#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Force live-mode behavior for evaluation.
os.environ["RISING_SUN_ENABLE_IDOC_DIRECTORY"] = "0"

from rising_sun.ground_truth import load_ground_truth_rows, load_ground_truth_rows_combined, match_ground_truth
from rising_sun.identity import normalize_person_name, normalize_supervision_number
from web.backend.main import REQUIRED_PAGE_COUNT, _extract_from_pdf, _verify_and_lookup


WORKBOOK = ROOT / "IDOC" / "Data" / "1. Processed Apps List.xlsx"
DATA_DIR_BY_YEAR = {
    "2025": ROOT / "IDOC" / "Data" / "PROCESSED APPS 2025",
    "2026": ROOT / "IDOC" / "Data" / "PROCESSED APPS 2026",
}


def discover_pdfs(years: list[str]) -> list[Path]:
    pdfs: list[Path] = []
    for year in years:
        directory = DATA_DIR_BY_YEAR[year]
        if not directory.exists():
            continue
        pdfs.extend(sorted(directory.rglob("*.pdf")))
    return pdfs


def truth_name_match_level(predicted_name: str, expected_name: str) -> str:
    predicted = normalize_person_name(predicted_name)
    expected = normalize_person_name(expected_name)
    if not predicted or not expected:
        return "none"
    if predicted == expected:
        return "exact"
    predicted_tokens = predicted.split()
    expected_tokens = expected.split()
    if predicted_tokens[:1] == expected_tokens[:1] and predicted_tokens[-1:] == expected_tokens[-1:]:
        return "first_last"
    if sorted(predicted_tokens) == sorted(expected_tokens):
        return "token_reordered"
    overlap = set(predicted_tokens) & set(expected_tokens)
    return "partial" if overlap else "none"


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    evaluated_rows = [row for row in rows if row["status"] == "evaluated"]
    matched_truth_rows = [row for row in evaluated_rows if row["truth_found"]]
    found_rows = [row for row in evaluated_rows if row["predicted_idoc_number"]]

    def count_where(items: list[dict[str, object]], key: str, value: object) -> int:
        return sum(1 for item in items if item.get(key) == value)

    verification_counts = {
        "green": count_where(evaluated_rows, "verification_status", "green"),
        "yellow": count_where(evaluated_rows, "verification_status", "yellow"),
        "red": count_where(evaluated_rows, "verification_status", "red"),
    }
    name_match_counts: dict[str, int] = {}
    pipeline_match_counts: dict[str, int] = {}
    for row in evaluated_rows:
        truth_level = str(row["truth_name_match_level"])
        pipeline_level = str(row["pipeline_name_match_level"])
        name_match_counts[truth_level] = name_match_counts.get(truth_level, 0) + 1
        pipeline_match_counts[pipeline_level] = pipeline_match_counts.get(pipeline_level, 0) + 1

    total = len(rows)
    evaluated = len(evaluated_rows)
    matched_truth = len(matched_truth_rows)
    predicted_found = len(found_rows)
    idoc_correct = count_where(matched_truth_rows, "idoc_match", True)
    rso_rows = [row for row in matched_truth_rows if row["expected_rso"] is not None and row["predicted_rso"] is not None]
    rso_correct = count_where(rso_rows, "rso_match", True)

    return {
        "total_files": total,
        "evaluated_files": evaluated,
        "skipped_non_4page": count_where(rows, "status", "skipped_non_4page"),
        "errors": count_where(rows, "status", "error"),
        "truth_rows_matched": matched_truth,
        "predicted_idoc_found": predicted_found,
        "idoc_exact_matches": idoc_correct,
        "idoc_accuracy_vs_truth": round(idoc_correct / matched_truth, 4) if matched_truth else 0.0,
        "idoc_found_rate": round(predicted_found / evaluated, 4) if evaluated else 0.0,
        "idoc_accuracy_when_found": round(idoc_correct / len([row for row in matched_truth_rows if row["predicted_idoc_number"]]), 4)
        if any(row["predicted_idoc_number"] for row in matched_truth_rows)
        else 0.0,
        "verification_status_counts": verification_counts,
        "truth_name_match_counts": name_match_counts,
        "pipeline_name_match_counts": pipeline_match_counts,
        "rso_rows_scored": len(rso_rows),
        "rso_exact_matches": rso_correct,
        "rso_accuracy_vs_truth": round(rso_correct / len(rso_rows), 4) if rso_rows else 0.0,
    }


async def evaluate(output_csv: Path, summary_json: Path, years: list[str], limit: int | None = None) -> None:
    if len(years) == 1:
        truth_rows = load_ground_truth_rows(WORKBOOK, sheet_name=years[0])
    else:
        truth_rows = load_ground_truth_rows_combined(WORKBOOK, sheet_names=years)

    pdf_paths = discover_pdfs(years)
    if limit is not None:
        pdf_paths = pdf_paths[:limit]

    rows: list[dict[str, object]] = []
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_pdf",
        "status",
        "page_count",
        "truth_found",
        "expected_name",
        "expected_idoc_number",
        "expected_rso",
        "predicted_name",
        "predicted_idoc_number",
        "predicted_rso",
        "idoc_match",
        "rso_match",
        "truth_name_match_level",
        "pipeline_name_match_level",
        "pipeline_match_score",
        "verification_status",
        "verification_reason",
        "extraction_method",
        "raw_capture",
        "candidate_count",
        "top_candidate_number",
        "top_candidate_name",
        "error",
    ]

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, pdf_path in enumerate(pdf_paths, start=1):
            row: dict[str, object] = {
                "source_pdf": str(pdf_path.relative_to(ROOT)),
                "status": "evaluated",
                "page_count": "",
                "truth_found": False,
                "expected_name": "",
                "expected_idoc_number": "",
                "expected_rso": None,
                "predicted_name": "",
                "predicted_idoc_number": "",
                "predicted_rso": None,
                "idoc_match": False,
                "rso_match": False,
                "truth_name_match_level": "none",
                "pipeline_name_match_level": "none",
                "pipeline_match_score": "",
                "verification_status": "",
                "verification_reason": "",
                "extraction_method": "",
                "raw_capture": "",
                "candidate_count": 0,
                "top_candidate_number": "",
                "top_candidate_name": "",
                "error": "",
            }

            try:
                truth = match_ground_truth(pdf_path, "", truth_rows)
                if truth is not None:
                    row["truth_found"] = True
                    row["expected_name"] = truth.source_name
                    row["expected_idoc_number"] = truth.supervision_number
                    row["expected_rso"] = truth.is_rso

                with fitz.open(pdf_path) as document:
                    page_count = len(document)
                row["page_count"] = page_count

                if REQUIRED_PAGE_COUNT is not None and page_count != REQUIRED_PAGE_COUNT:
                    row["status"] = "skipped_non_4page"
                else:
                    content = pdf_path.read_bytes()
                    extraction = _extract_from_pdf(content)
                    row["extraction_method"] = extraction.get("extraction_method", "")
                    row["raw_capture"] = extraction.get("raw_capture", "")
                    row["predicted_rso"] = (extraction.get("rso") or {}).get("is_rso")

                    if not row["truth_found"]:
                        truth = match_ground_truth(pdf_path, extraction.get("ocr_name", ""), truth_rows)
                        if truth is not None:
                            row["truth_found"] = True
                            row["expected_name"] = truth.source_name
                            row["expected_idoc_number"] = truth.supervision_number
                            row["expected_rso"] = truth.is_rso

                    result = await _verify_and_lookup(
                        candidates=list(extraction.get("candidates", [])),
                        raw_capture=extraction.get("raw_capture"),
                        extraction_method=extraction.get("extraction_method", ""),
                        ocr_names=extraction.get("ocr_names") or [],
                    )
                    info = result.get("idoc_info") or {}
                    verification = result.get("verification") or {}
                    candidate_results = result.get("candidate_results") or []

                    predicted_number = normalize_supervision_number(str(result.get("idoc_number", "") or ""))
                    predicted_name = str(info.get("name", "") or extraction.get("ocr_name", "") or "")

                    row["predicted_idoc_number"] = predicted_number
                    row["predicted_name"] = predicted_name
                    row["verification_status"] = verification.get("status", "")
                    row["verification_reason"] = verification.get("reason", "")
                    row["candidate_count"] = len(candidate_results)
                    if candidate_results:
                        row["top_candidate_number"] = candidate_results[0].get("idoc_number", "")
                        row["top_candidate_name"] = candidate_results[0].get("idoc_name", "")
                        row["pipeline_name_match_level"] = candidate_results[0].get("match_level", "none")
                        row["pipeline_match_score"] = candidate_results[0].get("match_score", "")

                    if truth is not None:
                        row["idoc_match"] = predicted_number == truth.supervision_number
                        row["rso_match"] = row["predicted_rso"] == truth.is_rso if row["predicted_rso"] is not None else False
                        row["truth_name_match_level"] = truth_name_match_level(predicted_name, truth.source_name)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)

            rows.append(row)
            writer.writerow(row)
            handle.flush()

            if index % 10 == 0:
                print(f"[{index}/{len(pdf_paths)}] processed {pdf_path.name}", flush=True)

    summary = build_summary(rows)
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the live IDOC pipeline without Processed Apps runtime lookup.")
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "output" / "live_pipeline_eval_v1_3_0.csv"),
        help="CSV path for per-document evaluation results.",
    )
    parser.add_argument(
        "--summary-json",
        default=str(ROOT / "output" / "live_pipeline_eval_v1_3_0_summary.json"),
        help="JSON path for evaluation summary metrics.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        choices=sorted(DATA_DIR_BY_YEAR),
        default=["2026"],
        help="Processed-app corpus year(s) to evaluate. Defaults to 2026 for parity with the previous evaluation corpus.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N PDFs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(evaluate(Path(args.output_csv), Path(args.summary_json), years=list(args.years), limit=args.limit))


if __name__ == "__main__":
    main()