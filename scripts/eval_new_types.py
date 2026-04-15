#!/usr/bin/env python3
"""Evaluate extraction on non-4-page PDFs (new document types).

Skips 4-page IDOC forms (already validated at 98.1%) and focuses on:
- 3-page Jotform digital applications (instant, no OCR)
- 2-page scanned Rising Sun packets
- 2-page scanned IDOC forms
- 3-page scanned RS/IDOC forms
- 6/8/10-page bundles
- 1/5-page edge cases
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import fitz

# Enable CUDA for RapidOCR if available
os.environ.setdefault("RISING_SUN_RAPIDOCR_USE_CUDA", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rising_sun.extractor import ApplicationExtractor
from rising_sun.ground_truth import load_ground_truth_rows, match_ground_truth
from rising_sun.identity import normalize_person_name, normalize_supervision_number

DATA_DIR = Path(__file__).resolve().parent.parent / "IDOC" / "Data" / "PROCESSED APPS 2026"
WORKBOOK = Path(__file__).resolve().parent.parent / "IDOC" / "Data" / "1. Processed Apps List.xlsx"
TEMPLATE = Path(__file__).resolve().parent.parent / "config" / "idoc_application_template.yml"
OUTPUT = Path(__file__).resolve().parent.parent / "output" / "new_types_eval.csv"


def main():
    # Find all non-4-page PDFs
    all_pdfs = sorted(DATA_DIR.glob("*.pdf"))
    print(f"Total PDFs: {len(all_pdfs)}")

    non_four: list[tuple[Path, int, bool]] = []
    for pdf_path in all_pdfs:
        doc = fitz.open(pdf_path)
        pages = doc.page_count
        total_text = sum(len(p.get_text().strip()) for p in doc)
        doc.close()
        is_digital = total_text > 100
        if pages != 4 or is_digital:
            non_four.append((pdf_path, pages, is_digital))

    print(f"Non-4-page or digital PDFs: {len(non_four)}")

    # Sort: digital first (fast), then by page count
    non_four.sort(key=lambda x: (not x[2], x[1], x[0].name))

    extractor = ApplicationExtractor(TEMPLATE)
    truth_rows = load_ground_truth_rows(WORKBOOK, sheet_name="2026") if WORKBOOK.exists() else []

    fieldnames = [
        "source_pdf", "page_count", "is_digital", "document_classification", "template",
        "supported", "name", "idoc_number", "fields_filled", "fields_total",
        "expected_name", "name_match", "expected_number", "number_match", "elapsed_s", "error",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-processed files
    done_pdfs: set[str] = set()
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            for row in csv.DictReader(f):
                done_pdfs.add(row["source_pdf"])
        print(f"Resuming: {len(done_pdfs)} already done, {len(non_four) - len(done_pdfs)} remaining")

    append_mode = bool(done_pdfs)
    csv_file = open(OUTPUT, "a" if append_mode else "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
    if not append_mode:
        writer.writeheader()
    csv_file.flush()

    errors = 0
    t0 = time.time()
    try:
        for i, (pdf_path, pages, is_digital) in enumerate(non_four):
            if pdf_path.name in done_pdfs:
                continue
            t1 = time.time()
            row: dict[str, str] = {
                "source_pdf": pdf_path.name,
                "page_count": str(pages),
                "is_digital": str(is_digital),
                "error": "",
            }
            try:
                result = extractor.extract_pdf(pdf_path, include_raw_text=True)
                e = result.get("extracted", {})
                fields = result.get("field_results", {})
                row["document_classification"] = result.get("document_classification", "")
                row["template"] = result.get("template", "")
                row["supported"] = str(result.get("supported_template", False))
                row["name"] = e.get("applicant", {}).get("name", "")
                row["idoc_number"] = e.get("applicant", {}).get("idoc_or_le_number", "") or e.get("applicant", {}).get("idoc_number", "")
                row["fields_filled"] = str(sum(1 for f in fields.values() if f.get("value")))
                row["fields_total"] = str(len(fields))

                if truth_rows:
                    truth = match_ground_truth(pdf_path, row["name"], truth_rows)
                    row["expected_name"] = truth.source_name if truth else ""
                    row["expected_number"] = truth.supervision_number if truth else ""
                    if truth:
                        norm = normalize_person_name(row["name"])
                        truth_norm = truth.normalized_name
                        norm_tokens = sorted(norm.split())
                        truth_tokens = sorted(truth_norm.split())
                        row["name_match"] = str(
                            norm == truth_norm
                            or norm_tokens == truth_tokens
                            or (norm.split()[:1] == truth_norm.split()[:1] and norm.split()[-1:] == truth_norm.split()[-1:])
                        )
                        extracted_num = normalize_supervision_number(row["idoc_number"])
                        row["number_match"] = str(extracted_num == truth.supervision_number)
            except Exception as ex:
                row["error"] = str(ex)
                row["document_classification"] = "ERROR"
                errors += 1

            row["elapsed_s"] = f"{time.time() - t1:.1f}"
            writer.writerow(row)
            csv_file.flush()

            status = "✓" if not row["error"] else "✗"
            name_info = row.get("name", "")[:25]
            print(f"  [{i+1}/{len(non_four)}] {status} {row['elapsed_s']:>5s}s {pages}p {'D' if is_digital else 'S'} {row.get('document_classification','?'):35s} {name_info:25s} | {pdf_path.name}")
    finally:
        csv_file.close()

    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Evaluation complete: {len(non_four)} PDFs in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Errors: {errors}")
    print(f"CSV: {OUTPUT}")


if __name__ == "__main__":
    main()
