from __future__ import annotations

import csv
from difflib import SequenceMatcher
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import fitz
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from rising_sun.calibration import export_review_crops
from rising_sun.classification import classify_document
from rising_sun.extractor import ApplicationExtractor
from rising_sun.ground_truth import load_ground_truth_rows, match_ground_truth
from rising_sun.identity import IdentityExtractor, _looks_like_date_number, _supervision_number_score, clean_pdf_stem_name, normalize_person_name, normalize_supervision_number
from rising_sun.image_ops import crop_image, mostly_blank
from rising_sun.name_ocr import NameOcrCandidate, available_name_ocr_backends, build_name_ocr_backend
from rising_sun.ocr import RapidOcrBackend
from rising_sun.pdf import render_pdf_page, render_pdf_pages
from rising_sun.review import annotate_result_reviews, collect_review_rows, write_review_csv
from rising_sun.train_name_ocr import train_name_ocr_model

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def digit_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        return max(len(left), len(right))
    return sum(left_char != right_char for left_char, right_char in zip(left, right))


def categorize_comparison(name_match: bool, extracted_number: str, expected_number: str) -> str:
    if not name_match:
        return "name_mismatch"
    if not extracted_number and not expected_number:
        return "both_blank"
    if extracted_number == expected_number:
        return "match"
    if expected_number and not extracted_number:
        return "expected_present_extracted_blank"
    if extracted_number and not expected_number:
        return "expected_blank_extracted_present"
    if len(extracted_number) == len(expected_number) and digit_distance(extracted_number, expected_number) == 1:
        return "single_digit_ocr_mismatch"
    if abs(len(extracted_number) - len(expected_number)) >= 2:
        return "possible_workbook_update_or_document_disagreement"
    return "multi_digit_ocr_mismatch"


def _first_last_name_match(left: str, right: str) -> bool:
    left_parts = left.split()
    right_parts = right.split()
    return bool(left_parts and right_parts and left_parts[:1] == right_parts[:1] and left_parts[-1:] == right_parts[-1:])


def _token_overlap_count(left: str, right: str) -> int:
    return len(set(left.split()).intersection(set(right.split())))


def _display_training_name(source_name: str) -> str:
    value = str(source_name or "").strip()
    if "," in value:
        last, first = [part.strip() for part in value.split(",", 1)]
        return f"{first} {last}".strip()
    return re.sub(r"\s+", " ", value)


def _dataset_split_for_key(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:2], 16)
    if bucket < 26:
        return "test"
    if bucket < 52:
        return "val"
    return "train"


def _applicant_split_key(pdf_path: Path, normalized_name: str) -> str:
    candidate = normalize_person_name(normalized_name)
    if candidate:
        return candidate
    return normalize_person_name(clean_pdf_stem_name(pdf_path)) or str(pdf_path)


def _name_failure_priority(row: dict[str, str]) -> tuple[int, int, float, float, float, str]:
    classification = row.get("classification", "")
    handwritten_rank = 0 if classification == "idoc_housing_application_v1" else 1
    exact_rank = 0 if row.get("exact_match") == "False" else 1
    first_last_rank = 0 if row.get("first_last_match") == "False" else 1
    token_overlap = float(row.get("token_overlap", "0") or 0)
    similarity = float(row.get("similarity_ratio", "0") or 0.0)
    confidence = float(row.get("confidence", "0") or 0.0)
    return (handwritten_rank, exact_rank + first_last_rank, token_overlap, similarity, confidence, row.get("source_pdf", ""))


def _variant_priority(variant: str) -> tuple[int, str]:
    order = {"tight": 0, "context": 1, "wide": 2}
    return (order.get(variant, 99), variant)


def _review_queue_status(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"approved", "ready", "train"}:
        return "approved"
    if normalized in {"skip", "rejected", "ignore"}:
        return "skip"
    return "pending"


def _expected_name_fields(row: dict[str, str]) -> tuple[str, str]:
    expected_name = str(row.get("label_text") or row.get("expected_name") or "")
    normalized_expected = str(row.get("normalized_label") or row.get("normalized_expected_name") or normalize_person_name(expected_name))
    return expected_name, normalized_expected


def _pick_best_name_candidate(candidates: list[NameOcrCandidate]) -> NameOcrCandidate | None:
    if not candidates:
        return None
    duplicate_counts = Counter(candidate.value for candidate in candidates)
    ranked = sorted(
        candidates,
        key=lambda candidate: (duplicate_counts[candidate.value], candidate.score, candidate.confidence, len(candidate.value)),
        reverse=True,
    )
    return ranked[0]


def discover_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*.pdf") if path.is_file())


def collect_supervision_candidate_counts(extractor: IdentityExtractor, pdf_path: Path) -> tuple[str, str, Counter[str]]:
    page = render_pdf_page(pdf_path, dpi=max(extractor.page_dpi, 300), page_number=0)
    raw = extractor._read_raw(page)
    page_text = extractor._join_lines(raw)
    classification = classify_document({"1": page_text}, 1)
    registration = extractor._estimate_registration(classification.name, raw, page.shape)
    candidate_texts: list[str] = []

    if classification.name == "idoc_housing_application_v1":
        for pattern in [
            r"IDOC#[:\s]*([A-Za-z0-9]{4,10})",
            r"IDOc#[:\s]*([A-Za-z0-9]{4,10})",
            r"IDoc#[:\s]*([A-Za-z0-9]{4,10})",
        ]:
            match = re.search(pattern, page_text, flags=re.IGNORECASE)
            if match:
                candidate_texts.append(match.group(1))
        for _, box in extractor._idoc_crop_boxes(registration):
            crop = crop_image(page, extractor._shift_box(box, registration, x_weight=0.8, y_weight=1.0), padding=8)
            candidate_texts.extend(extractor._ocr_crop_candidates(crop))
    else:
        match = re.search(r"IDOC\s*or\s*LE\s*#?\s*([A-Za-z0-9]{4,10})", page_text, flags=re.IGNORECASE)
        if match:
            candidate_texts.append(match.group(1))

        label_item = None
        for item in raw:
            normalized = re.sub(r"\s+", "", item[1].lower())
            if "idocorle" in normalized:
                label_item = item
                break

        if label_item is not None:
            label_box = label_item[0]
            x1 = min(point[0] for point in label_box)
            y1 = min(point[1] for point in label_box)
            x2 = max(point[0] for point in label_box)
            y2 = max(point[1] for point in label_box)
            label_height = max(1.0, y2 - y1)

            for box, text, _ in raw:
                bx1 = min(point[0] for point in box)
                by1 = min(point[1] for point in box)
                same_row = abs(by1 - y1) <= label_height * 1.4
                right_side = bx1 >= x2 - 20
                below_row = by1 >= y1 and by1 <= y1 + label_height * 3.0 and bx1 >= x1
                if same_row and right_side:
                    candidate_texts.append(text)
                elif below_row and re.search(r"[0-9A-Za-z]{4,}", text):
                    candidate_texts.append(text)

            left = max(0, int(x1 - 40))
            top = max(0, int(y1 - 30))
            right = min(page.shape[1], int(x2 + 420))
            bottom = min(page.shape[0], int(y2 + 120))
            candidate_texts.extend(extractor._ocr_crop_candidates(page[top:bottom, left:right], tighten=False))
        else:
            for _, box in extractor._packet_crop_boxes(classification.name):
                crop = crop_image(page, extractor._shift_box(box, registration, x_weight=1.0, y_weight=1.0), padding=8)
                candidate_texts.extend(extractor._ocr_crop_candidates(crop, tighten=False))
            predicted_box = extractor._predicted_packet_label_box(classification.name, registration)
            if predicted_box is not None:
                crop = crop_image(page, predicted_box, padding=8)
                candidate_texts.extend(extractor._ocr_crop_candidates(crop, tighten=False))

    counts: Counter[str] = Counter()
    for candidate_text in candidate_texts:
        for token in re.findall(r"[A-Za-z0-9/\\|!&$%()\[\]{}]{4,14}", candidate_text):
            candidate = normalize_supervision_number(token)
            if not candidate:
                continue
            if classification.name == "email_forward_application_packet" and _looks_like_date_number(candidate):
                continue
            counts[candidate] += 1
    return classification.name, registration.subtype, counts


def label_found_on_any_page(extractor: IdentityExtractor, pdf_path: Path, classification_name: str) -> bool:
    if classification_name == "unknown_document_type":
        return False
    doc = fitz.open(pdf_path)
    label_patterns = ["idocorle"] if classification_name != "idoc_housing_application_v1" else ["idoc#", "idoc"]
    for page_number in range(doc.page_count):
        page = render_pdf_page(pdf_path, dpi=extractor.page_dpi, page_number=page_number)
        raw = extractor._read_raw(page)
        text = extractor._join_lines(raw)
        compact = re.sub(r"\s+", "", text.lower())
        if any(pattern in compact for pattern in label_patterns):
            return True
    return False


def classify_failure_cause(
    mismatch_category: str,
    expected_present: bool,
    extracted_number: str,
    expected_number: str,
    label_any_page: bool,
) -> str:
    if mismatch_category == "possible_workbook_update_or_document_disagreement":
        return "likely_workbook_or_document_disagreement"
    if expected_present:
        return "candidate_ranking_issue"
    if mismatch_category == "expected_present_extracted_blank":
        return "localization_or_detection_issue" if label_any_page else "field_missing_or_unreadable_in_document"
    if extracted_number and expected_number and len(extracted_number) == len(expected_number) and digit_distance(extracted_number, expected_number) == 1:
        return "single_digit_ocr_recognition_issue"
    return "ocr_recognition_or_localization_issue"


@app.callback()
def main() -> None:
    """Rising Sun OCR pipeline commands."""


@app.command()
def process(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="PDF file or directory to process."),
    output_dir: Path = typer.Option(Path("output/json"), "--output-dir", help="Directory for extracted JSON files."),
    template: Path = typer.Option(Path("config/idoc_application_template.yml"), "--template", exists=True, readable=True),
    name_backend: str = typer.Option("rapid_ensemble", "--name-backend", help=f"Applicant-name OCR backend. Options: {', '.join(available_name_ocr_backends())}."),
    enable_idoc_directory: bool = typer.Option(False, "--enable-idoc-directory", help="Allow Processed Apps spreadsheet lookups for non-live processing workflows."),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Process only the first N PDFs."),
    include_raw_text: bool = typer.Option(True, "--raw-text/--no-raw-text", help="Include page-level OCR fallback text."),
    review_file: Path | None = typer.Option(None, "--review-file", help="Optional CSV path for low-confidence and unresolved fields."),
) -> None:
    extractor = ApplicationExtractor(template, name_backend=name_backend, enable_idoc_directory=enable_idoc_directory)
    pdf_paths = discover_pdfs(input_path)
    if limit is not None:
        pdf_paths = pdf_paths[:limit]

    if not pdf_paths:
        raise typer.Exit("No PDF files found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []

    base_dir = input_path if input_path.is_dir() else input_path.parent
    progress_columns = [SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn()]
    with Progress(*progress_columns, console=console) as progress:
        task_id = progress.add_task("Processing IDOC applications", total=len(pdf_paths))
        for pdf_path in pdf_paths:
            result = extractor.extract_pdf(pdf_path, include_raw_text=include_raw_text)
            annotate_result_reviews(result)
            relative = pdf_path.relative_to(base_dir).with_suffix(".json") if pdf_path.is_relative_to(base_dir) else Path(pdf_path.stem + ".json")
            output_path = output_dir / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2))
            manifest.append({"source_pdf": str(pdf_path), "output_json": str(output_path)})
            review_rows.extend(collect_review_rows(result))
            progress.advance(task_id)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    review_path = review_file or (output_dir / "review.csv")
    write_review_csv(review_rows, review_path)
    console.print(f"Wrote {len(pdf_paths)} JSON files to {output_dir}")
    console.print(f"Manifest: {manifest_path}")
    console.print(f"Review CSV: {review_path}")


@app.command("export-review-crops")
def export_review_crops_command(
    review_csv: Path = typer.Argument(..., exists=True, readable=True, help="Review CSV generated by the process command."),
    output_dir: Path = typer.Option(Path("output/review_crops"), "--output-dir", help="Directory for exported field crop images."),
    template: Path = typer.Option(Path("config/idoc_application_template.yml"), "--template", exists=True, readable=True),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Export only the first N review rows."),
) -> None:
    written = export_review_crops(review_csv, template, output_dir, limit=limit)
    console.print(f"Exported {written} review crops to {output_dir}")


@app.command("compare-ground-truth")
def compare_ground_truth(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="PDF file or directory to compare."),
    workbook: Path = typer.Argument(..., exists=True, readable=True, help="Workbook with ground-truth names and supervision numbers."),
    output_csv: Path = typer.Option(Path("output/ground_truth_compare.csv"), "--output-csv", help="CSV path for comparison results."),
    sheet_name: str = typer.Option("2026", "--sheet", help="Workbook sheet to use."),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Compare only the first N PDFs."),
) -> None:
    pdf_paths = discover_pdfs(input_path)
    if limit is not None:
        pdf_paths = pdf_paths[:limit]
    if not pdf_paths:
        raise typer.Exit("No PDF files found.")

    truth_rows = load_ground_truth_rows(workbook, sheet_name=sheet_name)
    extractor = IdentityExtractor()
    comparison_rows: list[dict[str, str | bool]] = []

    progress_columns = [SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn()]
    with Progress(*progress_columns, console=console) as progress:
        task_id = progress.add_task("Comparing name and supervision number", total=len(pdf_paths))
        for pdf_path in pdf_paths:
            identity = extractor.extract(pdf_path)
            truth = match_ground_truth(pdf_path, identity.name, truth_rows)
            expected_name = truth.source_name if truth else ""
            expected_number = truth.supervision_number if truth else ""
            extracted_number = normalize_supervision_number(identity.supervision_number)
            normalized_name = normalize_person_name(identity.name)
            truth_name = truth.normalized_name if truth else ""
            name_match = bool(
                truth
                and (
                    normalized_name == truth_name
                    or normalized_name.split()[:1] == truth_name.split()[:1] and normalized_name.split()[-1:] == truth_name.split()[-1:]
                )
            )
            mismatch_category = "no_ground_truth_row" if not truth else categorize_comparison(name_match, extracted_number, expected_number)
            comparison_rows.append({
                "source_pdf": str(pdf_path),
                "classification": identity.classification.name,
                "layout_subtype": identity.layout_subtype,
                "extracted_name": identity.name,
                "expected_name": expected_name,
                "name_match": name_match,
                "extracted_supervision_number": extracted_number,
                "expected_supervision_number": expected_number,
                "number_match": bool(truth and extracted_number == expected_number),
                "mismatch_category": mismatch_category,
            })
            progress.advance(task_id)

    import csv

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_pdf",
                "classification",
                "layout_subtype",
                "extracted_name",
                "expected_name",
                "name_match",
                "extracted_supervision_number",
                "expected_supervision_number",
                "number_match",
                "mismatch_category",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    matched_rows = [row for row in comparison_rows if row["expected_name"]]
    name_matches = sum(1 for row in matched_rows if row["name_match"])
    number_matches = sum(1 for row in matched_rows if row["number_match"])
    console.print(f"Wrote comparison CSV: {output_csv}")
    console.print(f"Matched workbook rows: {len(matched_rows)}/{len(comparison_rows)}")
    console.print(f"Name matches: {name_matches}/{len(matched_rows) if matched_rows else 0}")
    console.print(f"Supervision-number matches: {number_matches}/{len(matched_rows) if matched_rows else 0}")


@app.command("evaluate-extractor-names")
def evaluate_extractor_names(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="PDF file or directory to evaluate."),
    workbook: Path = typer.Argument(..., exists=True, readable=True, help="Workbook with ground-truth names."),
    output_csv: Path = typer.Option(Path("output/name_ocr_research.csv"), "--output-csv", help="CSV path for name OCR evaluation results."),
    template: Path = typer.Option(Path("config/idoc_application_template.yml"), "--template", exists=True, readable=True),
    name_backend: str = typer.Option("rapid_ensemble", "--name-backend", help=f"Applicant-name OCR backend. Options: {', '.join(available_name_ocr_backends())}."),
    enable_idoc_directory: bool = typer.Option(False, "--enable-idoc-directory", help="Allow Processed Apps spreadsheet lookups for non-live processing workflows."),
    sheet_name: str = typer.Option("2026", "--sheet", help="Workbook sheet to use."),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Evaluate only the first N PDFs."),
) -> None:
    pdf_paths = discover_pdfs(input_path)
    if limit is not None:
        pdf_paths = pdf_paths[:limit]
    if not pdf_paths:
        raise typer.Exit("No PDF files found.")

    truth_rows = load_ground_truth_rows(workbook, sheet_name=sheet_name)
    extractor = ApplicationExtractor(template, name_backend=name_backend, enable_idoc_directory=enable_idoc_directory)
    comparison_rows: list[dict[str, str | float | int | bool]] = []

    progress_columns = [SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn()]
    with Progress(*progress_columns, console=console) as progress:
        task_id = progress.add_task("Evaluating extractor name OCR", total=len(pdf_paths))
        for pdf_path in pdf_paths:
            result = extractor.extract_pdf(pdf_path, include_raw_text=True)
            field = result.get("field_results", {}).get("applicant.name", {})
            extracted_name = str(field.get("value", "") or "")
            normalized_name = normalize_person_name(extracted_name)
            truth = match_ground_truth(pdf_path, extracted_name, truth_rows)
            expected_name = truth.source_name if truth else ""
            expected_normalized = truth.normalized_name if truth else ""
            exact_match = bool(truth and normalized_name == expected_normalized)
            first_last_match = bool(truth and _first_last_name_match(normalized_name, expected_normalized))
            token_overlap = _token_overlap_count(normalized_name, expected_normalized) if truth else 0
            similarity = SequenceMatcher(None, normalized_name, expected_normalized).ratio() if truth else 0.0
            candidates = field.get("candidates", []) if isinstance(field.get("candidates"), list) else []
            candidate_summary = " ; ".join(
                f"{candidate.get('source', '')}={candidate.get('value', '')}"
                for candidate in candidates[:5]
                if isinstance(candidate, dict)
            )
            comparison_rows.append({
                "source_pdf": str(pdf_path),
                "classification": result.get("document_classification", ""),
                "supported_template": bool(result.get("supported_template", True)),
                "name_backend": name_backend,
                "expected_name": expected_name,
                "extracted_name": extracted_name,
                "normalized_expected_name": expected_normalized,
                "normalized_extracted_name": normalized_name,
                "exact_match": exact_match,
                "first_last_match": first_last_match,
                "token_overlap": token_overlap,
                "similarity_ratio": round(similarity, 4),
                "source": field.get("source", ""),
                "confidence": float(field.get("confidence", 0.0)),
                "candidate_count": len(candidates),
                "top_candidates": candidate_summary,
            })
            progress.advance(task_id)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_pdf",
                "classification",
                "supported_template",
                "name_backend",
                "expected_name",
                "extracted_name",
                "normalized_expected_name",
                "normalized_extracted_name",
                "exact_match",
                "first_last_match",
                "token_overlap",
                "similarity_ratio",
                "source",
                "confidence",
                "candidate_count",
                "top_candidates",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    matched_rows = [row for row in comparison_rows if row["expected_name"]]
    exact_matches = sum(1 for row in matched_rows if row["exact_match"])
    first_last_matches = sum(1 for row in matched_rows if row["first_last_match"])
    any_token_matches = sum(1 for row in matched_rows if int(row["token_overlap"]) > 0)
    source_breakdown = Counter(
        str(row["source"])
        for row in matched_rows
        if row["first_last_match"]
    )

    console.print(f"Wrote extractor name evaluation CSV: {output_csv}")
    console.print(f"Matched workbook rows: {len(matched_rows)}/{len(comparison_rows)}")
    console.print(f"Exact matches: {exact_matches}/{len(matched_rows) if matched_rows else 0}")
    console.print(f"First/last matches: {first_last_matches}/{len(matched_rows) if matched_rows else 0}")
    console.print(f"Any-token matches: {any_token_matches}/{len(matched_rows) if matched_rows else 0}")
    for source, count in source_breakdown.most_common():
        console.print(f"{source}: {count}")


@app.command("evaluate-name-crop-manifest")
def evaluate_name_crop_manifest(
    manifest_csv: Path = typer.Argument(..., exists=True, readable=True, help="Manifest generated by export-name-training-data or export-name-failure-crops."),
    output_csv: Path = typer.Option(Path("output/name_crop_backend_eval.csv"), "--output-csv", help="CSV path for crop-level backend evaluation results."),
    name_backend: str = typer.Option("rapid_ensemble", "--name-backend", help=f"Applicant-name OCR backend. Options: {', '.join(available_name_ocr_backends())}."),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Evaluate only the first N grouped PDFs in the manifest."),
    split: str | None = typer.Option(None, "--split", help="Optional manifest split filter, e.g. train, val, or test."),
    variants: str = typer.Option("tight,context,wide", "--variants", help="Comma-separated crop variants to include from the manifest."),
) -> None:
    rows = list(csv.DictReader(manifest_csv.open()))
    if split is not None:
        wanted_split = split.strip().lower()
        rows = [row for row in rows if str(row.get("split", "")).strip().lower() == wanted_split]

    allowed_variants = {part.strip().lower() for part in variants.split(",") if part.strip()}
    if allowed_variants:
        rows = [row for row in rows if str(row.get("variant", "")).strip().lower() in allowed_variants]

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        source_pdf = str(row.get("source_pdf", "")).strip()
        if not source_pdf:
            continue
        grouped_rows[source_pdf].append(row)

    grouped_items = sorted(grouped_rows.items())
    if limit is not None:
        grouped_items = grouped_items[:limit]
    if not grouped_items:
        raise typer.Exit("No manifest rows available after filtering.")

    backend = build_name_ocr_backend(name_backend, rapid_ocr=RapidOcrBackend(), normalize_name=normalize_person_name)
    comparison_rows: list[dict[str, str | float | int | bool]] = []

    progress_columns = [SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn()]
    with Progress(*progress_columns, console=console) as progress:
        task_id = progress.add_task("Evaluating name OCR on exported crops", total=len(grouped_items))
        for source_pdf, group in grouped_items:
            expected_name, normalized_expected = _expected_name_fields(group[0])
            classification = str(group[0].get("classification", ""))
            aggregated_candidates: list[NameOcrCandidate] = []

            for row in sorted(group, key=lambda item: _variant_priority(str(item.get("variant", "")))):
                image_path = Path(str(row.get("image_path", "")))
                if not image_path.exists():
                    continue
                crop = cv2.imread(str(image_path))
                if crop is None or crop.size == 0:
                    continue
                variant_label = str(row.get("variant", "tight") or "tight")
                crop_candidates = backend.extract_crop_candidates(crop, variant_label=variant_label)
                aggregated_candidates.extend(crop_candidates)

            best_candidate = _pick_best_name_candidate(aggregated_candidates)
            extracted_name = best_candidate.value if best_candidate else ""
            normalized_name = normalize_person_name(extracted_name)
            exact_match = bool(normalized_expected and normalized_name == normalized_expected)
            first_last_match = bool(normalized_expected and _first_last_name_match(normalized_name, normalized_expected))
            token_overlap = _token_overlap_count(normalized_name, normalized_expected) if normalized_expected else 0
            similarity = SequenceMatcher(None, normalized_name, normalized_expected).ratio() if normalized_expected else 0.0
            candidate_summary = " ; ".join(f"{candidate.source}={candidate.value}" for candidate in aggregated_candidates[:5])
            comparison_rows.append({
                "source_pdf": source_pdf,
                "classification": classification,
                "name_backend": name_backend,
                "expected_name": expected_name,
                "extracted_name": extracted_name,
                "normalized_expected_name": normalized_expected,
                "normalized_extracted_name": normalized_name,
                "exact_match": exact_match,
                "first_last_match": first_last_match,
                "token_overlap": token_overlap,
                "similarity_ratio": round(similarity, 4),
                "source": best_candidate.source if best_candidate else "",
                "confidence": float(best_candidate.confidence) if best_candidate else 0.0,
                "candidate_count": len(aggregated_candidates),
                "top_candidates": candidate_summary,
                "variants": ",".join(sorted({str(row.get("variant", "")) for row in group if str(row.get("variant", ""))})),
            })
            progress.advance(task_id)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_pdf",
                "classification",
                "name_backend",
                "expected_name",
                "extracted_name",
                "normalized_expected_name",
                "normalized_extracted_name",
                "exact_match",
                "first_last_match",
                "token_overlap",
                "similarity_ratio",
                "source",
                "confidence",
                "candidate_count",
                "top_candidates",
                "variants",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    matched_rows = [row for row in comparison_rows if row["expected_name"]]
    exact_matches = sum(1 for row in matched_rows if row["exact_match"])
    first_last_matches = sum(1 for row in matched_rows if row["first_last_match"])
    any_token_matches = sum(1 for row in matched_rows if int(row["token_overlap"]) > 0)

    console.print(f"Wrote crop-level name evaluation CSV: {output_csv}")
    console.print(f"Grouped PDFs evaluated: {len(comparison_rows)}")
    console.print(f"Exact matches: {exact_matches}/{len(matched_rows) if matched_rows else 0}")
    console.print(f"First/last matches: {first_last_matches}/{len(matched_rows) if matched_rows else 0}")
    console.print(f"Any-token matches: {any_token_matches}/{len(matched_rows) if matched_rows else 0}")


@app.command("export-name-training-data")
def export_name_training_data(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="PDF file or directory to export from."),
    workbook: Path = typer.Argument(..., exists=True, readable=True, help="Workbook with ground-truth names."),
    output_dir: Path = typer.Option(Path("output/name_training_dataset"), "--output-dir", help="Directory for labeled name-crop exports."),
    template: Path = typer.Option(Path("config/idoc_application_template.yml"), "--template", exists=True, readable=True),
    name_backend: str = typer.Option("rapid_ensemble", "--name-backend", help=f"Applicant-name OCR backend for crop variants. Options: {', '.join(available_name_ocr_backends())}."),
    enable_idoc_directory: bool = typer.Option(False, "--enable-idoc-directory", help="Allow Processed Apps spreadsheet lookups for non-live processing workflows."),
    sheet_name: str = typer.Option("2026", "--sheet", help="Workbook sheet to use."),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Export only the first N PDFs."),
    include_packets: bool = typer.Option(False, "--include-packets", help="Include Rising Sun packet forms in addition to handwritten IDOC forms."),
) -> None:
    pdf_paths = discover_pdfs(input_path)
    if limit is not None:
        pdf_paths = pdf_paths[:limit]
    if not pdf_paths:
        raise typer.Exit("No PDF files found.")

    truth_rows = load_ground_truth_rows(workbook, sheet_name=sheet_name)
    extractor = ApplicationExtractor(template, name_backend=name_backend, enable_idoc_directory=enable_idoc_directory)
    name_field = extractor.field_map.get("applicant.name")
    if name_field is None or name_field.box is None:
        raise typer.Exit("Template does not define applicant.name crop coordinates.")

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str | int | float]] = []

    progress_columns = [SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn()]
    with Progress(*progress_columns, console=console) as progress:
        task_id = progress.add_task("Exporting labeled name crops", total=len(pdf_paths))
        for pdf_path in pdf_paths:
            truth = match_ground_truth(pdf_path, "", truth_rows)
            if truth is None:
                progress.advance(task_id)
                continue

            pages = render_pdf_pages(pdf_path, dpi=extractor.template.render_dpi)
            if not pages:
                progress.advance(task_id)
                continue
            page1 = pages[0]
            page_text = extractor.ocr.read_text(page1, multiline=True).text
            classification = classify_document({"1": page_text}, len(pages))
            if classification.name != "idoc_housing_application_v1" and not include_packets:
                progress.advance(task_id)
                continue

            applicant_key = _applicant_split_key(pdf_path, truth.normalized_name)
            split = _dataset_split_for_key(applicant_key)
            label_text = _display_training_name(truth.source_name)
            for variant, crop_box in extractor.name_ocr_backend.candidate_boxes(name_field.box).items():
                crop = crop_image(page1, crop_box)
                if crop.size == 0 or mostly_blank(crop):
                    continue
                image_name = f"{pdf_path.stem.replace(' ', '_')}__{variant}.png"
                image_path = images_dir / image_name
                cv2.imwrite(str(image_path), crop)
                manifest_rows.append({
                    "source_pdf": str(pdf_path),
                    "applicant_key": applicant_key,
                    "image_path": str(image_path),
                    "image_relpath": str(image_path.relative_to(output_dir)),
                    "label_text": label_text,
                    "normalized_label": normalize_person_name(label_text),
                    "classification": classification.name,
                    "split": split,
                    "variant": variant,
                    "page": 1,
                    "left": crop_box[0],
                    "top": crop_box[1],
                    "right": crop_box[2],
                    "bottom": crop_box[3],
                    "name_backend": name_backend,
                })
            progress.advance(task_id)

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_pdf",
                "applicant_key",
                "image_path",
                "image_relpath",
                "label_text",
                "normalized_label",
                "classification",
                "split",
                "variant",
                "page",
                "left",
                "top",
                "right",
                "bottom",
                "name_backend",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    for split in ["train", "val", "test"]:
        split_rows = [row for row in manifest_rows if row["split"] == split]
        split_path = output_dir / f"{split}.jsonl"
        with split_path.open("w") as handle:
            for row in split_rows:
                handle.write(json.dumps({"image": row["image_relpath"], "text": row["label_text"], "applicant_key": row["applicant_key"]}) + "\n")

    console.print(f"Wrote labeled crop manifest: {manifest_path}")
    console.print(f"Exported image crops: {len(manifest_rows)}")
    split_counts = Counter(str(row["split"]) for row in manifest_rows)
    for split, count in split_counts.items():
        console.print(f"{split}: {count}")


@app.command("export-name-failure-crops")
def export_name_failure_crops(
    comparison_csv: Path = typer.Argument(..., exists=True, readable=True, help="CSV generated by evaluate-extractor-names."),
    output_dir: Path = typer.Option(Path("output/name_failure_crops"), "--output-dir", help="Directory for exported failure crops."),
    template: Path = typer.Option(Path("config/idoc_application_template.yml"), "--template", exists=True, readable=True),
    name_backend: str = typer.Option("rapid_ensemble", "--name-backend", help=f"Applicant-name crop backend for export. Options: {', '.join(available_name_ocr_backends())}."),
    enable_idoc_directory: bool = typer.Option(False, "--enable-idoc-directory", help="Allow Processed Apps spreadsheet lookups for non-live processing workflows."),
    handwritten_only: bool = typer.Option(True, "--handwritten-only/--all-classifications", help="Restrict the queue to handwritten IDOC forms."),
    include_token_matches: bool = typer.Option(False, "--include-token-matches", help="Keep rows that already share at least one name token with the expected value."),
    max_count: int | None = typer.Option(100, "--max-count", min=1, help="Maximum number of PDFs to export from the failure queue."),
) -> None:
    rows = list(csv.DictReader(comparison_csv.open()))
    filtered_rows: list[dict[str, str]] = []
    for row in rows:
        if not row.get("expected_name"):
            continue
        if handwritten_only and row.get("classification") != "idoc_housing_application_v1":
            continue
        if row.get("exact_match") == "True":
            continue
        if not include_token_matches and int(row.get("token_overlap", "0") or 0) > 0:
            continue
        filtered_rows.append(row)

    filtered_rows.sort(key=_name_failure_priority)
    if max_count is not None:
        filtered_rows = filtered_rows[:max_count]

    extractor = ApplicationExtractor(template, name_backend=name_backend, enable_idoc_directory=enable_idoc_directory)
    name_field = extractor.field_map.get("applicant.name")
    if name_field is None or name_field.box is None:
        raise typer.Exit("Template does not define applicant.name crop coordinates.")

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str | int | float]] = []

    progress_columns = [SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn()]
    with Progress(*progress_columns, console=console) as progress:
        task_id = progress.add_task("Exporting targeted failure crops", total=len(filtered_rows))
        for row in filtered_rows:
            pdf_path = Path(row["source_pdf"])
            pages = render_pdf_pages(pdf_path, dpi=extractor.template.render_dpi)
            if not pages:
                progress.advance(task_id)
                continue
            page1 = pages[0]
            stem = pdf_path.stem.replace(" ", "_")
            for variant, crop_box in extractor.name_ocr_backend.candidate_boxes(name_field.box).items():
                crop = crop_image(page1, crop_box)
                if crop.size == 0 or mostly_blank(crop):
                    continue
                image_name = f"{stem}__{variant}.png"
                image_path = images_dir / image_name
                cv2.imwrite(str(image_path), crop)
                manifest_rows.append({
                    "source_pdf": str(pdf_path),
                    "image_path": str(image_path),
                    "image_relpath": str(image_path.relative_to(output_dir)),
                    "expected_name": row.get("expected_name", ""),
                    "normalized_expected_name": row.get("normalized_expected_name", ""),
                    "extracted_name": row.get("extracted_name", ""),
                    "normalized_extracted_name": row.get("normalized_extracted_name", ""),
                    "classification": row.get("classification", ""),
                    "exact_match": row.get("exact_match", ""),
                    "first_last_match": row.get("first_last_match", ""),
                    "token_overlap": int(row.get("token_overlap", "0") or 0),
                    "similarity_ratio": float(row.get("similarity_ratio", "0") or 0.0),
                    "confidence": float(row.get("confidence", "0") or 0.0),
                    "source": row.get("source", ""),
                    "variant": variant,
                    "name_backend": name_backend,
                })
            progress.advance(task_id)

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_pdf",
                "image_path",
                "image_relpath",
                "expected_name",
                "normalized_expected_name",
                "extracted_name",
                "normalized_extracted_name",
                "classification",
                "exact_match",
                "first_last_match",
                "token_overlap",
                "similarity_ratio",
                "confidence",
                "source",
                "variant",
                "name_backend",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    console.print(f"Exported targeted failure crops: {len(manifest_rows)}")
    console.print(f"Manifest: {manifest_path}")


@app.command("build-name-review-queue")
def build_name_review_queue(
    failure_manifest: Path = typer.Argument(..., exists=True, readable=True, help="Manifest generated by export-name-failure-crops."),
    output_csv: Path | None = typer.Option(None, "--output-csv", help="CSV path for the editable review queue."),
) -> None:
    rows = list(csv.DictReader(failure_manifest.open()))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_pdf"]].append(row)

    if output_csv is None:
        output_csv = failure_manifest.parent / "review_queue.csv"

    review_rows: list[dict[str, str]] = []
    for source_pdf, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: _variant_priority(item.get("variant", "")))
        preferred = next((item for item in ordered if item.get("variant") == "tight"), ordered[0])
        variant_map = {item.get("variant", ""): item for item in ordered}
        review_rows.append({
            "source_pdf": source_pdf,
            "classification": preferred.get("classification", ""),
            "expected_name": preferred.get("expected_name", ""),
            "normalized_expected_name": preferred.get("normalized_expected_name", ""),
            "extracted_name": preferred.get("extracted_name", ""),
            "normalized_extracted_name": preferred.get("normalized_extracted_name", ""),
            "source": preferred.get("source", ""),
            "confidence": preferred.get("confidence", ""),
            "recommended_variant": preferred.get("variant", "tight"),
            "chosen_variant": preferred.get("variant", "tight"),
            "tight_image_relpath": variant_map.get("tight", {}).get("image_relpath", ""),
            "context_image_relpath": variant_map.get("context", {}).get("image_relpath", ""),
            "wide_image_relpath": variant_map.get("wide", {}).get("image_relpath", ""),
            "suggested_label": preferred.get("expected_name", ""),
            "reviewed_label": "",
            "review_status": "pending",
            "notes": "",
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_pdf",
                "classification",
                "expected_name",
                "normalized_expected_name",
                "extracted_name",
                "normalized_extracted_name",
                "source",
                "confidence",
                "recommended_variant",
                "chosen_variant",
                "tight_image_relpath",
                "context_image_relpath",
                "wide_image_relpath",
                "suggested_label",
                "reviewed_label",
                "review_status",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(review_rows)

    console.print(f"Wrote name review queue: {output_csv}")
    console.print(f"Review items: {len(review_rows)}")


@app.command("apply-name-review-queue")
def apply_name_review_queue(
    review_csv: Path = typer.Argument(..., exists=True, readable=True, help="Editable review queue generated by build-name-review-queue."),
    output_dir: Path = typer.Option(Path("output/name_training_dataset_reviewed"), "--output-dir", help="Directory for curated reviewed training examples."),
    bundle_dir: Path | None = typer.Option(None, "--bundle-dir", help="Root directory containing the failure crop images. Defaults to the review CSV directory."),
) -> None:
    rows = list(csv.DictReader(review_csv.open()))
    bundle_root = bundle_dir or review_csv.parent
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | float]] = []
    for row in rows:
        status = _review_queue_status(row.get("review_status", ""))
        if status != "approved":
            continue

        chosen_variant = str(row.get("chosen_variant") or row.get("recommended_variant") or "tight").strip().lower()
        variant_field = f"{chosen_variant}_image_relpath"
        image_relpath = row.get(variant_field, "") or row.get("tight_image_relpath", "")
        if not image_relpath:
            continue

        label_text = str(row.get("reviewed_label") or row.get("suggested_label") or row.get("expected_name") or "").strip()
        if not label_text:
            continue

        pdf_path = Path(row["source_pdf"])
        applicant_key = _applicant_split_key(pdf_path, label_text)
        split = _dataset_split_for_key(applicant_key)
        source_path = bundle_root / image_relpath
        if not source_path.exists():
            continue
        destination_name = f"{pdf_path.stem.replace(' ', '_')}__{chosen_variant}.png"
        destination_path = images_dir / destination_name
        shutil.copyfile(source_path, destination_path)

        manifest_rows.append({
            "source_pdf": str(pdf_path),
            "applicant_key": applicant_key,
            "image_path": str(destination_path),
            "image_relpath": str(destination_path.relative_to(output_dir)),
            "label_text": label_text,
            "normalized_label": normalize_person_name(label_text),
            "classification": row.get("classification", ""),
            "split": split,
            "variant": chosen_variant,
            "review_status": status,
            "notes": row.get("notes", ""),
        })

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_pdf",
                "applicant_key",
                "image_path",
                "image_relpath",
                "label_text",
                "normalized_label",
                "classification",
                "split",
                "variant",
                "review_status",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    for split in ["train", "val", "test"]:
        split_rows = [row for row in manifest_rows if row["split"] == split]
        split_path = output_dir / f"{split}.jsonl"
        with split_path.open("w") as handle:
            for row in split_rows:
                handle.write(json.dumps({"image": row["image_relpath"], "text": row["label_text"], "applicant_key": row["applicant_key"]}) + "\n")

    console.print(f"Wrote reviewed name training bundle: {manifest_path}")
    console.print(f"Approved reviewed examples: {len(manifest_rows)}")


@app.command("train-name-ocr-model")
def train_name_ocr_model_command(
    dataset_dir: Path = typer.Argument(..., exists=True, readable=True, help="Directory created by export-name-training-data."),
    output_dir: Path = typer.Option(Path("output/name_ocr_model"), "--output-dir", help="Directory for the fine-tuned model."),
    base_model: str = typer.Option("microsoft/trocr-small-handwritten", "--base-model", help="Pretrained TrOCR checkpoint to fine-tune."),
    num_train_epochs: float = typer.Option(6.0, "--epochs", min=0.1, help="Number of training epochs."),
    per_device_train_batch_size: int = typer.Option(4, "--train-batch-size", min=1, help="Per-device training batch size."),
    per_device_eval_batch_size: int = typer.Option(4, "--eval-batch-size", min=1, help="Per-device evaluation batch size."),
    learning_rate: float = typer.Option(5e-5, "--learning-rate", min=1e-7, help="Learning rate."),
    max_target_length: int = typer.Option(32, "--max-target-length", min=8, help="Maximum decoded name length."),
) -> None:
    train_name_ocr_model(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        base_model=base_model,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        max_target_length=max_target_length,
    )
    console.print(f"Saved fine-tuned name OCR model to: {output_dir}")


@app.command("export-supervision-crops")
def export_supervision_crops(
    comparison_csv: Path = typer.Argument(..., exists=True, readable=True, help="Comparison CSV generated by compare-ground-truth."),
    output_dir: Path = typer.Option(Path("output/supervision_crops"), "--output-dir", help="Directory for exported supervision-number crops."),
    include_categories: str = typer.Option(
        "single_digit_ocr_mismatch,multi_digit_ocr_mismatch,expected_present_extracted_blank,possible_workbook_update_or_document_disagreement",
        "--categories",
        help="Comma-separated mismatch categories to export.",
    ),
) -> None:
    categories = {item.strip() for item in include_categories.split(",") if item.strip()}
    rows = list(csv.DictReader(comparison_csv.open()))
    extractor = IdentityExtractor()
    written = 0
    manifest_rows: list[dict[str, str]] = []

    for row in rows:
        category = row.get("mismatch_category", "")
        if category not in categories:
            continue
        pdf_path = Path(row["source_pdf"])
        classification, subtype, crops = extractor.debug_supervision_crops(pdf_path)
        stem = pdf_path.stem.replace(" ", "_")
        target_dir = output_dir / category / subtype / stem
        target_dir.mkdir(parents=True, exist_ok=True)
        for label, image in crops.items():
            cv2.imwrite(str(target_dir / f"{label}.png"), image)
            written += 1

        manifest_rows.append({
            "source_pdf": str(pdf_path),
            "classification": classification,
            "layout_subtype": subtype,
            "mismatch_category": category,
            "expected_name": row.get("expected_name", ""),
            "extracted_name": row.get("extracted_name", ""),
            "expected_supervision_number": row.get("expected_supervision_number", ""),
            "extracted_supervision_number": row.get("extracted_supervision_number", ""),
            "crop_dir": str(target_dir),
        })

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_pdf",
                "classification",
                "layout_subtype",
                "mismatch_category",
                "expected_name",
                "extracted_name",
                "expected_supervision_number",
                "extracted_supervision_number",
                "crop_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    console.print(f"Exported {written} crop images to {output_dir}")
    console.print(f"Manifest: {manifest_path}")


@app.command("analyze-supervision-failures")
def analyze_supervision_failures(
    comparison_csv: Path = typer.Argument(..., exists=True, readable=True, help="Comparison CSV generated by compare-ground-truth."),
    output_csv: Path = typer.Option(Path("output/supervision_failure_analysis.csv"), "--output-csv", help="CSV path for failure analysis output."),
) -> None:
    rows = list(csv.DictReader(comparison_csv.open()))
    extractor = IdentityExtractor()
    analysis_rows: list[dict[str, str | int | bool]] = []

    failure_rows = [row for row in rows if row.get("expected_name") and row.get("number_match") != "True"]
    for row in failure_rows:
        pdf_path = Path(row["source_pdf"])
        classification_name, layout_subtype, candidate_counts = collect_supervision_candidate_counts(extractor, pdf_path)
        expected_number = row.get("expected_supervision_number", "")
        extracted_number = row.get("extracted_supervision_number", "")
        expected_present = expected_number in candidate_counts
        top_candidate, top_count = ("", 0)
        if candidate_counts:
            top_candidate, top_count = max(
                candidate_counts.items(),
                key=lambda item: (_supervision_number_score(item[0]), item[1], item[0]),
            )
        label_any_page = label_found_on_any_page(extractor, pdf_path, classification_name)
        likely_cause = classify_failure_cause(
            row.get("mismatch_category", ""),
            expected_present,
            extracted_number,
            expected_number,
            label_any_page,
        )
        analysis_rows.append({
            "source_pdf": str(pdf_path),
            "classification": classification_name,
            "layout_subtype": layout_subtype,
            "mismatch_category": row.get("mismatch_category", ""),
            "expected_supervision_number": expected_number,
            "extracted_supervision_number": extracted_number,
            "expected_present_in_candidates": expected_present,
            "expected_candidate_count": candidate_counts.get(expected_number, 0),
            "extracted_candidate_count": candidate_counts.get(extracted_number, 0),
            "top_candidate": top_candidate,
            "top_candidate_count": top_count,
            "label_found_any_page": label_any_page,
            "likely_cause": likely_cause,
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_pdf",
                "classification",
                "layout_subtype",
                "mismatch_category",
                "expected_supervision_number",
                "extracted_supervision_number",
                "expected_present_in_candidates",
                "expected_candidate_count",
                "extracted_candidate_count",
                "top_candidate",
                "top_candidate_count",
                "label_found_any_page",
                "likely_cause",
            ],
        )
        writer.writeheader()
        writer.writerows(analysis_rows)

    cause_counts = Counter(str(row["likely_cause"]) for row in analysis_rows)
    console.print(f"Wrote failure analysis CSV: {output_csv}")
    for cause, count in cause_counts.most_common():
        console.print(f"{cause}: {count}")


@app.command("evaluate-all")
def evaluate_all(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="PDF file or directory to evaluate."),
    output_csv: Path = typer.Option(Path("output/full_corpus_eval.csv"), "--output-csv", help="CSV path for evaluation results."),
    template: Path = typer.Option(Path("config/idoc_application_template.yml"), "--template", exists=True, readable=True),
    workbook: Path | None = typer.Option(None, "--workbook", exists=True, readable=True, help="Optional workbook for ground-truth comparison."),
    sheet_name: str = typer.Option("2026", "--sheet", help="Workbook sheet to use."),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Evaluate only the first N PDFs."),
    name_backend: str = typer.Option("rapid_ensemble", "--name-backend", help=f"Applicant-name OCR backend. Options: {', '.join(available_name_ocr_backends())}."),
    enable_idoc_directory: bool = typer.Option(False, "--enable-idoc-directory", help="Allow Processed Apps spreadsheet lookups for non-live processing workflows."),
) -> None:
    """Evaluate the full extraction pipeline on all PDF types (IDOC, Jotform, RS packets, bundles)."""
    extractor = ApplicationExtractor(template, name_backend=name_backend, enable_idoc_directory=enable_idoc_directory)
    pdf_paths = discover_pdfs(input_path)
    if limit is not None:
        pdf_paths = pdf_paths[:limit]
    if not pdf_paths:
        raise typer.Exit("No PDF files found.")

    truth_rows = load_ground_truth_rows(workbook, sheet_name=sheet_name) if workbook else []
    rows: list[dict[str, str]] = []

    progress_columns = [SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn()]

    fieldnames = [
        "source_pdf", "page_count", "document_classification", "template",
        "supported", "name", "idoc_number", "fields_filled", "fields_total",
        "expected_name", "name_match", "expected_number", "number_match", "error",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(output_csv, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    csv_file.flush()

    try:
        with Progress(*progress_columns, console=console) as progress:
            task_id = progress.add_task("Evaluating full corpus", total=len(pdf_paths))
            for pdf_path in pdf_paths:
                row: dict[str, str] = {"source_pdf": pdf_path.name, "error": ""}
                try:
                    result = extractor.extract_pdf(pdf_path, include_raw_text=True)
                    e = result.get("extracted", {})
                    fields = result.get("field_results", {})
                    row["page_count"] = str(result.get("page_count", len(fitz.open(pdf_path))))
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

                rows.append(row)
                writer.writerow(row)
                csv_file.flush()
                progress.advance(task_id)
    finally:
        csv_file.close()

    # Summary
    total = len(rows)
    by_class: dict[str, int] = {}
    for r in rows:
        cls = r.get("document_classification", "unknown")
        by_class[cls] = by_class.get(cls, 0) + 1
    has_name = sum(1 for r in rows if r.get("name"))
    has_error = sum(1 for r in rows if r.get("error"))

    console.print(f"\n{'='*60}")
    console.print(f"Full Corpus Evaluation ({total} PDFs)")
    console.print(f"{'='*60}")
    console.print(f"\nBy classification:")
    for cls in sorted(by_class):
        pct = by_class[cls] / total * 100
        console.print(f"  {cls:40s}: {by_class[cls]:4d} ({pct:5.1f}%)")
    console.print(f"\nName extracted: {has_name}/{total}")
    console.print(f"Errors: {has_error}/{total}")
    console.print(f"\nCSV: {output_csv}")


if __name__ == "__main__":
    app()
