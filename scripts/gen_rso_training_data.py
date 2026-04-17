#!/usr/bin/env python3
"""Generate training data for the RSO (Registered Sex Offender) checkbox classifier.

Crops the RSO checkbox region from page 1 of each 4-page IDOC housing application
and labels using ground truth from the processed applications spreadsheet.

The RSO checkbox region includes both the "yes" and "no" boxes from the template:
  yes: [0.643, 0.552, 0.663, 0.575]
  no:  [0.698, 0.552, 0.718, 0.575]

We crop a wider area that covers both checkboxes plus surrounding context.

Usage:
    python scripts/gen_rso_training_data.py \
        --output-dir output/rso_training_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR_2026 = ROOT / "IDOC" / "Data" / "PROCESSED APPS 2026"
DATA_DIR_2025 = ROOT / "IDOC" / "Data" / "PROCESSED APPS 2025"
WORKBOOK = ROOT / "IDOC" / "Data" / "1. Processed Apps List.xlsx"

# Wider crop region covering both yes/no checkboxes plus the question text.
# Gives the model enough context to see "Do you need to register as a sex offender?"
# plus both checkbox answers.
RSO_CROP_BOX = (0.50, 0.540, 0.75, 0.585)

# Tighter crops centered on the yes and no checkboxes individually
RSO_YES_BOX = (0.630, 0.545, 0.675, 0.580)
RSO_NO_BOX = (0.685, 0.545, 0.730, 0.580)

RENDER_DPI = 225


def _split_bucket(name: str) -> str:
    """Deterministic train/val/test split by name hash (80/10/10)."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    elif h < 90:
        return "val"
    return "test"


def _crop_region(page_image, box: tuple[float, float, float, float]):
    """Crop a normalized box from a page image (numpy array)."""
    import numpy as np

    h, w = page_image.shape[:2]
    x1 = int(box[0] * w)
    y1 = int(box[1] * h)
    x2 = int(box[2] * w)
    y2 = int(box[3] * h)
    # Clamp
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return page_image[y1:y2, x1:x2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RSO checkbox training crops")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    args = parser.parse_args()

    import cv2
    import numpy as np
    import pymupdf

    from rising_sun.ground_truth import load_ground_truth_rows, match_ground_truth
    from rising_sun.identity import clean_pdf_stem_name

    output_dir: Path = args.output_dir
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth from both sheets
    rows_2025 = load_ground_truth_rows(args.workbook, sheet_name="2025")
    rows_2026 = load_ground_truth_rows(args.workbook, sheet_name="2026")
    all_rows = rows_2025 + rows_2026

    rso_by_name = {}
    for row in all_rows:
        rso_by_name[row.normalized_name] = row.is_rso
        if row.first_last_key:
            rso_by_name[row.first_last_key] = row.is_rso

    logger.info(f"Ground truth: {len(rows_2025)} from 2025, {len(rows_2026)} from 2026")
    logger.info(f"RSO positive: {sum(1 for r in all_rows if r.is_rso)}")

    # Collect PDFs from both directories
    pdfs: list[tuple[Path, str]] = []  # (path, year)
    for data_dir, year, gt_rows in [
        (DATA_DIR_2026, "2026", rows_2026),
        (DATA_DIR_2025, "2025", rows_2025),
    ]:
        if data_dir.exists():
            for pdf in sorted(data_dir.glob("*.pdf")):
                pdfs.append((pdf, year))
            logger.info(f"Found {sum(1 for p, y in pdfs if y == year)} PDFs in {data_dir.name}")

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    stats = {"total": 0, "matched": 0, "rso_yes": 0, "rso_no": 0, "skipped_pages": 0}

    for i, (pdf_path, year) in enumerate(pdfs):
        if args.limit and stats["total"] >= args.limit:
            break

        if (i + 1) % 100 == 0 or i == 0:
            logger.info(f"[{i + 1}/{len(pdfs)}] {pdf_path.name}")

        try:
            doc = pymupdf.open(pdf_path)
            if len(doc) != 4:
                stats["skipped_pages"] += 1
                doc.close()
                continue

            # Render page 1
            page = doc[0]
            mat = pymupdf.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            doc.close()
        except Exception as e:
            logger.warning(f"Failed to render {pdf_path.name}: {e}")
            continue

        # Match to ground truth
        name = clean_pdf_stem_name(pdf_path)
        gt_rows_for_year = rows_2025 if year == "2025" else rows_2026
        gt_match = match_ground_truth(pdf_path, name, gt_rows_for_year)

        if gt_match is None:
            # Try matching against all rows
            gt_match = match_ground_truth(pdf_path, name, all_rows)

        if gt_match is None:
            continue

        stats["matched"] += 1
        label = "yes" if gt_match.is_rso else "no"
        if gt_match.is_rso:
            stats["rso_yes"] += 1
        else:
            stats["rso_no"] += 1

        stats["total"] += 1
        stem = pdf_path.stem
        split = _split_bucket(name)

        # Generate crop variants
        crop_variants = [
            ("wide", RSO_CROP_BOX),
            ("yes_box", RSO_YES_BOX),
            ("no_box", RSO_NO_BOX),
        ]

        for variant_name, crop_box in crop_variants:
            crop = _crop_region(img, crop_box)
            if crop.size == 0:
                continue

            fname = f"{stem}_{variant_name}.png"
            img_path = images_dir / fname
            cv2.imwrite(str(img_path), crop)

            row = {
                "image": f"images/{fname}",
                "text": label,
                "task": "rso",
                "variant": variant_name,
                "source_year": year,
            }
            splits[split].append(row)

    # Write JSONL split files
    for split_name, split_rows in splits.items():
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for row in split_rows:
                f.write(json.dumps({"image": row["image"], "text": row["text"], "task": row["task"]}) + "\n")
        n_yes = sum(1 for r in split_rows if r["text"] == "yes")
        n_no = sum(1 for r in split_rows if r["text"] == "no")
        logger.info(f"{split_name}.jsonl: {len(split_rows)} samples ({n_yes} yes, {n_no} no)")

    # Write full manifest
    all_samples = splits["train"] + splits["val"] + splits["test"]
    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for row in all_samples:
            f.write(json.dumps(row) + "\n")

    logger.info(
        f"Done: {len(all_samples)} total crops from {stats['matched']} matched PDFs "
        f"(yes={stats['rso_yes']}, no={stats['rso_no']}, "
        f"skipped_pages={stats['skipped_pages']})"
    )


if __name__ == "__main__":
    main()
