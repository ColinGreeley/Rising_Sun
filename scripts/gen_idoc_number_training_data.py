#!/usr/bin/env python3
"""Generate IDOC number region crops with ground-truth labels for TrOCR fine-tuning.

For each 4-page PDF with a known IDOC number (from the spreadsheet), generates
multiple crop variants of the IDOC# field region at different DPIs and with
different augmentations.  Output is a flat directory of images plus JSONL splits.

Usage:
    python scripts/gen_idoc_number_training_data.py \
        --output-dir output/idoc_number_training
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rising_sun.identity import clean_pdf_stem_name
from rising_sun.idoc_lookup import IdocDirectory
from rising_sun.pdf import render_pdf_page

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "IDOC" / "Data" / "PROCESSED APPS 2026"

# Region crop boxes (normalised 0-1 coordinates)
CROP_DEFAULT = (0.62, 0.24, 0.95, 0.34)
CROP_TIGHT = (0.72, 0.238, 0.90, 0.298)
CROP_MID = (0.67, 0.238, 0.92, 0.31)


def _crop_region(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
    return img[y1:y2, x1:x2]


def _apply_clahe(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _apply_binary(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _split_bucket(name: str) -> str:
    """Deterministic train/val/test split by name hash (80/10/10)."""
    h = int(hashlib.sha256(name.encode()).hexdigest()[:4], 16) % 100
    if h < 80:
        return "train"
    elif h < 90:
        return "val"
    else:
        return "test"


def generate_crops_for_pdf(
    pdf_path: Path,
    idoc_number: str,
    images_dir: Path,
) -> list[dict]:
    """Generate multiple crop variants and return manifest rows."""
    stem = pdf_path.stem.replace(" ", "_")
    rows = []

    variants: list[tuple[str, int, tuple, str | None]] = [
        # (variant_name, dpi, crop_box, augmentation)
        ("tight_225", 225, CROP_TIGHT, None),
        ("default_225", 225, CROP_DEFAULT, None),
        ("mid_225", 225, CROP_MID, None),
        ("tight_225_clahe", 225, CROP_TIGHT, "clahe"),
        ("tight_225_binary", 225, CROP_TIGHT, "binary"),
        ("tight_300", 300, CROP_TIGHT, None),
        ("default_300", 300, CROP_DEFAULT, None),
        ("tight_300_clahe", 300, CROP_TIGHT, "clahe"),
        ("tight_300_binary", 300, CROP_TIGHT, "binary"),
        ("tight_400", 400, CROP_TIGHT, None),
        ("tight_400_binary", 400, CROP_TIGHT, "binary"),
    ]

    # Cache rendered pages by DPI
    page_cache: dict[int, np.ndarray] = {}

    for variant_name, dpi, crop_box, augmentation in variants:
        if dpi not in page_cache:
            page_cache[dpi] = render_pdf_page(pdf_path, dpi=dpi, page_number=0)

        page_img = page_cache[dpi]
        crop = _crop_region(page_img, crop_box)

        if augmentation == "clahe":
            crop = _apply_clahe(crop)
        elif augmentation == "binary":
            crop = _apply_binary(crop)

        filename = f"{stem}__{variant_name}.png"
        out_path = images_dir / filename
        cv2.imwrite(str(out_path), crop)

        rows.append({
            "image": f"images/{filename}",
            "text": idoc_number,
            "source_pdf": pdf_path.name,
            "variant": variant_name,
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IDOC number training crops")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    directory = IdocDirectory()
    pdfs = sorted(args.data_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdfs)} total PDFs")

    # Filter to 4-page PDFs with known IDOC numbers
    eligible: list[tuple[Path, str]] = []
    for pdf_path in pdfs:
        doc = pymupdf.open(pdf_path)
        n_pages = len(doc)
        doc.close()
        if n_pages != 4:
            continue
        name = clean_pdf_stem_name(pdf_path)
        nums = directory.lookup_by_name(name)
        if nums and len(nums[0]) >= 5:
            eligible.append((pdf_path, nums[0]))

    logger.info(f"Eligible (4-page + known IDOC#): {len(eligible)}")

    if args.limit:
        eligible = eligible[: args.limit]

    all_rows: list[dict] = []
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for i, (pdf_path, idoc_number) in enumerate(eligible):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info(f"[{i + 1}/{len(eligible)}] {pdf_path.name} → {idoc_number}")

        try:
            rows = generate_crops_for_pdf(pdf_path, idoc_number, images_dir)
        except Exception as e:
            logger.warning(f"Failed {pdf_path.name}: {e}")
            continue

        name = clean_pdf_stem_name(pdf_path)
        split = _split_bucket(name)
        for row in rows:
            row["split"] = split
            splits[split].append(row)
        all_rows.extend(rows)

    # Write JSONL split files
    for split_name, split_rows in splits.items():
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for row in split_rows:
                f.write(json.dumps({"image": row["image"], "text": row["text"]}) + "\n")
        logger.info(f"{split_name}.jsonl: {len(split_rows)} samples")

    # Write full manifest
    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    logger.info(
        f"Done: {len(all_rows)} total crops from {len(eligible)} PDFs "
        f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})"
    )


if __name__ == "__main__":
    main()
