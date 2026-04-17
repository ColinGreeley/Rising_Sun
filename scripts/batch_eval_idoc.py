#!/usr/bin/env python3
"""Batch evaluation of IDOC number extraction + database verification.

Processes all 4-page PDFs using a multi-strategy extraction pipeline:
1. Region crop OCR (targets the IDOC# field directly)
2. Full-page OCR with regex extraction
3. Augmented OCR (CLAHE, higher DPI)
4. 5-digit minimum filter to reject truncated reads
5. Fuzzy matching against the known IDOC directory spreadsheet
6. Name-based spreadsheet fallback when OCR fails entirely

Usage:
    python scripts/batch_eval_idoc.py [--limit N] [--output PATH]
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import pymupdf
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rising_sun.identity import (
    clean_pdf_stem_name,
    normalize_person_name,
    normalize_supervision_candidates,
    person_name_key,
)
from rising_sun.idoc_lookup import (
    IdocDirectory,
    filter_candidates_by_length,
)
from rising_sun.ocr import RapidOcrBackend
from rising_sun.pdf import render_pdf_page

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "IDOC" / "Data" / "PROCESSED APPS 2026"
DATA_DIR_2025 = Path(__file__).resolve().parent.parent / "IDOC" / "Data" / "PROCESSED APPS 2025"
IDOC_DETAIL_URL = "https://www.idoc.idaho.gov/content/prisons/resident-client-search/details/{idoc_number}"
MAX_CANDIDATES_TO_TRY = 6
REQUEST_DELAY = 0.3  # seconds between HTTP requests to be polite
PER_PDF_TIMEOUT = 90  # seconds max per PDF (augmentation can be very slow)

# Region crop boxes for the IDOC# field on page 1 of the housing application
IDOC_CROP_DEFAULT = (0.62, 0.24, 0.95, 0.34)
IDOC_CROP_TIGHT = (0.72, 0.238, 0.90, 0.298)

TROCR_MODEL_DIR = Path(__file__).resolve().parent.parent / "output" / "trocr_model_v3b"
NUMBER_MODEL_DIR = Path(__file__).resolve().parent.parent / "output" / "trocr_number_model_v2"
NAME_MODEL_DIR = Path(__file__).resolve().parent.parent / "output" / "trocr_name_model_v2"

# Name field crop boxes
NAME_CROP_TIGHT = (0.14, 0.248, 0.60, 0.282)
NAME_CROP_CONTEXT = (0.10, 0.238, 0.64, 0.290)
NAME_CROP_WIDE = (0.06, 0.230, 0.68, 0.300)

# ---------------------------------------------------------------------------
# Shared OCR extraction (mirrors backend logic)
# ---------------------------------------------------------------------------

_IDOC_CAPTURE_CC = r"[A-Za-z0-9.,\-/\\()\[\]{}!|&$%'\" ]"
_IDOC_PATTERNS = [
    re.compile(rf"ID[O0]C\s*#\s*:?\s*({_IDOC_CAPTURE_CC}{{4,15}})", re.IGNORECASE),
    re.compile(rf"IDOC\s*(?:or\s*LE\s*)?#\s*:?\s*({_IDOC_CAPTURE_CC}{{4,15}})", re.IGNORECASE),
    re.compile(rf"IDOC\s*Number\s*:?\s*({_IDOC_CAPTURE_CC}{{4,15}})", re.IGNORECASE),
]
_STANDALONE_NUMBER_RE = re.compile(r"\b(\d{5,6})\b")

_OCR_DIGIT_MAP = str.maketrans("OoIlLJZSBG/\\|!", "00111125861111")


def _ocr_to_digits(value: str) -> str | None:
    cleaned = re.sub(r"[^A-Za-z0-9/\\|!]", "", value)
    digits = cleaned.translate(_OCR_DIGIT_MAP)
    if digits.isdigit() and 4 <= len(digits) <= 8:
        return digits
    return None


def _extract_raw_from_text(text: str) -> str | None:
    for pattern in _IDOC_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).strip()
    return None


def _extract_candidates_from_text(text: str) -> tuple[str | None, list[str]]:
    raw = _extract_raw_from_text(text)
    candidates: list[str] = []
    if raw:
        if raw.isdigit() and 4 <= len(raw) <= 8:
            candidates = [raw]
        else:
            candidates = normalize_supervision_candidates(raw)
            if not candidates:
                single = _ocr_to_digits(raw)
                if single:
                    candidates = [single]
    # Also extract standalone 5-6 digit numbers from the full text
    for m in _STANDALONE_NUMBER_RE.finditer(text):
        num = m.group(1)
        if num not in candidates:
            candidates.append(num)
    if candidates:
        return raw, candidates
    return raw, []


# ---------------------------------------------------------------------------
# TrOCR inference helper
# ---------------------------------------------------------------------------

def _trocr_extract_candidates(
    page_image: np.ndarray,
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    directory: IdocDirectory | None = None,
) -> list[str]:
    """Run fine-tuned TrOCR on tight crops with multi-beam and return digit candidates.

    If *directory* is provided, snaps raw OCR digits to known IDOC numbers
    via fuzzy matching before returning.
    """
    raw_digits: list[str] = []
    for box in [IDOC_CROP_TIGHT, IDOC_CROP_DEFAULT]:
        crop = _crop_idoc_region(page_image, box)
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
        if next(model.parameters()).is_cuda:
            pixel_values = pixel_values.cuda()
        with torch.no_grad():
            ids = model.generate(
                pixel_values,
                max_new_tokens=12,
                num_beams=8,
                num_return_sequences=4,
            )
        preds = processor.batch_decode(ids, skip_special_tokens=True)
        for pred in preds:
            digits = re.sub(r"[^0-9]", "", pred.strip())
            if digits and 5 <= len(digits) <= 6 and digits not in raw_digits:
                raw_digits.append(digits)

    if directory is None:
        return raw_digits

    # Snap against known directory
    snapped: list[str] = []
    remaining: list[str] = []
    for d in raw_digits:
        if directory.is_known(d):
            if d not in snapped:
                snapped.append(d)
        else:
            fuzzy = directory.fuzzy_match(d)
            if fuzzy:
                for f in fuzzy:
                    if f not in snapped:
                        snapped.append(f)
            else:
                remaining.append(d)
    for d in remaining:
        if d not in snapped:
            snapped.append(d)
    return snapped


def _trocr_extract_name(
    page_image: np.ndarray,
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    directory: IdocDirectory | None = None,
) -> str | None:
    """OCR the applicant name field using TrOCR with multi-beam candidates."""
    all_name_candidates: list[str] = []
    for box in [NAME_CROP_TIGHT, NAME_CROP_CONTEXT, NAME_CROP_WIDE]:
        crop = _crop_idoc_region(page_image, box)
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
        if next(model.parameters()).is_cuda:
            pixel_values = pixel_values.cuda()
        with torch.no_grad():
            ids = model.generate(
                pixel_values,
                max_new_tokens=40,
                num_beams=4,
                num_return_sequences=3,
            )
        preds = processor.batch_decode(ids, skip_special_tokens=True)
        for pred in preds:
            pred = pred.strip()
            if pred and len(pred) >= 3 and " " in pred and pred not in all_name_candidates:
                all_name_candidates.append(pred)
    if not all_name_candidates:
        return None
    # Prefer directory-confirmed name
    if directory:
        for cand in all_name_candidates:
            numbers = directory.lookup_by_name(cand)
            if numbers:
                return cand
    return all_name_candidates[0]


# ---------------------------------------------------------------------------
# Region crop helpers
# ---------------------------------------------------------------------------

def _crop_idoc_region(
    page_image: np.ndarray,
    box: tuple[float, float, float, float] = IDOC_CROP_DEFAULT,
) -> np.ndarray:
    """Crop the IDOC# field region from a full-page image."""
    h, w = page_image.shape[:2]
    x1 = int(box[0] * w)
    y1 = int(box[1] * h)
    x2 = int(box[2] * w)
    y2 = int(box[3] * h)
    return page_image[y1:y2, x1:x2]


def _extract_from_region_crop(
    ocr_backend: RapidOcrBackend,
    page_image: np.ndarray,
) -> tuple[str | None, list[str]]:
    """OCR just the IDOC# region and extract candidates."""
    crop = _crop_idoc_region(page_image, IDOC_CROP_DEFAULT)
    ocr_text = ocr_backend.read_text(crop, multiline=True).text
    raw, candidates = _extract_candidates_from_text(ocr_text)
    if candidates:
        return raw, candidates

    crop_tight = _crop_idoc_region(page_image, IDOC_CROP_TIGHT)
    ocr_text2 = ocr_backend.read_text(crop_tight, multiline=True).text
    raw2, candidates2 = _extract_candidates_from_text(ocr_text2)
    if candidates2:
        return raw2, candidates2

    return raw or raw2, []


# ---------------------------------------------------------------------------
# Image augmentations for retry
# ---------------------------------------------------------------------------

def _augmented_images(page_image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Generate augmented variants of a page image for OCR retry."""
    variants: list[tuple[str, np.ndarray]] = []

    # 1. Contrast boost (CLAHE)
    gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(("clahe", cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)))

    # 2. Simple binarization (Otsu)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("binary", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)))

    # 3. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
    variants.append(("adaptive", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)))

    # 4. Sharpened
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(page_image, -1, kernel)
    variants.append(("sharpen", sharpened))

    return variants


def _extract_with_augmentations(
    ocr_backend: RapidOcrBackend,
    pdf_path: Path,
    base_candidates: list[str],
    base_raw: str | None,
    base_image: np.ndarray | None = None,
) -> tuple[str | None, list[str], str]:
    """Try region crops and augmentations to extract IDOC candidates.

    Strategy order:
    1. Region crop on existing 225dpi image (default + tight boxes)
    2. CLAHE on 225dpi crop
    3. Region crop at 300dpi
    4. CLAHE on 300dpi crop
    5. Full-page CLAHE on 225dpi (fallback)

    Returns (raw_capture, candidates, method_used).
    """
    if base_candidates:
        return base_raw, base_candidates, "base_225dpi"

    all_raw: list[str | None] = []

    # 1. Region crop on existing 225dpi image
    if base_image is not None:
        raw, cands = _extract_from_region_crop(ocr_backend, base_image)
        if cands:
            return raw, cands, "region_crop_225dpi"
        if raw:
            all_raw.append(raw)

        # 2. CLAHE on 225dpi crop
        crop = _crop_idoc_region(base_image, IDOC_CROP_DEFAULT)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        clahe_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        ocr_text = ocr_backend.read_text(clahe_img, multiline=True).text
        raw, cands = _extract_candidates_from_text(ocr_text)
        if cands:
            return raw, cands, "clahe_crop_225dpi"
        if raw:
            all_raw.append(raw)

    # 3. Region crop at 300dpi
    page_300 = render_pdf_page(pdf_path, dpi=300, page_number=0)
    raw, cands = _extract_from_region_crop(ocr_backend, page_300)
    if cands:
        return raw, cands, "region_crop_300dpi"
    if raw:
        all_raw.append(raw)

    # 4. CLAHE on 300dpi crop
    crop_300 = _crop_idoc_region(page_300, IDOC_CROP_DEFAULT)
    gray = cv2.cvtColor(crop_300, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    clahe_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    ocr_text = ocr_backend.read_text(clahe_img, multiline=True).text
    raw, cands = _extract_candidates_from_text(ocr_text)
    if cands:
        return raw, cands, "clahe_crop_300dpi"
    if raw:
        all_raw.append(raw)

    # 5. Region crop at 400dpi
    page_400 = render_pdf_page(pdf_path, dpi=400, page_number=0)
    raw, cands = _extract_from_region_crop(ocr_backend, page_400)
    if cands:
        return raw, cands, "region_crop_400dpi"
    if raw:
        all_raw.append(raw)

    # 6. Binary threshold on 400dpi crop
    crop_400 = _crop_idoc_region(page_400, IDOC_CROP_DEFAULT)
    gray = cv2.cvtColor(crop_400, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    ocr_text = ocr_backend.read_text(bin_img, multiline=True).text
    raw, cands = _extract_candidates_from_text(ocr_text)
    if cands:
        return raw, cands, "binary_crop_400dpi"
    if raw:
        all_raw.append(raw)

    first_raw = next((r for r in all_raw if r), None)
    return first_raw, [], "augmentation_failed"


# ---------------------------------------------------------------------------
# IDOC website parsing (mirrors backend)
# ---------------------------------------------------------------------------

from lxml import html as lxml_html


def _parse_idoc_detail_page(page_html: str) -> dict[str, Any]:
    tree = lxml_html.fromstring(page_html)
    region = tree.xpath('//div[contains(@class, "region-content")]')
    if not region:
        return {}
    content_text = region[0].text_content()
    info: dict[str, Any] = {}

    name_match = re.search(r"Resident/Client Search\s+(.+?)\s*IDOC\s*#", content_text, re.DOTALL)
    if name_match:
        info["name"] = re.sub(r"\s+", " ", name_match.group(1)).strip()

    idoc_match = re.search(r"IDOC\s*#:\s*(\d+)", content_text)
    if idoc_match:
        info["idoc_number"] = idoc_match.group(1)

    return info


def _names_match(name_a: str, name_b: str) -> bool:
    key_a = person_name_key(name_a)
    key_b = person_name_key(name_b)
    if key_a is None or key_b is None:
        return False
    return key_a == key_b


def _name_similarity(name_a: str, name_b: str) -> str:
    if _names_match(name_a, name_b):
        return "exact"
    norm_a = normalize_person_name(name_a).split()
    norm_b = normalize_person_name(name_b).split()
    if not norm_a or not norm_b:
        return "none"
    if norm_a[-1] == norm_b[-1]:
        return "partial"
    if norm_a[0] == norm_b[0]:
        return "partial"
    return "none"


# ---------------------------------------------------------------------------
# Main batch pipeline
# ---------------------------------------------------------------------------

async def process_pdf(
    pdf_path: Path,
    ocr_backend: RapidOcrBackend,
    client: httpx.AsyncClient,
    directory: IdocDirectory,
    number_processor: TrOCRProcessor | None = None,
    number_model: VisionEncoderDecoderModel | None = None,
    name_processor: TrOCRProcessor | None = None,
    name_model: VisionEncoderDecoderModel | None = None,
) -> dict[str, Any]:
    """Process a single PDF: extract IDOC candidates, look up, verify.

    Uses multi-strategy extraction:
    1. Embedded text → 2. Full-page OCR 225dpi → 3. Region crop + augmentations
    Then: 5-digit filter → fuzzy match against known directory → name fallback
    """
    filename_name = clean_pdf_stem_name(pdf_path)
    result: dict[str, Any] = {
        "filename": pdf_path.name,
        "filename_name": filename_name,
    }

    # Check page count
    doc = pymupdf.open(pdf_path)
    page_count = len(doc)
    result["page_count"] = page_count

    # Try embedded text
    raw_capture = None
    candidates: list[str] = []
    for page in doc:
        text = page.get_text()
        if len(text.strip()) > 50:
            raw_capture, candidates = _extract_candidates_from_text(text)
            if candidates:
                break
    doc.close()

    # Fall back to OCR
    method = "embedded_text"
    page_image_225 = None
    if not candidates:
        page_image_225 = render_pdf_page(pdf_path, dpi=225, page_number=0)
        ocr_text = ocr_backend.read_text(page_image_225, multiline=True).text
        raw_capture, candidates = _extract_candidates_from_text(ocr_text)
        method = "base_225dpi"

    # If still no candidates, try region crops + augmentations
    if not candidates:
        raw_capture, candidates, method = _extract_with_augmentations(
            ocr_backend, pdf_path, candidates, raw_capture, base_image=page_image_225
        )

    # TrOCR: add fine-tuned model predictions as additional candidates
    if number_processor is not None and number_model is not None:
        trocr_image = page_image_225 if page_image_225 is not None else render_pdf_page(pdf_path, dpi=225, page_number=0)
        trocr_cands = _trocr_extract_candidates(trocr_image, number_processor, number_model, directory)
        for tc in trocr_cands:
            if tc not in candidates:
                candidates.append(tc)
        if trocr_cands and not method.startswith("trocr"):
            method = f"{method}+trocr"

    # TrOCR name extraction for cross-check
    ocr_name = None
    if name_processor is not None and name_model is not None:
        trocr_image = page_image_225 if page_image_225 is not None else render_pdf_page(pdf_path, dpi=225, page_number=0)
        ocr_name = _trocr_extract_name(trocr_image, name_processor, name_model, directory)

    # Filter: require 5+ digit candidates
    candidates = filter_candidates_by_length(candidates)

    # Fuzzy match against known IDOC directory
    # Use OCR-extracted name as additional signal when available
    lookup_name = filename_name
    if ocr_name and not lookup_name:
        lookup_name = ocr_name
    directory_match = None
    if candidates:
        best_num, match_method = directory.best_match(candidates, lookup_name)
        if best_num:
            directory_match = best_num
            if best_num not in candidates:
                candidates.insert(0, best_num)
            method = f"{method}+{match_method}"

    # Name-based spreadsheet fallback:
    # - When OCR found nothing at all
    # - When OCR candidates exist but none confirmed by directory (likely wrong numbers)
    fallback_name = filename_name or ocr_name
    if fallback_name and not directory_match:
        fb_number, fb_name = directory.name_fallback(fallback_name)
        if fb_number:
            if not candidates:
                candidates = [fb_number]
                method = "spreadsheet_name_fallback"
            else:
                # Prepend fallback so it's tried first during DB lookup
                candidates.insert(0, fb_number)
                method = f"{method}+name_fallback"

    result["raw_capture"] = raw_capture
    result["num_candidates"] = len(candidates)
    result["candidates"] = ",".join(candidates[:MAX_CANDIDATES_TO_TRY])
    result["extraction_method"] = method
    result["ocr_name"] = ocr_name or ""

    if not candidates:
        result["status"] = "no_candidates"
        result["verification"] = "red"
        result["reason"] = "No IDOC number found"
        return result

    # Look up candidates against IDOC database
    to_try = candidates[:MAX_CANDIDATES_TO_TRY]
    found_results: list[tuple[str, dict[str, Any]]] = []

    for candidate in to_try:
        url = IDOC_DETAIL_URL.format(idoc_number=candidate)
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                info = _parse_idoc_detail_page(resp.text)
                if info.get("idoc_number"):
                    found_results.append((candidate, info))
        except httpx.HTTPError as e:
            logger.warning(f"HTTP error for {candidate}: {e}")
        await asyncio.sleep(REQUEST_DELAY)

    result["candidates_tried"] = len(to_try)
    result["candidates_found"] = len(found_results)

    if not found_results:
        result["idoc_number"] = candidates[0]
        result["status"] = "not_in_database"
        result["verification"] = "red"
        result["reason"] = f"Tried {len(to_try)} candidates, none in IDOC database"
        return result

    # Pick best match by name
    best_candidate = None
    best_info = None
    best_sim = "none"

    for candidate, info in found_results:
        db_name = info.get("name", "")
        sim = _name_similarity(filename_name, db_name) if filename_name else "none"
        if sim == "exact":
            best_candidate, best_info, best_sim = candidate, info, sim
            break
        if sim == "partial" and best_sim == "none":
            best_candidate, best_info, best_sim = candidate, info, sim
        elif best_candidate is None:
            best_candidate, best_info, best_sim = candidate, info, sim

    result["idoc_number"] = best_candidate
    result["db_name"] = best_info.get("name", "") if best_info else ""
    result["name_match"] = best_sim

    if best_sim == "exact":
        result["status"] = "verified"
        result["verification"] = "green"
        result["reason"] = "Name match confirmed"
    elif best_sim == "partial":
        result["status"] = "partial_match"
        result["verification"] = "yellow"
        result["reason"] = f"Partial name: DB='{result['db_name']}' vs file='{filename_name}'"
    else:
        result["status"] = "name_mismatch"
        result["verification"] = "red"
        result["reason"] = f"Name mismatch: DB='{result['db_name']}' vs file='{filename_name}'"

    return result


async def run_batch(data_dir: Path, limit: int | None, output_path: Path) -> None:
    """Run the batch evaluation on all 4-page PDFs."""
    pdfs = sorted(data_dir.glob("*.pdf"))
    # Also include 2025 data directory
    if DATA_DIR_2025.exists() and data_dir != DATA_DIR_2025:
        pdfs_2025 = sorted(DATA_DIR_2025.glob("*.pdf"))
        logger.info(f"Found {len(pdfs_2025)} PDFs in 2025 directory")
        pdfs = pdfs + pdfs_2025
    logger.info(f"Found {len(pdfs)} total PDFs")

    # Filter to 4-page PDFs only
    eligible: list[Path] = []
    for pdf_path in pdfs:
        try:
            doc = pymupdf.open(pdf_path)
            n_pages = len(doc)
            doc.close()
            if n_pages == 4:
                eligible.append(pdf_path)
        except Exception:
            continue
    logger.info(f"Eligible (4-page): {len(eligible)} PDFs")
    pdfs = eligible

    if limit:
        pdfs = pdfs[:limit]
        logger.info(f"Limiting to {limit} PDFs")

    ocr_backend = RapidOcrBackend()
    directory = IdocDirectory()

    # Load specialized number model (preferred) or fallback to multi-task model
    number_processor = None
    number_model = None
    for model_dir in [NUMBER_MODEL_DIR, TROCR_MODEL_DIR]:
        if model_dir.exists():
            logger.info(f"Loading number TrOCR model from {model_dir}")
            number_processor = TrOCRProcessor.from_pretrained(str(model_dir))
            number_model = VisionEncoderDecoderModel.from_pretrained(str(model_dir))
            number_model.eval()
            if torch.cuda.is_available():
                number_model = number_model.cuda()
            logger.info("Number TrOCR model loaded")
            break
    if number_processor is None:
        logger.warning("No number TrOCR model found, skipping TrOCR number extraction")

    # Load specialized name model
    name_processor = None
    name_model = None
    if NAME_MODEL_DIR.exists():
        logger.info(f"Loading name TrOCR model from {NAME_MODEL_DIR}")
        name_processor = TrOCRProcessor.from_pretrained(str(NAME_MODEL_DIR))
        name_model = VisionEncoderDecoderModel.from_pretrained(str(NAME_MODEL_DIR))
        name_model.eval()
        if torch.cuda.is_available():
            name_model = name_model.cuda()
        logger.info("Name TrOCR model loaded")
    else:
        logger.warning(f"Name TrOCR model not found at {NAME_MODEL_DIR}, skipping")

    results: list[dict[str, Any]] = []

    fieldnames = [
        "filename", "filename_name", "page_count",
        "raw_capture", "num_candidates", "candidates", "extraction_method",
        "ocr_name",
        "idoc_number", "db_name", "name_match",
        "candidates_tried", "candidates_found",
        "status", "verification", "reason",
    ]

    # Open CSV and write header + rows incrementally
    csv_file = open(output_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    csv_file.flush()

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            for i, pdf_path in enumerate(pdfs):
                logger.info(f"[{i+1}/{len(pdfs)}] {pdf_path.name}")
                try:
                    result = await process_pdf(pdf_path, ocr_backend, client, directory, number_processor, number_model, name_processor, name_model)
                except Exception as e:
                    logger.error(f"Error processing {pdf_path.name}: {e}")
                    result = {
                        "filename": pdf_path.name,
                        "filename_name": clean_pdf_stem_name(pdf_path),
                        "status": "error",
                        "verification": "red",
                        "reason": str(e),
                    }
                results.append(result)
                writer.writerow(result)
                csv_file.flush()
    finally:
        csv_file.close()

    logger.info(f"Results written to {output_path}")

    # Print summary
    total = len(results)
    by_status: dict[str, int] = {}
    by_verification: dict[str, int] = {}
    for r in results:
        s = r.get("status", "unknown")
        v = r.get("verification", "unknown")
        by_status[s] = by_status.get(s, 0) + 1
        by_verification[v] = by_verification.get(v, 0) + 1

    print(f"\n{'='*60}")
    print(f"Batch Evaluation Summary ({total} PDFs)")
    print(f"{'='*60}")
    print("\nBy status:")
    for s in sorted(by_status):
        pct = by_status[s] / total * 100
        print(f"  {s:25s}: {by_status[s]:4d} ({pct:5.1f}%)")
    print("\nBy verification tier:")
    for v in ["green", "yellow", "red"]:
        count = by_verification.get(v, 0)
        pct = count / total * 100
        print(f"  {v:25s}: {count:4d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Batch IDOC extraction evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N PDFs")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "output" / "batch_eval_idoc.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    asyncio.run(run_batch(DATA_DIR, args.limit, Path(args.output)))


if __name__ == "__main__":
    main()
