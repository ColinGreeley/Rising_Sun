from __future__ import annotations

import base64
import logging
import os
import re
from pathlib import Path
from typing import Any

# Enable CUDA for RapidOCR if available
os.environ.setdefault("RISING_SUN_RAPIDOCR_USE_CUDA", "1")

import cv2
import httpx
import numpy as np
import pymupdf
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from lxml import html
from PIL import Image
from transformers.models.trocr.processing_trocr import TrOCRProcessor
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import VisionEncoderDecoderModel

from rising_sun.identity import (
    normalize_person_name,
    normalize_supervision_candidates,
    person_name_key,
)
from rising_sun.idoc_lookup import (
    IdocDirectory,
    filter_candidates_by_length,
)
from rising_sun.models import NormalizedBox
from rising_sun.name_ocr import RapidEnsembleNameOcrBackend
from rising_sun.ocr import RapidOcrBackend
from rising_sun.pdf import render_pdf_page, render_pdf_pages
from rising_sun.rso_detector import detect_rso_checkbox

logger = logging.getLogger(__name__)

IDOC_DETAIL_URL = "https://www.idoc.idaho.gov/content/prisons/resident-client-search/details/{idoc_number}"
REQUIRED_PAGE_COUNT = 4  # Only process 4-page IDOC housing applications
MAX_CANDIDATES_TO_TRY = 6

# Region crop box for the IDOC# field on page 1 of the housing application.
# Expressed as (x1_frac, y1_frac, x2_frac, y2_frac) of the page image.
IDOC_CROP_DEFAULT = (0.62, 0.24, 0.95, 0.34)
IDOC_CROP_TIGHT = (0.72, 0.238, 0.90, 0.298)

# Fine-tuned TrOCR v2 model (multi-task: IDOC numbers + applicant names)
# RISING_SUN_MODEL_DIR env var allows PyInstaller / Docker to override the path.
TROCR_MODEL_DIR = Path(os.environ.get(
    "RISING_SUN_MODEL_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "output" / "trocr_model_v3b"),
))

NUMBER_MODEL_DIR = Path(os.environ.get(
    "RISING_SUN_NUMBER_MODEL_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "output" / "trocr_number_model_v2"),
))

NAME_MODEL_DIR = Path(os.environ.get(
    "RISING_SUN_NAME_MODEL_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "output" / "trocr_name_model_v2"),
))

# Name field crop boxes (from config/idoc_application_template.yml)
NAME_CROP_TIGHT = (0.14, 0.248, 0.60, 0.282)
NAME_CROP_CONTEXT = (0.10, 0.238, 0.64, 0.290)
NAME_CROP_WIDE = (0.08, 0.235, 0.68, 0.295)

app = FastAPI(title="Rising Sun IDOC Lookup", version="0.7.0")

# CORS: allow local dev and any production origin (Caddy handles HTTPS termination)
_cors_origins = os.environ.get("CORS_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr = RapidOcrBackend()
name_ocr = RapidEnsembleNameOcrBackend(ocr, normalize_person_name)

# Load IDOC directory (spreadsheet of known numbers ↔ names).
# Gracefully degrades when the spreadsheet is unavailable (e.g. production).
try:
    idoc_directory = IdocDirectory()
except Exception:
    logger.warning("Could not load IDOC directory — running without spreadsheet lookup")
    idoc_directory = IdocDirectory.__new__(IdocDirectory)
    idoc_directory._by_number = {}
    idoc_directory._by_name_key = {}
    idoc_directory._all_numbers = set()

# Load TrOCR model at startup
trocr_processor: TrOCRProcessor | None = None
trocr_model: VisionEncoderDecoderModel | None = None
number_trocr_processor: TrOCRProcessor | None = None
number_trocr_model: VisionEncoderDecoderModel | None = None
name_trocr_processor: TrOCRProcessor | None = None
name_trocr_model: VisionEncoderDecoderModel | None = None


def _load_trocr_bundle(model_dir: Path, label: str) -> tuple[TrOCRProcessor | None, VisionEncoderDecoderModel | None]:
    if not model_dir.exists():
        logger.warning("%s model not found at %s", label, model_dir)
        return None, None
    logger.info("Loading %s model from %s", label, model_dir)
    processor = TrOCRProcessor.from_pretrained(str(model_dir))
    model = VisionEncoderDecoderModel.from_pretrained(str(model_dir))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    logger.info("%s model loaded (CUDA=%s)", label, torch.cuda.is_available())
    return processor, model


trocr_processor, trocr_model = _load_trocr_bundle(TROCR_MODEL_DIR, "TrOCR")

if NUMBER_MODEL_DIR.exists() and NUMBER_MODEL_DIR != TROCR_MODEL_DIR:
    number_trocr_processor, number_trocr_model = _load_trocr_bundle(NUMBER_MODEL_DIR, "TrOCR number")
else:
    number_trocr_processor, number_trocr_model = trocr_processor, trocr_model

if NAME_MODEL_DIR.exists() and NAME_MODEL_DIR != TROCR_MODEL_DIR:
    name_trocr_processor, name_trocr_model = _load_trocr_bundle(NAME_MODEL_DIR, "TrOCR name")
else:
    name_trocr_processor, name_trocr_model = trocr_processor, trocr_model

# RSO detection now uses template matching (rso_detector module), no model needed.

# ---------------------------------------------------------------------------
# IDOC number extraction from PDF text
# ---------------------------------------------------------------------------
_IDOC_CAPTURE_CC = r"[A-Za-z0-9.,\-/\\()\[\]{}!|&$%'\" ]"
_IDOC_PATTERNS = [
    re.compile(rf"ID[O0]C\s*#\s*:?\s*({_IDOC_CAPTURE_CC}{{4,15}})", re.IGNORECASE),
    re.compile(rf"IDOC\s*(?:or\s*LE\s*)?#\s*:?\s*({_IDOC_CAPTURE_CC}{{4,15}})", re.IGNORECASE),
    re.compile(rf"IDOC\s*Number\s*:?\s*({_IDOC_CAPTURE_CC}{{4,15}})", re.IGNORECASE),
]
_STANDALONE_NUMBER_RE = re.compile(r"\b(\d{5,6})\b")

# Common OCR letter-to-digit substitutions (matches rising_sun.identity)
_OCR_DIGIT_MAP = str.maketrans("OoIlLJZSBG/\\|!", "00111125861111")


def _ocr_to_digits(value: str) -> str | None:
    """Try to convert an OCR-read value to a pure-digit IDOC number."""
    # Strip everything except alphanumeric and OCR-ambiguous characters
    cleaned = re.sub(r"[^A-Za-z0-9/\\|!]", "", value)
    digits = cleaned.translate(_OCR_DIGIT_MAP)
    if digits.isdigit() and 4 <= len(digits) <= 8:
        return digits
    return None


def _extract_idoc_raw_from_text(text: str) -> str | None:
    """Return the raw captured string (before digit normalization)."""
    for pattern in _IDOC_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).strip()
    return None


def _extract_idoc_candidates_from_text(text: str) -> tuple[str | None, list[str]]:
    """Return (raw_capture, list_of_digit_candidates) from OCR text."""
    raw = _extract_idoc_raw_from_text(text)
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


def _image_to_data_uri(img: np.ndarray) -> str | None:
    """Encode a BGR numpy image as a base64 PNG data URI."""
    success, buffer = cv2.imencode(".png", img)
    if not success:
        return None
    return "data:image/png;base64," + base64.b64encode(buffer).decode()


def _extract_from_region_crop(page_image: np.ndarray) -> tuple[str | None, list[str]]:
    """OCR just the IDOC# region of the page image and extract candidates."""
    # Try the default (wider) crop first
    crop = _crop_idoc_region(page_image, IDOC_CROP_DEFAULT)
    ocr_text = ocr.read_text(crop, multiline=True).text
    raw, candidates = _extract_idoc_candidates_from_text(ocr_text)
    if candidates:
        return raw, candidates

    # Try tight crop
    crop_tight = _crop_idoc_region(page_image, IDOC_CROP_TIGHT)
    ocr_text2 = ocr.read_text(crop_tight, multiline=True).text
    raw2, candidates2 = _extract_idoc_candidates_from_text(ocr_text2)
    if candidates2:
        return raw2, candidates2

    return raw or raw2, []


def _trocr_read_crop(
    page_image: np.ndarray,
    box: tuple[float, float, float, float],
    max_tokens: int = 12,
    processor: TrOCRProcessor | None = None,
    model: VisionEncoderDecoderModel | None = None,
    num_beams: int = 4,
) -> str:
    """Run TrOCR on a single crop and return raw prediction text."""
    processor = processor or trocr_processor
    model = model or trocr_model
    if processor is None or model is None:
        return ""
    crop = _crop_idoc_region(page_image, box)
    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
    if next(model.parameters()).is_cuda:
        pixel_values = pixel_values.cuda()
    with torch.no_grad():
        ids = model.generate(pixel_values, max_new_tokens=max_tokens, num_beams=num_beams)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _trocr_read_crop_multi(
    page_image: np.ndarray,
    box: tuple[float, float, float, float],
    max_tokens: int = 12,
    processor: TrOCRProcessor | None = None,
    model: VisionEncoderDecoderModel | None = None,
    num_beams: int = 8,
    num_return_sequences: int = 4,
) -> list[str]:
    """Run TrOCR on a crop and return multiple beam hypotheses."""
    processor = processor or trocr_processor
    model = model or trocr_model
    if processor is None or model is None:
        return []
    crop = _crop_idoc_region(page_image, box)
    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
    if next(model.parameters()).is_cuda:
        pixel_values = pixel_values.cuda()
    with torch.no_grad():
        ids = model.generate(
            pixel_values,
            max_new_tokens=max_tokens,
            num_beams=num_beams,
            num_return_sequences=min(num_return_sequences, num_beams),
        )
    return [t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)]


def _trocr_extract_candidates(page_image: np.ndarray) -> list[str]:
    """Run fine-tuned TrOCR on tight+default crops and return digit candidates.

    Uses multi-beam generation and snaps results against the known IDOC
    directory to recover from minor OCR digit errors.
    """
    if number_trocr_processor is None or number_trocr_model is None:
        return []
    raw_digits: list[str] = []
    for box in [IDOC_CROP_TIGHT, IDOC_CROP_DEFAULT]:
        preds = _trocr_read_crop_multi(
            page_image,
            box,
            max_tokens=12,
            processor=number_trocr_processor,
            model=number_trocr_model,
            num_beams=8,
            num_return_sequences=4,
        )
        for pred in preds:
            digits = re.sub(r"[^0-9]", "", pred)
            if digits and 5 <= len(digits) <= 6 and digits not in raw_digits:
                raw_digits.append(digits)

    # Snap raw OCR digits against the known directory via fuzzy matching.
    # Prioritise direct hits, then fuzzy matches, then raw digits.
    snapped: list[str] = []
    remaining: list[str] = []
    for d in raw_digits:
        if idoc_directory.is_known(d):
            if d not in snapped:
                snapped.append(d)
        else:
            fuzzy = idoc_directory.fuzzy_match(d)
            if fuzzy:
                for f in fuzzy:
                    if f not in snapped:
                        snapped.append(f)
            else:
                remaining.append(d)
    # Append un-snapped raw digits at the end as fallback
    for d in remaining:
        if d not in snapped:
            snapped.append(d)
    return snapped


def _trocr_extract_name(page_image: np.ndarray) -> str | None:
    """OCR the applicant name field using TrOCR + RapidOCR ensemble fallback.

    Generates multiple candidates from different crop regions and beam
    hypotheses, then picks the best one that looks like a real name
    (2+ alpha tokens).  Prefers candidates that match a known name in
    the IDOC directory.  Falls back to the RapidOCR ensemble pipeline
    when TrOCR alone can't find a directory-confirmed match.
    """
    all_name_candidates: list[str] = []

    # Phase 1: TrOCR multi-beam candidates
    if name_trocr_processor is not None and name_trocr_model is not None:
        for box in [NAME_CROP_TIGHT, NAME_CROP_CONTEXT, NAME_CROP_WIDE]:
            preds = _trocr_read_crop_multi(
                page_image,
                box,
                max_tokens=40,
                processor=name_trocr_processor,
                model=name_trocr_model,
                num_beams=4,
                num_return_sequences=3,
            )
            for pred in preds:
                pred = pred.strip()
                if pred and len(pred) >= 3 and " " in pred and pred not in all_name_candidates:
                    all_name_candidates.append(pred)

    # Phase 2: RapidOCR ensemble candidates (always runs as fallback/supplement)
    rapid_candidates = name_ocr.extract_candidates(
        page_image, page_text="", box=NAME_CROP_TIGHT,
    )
    for rc in rapid_candidates:
        if rc.value and rc.value not in all_name_candidates:
            all_name_candidates.append(rc.value)

    if not all_name_candidates:
        return None

    # Prefer a candidate that matches a known name in the directory
    for cand in all_name_candidates:
        numbers = idoc_directory.lookup_by_name(cand)
        if numbers:
            return cand

    # Fall back to first plausible name
    return all_name_candidates[0]


def _predict_rso(content: bytes) -> dict[str, Any] | None:
    """Run template-matching RSO checkbox detection across all pages."""
    rso_pages = render_pdf_pages(content, dpi=225)
    rso_result = detect_rso_checkbox(rso_pages)
    prediction = rso_result["prediction"]

    # Crop the RSO checkbox area from the matched page for visualization.
    # Use the dynamic crop_box computed from the actual template match location.
    page_idx = rso_result.get("page", -1)
    rso_page = rso_pages[page_idx] if 0 <= page_idx < len(rso_pages) else (rso_pages[0] if rso_pages else None)
    rso_crop_image: str | None = None
    if rso_page is not None and rso_result.get("crop_box"):
        rso_crop_image = _image_to_data_uri(_crop_idoc_region(rso_page, rso_result["crop_box"]))
    elif rso_page is not None:
        # Fallback fixed crop for text-fallback detections (no template match)
        rso_crop_image = _image_to_data_uri(_crop_idoc_region(rso_page, (0.50, 0.530, 0.75, 0.595)))

    return {
        "prediction": prediction,
        "decision": prediction,
        "is_rso": prediction == "yes",
        "needs_review": rso_result["method"] == "default",
        "probability_yes": rso_result["scores"]["yes_score"],
        "probability_no": rso_result["scores"]["no_score"],
        "confidence": rso_result["confidence"],
        "score_yes": rso_result["scores"]["yes_score"],
        "score_no": rso_result["scores"]["no_score"],
        "method": rso_result["method"],
        "page": rso_result["page"],
        "crop_image": rso_crop_image,
    }


def _extract_from_pdf(content: bytes) -> dict[str, Any]:
    """Extract page count, raw IDOC text, and candidate numbers from PDF bytes.

    Pure OCR pipeline — no spreadsheet or directory matching.
    1. Try embedded text (digital PDFs)
    2. Region crop OCR at 225 DPI (targets the IDOC# field directly)
    2b. Full-page OCR at 225 DPI with regex extraction
    2c. CLAHE on 225 DPI crop
    3. Region crop OCR at 300 DPI + CLAHE
    4a. Region crop at 400 DPI
    4b. 400 DPI + binary threshold (Otsu)
    5. TrOCR fine-tuned model on tight/default crops
    6. Filter candidates to 5+ digits
    """
    doc = pymupdf.open(stream=content, filetype="pdf")
    page_count = len(doc)

    # Try embedded text first (digital PDFs)
    raw_capture = None
    candidates: list[str] = []
    extraction_method = "none"
    ocr_name: str | None = None
    page_image: np.ndarray | None = None
    for page in doc:
        text = page.get_text()
        if len(text.strip()) > 50:
            raw_capture, candidates = _extract_idoc_candidates_from_text(text)
            if candidates:
                extraction_method = "embedded_text"
                break
    doc.close()

    # Fall back to OCR strategies for scanned PDFs
    if not candidates:
        # Render from the uploaded bytes directly so Windows file locking does
        # not block PyMuPDF from reopening a still-open temporary file.
        page_image = render_pdf_page(content, dpi=225, page_number=0)
        raw_capture, candidates = _extract_from_region_crop(page_image)
        if candidates:
            extraction_method = "region_crop_225dpi"

        # Strategy 2: Full-page OCR at 225 DPI
        if not candidates:
            ocr_text = ocr.read_text(page_image, multiline=True).text
            raw_capture, candidates = _extract_idoc_candidates_from_text(ocr_text)
            if candidates:
                extraction_method = "base_225dpi"

        # Strategy 2b: CLAHE on 225 DPI crop
        if not candidates:
            crop_225 = _crop_idoc_region(page_image, IDOC_CROP_DEFAULT)
            gray_225 = cv2.cvtColor(crop_225, cv2.COLOR_BGR2GRAY)
            clahe_225 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_225 = clahe_225.apply(gray_225)
            clahe_img_225 = cv2.cvtColor(enhanced_225, cv2.COLOR_GRAY2BGR)
            ocr_text = ocr.read_text(clahe_img_225, multiline=True).text
            raw_cl, cands_cl = _extract_idoc_candidates_from_text(ocr_text)
            if cands_cl:
                raw_capture, candidates = raw_cl, cands_cl
                extraction_method = "clahe_crop_225dpi"

        # Strategy 3: Region crop at 300 DPI with CLAHE
        if not candidates:
            page_300 = render_pdf_page(content, dpi=300, page_number=0)
            raw_300, cands_300 = _extract_from_region_crop(page_300)
            if cands_300:
                raw_capture, candidates = raw_300, cands_300
                extraction_method = "region_crop_300dpi"
            else:
                crop_300 = _crop_idoc_region(page_300, IDOC_CROP_DEFAULT)
                gray = cv2.cvtColor(crop_300, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                clahe_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                ocr_text = ocr.read_text(clahe_img, multiline=True).text
                raw_c, cands_c = _extract_idoc_candidates_from_text(ocr_text)
                if cands_c:
                    raw_capture, candidates = raw_c, cands_c
                    extraction_method = "clahe_crop_300dpi"

        # Strategy 4a: Region crop at 400 DPI
        if not candidates:
            page_400 = render_pdf_page(content, dpi=400, page_number=0)
            raw_400, cands_400 = _extract_from_region_crop(page_400)
            if cands_400:
                raw_capture, candidates = raw_400, cands_400
                extraction_method = "region_crop_400dpi"

        # Strategy 4b: 400 DPI + binary threshold
        if not candidates:
            crop_400 = _crop_idoc_region(page_400, IDOC_CROP_DEFAULT)
            gray_400 = cv2.cvtColor(crop_400, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray_400, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            ocr_text = ocr.read_text(binary_img, multiline=True).text
            raw_b, cands_b = _extract_idoc_candidates_from_text(ocr_text)
            if cands_b:
                raw_capture, candidates = raw_b, cands_b
                extraction_method = "binary_crop_400dpi"

        # Strategy 5: TrOCR fine-tuned model
        trocr_cands = _trocr_extract_candidates(page_image)
        for tc in trocr_cands:
            if tc not in candidates:
                candidates.append(tc)
        if trocr_cands:
            extraction_method = f"{extraction_method}+trocr" if extraction_method != "none" else "trocr"

        # Extract applicant name via TrOCR for cross-check
        ocr_name = _trocr_extract_name(page_image)

    # Ensure page_image is available for crop visualization (even for digital PDFs)
    if page_image is None:
        page_image = render_pdf_page(content, dpi=225, page_number=0)

    # RSO checkbox prediction via template matching (works on any page count)
    rso_result = _predict_rso(content)

    # Filter: require 5+ digit candidates
    candidates = filter_candidates_by_length(candidates)

    # Crop visualization images
    idoc_crop_image = _image_to_data_uri(_crop_idoc_region(page_image, IDOC_CROP_DEFAULT))
    name_crop_image = _image_to_data_uri(_crop_idoc_region(page_image, NAME_CROP_WIDE))

    return {
        "page_count": page_count,
        "raw_capture": raw_capture,
        "candidates": candidates,
        "extraction_method": extraction_method,
        "ocr_name": ocr_name,
        "rso": rso_result,
        "idoc_crop_image": idoc_crop_image,
        "name_crop_image": name_crop_image,
    }


# ---------------------------------------------------------------------------
# IDOC website scraper
# ---------------------------------------------------------------------------

def _parse_idoc_detail_page(page_html: str) -> dict[str, Any]:
    """Parse the IDOC resident/client search detail page HTML."""
    tree = html.fromstring(page_html)

    # The resident info lives inside div.region-content
    region = tree.xpath('//div[contains(@class, "region-content")]')
    if not region:
        return {}
    content_text = region[0].text_content()

    info: dict[str, Any] = {}

    # Name: appears between "Resident/Client Search" heading and "IDOC #:"
    name_match = re.search(
        r"Resident/Client Search\s+(.+?)\s*IDOC\s*#",
        content_text,
        re.DOTALL,
    )
    if name_match:
        info["name"] = re.sub(r"\s+", " ", name_match.group(1)).strip()

    # IDOC #
    idoc_match = re.search(r"IDOC\s*#:\s*(\d+)", content_text)
    if idoc_match:
        info["idoc_number"] = idoc_match.group(1)

    # Status: appears after "Status:" with possible whitespace/newlines
    status_match = re.search(r"Status:\s*(.+?)(?:Age:|$)", content_text, re.DOTALL)
    if status_match:
        info["status"] = re.sub(r"\s+", " ", status_match.group(1)).strip()

    # Age
    age_match = re.search(r"Age:\s*(\d+)", content_text)
    if age_match:
        info["age"] = int(age_match.group(1))

    # Mailing Address — between "Mailing Address" and "Phone Number"
    addr_match = re.search(
        r"Mailing Address\s*(.+?)\s*Phone Number",
        content_text,
        re.DOTALL,
    )
    if addr_match:
        info["mailing_address"] = re.sub(r"\s+", " ", addr_match.group(1)).strip()

    # Phone Number
    phone_match = re.search(r"Phone Number\s*(\d{7,})", content_text, re.DOTALL)
    if phone_match:
        raw = phone_match.group(1)
        if len(raw) == 10:
            info["phone"] = f"({raw[:3]}) {raw[3:6]}-{raw[6:]}"
        else:
            info["phone"] = raw

    # Sentence table rows — use the actual <table> in the page
    # Each data cell includes its header label as prefix text, so we strip it.
    _CELL_PREFIXES = [
        "Offense", "Sentencing County", "Case No.",
        "Status", "Released to Supervision*",
        "Sentence Satisfaction Date",
    ]
    sentences: list[dict[str, str]] = []
    for tbl in tree.xpath('//table'):
        rows = tbl.xpath('.//tr')
        for row in rows:
            cells_raw = [re.sub(r"\s+", " ", c.text_content()).strip() for c in row.xpath('.//td')]
            cells_raw = [c for c in cells_raw if c]
            if len(cells_raw) < 4:
                continue
            # Strip known header prefixes from cell text
            cells: list[str] = []
            for cell_text, prefix in zip(cells_raw, _CELL_PREFIXES):
                if cell_text.startswith(prefix):
                    cell_text = cell_text[len(prefix):].strip()
                cells.append(cell_text)
            # Append remaining cells (if any) without prefix stripping
            cells.extend(cells_raw[len(_CELL_PREFIXES):])
            if not cells[0]:
                continue
            entry: dict[str, str] = {
                "offense": cells[0],
                "county": cells[1],
                "case_number": cells[2],
                "sentence_status": cells[3],
            }
            if len(cells) >= 5:
                entry["released_to_supervision"] = cells[4]
            if len(cells) >= 6:
                entry["termination_date"] = cells[5]
            sentences.append(entry)
    if sentences:
        info["sentences"] = sentences

    return info


async def _lookup_idoc(idoc_number: str) -> dict[str, Any]:
    """Fetch and parse the IDOC detail page for the given number."""
    url = IDOC_DETAIL_URL.format(idoc_number=idoc_number)
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(url)
    if resp.status_code != 200:
        return {"error": f"IDOC lookup returned status {resp.status_code}"}
    info = _parse_idoc_detail_page(resp.text)
    if not info.get("idoc_number"):
        return {"error": "No resident found for this IDOC number"}
    info["idoc_url"] = url
    return info


# ---------------------------------------------------------------------------
# Multi-candidate verification pipeline
# ---------------------------------------------------------------------------

def _name_similarity(name_a: str, name_b: str) -> str:
    """Compare two names using nickname-aware canonicalization.

    Order-agnostic: handles TrOCR reading "Last First" from forms while the
    IDOC website shows "First Last".

    Returns 'exact' (full match), 'partial' (at least one token match), or 'none'.
    """
    key_a = person_name_key(name_a)
    key_b = person_name_key(name_b)
    if key_a and key_b:
        if key_a == key_b:
            return "exact"
        # Handle reversed name order (e.g. "Evans Aaron" vs "Aaron Evans")
        if key_a == (key_b[1], key_b[0]):
            return "exact"
    norm_a = set(normalize_person_name(name_a).split())
    norm_b = set(normalize_person_name(name_b).split())
    if not norm_a or not norm_b:
        return "none"
    if norm_a & norm_b:
        return "partial"
    return "none"


async def _verify_and_lookup(
    candidates: list[str],
    raw_capture: str | None,
    extraction_method: str = "",
    ocr_name: str | None = None,
) -> dict[str, Any]:
    """Try candidate IDOC numbers against the IDOC website.

    Returns the best match, preferring OCR-name-confirmed results.
    """
    if not candidates:
        return {
            "idoc_number": None,
            "idoc_info": {"error": "Could not extract any IDOC number candidates from the PDF."},
            "verification": {
                "status": "red",
                "reason": "No IDOC number found in document",
                "raw_capture": raw_capture,
                "extraction_method": extraction_method,
                "candidates_tried": [],
            },
        }

    # Limit candidates to avoid excessive requests
    to_try = candidates[:MAX_CANDIDATES_TO_TRY]

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        results: list[tuple[str, dict[str, Any]]] = []
        for candidate in to_try:
            url = IDOC_DETAIL_URL.format(idoc_number=candidate)
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    info = _parse_idoc_detail_page(resp.text)
                    if info.get("idoc_number"):
                        info["idoc_url"] = url
                        results.append((candidate, info))
            except httpx.HTTPError:
                continue

    tried = [{"number": c, "found": any(c == r[0] for r in results)} for c in to_try]

    if not results:
        return {
            "idoc_number": candidates[0],
            "idoc_info": {"error": "No resident found for any IDOC number candidate"},
            "verification": {
                "status": "red",
                "reason": f"Tried {len(to_try)} candidate(s), none found in IDOC database",
                "raw_capture": raw_capture,
                "extraction_method": extraction_method,
                "candidates_tried": tried,
            },
        }

    # Pick the best match, preferring OCR-name-confirmed results.
    match_name = ocr_name
    best_candidate = None
    best_info = None
    best_sim = "none"

    for candidate, info in results:
        db_name = info.get("name", "")
        sim = _name_similarity(match_name, db_name) if match_name else "none"
        if sim == "exact":
            best_candidate, best_info, best_sim = candidate, info, sim
            break
        if sim == "partial" and best_sim == "none":
            best_candidate, best_info, best_sim = candidate, info, sim
        elif best_candidate is None:
            best_candidate, best_info, best_sim = candidate, info, sim

    chosen_number, chosen_info = best_candidate, best_info

    # Name cross-check details
    name_crosscheck: dict[str, Any] = {"ocr_name": ocr_name}
    idoc_name = chosen_info.get("name")
    if match_name and idoc_name:
        name_crosscheck["idoc_name"] = idoc_name
        name_crosscheck["match"] = best_sim in ("exact", "partial")
        name_crosscheck["match_level"] = best_sim

    verification_status = "green"
    reason = f"IDOC #{chosen_number} verified — {idoc_name or 'unknown'}"
    if match_name and idoc_name and best_sim == "none":
        verification_status = "yellow"
        reason = f"IDOC #{chosen_number} found but name mismatch: form says \"{match_name}\", IDOC says \"{idoc_name}\""

    return {
        "idoc_number": chosen_number,
        "idoc_info": chosen_info,
        "verification": {
            "status": verification_status,
            "reason": reason,
            "raw_capture": raw_capture,
            "extraction_method": extraction_method,
            "candidates_tried": tried,
            "total_found": len(results),
            "name_crosscheck": name_crosscheck,
        },
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.post("/api/extract")
async def extract_pdf(file: UploadFile) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50 MB)")

    extraction = _extract_from_pdf(content)

    # Reject non-4-page PDFs
    if REQUIRED_PAGE_COUNT is not None and extraction["page_count"] != REQUIRED_PAGE_COUNT:
        raise HTTPException(
            status_code=422,
            detail=f"Only {REQUIRED_PAGE_COUNT}-page IDOC housing applications are supported. "
                   f"This PDF has {extraction['page_count']} pages.",
        )

    candidates = extraction["candidates"]
    extraction_method = extraction["extraction_method"]
    ocr_name = extraction.get("ocr_name")
    match_name = ocr_name or ""

    # Phase 2: Fuzzy match OCR candidates against known IDOC directory
    directory_match: str | None = None
    if candidates and idoc_directory.known_numbers:
        best_num, match_method = idoc_directory.best_match(candidates, match_name)
        if best_num:
            directory_match = best_num
            if best_num not in candidates:
                candidates.insert(0, best_num)
            extraction_method = f"{extraction_method}+{match_method}"

    # Phase 3: Name-based fallback when OCR found nothing or directory didn't confirm
    if match_name and not directory_match and idoc_directory.known_numbers:
        fb_number, _fb_name = idoc_directory.name_fallback(match_name)
        if fb_number:
            if not candidates:
                candidates = [fb_number]
                extraction_method = "spreadsheet_name_fallback"
            else:
                candidates.insert(0, fb_number)
                extraction_method = f"{extraction_method}+name_fallback"

    result = await _verify_and_lookup(
        candidates=candidates,
        raw_capture=extraction["raw_capture"],
        extraction_method=extraction_method,
        ocr_name=ocr_name,
    )

    response = {
        "filename": file.filename,
        "page_count": extraction["page_count"],
        "extraction_method": extraction_method,
        **result,
    }

    info = response.get("idoc_info") or {}
    if info.get("name"):
        response["resolved_name"] = info["name"]
        response["resolved_name_source"] = "idoc_website"
    elif ocr_name:
        response["resolved_name"] = ocr_name
        response["resolved_name_source"] = "ocr"

    # Bundle crop images for frontend visualization
    rso_data = extraction.get("rso") or {}
    rso_crop = rso_data.pop("crop_image", None)
    crop_images: dict[str, str | None] = {
        "idoc_field": extraction.get("idoc_crop_image"),
        "name_field": extraction.get("name_crop_image"),
        "rso_field": rso_crop,
    }
    if any(v for v in crop_images.values()):
        response["crop_images"] = crop_images

    # Include RSO prediction if available
    if extraction.get("rso"):
        response["rso"] = extraction["rso"]

    return response


# ---------------------------------------------------------------------------
# Serve frontend static build (production — when dist/ exists)
# ---------------------------------------------------------------------------
_FRONTEND_DIR = Path(os.environ.get(
    "RISING_SUN_FRONTEND_DIR",
    str(Path(__file__).resolve().parent.parent / "frontend" / "dist"),
))

if _FRONTEND_DIR.is_dir():
    # Catch-all: serve index.html for client-side routing
    @app.get("/{path:path}")
    async def _spa_fallback(path: str):
        file_path = _FRONTEND_DIR / path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
