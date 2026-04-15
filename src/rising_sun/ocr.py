from __future__ import annotations

from dataclasses import dataclass
import os

import cv2
import numpy as np
import pytesseract
from rapidocr_onnxruntime import RapidOCR
from pytesseract import TesseractError

from rising_sun.image_ops import normalize_text


@dataclass(frozen=True)
class OCRTextResult:
    text: str
    confidence: float
    lines: list[str]


def _cuda_available() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _make_rapid_ocr() -> RapidOCR:
    force_cuda = os.getenv("RISING_SUN_RAPIDOCR_USE_CUDA", "").strip().lower() in {"1", "true", "yes"}
    if force_cuda and _cuda_available():
        return RapidOCR(det_use_cuda=True, rec_use_cuda=True, cls_use_cuda=True)
    return RapidOCR(det_use_cuda=False, rec_use_cuda=False, cls_use_cuda=False)


class RapidOcrBackend:
    def __init__(self) -> None:
        self._engine = _make_rapid_ocr()

    def read_text(self, image: np.ndarray, multiline: bool = False) -> OCRTextResult:
        result, _ = self._engine(image)
        if not result:
            return OCRTextResult(text="", confidence=0.0, lines=[])

        ordered = sorted(result, key=lambda item: (min(point[1] for point in item[0]), min(point[0] for point in item[0])))
        lines = [normalize_text(item[1], multiline=False) for item in ordered if item[1].strip()]
        text = normalize_text("\n".join(lines) if multiline else " ".join(lines), multiline=multiline)
        confidence = float(sum(float(item[2]) for item in ordered) / len(ordered))
        return OCRTextResult(text=text, confidence=confidence, lines=lines)


class TesseractDigitBackend:
    """Tesseract backend specialized for reading digit-only crops."""

    DIGIT_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789"
    SINGLE_LINE_CONFIG = "--psm 7"

    def __init__(self, timeout_seconds: float = 5.0) -> None:
        self.timeout_seconds = timeout_seconds

    def read_digits(self, image: np.ndarray) -> list[str]:
        variants = self._prepare_variants(image)
        candidates: list[str] = []
        for variant in variants:
            text = self._read_variant(variant, self.DIGIT_CONFIG)
            if text:
                candidates.append(text)
        return candidates

    def read_text_single_line(self, image: np.ndarray) -> list[str]:
        variants = self._prepare_variants(image)
        candidates: list[str] = []
        for variant in variants:
            text = self._read_variant(variant, self.SINGLE_LINE_CONFIG)
            if text:
                candidates.append(text)
        return candidates

    def _read_variant(self, image: np.ndarray, config: str) -> str:
        try:
            return pytesseract.image_to_string(
                image,
                config=config,
                timeout=self.timeout_seconds,
            ).strip()
        except (RuntimeError, TesseractError, OSError):
            return ""

    def _prepare_variants(self, image: np.ndarray) -> list[np.ndarray]:
        if image.size == 0:
            return []
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        upscaled = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, inv = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return [upscaled, binary, inv]
