from __future__ import annotations

import re
import unicodedata

import cv2
import numpy as np

from rising_sun.models import NormalizedBox


def pixel_box(image: np.ndarray, box: NormalizedBox) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    left = max(0, min(width - 1, int(box[0] * width)))
    top = max(0, min(height - 1, int(box[1] * height)))
    right = max(left + 1, min(width, int(box[2] * width)))
    bottom = max(top + 1, min(height, int(box[3] * height)))
    return left, top, right, bottom


def crop_image(image: np.ndarray, box: NormalizedBox, padding: int = 4) -> np.ndarray:
    left, top, right, bottom = pixel_box(image, box)
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.shape[1], right + padding)
    bottom = min(image.shape[0], bottom + padding)
    return image[top:bottom, left:right].copy()


def prepare_text_crop(image: np.ndarray, multiline: bool) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal_span = max(15, image.shape[1] // (12 if multiline else 10))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_span, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    cleaned = cv2.subtract(binary, horizontal_lines)

    if multiline:
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, image.shape[0] // 12)))
        vertical_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, vertical_kernel)
        cleaned = cv2.subtract(cleaned, vertical_lines)

    restored = cv2.bitwise_not(cleaned)
    return cv2.resize(restored, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


def checkbox_score(image: np.ndarray, box: NormalizedBox) -> float:
    crop = crop_image(image, box, padding=3)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = binary.shape[:2]
    margin_y = max(1, int(height * 0.18))
    margin_x = max(1, int(width * 0.18))
    inner = binary[margin_y : height - margin_y, margin_x : width - margin_x]
    if inner.size == 0:
        return 0.0
    return float(inner.mean() / 255.0)


def mostly_blank(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray < 230)) < 0.01


def normalize_text(text: str, multiline: bool) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u2019", "'")
    if multiline:
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        normalized = re.sub(r" *\n *", "\n", normalized)
        return normalized.strip(" _,-\n")

    normalized = normalized.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" _,-")
