"""RSO (Registered Sex Offender) checkbox detection via template matching.

Uses dual "sex offender" text templates (V1 and V2 form versions) to locate the
RSO question on any page, then reads the Yes/No checkbox fill levels using
calibrated offsets from the matched template position.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Template images (grayscale, loaded lazily)
_tmpl_v1_gray: np.ndarray | None = None
_tmpl_v2_gray: np.ndarray | None = None

# Calibrated offsets from template match location to checkbox centres (normalised)
_V1_YES_OFF = (0.4862, 0.0020)
_V1_NO_OFF = (0.5447, 0.0034)
_V2_YES_OFF = (0.4725, 0.0030)
_V2_NO_OFF = (0.5282, 0.0024)

_BOX_W = 0.016
_BOX_H = 0.014
_TMPL_CONF_THRESHOLD = 0.55
_MARGIN_FRAC = 0.18


def _load_templates() -> tuple[np.ndarray, np.ndarray]:
    global _tmpl_v1_gray, _tmpl_v2_gray
    if _tmpl_v1_gray is None:
        img = cv2.imread(str(_ASSETS_DIR / "tmpl_sex_offender_v1.png"))
        _tmpl_v1_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if _tmpl_v2_gray is None:
        img = cv2.imread(str(_ASSETS_DIR / "tmpl_sex_offender_v2.png"))
        _tmpl_v2_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return _tmpl_v1_gray, _tmpl_v2_gray


def _checkbox_score(gray: np.ndarray, box: tuple[float, float, float, float]) -> float:
    h, w = gray.shape
    x1 = max(0, int(box[0] * w))
    y1 = max(0, int(box[1] * h))
    x2 = min(w, int(box[2] * w))
    y2 = min(h, int(box[3] * h))
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * _MARGIN_FRAC), int(bh * _MARGIN_FRAC)
    crop = gray[y1 + my : y2 - my, x1 + mx : x2 - mx]
    if crop.size == 0:
        return 0.0
    return 1.0 - crop.mean() / 255.0


def detect_rso_checkbox(page_images: list[np.ndarray]) -> dict[str, Any]:
    """Detect RSO checkbox status across multiple page images.

    Parameters
    ----------
    page_images : list[np.ndarray]
        BGR page images (rendered at 225 DPI).

    Returns
    -------
    dict with keys:
        - prediction: "yes" | "no"
        - confidence: float (template match confidence)
        - method: str describing detection method
        - scores: dict with yes_score and no_score
        - page: int (0-based page index where match was found)
        - version: "V1" | "V2" (matched template version)
    """
    tmpl_v1, tmpl_v2 = _load_templates()

    best_conf = 0.0
    best_info: tuple[int, str, float, float] | None = None

    for pg_idx, img in enumerate(page_images[:4]):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        res1 = cv2.matchTemplate(gray, tmpl_v1, cv2.TM_CCOEFF_NORMED)
        _, c1, _, l1 = cv2.minMaxLoc(res1)
        res2 = cv2.matchTemplate(gray, tmpl_v2, cv2.TM_CCOEFF_NORMED)
        _, c2, _, l2 = cv2.minMaxLoc(res2)

        if c1 >= c2:
            conf, loc, ver = c1, l1, "V1"
            yes_off, no_off = _V1_YES_OFF, _V1_NO_OFF
        else:
            conf, loc, ver = c2, l2, "V2"
            yes_off, no_off = _V2_YES_OFF, _V2_NO_OFF

        if conf > best_conf:
            sx, sy = loc[0] / w, loc[1] / h
            yes_box = (sx + yes_off[0], sy + yes_off[1],
                       sx + yes_off[0] + _BOX_W, sy + yes_off[1] + _BOX_H)
            no_box = (sx + no_off[0], sy + no_off[1],
                      sx + no_off[0] + _BOX_W, sy + no_off[1] + _BOX_H)
            ys = _checkbox_score(gray, yes_box)
            ns = _checkbox_score(gray, no_box)
            best_conf = conf
            best_info = (pg_idx, ver, ys, ns)

    if best_conf >= _TMPL_CONF_THRESHOLD and best_info is not None:
        pg_idx, ver, ys, ns = best_info
        prediction = "yes" if ys > ns else "no"
        return {
            "prediction": prediction,
            "confidence": float(best_conf),
            "method": f"template_{ver}",
            "scores": {"yes_score": float(ys), "no_score": float(ns)},
            "page": pg_idx,
            "version": ver,
        }

    # Text fallback: check for unicode checkbox markers near "sex offender"
    text_result = _detect_rso_text_fallback(page_images)
    if text_result is not None:
        return text_result

    return {
        "prediction": "no",
        "confidence": float(best_conf),
        "method": "default",
        "scores": {"yes_score": 0.0, "no_score": 0.0},
        "page": -1,
        "version": "none",
    }


def _detect_rso_text_fallback(page_images: list[np.ndarray]) -> dict[str, Any] | None:
    """Try to detect RSO from embedded PDF text markers (☒☑✓✔)."""
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        return None
    # This fallback is only usable when we have access to the PDF text.
    # In the pipeline it's handled at a higher level via page_raw_text.
    return None
