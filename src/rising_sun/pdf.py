from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pymupdf


def render_pdf_page(pdf_path: Path, dpi: int, page_number: int = 0) -> np.ndarray:
    document = pymupdf.open(pdf_path)
    scale = dpi / 72.0
    matrix = pymupdf.Matrix(scale, scale)
    page = document.load_page(page_number)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    document.close()
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def render_pdf_pages(pdf_path: Path, dpi: int) -> list[np.ndarray]:
    document = pymupdf.open(pdf_path)
    scale = dpi / 72.0
    matrix = pymupdf.Matrix(scale, scale)
    pages: list[np.ndarray] = []
    for page in document:
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
        pages.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    document.close()
    return pages
