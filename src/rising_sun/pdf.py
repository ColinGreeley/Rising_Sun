from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pymupdf


PdfSource = Path | str | bytes


def _open_pdf_document(pdf_source: PdfSource) -> pymupdf.Document:
    if isinstance(pdf_source, (bytes, bytearray)):
        return pymupdf.open(stream=bytes(pdf_source), filetype="pdf")
    return pymupdf.open(pdf_source)


def render_pdf_page(pdf_source: PdfSource, dpi: int, page_number: int = 0) -> np.ndarray:
    document = _open_pdf_document(pdf_source)
    scale = dpi / 72.0
    matrix = pymupdf.Matrix(scale, scale)
    page = document.load_page(page_number)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    document.close()
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def render_pdf_pages(pdf_source: PdfSource, dpi: int) -> list[np.ndarray]:
    document = _open_pdf_document(pdf_source)
    scale = dpi / 72.0
    matrix = pymupdf.Matrix(scale, scale)
    pages: list[np.ndarray] = []
    for page in document:
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
        pages.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    document.close()
    return pages
