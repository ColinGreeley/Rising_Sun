from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import cv2

from rising_sun.config_loader import load_template
from rising_sun.image_ops import crop_image
from rising_sun.models import FieldSpec
from rising_sun.pdf import render_pdf_pages


@dataclass(frozen=True)
class ReviewEntry:
    source_pdf: Path
    field: str
    page: int
    status: str


def _sanitize_fragment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "item"


def _field_crop_box(field: FieldSpec) -> tuple[float, float, float, float] | None:
    if field.box is not None:
        return field.box
    if not field.boxes:
        return None
    left = min(box[0] for box in field.boxes.values())
    top = min(box[1] for box in field.boxes.values())
    right = max(box[2] for box in field.boxes.values())
    bottom = max(box[3] for box in field.boxes.values())
    return (left, top, right, bottom)


def load_review_entries(review_csv: Path) -> list[ReviewEntry]:
    rows: list[ReviewEntry] = []
    with review_csv.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") == "unsupported_document":
                continue
            field = row.get("field", "")
            page = row.get("page", "")
            if not field or not page:
                continue
            rows.append(
                ReviewEntry(
                    source_pdf=Path(row["source_pdf"]),
                    field=field,
                    page=int(page),
                    status=row.get("status", ""),
                )
            )
    return rows


def export_review_crops(review_csv: Path, template_path: Path, output_dir: Path, limit: int | None = None) -> int:
    template = load_template(template_path)
    field_map = {field.key: field for field in template.fields}
    entries = load_review_entries(review_csv)
    if limit is not None:
        entries = entries[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    page_cache: dict[tuple[Path, int], object] = {}

    for entry in entries:
        field = field_map.get(entry.field)
        if field is None:
            continue
        crop_box = _field_crop_box(field)
        if crop_box is None:
            continue

        cache_key = (entry.source_pdf, template.render_dpi)
        pages = page_cache.get(cache_key)
        if pages is None:
            pages = render_pdf_pages(entry.source_pdf, dpi=template.render_dpi)
            page_cache[cache_key] = pages

        if entry.page > len(pages):
            continue

        crop = crop_image(pages[entry.page - 1], crop_box, padding=5)
        pdf_dir = output_dir / _sanitize_fragment(entry.source_pdf.stem)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        target_name = f"p{entry.page:02d}_{_sanitize_fragment(entry.field)}_{_sanitize_fragment(entry.status)}.png"
        target_path = pdf_dir / target_name
        cv2.imwrite(str(target_path), crop)
        written += 1

    return written
