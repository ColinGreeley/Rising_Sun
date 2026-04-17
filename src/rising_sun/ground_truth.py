from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rising_sun.identity import clean_pdf_stem_name, normalize_person_name, normalize_supervision_number, person_name_key


@dataclass(frozen=True)
class GroundTruthRow:
    source_name: str
    supervision_number: str
    normalized_name: str
    first_last_key: tuple[str, str] | None
    is_rso: bool = False


def _parse_supervision_and_rso(raw_value: str, rso_column_value: str | None = None) -> tuple[str, bool]:
    """Extract a clean supervision number and RSO flag from a raw cell value.

    Handles 2025-style embedded RSO (e.g. ``RSO/123456``, ``123456/RSO``)
    and 2026-style separate RSO column.

    Returns ``(normalized_number, is_rso)``.
    """
    raw = str(raw_value or "").strip()

    # Check dedicated RSO column first (2026 style)
    is_rso = bool(rso_column_value and str(rso_column_value).strip().upper() == "YES")

    # Check for RSO embedded in the supervision value (2025 style)
    if "RSO" in raw.upper():
        is_rso = True
        # Extract the numeric portion
        digits = re.findall(r"\d{5,6}", raw)
        if digits:
            return normalize_supervision_number(digits[0]), is_rso
        # Fall through to normal normalization after stripping RSO
        raw = re.sub(r"RSO\s*/?\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = raw.strip("/").strip()

    # Skip non-numeric entries (Denied, NS, MISD, FED, etc.)
    stripped = re.sub(r"[^0-9]", "", raw)
    if len(stripped) < 5:
        return "", is_rso

    return normalize_supervision_number(raw), is_rso


def load_ground_truth_rows(
    workbook_path: Path,
    sheet_name: str = "2026",
) -> list[GroundTruthRow]:
    frame = pd.read_excel(workbook_path, sheet_name=sheet_name)
    if "Name" in frame.columns:
        name_column = "Name"
        supervision_column = "Supervision #"
    else:
        name_column = "Last, First Name"
        supervision_column = "Supervision #"

    has_rso_column = "RSO" in frame.columns

    rows: list[GroundTruthRow] = []
    for _, row in frame.iterrows():
        source_name = str(row.get(name_column, "") or "").strip()
        normalized_name = normalize_person_name(source_name)
        if not normalized_name:
            continue

        rso_col_val = str(row.get("RSO", "") or "").strip() if has_rso_column else None
        sup_number, is_rso = _parse_supervision_and_rso(
            row.get(supervision_column, ""),
            rso_col_val,
        )

        rows.append(
            GroundTruthRow(
                source_name=source_name,
                supervision_number=sup_number,
                normalized_name=normalized_name,
                first_last_key=person_name_key(source_name),
                is_rso=is_rso,
            )
        )
    return rows


def load_ground_truth_rows_combined(
    workbook_path: Path,
    sheet_names: list[str] | None = None,
) -> list[GroundTruthRow]:
    """Load ground truth from multiple sheets and merge.

    Defaults to ``["2025", "2026"]``.  Skips entries without a valid
    supervision number (Denied, NS, MISD, FED, etc.).
    """
    if sheet_names is None:
        sheet_names = ["2025", "2026"]

    combined: list[GroundTruthRow] = []
    seen: set[tuple[str, str]] = set()  # (normalized_name, supervision_number)
    for sheet in sheet_names:
        for row in load_ground_truth_rows(workbook_path, sheet_name=sheet):
            if not row.supervision_number:
                continue
            key = (row.normalized_name, row.supervision_number)
            if key in seen:
                continue
            seen.add(key)
            combined.append(row)
    return combined


def match_ground_truth(pdf_path: Path, extracted_name: str, rows: list[GroundTruthRow]) -> GroundTruthRow | None:
    candidates = [clean_pdf_stem_name(pdf_path), extracted_name]

    for candidate in candidates:
        normalized = normalize_person_name(candidate)
        if not normalized:
            continue
        exact = [row for row in rows if row.normalized_name == normalized]
        if len(exact) == 1:
            return exact[0]

    for candidate in candidates:
        key = person_name_key(candidate)
        if key is None:
            continue
        first_last_matches = [row for row in rows if row.first_last_key == key]
        if len(first_last_matches) == 1:
            return first_last_matches[0]
        if len(first_last_matches) > 1:
            candidate_tokens = set(normalize_person_name(candidate).split())
            ranked = sorted(
                first_last_matches,
                key=lambda row: len(candidate_tokens.intersection(set(row.normalized_name.split()))),
                reverse=True,
            )
            if ranked:
                return ranked[0]

    return None