from __future__ import annotations

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


def load_ground_truth_rows(workbook_path: Path, sheet_name: str = "2026") -> list[GroundTruthRow]:
    frame = pd.read_excel(workbook_path, sheet_name=sheet_name)
    if "Name" in frame.columns:
        name_column = "Name"
        supervision_column = "Supervision #"
    else:
        name_column = "Last, First Name"
        supervision_column = "Supervision #"

    rows: list[GroundTruthRow] = []
    for _, row in frame.iterrows():
        source_name = str(row.get(name_column, "") or "").strip()
        normalized_name = normalize_person_name(source_name)
        if not normalized_name:
            continue
        rows.append(
            GroundTruthRow(
                source_name=source_name,
                supervision_number=normalize_supervision_number(row.get(supervision_column, "")),
                normalized_name=normalized_name,
                first_last_key=person_name_key(source_name),
            )
        )
    return rows


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