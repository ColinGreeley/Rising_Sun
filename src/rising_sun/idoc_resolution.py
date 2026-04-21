from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from rising_sun.identity import normalize_person_name, person_name_key


@dataclass(frozen=True)
class NameMatchResult:
    ocr_name: str | None
    idoc_name: str
    level: str
    score: float
    sequence_ratio: float
    overlap_count: int
    first_name_match: bool
    last_name_match: bool


@dataclass(frozen=True)
class RankedIdocCandidate:
    idoc_number: str
    idoc_name: str
    matched_ocr_name: str | None
    match_level: str
    match_score: float
    sequence_ratio: float
    overlap_count: int
    first_name_match: bool
    last_name_match: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "idoc_number": self.idoc_number,
            "idoc_name": self.idoc_name,
            "matched_ocr_name": self.matched_ocr_name,
            "match_level": self.match_level,
            "match_score": round(self.match_score, 3),
            "sequence_ratio": round(self.sequence_ratio, 3),
            "overlap_count": self.overlap_count,
            "first_name_match": self.first_name_match,
            "last_name_match": self.last_name_match,
        }


def _unique_names(names: list[str] | None) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for name in names or []:
        normalized = normalize_person_name(name)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(str(name).strip())
    return unique


def score_name_match(ocr_name: str, idoc_name: str) -> NameMatchResult:
    normalized_ocr = normalize_person_name(ocr_name)
    normalized_idoc = normalize_person_name(idoc_name)
    if not normalized_ocr or not normalized_idoc:
        return NameMatchResult(
            ocr_name=ocr_name or None,
            idoc_name=idoc_name,
            level="none",
            score=0.0,
            sequence_ratio=0.0,
            overlap_count=0,
            first_name_match=False,
            last_name_match=False,
        )

    ocr_tokens = normalized_ocr.split()
    idoc_tokens = normalized_idoc.split()
    overlap = set(ocr_tokens) & set(idoc_tokens)
    overlap_count = len(overlap)
    overlap_ratio = overlap_count / max(len(ocr_tokens), len(idoc_tokens))
    same_token_set = len(ocr_tokens) == len(idoc_tokens) and set(ocr_tokens) == set(idoc_tokens)

    ocr_key = person_name_key(ocr_name)
    idoc_key = person_name_key(idoc_name)
    first_name_match = bool(ocr_key and idoc_key and ocr_key[0] == idoc_key[0])
    last_name_match = bool(ocr_key and idoc_key and ocr_key[1] == idoc_key[1])

    forward_ratio = SequenceMatcher(None, normalized_ocr, normalized_idoc).ratio()
    reversed_ocr = " ".join(reversed(ocr_tokens)) if len(ocr_tokens) >= 2 else normalized_ocr
    reversed_ratio = SequenceMatcher(None, reversed_ocr, normalized_idoc).ratio()
    sequence_ratio = max(forward_ratio, reversed_ratio)

    score = overlap_ratio * 45.0 + sequence_ratio * 35.0
    if normalized_ocr == normalized_idoc:
        score += 45.0
    if same_token_set:
        score += 30.0
    if first_name_match:
        score += 25.0
    if last_name_match:
        score += 35.0
    if first_name_match and last_name_match:
        score += 30.0
    if overlap_count == 0 and not last_name_match:
        score -= 30.0

    if normalized_ocr == normalized_idoc or same_token_set:
        level = "exact"
    elif first_name_match and last_name_match:
        level = "exact"
    elif last_name_match and (overlap_count >= 2 or sequence_ratio >= 0.78):
        level = "strong"
    elif overlap_count >= 1 or sequence_ratio >= 0.60:
        level = "partial"
    elif score >= 20.0:
        level = "weak"
    else:
        level = "none"

    return NameMatchResult(
        ocr_name=ocr_name or None,
        idoc_name=idoc_name,
        level=level,
        score=score,
        sequence_ratio=sequence_ratio,
        overlap_count=overlap_count,
        first_name_match=first_name_match,
        last_name_match=last_name_match,
    )


def best_name_match(ocr_names: list[str] | None, idoc_name: str) -> NameMatchResult:
    names = _unique_names(ocr_names)
    if not names:
        return NameMatchResult(
            ocr_name=None,
            idoc_name=idoc_name,
            level="none",
            score=0.0,
            sequence_ratio=0.0,
            overlap_count=0,
            first_name_match=False,
            last_name_match=False,
        )

    ranked = [score_name_match(name, idoc_name) for name in names]
    ranked.sort(
        key=lambda item: (
            item.score,
            item.last_name_match,
            item.first_name_match,
            item.overlap_count,
            item.sequence_ratio,
        ),
        reverse=True,
    )
    return ranked[0]


def rank_verified_candidates(
    results: list[tuple[str, dict[str, Any]]],
    ocr_names: list[str] | None,
) -> list[RankedIdocCandidate]:
    ranked: list[RankedIdocCandidate] = []
    for candidate, info in results:
        idoc_name = str(info.get("name", "") or "")
        match = best_name_match(ocr_names, idoc_name)
        ranked.append(
            RankedIdocCandidate(
                idoc_number=candidate,
                idoc_name=idoc_name,
                matched_ocr_name=match.ocr_name,
                match_level=match.level,
                match_score=match.score,
                sequence_ratio=match.sequence_ratio,
                overlap_count=match.overlap_count,
                first_name_match=match.first_name_match,
                last_name_match=match.last_name_match,
            )
        )

    ranked.sort(
        key=lambda item: (
            item.match_score,
            item.last_name_match,
            item.first_name_match,
            item.overlap_count,
            item.sequence_ratio,
        ),
        reverse=True,
    )
    return ranked
