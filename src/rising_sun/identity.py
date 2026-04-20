from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rising_sun.classification import DocumentClassification, classify_document
from rising_sun.image_ops import crop_image, prepare_text_crop
from rising_sun.ocr import TesseractDigitBackend, _make_rapid_ocr
from rising_sun.pdf import render_pdf_page


def clean_pdf_stem_name(pdf_path: Path) -> str:
    return re.sub(r"\s+\d{1,2}-\d{1,2}-\d{2}$", "", pdf_path.stem).strip()


def normalize_person_name(name: str) -> str:
    value = str(name or "").strip()
    value = re.sub(r"\s+", " ", value)
    if "," in value:
        last, first = [part.strip() for part in value.split(",", 1)]
        value = f"{first} {last}"
    value = re.sub(r"[^A-Za-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


# Common nickname → canonical first name (lowercase)
_NICKNAMES: dict[str, str] = {
    "bob": "robert", "bobby": "robert", "rob": "robert", "robby": "robert",
    "bill": "william", "billy": "william", "will": "william", "willy": "william", "willie": "william",
    "jim": "james", "jimmy": "james", "jamie": "james",
    "joe": "joseph", "joey": "joseph",
    "mike": "michael", "mikey": "michael",
    "dan": "daniel", "danny": "daniel",
    "dave": "david", "davey": "david",
    "tom": "thomas", "tommy": "thomas",
    "dick": "richard", "rick": "richard", "ricky": "richard", "rich": "richard",
    "steve": "steven", "stevie": "steven",
    "tony": "anthony",
    "chris": "christopher",
    "matt": "matthew",
    "pat": "patrick",
    "ed": "edward", "eddie": "edward", "ted": "edward", "teddy": "edward",
    "larry": "lawrence",
    "jerry": "gerald", "gerry": "gerald",
    "terry": "terrence", "terri": "terrence",
    "charlie": "charles", "chuck": "charles", "chas": "charles",
    "sam": "samuel", "sammy": "samuel",
    "jack": "john", "johnny": "john", "jon": "john",
    "alex": "alexander", "al": "albert",
    "ben": "benjamin", "benny": "benjamin",
    "ken": "kenneth", "kenny": "kenneth",
    "tim": "timothy", "timmy": "timothy",
    "brad": "bradley",
    "fred": "frederick", "freddy": "frederick",
    "jeff": "jeffrey",
    "greg": "gregory",
    "nick": "nicholas",
    "andy": "andrew", "drew": "andrew",
    "ron": "ronald", "ronny": "ronald",
    "don": "donald", "donny": "donald",
    "liz": "elizabeth", "beth": "elizabeth", "lizzy": "elizabeth",
    "kathy": "katherine", "kate": "katherine", "katie": "katherine", "katy": "katherine",
    "jenny": "jennifer", "jen": "jennifer",
    "chris": "christine",
    "sue": "susan", "suzy": "susan",
    "pam": "pamela",
    "peggy": "margaret", "maggie": "margaret", "meg": "margaret",
    "debbie": "deborah", "deb": "deborah",
    "barb": "barbara",
    "becky": "rebecca",
    "cindy": "cynthia",
    "sandy": "sandra",
    "mandy": "amanda",
    "vicky": "victoria", "vicki": "victoria",
    "cathy": "catherine",
}


def _canonical_first(name: str) -> str:
    """Return the canonical form of a first name (lowercase)."""
    return _NICKNAMES.get(name, name)


def person_name_key(name: str) -> tuple[str, str] | None:
    tokens = normalize_person_name(name).split()
    if len(tokens) < 2:
        return None
    first = _canonical_first(tokens[0])
    last = tokens[-1]
    return first, last


def normalize_supervision_number(value: str) -> str:
    candidates = normalize_supervision_candidates(value)
    result = max(candidates, key=_supervision_number_score, default="")
    return result.lstrip("0") or result


def normalize_supervision_candidates(value: str) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []

    if re.fullmatch(r"\d+\.0+", raw):
        raw = raw.split(".", 1)[0]

    cleaned = re.sub(r"[^A-Za-z0-9/\\|!&$%()\[\]{}]", "", raw.upper())
    if not cleaned:
        return []

    digit_like = sum(char.isdigit() for char in cleaned)
    if digit_like < max(3, len(cleaned) - 2):
        return []

    substitutions = {
        "0": ["0"],
        "1": ["1"],
        "2": ["2"],
        "3": ["3"],
        "4": ["4"],
        "5": ["5"],
        "6": ["6"],
        "7": ["7"],
        "8": ["8"],
        "9": ["9"],
        "I": ["1"],
        "L": ["1"],
        "J": ["1"],
        "/": ["1"],
        "\\": ["1"],
        "|": ["1"],
        "!": ["1"],
        "[": ["1"],
        "]": ["1"],
        "{": ["1"],
        "}": ["1"],
        "O": ["0"],
        "Q": ["0"],
        "D": ["0"],
        "C": ["0"],
        "U": ["0"],
        "S": ["5"],
        "Z": ["2"],
        "B": ["8"],
        "&": ["8"],
        "$": ["8"],
        "%": ["8"],
        "G": ["6"],
        "E": ["6"],
        "A": ["4"],
        "H": ["4"],
        "T": ["7"],
        "(": ["", "1"],
        ")": ["", "1"],
    }

    variants = [""]
    for char in cleaned:
        options = substitutions.get(char, [""])
        next_variants: list[str] = []
        for base in variants:
            for option in options:
                candidate = base + option
                if len(candidate) <= 8:
                    next_variants.append(candidate)
        variants = next_variants[:16]
        if not variants:
            break

    normalized = sorted({candidate for candidate in variants if 4 <= len(candidate) <= 8 and candidate.isdigit()})
    return normalized


def _supervision_number_score(value: str) -> tuple[int, int]:
    length = len(value)
    length_score = {
        6: 5,
        5: 4,
        7: 4,
        8: 3,
        4: 2,
    }.get(length, 0)
    return length_score, length


def _digit_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        return max(len(left), len(right))
    return sum(left_char != right_char for left_char, right_char in zip(left, right))


def _most_common(values: list[str]) -> str:
    if not values:
        return ""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda item: (item[1], _supervision_number_score(item[0]), item[0]))[0]


def _looks_like_date_number(value: str) -> bool:
    if len(value) != 8 or not value.isdigit():
        return False
    if value.endswith(("2024", "2025", "2026", "2027")):
        return True
    month = int(value[:2])
    day = int(value[2:4])
    year = int(value[4:])
    return 1 <= month <= 12 and 1 <= day <= 31 and 2024 <= year <= 2027


def _is_plausible_idoc_number(value: str) -> bool:
    return value.isdigit() and 5 <= len(value) <= 6 and not _looks_like_date_number(value)


REFERENCE_ANCHORS: dict[str, dict[str, tuple[float, float]]] = {
    "idoc_housing_application_v1": {
        "idoc": (0.317, 0.065),
        "name": (0.067, 0.250),
        "gender": (0.082, 0.296),
    },
    "rising_sun_application_packet": {
        "idoc_or_le": (0.583, 0.317),
        "name": (0.103, 0.347),
        "gender": (0.719, 0.349),
    },
    "email_forward_application_packet": {
        "idoc_or_le": (0.204, 0.419),
        "name": (0.201, 0.447),
        "gender": (0.204, 0.475),
    },
}


@dataclass(frozen=True)
class LayoutRegistration:
    subtype: str
    shift_x: float
    shift_y: float
    anchor_count: int


@dataclass(frozen=True)
class IdentityExtraction:
    name: str
    supervision_number: str
    classification: DocumentClassification
    page_text: str
    layout_subtype: str


class IdentityExtractor:
    def __init__(self, page_dpi: int = 225, enable_tesseract: bool = False) -> None:
        self.page_dpi = page_dpi
        self.enable_tesseract = enable_tesseract
        self._ocr = _make_rapid_ocr()
        self._tesseract = TesseractDigitBackend() if enable_tesseract else None

    def extract(self, pdf_path: Path) -> IdentityExtraction:
        page_image = render_pdf_page(pdf_path, dpi=self.page_dpi, page_number=0)
        raw_results = self._read_raw(page_image)
        page_text = self._join_lines(raw_results)
        classification = classify_document({"1": page_text}, 1)
        return self.extract_from_page(pdf_path, page_image, raw_results, page_text, classification)

    def debug_supervision_crops(self, pdf_path: Path, dpi: int = 400) -> tuple[str, str, dict[str, np.ndarray]]:
        page_image = render_pdf_page(pdf_path, dpi=dpi, page_number=0)
        raw_results = self._read_raw(page_image)
        page_text = self._join_lines(raw_results)
        classification = classify_document({"1": page_text}, 1)
        registration = self._estimate_registration(classification.name, raw_results, page_image.shape)

        crops: dict[str, np.ndarray] = {}
        if classification.name == "idoc_housing_application_v1":
            for label, box in self._idoc_crop_boxes(registration):
                crops[label] = crop_image(page_image, self._shift_box(box, registration, x_weight=0.8, y_weight=1.0), padding=6)
        else:
            for label, box in self._packet_crop_boxes(classification.name):
                crops[label] = crop_image(page_image, self._shift_box(box, registration, x_weight=1.0, y_weight=1.0), padding=6)

            label_item = None
            for item in raw_results:
                normalized = re.sub(r"\s+", "", item[1].lower())
                if "idocorle" in normalized:
                    label_item = item
                    break
            if label_item is not None:
                label_box = label_item[0]
                x1 = min(point[0] for point in label_box)
                y1 = min(point[1] for point in label_box)
                x2 = max(point[0] for point in label_box)
                y2 = max(point[1] for point in label_box)
                left = max(0, int(x1 - 40))
                top = max(0, int(y1 - 30))
                right = min(page_image.shape[1], int(x2 + 420))
                bottom = min(page_image.shape[0], int(y2 + 120))
                crops["label_relative"] = page_image[top:bottom, left:right].copy()
            else:
                predicted_box = self._predicted_packet_label_box(classification.name, registration)
                if predicted_box is not None:
                    crops["predicted_label_relative"] = crop_image(page_image, predicted_box, padding=6)

        return classification.name, registration.subtype, crops

    def extract_from_page(
        self,
        pdf_path: Path,
        page_image: np.ndarray,
        raw_results: list[tuple[Any, str, float]] | None,
        page_text: str,
        classification: DocumentClassification,
    ) -> IdentityExtraction:
        name = clean_pdf_stem_name(pdf_path)
        registration = self._estimate_registration(classification.name, raw_results or [], page_image.shape)
        if classification.name == "idoc_housing_application_v1":
            supervision_number = self._extract_idoc_number(pdf_path, page_image, page_text, registration)
        else:
            supervision_number = self._extract_rising_sun_number(
                pdf_path,
                classification.name,
                page_image,
                raw_results or [],
                page_text,
                registration,
            )
        return IdentityExtraction(
            name=name,
            supervision_number=supervision_number,
            classification=classification,
            page_text=page_text,
            layout_subtype=registration.subtype,
        )

    def _read_raw(self, image: np.ndarray) -> list[tuple[Any, str, float]]:
        result, _ = self._ocr(image)
        return result or []

    def _idoc_crop_boxes(self, registration: LayoutRegistration) -> list[tuple[str, tuple[float, float, float, float]]]:
        boxes = [
            ("default", (0.62, 0.24, 0.95, 0.34)),
            ("mid", (0.66, 0.236, 0.93, 0.31)),
            ("tight", (0.72, 0.238, 0.90, 0.298)),
        ]
        if registration.subtype in {"idoc_standard", "idoc_left_shifted", "idoc_upshifted"}:
            boxes.append(("left_tight", (0.68, 0.238, 0.90, 0.298)))
        return boxes

    def _join_lines(self, raw_results: list[tuple[Any, str, float]]) -> str:
        ordered = sorted(raw_results, key=lambda item: (min(point[1] for point in item[0]), min(point[0] for point in item[0])))
        return "\n".join(item[1].strip() for item in ordered if item[1].strip())

    def _ocr_crop_candidates(
        self,
        crop: np.ndarray,
        use_tesseract: bool | None = None,
        tighten: bool = True,
    ) -> list[str]:
        candidates: list[str] = []
        proposals = self._number_region_proposals(crop) if tighten else [crop]
        for proposal in proposals:
            variant_images = self._variant_images(proposal)
            variants = [
                proposal,
                prepare_text_crop(proposal, multiline=False),
                variant_images["gray"],
                variant_images["binary"],
                variant_images["line_stripped"],
                variant_images["line_stripped_dilated"],
            ]
            for variant in variants:
                result, _ = self._ocr(variant)
                if not result:
                    continue
                ordered = sorted(result, key=lambda item: (min(point[1] for point in item[0]), min(point[0] for point in item[0])))
                candidates.append(" ".join(item[1] for item in ordered if item[1].strip()))
                candidates.extend(item[1] for item in ordered if item[1].strip())
        if use_tesseract is None:
            use_tesseract = self.enable_tesseract
        if use_tesseract and self._tesseract is not None:
            candidates.extend(self._tesseract.read_digits(crop))
        return candidates

    def _number_region_proposals(self, crop: np.ndarray) -> list[np.ndarray]:
        proposals = [crop]
        seen_shapes = {crop.shape[:2]}
        for min_x_ratio in [0.18, 0.30]:
            tightened = self._tight_number_crop(crop, min_x_ratio=min_x_ratio)
            if tightened is None:
                continue
            key = tightened.shape[:2]
            if key in seen_shapes:
                continue
            seen_shapes.add(key)
            proposals.append(tightened)
        return proposals

    def _tight_number_crop(self, crop: np.ndarray, min_x_ratio: float) -> np.ndarray | None:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, crop.shape[1] // 6), 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        cleaned = cv2.subtract(binary, horizontal_lines)
        cleaned = cv2.dilate(cleaned, np.ones((2, 2), np.uint8), iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        crop_h, crop_w = crop.shape[:2]
        min_area = max(25, int(crop_h * crop_w * 0.001))
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            center_x = x + width / 2
            center_y = y + height / 2
            if area < min_area:
                continue
            if height < max(10, int(crop_h * 0.12)):
                continue
            if center_x < crop_w * min_x_ratio:
                continue
            if center_y > crop_h * 0.68:
                continue
            boxes.append((x, y, x + width, y + height))

        if not boxes:
            return None

        x1 = max(0, min(box[0] for box in boxes) - 8)
        y1 = max(0, min(box[1] for box in boxes) - 6)
        x2 = min(crop_w, max(box[2] for box in boxes) + 10)
        y2 = min(crop_h, max(box[3] for box in boxes) + 6)
        tightened = crop[y1:y2, x1:x2].copy()
        if tightened.size == 0 or tightened.shape[1] >= crop.shape[1] * 0.97:
            return None
        return tightened

    def _variant_images(self, crop: np.ndarray) -> dict[str, np.ndarray]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dilated = cv2.dilate(inv, np.ones((2, 2), np.uint8), iterations=1)
        eroded = cv2.erode(inv, np.ones((2, 2), np.uint8), iterations=1)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, gray.shape[1] // 5), 1))
        horizontal_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horizontal_kernel)
        line_stripped = cv2.subtract(inv, horizontal_lines)
        line_stripped_dilated = cv2.dilate(line_stripped, np.ones((2, 2), np.uint8), iterations=1)
        return {
            "orig": cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
            "gray": cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
            "blur": cv2.resize(blur, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
            "binary": cv2.resize(binary, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
            "inv": cv2.resize(inv, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
            "dilated": cv2.resize(dilated, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
            "eroded": cv2.resize(eroded, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
            "line_stripped": cv2.resize(cv2.bitwise_not(line_stripped), None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
            "line_stripped_dilated": cv2.resize(cv2.bitwise_not(line_stripped_dilated), None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
        }

    def _normalized_candidates_for_variants(self, crop: np.ndarray, variant_names: list[str]) -> list[str]:
        normalized: list[str] = []
        variants = self._variant_images(crop)
        for variant_name in variant_names:
            variant = variants.get(variant_name)
            if variant is None:
                continue
            result, _ = self._ocr(variant)
            for item in result or []:
                for token in re.findall(r"[A-Za-z0-9/\\|!&$%()\[\]{}]{4,14}", item[1]):
                    candidate = normalize_supervision_number(token)
                    if candidate:
                        normalized.append(candidate)
        return normalized

    def _extract_candidate_tokens(self, candidate_texts: list[str]) -> list[str]:
        extracted: list[str] = []
        for candidate in candidate_texts:
            match = re.search(r"IDOC#[:\s]*([A-Za-z0-9]{4,10})", candidate, flags=re.IGNORECASE)
            if match:
                extracted.append(match.group(1))
            extracted.extend(re.findall(r"[A-Za-z0-9/\\|!&$%()\[\]{}]{4,10}", candidate))
        return extracted

    def _normalized_candidate_counts(
        self,
        values: list[str],
        reject_date_like: bool = False,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in values:
            normalized = normalize_supervision_number(value)
            if not normalized:
                continue
            if reject_date_like and _looks_like_date_number(normalized):
                continue
            counts[normalized] = counts.get(normalized, 0) + 1
        return counts

    def _best_normalized_candidate(
        self,
        values: list[str],
        reject_date_like: bool = False,
        prefer_frequency: bool = False,
    ) -> str:
        best = ""
        best_score = (-1, -1)
        counts = self._normalized_candidate_counts(values, reject_date_like=reject_date_like)
        for value in values:
            normalized = normalize_supervision_number(value)
            if not normalized:
                continue
            if reject_date_like and _looks_like_date_number(normalized):
                continue
            score = _supervision_number_score(normalized)
            if score > best_score:
                best = normalized
                best_score = score
        if prefer_frequency and counts:
            return max(counts.items(), key=lambda item: (_supervision_number_score(item[0]), item[1], item[0]))[0]
        return best

    def _prefer_high_dpi_consensus(
        self,
        current_best: str,
        high_dpi_values: list[str],
        reject_date_like: bool = False,
    ) -> str:
        counts = self._normalized_candidate_counts(high_dpi_values, reject_date_like=reject_date_like)
        if not counts:
            return current_best

        high_dpi_best = max(counts.items(), key=lambda item: (_supervision_number_score(item[0]), item[1], item[0]))[0]
        high_dpi_score = _supervision_number_score(high_dpi_best)
        current_score = _supervision_number_score(current_best)
        if high_dpi_score > current_score:
            return high_dpi_best

        if (
            current_best
            and high_dpi_best != current_best
            and high_dpi_score == current_score
            and counts.get(current_best, 0) <= 2
            and counts[high_dpi_best] >= 8
        ):
            return high_dpi_best

        return current_best

    def _find_anchor_boxes(
        self,
        raw_results: list[tuple[Any, str, float]],
        image_shape: tuple[int, ...],
    ) -> dict[str, tuple[float, float, float, float]]:
        anchors: dict[str, tuple[float, float, float, float]] = {}
        height, width = image_shape[:2]
        for box, text, _ in raw_results:
            compact = "".join(char for char in text.lower() if char.isalnum())
            x1 = min(point[0] for point in box) / width
            y1 = min(point[1] for point in box) / height
            x2 = max(point[0] for point in box) / width
            y2 = max(point[1] for point in box) / height
            if "idocorle" in compact and "idoc_or_le" not in anchors:
                anchors["idoc_or_le"] = (x1, y1, x2, y2)
            elif "idochousingapplication" in compact and "idoc" not in anchors:
                anchors["idoc"] = (x1, y1, x2, y2)
            elif compact.startswith("1name") or compact.startswith("name"):
                anchors.setdefault("name", (x1, y1, x2, y2))
            elif compact.startswith("2gender") or compact.startswith("gender"):
                anchors.setdefault("gender", (x1, y1, x2, y2))
        return anchors

    def _estimate_registration(
        self,
        classification_name: str,
        raw_results: list[tuple[Any, str, float]],
        image_shape: tuple[int, ...],
    ) -> LayoutRegistration:
        references = REFERENCE_ANCHORS.get(classification_name, {})
        anchors = self._find_anchor_boxes(raw_results, image_shape)
        deltas: list[tuple[float, float]] = []
        for key, (ref_x, ref_y) in references.items():
            if key not in anchors:
                continue
            anchor_x, anchor_y, _, _ = anchors[key]
            deltas.append((anchor_x - ref_x, anchor_y - ref_y))

        if deltas:
            shift_x = sum(item[0] for item in deltas) / len(deltas)
            shift_y = sum(item[1] for item in deltas) / len(deltas)
        else:
            shift_x = 0.0
            shift_y = 0.0

        subtype = self._classify_layout_subtype(classification_name, anchors, shift_x, shift_y)
        return LayoutRegistration(subtype=subtype, shift_x=shift_x, shift_y=shift_y, anchor_count=len(deltas))

    def _classify_layout_subtype(
        self,
        classification_name: str,
        anchors: dict[str, tuple[float, float, float, float]],
        shift_x: float,
        shift_y: float,
    ) -> str:
        if classification_name == "idoc_housing_application_v1":
            name_anchor = anchors.get("name")
            if not anchors:
                return "idoc_unanchored"
            if name_anchor and name_anchor[0] < 0.060:
                return "idoc_left_shifted"
            if shift_y < -0.010:
                return "idoc_upshifted"
            if shift_x > 0.010:
                return "idoc_right_shifted"
            return "idoc_standard"

        if classification_name == "rising_sun_application_packet":
            if "idoc_or_le" not in anchors:
                return "packet_label_missing"
            if shift_x < -0.010:
                return "packet_left_shifted"
            if shift_x > 0.010:
                return "packet_right_shifted"
            if shift_y > 0.010:
                return "packet_downshifted"
            return "packet_scanned_standard"

        if classification_name == "email_forward_application_packet":
            if "idoc_or_le" not in anchors:
                return "email_forward_label_missing"
            if abs(shift_x) > 0.010 or abs(shift_y) > 0.010:
                return "email_forward_shifted"
            return "email_forward_standard"

        return classification_name

    def _shift_box(
        self,
        box: tuple[float, float, float, float],
        registration: LayoutRegistration,
        x_weight: float = 1.0,
        y_weight: float = 1.0,
    ) -> tuple[float, float, float, float]:
        shift_x = max(-0.05, min(0.05, registration.shift_x * x_weight))
        shift_y = max(-0.05, min(0.05, registration.shift_y * y_weight))
        x1 = max(0.0, min(0.98, box[0] + shift_x))
        y1 = max(0.0, min(0.98, box[1] + shift_y))
        x2 = max(x1 + 0.01, min(1.0, box[2] + shift_x))
        y2 = max(y1 + 0.01, min(1.0, box[3] + shift_y))
        return x1, y1, x2, y2

    def _packet_crop_boxes(self, classification_name: str) -> list[tuple[str, tuple[float, float, float, float]]]:
        if classification_name == "email_forward_application_packet":
            return [
                ("fixed_default", (0.15, 0.395, 0.64, 0.47)),
                ("midwide", (0.22, 0.401, 0.56, 0.462)),
                ("tight", (0.30, 0.403, 0.51, 0.456)),
            ]
        return [
            ("fixed_default", (0.54, 0.29, 0.86, 0.37)),
            ("midwide", (0.60, 0.295, 0.84, 0.36)),
            ("tight", (0.67, 0.305, 0.82, 0.35)),
        ]

    def _predicted_packet_label_box(
        self,
        classification_name: str,
        registration: LayoutRegistration,
    ) -> tuple[float, float, float, float] | None:
        references = REFERENCE_ANCHORS.get(classification_name, {})
        anchor = references.get("idoc_or_le")
        if anchor is None:
            return None
        anchor_x = anchor[0] + registration.shift_x
        anchor_y = anchor[1] + registration.shift_y
        if classification_name == "email_forward_application_packet":
            return (
                max(0.0, anchor_x - 0.04),
                max(0.0, anchor_y - 0.03),
                min(1.0, anchor_x + 0.42),
                min(1.0, anchor_y + 0.05),
            )
        return (
            max(0.0, anchor_x - 0.05),
            max(0.0, anchor_y - 0.03),
            min(1.0, anchor_x + 0.30),
            min(1.0, anchor_y + 0.06),
        )

    def _extract_idoc_number(
        self,
        pdf_path: Path,
        page_image: np.ndarray,
        page_text: str,
        registration: LayoutRegistration,
    ) -> str:
        candidates: list[str] = []
        default_crop_candidates: list[str] = []

        for pattern in [
            r"IDOC#[:\s]*([A-Za-z0-9]{4,10})",
            r"IDOc#[:\s]*([A-Za-z0-9]{4,10})",
            r"IDoc#[:\s]*([A-Za-z0-9]{4,10})",
        ]:
            match = re.search(pattern, page_text, flags=re.IGNORECASE)
            if match:
                candidates.append(match.group(1))

        for label, box in self._idoc_crop_boxes(registration):
            crop = crop_image(page_image, self._shift_box(box, registration, x_weight=0.8, y_weight=1.0), padding=8)
            crop_candidates = self._ocr_crop_candidates(crop)
            candidates.extend(crop_candidates)
            if label == "default":
                default_crop_candidates.extend(crop_candidates)

        extracted = self._extract_candidate_tokens(candidates)
        default_extracted = self._extract_candidate_tokens(default_crop_candidates)

        extracted = [value for value in extracted if _is_plausible_idoc_number(normalize_supervision_number(value))]
        default_extracted = [value for value in default_extracted if _is_plausible_idoc_number(normalize_supervision_number(value))]

        best = self._best_normalized_candidate(extracted)
        voted_best = self._best_normalized_candidate(extracted, prefer_frequency=True)
        all_counts = self._normalized_candidate_counts(extracted)
        default_counts = self._normalized_candidate_counts(default_extracted)
        if (
            voted_best
            and voted_best != best
            and _supervision_number_score(voted_best) == _supervision_number_score(best)
            and all_counts.get(best, 0) <= 3
            and default_counts.get(voted_best, 0) >= 6
        ):
            best = voted_best

        if _supervision_number_score(best)[0] >= 5 and registration.subtype != "idoc_left_shifted":
            return best

        high_dpi_page = render_pdf_page(pdf_path, dpi=max(self.page_dpi, 300), page_number=0)
        high_dpi_raw = self._read_raw(high_dpi_page)
        high_dpi_registration = self._estimate_registration("idoc_housing_application_v1", high_dpi_raw, high_dpi_page.shape)
        high_dpi_candidates: list[str] = []
        for _, box in self._idoc_crop_boxes(high_dpi_registration):
            retry_crop = crop_image(high_dpi_page, self._shift_box(box, high_dpi_registration, x_weight=0.8, y_weight=1.0), padding=4)
            for candidate in self._ocr_crop_candidates(retry_crop):
                high_dpi_candidates.extend(re.findall(r"[A-Za-z0-9/\\|!&$%()\[\]{}]{4,12}", candidate))

        high_dpi_candidates = [
            value for value in high_dpi_candidates if _is_plausible_idoc_number(normalize_supervision_number(value))
        ]

        retry_best = self._prefer_high_dpi_consensus(best, high_dpi_candidates)
        if retry_best != best:
            return retry_best

        if registration.subtype == "idoc_left_shifted":
            consensus = self._idoc_left_shifted_consensus(pdf_path, registration)
            if consensus:
                return consensus
        return best if _is_plausible_idoc_number(best) else ""

    def _idoc_left_shifted_consensus(self, pdf_path: Path, registration: LayoutRegistration) -> str:
        page = render_pdf_page(pdf_path, dpi=max(self.page_dpi, 400), page_number=0)
        raw = self._read_raw(page)
        refined_registration = self._estimate_registration("idoc_housing_application_v1", raw, page.shape)
        crop = crop_image(page, self._shift_box((0.72, 0.238, 0.90, 0.298), refined_registration, x_weight=0.8, y_weight=1.0), padding=6)
        inv_candidates = [candidate for candidate in self._normalized_candidates_for_variants(crop, ["inv"]) if len(candidate) == 6]
        dilated_candidates = [candidate for candidate in self._normalized_candidates_for_variants(crop, ["dilated"]) if len(candidate) == 6]
        common = [candidate for candidate in inv_candidates if candidate in dilated_candidates]
        return _most_common(common)

    def _extract_rising_sun_number(
        self,
        pdf_path: Path,
        classification_name: str,
        page_image: np.ndarray,
        raw_results: list[tuple[Any, str, float]],
        page_text: str,
        registration: LayoutRegistration,
    ) -> str:
        constrain_to_idoc_shape = classification_name == "jotform_application"
        direct_pattern_candidate = ""
        pattern = re.search(r"IDOC\s*or\s*LE\s*#?\s*([A-Za-z0-9]{4,10})", page_text, flags=re.IGNORECASE)
        if pattern:
            direct_pattern_candidate = normalize_supervision_number(pattern.group(1))
            if constrain_to_idoc_shape and not _is_plausible_idoc_number(direct_pattern_candidate):
                direct_pattern_candidate = ""

        label_item = None
        for item in raw_results:
            normalized = re.sub(r"\s+", "", item[1].lower())
            if "idocorle" in normalized:
                label_item = item
                break

        candidates: list[str] = []
        if label_item is not None:
            label_box = label_item[0]
            x1 = min(point[0] for point in label_box)
            y1 = min(point[1] for point in label_box)
            x2 = max(point[0] for point in label_box)
            y2 = max(point[1] for point in label_box)
            label_height = max(1.0, y2 - y1)

            for box, text, _ in raw_results:
                bx1 = min(point[0] for point in box)
                by1 = min(point[1] for point in box)
                bx2 = max(point[0] for point in box)
                by2 = max(point[1] for point in box)
                same_row = abs(by1 - y1) <= label_height * 1.4
                right_side = bx1 >= x2 - 20
                below_row = by1 >= y1 and by1 <= y1 + label_height * 3.0 and bx1 >= x1
                if same_row and right_side:
                    candidates.append(text)
                elif below_row and re.search(r"[0-9A-Za-z]{4,}", text):
                    candidates.append(text)

            height, width = page_image.shape[:2]
            left = max(0, int(x1 - 40))
            top = max(0, int(y1 - 30))
            right = min(width, int(x2 + 420))
            bottom = min(height, int(y2 + 120))
            crop = page_image[top:bottom, left:right]
            candidates.extend(self._ocr_crop_candidates(crop, tighten=False))
        else:
            for _, box in self._packet_crop_boxes(classification_name):
                fixed_crop = crop_image(page_image, self._shift_box(box, registration, x_weight=1.0, y_weight=1.0), padding=8)
                candidates.extend(self._ocr_crop_candidates(fixed_crop, tighten=False))
            predicted_box = self._predicted_packet_label_box(classification_name, registration)
            if predicted_box is not None:
                predicted_crop = crop_image(page_image, predicted_box, padding=8)
                candidates.extend(self._ocr_crop_candidates(predicted_crop, tighten=False))

        extracted_tokens: list[str] = []
        for candidate in candidates:
            for token in re.findall(r"[A-Za-z0-9/\\|!&$%()\[\]{}]{4,12}", candidate):
                extracted_tokens.append(token)

        if direct_pattern_candidate:
            extracted_tokens.append(direct_pattern_candidate)

        if constrain_to_idoc_shape:
            extracted_tokens = [
                value for value in extracted_tokens if _is_plausible_idoc_number(normalize_supervision_number(value))
            ]

        best = self._best_normalized_candidate(
            extracted_tokens,
            reject_date_like=classification_name == "email_forward_application_packet",
        )
        if constrain_to_idoc_shape:
            return best if _is_plausible_idoc_number(best) else ""
        if classification_name != "rising_sun_application_packet":
            return best

        high_dpi_page = render_pdf_page(pdf_path, dpi=max(self.page_dpi, 300), page_number=0)
        best_score = _supervision_number_score(best)
        high_dpi_tokens: list[str] = []
        for _, box in self._packet_crop_boxes(classification_name):
            retry_crop = crop_image(high_dpi_page, self._shift_box(box, registration, x_weight=1.0, y_weight=1.0), padding=4)
            retry_tokens: list[str] = []
            for candidate in self._ocr_crop_candidates(retry_crop, tighten=False):
                retry_tokens.extend(re.findall(r"[A-Za-z0-9/\\|!&$%()\[\]{}]{4,12}", candidate))
            high_dpi_tokens.extend(retry_tokens)
            retry_best = self._best_normalized_candidate(
                retry_tokens,
                reject_date_like=classification_name == "email_forward_application_packet",
            )
            retry_score = _supervision_number_score(retry_best)
            if retry_score > best_score:
                best = retry_best
                best_score = retry_score
            if retry_best and best and retry_score == best_score and _digit_distance(retry_best, best) <= 1:
                best = retry_best

        predicted_box = self._predicted_packet_label_box(classification_name, registration)
        if predicted_box is not None:
            retry_crop = crop_image(high_dpi_page, predicted_box, padding=4)
            retry_tokens: list[str] = []
            for candidate in self._ocr_crop_candidates(retry_crop, tighten=False):
                retry_tokens.extend(re.findall(r"[A-Za-z0-9/\\|!&$%()\[\]{}]{4,12}", candidate))
            high_dpi_tokens.extend(retry_tokens)
            retry_best = self._best_normalized_candidate(
                retry_tokens,
                reject_date_like=classification_name == "email_forward_application_packet",
            )
            retry_score = _supervision_number_score(retry_best)
            if retry_score > best_score:
                best = retry_best
                best_score = retry_score

        high_dpi_consensus = self._prefer_high_dpi_consensus(
            best,
            high_dpi_tokens,
            reject_date_like=classification_name == "email_forward_application_packet",
        )
        if high_dpi_consensus != best:
            best = high_dpi_consensus
            best_score = _supervision_number_score(best)

        if registration.subtype == "packet_right_shifted":
            packet_override = self._packet_right_shifted_override(pdf_path, registration, best)
            if packet_override:
                return packet_override
        return best

    def _packet_right_shifted_override(self, pdf_path: Path, registration: LayoutRegistration, current_best: str) -> str:
        if not current_best or len(current_best) != 6:
            return ""
        page = render_pdf_page(pdf_path, dpi=max(self.page_dpi, 400), page_number=0)
        raw = self._read_raw(page)
        refined_registration = self._estimate_registration("rising_sun_application_packet", raw, page.shape)
        crop = crop_image(page, self._shift_box((0.60, 0.295, 0.84, 0.36), refined_registration, x_weight=1.0, y_weight=1.0), padding=6)
        blur_candidates = [candidate for candidate in self._normalized_candidates_for_variants(crop, ["blur"]) if len(candidate) == 6]
        blur_best = _most_common(blur_candidates)
        if blur_best and _digit_distance(blur_best, current_best) == 1:
            return blur_best
        return ""