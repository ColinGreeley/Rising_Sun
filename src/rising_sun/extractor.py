from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from rising_sun.classification import classify_document
from rising_sun.config_loader import load_template
from rising_sun.identity import IdentityExtractor, normalize_person_name, person_name_key
from rising_sun.idoc_lookup import IdocDirectory
from rising_sun.image_ops import checkbox_score, crop_image, mostly_blank, prepare_text_crop
from rising_sun.jotform_parser import parse_jotform_application
from rising_sun.models import FieldSpec
from rising_sun.name_ocr import build_name_ocr_backend
from rising_sun.ocr import OCRTextResult, RapidOcrBackend
from rising_sun.pdf import render_pdf_pages
from rising_sun.rising_sun_packet import parse_rising_sun_packet
from rising_sun.rso_detector import detect_rso_checkbox
from rising_sun.rso_detector import detect_rso_checkbox


logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


BINARY_FIELD_QUESTION_HINTS: dict[str, tuple[int, str]] = {
    "requirements.sex_offender_registration": (1, "doyouneedtoregisterasasexoffender"),
    "requirements.backup_plan": (1, "isthisabackupplanforaninterstatecompactoranicedetainer"),
    "requirements.children_visitation_required": (1, "isatransitionalhomewithvisitationorovernightstaysforchildrenrequired"),
    "history.disciplined_or_discharged": (1, "haveyoueverbeendisciplinedandinvoluntarilydischargedfromatransitionalhome"),
    "history.violent_crimes_or_dor": (1, "haveyouhadanyviolentcrimesordisciplinaryoffensereportsdorforviolence"),
    "release.on_probation_or_parole": (1, "willyoubeonprobationorparolewhenyouarerelease"),
    "release.parole_hearing_completed": (1, "haveyouhadyourparolehearing"),
    "transportation.vehicle_on_site": (1, "willyouhaveavehicleonsite"),
    "background.served_in_military": (1, "haveyoueverservedinthemilitary"),
    "background.enrolled_at_va": (1, "enrolledattheva"),
    "employment.has_employment_upon_release": (1, "doyouhaveemploymentuponrelease"),
    "support.medical_or_mental_health_support": (2, "doyouneedmedicalormentalhealthsupport"),
    "support.communicable_disease": (2, "doyouhaveanycontagiousorcommunicablediseases"),
    "benefits.plan_to_apply": (2, "doyouplantoapplyforssissdissrbmedicareormedicaid"),
    "addictions.has_addictions": (2, "doyouhaveaddictions"),
    "addictions.under_influence_during_crime": (2, "wereyouundertheinfluenceofdrugsalcoholwhencrimewascommitted"),
    "preferences.faith_based_provider": (2, "wouldyoupreferafaithbasedhousingprovider"),
    "housing.previously_in_transitional_home": (2, "haveyoupreviouslyresidedinatransitionalhome"),
}

def assign_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _clean_capture(value: str) -> str:
    value = value.strip(" _:-")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _capture(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return ""
    return _clean_capture(match.group(1))


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _extract_labeled_block(lines: list[str], start_pattern: str, stop_pattern: str) -> str:
    start_regex = re.compile(start_pattern, flags=re.IGNORECASE)
    stop_regex = re.compile(stop_pattern, flags=re.IGNORECASE)
    collecting = False
    parts: list[str] = []

    for line in lines:
        if not collecting:
            match = start_regex.match(line)
            if not match:
                continue
            collecting = True
            suffix = _clean_capture(line[match.end() :])
            if suffix:
                parts.append(suffix)
            continue

        if stop_regex.match(line):
            break
        parts.append(line)

    return _clean_capture(" ".join(parts))


def _capture_between_labels(text: str, start_pattern: str, stop_pattern: str) -> str:
    match = re.search(start_pattern + r"\s*(.*?)\s*" + stop_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return _clean_capture(match.group(1))


def _block_between(lines: list[str], start_prefix: str, stop_prefix: str) -> list[str]:
    try:
        start = next(index for index, line in enumerate(lines) if line.startswith(start_prefix))
        stop = next(index for index, line in enumerate(lines[start + 1 :], start + 1) if line.startswith(stop_prefix))
    except StopIteration:
        return []
    return [line for line in lines[start + 1 : stop] if line]


def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9/]+", "", text.lower().replace("0", "o"))


def _normalize_checkbox_window(text: str) -> str:
    normalized = text.lower().replace("0", "o")
    for source, target in {
        "口": "x",
        "区": "x",
        "☑": "x",
        "☒": "x",
        "α": "x",
    }.items():
        normalized = normalized.replace(source, target)
    return re.sub(r"\s+", "", normalized)


def _parse_binary_answer_from_text(page_text: str, anchor: str) -> str | None:
    compact = _compact_text(page_text)
    start = compact.find(anchor)
    if start == -1:
        lines = _lines(page_text)
        compact_lines = [_compact_text(line) for line in lines]
        for index, compact_line in enumerate(compact_lines):
            combined = compact_line + (compact_lines[index + 1] if index + 1 < len(compact_lines) else "")
            if anchor in combined:
                raw_window = " ".join(lines[index : index + 2])
                normalized_window = _normalize_checkbox_window(raw_window)
                if re.search(r"y[x/.,]+n", normalized_window):
                    return "yes"
                if re.search(r"yn[x/.,k]+", normalized_window):
                    return "no"
                if re.search(r"y(?:x|[.,])$", normalized_window):
                    return "yes"
                if re.search(r"yn(?:x|[.,k/])+$", normalized_window):
                    return "no"
                if re.search(r"y$", normalized_window) and "n" not in normalized_window:
                    return "yes"
                return None
        return None
    window = compact[start : start + len(anchor) + 40]
    tail = window[len(anchor) :]

    if re.search(r"y[/x]+n", tail):
        return "yes"
    if re.search(r"yn[/x]+", tail):
        return "no"
    if re.search(r"yn[xk]+", tail):
        return "no"
    if re.search(r"yno(?:if|$)", tail):
        return "no"
    if re.search(r"yyesn", tail):
        return "yes"
    if re.search(r"y$", tail):
        return "yes"
    return None


def _normalize_date_value(value: str) -> str:
    cleaned = _clean_capture(value)
    return cleaned if re.fullmatch(r"[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4}", cleaned) else ""


def _normalize_age_value(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if not digits:
        return ""
    age = int(digits)
    return str(age) if 17 <= age <= 100 else ""


def _normalize_ssn_last4(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    return digits[-4:] if len(digits) == 4 else ""


def _normalize_idoc_number(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9-]", "", value)
    return cleaned if len(cleaned) >= 4 and sum(char.isdigit() for char in cleaned) >= 3 else ""


def _normalize_gender_value(value: str) -> str:
    compact = re.sub(r"[^a-z]", "", value.lower())
    if compact in {"m", "male"}:
        return "Male"
    if compact in {"f", "female"}:
        return "Female"
    return ""


def _normalize_phone_value(value: str) -> str:
    cleaned = _clean_capture(value)
    digits = re.sub(r"\D", "", cleaned)
    return cleaned if 7 <= len(digits) <= 12 else ""


def _normalize_name_value(value: str) -> str:
    cleaned = value
    # Strip form label fragments that leak into the name region
    # Colons may already be stripped by _clean_capture, so make them optional
    cleaned = re.sub(
        r"\b(?:ne:?|me:?|Age:?|DOB:?|D\.?O\.?B:?|Last\s*\d+\s*digits?\s*(?:of\s*)?(?:SSN)?:?|"
        r"IDoc#?:?|ID[O0]C#?:?|special\s*accommodations?|of\s*information|"
        r"Gender:?|Date\s*of\s*Birth:?|Hia\d+)\b.*",
        "", cleaned, flags=re.IGNORECASE,
    ).strip()
    # Also catch "Age:" without word boundary (e.g. "adAge: 3S")
    cleaned = re.sub(r"[Aa]ge\s*:\s*\d.*$", "", cleaned).strip()
    # Remove stray punctuation/underscores
    cleaned = re.sub(r"[_|]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z'.-]*", cleaned)
    stopwords = {
        "name",
        "ame",
        "lame",
        "current",
        "status",
        "please",
        "check",
        "one",
        "relationship",
        "phone",
        "number",
        "include",
        "area",
        "code",
        "gender",
        "dob",
        "idoc",
        "doc",
        "information",
        "lack",
        "due",
        "needs",
        "helps",
        "support",
        "better",
        "reentry",
        "success",
        "special",
        "accommodations",
        "for",
        "your",
        "and",
        "id",
        "ido",
        "idc",
        "doc",
        "ob",
        "oa",
        "ao",
        "ht",
        "qd",
        "fit",
    }
    blocked_substrings = (
        "inform",
        "ormat",
        "status",
        "relation",
        "phone",
        "number",
        "accommod",
        "support",
        "reentry",
        "success",
        "current",
        "please",
        "check",
        "include",
        "lack",
        "leav",
        "conven",
        "name",
    )
    if not raw_tokens:
        return ""

    best_tokens: list[str] = []
    best_score = float("-inf")
    max_window = min(4, len(raw_tokens))
    for start in range(len(raw_tokens)):
        for width in range(1, max_window + 1):
            window = raw_tokens[start : start + width]
            if not window:
                continue
            score = sum(min(len(token), 10) for token in window)
            score += len(window) * 3
            if len(window) >= 2:
                score += 4
            score -= sum(3 for token in window if len(token) <= 2)
            score -= sum(6 for token in window if token.lower() in stopwords)
            score -= sum(6 for token in window if any(fragment in token.lower() for fragment in blocked_substrings))
            if all(token.lower() not in stopwords for token in window):
                score += 2
            if score > best_score:
                best_tokens = window
                best_score = score

    if len(best_tokens) > 2 and len(best_tokens[0]) <= 2:
        best_tokens = best_tokens[1:]

    while best_tokens and (best_tokens[0].lower() in stopwords or any(fragment in best_tokens[0].lower() for fragment in blocked_substrings)):
        best_tokens = best_tokens[1:]
    while best_tokens and (best_tokens[-1].lower() in stopwords or any(fragment in best_tokens[-1].lower() for fragment in blocked_substrings)):
        best_tokens = best_tokens[:-1]

    cleaned = " ".join(best_tokens)
    cleaned = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", cleaned).strip()
    if sum(c.isalpha() for c in cleaned) < 2:
        return ""
    return cleaned


def _directory_display_name(value: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return ""
    if "," in cleaned:
        last, first = [part.strip() for part in cleaned.split(",", 1)]
        cleaned = f"{first} {last}"
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_county_value(value: str) -> str:
    cleaned = _clean_capture(value).replace("_", "")
    if not cleaned:
        return ""
    if not re.fullmatch(r"[A-Za-z /-]{2,40}", cleaned):
        return ""
    return cleaned.title()


def normalize_text_value(field_key: str, value: str) -> str:
    cleaned = _clean_capture(value)
    if not cleaned:
        return ""

    if field_key in {"applicant.name", "additional_form.name"}:
        return _normalize_name_value(cleaned)
    if field_key == "applicant.idoc_number":
        return _normalize_idoc_number(cleaned)
    if field_key == "applicant.dob" or field_key == "signing.application_date" or field_key == "addictions.date_of_last_use":
        return _normalize_date_value(cleaned)
    if field_key == "applicant.age":
        return _normalize_age_value(cleaned)
    if field_key == "applicant.ssn_last4":
        return _normalize_ssn_last4(cleaned)
    if field_key == "applicant.gender":
        return _normalize_gender_value(cleaned)
    if field_key in {"contacts.personal_phone", "contacts.emergency_phone", "employment.employer_phone_or_email"}:
        return _normalize_phone_value(cleaned)
    if field_key == "criminal.conviction_county":
        return _normalize_county_value(cleaned)
    return cleaned


def derive_checkbox_overrides(page_raw_text: dict[str, str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for field_key, (page_number, anchor) in BINARY_FIELD_QUESTION_HINTS.items():
        page_text = page_raw_text.get(str(page_number), "")
        answer = _parse_binary_answer_from_text(page_text, anchor)
        if answer:
            overrides[field_key] = answer
    return overrides


def derive_overrides(page_raw_text: dict[str, str], pdf_path: Path) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    page1 = page_raw_text.get("1", "")
    page2 = page_raw_text.get("2", "")
    lines1 = _lines(page1)
    lines2 = _lines(page2)

    overrides["applicant.idoc_number"] = _capture(r"IDoc#:\s*([^\n]+)", page1)
    overrides["applicant.dob"] = _capture(r"DOB:\s*([A-Za-z0-9./-]+)", page1)
    overrides["applicant.age"] = _capture(r"Age:\s*([0-9.]+)", page1)
    overrides["applicant.ssn_last4"] = _capture(r"Last 4 digits of SSN:\s*([A-Za-z0-9./-]+)", page1)
    overrides["release.tpd_or_needed_date"] = _capture(r"housing needed:\s*([^\n]+?)\s+Have you had your Parole Hearing", page1)
    overrides["addictions.drug_of_choice"] = _capture(r"drug\(s\)of Choice:\s*([^\n]+?)\s*Dateoflastuse", page2)
    overrides["signing.application_date"] = _capture(r"Date:\s*([0-9./-]+)", page2)
    overrides["additional_form.idoc_number"] = overrides.get("applicant.idoc_number", "")
    overrides["contacts.personal_phone"] = _capture_between_labels(page1, r"3\.\s*Personal Phone Number\s*:?", r"4\.")
    overrides["contacts.emergency_name_relationship"] = _capture_between_labels(page1, r"4\.\s*Emergency Contact Name\s*&\s*Relationship\s*:?", r"5\.")
    overrides["contacts.emergency_phone"] = _capture_between_labels(page1, r"5\.\s*Emergency Contact Phone Number(?:\s*\(include area code\))?\s*:?", r"6\.")
    overrides["criminal.most_recent_convictions"] = _capture_between_labels(page1, r"7\.\s*What was your most recent conviction\(s\)\?", r"8\.")
    overrides["criminal.conviction_county"] = _capture_between_labels(page1, r"8\.\s*What county was the crime committed in\?", r"9\.")
    overrides["support.prescribed_medications"] = _extract_labeled_block(lines2, r"21\.\s*List any prescribed medications you currently take\.\s*", r"22\.")
    overrides["addictions.date_of_last_use"] = _capture(r"Date\s*of\s*last\s*use:?\s*([A-Za-z0-9./-]+)", page2)

    providers = _block_between(lines2, "27.", "28.")
    if providers:
        provider_text = " ".join(providers[:-1] if len(providers) > 1 else providers)
        provider_text = re.sub(r"\s+", " ", provider_text).strip()
        if provider_text:
            overrides["housing.preferred_providers"] = provider_text
        last_line = providers[-1].strip()
        if last_line and len(last_line) <= 40:
            overrides["contacts.case_manager_or_po"] = last_line

    return {key: value for key, value in overrides.items() if value != ""}


def looks_like_prompt_text(value: str) -> bool:
    lowered = value.lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    prompt_markers = [
        "please check",
        "additional information form",
        "housing application",
        "phone number",
        "current status",
        "company:",
        "page 3 of 4",
        "page 4 of 4",
        "v2.0",
        "special accommodations",
        "of information",
        "for your convenience",
        "rising sun sober living",
    ]
    compact_markers = [
        "page3of4",
        "page4of4",
        "idocdatasensitivityclassification",
        "pleasecheckboxes",
    ]
    return any(marker in lowered for marker in prompt_markers) or any(marker in compact for marker in compact_markers)


class ApplicationExtractor:
    def __init__(
        self,
        template_path: Path,
        name_backend: str = "rapid_ensemble",
        enable_idoc_directory: bool | None = None,
    ) -> None:
        self.template_path = template_path
        self.template = load_template(template_path)
        self.field_map = {field.key: field for field in self.template.fields}
        self.ocr = RapidOcrBackend()
        self.identity = IdentityExtractor(page_dpi=max(225, self.template.render_dpi))
        self.name_backend = name_backend
        self.enable_idoc_directory = _env_flag("RISING_SUN_ENABLE_IDOC_DIRECTORY", default=False) if enable_idoc_directory is None else enable_idoc_directory
        if self.enable_idoc_directory:
            try:
                self.idoc_directory = IdocDirectory()
            except Exception:
                logger.warning("Could not load IDOC directory in ApplicationExtractor")
                self.idoc_directory = None
        else:
            self.idoc_directory = None
        self.name_ocr_backend = build_name_ocr_backend(
            name_backend,
            rapid_ocr=self.ocr,
            normalize_name=lambda value: normalize_text_value("applicant.name", value),
        )

    def extract_pdf(self, pdf_path: Path, include_raw_text: bool = True) -> dict[str, Any]:
        # First, check for digital text (Jotform applications) — no OCR needed
        digital_text = self._extract_digital_text(pdf_path)
        if digital_text:
            all_digital = "\n".join(digital_text.values())
            compact = re.sub(r"[^a-z0-9]+", "", all_digital.lower())
            if "jotform" in compact and "applicationforhousing" in compact:
                return parse_jotform_application(pdf_path, digital_text)

        pages = render_pdf_pages(pdf_path, dpi=self.template.render_dpi)
        page_raw_text: dict[str, str] = {}
        for index, page in enumerate(pages, start=1):
            if include_raw_text or index == 1:
                page_raw_text[str(index)] = self.ocr.read_text(page, multiline=True).text

        classification = classify_document(page_raw_text, len(pages))
        identity = self.identity.extract(pdf_path)

        # Multi-page bundles (6/8/10 pages): identify IDOC pages and extract from them
        if len(pages) > 5 and classification.name == "idoc_housing_application_v1":
            idoc_pages, idoc_raw = self._select_idoc_pages(pages, page_raw_text)
            if idoc_pages:
                return self._extract_idoc_fields(pdf_path, idoc_pages, idoc_raw, classification, identity, include_raw_text, original_page_raw_text=page_raw_text)

        if classification.name == "jotform_application":
            result = parse_jotform_application(pdf_path, digital_text or page_raw_text)
            if identity.supervision_number:
                result["field_results"]["applicant.idoc_or_le_number"]["value"] = identity.supervision_number
                result["field_results"]["applicant.idoc_or_le_number"]["confidence"] = 0.9
                result["extracted"].setdefault("applicant", {})["idoc_or_le_number"] = identity.supervision_number
            return result

        if classification.name == "rising_sun_application_packet":
            packet_result = parse_rising_sun_packet(pdf_path, page_raw_text, classification.name, classification.reason)
            packet_result["field_results"]["applicant.idoc_or_le_number"]["value"] = identity.supervision_number
            packet_result["field_results"]["applicant.idoc_or_le_number"]["confidence"] = 0.9 if identity.supervision_number else 0.0
            packet_result["extracted"].setdefault("applicant", {})["idoc_or_le_number"] = identity.supervision_number
            return packet_result

        if not classification.supported:
            return {
                "source_pdf": str(pdf_path),
                "template": self.template.name,
                "document_classification": classification.name,
                "supported_template": False,
                "classification_reason": classification.reason,
                "page_count": len(pages),
                "field_results": {},
                "extracted": {},
                "page_raw_text": page_raw_text,
            }

        return self._extract_idoc_fields(pdf_path, pages, page_raw_text, classification, identity, include_raw_text)

    @staticmethod
    def _extract_digital_text(pdf_path: Path) -> dict[str, str] | None:
        """Extract embedded text from a PDF. Returns None if the PDF is scanned (no text)."""
        import pymupdf
        doc = pymupdf.open(pdf_path)
        pages: dict[str, str] = {}
        total_text = 0
        for i, page in enumerate(doc, start=1):
            text = page.get_text()
            pages[str(i)] = text
            total_text += len(text.strip())
        doc.close()
        return pages if total_text > 100 else None

    def _select_idoc_pages(self, pages: list, page_raw_text: dict[str, str]) -> tuple[list, dict[str, str]]:
        """From a multi-page bundle, select the 4 pages that form the IDOC application."""
        compact_fn = lambda t: re.sub(r"[^a-z0-9]+", "", t.lower())

        idoc_indices: list[int] = []
        for i, _page in enumerate(pages):
            text = page_raw_text.get(str(i + 1), "")
            compact = compact_fn(text)
            # IDOC form pages have specific content markers
            is_idoc = "idochousingapplication" in compact or "currentstatuspleasecheckone" in compact
            is_idoc = is_idoc or ("cmpocontactemail" in compact)
            # Page 2 markers
            is_idoc = is_idoc or ("doyouneedmedical" in compact and "mentalhealthsupport" in compact)
            # Page 3/4 markers (additional information form)
            is_idoc = is_idoc or ("additionalinformationform" in compact)
            if is_idoc:
                idoc_indices.append(i)

        if not idoc_indices:
            return [], {}

        # For 8-page double-scanned bundles (pages interleaved with blanks),
        # try to identify the 4 real IDOC pages
        selected = [pages[i] for i in idoc_indices[:4]]
        selected_text: dict[str, str] = {}
        for j, i in enumerate(idoc_indices[:4]):
            selected_text[str(j + 1)] = page_raw_text.get(str(i + 1), "")
        return selected, selected_text

    def _extract_idoc_fields(
        self,
        pdf_path: Path,
        pages: list,
        page_raw_text: dict[str, str],
        classification,
        identity,
        include_raw_text: bool,
        original_page_raw_text: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Extract fields from IDOC housing application pages using the template."""
        overrides = derive_overrides(page_raw_text, pdf_path)
        special_text_results: dict[str, dict[str, Any]] = {}
        name_field = self.field_map.get("applicant.name")
        if name_field is not None and pages:
            name_result = self._extract_name_field_result(pages[0], page_raw_text.get("1", ""), name_field)
            name_result = self._postprocess_name_field_result(pdf_path, identity, name_result)
            if name_result.get("value"):
                special_text_results["applicant.name"] = dict(name_result)
                special_text_results["additional_form.name"] = dict(name_result)
        if identity.supervision_number:
            overrides["applicant.idoc_number"] = identity.supervision_number
        checkbox_overrides = derive_checkbox_overrides(page_raw_text)

        field_results: dict[str, dict[str, Any]] = {}
        structured: dict[str, Any] = {}

        for field in self.template.fields:
            if field.page > len(pages):
                continue
            page_image = pages[field.page - 1]
            special_result = special_text_results.get(field.key)
            if special_result is not None:
                result = {key: value for key, value in special_result.items() if key != "source"}
                source = str(special_result.get("source", "name_candidate_ensemble"))
            else:
                result = self._extract_field(page_image, field)
                source = "checkbox_crop" if field.kind == "checkbox_group" else "region_ocr"

            if special_result is None and field.key in overrides:
                result["value"] = overrides[field.key]
                result["confidence"] = max(float(result.get("confidence", 0.0)), 0.65)
                source = "raw_text_override"
            elif special_result is None and field.kind == "checkbox_group":
                selected_options = list(result["value"].get("selected_options", []))
                score_keys = set(result["value"].get("scores", {}).keys())
                if score_keys == {"yes", "no"} and len(selected_options) != 1 and field.key in checkbox_overrides:
                    answer = checkbox_overrides[field.key]
                    result["value"]["selected_options"] = [answer]
                    source = "raw_text_checkbox_override"
            elif (
                special_result is None
                and field.kind != "checkbox_group"
                and isinstance(result.get("value"), str)
                and looks_like_prompt_text(result["value"])
            ):
                result["value"] = ""
                result["confidence"] = 0.0
                source = "prompt_filtered_blank"

            if field.kind != "checkbox_group" and isinstance(result.get("value"), str):
                normalized_value = normalize_text_value(field.key, result["value"])
                if normalized_value != result["value"]:
                    result["value"] = normalized_value
                    if not normalized_value:
                        result["confidence"] = 0.0
                        source = "normalized_blank"

            field_results[field.key] = {
                "page": field.page,
                "kind": field.kind,
                "source": source,
                **result,
            }
            assign_nested(structured, field.key, result["value"])

        # RSO override: use template-matching detector at 225 DPI for reliable
        # multi-page, multi-version checkbox detection.
        rso_key = "requirements.sex_offender_registration"
        if rso_key in field_results:
            rso_pages = render_pdf_pages(pdf_path, dpi=225)
            rso_result = detect_rso_checkbox(rso_pages)
            answer = rso_result["prediction"]
            rso_scores = rso_result["scores"]
            field_results[rso_key] = {
                "page": rso_result["page"] + 1,
                "kind": "checkbox_group",
                "source": f"rso_template_{rso_result['method']}",
                "value": {
                    "selected_options": [answer],
                    "scores": {"yes": rso_scores["yes_score"], "no": rso_scores["no_score"]},
                    "threshold": 0.0,
                },
                "confidence": rso_result["confidence"],
            }
            structured.setdefault("requirements", {})["sex_offender_registration"] = {
                "selected_options": [answer],
                "scores": {"yes": rso_scores["yes_score"], "no": rso_scores["no_score"]},
                "threshold": 0.0,
            }

        return {
            "source_pdf": str(pdf_path),
            "template": self.template.name,
            "document_classification": classification.name,
            "supported_template": True,
            "classification_reason": classification.reason,
            "page_count": len(pages),
            "field_results": field_results,
            "extracted": structured,
            "page_raw_text": original_page_raw_text if original_page_raw_text else page_raw_text,
        }

    def _extract_field(self, page_image, field: FieldSpec) -> dict[str, Any]:
        if field.kind == "checkbox_group":
            scores = {name: checkbox_score(page_image, box) for name, box in (field.boxes or {}).items()}
            selected = [name for name, score in scores.items() if score >= field.threshold]
            return {
                "value": {
                    "selected_options": selected,
                    "scores": scores,
                    "threshold": field.threshold,
                },
                "confidence": 1.0,
            }

        if field.box is None:
            return {"value": "", "confidence": 0.0}

        raw_crop = crop_image(page_image, field.box)
        if mostly_blank(raw_crop):
            return {"value": "", "confidence": 1.0}

        multiline = field.kind == "multiline_text"
        result = self._read_text_variants(raw_crop, multiline=multiline)
        return {"value": result.text, "confidence": result.confidence}

    def _extract_name_field_result(self, page_image, page_text: str, field: FieldSpec) -> dict[str, Any]:
        if field.box is None:
            return {"value": "", "confidence": 0.0, "candidates": []}

        candidates = self.name_ocr_backend.extract_candidates(page_image, page_text, field.box)
        if not candidates:
            return {"value": "", "confidence": 0.0, "candidates": []}

        best = candidates[0]
        return {
            "value": best.value,
            "confidence": best.confidence,
            "candidates": [
                {
                    "source": candidate.source,
                    "value": candidate.value,
                    "confidence": round(candidate.confidence, 3),
                    "score": round(candidate.score, 3),
                }
                for candidate in candidates[:8]
            ],
            "source": best.source,
        }

    def _postprocess_name_field_result(self, pdf_path: Path, identity, name_result: dict[str, Any]) -> dict[str, Any]:
        directory = self.idoc_directory
        if directory is None:
            return name_result

        directory_name = ""
        directory_source = ""

        if getattr(identity, "supervision_number", ""):
            directory_name = _directory_display_name(directory.lookup_by_number(identity.supervision_number) or "")
            if directory_name:
                directory_source = "directory_by_number"

        if not directory_name:
            return name_result

        updated = dict(name_result)
        updated_candidates = list(updated.get("candidates", []))
        directory_key = person_name_key(directory_name)
        current_value = str(updated.get("value", "") or "")
        current_key = person_name_key(current_value)

        if not current_value:
            updated["value"] = directory_name
            updated["confidence"] = max(float(updated.get("confidence", 0.0)), 0.85)
            updated["source"] = directory_source
        elif directory_key and current_key and directory_key == current_key:
            updated["value"] = directory_name
            updated["confidence"] = max(float(updated.get("confidence", 0.0)), 0.9)
            updated["source"] = f"{updated.get('source', 'name_candidate_ensemble')}+{directory_source}_canonicalized"
        elif normalize_person_name(current_value) == normalize_person_name(directory_name):
            updated["value"] = directory_name
            updated["confidence"] = max(float(updated.get("confidence", 0.0)), 0.9)
            updated["source"] = f"{updated.get('source', 'name_candidate_ensemble')}+{directory_source}_canonicalized"
        else:
            return name_result

        directory_candidate = {
            "source": directory_source,
            "value": directory_name,
            "confidence": round(float(updated.get("confidence", 0.0)), 3),
            "score": 999.0,
        }
        seen_values = {normalize_person_name(directory_name)}
        deduped_candidates = [directory_candidate]
        for candidate in updated_candidates:
            candidate_value = str(candidate.get("value", "") or "")
            normalized_candidate = normalize_person_name(candidate_value)
            if not normalized_candidate or normalized_candidate in seen_values:
                continue
            seen_values.add(normalized_candidate)
            deduped_candidates.append(candidate)
        updated["candidates"] = deduped_candidates[:8]
        return updated

    def _read_text_variants(self, crop, multiline: bool) -> OCRTextResult:
        variants = [
            self.ocr.read_text(crop, multiline=multiline),
            self.ocr.read_text(prepare_text_crop(crop, multiline=multiline), multiline=multiline),
        ]
        ranked = sorted(
            variants,
            key=lambda item: (bool(item.text), len(item.text), item.confidence),
            reverse=True,
        )
        return ranked[0]
