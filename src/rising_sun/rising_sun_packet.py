from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _normalize_spaces(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _lines(text: str) -> list[str]:
    return [line.strip() for line in _normalize_spaces(text).splitlines() if line.strip()]


def _capture(pattern: str, text: str, flags: int = 0) -> str:
    match = re.search(pattern, text, flags)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip(" _:-")


def _capture_between(text: str, start: str, stop: str) -> str:
    pattern = re.escape(start) + r"\s*(.*?)\s*" + re.escape(stop)
    return _capture(pattern, text, flags=re.IGNORECASE | re.DOTALL)


def _stem_name(pdf_path: Path) -> str:
    return re.sub(r"\s+\d{1,2}-\d{1,2}-\d{2}$", "", pdf_path.stem).strip()


def _clean_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" _:-.,")


def _valid_gender(value: str) -> str:
    lowered = value.lower().strip()
    if lowered in {"male", "m"}:
        return "Male"
    if lowered in {"female", "f"}:
        return "Female"
    return ""


def _valid_numeric_identifier(value: str) -> str:
    cleaned = re.sub(r"[^0-9]", "", value)
    return cleaned if len(cleaned) >= 4 else ""


def _valid_date(value: str) -> str:
    cleaned = _clean_value(value)
    return cleaned if re.fullmatch(r"[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4}", cleaned) else ""


def _valid_phone(value: str) -> str:
    cleaned = _clean_value(value)
    digits = re.sub(r"\D", "", cleaned)
    return cleaned if len(digits) >= 10 else ""


def _valid_email(value: str) -> str:
    cleaned = _clean_value(value)
    return cleaned if re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", cleaned) else ""


def _valid_city(value: str) -> str:
    cleaned = _clean_value(value)
    allowed = {
        "Boise",
        "Caldwell",
        "Lewiston",
        "Twin Falls",
        "Idaho Falls",
        "Pocatello",
    }
    for city in allowed:
        if city.lower() in cleaned.lower():
            return city
    return ""


def _bool_from_regex(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    answer = match.group(1).strip().lower()
    if answer in {"yes", "no"}:
        return answer
    return ""


def _find_line_after(lines: list[str], anchor_pattern: str, *, max_offset: int = 4, ignore: set[str] | None = None) -> str:
    ignore = ignore or set()
    for index, line in enumerate(lines):
        if re.search(anchor_pattern, line, flags=re.IGNORECASE):
            for candidate in lines[index + 1 : index + 1 + max_offset]:
                normalized = candidate.lower()
                if candidate and all(token not in normalized for token in ignore):
                    return candidate
    return ""


def _bool_field(value: str, source: str = "raw_text_parser", confidence: float = 0.9) -> dict[str, Any]:
    selected = [value] if value in {"yes", "no"} else []
    scores = {"yes": 1.0 if value == "yes" else 0.0, "no": 1.0 if value == "no" else 0.0}
    return {
        "kind": "checkbox_group",
        "source": source,
        "value": {"selected_options": selected, "scores": scores, "threshold": 0.5},
        "confidence": confidence if selected else 0.0,
    }


def _text_field(value: str, source: str = "raw_text_parser", confidence: float = 0.85) -> dict[str, Any]:
    return {
        "kind": "text",
        "source": source,
        "value": value.strip(),
        "confidence": confidence if value.strip() else 0.0,
    }


def _multiline_field(value: str, source: str = "raw_text_parser", confidence: float = 0.85) -> dict[str, Any]:
    return {
        "kind": "multiline_text",
        "source": source,
        "value": value.strip(),
        "confidence": confidence if value.strip() else 0.0,
    }


def _bool_from_context(text: str, anchor_pattern: str) -> str:
    compact_lines = _lines(text)
    for index, line in enumerate(compact_lines):
        if re.search(anchor_pattern, line, flags=re.IGNORECASE):
            window = compact_lines[index : index + 4]
            joined = " ".join(window).lower()
            if re.search(r"\byes\b", joined) and not re.search(r"\bno\b", joined):
                return "yes"
            if re.search(r"\bno\b", joined) and not re.search(r"\byes\b", joined):
                return "no"
    return ""


def _assign_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def parse_rising_sun_packet(pdf_path: Path, page_raw_text: dict[str, str], classification_name: str, classification_reason: str) -> dict[str, Any]:
    page1 = page_raw_text.get("1", "")
    page2 = page_raw_text.get("2", "")
    page3 = page_raw_text.get("3", "")
    lines1 = _lines(page1)
    lines2 = _lines(page2)

    application_date = _valid_date(_capture(r"Today's Date:?\s*([0-9/.-]+)", page1, flags=re.IGNORECASE))
    idoc_or_le_number = _valid_numeric_identifier(_capture(r"IDOC\s*or\s*LE\s*#\s*([A-Za-z0-9-]+)", page1, flags=re.IGNORECASE))
    gender = _valid_gender(_find_line_after(lines1, r"^gender:?$", ignore={"name", "please print clearly", "address", "physical address"}))
    dob = _valid_date(_capture(r"Date of Birth\s*([0-9/.-]+)", page1, flags=re.IGNORECASE))
    age = _capture(r"Age:?\s*([0-9]{1,3})", page1, flags=re.IGNORECASE)
    phone = _valid_phone(_capture(r"Phone(?: Number)?:?\s*([0-9() -]{7,})", page1, flags=re.IGNORECASE | re.DOTALL))
    email = _valid_email(_capture(r"[Ee]-?mail\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+)", page1, flags=re.IGNORECASE | re.DOTALL))
    emergency_contact = _capture(r"Emergency Contact:?\s*([A-Za-z0-9 .()/-]{6,})\s*Name(?:\s*&\s*Phone| and Phone)?", page1, flags=re.IGNORECASE)
    preferred_city = _valid_city(_capture_between(page1, "preferred city of", "Have you previously") or _find_line_after(lines1, r"preferred city of", ignore={"housing", "please selecta"}))
    drug_of_choice = _clean_value(_capture(r"Drug\(s\) of Choice:?\s*([^\n]+)", page1, flags=re.IGNORECASE) or _capture(r"Alcohol and/or drugs\s*([^\n]+)", page2, flags=re.IGNORECASE))
    if "?" in drug_of_choice.lower():
        drug_of_choice = ""
    date_of_last_use = _valid_date(_capture(r"Date[- ]of[- ]Last[- ]Use:?\s*([0-9/.-]+)", page1, flags=re.IGNORECASE) or _capture(r"Date of last use\s*([0-9/.-]+)", page2, flags=re.IGNORECASE))
    housing_needed_by = _valid_date(_capture(r"Date Housing Needed:?\s*([^\n]+)", page1, flags=re.IGNORECASE) or _capture(r"Housing needed by\s*([^\n]+)", page2, flags=re.IGNORECASE))
    medications = _capture(r"List any medications you are prescribed:?\s*([^\n]+)", page1, flags=re.IGNORECASE) or _capture(r"List any medications\s*([^\n]+)\s*you are prescribed", page2, flags=re.IGNORECASE)
    charges = _capture(r"List all charges:?\s*(.*?)\s*(?:Name of Case Manager|Will you be on probation|I have completed)", page2, flags=re.IGNORECASE | re.DOTALL)
    violent_explanation = _capture(r"violent charges:?\s*(.*?)\s*(?:Name of Case Manager|Will you be on probation|I have completed)", page2, flags=re.IGNORECASE | re.DOTALL)
    case_manager = _capture(r"Name of Case\s*Manager:?\s*([^\n]+)", page2, flags=re.IGNORECASE) or _capture(r"Name of CaseManager,? if applicable:?\s*([^\n]+)", page2, flags=re.IGNORECASE)
    if re.search(r"if applicable|name of case", case_manager, flags=re.IGNORECASE):
        case_manager = ""
    county_reporting_to = _capture(r"Name of Idaho County you will be reporting to:?\s*([^\n]+)", page2, flags=re.IGNORECASE)
    probation_officer = _capture(r"Probation/Parole Officer(?: Name)?(?:, if known)?:?\s*([^\n]+)", page2, flags=re.IGNORECASE)
    if re.search(r"name,? ?if known", probation_officer, flags=re.IGNORECASE):
        probation_officer = ""
    signature_name = _capture(r"Your Signature \(Type Name\)\s*([^\n]+)", page2, flags=re.IGNORECASE) or _capture(r"Print Name:?\s*([^\n]+)", page2, flags=re.IGNORECASE)
    application_date = application_date or _valid_date(_capture(r"Date\s*([0-9/.-]+)", page2, flags=re.IGNORECASE) or _capture(r"Date\s*([0-9/.-]+)", page3, flags=re.IGNORECASE))

    page1_joined = "\n".join(lines1)
    page2_joined = "\n".join(lines2)

    field_results: dict[str, dict[str, Any]] = {
        "applicant.name": _text_field(_stem_name(pdf_path)),
        "applicant.idoc_or_le_number": _text_field(idoc_or_le_number),
        "applicant.gender": _text_field(gender),
        "applicant.date_of_birth": _text_field(dob),
        "applicant.age": _text_field(age),
        "contacts.phone": _text_field(phone),
        "contacts.email": _text_field(email),
        "contacts.emergency_contact": _text_field(emergency_contact),
        "housing.preferred_city": _text_field(preferred_city),
        "addictions.drug_of_choice": _text_field(drug_of_choice),
        "addictions.date_of_last_use": _text_field(date_of_last_use),
        "housing.needed_by": _text_field(housing_needed_by),
        "support.prescribed_medications": _text_field(medications),
        "criminal.charges": _multiline_field(charges),
        "criminal.violent_charge_explanation": _multiline_field(violent_explanation),
        "contacts.case_manager": _text_field(case_manager),
        "supervision.reporting_county": _text_field(county_reporting_to),
        "supervision.probation_or_parole_officer": _text_field(probation_officer),
        "signing.signature_name": _text_field(signature_name),
        "signing.application_date": _text_field(application_date),
        "history.previously_resided_at_rising_sun": _bool_field(_bool_from_regex(page1, r"Have you previously\s*(Yes|No)\s*resided at Rising Sun") or _bool_from_context(page1_joined, r"previously\s+resided\s+at\s+rising\s+sun")),
        "history.alcoholic_or_addict": _bool_field(_bool_from_regex(page1, r"Are you an\s*(Yes|No)\s*alcoholic/addict") or _bool_from_context(page1_joined, r"alcoholic/addict")),
        "requirements.sex_offender_registration": _bool_field(_bool_from_regex(page2, r"register\s*(Yes|No)\s*as a sex offender") or _bool_from_context(page2_joined, r"register\s+as\s+a\s+sex\s+offender")),
        "employment.currently_employed": _bool_field(_bool_from_regex(page2, r"currently\s*(Yes|No)\s*employed") or _bool_from_context(page2_joined, r"currently\s+employed")),
        "criminal.convicted_misdemeanor_or_felony": _bool_field(_bool_from_regex(page1 + "\n" + page2, r"convicted of a\s*(Yes|No)\s*misdemeanor or\s*felony") or _bool_from_context(page1_joined + "\n" + page2_joined, r"convicted\s+of\s+a\s+misdemeanor\s+or\s+felony")),
        "criminal.under_influence_during_crime": _bool_field(_bool_from_regex(page2, r"influence of\s*(Yes|No)\s*drugs/alcohol") or _bool_from_context(page2_joined, r"under\s+the\s+influence.*crime\s+was\s+committed|where\s+you\s+under\s+the\s+influence")),
        "supervision.on_probation_or_parole_while_housed": _bool_field(_bool_from_regex(page2, r"Will you be on\s*(Yes|No)\s*probation or parole") or _bool_from_context(page2_joined, r"probation\s+or\s+parole\s+while\s+in\s+housing")),
    }

    structured: dict[str, Any] = {}
    for field_name, field in field_results.items():
        _assign_nested(structured, field_name, field["value"])

    return {
        "source_pdf": str(pdf_path),
        "template": "rising_sun_application_packet_v1",
        "document_classification": classification_name,
        "supported_template": True,
        "classification_reason": classification_reason,
        "page_count": max(int(key) for key in page_raw_text.keys()),
        "field_results": {
            key: {"page": 1 if not key.startswith(("criminal.", "supervision.", "signing.")) else 2, **value}
            for key, value in field_results.items()
        },
        "extracted": structured,
        "page_raw_text": page_raw_text,
    }
