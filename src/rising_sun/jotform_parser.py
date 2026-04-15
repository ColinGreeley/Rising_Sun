from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _valid_date(value: str) -> str:
    cleaned = _clean(value)
    return cleaned if re.fullmatch(r"[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4}", cleaned) else ""


def _valid_phone(value: str) -> str:
    cleaned = _clean(value)
    digits = re.sub(r"\D", "", cleaned)
    return cleaned if len(digits) >= 7 else ""


def _valid_email(value: str) -> str:
    cleaned = _clean(value)
    return cleaned if re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", cleaned) else ""


def _valid_gender(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"male", "m"}:
        return "Male"
    if lowered in {"female", "f"}:
        return "Female"
    return ""


def _valid_numeric_id(value: str) -> str:
    cleaned = re.sub(r"[^0-9]", "", value).lstrip("0")
    return cleaned if 5 <= len(cleaned) <= 6 else ""


def _text_field(value: str, source: str = "jotform_text", confidence: float = 0.95) -> dict[str, Any]:
    return {
        "kind": "text",
        "source": source,
        "value": value.strip(),
        "confidence": confidence if value.strip() else 0.0,
    }


def _multiline_field(value: str, source: str = "jotform_text", confidence: float = 0.95) -> dict[str, Any]:
    return {
        "kind": "multiline_text",
        "source": source,
        "value": value.strip(),
        "confidence": confidence if value.strip() else 0.0,
    }


def _bool_field(value: str, source: str = "jotform_text", confidence: float = 0.95) -> dict[str, Any]:
    selected = [value] if value in {"yes", "no"} else []
    scores = {"yes": 1.0 if value == "yes" else 0.0, "no": 1.0 if value == "no" else 0.0}
    return {
        "kind": "checkbox_group",
        "source": source,
        "value": {"selected_options": selected, "scores": scores, "threshold": 0.5},
        "confidence": confidence if selected else 0.0,
    }


def _yes_no(value: str) -> str:
    lowered = value.strip().lower()
    if lowered == "yes":
        return "yes"
    if lowered == "no":
        return "no"
    return ""


def _assign_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _extract_digital_text(pdf_path: Path) -> dict[str, str]:
    """Extract text directly from a digital PDF using PyMuPDF."""
    doc = pymupdf.open(pdf_path)
    pages: dict[str, str] = {}
    for i, page in enumerate(doc, start=1):
        pages[str(i)] = page.get_text()
    doc.close()
    return pages


def _find_value_after(lines: list[str], anchor: str, *, max_lines: int = 3) -> str:
    """Find the value on lines immediately after a line matching the anchor."""
    anchor_lower = anchor.lower()
    for i, line in enumerate(lines):
        if anchor_lower in line.lower():
            # Check if there's a value on the same line after the anchor
            idx = line.lower().index(anchor_lower)
            rest = line[idx + len(anchor):].strip()
            if rest:
                return _clean(rest)
            # Otherwise look at subsequent lines
            parts: list[str] = []
            for j in range(i + 1, min(i + 1 + max_lines, len(lines))):
                candidate = lines[j].strip()
                if not candidate:
                    continue
                # Stop if this looks like another label
                if _is_label_line(candidate):
                    break
                parts.append(candidate)
            return _clean(" ".join(parts))
    return ""


def _find_multiline_after(lines: list[str], anchor: str, stop_anchor: str | None = None) -> str:
    """Collect multiple lines after an anchor until a stop anchor or another label."""
    anchor_lower = anchor.lower()
    stop_lower = stop_anchor.lower() if stop_anchor else None
    collecting = False
    parts: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not collecting:
            if anchor_lower in stripped.lower():
                collecting = True
                # Check same-line content
                idx = stripped.lower().index(anchor_lower)
                rest = stripped[idx + len(anchor):].strip()
                if rest:
                    parts.append(rest)
            continue
        if stop_lower and stop_lower in stripped.lower():
            break
        if not stripped:
            continue
        parts.append(stripped)
    return _clean(" ".join(parts))


# Jotform labels that appear on their own line
_JOTFORM_LABELS = {
    "today's date", "idoc or le#", "name", "gender", "physical address",
    "city", "state", "zipcode", "e-mail", "date of birth", "age:",
    "phone number", "emergency contact:", "name & phone",
    "please select a",
    "preferred city of", "have you previously", "are you an",
    "alcohol and/or drugs", "date of last use", "do you have to register",
    "housing needed by", "phone:", "are you currently", "employer:",
    "list any medications", "have you been", "where you under",
    "list all charges", "explain in detail", "name of case",
    "will you be on", "probation/parole", "your signature",
    "date", "you can edit this",
}


def _is_label_line(line: str) -> bool:
    lowered = line.strip().lower()
    if not lowered:
        return False
    return any(lowered.startswith(label) for label in _JOTFORM_LABELS)


def _find_bool_after(lines: list[str], anchor: str) -> str:
    value = _find_value_after(lines, anchor, max_lines=2)
    return _yes_no(value)


def parse_jotform_application(pdf_path: Path, page_raw_text: dict[str, str] | None = None) -> dict[str, Any]:
    """Parse a digital-text Jotform application PDF.

    These are email-forwarded Jotform submissions with structured key-value
    pairs directly in the PDF text layer. No OCR needed.
    """
    if page_raw_text is None:
        page_raw_text = _extract_digital_text(pdf_path)

    all_text = "\n".join(page_raw_text.get(str(i), "") for i in range(1, len(page_raw_text) + 1))
    lines = [line.strip() for line in all_text.splitlines() if line.strip()]

    # Extract fields from the structured key-value format
    name = _find_value_after(lines, "Name", max_lines=1)
    # Filter out common false positives from the name field
    if name and any(kw in name.lower() for kw in ["gender", "physical", "address", "male", "female"]):
        name = ""
    # Strip parenthetical suffixes like "(Child)" from names
    if name:
        name = re.sub(r"\s*\(.*?\)\s*$", "", name).strip()
    idoc_or_le = _valid_numeric_id(_find_value_after(lines, "IDOC or LE#", max_lines=1))
    gender = _valid_gender(_find_value_after(lines, "Gender", max_lines=1))
    dob = _valid_date(_find_value_after(lines, "Date of Birth", max_lines=1))
    age = _find_value_after(lines, "Age:", max_lines=1)
    if age:
        digits = re.sub(r"\D", "", age)
        age = digits if digits and 17 <= int(digits) <= 100 else ""
    phone = _valid_phone(_find_value_after(lines, "Phone Number", max_lines=1))
    if not phone:
        phone = _valid_phone(_find_value_after(lines, "Phone:", max_lines=1))
    email = _valid_email(_find_value_after(lines, "E-mail", max_lines=1))
    emergency_contact = _find_value_after(lines, "Name & Phone", max_lines=1)
    if not emergency_contact:
        emergency_contact = _find_value_after(lines, "Emergency Contact:", max_lines=3)
    todays_date = _valid_date(_find_value_after(lines, "Today's Date", max_lines=1))
    preferred_city = _find_value_after(lines, "preferred city of", max_lines=2)
    # Clean up city — remove label fragments
    if preferred_city:
        preferred_city = re.sub(r"(?i)housing:?\s*", "", preferred_city).strip()
    drug_of_choice = _find_value_after(lines, "Alcohol and/or drugs", max_lines=2)
    if drug_of_choice:
        drug_of_choice = re.sub(r"(?i)of choice\s*", "", drug_of_choice).strip()
    date_of_last_use = _valid_date(_find_value_after(lines, "Date of last use", max_lines=1))
    housing_needed_by = _valid_date(_find_multiline_after(lines, "Housing needed by", "Phone"))
    medications = _find_value_after(lines, "List any medications", max_lines=2)
    if medications:
        medications = re.sub(r"(?i)you are prescribed:?\s*", "", medications).strip()
    employer = _find_value_after(lines, "Employer:", max_lines=1)
    charges = _find_multiline_after(lines, "EXPLAIN IN DETAIL ALL", "Name of Case")
    if not charges:
        charges = _find_multiline_after(lines, "List all charges", "Name of Case")
    if charges:
        charges = re.sub(r"(?i)\s*EXPLAIN IN DETAIL ALL\s*VIOLENT CHARGES:?\s*", "\n", charges).strip()
        charges = re.sub(r"(?i)^VIOLENT CHARGES:?\s*", "", charges).strip()
    case_manager = _find_value_after(lines, "Name of Case", max_lines=2)
    if case_manager:
        case_manager = re.sub(r"(?i)Manager:?\s*", "", case_manager).strip()
    probation_officer = _find_value_after(lines, "Probation/Parole", max_lines=2)
    if probation_officer:
        probation_officer = re.sub(r"(?i)Officer,?\s*if known:?\s*", "", probation_officer).strip()
    county = _find_value_after(lines, "Name of Idaho County", max_lines=1)
    if county:
        county = re.sub(r"(?i)you will be reporting to:?\s*", "", county).strip()
    signature = _find_value_after(lines, "Name)", max_lines=1)
    if not signature or len(signature) < 3:
        signature = _find_value_after(lines, "Your Signature", max_lines=3)
        if signature:
            signature = re.sub(r"(?i)^\(Type\s*Name\)?\s*", "", signature).strip()
    # Remove any leftover label fragments
    if signature:
        signature = re.sub(r"(?i)^\(Type.*?\)\s*", "", signature).strip()
    signing_date = _valid_date(_find_value_after(lines, "Date", max_lines=1))
    if not signing_date:
        signing_date = todays_date

    # Boolean fields
    previously_resided = _find_bool_after(lines, "resided at Rising Sun")
    alcoholic_addict = _find_bool_after(lines, "alcoholic/addict")
    sex_offender = _find_bool_after(lines, "sex offender?") or _find_bool_after(lines, "register as a sex offender")
    currently_employed = _find_bool_after(lines, "Are you currently")
    if not currently_employed:
        currently_employed = _find_bool_after(lines, "employed?")
    convicted = _find_bool_after(lines, "convicted of a")
    if not convicted:
        convicted = _find_bool_after(lines, "misdemeanor or")
    under_influence = _find_bool_after(lines, "influence of")
    if not under_influence:
        under_influence = _find_bool_after(lines, "drugs/alcohol when")
    on_probation = _find_bool_after(lines, "Will you be on")
    if not on_probation:
        on_probation = _find_bool_after(lines, "probation or parole while")

    # Also try to get name from the email subject line
    if not name:
        for line in lines:
            m = re.search(r"Re:\s*(.+?)\s*-\s*RISING SUN", line, re.IGNORECASE)
            if m:
                name = _clean(m.group(1))
                break

    # Fallback: use filename stem
    if not name:
        name = re.sub(r"\s+\d{1,2}-\d{1,2}-\d{2,4}$", "", pdf_path.stem).strip()

    field_results: dict[str, dict[str, Any]] = {
        "applicant.name": _text_field(name),
        "applicant.idoc_or_le_number": _text_field(idoc_or_le),
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
        "employment.currently_employed": _bool_field(currently_employed),
        "employment.employer": _text_field(employer),
        "criminal.convicted_misdemeanor_or_felony": _bool_field(convicted),
        "criminal.under_influence_during_crime": _bool_field(under_influence),
        "criminal.charges": _multiline_field(charges),
        "contacts.case_manager": _text_field(case_manager),
        "supervision.on_probation_or_parole_while_housed": _bool_field(on_probation),
        "supervision.probation_or_parole_officer": _text_field(probation_officer),
        "supervision.reporting_county": _text_field(county),
        "requirements.sex_offender_registration": _bool_field(sex_offender),
        "history.previously_resided_at_rising_sun": _bool_field(previously_resided),
        "history.alcoholic_or_addict": _bool_field(alcoholic_addict),
        "signing.signature_name": _text_field(signature),
        "signing.application_date": _text_field(signing_date or todays_date),
    }

    structured: dict[str, Any] = {}
    for field_name, field in field_results.items():
        _assign_nested(structured, field_name, field["value"])

    page_count = len(page_raw_text)
    return {
        "source_pdf": str(pdf_path),
        "template": "jotform_application_v1",
        "document_classification": "jotform_application",
        "supported_template": True,
        "classification_reason": "Digital Jotform application with structured text fields.",
        "page_count": page_count,
        "field_results": {
            key: {"page": 1 if not key.startswith(("criminal.", "supervision.", "signing.", "employment.")) else 2, **value}
            for key, value in field_results.items()
        },
        "extracted": structured,
        "page_raw_text": page_raw_text,
    }
