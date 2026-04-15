from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Any


TEXT_CONFIDENCE_THRESHOLD = 0.8
HANDWRITTEN_NAME_CONFIDENCE_THRESHOLD = 0.9


def _selected_binary_option(result: dict[str, Any], field_name: str) -> str | None:
    field = result.get("field_results", {}).get(field_name, {})
    value = field.get("value") or {}
    selected_options = list(value.get("selected_options", []))
    if len(selected_options) == 1:
        return selected_options[0]
    return None


def _should_skip_blank_review(result: dict[str, Any], field_name: str) -> bool:
    if field_name in {"benefits.reinstatement_date", "benefits.previous_benefits_duration", "support.prescribed_medications", "contacts.case_manager_or_po"}:
        return True

    if field_name in {"employment.employer", "employment.employer_phone_or_email"}:
        return _selected_binary_option(result, "employment.has_employment_upon_release") != "yes"

    if field_name == "addictions.date_of_last_use":
        return _selected_binary_option(result, "addictions.has_addictions") != "yes"

    if field_name == "housing.transitional_home_name_city":
        return _selected_binary_option(result, "housing.previously_in_transitional_home") != "yes"

    if field_name == "history.violent_or_discharge_explanation":
        return (
            _selected_binary_option(result, "history.disciplined_or_discharged") != "yes"
            and _selected_binary_option(result, "history.violent_crimes_or_dor") != "yes"
        )

    return False


def _preview(value: Any, limit: int = 120) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _name_like_token_count(value: str) -> int:
    return len(re.findall(r"[A-Za-z][A-Za-z'.-]*", value))


def _is_handwritten_name_source(source: str) -> bool:
    return source.startswith("name_crop_") or source == "page_text_name_regex"


def build_field_review(result: dict[str, Any], field_name: str, field: dict[str, Any]) -> dict[str, Any] | None:
    kind = field.get("kind")
    page = field.get("page")
    confidence = float(field.get("confidence", 0.0))
    source = str(field.get("source", "") or "")
    value = field.get("value")

    if kind == "checkbox_group":
        selected_options = list((value or {}).get("selected_options", []))
        score_keys = set((value or {}).get("scores", {}).keys())
        if score_keys == {"yes", "no"} and len(selected_options) != 1:
            return {
                "needs_review": True,
                "status": "binary_checkbox_conflict" if len(selected_options) > 1 else "binary_checkbox_unresolved",
                "severity": "medium",
                "message": "Checkbox answer could not be resolved cleanly.",
                "page": page,
                "source": source,
            }
        return None

    text_value = str(value or "").strip()
    if not text_value:
        if _should_skip_blank_review(result, field_name):
            return None
        return {
            "needs_review": True,
            "status": "blank_text",
            "severity": "high",
            "message": "Field is blank and should be reviewed.",
            "page": page,
            "source": source,
        }

    if field_name == "applicant.name":
        token_count = _name_like_token_count(text_value)
        candidate_count = len(field.get("candidates", [])) if isinstance(field.get("candidates"), list) else 0
        reasons: list[str] = []
        if _is_handwritten_name_source(source):
            reasons.append("handwritten_name")
        if confidence < HANDWRITTEN_NAME_CONFIDENCE_THRESHOLD:
            reasons.append("low_confidence")
        if token_count < 2:
            reasons.append("incomplete_name")
        if reasons:
            severity = "high" if "incomplete_name" in reasons or confidence < 0.75 else "medium"
            reason_text = ", ".join(reasons).replace("_", " ")
            candidate_text = f" {candidate_count} candidates available." if candidate_count else ""
            return {
                "needs_review": True,
                "status": "manual_name_review",
                "severity": severity,
                "message": f"Applicant name should be reviewed: {reason_text}.{candidate_text}",
                "page": page,
                "source": source,
            }
        return None

    if confidence < TEXT_CONFIDENCE_THRESHOLD:
        return {
            "needs_review": True,
            "status": "low_confidence_text",
            "severity": "medium",
            "message": "Text confidence is below the review threshold.",
            "page": page,
            "source": source,
        }

    return None


def annotate_result_reviews(result: dict[str, Any]) -> dict[str, Any]:
    field_results = result.get("field_results", {})
    review_summary = {"needs_review_count": 0, "high_severity_count": 0, "fields": []}

    for field_name, field in field_results.items():
        review = build_field_review(result, field_name, field)
        if review is None:
            field.pop("review", None)
            continue
        field["review"] = review
        review_summary["needs_review_count"] += 1
        if review.get("severity") == "high":
            review_summary["high_severity_count"] += 1
        review_summary["fields"].append({
            "field": field_name,
            "status": review.get("status", ""),
            "severity": review.get("severity", ""),
            "message": review.get("message", ""),
            "page": review.get("page", ""),
        })

    result["review_summary"] = review_summary
    return result


def collect_review_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_pdf = result.get("source_pdf", "")
    if not result.get("supported_template", True):
        rows.append({
            "source_pdf": source_pdf,
            "field": "__document__",
            "page": "",
            "kind": "document",
            "status": "unsupported_document",
            "confidence": "",
            "source": result.get("document_classification", ""),
            "value_preview": result.get("classification_reason", ""),
        })
        return rows

    for field_name, field in result.get("field_results", {}).items():
        kind = field.get("kind")
        page = field.get("page")
        confidence = float(field.get("confidence", 0.0))
        source = field.get("source", "")
        value = field.get("value")

        review = build_field_review(result, field_name, field)
        if review is not None and review.get("status") not in {"blank_text", "low_confidence_text", "binary_checkbox_unresolved", "binary_checkbox_conflict"}:
            rows.append({
                "source_pdf": source_pdf,
                "field": field_name,
                "page": page,
                "kind": kind,
                "status": review.get("status", "manual_review"),
                "confidence": confidence,
                "source": source,
                "value_preview": _preview(value),
            })

        if kind == "checkbox_group":
            selected_options = list((value or {}).get("selected_options", []))
            score_keys = set((value or {}).get("scores", {}).keys())
            if score_keys == {"yes", "no"}:
                if len(selected_options) == 0:
                    rows.append({
                        "source_pdf": source_pdf,
                        "field": field_name,
                        "page": page,
                        "kind": kind,
                        "status": "binary_checkbox_unresolved",
                        "confidence": confidence,
                        "source": source,
                        "value_preview": "",
                    })
                elif len(selected_options) > 1:
                    rows.append({
                        "source_pdf": source_pdf,
                        "field": field_name,
                        "page": page,
                        "kind": kind,
                        "status": "binary_checkbox_conflict",
                        "confidence": confidence,
                        "source": source,
                        "value_preview": ", ".join(selected_options),
                    })
            continue

        text_value = str(value or "").strip()
        if not text_value:
            if _should_skip_blank_review(result, field_name):
                continue
            rows.append({
                "source_pdf": source_pdf,
                "field": field_name,
                "page": page,
                "kind": kind,
                "status": "blank_text",
                "confidence": confidence,
                "source": source,
                "value_preview": "",
            })
        elif confidence < TEXT_CONFIDENCE_THRESHOLD:
            rows.append({
                "source_pdf": source_pdf,
                "field": field_name,
                "page": page,
                "kind": kind,
                "status": "low_confidence_text",
                "confidence": confidence,
                "source": source,
                "value_preview": _preview(text_value),
            })

    return rows


def write_review_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_pdf", "field", "page", "kind", "status", "confidence", "source", "value_preview"],
        )
        writer.writeheader()
        writer.writerows(rows)
