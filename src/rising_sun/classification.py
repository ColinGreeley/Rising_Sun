from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentClassification:
    name: str
    supported: bool
    reason: str


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def classify_document(page_raw_text: dict[str, str], page_count: int) -> DocumentClassification:
    first_page = page_raw_text.get("1", "")
    compact = _compact(first_page)

    if "idochousingapplication" in compact and ("cmpocontactemail" in compact or "currentstatuspleasecheckone" in compact):
        return DocumentClassification(
            name="idoc_housing_application_v1",
            supported=True,
            reason="Matched IDOC housing application title and first-page prompts.",
        )

    if "jotform" in compact and "applicationforhousing" in compact:
        return DocumentClassification(
            name="jotform_application",
            supported=True,
            reason="Matched a digital Jotform application with structured text fields.",
        )

    if "risingsunsoberliving" in compact or "applicationforhousing" in compact:
        return DocumentClassification(
            name="rising_sun_application_packet",
            supported=True,
            reason="Matched the Rising Sun housing application packet.",
        )

    # For multi-page bundles, scan all pages for a match
    if page_count > 4:
        for page_num in range(2, page_count + 1):
            page_text = page_raw_text.get(str(page_num), "")
            page_compact = _compact(page_text)
            if "idochousingapplication" in page_compact and ("cmpocontactemail" in page_compact or "currentstatuspleasecheckone" in page_compact):
                return DocumentClassification(
                    name="idoc_housing_application_v1",
                    supported=True,
                    reason=f"Matched IDOC housing application on page {page_num} of {page_count}-page bundle.",
                )
            if "risingsunsoberliving" in page_compact or "applicationforhousing" in page_compact:
                return DocumentClassification(
                    name="rising_sun_application_packet",
                    supported=True,
                    reason=f"Matched Rising Sun packet on page {page_num} of {page_count}-page bundle.",
                )

    return DocumentClassification(
        name="unknown_document_type",
        supported=False,
        reason=f"No supported template fingerprint matched across {page_count} rendered pages.",
    )
