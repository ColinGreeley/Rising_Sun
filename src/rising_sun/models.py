from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NormalizedBox = tuple[float, float, float, float]
FieldKind = Literal["text", "multiline_text", "checkbox_group"]


@dataclass(frozen=True)
class FieldSpec:
    key: str
    page: int
    kind: FieldKind
    box: NormalizedBox | None = None
    boxes: dict[str, NormalizedBox] | None = None
    threshold: float = 0.055


@dataclass(frozen=True)
class TemplateConfig:
    name: str
    render_dpi: int
    fields: list[FieldSpec]
