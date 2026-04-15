from __future__ import annotations

from pathlib import Path

import yaml

from rising_sun.models import FieldSpec, TemplateConfig


def _normalize_option_name(name: object) -> str:
    if name is True:
        return "yes"
    if name is False:
        return "no"
    return str(name)


def load_template(path: Path) -> TemplateConfig:
    payload = yaml.safe_load(path.read_text())
    fields = [
        FieldSpec(
            key=item["key"],
            page=int(item["page"]),
            kind=item["kind"],
            box=tuple(item["box"]) if item.get("box") else None,
            boxes={_normalize_option_name(name): tuple(coords) for name, coords in item.get("boxes", {}).items()} or None,
            threshold=float(item.get("threshold", 0.055)),
        )
        for item in payload["fields"]
    ]
    return TemplateConfig(
        name=payload["template_name"],
        render_dpi=int(payload.get("render_dpi", 150)),
        fields=fields,
    )
