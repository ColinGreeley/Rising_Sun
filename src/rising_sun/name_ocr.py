from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Protocol
from urllib.request import urlretrieve

from rising_sun.image_ops import crop_image, mostly_blank, prepare_text_crop
from rising_sun.models import NormalizedBox
from rising_sun.ocr import RapidOcrBackend


DEFAULT_KRAKEN_MODEL = "10.5281/zenodo.13788177"
KRAKEN_CACHE_DIR = Path.home() / ".cache" / "rising_sun" / "kraken"


@dataclass(frozen=True)
class NameOcrCandidate:
    source: str
    value: str
    confidence: float
    score: float = 0.0


class NameOcrBackend(Protocol):
    backend_name: str

    def candidate_boxes(self, box: NormalizedBox) -> dict[str, NormalizedBox]:
        ...

    def extract_crop_candidates(self, crop, variant_label: str = "tight") -> list[NameOcrCandidate]:
        ...

    def extract_candidates(self, page_image, page_text: str, box: NormalizedBox) -> list[NameOcrCandidate]:
        ...


def _clean_capture(value: str) -> str:
    value = str(value or "").strip(" _:-")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _lines(text: str) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9/]+", "", str(text or "").lower().replace("0", "o"))


def _score_name_candidate(candidate: NameOcrCandidate, duplicate_count: int, source_weights: dict[str, float]) -> float:
    lowered = candidate.value.lower()
    token_count = len(re.findall(r"[A-Za-z][A-Za-z'.-]*", candidate.value))
    alpha_count = sum(char.isalpha() for char in candidate.value)
    compact_length = max(1, len(re.sub(r"\s+", "", candidate.value)))
    alpha_ratio = alpha_count / compact_length
    has_digits = any(char.isdigit() for char in candidate.value)
    suspicious = any(marker in lowered for marker in ["age", "dob", "ssn", "idoc", "gender", "special", "accommodation"])

    score = source_weights.get(candidate.source, 1.0)
    score += min(token_count, 4) * 3.0
    score += 4.0 if token_count >= 2 else -1.0
    score += min(alpha_count, 24) * 0.18
    score += alpha_ratio * 4.0
    score += candidate.confidence * 3.0
    score += duplicate_count * 1.25
    if has_digits:
        score -= 6.0
    if suspicious:
        score -= 8.0
    if len(candidate.value) < 5:
        score -= 4.0
    return score


def _rank_name_candidates(candidates: list[NameOcrCandidate], source_weights: dict[str, float]) -> list[NameOcrCandidate]:
    if not candidates:
        return []

    candidate_counts = Counter(candidate.value for candidate in candidates)
    ranked = [
        NameOcrCandidate(
            source=candidate.source,
            value=candidate.value,
            confidence=candidate.confidence,
            score=_score_name_candidate(candidate, candidate_counts[candidate.value], source_weights),
        )
        for candidate in candidates
    ]
    ranked.sort(key=lambda item: (item.score, item.confidence, len(item.value)), reverse=True)

    unique_ranked: list[NameOcrCandidate] = []
    seen_values: set[str] = set()
    for candidate in ranked:
        if candidate.value in seen_values:
            continue
        seen_values.add(candidate.value)
        unique_ranked.append(candidate)
    return unique_ranked


class RapidEnsembleNameOcrBackend:
    backend_name = "rapid_ensemble"
    source_weights = {
        "page_text_name_regex": 8.0,
        "name_crop_wide_raw": 8.0,
        "name_crop_context_raw": 7.0,
        "name_crop_wide_prepared": 6.0,
        "name_crop_context_prepared": 5.5,
        "name_crop_tight_raw": 4.0,
    }

    def __init__(self, rapid_ocr: RapidOcrBackend, normalize_name: Callable[[str], str]) -> None:
        self.rapid_ocr = rapid_ocr
        self.normalize_name = normalize_name

    def candidate_boxes(self, box: NormalizedBox) -> dict[str, NormalizedBox]:
        left, top, right, bottom = box
        return {
            "tight": box,
            "context": (
                max(0.0, left - 0.04),
                max(0.0, top - 0.010),
                min(1.0, right + 0.04),
                min(1.0, bottom + 0.008),
            ),
            "wide": (
                max(0.0, left - 0.06),
                max(0.0, top - 0.013),
                min(1.0, right + 0.08),
                min(1.0, bottom + 0.013),
            ),
        }

    def extract_candidates(self, page_image, page_text: str, box: NormalizedBox) -> list[NameOcrCandidate]:
        candidates: list[NameOcrCandidate] = []

        for candidate_text in self._text_candidates_from_page(page_text):
            normalized = self.normalize_name(candidate_text)
            if normalized and len(normalized.split()) >= 2:
                candidates.append(NameOcrCandidate(source="page_text_name_regex", value=normalized, confidence=0.78))

        for label, crop_box in self.candidate_boxes(box).items():
            crop = crop_image(page_image, crop_box)
            if crop.size == 0 or mostly_blank(crop):
                continue
            candidates.extend(self.extract_crop_candidates(crop, label))

        return _rank_name_candidates(candidates, self.source_weights)

    def extract_crop_candidates(self, crop, variant_label: str = "tight") -> list[NameOcrCandidate]:
        if crop.size == 0 or mostly_blank(crop):
            return []

        candidates: list[NameOcrCandidate] = []
        variants = [(f"name_crop_{variant_label}_raw", crop)]
        if variant_label in {"wide", "context"}:
            variants.append((f"name_crop_{variant_label}_prepared", prepare_text_crop(crop, multiline=False)))
        for source, variant_image in variants:
            result = self.rapid_ocr.read_text(variant_image, multiline=False)
            normalized = self.normalize_name(result.text)
            if normalized:
                candidates.append(NameOcrCandidate(source=source, value=normalized, confidence=result.confidence))
        return _rank_name_candidates(candidates, self.source_weights)

    def _text_candidates_from_page(self, page_text: str) -> list[str]:
        if not page_text:
            return []

        stop_patterns = [
            r"ID[O0]C#?",
            r"Last\s*4\s*digits",
            r"DOB",
            r"D\.?O\.?B",
            r"2\.?\s*Gender",
            r"Gender",
            r"special\s+accommodations",
        ]
        stops = "|".join(stop_patterns)
        patterns = [rf"1\.?\s*Name:?\s*(.+?)\s*(?:{stops})"]

        candidates: list[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, page_text, flags=re.IGNORECASE | re.DOTALL):
                captured = _clean_capture(match.group(1))
                if captured:
                    candidates.append(captured)

        lines = _lines(page_text)
        for index, line in enumerate(lines):
            compact = _compact_text(line)
            if "1name" not in compact and not compact.startswith("name"):
                continue
            tail = re.sub(r"^.*?name:?", "", line, flags=re.IGNORECASE).strip()
            if tail:
                candidates.append(tail)
            if index + 1 < len(lines) and not re.search(r"(?:gender|dob|ssn|idoc)", lines[index + 1], flags=re.IGNORECASE):
                candidates.append(lines[index + 1])

        unique_candidates: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            cleaned = _clean_capture(candidate)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique_candidates.append(cleaned)
        return unique_candidates

class EasyOcrNameOcrBackend:
    backend_name = "easyocr"
    source_weights = {
        "easyocr_wide_raw": 8.0,
        "easyocr_context_raw": 7.0,
        "easyocr_wide_prepared": 6.0,
        "easyocr_context_prepared": 5.5,
        "easyocr_tight_raw": 4.0,
    }

    def __init__(self, normalize_name: Callable[[str], str]) -> None:
        self.normalize_name = normalize_name
        self._reader = None

    def candidate_boxes(self, box: NormalizedBox) -> dict[str, NormalizedBox]:
        return RapidEnsembleNameOcrBackend.candidate_boxes(self, box)

    def extract_candidates(self, page_image, page_text: str, box: NormalizedBox) -> list[NameOcrCandidate]:
        self._ensure_loaded()
        candidates: list[NameOcrCandidate] = []

        for label, crop_box in self.candidate_boxes(box).items():
            crop = crop_image(page_image, crop_box)
            if crop.size == 0 or mostly_blank(crop):
                continue
            candidates.extend(self.extract_crop_candidates(crop, label))

        return _rank_name_candidates(candidates, self.source_weights)

    def extract_crop_candidates(self, crop, variant_label: str = "tight") -> list[NameOcrCandidate]:
        self._ensure_loaded()
        if crop.size == 0 or mostly_blank(crop):
            return []

        import cv2

        candidates: list[NameOcrCandidate] = []
        variants = [(f"easyocr_{variant_label}_raw", crop)]
        if variant_label in {"wide", "context"}:
            variants.append((f"easyocr_{variant_label}_prepared", prepare_text_crop(crop, multiline=False)))
        for source, variant_image in variants:
            gray = cv2.cvtColor(variant_image, cv2.COLOR_BGR2GRAY) if len(variant_image.shape) == 3 else variant_image
            height, width = gray.shape[:2]
            result = self._reader.recognize(
                gray,
                horizontal_list=[[0, width, 0, height]],
                free_list=[],
                detail=1,
                paragraph=False,
                decoder="beamsearch",
                beamWidth=5,
                batch_size=1,
                workers=0,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -'",
                reformat=False,
            )
            if not result:
                continue
            ordered = sorted(result, key=lambda item: min(point[0] for point in item[0]))
            text = " ".join(str(item[1]).strip() for item in ordered if str(item[1]).strip())
            confidence_values = [float(item[2]) for item in ordered if len(item) > 2]
            normalized = self.normalize_name(text)
            if normalized:
                confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
                candidates.append(NameOcrCandidate(source=source, value=normalized, confidence=confidence))
        return _rank_name_candidates(candidates, self.source_weights)

    def _ensure_loaded(self) -> None:
        if self._reader is not None:
            return

        import easyocr

        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)


class TrOcrSmallHandwrittenBackend:
    backend_name = "trocr_small_handwritten"

    def __init__(self, normalize_name: Callable[[str], str], model_name: str = "microsoft/trocr-small-handwritten") -> None:
        self.normalize_name = normalize_name
        self.model_name = model_name
        self._processor = None
        self._model = None
        self._device = None

    def candidate_boxes(self, box: NormalizedBox) -> dict[str, NormalizedBox]:
        return RapidEnsembleNameOcrBackend.candidate_boxes(self, box)

    def extract_candidates(self, page_image, page_text: str, box: NormalizedBox) -> list[NameOcrCandidate]:
        candidates: list[NameOcrCandidate] = []

        for label, crop_box in self.candidate_boxes(box).items():
            crop = crop_image(page_image, crop_box)
            if crop.size == 0 or mostly_blank(crop):
                continue
            candidates.extend(self.extract_crop_candidates(crop, label))
        return candidates

    def extract_crop_candidates(self, crop, variant_label: str = "tight") -> list[NameOcrCandidate]:
        self._ensure_loaded()
        if crop.size == 0 or mostly_blank(crop):
            return []

        from PIL import Image
        import torch

        image = Image.fromarray(crop[:, :, ::-1]).convert("RGB")
        pixel_values = self._processor(image, return_tensors="pt").pixel_values
        if self._device is not None:
            pixel_values = pixel_values.to(self._device)
        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values)
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        normalized = self.normalize_name(text)
        if not normalized:
            return []
        return [NameOcrCandidate(source=f"trocr_{variant_label}", value=normalized, confidence=0.55)]

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        device = None
        try:
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = None

        self._processor = TrOCRProcessor.from_pretrained(self.model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self._model.generation_config.max_length = 24
        self._model.eval()
        if device is not None:
            self._model = self._model.to(device)
        self._device = device


class KrakenNameOcrBackend:
    backend_name = "kraken"
    source_weights = {
        "kraken_wide_raw": 8.0,
        "kraken_context_raw": 7.0,
        "kraken_wide_prepared": 6.0,
        "kraken_context_prepared": 5.5,
        "kraken_tight_prepared": 4.5,
        "kraken_tight_raw": 4.0,
    }

    def __init__(self, normalize_name: Callable[[str], str], model_spec: str = DEFAULT_KRAKEN_MODEL) -> None:
        self.normalize_name = normalize_name
        self.model_spec = model_spec
        self._model = None

    def candidate_boxes(self, box: NormalizedBox) -> dict[str, NormalizedBox]:
        return RapidEnsembleNameOcrBackend.candidate_boxes(self, box)

    def extract_candidates(self, page_image, page_text: str, box: NormalizedBox) -> list[NameOcrCandidate]:
        candidates: list[NameOcrCandidate] = []

        for label, crop_box in self.candidate_boxes(box).items():
            crop = crop_image(page_image, crop_box)
            if crop.size == 0 or mostly_blank(crop):
                continue
            candidates.extend(self.extract_crop_candidates(crop, label))

        return _rank_name_candidates(candidates, self.source_weights)

    def extract_crop_candidates(self, crop, variant_label: str = "tight") -> list[NameOcrCandidate]:
        self._ensure_loaded()
        if crop.size == 0 or mostly_blank(crop):
            return []

        candidates: list[NameOcrCandidate] = []
        variants = [(f"kraken_{variant_label}_raw", crop), (f"kraken_{variant_label}_prepared", prepare_text_crop(crop, multiline=False))]
        for source, variant_image in variants:
            text, confidence = self._recognize(variant_image)
            normalized = self.normalize_name(text)
            if normalized:
                candidates.append(NameOcrCandidate(source=source, value=normalized, confidence=confidence))
        return _rank_name_candidates(candidates, self.source_weights)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from kraken.lib import models

        model_path = self._resolve_model_path(self.model_spec)
        self._model = models.load_any(str(model_path), device="cpu")

    def _resolve_model_path(self, model_spec: str) -> Path:
        candidate = Path(model_spec).expanduser()
        if candidate.exists():
            return candidate.resolve()

        from kraken import repo

        record = repo.get_description(model_spec, callback=lambda *args, **kwargs: None)
        model_url = next(
            (item["url"] for item in record.distribution if item["url"].endswith(".mlmodel")),
            None,
        )
        if model_url is None:
            raise ValueError(f"Kraken model '{model_spec}' does not expose a .mlmodel artifact")

        KRAKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_path = KRAKEN_CACHE_DIR / Path(model_url).name
        if not model_path.exists():
            urlretrieve(model_url, model_path)
        return model_path

    def _recognize(self, image) -> tuple[str, float]:
        from kraken import rpred
        from kraken.containers import BaselineLine, Segmentation
        from PIL import Image
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        pil_image = Image.fromarray(gray).convert("L")
        width, height = pil_image.size
        if width == 0 or height == 0:
            return "", 0.0

        max_x = max(0, width - 1)
        max_y = max(0, height - 1)
        baseline_y = max(0, min(max_y, int(height * 0.78)))
        segmentation = Segmentation(
            type="baselines",
            imagename="name-line",
            text_direction="horizontal-lr",
            script_detection=False,
            lines=[
                BaselineLine(
                    id="0",
                    baseline=[(0, baseline_y), (max_x, baseline_y)],
                    boundary=[(0, 0), (max_x, 0), (max_x, max_y), (0, max_y)],
                )
            ],
        )
        records = list(rpred.rpred(self._model, pil_image, segmentation))
        if not records:
            return "", 0.0

        predictions: list[str] = []
        confidences: list[float] = []
        for record in records:
            prediction = str(getattr(record, "prediction", "") or "").strip()
            if prediction:
                predictions.append(prediction)
            for value in getattr(record, "confidences", []) or []:
                confidences.append(float(value))
        text = " ".join(predictions).strip()
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return text, confidence


def available_name_ocr_backends() -> list[str]:
    return ["rapid_ensemble", "easyocr", "kraken", "kraken:<model-id-or-path>", "trocr_small_handwritten", "trocr:<checkpoint-path>"]


def build_name_ocr_backend(name: str, rapid_ocr: RapidOcrBackend, normalize_name: Callable[[str], str]) -> NameOcrBackend:
    trimmed = name.strip()
    normalized = trimmed.lower()
    if normalized == "rapid_ensemble":
        return RapidEnsembleNameOcrBackend(rapid_ocr=rapid_ocr, normalize_name=normalize_name)
    if normalized == "easyocr":
        return EasyOcrNameOcrBackend(normalize_name=normalize_name)
    if normalized == "kraken":
        return KrakenNameOcrBackend(normalize_name=normalize_name)
    if normalized.startswith("kraken:"):
        model_spec = trimmed.split(":", 1)[1].strip()
        if not model_spec:
            raise ValueError("Kraken backend requires a model DOI or local model path after 'kraken:'")
        return KrakenNameOcrBackend(normalize_name=normalize_name, model_spec=model_spec)
    if normalized == "trocr_small_handwritten":
        return TrOcrSmallHandwrittenBackend(normalize_name=normalize_name)
    if normalized.startswith("trocr:"):
        checkpoint = trimmed.split(":", 1)[1].strip()
        if not checkpoint:
            raise ValueError("TrOCR backend requires a checkpoint path after 'trocr:'")
        return TrOcrSmallHandwrittenBackend(normalize_name=normalize_name, model_name=checkpoint)
    if Path(trimmed).exists():
        return TrOcrSmallHandwrittenBackend(normalize_name=normalize_name, model_name=trimmed)
    raise ValueError(f"Unknown name OCR backend: {name}")