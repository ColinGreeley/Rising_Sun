from __future__ import annotations

from difflib import SequenceMatcher
import json
from pathlib import Path

from rising_sun.identity import normalize_person_name


def _first_last_name_match(left: str, right: str) -> bool:
    left_parts = left.split()
    right_parts = right.split()
    return bool(left_parts and right_parts and left_parts[:1] == right_parts[:1] and left_parts[-1:] == right_parts[-1:])


def _levenshtein_distance(left: list[str] | str, right: list[str] | str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_value != right_value)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def train_name_ocr_model(
    dataset_dir: Path,
    output_dir: Path,
    base_model: str = "microsoft/trocr-small-handwritten",
    num_train_epochs: float = 6.0,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_target_length: int = 32,
) -> None:
    try:
        import numpy as np
        import torch
        from PIL import Image
        from torch.utils.data import Dataset
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            TrOCRProcessor,
            VisionEncoderDecoderModel,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training requires the optional handwriting dependencies. Install the project with the 'handwriting' extra."
        ) from exc

    class NameCropDataset(Dataset):
        def __init__(self, dataset_root: Path, split_name: str, processor: TrOCRProcessor, max_length: int) -> None:
            self.dataset_root = dataset_root
            self.processor = processor
            self.max_length = max_length
            split_path = dataset_root / f"{split_name}.jsonl"
            self.rows = [json.loads(line) for line in split_path.read_text().splitlines() if line.strip()]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            row = self.rows[index]
            image = Image.open(self.dataset_root / row["image"]).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
            labels = self.processor.tokenizer(
                row["text"],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            labels = labels.clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel_values, "labels": labels}

    def collate_fn(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.stack([feature["pixel_values"] for feature in features]),
            "labels": torch.stack([feature["labels"] for feature in features]),
        }

    def compute_metrics(eval_pred) -> dict[str, float]:
        predictions = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        labels = eval_pred.label_ids
        labels = np.where(labels == -100, processor.tokenizer.pad_token_id, labels)

        predicted_texts = processor.batch_decode(predictions, skip_special_tokens=True)
        label_texts = processor.batch_decode(labels, skip_special_tokens=True)

        total = max(1, len(label_texts))
        exact_matches = 0
        first_last_matches = 0
        total_token_overlap = 0
        total_similarity = 0.0
        char_distance = 0
        char_total = 0
        word_distance = 0
        word_total = 0

        for predicted_text, label_text in zip(predicted_texts, label_texts):
            predicted_normalized = normalize_person_name(predicted_text)
            label_normalized = normalize_person_name(label_text)

            if predicted_normalized == label_normalized:
                exact_matches += 1
            if _first_last_name_match(predicted_normalized, label_normalized):
                first_last_matches += 1

            total_token_overlap += len(set(predicted_normalized.split()).intersection(set(label_normalized.split())))
            total_similarity += SequenceMatcher(None, predicted_normalized, label_normalized).ratio()

            char_distance += _levenshtein_distance(predicted_normalized, label_normalized)
            char_total += max(1, len(label_normalized))

            predicted_words = predicted_normalized.split()
            label_words = label_normalized.split()
            word_distance += _levenshtein_distance(predicted_words, label_words)
            word_total += max(1, len(label_words))

        return {
            "exact_match": exact_matches / total,
            "first_last_match": first_last_matches / total,
            "avg_token_overlap": total_token_overlap / total,
            "avg_similarity": total_similarity / total,
            "cer": char_distance / char_total,
            "wer": word_distance / word_total,
        }

    processor = TrOCRProcessor.from_pretrained(base_model)
    model = VisionEncoderDecoderModel.from_pretrained(base_model)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = max_target_length
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 2
    model.generation_config.length_penalty = 1.0
    model.generation_config.num_beams = 2

    train_dataset = NameCropDataset(dataset_dir, "train", processor, max_target_length)
    eval_dataset = NameCropDataset(dataset_dir, "val", processor, max_target_length)
    test_path = dataset_dir / "test.jsonl"
    test_dataset = NameCropDataset(dataset_dir, "test", processor, max_target_length) if test_path.exists() else None

    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        use_cpu=not torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_first_last_match",
        greater_is_better=True,
        report_to="none",
        generation_max_length=max_target_length,
        generation_num_beams=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")
    metrics_path = output_dir / "metrics.json"
    metrics_payload: dict[str, dict[str, float]] = {"eval": {key: float(value) for key, value in eval_metrics.items() if isinstance(value, (int, float))}}
    if test_dataset is not None:
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        metrics_payload["test"] = {key: float(value) for key, value in test_metrics.items() if isinstance(value, (int, float))}
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))