#!/usr/bin/env python3
"""Fine-tune TrOCR on IDOC number region crops.

Unlike the name OCR model, this targets a much simpler task:
- Output vocabulary: just digits 0-9
- Output length: 5-6 characters
- Input: cropped region of a scanned form with handwritten number

Usage:
    python scripts/train_idoc_number_ocr.py \
        --dataset-dir output/idoc_number_training \
        --output-dir output/idoc_number_model_v1 \
        --epochs 15
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def train(
    dataset_dir: Path,
    output_dir: Path,
    base_model: str = "microsoft/trocr-small-handwritten",
    num_train_epochs: float = 15.0,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 4e-5,
    max_target_length: int = 12,
) -> None:
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

    class IdocNumberDataset(Dataset):
        def __init__(self, root: Path, split: str, processor: TrOCRProcessor, max_len: int) -> None:
            self.root = root
            self.processor = processor
            self.max_len = max_len
            path = root / f"{split}.jsonl"
            self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            row = self.rows[idx]
            image = Image.open(self.root / row["image"]).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
            labels = self.processor.tokenizer(
                row["text"],
                padding="max_length",
                max_length=self.max_len,
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            labels = labels.clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel_values, "labels": labels}

    def collate_fn(features: list[dict]) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }

    def compute_metrics(eval_pred) -> dict[str, float]:
        preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        labels = eval_pred.label_ids
        labels = np.where(labels == -100, processor.tokenizer.pad_token_id, labels)

        pred_texts = processor.batch_decode(preds, skip_special_tokens=True)
        label_texts = processor.batch_decode(labels, skip_special_tokens=True)

        # Strip to digits only for comparison
        def digits_only(s: str) -> str:
            return "".join(c for c in s if c.isdigit())

        total = max(1, len(label_texts))
        exact = 0
        digit_exact = 0
        char_dist = 0
        char_total = 0

        for pred, label in zip(pred_texts, label_texts):
            pred_d = digits_only(pred)
            label_d = digits_only(label)
            if pred.strip() == label.strip():
                exact += 1
            if pred_d == label_d:
                digit_exact += 1
            dist = _levenshtein(pred_d, label_d)
            char_dist += dist
            char_total += max(1, len(label_d))

        return {
            "exact_match": exact / total,
            "digit_exact_match": digit_exact / total,
            "cer": char_dist / max(1, char_total),
        }

    logger.info(f"Loading base model: {base_model}")
    processor = TrOCRProcessor.from_pretrained(base_model)
    model = VisionEncoderDecoderModel.from_pretrained(base_model)

    # Configure decoder for digit sequences
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = max_target_length
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 0  # digits can repeat
    model.generation_config.length_penalty = 1.0
    model.generation_config.num_beams = 4  # more beams for short sequences

    train_ds = IdocNumberDataset(dataset_dir, "train", processor, max_target_length)
    val_ds = IdocNumberDataset(dataset_dir, "val", processor, max_target_length)
    test_path = dataset_dir / "test.jsonl"
    test_ds = IdocNumberDataset(dataset_dir, "test", processor, max_target_length) if test_path.exists() else None

    logger.info(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds) if test_ds else 0}")

    output_dir.mkdir(parents=True, exist_ok=True)

    has_gpu = torch.cuda.is_available()
    logger.info(f"GPU available: {has_gpu}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        use_cpu=not has_gpu,
        fp16=has_gpu,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_digit_exact_match",
        greater_is_better=True,
        report_to="none",
        generation_max_length=max_target_length,
        generation_num_beams=4,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    # Evaluate
    eval_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="eval")
    logger.info(f"Eval metrics: {eval_metrics}")

    metrics_payload: dict[str, dict] = {
        "eval": {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))}
    }

    if test_ds is not None:
        test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
        logger.info(f"Test metrics: {test_metrics}")
        metrics_payload["test"] = {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}

    # Save
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info(f"Model saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IDOC number OCR model")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="microsoft/trocr-small-handwritten")
    parser.add_argument("--epochs", type=float, default=15.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
