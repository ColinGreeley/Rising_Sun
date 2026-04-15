#!/usr/bin/env python3
"""Fine-tune TrOCR v2 with LoRA + runtime augmentations + early stopping.

Multi-task model that reads both IDOC numbers and applicant names from form crops.

Key improvements over v1:
- LoRA (Low-Rank Adaptation): freezes pretrained weights, trains small adapters
  → prevents memorization/overfitting
- Runtime image augmentations: rotation, noise, contrast jitter, elastic distortion
  → each epoch sees different versions of same crops
- Early stopping on validation loss (not training metric)
  → stops before generalization degrades
- Label smoothing: prevents overconfident predictions

Usage:
    python scripts/train_trocr_v2.py \
        --dataset-dir output/combined_training_v2 \
        --output-dir output/trocr_model_v2 \
        --epochs 30 \
        --batch-size 8 \
        --lr 5e-4
"""
from __future__ import annotations

import argparse
import json
import logging
import random
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
    num_train_epochs: float = 30.0,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 5e-4,
    max_target_length: int = 32,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    patience: int = 5,
    label_smoothing: float = 0.1,
) -> None:
    import numpy as np
    import torch
    import torchvision.transforms.v2 as T
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    # ----- Runtime augmentations (training only) -----
    train_augment = T.Compose([
        T.RandomRotation(degrees=3, fill=255),
        T.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05), shear=2, fill=255),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.RandomGrayscale(p=0.1),
    ])

    class MultiTaskDataset(Dataset):
        def __init__(self, root: Path, split: str, processor: TrOCRProcessor,
                     max_len: int, augment: bool = False,
                     decoder_start_token_id: int = 2) -> None:
            self.root = root
            self.processor = processor
            self.max_len = max_len
            self.augment = augment
            self.decoder_start_token_id = decoder_start_token_id
            path = root / f"{split}.jsonl"
            self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            row = self.rows[idx]
            image = Image.open(self.root / row["image"]).convert("RGB")

            if self.augment:
                image = train_augment(image)

            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
            labels = self.processor.tokenizer(
                row["text"],
                padding="max_length",
                max_length=self.max_len,
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            labels = labels.clone()
            # Build decoder_input_ids by shifting labels right (prepend decoder_start, replace -100 with pad)
            # We must provide these explicitly because HF Trainer strips labels from inputs
            # when label_smoothing > 0, and VisionEncoderDecoderModel needs labels to auto-create them.
            decoder_input_ids = labels.clone()
            decoder_input_ids[decoder_input_ids == self.processor.tokenizer.pad_token_id] = -100
            # Shift right: prepend decoder_start_token_id, drop last token
            shifted = torch.full_like(decoder_input_ids, self.processor.tokenizer.pad_token_id)
            shifted[0] = self.decoder_start_token_id
            shifted[1:] = labels[:-1]
            shifted[shifted == -100] = self.processor.tokenizer.pad_token_id
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel_values, "labels": labels, "decoder_input_ids": shifted}

    def collate_fn(features: list[dict]) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
            "decoder_input_ids": torch.stack([f["decoder_input_ids"] for f in features]),
        }

    def compute_metrics(eval_pred) -> dict[str, float]:
        preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        labels = eval_pred.label_ids
        labels = np.where(labels == -100, processor.tokenizer.pad_token_id, labels)

        pred_texts = processor.batch_decode(preds, skip_special_tokens=True)
        label_texts = processor.batch_decode(labels, skip_special_tokens=True)

        def digits_only(s: str) -> str:
            return "".join(c for c in s if c.isdigit())

        total = max(1, len(label_texts))
        exact = 0
        digit_exact = 0
        name_exact = 0
        char_dist = 0
        char_total = 0
        n_number = 0
        n_name = 0

        for pred, label in zip(pred_texts, label_texts):
            pred_s = pred.strip()
            label_s = label.strip()

            # Determine task by label content
            is_number = label_s.isdigit()

            if pred_s == label_s:
                exact += 1

            if is_number:
                n_number += 1
                pred_d = digits_only(pred_s)
                label_d = digits_only(label_s)
                if pred_d == label_d:
                    digit_exact += 1
                dist = _levenshtein(pred_d, label_d)
                char_dist += dist
                char_total += max(1, len(label_d))
            else:
                n_name += 1
                if pred_s.lower() == label_s.lower():
                    name_exact += 1
                dist = _levenshtein(pred_s.lower(), label_s.lower())
                char_dist += dist
                char_total += max(1, len(label_s))

        return {
            "exact_match": exact / total,
            "digit_exact_match": digit_exact / max(1, n_number) if n_number else 0.0,
            "name_exact_match": name_exact / max(1, n_name) if n_name else 0.0,
            "cer": char_dist / max(1, char_total),
        }

    # ----- Model setup -----
    logger.info(f"Loading base model: {base_model}")
    processor = TrOCRProcessor.from_pretrained(base_model)
    model = VisionEncoderDecoderModel.from_pretrained(base_model)

    # Configure for text generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = max_target_length
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.length_penalty = 1.0
    model.generation_config.num_beams = 4

    # ----- Apply LoRA -----
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "key", "value", "q_proj", "v_proj", "k_proj", "out_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ----- Datasets -----
    dec_start = processor.tokenizer.cls_token_id
    train_ds = MultiTaskDataset(dataset_dir, "train", processor, max_target_length, augment=True, decoder_start_token_id=dec_start)
    val_ds = MultiTaskDataset(dataset_dir, "val", processor, max_target_length, augment=False, decoder_start_token_id=dec_start)
    test_path = dataset_dir / "test.jsonl"
    test_ds = MultiTaskDataset(dataset_dir, "test", processor, max_target_length, augment=False, decoder_start_token_id=dec_start) if test_path.exists() else None

    logger.info(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds) if test_ds else 0}")

    output_dir.mkdir(parents=True, exist_ok=True)

    has_gpu = torch.cuda.is_available()
    logger.info(f"GPU available: {has_gpu}")

    # ----- Training arguments -----
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        label_smoothing_factor=label_smoothing,
        lr_scheduler_type="cosine",
        use_cpu=not has_gpu,
        fp16=has_gpu,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,  # lower val loss = better
        report_to="none",
        generation_max_length=max_target_length,
        generation_num_beams=4,
    )

    # ----- Trainer with early stopping -----
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        processing_class=processor,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    logger.info(f"Starting training (max {num_train_epochs} epochs, patience={patience})...")
    trainer.train()

    # ----- Evaluation -----
    eval_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="eval")
    logger.info(f"Eval metrics: {eval_metrics}")

    metrics_payload: dict[str, dict] = {
        "eval": {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        "config": {
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "label_smoothing": label_smoothing,
            "patience": patience,
            "lr_scheduler": "cosine",
            "augmentations": "rotation+affine+colorjitter+blur+grayscale",
        },
    }

    if test_ds is not None:
        test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
        logger.info(f"Test metrics: {test_metrics}")
        metrics_payload["test"] = {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}

    # ----- Save -----
    # Merge LoRA weights back into the base model for easy inference
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n")
    logger.info(f"Merged model saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TrOCR v2 with LoRA + augmentations")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="microsoft/trocr-small-handwritten")
    parser.add_argument("--epochs", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-target-length", type=int, default=32)
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_target_length=args.max_target_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
    )


if __name__ == "__main__":
    main()
