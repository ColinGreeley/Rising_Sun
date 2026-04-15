#!/usr/bin/env python3
"""K-fold cross-validation for TrOCR v2 with LoRA + augmentations.

Splits by person (not crop) so all crops from a person are in the same fold.
For each fold, trains on k-1 folds (with 10% held-out val for early stopping)
and evaluates on the held-out fold.

Reports per-fold and aggregate metrics.

Usage:
    python scripts/kfold_cv_trocr_v2.py \
        --dataset-dir output/combined_training_v2 \
        --output-dir output/kfold_cv_v2 \
        --k 5 \
        --epochs 30 \
        --batch-size 8 \
        --lr 5e-4
"""
from __future__ import annotations

import argparse
import hashlib
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


def _person_key(image_path: str) -> str:
    """Extract person identifier from image filename."""
    fname = image_path.split("/")[-1]
    return fname.rsplit("__", 1)[0]


def _load_all_rows(dataset_dir: Path) -> list[dict]:
    """Load all rows from train/val/test JSONL files."""
    rows = []
    for split in ["train", "val", "test"]:
        path = dataset_dir / f"{split}.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _assign_folds(persons: list[str], k: int) -> dict[str, int]:
    """Deterministic fold assignment by person name hash."""
    assignment = {}
    for person in sorted(persons):
        h = int(hashlib.sha256(person.encode()).hexdigest()[:8], 16)
        assignment[person] = h % k
    return assignment


def train_fold(
    fold_idx: int,
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
    dataset_dir: Path,
    fold_dir: Path,
    base_model: str,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    learning_rate: float,
    max_target_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    patience: int,
    label_smoothing: float,
) -> dict:
    """Train and evaluate a single fold. Returns test metrics dict."""
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

    logger.info(f"=== Fold {fold_idx} ===")
    logger.info(f"  train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")

    # Write temporary JSONL files for this fold
    fold_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        (fold_dir / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )

    # Runtime augmentations
    train_augment = T.Compose([
        T.RandomRotation(degrees=3, fill=255),
        T.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05), shear=2, fill=255),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.RandomGrayscale(p=0.1),
    ])

    class FoldDataset(Dataset):
        def __init__(self, rows: list[dict], root: Path, processor: TrOCRProcessor,
                     max_len: int, augment: bool = False,
                     decoder_start_token_id: int = 2) -> None:
            self.rows = rows
            self.root = root
            self.processor = processor
            self.max_len = max_len
            self.augment = augment
            self.decoder_start_token_id = decoder_start_token_id

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
            decoder_input_ids = labels.clone()
            decoder_input_ids[decoder_input_ids == self.processor.tokenizer.pad_token_id] = -100
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

    # Load fresh model from base
    processor = TrOCRProcessor.from_pretrained(base_model)
    model = VisionEncoderDecoderModel.from_pretrained(base_model)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = max_target_length
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.length_penalty = 1.0
    model.generation_config.num_beams = 4

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "key", "value", "q_proj", "v_proj", "k_proj", "out_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dec_start = processor.tokenizer.cls_token_id
    train_ds = FoldDataset(train_rows, dataset_dir, processor, max_target_length, augment=True, decoder_start_token_id=dec_start)
    val_ds = FoldDataset(val_rows, dataset_dir, processor, max_target_length, augment=False, decoder_start_token_id=dec_start)
    test_ds = FoldDataset(test_rows, dataset_dir, processor, max_target_length, augment=False, decoder_start_token_id=dec_start)

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

    has_gpu = torch.cuda.is_available()
    checkpoint_dir = fold_dir / "checkpoints"

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_dir),
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
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    logger.info(f"Fold {fold_idx}: Starting training...")
    trainer.train()

    # Evaluate on held-out test fold
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    logger.info(f"Fold {fold_idx} test metrics: {test_metrics}")

    # Clean up GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}


def run_kfold(
    dataset_dir: Path,
    output_dir: Path,
    k: int = 5,
    base_model: str = "microsoft/trocr-small-handwritten",
    num_train_epochs: float = 30.0,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 5e-4,
    max_target_length: int = 32,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    patience: int = 5,
    label_smoothing: float = 0.1,
    val_fraction: float = 0.12,
) -> None:
    """Run k-fold cross-validation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data
    all_rows = _load_all_rows(dataset_dir)
    logger.info(f"Loaded {len(all_rows)} total rows from {dataset_dir}")

    # Group by person
    person_rows: dict[str, list[dict]] = {}
    for row in all_rows:
        person = _person_key(row["image"])
        person_rows.setdefault(person, []).append(row)

    persons = sorted(person_rows.keys())
    logger.info(f"Found {len(persons)} unique persons")

    # Assign persons to folds
    fold_assignment = _assign_folds(persons, k)
    fold_persons: dict[int, list[str]] = {i: [] for i in range(k)}
    for person, fold in fold_assignment.items():
        fold_persons[fold].append(person)

    for i in range(k):
        n_persons = len(fold_persons[i])
        n_crops = sum(len(person_rows[p]) for p in fold_persons[i])
        logger.info(f"  Fold {i}: {n_persons} persons, {n_crops} crops")

    # Run each fold
    all_fold_metrics: list[dict] = []

    for fold_idx in range(k):
        # Test = this fold's persons
        test_persons = set(fold_persons[fold_idx])
        train_val_persons = [p for p in persons if p not in test_persons]

        # Split train_val into train + val (for early stopping)
        # Deterministic: use hash to pick val persons from training pool
        val_count = max(1, int(len(train_val_persons) * val_fraction))
        # Sort by hash for deterministic split
        sorted_by_hash = sorted(
            train_val_persons,
            key=lambda p: hashlib.sha256(f"val_split_{fold_idx}_{p}".encode()).hexdigest()
        )
        val_persons = set(sorted_by_hash[:val_count])
        train_persons = set(sorted_by_hash[val_count:])

        train_rows = [r for p in train_persons for r in person_rows[p]]
        val_rows = [r for p in val_persons for r in person_rows[p]]
        test_rows = [r for p in test_persons for r in person_rows[p]]

        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_metrics = train_fold(
            fold_idx=fold_idx,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            dataset_dir=dataset_dir,
            fold_dir=fold_dir,
            base_model=base_model,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            max_target_length=max_target_length,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            patience=patience,
            label_smoothing=label_smoothing,
        )

        all_fold_metrics.append(fold_metrics)

        # Save fold result immediately (in case of crash)
        (fold_dir / "test_metrics.json").write_text(
            json.dumps(fold_metrics, indent=2) + "\n"
        )

    # Aggregate metrics
    logger.info("\n" + "=" * 60)
    logger.info("K-FOLD CROSS-VALIDATION RESULTS")
    logger.info("=" * 60)

    metric_keys = sorted({k for m in all_fold_metrics for k in m.keys()})
    summary: dict[str, dict[str, float]] = {}

    for key in metric_keys:
        values = [m[key] for m in all_fold_metrics if key in m]
        if not values:
            continue
        import statistics
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[key] = {"mean": mean, "std": std, "values": values}
        logger.info(f"  {key}: {mean:.4f} ± {std:.4f}  ({', '.join(f'{v:.4f}' for v in values)})")

    # Save summary
    results = {
        "k": k,
        "n_persons": len(persons),
        "n_crops": len(all_rows),
        "fold_sizes": {
            str(i): {
                "n_persons": len(fold_persons[i]),
                "n_crops": sum(len(person_rows[p]) for p in fold_persons[i]),
            }
            for i in range(k)
        },
        "per_fold": all_fold_metrics,
        "summary": summary,
        "config": {
            "base_model": base_model,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "label_smoothing": label_smoothing,
            "patience": patience,
            "epochs": num_train_epochs,
            "lr": learning_rate,
            "val_fraction": val_fraction,
        },
    }

    results_path = output_dir / "kfold_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    logger.info(f"\nResults saved to {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="K-fold CV for TrOCR v2")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--base-model", default="microsoft/trocr-small-handwritten")
    parser.add_argument("--epochs", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-target-length", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.12)
    args = parser.parse_args()

    run_kfold(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        k=args.k,
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
        val_fraction=args.val_fraction,
    )


if __name__ == "__main__":
    main()
