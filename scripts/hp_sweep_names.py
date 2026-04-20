#!/usr/bin/env python3
"""Fast hyperparameter sweep for TrOCR name fine-tuning.

Runs each config for only a few epochs with frequent eval to quickly
identify which settings avoid overfitting.  Results are saved to a
CSV for side-by-side comparison.

Usage:
    PYTHONPATH=src python scripts/hp_sweep_names.py \
        --dataset-dir output/combined_training_v4 \
        --output-dir output/name_hp_sweep_v1 \
        --max-epochs 5
"""
from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import logging
import sys
import time
from dataclasses import dataclass, field
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


# ---- Hyperparameter grid ----
# Each key maps to a list of values to sweep.  Combinations are generated
# as a full grid, but you can prune by setting --max-trials.
HP_GRID = {
    "lr": [5e-5, 1e-4, 3e-4],
    "lora_r": [8, 16],
    "lora_dropout": [0.15, 0.3],
    "weight_decay": [0.05, 0.1],
    "label_smoothing": [0.0],
    "batch_size": [8],
}


@dataclass
class TrialConfig:
    lr: float
    lora_r: int
    lora_dropout: float
    weight_decay: float
    label_smoothing: float
    batch_size: int
    # Fixed for all trials
    lora_alpha_ratio: int = 2  # lora_alpha = lora_r * ratio
    base_model: str = "microsoft/trocr-base-handwritten"
    max_target_length: int = 48
    eval_beams: int = 1
    num_workers: int = 4

    @property
    def lora_alpha(self) -> int:
        return self.lora_r * self.lora_alpha_ratio

    @property
    def tag(self) -> str:
        return (
            f"lr{self.lr:.0e}_r{self.lora_r}_do{self.lora_dropout}"
            f"_wd{self.weight_decay}_ls{self.label_smoothing}_bs{self.batch_size}"
        )


def generate_trials() -> list[TrialConfig]:
    keys = list(HP_GRID.keys())
    combos = list(itertools.product(*(HP_GRID[k] for k in keys)))
    return [TrialConfig(**dict(zip(keys, vals))) for vals in combos]


def run_trial(
    cfg: TrialConfig,
    dataset_dir: Path,
    output_dir: Path,
    max_epochs: int,
    eval_steps: int,
) -> dict:
    """Run a single short training trial, return summary metrics."""
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

    trial_dir = output_dir / cfg.tag
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Augmentations (same as main training script)
    train_augment = T.Compose([
        T.RandomRotation(degrees=3, fill=255),
        T.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05), shear=2, fill=255),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.RandomGrayscale(p=0.1),
    ])

    class NameDataset(Dataset):
        def __init__(self, root, split, processor, max_len, augment=False, decoder_start_token_id=2):
            self.root = root
            self.processor = processor
            self.max_len = max_len
            self.augment = augment
            self.decoder_start_token_id = decoder_start_token_id
            path = root / f"{split}.jsonl"
            self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            self.rows = [r for r in self.rows if r.get("task") == "name"]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows[idx]
            image = Image.open(self.root / row["image"]).convert("RGB")
            if self.augment:
                image = train_augment(image)
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
            labels = self.processor.tokenizer(
                row["text"], padding="max_length", max_length=self.max_len,
                truncation=True, return_tensors="pt",
            ).input_ids.squeeze(0)
            labels = labels.clone()
            decoder_input_ids = labels.clone()
            shifted = torch.full_like(decoder_input_ids, self.processor.tokenizer.pad_token_id)
            shifted[0] = self.decoder_start_token_id
            shifted[1:] = labels[:-1]
            shifted[shifted == -100] = self.processor.tokenizer.pad_token_id
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel_values, "labels": labels, "decoder_input_ids": shifted}

    def collate_fn(features):
        return {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
            "decoder_input_ids": torch.stack([f["decoder_input_ids"] for f in features]),
        }

    # ---- Model setup ----
    processor = TrOCRProcessor.from_pretrained(cfg.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(cfg.base_model)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = cfg.max_target_length
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.length_penalty = 1.0
    model.generation_config.num_beams = 4

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["query", "key", "value", "q_proj", "v_proj", "k_proj", "out_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    dec_start = processor.tokenizer.cls_token_id
    train_ds = NameDataset(dataset_dir, "train", processor, cfg.max_target_length, augment=True, decoder_start_token_id=dec_start)
    val_ds = NameDataset(dataset_dir, "val", processor, cfg.max_target_length, augment=False, decoder_start_token_id=dec_start)

    logger.info(f"Trial {cfg.tag}: train={len(train_ds)}, val={len(val_ds)}, trainable={trainable:,}/{total_params:,}")

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        labels_arr = eval_pred.label_ids
        labels_arr = np.where(labels_arr == -100, processor.tokenizer.pad_token_id, labels_arr)
        pred_texts = processor.batch_decode(preds, skip_special_tokens=True)
        label_texts = processor.batch_decode(labels_arr, skip_special_tokens=True)
        total = max(1, len(label_texts))
        exact = sum(1 for p, l in zip(pred_texts, label_texts) if p.strip().lower() == l.strip().lower())
        char_dist = sum(_levenshtein(p.strip().lower(), l.strip().lower()) for p, l in zip(pred_texts, label_texts))
        char_total = sum(max(1, len(l.strip())) for l in label_texts)
        return {
            "name_accuracy": exact / total,
            "cer": char_dist / max(1, char_total),
        }

    has_gpu = torch.cuda.is_available()
    if has_gpu:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    use_bf16 = has_gpu and torch.cuda.is_bf16_supported()
    use_fp16 = has_gpu and not use_bf16

    steps_per_epoch = len(train_ds) // cfg.batch_size
    actual_eval_steps = min(eval_steps, steps_per_epoch)  # At least once per epoch

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(trial_dir),
        do_train=True,
        do_eval=True,
        predict_with_generate=True,
        eval_strategy="steps",
        eval_steps=actual_eval_steps,
        save_strategy="steps",
        save_steps=actual_eval_steps,
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=max(cfg.batch_size * 2, 16),
        num_train_epochs=max_epochs,
        warmup_ratio=0.1,
        weight_decay=cfg.weight_decay,
        label_smoothing_factor=cfg.label_smoothing,
        lr_scheduler_type="cosine",
        use_cpu=not has_gpu,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=cfg.num_workers,
        dataloader_pin_memory=has_gpu,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_name_accuracy",
        greater_is_better=True,
        report_to="none",
        generation_max_length=cfg.max_target_length,
        generation_num_beams=cfg.eval_beams,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        processing_class=processor,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # Collect eval trajectory from trainer state
    eval_entries = [e for e in trainer.state.log_history if "eval_loss" in e]
    train_entries = [e for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]

    best_eval = min(eval_entries, key=lambda e: e.get("eval_loss", 999)) if eval_entries else {}
    last_eval = eval_entries[-1] if eval_entries else {}
    first_eval = eval_entries[0] if eval_entries else {}
    last_train = train_entries[-1] if train_entries else {}

    result = {
        "tag": cfg.tag,
        "lr": cfg.lr,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "weight_decay": cfg.weight_decay,
        "label_smoothing": cfg.label_smoothing,
        "batch_size": cfg.batch_size,
        "trainable_params": trainable,
        "epochs_completed": last_eval.get("epoch", 0),
        "elapsed_sec": round(elapsed),
        # Best eval
        "best_eval_loss": best_eval.get("eval_loss"),
        "best_name_acc": best_eval.get("eval_name_accuracy"),
        "best_cer": best_eval.get("eval_cer"),
        "best_epoch": best_eval.get("epoch"),
        # Last eval
        "last_eval_loss": last_eval.get("eval_loss"),
        "last_name_acc": last_eval.get("eval_name_accuracy"),
        "last_cer": last_eval.get("eval_cer"),
        # First eval (to see starting point)
        "first_eval_loss": first_eval.get("eval_loss"),
        "first_name_acc": first_eval.get("eval_name_accuracy"),
        # Train-eval gap (overfitting signal)
        "last_train_loss": last_train.get("loss"),
        "train_eval_gap": (last_eval.get("eval_loss", 0) or 0) - (last_train.get("loss", 0) or 0),
        # Trend: is eval loss going up? (last - best)
        "eval_loss_trend": (last_eval.get("eval_loss", 0) or 0) - (best_eval.get("eval_loss", 0) or 0),
    }

    # Save detailed log per trial
    detail_path = trial_dir / "eval_trajectory.json"
    detail_path.write_text(json.dumps(eval_entries, indent=2))

    # Cleanup GPU memory
    del trainer, model, train_ds, val_ds
    gc.collect()
    if has_gpu:
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Fast HP sweep for TrOCR name OCR")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=5, help="Epochs per trial (default 5)")
    parser.add_argument("--eval-steps", type=int, default=500, help="Eval every N steps")
    parser.add_argument("--max-trials", type=int, default=0, help="Limit total trials (0=all)")
    parser.add_argument("--resume", action="store_true", help="Skip trials with existing results")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = args.output_dir / "sweep_results.csv"

    trials = generate_trials()
    logger.info(f"Generated {len(trials)} trial configs from grid")

    if args.max_trials > 0:
        trials = trials[:args.max_trials]
        logger.info(f"Limiting to {args.max_trials} trials")

    # Load existing results for --resume
    existing_tags = set()
    existing_results = []
    if args.resume and results_csv.exists():
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_tags.add(row["tag"])
                existing_results.append(row)
        logger.info(f"Resuming: {len(existing_tags)} trials already completed")

    all_results = list(existing_results)

    for i, cfg in enumerate(trials):
        if cfg.tag in existing_tags:
            logger.info(f"[{i+1}/{len(trials)}] SKIP {cfg.tag} (already done)")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(trials)}] Starting trial: {cfg.tag}")
        logger.info(f"{'='*60}")

        try:
            result = run_trial(cfg, args.dataset_dir, args.output_dir, args.max_epochs, args.eval_steps)
            all_results.append(result)
            logger.info(
                f"Trial {cfg.tag} done: best_eval_loss={result['best_eval_loss']:.4f} "
                f"best_name_acc={result['best_name_acc']:.4f} "
                f"gap={result['train_eval_gap']:.2f}"
            )
        except Exception as e:
            logger.error(f"Trial {cfg.tag} failed: {e}")
            all_results.append({"tag": cfg.tag, "error": str(e)})

        # Write CSV after every trial (crash-safe)
        if all_results:
            fieldnames = list(all_results[0].keys())
            # Ensure all keys present
            for r in all_results:
                for k in r:
                    if k not in fieldnames:
                        fieldnames.append(k)
            with open(results_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Sweep complete. Results saved to {results_csv}")
    # Sort by best eval loss
    valid = [r for r in all_results if r.get("best_eval_loss") is not None]
    if valid:
        valid.sort(key=lambda r: float(r["best_eval_loss"]))
        logger.info("\nTop 5 by best eval loss:")
        for r in valid[:5]:
            logger.info(
                f"  {r['tag']}: eval_loss={float(r['best_eval_loss']):.4f} "
                f"name_acc={float(r['best_name_acc']):.4f} "
                f"gap={float(r.get('train_eval_gap', 0)):.2f}"
            )


if __name__ == "__main__":
    main()
