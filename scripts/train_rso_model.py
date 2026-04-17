#!/usr/bin/env python3
"""Train a TrOCR-based RSO checkbox classifier.

Fine-tunes TrOCR (same architecture as the IDOC# / name model) to predict
"yes" or "no" from RSO checkbox crop images.

The model is small (outputs only 1 token) so training is fast.  Uses LoRA
for parameter-efficient fine-tuning with augmentations and early stopping.

Usage:
    python scripts/train_rso_model.py \
        --dataset-dir output/rso_training_v1 \
        --output-dir output/rso_model_v1 \
        --epochs 20 \
        --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train(
    dataset_dir: Path,
    output_dir: Path,
    base_model: str = "microsoft/trocr-small-handwritten",
    num_train_epochs: float = 20.0,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    learning_rate: float = 5e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    patience: int = 5,
    label_smoothing: float = 0.1,
    dataloader_num_workers: int = 4,
    eval_num_beams: int = 1,
    target_pos_ratio: float = 0.3,
) -> None:
    import numpy as np
    import torch
    import torchvision.transforms.v2 as T
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        TrainerCallback,
        Seq2SeqTrainingArguments,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    class EvalSummaryCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                return control
            summary = []
            if "eval_loss" in metrics:
                summary.append(f"val_loss={metrics['eval_loss']:.4f}")
            if "eval_accuracy" in metrics:
                summary.append(f"val_accuracy={metrics['eval_accuracy']:.4f}")
            if "eval_precision" in metrics:
                summary.append(f"val_precision={metrics['eval_precision']:.4f}")
            if "eval_recall" in metrics:
                summary.append(f"val_recall={metrics['eval_recall']:.4f}")
            if "eval_f1" in metrics:
                summary.append(f"val_f1={metrics['eval_f1']:.4f}")
            if "eval_specificity" in metrics:
                summary.append(f"val_specificity={metrics['eval_specificity']:.4f}")
            if "eval_balanced_accuracy" in metrics:
                summary.append(f"val_bal_acc={metrics['eval_balanced_accuracy']:.4f}")
            if summary:
                logger.info("Validation summary: %s", ", ".join(summary))
            return control

    class BalancedSeq2SeqTrainer(Seq2SeqTrainer):
        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset")

            rows = getattr(self.train_dataset, "rows", None)
            if rows is None:
                return super().get_train_dataloader()

            labels = [1 if r.get("text") == "yes" else 0 for r in rows]
            n_pos = sum(labels)
            n_neg = max(0, len(labels) - n_pos)
            if n_pos == 0 or n_neg == 0:
                return super().get_train_dataloader()

            # Reweight to a configurable expected positive/negative frequency.
            pos_ratio = min(0.9, max(0.1, target_pos_ratio))
            w_pos = pos_ratio / n_pos
            w_neg = (1.0 - pos_ratio) / n_neg
            sample_weights = [w_pos if y == 1 else w_neg for y in labels]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            logger.info(
                "Using weighted sampler for RSO training (target yes/no ~= %.2f/%.2f per epoch)",
                pos_ratio,
                1.0 - pos_ratio,
            )

            return DataLoader(
                self.train_dataset,
                batch_size=self._train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=self.args.dataloader_num_workers > 0,
            )

    max_target_length = 4  # "yes" or "no" — very short

    # ----- Augmentations -----
    train_augment = T.Compose([
        T.RandomRotation(degrees=3, fill=255),
        T.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05), shear=2, fill=255),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.RandomGrayscale(p=0.1),
    ])

    class RSODataset(Dataset):
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

    def compute_metrics(eval_pred) -> dict[str, float]:
        preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        labels = eval_pred.label_ids
        labels = np.where(labels == -100, processor.tokenizer.pad_token_id, labels)

        pred_texts = processor.batch_decode(preds, skip_special_tokens=True)
        label_texts = processor.batch_decode(labels, skip_special_tokens=True)

        total = max(1, len(label_texts))
        exact = 0
        tp = fp = fn = tn = 0

        for pred, label in zip(pred_texts, label_texts):
            pred_s = pred.strip().lower()
            label_s = label.strip().lower()

            if pred_s == label_s:
                exact += 1

            pred_yes = pred_s == "yes"
            label_yes = label_s == "yes"

            if pred_yes and label_yes:
                tp += 1
            elif pred_yes and not label_yes:
                fp += 1
            elif not pred_yes and label_yes:
                fn += 1
            else:
                tn += 1

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        balanced_accuracy = 0.5 * (recall + specificity)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)

        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0.0

        return {
            "accuracy": exact / total,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
            "f1": f1,
            "mcc": mcc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    # ----- Model setup -----
    logger.info(f"Loading base model: {base_model}")
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

    # ----- LoRA -----
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
    train_ds = RSODataset(dataset_dir, "train", processor, max_target_length, augment=True, decoder_start_token_id=dec_start)
    val_ds = RSODataset(dataset_dir, "val", processor, max_target_length, augment=False, decoder_start_token_id=dec_start)
    test_path = dataset_dir / "test.jsonl"
    test_ds = RSODataset(dataset_dir, "test", processor, max_target_length, augment=False, decoder_start_token_id=dec_start) if test_path.exists() else None

    logger.info(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds) if test_ds else 0}")

    n_yes_train = sum(1 for r in train_ds.rows if r["text"] == "yes")
    n_no_train = sum(1 for r in train_ds.rows if r["text"] == "no")
    logger.info(f"Train class distribution: yes={n_yes_train}, no={n_no_train}")

    output_dir.mkdir(parents=True, exist_ok=True)

    has_gpu = torch.cuda.is_available()
    logger.info(f"GPU available: {has_gpu}")

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
    logger.info(
        "Training config: train_batch=%s eval_batch=%s workers=%s eval_beams=%s precision=%s",
        per_device_train_batch_size,
        per_device_eval_batch_size,
        dataloader_num_workers,
        eval_num_beams,
        "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32",
    )

    # ----- Training arguments -----
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        label_smoothing_factor=label_smoothing,
        lr_scheduler_type="cosine",
        use_cpu=not has_gpu,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=has_gpu,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_balanced_accuracy",
        greater_is_better=True,
        report_to="none",
        generation_max_length=max_target_length,
        generation_num_beams=eval_num_beams,
    )

    # ----- Trainer -----
    trainer = BalancedSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        processing_class=processor,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=patience),
            EvalSummaryCallback(),
        ],
    )

    logger.info(f"Starting RSO training (max {num_train_epochs} epochs, patience={patience})...")
    trainer.train()

    # Disable early-stopping callback for manual post-train eval/test passes.
    trainer.remove_callback(EarlyStoppingCallback)

    # ----- Evaluation -----
    eval_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="eval")
    logger.info(f"Eval metrics: {eval_metrics}")

    metrics_payload: dict[str, dict] = {
        "eval": {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        "config": {
            "task": "rso_checkbox",
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "label_smoothing": label_smoothing,
            "patience": patience,
            "target_pos_ratio": target_pos_ratio,
        },
    }

    if test_ds is not None:
        test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
        logger.info(f"Test metrics: {test_metrics}")
        metrics_payload["test"] = {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}

    # ----- Save -----
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n")
    logger.info(f"RSO model saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RSO checkbox classifier (TrOCR + LoRA)")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="microsoft/trocr-small-handwritten")
    parser.add_argument("--epochs", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-beams", type=int, default=1)
    parser.add_argument("--target-pos-ratio", type=float, default=0.3)
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        dataloader_num_workers=args.num_workers,
        eval_num_beams=args.eval_beams,
        target_pos_ratio=args.target_pos_ratio,
    )


if __name__ == "__main__":
    main()
