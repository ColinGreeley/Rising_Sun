#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


def _fmt_value(value: float) -> str:
    return str(value).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequential RSO training sweeps and collect metrics")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=float, nargs="+", default=[20.0])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[5e-4])
    parser.add_argument("--target-pos-ratios", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    parser.add_argument("--lora-ranks", type=int, nargs="+", default=[16])
    parser.add_argument("--lora-alphas", type=int, nargs="+", default=[32])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-beams", type=int, default=1)
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | int | str]] = []
    script_path = Path(__file__).resolve().parent / "train_rso_model.py"

    for epochs, batch_size, learning_rate, target_pos_ratio, lora_rank, lora_alpha in itertools.product(
        args.epochs,
        args.batch_sizes,
        args.learning_rates,
        args.target_pos_ratios,
        args.lora_ranks,
        args.lora_alphas,
    ):
        run_name = (
            f"rso_ratio{_fmt_value(target_pos_ratio)}"
            f"_lr{_fmt_value(learning_rate)}"
            f"_bs{batch_size}"
            f"_ep{_fmt_value(epochs)}"
            f"_r{lora_rank}"
            f"_a{lora_alpha}"
        )
        output_dir = output_root / run_name
        command = [
            sys.executable,
            str(script_path),
            "--dataset-dir",
            str(args.dataset_dir),
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--lr",
            str(learning_rate),
            "--lora-r",
            str(lora_rank),
            "--lora-alpha",
            str(lora_alpha),
            "--target-pos-ratio",
            str(target_pos_ratio),
            "--num-workers",
            str(args.num_workers),
            "--eval-beams",
            str(args.eval_beams),
        ]
        if args.eval_batch_size is not None:
            command.extend(["--eval-batch-size", str(args.eval_batch_size)])

        print(f"=== Running {run_name} ===", flush=True)
        subprocess.run(command, check=True)

        metrics_path = output_dir / "metrics.json"
        payload = json.loads(metrics_path.read_text())
        eval_metrics = payload.get("eval", {})
        test_metrics = payload.get("test", {})
        row = {
            "run_name": run_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "target_pos_ratio": target_pos_ratio,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "eval_balanced_accuracy": eval_metrics.get("eval_balanced_accuracy", 0.0),
            "eval_precision": eval_metrics.get("eval_precision", 0.0),
            "eval_recall": eval_metrics.get("eval_recall", 0.0),
            "eval_specificity": eval_metrics.get("eval_specificity", 0.0),
            "test_balanced_accuracy": test_metrics.get("test_balanced_accuracy", 0.0),
            "test_precision": test_metrics.get("test_precision", 0.0),
            "test_recall": test_metrics.get("test_recall", 0.0),
            "test_specificity": test_metrics.get("test_specificity", 0.0),
        }
        summary_rows.append(row)

        summary_rows.sort(
            key=lambda item: (
                float(item["eval_balanced_accuracy"]),
                float(item["test_balanced_accuracy"]),
            ),
            reverse=True,
        )

        summary_json_path = output_root / "summary.json"
        summary_json_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n")

        summary_csv_path = output_root / "summary.csv"
        with summary_csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    if summary_rows:
        best = summary_rows[0]
        print(
            "Best run:",
            best["run_name"],
            f"eval_bal_acc={best['eval_balanced_accuracy']}",
            f"test_bal_acc={best['test_balanced_accuracy']}",
            flush=True,
        )


if __name__ == "__main__":
    main()