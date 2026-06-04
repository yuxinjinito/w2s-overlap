#!/usr/bin/env python3
"""Run extra Dream residual-filtering weak-label baselines.

This script continues after `run_dream_w2s_baselines.py`. It reuses the saved
weak labels and eval examples, then trains additional LoRA baselines on residual
matched / residual middle / weak-label-balanced subsets without recomputing the
weak probe or rerunning the base/ground-truth baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from run_dream_w2s_baselines import (
    Example,
    clear_memory,
    evaluate_yes_no,
    load_residual_scores,
    make_residual_middle_subset,
    train_lora_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-output-dir", required=True)
    parser.add_argument("--residual-filter-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--runs",
        default="matched_all,matched_balanced,middle_balanced",
        help=(
            "Comma-separated extras to run. Options: matched_all, "
            "matched_balanced, middle_unbalanced, middle_balanced, "
            "random_balanced."
        ),
    )
    parser.add_argument("--residual-score-col", default="residual_l2")
    parser.add_argument("--residual-keep-middle-frac", type=float, default=0.5)
    parser.add_argument("--residual-filter-map-train", choices=["all", "train", "heldout"], default="all")
    parser.add_argument("--min-residual-filter-examples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strong-model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--strong-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-adapters", action="store_true")
    parser.add_argument("--random-control-count", type=int, default=3)
    parser.add_argument(
        "--random-control-size",
        type=int,
        default=None,
        help="Optional size for random balanced controls. Defaults to the middle_balanced size.",
    )
    return parser.parse_args()


def read_strong_train(path: Path) -> tuple[list[Example], list[int]]:
    examples: list[Example] = []
    weak_labels: list[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            examples.append(Example(id=row["id"], text=row["text"], label=int(row["label"])))
            weak_labels.append(int(row["weak_label"]))
    return examples, weak_labels


def read_eval_examples(path: Path) -> tuple[list[Example], list[dict]]:
    rows = []
    examples = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            examples.append(Example(id=row["id"], text=row["text"], label=int(row["label"])))
    return examples, rows


def weak_label_balance(
    examples: list[Example],
    labels: list[int],
    seed: int,
) -> tuple[list[Example], list[int]]:
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for idx, label in enumerate(labels):
        by_label[int(label)].append(idx)
    if not by_label[0] or not by_label[1]:
        return examples, labels

    rng = np.random.default_rng(seed)
    n = min(len(by_label[0]), len(by_label[1]))
    selected: list[int] = []
    for label in [0, 1]:
        perm = rng.permutation(by_label[label])[:n]
        selected.extend(int(i) for i in perm)
    selected = [selected[int(i)] for i in rng.permutation(len(selected))]
    return [examples[i] for i in selected], [labels[i] for i in selected]


def random_weak_label_balanced_subset(
    examples: list[Example],
    labels: list[int],
    size: int,
    seed: int,
) -> tuple[list[Example], list[int]]:
    if size <= 0:
        raise ValueError("Random balanced subset size must be positive.")
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for idx, label in enumerate(labels):
        by_label[int(label)].append(idx)
    if not by_label[0] or not by_label[1]:
        return examples, labels

    per_label = max(1, size // 2)
    per_label = min(per_label, len(by_label[0]), len(by_label[1]))
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in [0, 1]:
        selected.extend(int(i) for i in rng.choice(by_label[label], size=per_label, replace=False))
    selected = [selected[int(i)] for i in rng.permutation(len(selected))]
    return [examples[i] for i in selected], [labels[i] for i in selected]


def summarize_train_subset(examples: list[Example], labels: list[int]) -> dict:
    correct = [int(label == ex.label) for ex, label in zip(examples, labels)]
    return {
        "train_examples": len(examples),
        "true_label_mean": float(np.mean([ex.label for ex in examples])) if examples else math.nan,
        "weak_label_mean": float(np.mean(labels)) if labels else math.nan,
        "weak_label_accuracy": float(np.mean(correct)) if correct else math.nan,
    }


def write_prediction_csv(path: Path, base_rows: list[dict], columns: dict[str, list[dict]]) -> None:
    base_fields = list(base_rows[0].keys()) if base_rows else []
    fieldnames = base_fields[:]
    for name in columns:
        fieldnames.extend([f"{name}_prob_label1", f"{name}_pred", f"{name}_correct"])

    by_id = {
        name: {row["id"]: row for row in rows}
        for name, rows in columns.items()
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for base in base_rows:
            out = dict(base)
            for name, rows_by_id in by_id.items():
                row = rows_by_id.get(base["id"], {})
                out[f"{name}_prob_label1"] = row.get("prob_label1")
                out[f"{name}_pred"] = row.get("pred")
                out[f"{name}_correct"] = row.get("correct")
            writer.writerow(out)


def write_subset_csv(path: Path, subsets: dict[str, tuple[list[Example], list[int]]]) -> None:
    fieldnames = ["run_name", "id", "label", "weak_label", "weak_correct", "text"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run_name, (examples, labels) in subsets.items():
            for ex, weak_label in zip(examples, labels):
                writer.writerow(
                    {
                        "run_name": run_name,
                        "id": ex.id,
                        "label": ex.label,
                        "weak_label": int(weak_label),
                        "weak_correct": int(int(weak_label) == ex.label),
                        "text": ex.text,
                    }
                )


def matched_subset(
    examples: list[Example],
    labels: list[int],
    residual_scores: dict[str, float],
) -> tuple[list[Example], list[int]]:
    kept_examples = []
    kept_labels = []
    for ex, label in zip(examples, labels):
        if ex.id in residual_scores:
            kept_examples.append(ex)
            kept_labels.append(int(label))
    return kept_examples, kept_labels


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this script on a CUDA GPU machine.")

    baseline_dir = Path(args.baseline_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_runs = [name.strip() for name in args.runs.split(",") if name.strip()]
    allowed_runs = {"matched_all", "matched_balanced", "middle_unbalanced", "middle_balanced", "random_balanced"}
    unknown_runs = sorted(set(requested_runs) - allowed_runs)
    if unknown_runs:
        raise SystemExit(f"Unknown run(s): {', '.join(unknown_runs)}")

    strong_train, weak_labels = read_strong_train(baseline_dir / "strong_train_labels.csv")
    eval_examples, base_eval_rows = read_eval_examples(baseline_dir / "eval_predictions.csv")
    residual_scores = load_residual_scores(
        args.residual_filter_csv,
        args.residual_score_col,
        args.residual_filter_map_train,
    )

    matched_examples, matched_labels = matched_subset(strong_train, weak_labels, residual_scores)
    if len(matched_examples) < args.min_residual_filter_examples:
        raise SystemExit(
            f"Only {len(matched_examples)} examples matched residual scores, "
            f"below --min-residual-filter-examples={args.min_residual_filter_examples}."
        )

    middle_examples, middle_labels, middle_rows, middle_summary = make_residual_middle_subset(
        strong_train,
        weak_labels,
        residual_scores,
        args.residual_keep_middle_frac,
        args.min_residual_filter_examples,
    )
    matched_balanced_examples, matched_balanced_labels = weak_label_balance(matched_examples, matched_labels, args.seed)
    middle_balanced_examples, middle_balanced_labels = weak_label_balance(middle_examples, middle_labels, args.seed)
    random_control_size = args.random_control_size or len(middle_balanced_examples)

    run_subsets = {
        "matched_all": (matched_examples, matched_labels),
        "matched_balanced": (matched_balanced_examples, matched_balanced_labels),
        "middle_unbalanced": (middle_examples, middle_labels),
        "middle_balanced": (middle_balanced_examples, middle_balanced_labels),
    }
    if "random_balanced" in requested_runs:
        requested_runs = [name for name in requested_runs if name != "random_balanced"]
        for idx in range(args.random_control_count):
            run_name = f"random_balanced_{idx}"
            run_subsets[run_name] = random_weak_label_balanced_subset(
                matched_examples,
                matched_labels,
                random_control_size,
                args.seed + 1000 + idx,
            )
            requested_runs.append(run_name)

    prediction_columns = {}
    run_reports = {}
    for run_name in requested_runs:
        train_examples, train_labels = run_subsets[run_name]
        model, tokenizer, report = train_lora_model(
            args,
            train_examples,
            train_labels,
            run_name,
            output_dir,
        )
        eval_summary, eval_rows = evaluate_yes_no(
            model,
            tokenizer,
            eval_examples,
            args.strong_batch_size,
            args.device,
            args.max_length,
            f"eval {run_name}",
        )
        prediction_columns[run_name] = eval_rows
        run_reports[run_name] = {
            **report,
            "train_subset": summarize_train_subset(train_examples, train_labels),
            "eval": eval_summary,
        }
        del model
        del tokenizer
        clear_memory()

    write_prediction_csv(output_dir / "eval_predictions.csv", base_eval_rows, prediction_columns)
    write_subset_csv(
        output_dir / "train_subsets.csv",
        {run_name: run_subsets[run_name] for run_name in requested_runs},
    )
    with (output_dir / "middle_residual_filter_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "label",
                "weak_label",
                "weak_correct",
                "residual_rank",
                "residual_score",
                "kept",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(middle_rows)

    summary = {
        "baseline_output_dir": str(baseline_dir),
        "residual_filter_csv": str(args.residual_filter_csv),
        "residual_score_col": args.residual_score_col,
        "residual_keep_middle_frac": args.residual_keep_middle_frac,
        "residual_filter_map_train": args.residual_filter_map_train,
        "n_strong_train_from_baseline": len(strong_train),
        "n_residual_matched": len(matched_examples),
        "random_control_count": args.random_control_count,
        "random_control_size": random_control_size,
        "middle_filter": middle_summary,
        "available_train_subsets": {
            name: summarize_train_subset(*subset)
            for name, subset in run_subsets.items()
        },
        "runs": run_reports,
        "outputs": {
            "summary": str(output_dir / "summary.json"),
            "eval_predictions": str(output_dir / "eval_predictions.csv"),
            "train_subsets": str(output_dir / "train_subsets.csv"),
            "middle_residual_filter_rows": str(output_dir / "middle_residual_filter_rows.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
