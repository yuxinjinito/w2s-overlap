#!/usr/bin/env python3
"""Analyze correctness inside weak-confidence partitions.

This is a diagnostic for the first step of the paper's overlap detector:

    low weak confidence  -> hard-only candidate
    high weak confidence -> easy-or-overlap candidate

The detector itself does not use ground-truth labels, but this script uses the
CSV labels afterwards to measure how noisy each candidate partition is.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--confidence-col", default="weak_confidence")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--pred-col", default="weak_pred")
    parser.add_argument("--correct-col", default="weak_correct")
    parser.add_argument("--prob-col", default="weak_prob_label1")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-partition-size", type=int, default=5)
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--wrap-width", type=int, default=110)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--annotated-output", default=None)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def one_change_point_threshold(values: np.ndarray, min_partition_size: int) -> tuple[float, int, float]:
    """Find one split in sorted confidence scores by minimizing within-segment SSE."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2 * min_partition_size + 1:
        raise ValueError(
            f"Need at least {2 * min_partition_size + 1} non-NaN confidence values, got {len(values)}"
        )

    sorted_values = np.sort(values)
    prefix = np.concatenate([[0.0], np.cumsum(sorted_values)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(sorted_values * sorted_values)])
    n = len(sorted_values)

    def segment_sse(start: int, end: int) -> float:
        count = end - start
        total = prefix[end] - prefix[start]
        total_sq = prefix_sq[end] - prefix_sq[start]
        return float(total_sq - (total * total / count))

    best_k = min_partition_size
    best_sse = float("inf")
    for k in range(min_partition_size, n - min_partition_size + 1):
        sse = segment_sse(0, k) + segment_sse(k, n)
        if sse < best_sse:
            best_sse = sse
            best_k = k

    threshold_index = min(best_k, n - 1)
    return float(sorted_values[threshold_index]), int(best_k), float(best_sse)


def ensure_correctness(df: pd.DataFrame, label_col: str, pred_col: str, correct_col: str) -> pd.Series:
    if correct_col in df.columns:
        return df[correct_col].astype(int)
    require_columns(df, [label_col, pred_col])
    return (df[pred_col].astype(int) == df[label_col].astype(int)).astype(int)


def summarize_group(df: pd.DataFrame, confidence_col: str, correct_col: str) -> dict[str, float | int | None]:
    n = int(len(df))
    if n == 0:
        return {
            "n": 0,
            "correct": 0,
            "wrong": 0,
            "accuracy": None,
            "confidence_mean": None,
            "confidence_median": None,
            "confidence_min": None,
            "confidence_max": None,
        }
    correct = int(df[correct_col].sum())
    return {
        "n": n,
        "correct": correct,
        "wrong": int(n - correct),
        "accuracy": float(correct / n),
        "confidence_mean": float(df[confidence_col].mean()),
        "confidence_median": float(df[confidence_col].median()),
        "confidence_min": float(df[confidence_col].min()),
        "confidence_max": float(df[confidence_col].max()),
    }


def format_summary_row(name: str, summary: dict[str, float | int | None]) -> str:
    accuracy = summary["accuracy"]
    accuracy_str = "NA" if accuracy is None else f"{accuracy:.4f}"
    return (
        f"- {name}: n={summary['n']}, correct={summary['correct']}, wrong={summary['wrong']}, "
        f"accuracy={accuracy_str}, confidence mean/median="
        f"{format_optional(summary['confidence_mean'])}/{format_optional(summary['confidence_median'])}, "
        f"range={format_optional(summary['confidence_min'])}-{format_optional(summary['confidence_max'])}"
    )


def format_optional(value: float | int | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def optional_value(row: pd.Series, column: str) -> str:
    if column not in row.index:
        return "NA"
    value = row[column]
    if pd.isna(value):
        return "NA"
    return str(value)


def format_examples(title: str, df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    lines = [title, "=" * len(title), ""]
    if len(df) == 0:
        lines.extend(["No examples.", ""])
        return lines

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        text = optional_value(row, args.text_col)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join(textwrap.wrap(text, width=args.wrap_width, replace_whitespace=False))
        lines.extend(
            [
                f"[{rank}] id={optional_value(row, 'id')} | partition={optional_value(row, 'confidence_partition')}",
                (
                    f"confidence={float(row[args.confidence_col]):.6f} | "
                    f"prob_label1={optional_value(row, args.prob_col)} | "
                    f"label={optional_value(row, args.label_col)} | "
                    f"pred={optional_value(row, args.pred_col)} | "
                    f"correct={optional_value(row, args.correct_col)}"
                ),
                text,
                "",
            ]
        )
    return lines


def write_report(df: pd.DataFrame, args: argparse.Namespace, threshold: float, split_index: int, split_sse: float) -> None:
    hard = df[df["confidence_partition"] == "hard_only_candidate"]
    nonhard = df[df["confidence_partition"] == "easy_or_overlap_candidate"]

    high_correct = nonhard[nonhard[args.correct_col] == 1].sort_values(args.confidence_col, ascending=False)
    high_wrong = nonhard[nonhard[args.correct_col] == 0].sort_values(args.confidence_col, ascending=False)
    hard_correct = hard[hard[args.correct_col] == 1].sort_values(args.confidence_col, ascending=True)
    hard_wrong = hard[hard[args.correct_col] == 0].sort_values(args.confidence_col, ascending=True)

    lines = [
        f"CSV: {args.csv_path}",
        f"n_examples: {len(df)}",
        "",
        "Threshold",
        "=========",
        f"method: one-change-point on sorted weak_confidence, matching the paper's first detector step",
        f"threshold: {threshold:.9f}",
        f"split_index_in_sorted_confidence: {split_index}",
        f"split_sse: {split_sse:.9f}",
        "",
        "Partition Correctness",
        "=====================",
        format_summary_row("overall", summarize_group(df, args.confidence_col, args.correct_col)),
        format_summary_row("hard_only_candidate / low confidence", summarize_group(hard, args.confidence_col, args.correct_col)),
        format_summary_row(
            "easy_or_overlap_candidate / high confidence",
            summarize_group(nonhard, args.confidence_col, args.correct_col),
        ),
        "",
        "Interpretation note",
        "===================",
        "The paper's detector does not use ground-truth labels to create these partitions.",
        "Here, labels are only used afterwards to estimate how many candidate points are correct or wrong.",
        "This script separates hard-only vs non-hard only. It does not separate easy-only vs overlap,",
        "because that second step requires activation-similarity to hard-only candidates.",
        "",
    ]

    examples = [
        ("Highest-confidence correct examples in easy_or_overlap_candidate", high_correct.head(args.n_examples)),
        ("Highest-confidence wrong examples in easy_or_overlap_candidate", high_wrong.head(args.n_examples)),
        ("Lowest-confidence correct examples in hard_only_candidate", hard_correct.head(args.n_examples)),
        ("Lowest-confidence wrong examples in hard_only_candidate", hard_wrong.head(args.n_examples)),
    ]
    for title, rows in examples:
        lines.extend(format_examples(title, rows, args))

    output = Path(args.report_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv_path)
    require_columns(df, [args.confidence_col, args.label_col, args.pred_col])

    df = df.copy()
    df[args.correct_col] = ensure_correctness(df, args.label_col, args.pred_col, args.correct_col)

    if args.threshold is None:
        threshold, split_index, split_sse = one_change_point_threshold(
            df[args.confidence_col].to_numpy(),
            args.min_partition_size,
        )
    else:
        threshold = float(args.threshold)
        split_index = int((df[args.confidence_col] <= threshold).sum())
        split_sse = float("nan")

    hard_mask = df[args.confidence_col] <= threshold
    df["confidence_threshold"] = threshold
    df["confidence_partition"] = np.where(
        hard_mask,
        "hard_only_candidate",
        "easy_or_overlap_candidate",
    )

    write_report(df, args, threshold, split_index, split_sse)

    if args.annotated_output:
        output = Path(args.annotated_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        print(f"Wrote annotated CSV to {output}")
    print(f"Wrote partition report to {args.report_output}")


if __name__ == "__main__":
    main()
