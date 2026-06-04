#!/usr/bin/env python3
"""Summarize an inference/confidence CSV.

This helper turns the per-example table from run_inference_confidence.py into a
compact report that is easy to paste into a mentor update.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--examples", type=int, default=5)
    return parser.parse_args()


def maybe_mean(df: pd.DataFrame, column: str):
    if column not in df or df[column].isna().all():
        return None
    return float(df[column].mean())


def confidence_summary(df: pd.DataFrame, prefix: str) -> dict:
    col = f"{prefix}_confidence"
    if col not in df:
        return {}
    series = df[col]
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "q10": float(series.quantile(0.10)),
        "q90": float(series.quantile(0.90)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def compact_examples(df: pd.DataFrame, n: int) -> list[dict]:
    cols = [
        "id",
        "label",
        "weak_prob_label1",
        "weak_confidence",
        "weak_pred",
        "weak_correct",
        "strong_prob_label1",
        "strong_confidence",
        "strong_pred",
        "strong_correct",
    ]
    existing = [col for col in cols if col in df.columns]
    rows = df.sort_values("weak_confidence", ascending=False).head(n)
    return rows[existing].to_dict(orient="records")


def main() -> None:
    args = parse_args()
    path = Path(args.csv_path)
    df = pd.read_csv(path)

    summary = {
        "path": str(path),
        "n_examples": int(len(df)),
        "dataset": None if "dataset" not in df else str(df["dataset"].iloc[0]),
        "split": None if "split" not in df else str(df["split"].iloc[0]),
        "weak_accuracy": maybe_mean(df, "weak_correct"),
        "strong_accuracy": maybe_mean(df, "strong_correct"),
        "weak_confidence": confidence_summary(df, "weak"),
        "strong_confidence": confidence_summary(df, "strong"),
    }

    if {"weak_pred", "strong_pred"}.issubset(df.columns):
        summary["weak_strong_agreement"] = float((df["weak_pred"] == df["strong_pred"]).mean())

    if {"weak_correct", "strong_correct"}.issubset(df.columns):
        summary["weak_correct_strong_wrong"] = int(((df["weak_correct"] == 1) & (df["strong_correct"] == 0)).sum())
        summary["weak_wrong_strong_correct"] = int(((df["weak_correct"] == 0) & (df["strong_correct"] == 1)).sum())
        summary["both_correct"] = int(((df["weak_correct"] == 1) & (df["strong_correct"] == 1)).sum())
        summary["both_wrong"] = int(((df["weak_correct"] == 0) & (df["strong_correct"] == 0)).sum())

    summary["top_weak_confidence_examples"] = compact_examples(df, args.examples)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
