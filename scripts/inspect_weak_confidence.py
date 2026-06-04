#!/usr/bin/env python3
"""Plot weak confidence scores and print high/low-confidence examples."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Per-example CSV with weak_confidence and text columns.")
    parser.add_argument("--confidence-col", default="weak_confidence")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--pred-col", default="weak_pred")
    parser.add_argument("--correct-col", default="weak_correct")
    parser.add_argument("--prob-col", default="weak_prob_label1")
    parser.add_argument("--plot-output", required=True)
    parser.add_argument("--examples-output", required=True)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--n-examples", type=int, default=8)
    parser.add_argument("--wrap-width", type=int, default=110)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def optional_value(row: pd.Series, column: str) -> str:
    if column not in row.index:
        return "NA"
    value = row[column]
    if pd.isna(value):
        return "NA"
    return str(value)


def plot_confidences(df: pd.DataFrame, confidence_col: str, output: Path, bins: int) -> None:
    import matplotlib.pyplot as plt

    values = df[confidence_col].dropna()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(values, bins=bins, range=(0.0, 1.0), color="#386cb0", edgecolor="white", linewidth=0.8)
    ax.axvline(values.mean(), color="#ef3b2c", linestyle="--", linewidth=1.5, label=f"mean={values.mean():.3f}")
    ax.axvline(values.median(), color="#31a354", linestyle=":", linewidth=1.8, label=f"median={values.median():.3f}")
    ax.set_title("Weak Confidence Distribution")
    ax.set_xlabel("weak confidence")
    ax.set_ylabel("number of examples")
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def format_examples(
    title: str,
    rows: pd.DataFrame,
    args: argparse.Namespace,
) -> list[str]:
    lines = [title, "=" * len(title), ""]
    for rank, (_, row) in enumerate(rows.iterrows(), start=1):
        text = str(row[args.text_col]).replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join(textwrap.wrap(text, width=args.wrap_width, replace_whitespace=False))
        lines.extend(
            [
                f"[{rank}] id={optional_value(row, 'id')}",
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


def write_examples(df: pd.DataFrame, args: argparse.Namespace, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    high = df.sort_values(args.confidence_col, ascending=False).head(args.n_examples)
    low = df.sort_values(args.confidence_col, ascending=True).head(args.n_examples)

    lines = [
        f"CSV: {args.csv_path}",
        f"n_examples: {len(df)}",
        f"confidence_col: {args.confidence_col}",
        "",
    ]
    lines.extend(format_examples("High Confidence Examples", high, args))
    lines.extend(format_examples("Low Confidence Examples", low, args))

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)
    require_columns(df, [args.confidence_col, args.text_col])

    plot_confidences(df, args.confidence_col, Path(args.plot_output), args.bins)
    write_examples(df, args, Path(args.examples_output))

    print(f"Wrote plot to {args.plot_output}")
    print(f"Wrote examples to {args.examples_output}")


if __name__ == "__main__":
    main()
