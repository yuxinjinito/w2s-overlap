#!/usr/bin/env python3
"""Compare weak-to-strong mapping losses with weak confidence/correctness.

This script is the follow-up analysis for `run_representation_mapping.py`.
It merges the representation-mapping output with the Changho-style weak-probe
confidence CSV, then reports whether mapping losses are related to weak
confidence or weak correctness.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mapping_csv", help="CSV produced by run_representation_mapping.py")
    parser.add_argument(
        "--confidence-csv",
        default=None,
        help="CSV with weak_confidence, weak_pred, weak_correct, usually from run_changho_style_probe.py",
    )
    parser.add_argument("--merge-on", choices=["id", "text"], default="id")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--confidence-col", default="weak_confidence")
    parser.add_argument("--correct-col", default="weak_correct")
    parser.add_argument("--pred-col", default="weak_pred")
    parser.add_argument("--prob-col", default="weak_prob_label1")
    parser.add_argument("--primary-loss", default="linear_l2")
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--wrap-width", type=int, default=110)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--merged-output", default=None)
    parser.add_argument("--plot-output", default=None, help="Optional PNG scatter plot output.")
    parser.add_argument(
        "--plot-heldout-only",
        action="store_true",
        help="When map_train exists, plot only map_train=0 examples.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def read_csv_with_string_id(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    return df


def metric_columns(df: pd.DataFrame) -> list[str]:
    suffixes = ("_l2", "_mse", "_cosine")
    return [col for col in df.columns if col.endswith(suffixes)]


def merge_tables(mapping_df: pd.DataFrame, conf_df: pd.DataFrame | None, merge_on: str) -> pd.DataFrame:
    if conf_df is None:
        return mapping_df.copy()

    require_columns(mapping_df, [merge_on], "mapping CSV")
    require_columns(conf_df, [merge_on], "confidence CSV")

    keep_cols = [
        merge_on,
        "weak_confidence",
        "weak_prob_label1",
        "weak_pred",
        "weak_correct",
        "prediction_source",
    ]
    if merge_on != "text" and "text" in conf_df.columns:
        keep_cols.append("text")
    keep_cols = [col for col in keep_cols if col in conf_df.columns]

    merged = mapping_df.merge(
        conf_df[keep_cols],
        on=merge_on,
        how="left",
        suffixes=("", "_conf"),
        validate="one_to_one",
    )

    if "text_conf" in merged.columns and "text" in merged.columns:
        merged["text_matches_confidence_csv"] = merged["text"] == merged["text_conf"]
    return merged


def summarize_series(series: pd.Series) -> dict[str, float | int | None]:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "q10": None,
            "q90": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "q10": float(series.quantile(0.10)),
        "q90": float(series.quantile(0.90)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def summarize_by_group(df: pd.DataFrame, group_col: str, metrics: list[str]) -> dict[str, dict[str, dict]]:
    if group_col not in df.columns:
        return {}
    out: dict[str, dict[str, dict]] = {}
    for value, group in df.groupby(group_col, dropna=False, observed=False):
        key = "NA" if pd.isna(value) else str(value)
        out[key] = {metric: summarize_series(group[metric]) for metric in metrics}
    return out


def summarize_by_two_groups(
    df: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
) -> dict[str, dict[str, dict]]:
    if any(col not in df.columns for col in group_cols):
        return {}
    out: dict[str, dict[str, dict]] = {}
    for values, group in df.groupby(group_cols, dropna=False, observed=False):
        if not isinstance(values, tuple):
            values = (values,)
        key = ", ".join(f"{col}={value}" for col, value in zip(group_cols, values))
        out[key] = {metric: summarize_series(group[metric]) for metric in metrics}
    return out


def correlations(df: pd.DataFrame, x_cols: list[str], y_cols: list[str]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for x_col in x_cols:
        out[x_col] = {}
        for y_col in y_cols:
            if x_col not in df.columns or y_col not in df.columns:
                continue
            pair = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 2 or pair[x_col].nunique() < 2 or pair[y_col].nunique() < 2:
                corr = None
            else:
                corr = float(pair[x_col].corr(pair[y_col]))
            out[x_col][y_col] = corr
    return out


def correlations_by_group(
    df: pd.DataFrame,
    group_col: str,
    x_cols: list[str],
    y_cols: list[str],
) -> dict[str, dict[str, dict[str, float | None]]]:
    if group_col not in df.columns:
        return {}
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for value, group in df.groupby(group_col, dropna=False):
        key = "NA" if pd.isna(value) else str(value)
        out[key] = correlations(group, x_cols, y_cols)
    return out


def add_loss_quantiles(df: pd.DataFrame, primary_loss: str) -> pd.DataFrame:
    if primary_loss not in df.columns:
        return df
    out = df.copy()
    out["primary_loss_quantile"] = pd.qcut(
        out[primary_loss].rank(method="first"),
        q=4,
        labels=["lowest", "low_mid", "high_mid", "highest"],
    )
    return out


def optional_value(row: pd.Series, column: str) -> str:
    if column not in row.index:
        return "NA"
    value = row[column]
    if pd.isna(value):
        return "NA"
    return str(value)


def format_examples(title: str, rows: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    lines = [title, "=" * len(title), ""]
    if len(rows) == 0:
        lines.extend(["No examples.", ""])
        return lines

    for rank, (_, row) in enumerate(rows.iterrows(), start=1):
        text = optional_value(row, args.text_col).replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join(textwrap.wrap(text, width=args.wrap_width, replace_whitespace=False))
        lines.extend(
            [
                f"[{rank}] id={optional_value(row, 'id')} | map_train={optional_value(row, 'map_train')}",
                (
                    f"{args.primary_loss}={float(row[args.primary_loss]):.6f} | "
                    f"weak_confidence={optional_value(row, args.confidence_col)} | "
                    f"weak_correct={optional_value(row, args.correct_col)} | "
                    f"label={optional_value(row, args.label_col)} | "
                    f"weak_pred={optional_value(row, args.pred_col)}"
                ),
                text,
                "",
            ]
        )
    return lines


def format_metric_block(title: str, stats: dict[str, dict]) -> list[str]:
    lines = [title, "=" * len(title), ""]
    if not stats:
        lines.extend(["NA", ""])
        return lines
    for metric, values in stats.items():
        lines.append(
            f"- {metric}: n={values['n']}, mean={fmt(values['mean'])}, median={fmt(values['median'])}, "
            f"q10={fmt(values['q10'])}, q90={fmt(values['q90'])}, range={fmt(values['min'])}-{fmt(values['max'])}"
        )
    lines.append("")
    return lines


def fmt(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.6f}"


def write_report(df: pd.DataFrame, summary: dict, args: argparse.Namespace, metrics: list[str]) -> None:
    lines = [
        f"Mapping CSV: {args.mapping_csv}",
        f"Confidence CSV: {args.confidence_csv or 'NA'}",
        f"n_examples: {len(df)}",
        f"primary_loss: {args.primary_loss}",
        "",
    ]

    primary_stats = {args.primary_loss: summarize_series(df[args.primary_loss])} if args.primary_loss in df.columns else {}
    lines.extend(format_metric_block("Primary Loss", primary_stats))

    if "map_train" in df.columns:
        lines.extend(["By Mapping Split", "================", ""])
        for value, name in [(1, "mapping train"), (0, "heldout")]:
            sub = df[df["map_train"] == value]
            if len(sub) == 0:
                continue
            lines.append(f"{name}: n={len(sub)}")
            for metric in metrics:
                values = summarize_series(sub[metric])
                lines.append(f"- {metric}: mean={fmt(values['mean'])}, median={fmt(values['median'])}")
            lines.append("")

    if args.correct_col in df.columns:
        lines.extend(["By Weak Correctness", "===================", ""])
        for value, name in [(1, "weak correct"), (0, "weak wrong")]:
            sub = df[df[args.correct_col] == value]
            if len(sub) == 0:
                continue
            lines.append(f"{name}: n={len(sub)}")
            for metric in metrics:
                values = summarize_series(sub[metric])
                lines.append(f"- {metric}: mean={fmt(values['mean'])}, median={fmt(values['median'])}")
            lines.append("")

    if "map_train" in df.columns and args.correct_col in df.columns:
        lines.extend(["By Mapping Split And Weak Correctness", "=====================================", ""])
        for map_value, map_name in [(1, "mapping train"), (0, "heldout")]:
            for correct_value, correct_name in [(1, "weak correct"), (0, "weak wrong")]:
                sub = df[(df["map_train"] == map_value) & (df[args.correct_col] == correct_value)]
                if len(sub) == 0:
                    continue
                lines.append(f"{map_name}, {correct_name}: n={len(sub)}")
                for metric in metrics:
                    values = summarize_series(sub[metric])
                    lines.append(f"- {metric}: mean={fmt(values['mean'])}, median={fmt(values['median'])}")
                lines.append("")

    lines.extend(["Correlations", "============", ""])
    for loss_col, corr_map in summary.get("correlations", {}).items():
        rendered = ", ".join(f"{target}={fmt(value)}" for target, value in corr_map.items())
        lines.append(f"- {loss_col}: {rendered}")
    lines.append("")

    if summary.get("correlations_by_map_train"):
        lines.extend(["Correlations By Mapping Split", "=============================", ""])
        for split_value, corr_group in summary["correlations_by_map_train"].items():
            split_name = "heldout" if split_value == "0" else "mapping train" if split_value == "1" else split_value
            lines.append(f"{split_name}:")
            for loss_col, corr_map in corr_group.items():
                rendered = ", ".join(f"{target}={fmt(value)}" for target, value in corr_map.items())
                lines.append(f"- {loss_col}: {rendered}")
            lines.append("")

    if args.primary_loss in df.columns:
        high = df.sort_values(args.primary_loss, ascending=False).head(args.n_examples)
        low = df.sort_values(args.primary_loss, ascending=True).head(args.n_examples)
        lines.extend(format_examples(f"Highest {args.primary_loss} Examples", high, args))
        lines.extend(format_examples(f"Lowest {args.primary_loss} Examples", low, args))

    output = Path(args.report_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def maybe_write_plot(df: pd.DataFrame, args: argparse.Namespace) -> None:
    if not args.plot_output:
        return
    if args.primary_loss not in df.columns or args.confidence_col not in df.columns:
        print("Skipping plot: primary loss or confidence column is missing.")
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Skipping plot: matplotlib is not installed.")
        return

    plot_cols = [args.primary_loss, args.confidence_col, args.correct_col]
    if "map_train" in df.columns:
        plot_cols.append("map_train")
    plot_df = df[plot_cols].copy()
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce").dropna()
    if args.plot_heldout_only and "map_train" in plot_df.columns:
        plot_df = plot_df[plot_df["map_train"] == 0]
    if len(plot_df) == 0:
        print("Skipping plot: no numeric rows.")
        return

    colors = plot_df[args.correct_col].map({1: "#31a354", 0: "#ef3b2c"}).fillna("#636363")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if "map_train" in plot_df.columns and not args.plot_heldout_only:
        train = plot_df["map_train"] == 1
        ax.scatter(
            plot_df.loc[~train, args.confidence_col],
            plot_df.loc[~train, args.primary_loss],
            c=colors.loc[~train],
            alpha=0.85,
            edgecolors="none",
            marker="o",
            label="heldout",
        )
        ax.scatter(
            plot_df.loc[train, args.confidence_col],
            plot_df.loc[train, args.primary_loss],
            c=colors.loc[train],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
            marker="x",
            label="mapping train",
        )
        ax.legend(frameon=False)
    else:
        ax.scatter(plot_df[args.confidence_col], plot_df[args.primary_loss], c=colors, alpha=0.8, edgecolors="none")
    ax.set_xlabel(args.confidence_col)
    ax.set_ylabel(args.primary_loss)
    title = "Mapping Loss vs Weak Confidence"
    if args.plot_heldout_only:
        title += " (heldout only)"
    ax.set_title(title)
    fig.tight_layout()

    output = Path(args.plot_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mapping_df = read_csv_with_string_id(args.mapping_csv)
    conf_df = read_csv_with_string_id(args.confidence_csv) if args.confidence_csv else None

    df = merge_tables(mapping_df, conf_df, args.merge_on)
    if args.correct_col in df.columns:
        df[args.correct_col] = pd.to_numeric(df[args.correct_col], errors="coerce")
    if args.primary_loss not in df.columns:
        raise ValueError(f"Primary loss column not found: {args.primary_loss}")
    df = add_loss_quantiles(df, args.primary_loss)

    metrics = metric_columns(mapping_df)
    target_cols = [col for col in [args.confidence_col, args.correct_col, args.prob_col] if col in df.columns]
    summary = {
        "mapping_csv": args.mapping_csv,
        "confidence_csv": args.confidence_csv,
        "n_examples": int(len(df)),
        "merge_on": args.merge_on,
        "primary_loss": args.primary_loss,
        "metrics": {metric: summarize_series(df[metric]) for metric in metrics},
        "by_map_train": summarize_by_group(df, "map_train", metrics),
        "by_weak_correct": summarize_by_group(df, args.correct_col, metrics),
        "by_map_train_and_weak_correct": summarize_by_two_groups(df, ["map_train", args.correct_col], metrics),
        "by_primary_loss_quantile": summarize_by_group(df, "primary_loss_quantile", metrics),
        "correlations": correlations(df, metrics, target_cols),
        "correlations_by_map_train": correlations_by_group(df, "map_train", metrics, target_cols),
    }

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_report(df, summary, args, metrics)

    if args.merged_output:
        merged_path = Path(args.merged_output)
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(merged_path, index=False)

    maybe_write_plot(df, args)

    print(json.dumps(summary, indent=2))
    print(f"Wrote summary to {args.summary_output}")
    print(f"Wrote report to {args.report_output}")
    if args.merged_output:
        print(f"Wrote merged CSV to {args.merged_output}")
    if args.plot_output:
        print(f"Plot requested at {args.plot_output}")


if __name__ == "__main__":
    main()
