#!/usr/bin/env python3
"""Plot weak-confidence screening metrics across datasets.

The figure is meant for the experiment report: it shows why SciQ and PAWS were
chosen for the next representation-mapping stage after the first weak-confidence
screening pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SELECTED = {"sciq", "paws"}
DISPLAY_NAMES = {
    "amazon_polarity": "Amazon",
    "anli-r2": "ANLI-R2",
    "boolq": "BoolQ",
    "cola": "CoLA",
    "hellaswag": "HellaSwag",
    "paws": "PAWS",
    "sciq": "SciQ",
    "sst2": "SST-2",
    "wic": "WiC",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="/Users/yuxinjin/Downloads/w2s_confidence_paper10_0524",
        help="Directory containing *_qwen05_probe_*_1000.csv files.",
    )
    parser.add_argument(
        "--output",
        default="overleaf/w2s_project/sections/experimentreports/fig/confidence_dataset_selection_9.png",
    )
    parser.add_argument(
        "--metrics-output",
        default="overleaf/w2s_project/sections/experimentreports/fig/confidence_dataset_selection_9_metrics.csv",
    )
    parser.add_argument("--bins", type=int, default=20)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_metrics(input_dir: Path, bins: int) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(input_dir.glob("*_qwen05_probe_*_1000.csv")):
        df = pd.read_csv(csv_path)
        if "weak_confidence" not in df or "dataset" not in df:
            continue
        dataset = str(df["dataset"].iloc[0])
        conf = df["weak_confidence"].astype(float).to_numpy()
        hist, _ = np.histogram(conf, bins=bins, range=(0.0, 1.0))
        weak_accuracy = float(df["weak_correct"].astype(float).mean()) if "weak_correct" in df else np.nan
        median_confidence = float(np.median(conf))
        spread_q90_q10 = float(np.quantile(conf, 0.90) - np.quantile(conf, 0.10))
        high_conf_mass = float(np.mean(conf > 0.90))
        # A compact screening score for this report: prefer broad distributions,
        # penalize confidence collapse near 1, and prefer a middle-ish median.
        selection_score = spread_q90_q10 - high_conf_mass - abs(median_confidence - 0.65)
        rows.append(
            {
                "dataset": dataset,
                "display_name": DISPLAY_NAMES.get(dataset, dataset),
                "n_examples": len(df),
                "weak_accuracy": weak_accuracy,
                "median_confidence": median_confidence,
                "q10_confidence": float(np.quantile(conf, 0.10)),
                "q90_confidence": float(np.quantile(conf, 0.90)),
                "spread_q90_q10": spread_q90_q10,
                "high_conf_mass_gt_0_90": high_conf_mass,
                "selection_score": float(selection_score),
                "hist_counts": hist.tolist(),
                "selected": dataset in SELECTED,
            }
        )
    data = pd.DataFrame(rows)
    return data.sort_values(["selection_score", "dataset"], ascending=[False, True]).reset_index(drop=True)


def draw_metric_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    value: float,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] = (205, 211, 219),
) -> None:
    draw.rounded_rectangle([x, y, x + width, y + height], radius=4, outline=outline, width=1, fill=(248, 250, 252))
    draw.rounded_rectangle([x, y, x + int(width * value), y + height], radius=4, fill=fill)


def draw_figure(data: pd.DataFrame, output: Path, bins: int) -> None:
    width = 1800
    header_h = 165
    row_h = 106
    footer_h = 24
    height = header_h + row_h * len(data) + footer_h
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    title_font = load_font(38, bold=True)
    subtitle_font = load_font(22)
    label_font = load_font(24, bold=True)
    small_font = load_font(19)
    tiny_font = load_font(17)
    value_font = load_font(21)

    text = (23, 33, 43)
    muted = (93, 104, 120)
    grid = (229, 233, 240)
    selected_green = (22, 128, 86)
    selected_bg = (232, 247, 239)
    grey_bar = (104, 121, 140)
    blue = (50, 100, 165)
    orange = (226, 129, 52)

    draw.text((60, 34), "Weak-Confidence Screening Across 9 Datasets", fill=text, font=title_font)
    draw.text(
        (60, 84),
        "Sorted by selection score: broad confidence spread, less high-confidence pile-up, and a median away from saturation.",
        fill=muted,
        font=subtitle_font,
    )

    x_name = 60
    x_hist = 300
    hist_w = 690
    x_score = 1060
    x_spread = 1255
    x_high = 1450
    x_med = 1620
    metric_w = 120

    draw.text((x_name, 130), "Dataset", fill=muted, font=small_font)
    draw.text((x_hist, 130), "confidence histogram", fill=muted, font=small_font)
    draw.text((x_score, 130), "selection score", fill=muted, font=small_font)
    draw.text((x_spread, 130), "q90-q10", fill=muted, font=small_font)
    draw.text((x_high, 130), "P(conf > .90)", fill=muted, font=small_font)
    draw.text((x_med, 130), "median", fill=muted, font=small_font)

    for i, row in data.iterrows():
        y = header_h + i * row_h
        selected = bool(row["selected"])
        if selected:
            draw.rounded_rectangle([38, y + 8, width - 38, y + row_h - 8], radius=14, fill=selected_bg)
            draw.text((x_name - 28, y + 31), "*", fill=selected_green, font=label_font)
        else:
            draw.line([38, y, width - 38, y], fill=grid, width=1)

        name_color = selected_green if selected else text
        draw.text((x_name, y + 23), row["display_name"], fill=name_color, font=label_font)
        draw.text((x_name, y + 53), f"n={int(row['n_examples'])}", fill=muted, font=tiny_font)

        # Histogram as compact vertical bars.
        hist = np.array(row["hist_counts"], dtype=float)
        bar_gap = 3
        bar_w = int((hist_w - bar_gap * (bins - 1)) / bins)
        base_y = y + 79
        max_bar_h = 64
        bar_color = selected_green if selected else grey_bar
        row_max = max(float(hist.max()), 1.0)
        axis_x = x_hist - 8
        draw.line([axis_x, base_y - max_bar_h, axis_x, base_y], fill=(72, 84, 101), width=3)
        draw.line([axis_x - 4, base_y, x_hist + hist_w + 2, base_y], fill=(172, 181, 194), width=2)
        for j, count in enumerate(hist):
            h = int(max_bar_h * count / row_max)
            x0 = x_hist + j * (bar_w + bar_gap)
            draw.rectangle([x0, base_y - h, x0 + bar_w, base_y], fill=bar_color)
        for tick, label in [(0.0, "0"), (0.5, ".5"), (1.0, "1")]:
            x_tick = x_hist + int(tick * hist_w)
            draw.line([x_tick, base_y + 2, x_tick, base_y + 8], fill=muted, width=1)
            draw.text((x_tick - 9, base_y + 11), label, fill=muted, font=tiny_font)

        score = float(row["selection_score"])
        med = float(row["median_confidence"])
        spread = float(row["spread_q90_q10"])
        high = float(row["high_conf_mass_gt_0_90"])
        metric_color = selected_green if selected else blue
        high_color = selected_green if selected else orange

        draw_metric_bar(draw, x_score, y + 28, metric_w, 16, max(min(score, 1.0), 0.0), metric_color)
        draw.text((x_score, y + 53), f"{score:.2f}", fill=text, font=value_font)
        draw_metric_bar(draw, x_spread, y + 23, metric_w, 16, min(spread, 1.0), metric_color)
        draw.text((x_spread, y + 47), f"{spread:.2f}", fill=text, font=value_font)
        draw_metric_bar(draw, x_high, y + 23, metric_w, 16, high, high_color)
        draw.text((x_high, y + 47), f"{high:.2f}", fill=text, font=value_font)
        draw_metric_bar(draw, x_med, y + 23, metric_w, 16, med, metric_color)
        draw.text((x_med, y + 47), f"{med:.2f}", fill=text, font=value_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output = Path(args.output)
    metrics_output = Path(args.metrics_output)
    data = load_metrics(input_dir, args.bins)
    if data.empty:
        raise SystemExit(f"No matching CSV files found in {input_dir}")

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    data.drop(columns=["hist_counts"]).to_csv(metrics_output, index=False)
    draw_figure(data, output, args.bins)
    print(f"Wrote figure to {output}")
    print(f"Wrote metrics to {metrics_output}")


if __name__ == "__main__":
    main()
