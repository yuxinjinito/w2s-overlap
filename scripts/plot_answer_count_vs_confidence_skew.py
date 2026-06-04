#!/usr/bin/env python3
"""Plot dataset answer/class count against weak-confidence skew.

This intentionally uses only the Python standard library and writes SVG so the
plot can be generated on a lightweight local machine without matplotlib.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path


INPUT = Path("experiments/2026-05-28-dataset-answer-counts.csv")
OUTPUT = Path("experiments/2026-05-28-answer-count-vs-confidence-skew.svg")


COLORS = {
    "very skewed/high-confidence": "#d95f02",
    "moderately skewed or mixed": "#7570b3",
    "healthier / less skewed": "#1b9e77",
    "low-confidence / weak probe near chance": "#666666",
}


def sx(x: float) -> float:
    # Map original answer count in [1.75, 4.25] to plot x.
    return 120 + (x - 1.75) / (4.25 - 1.75) * 690


def sy(y: float) -> float:
    # Map confidence mass in [0, 0.85] to plot y.
    return 520 - y / 0.85 * 410


def text(x: float, y: float, value: str, size: int = 18, anchor: str = "middle", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="#202020">{html.escape(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str = "#dddddd", width: float = 1.0) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"/>'


def circle(x: float, y: float, r: float, color: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" fill-opacity="0.88" stroke="#ffffff" stroke-width="2"/>'


def main() -> None:
    rows = []
    with INPUT.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["x"] = float(row["original_choices_or_classes"])
            row["y"] = float(row["high_confidence_mass_gt_0_90"])
            rows.append(row)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="680" viewBox="0 0 1000 680">',
        '<rect width="1000" height="680" fill="#ffffff"/>',
        text(500, 52, "Answer/Class Count vs Weak-Confidence Skew", 30, weight="700"),
        text(500, 84, "Y-axis: fraction of examples with weak confidence > 0.9", 17),
    ]

    # Plot area.
    left, right, top, bottom = 120, 810, 110, 520
    parts.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#fbfbfb" stroke="#cfcfcf"/>')

    for y in [0.0, 0.2, 0.4, 0.6, 0.8]:
        yy = sy(y)
        parts.append(line(left, yy, right, yy, "#e4e4e4"))
        parts.append(text(left - 16, yy + 6, f"{y:.1f}", 15, "end"))

    for x in [2, 3, 4]:
        xx = sx(x)
        parts.append(line(xx, top, xx, bottom, "#eeeeee"))
        parts.append(text(xx, bottom + 36, str(x), 18))

    parts.append(text((left + right) / 2, 610, "Original number of answer choices / classes", 19))
    parts.append(
        '<text x="38" y="315" font-family="Arial, Helvetica, sans-serif" font-size="19" '
        'text-anchor="middle" fill="#202020" transform="rotate(-90 38 315)">High-confidence mass</text>'
    )

    # Draw points with a tiny deterministic jitter for datasets sharing the same x.
    by_x_seen: dict[int, int] = {}
    label_offsets = {
        "Amazon": (10, -16, "start"),
        "SST-2": (10, -10, "start"),
        "Twitter Sentiment": (10, 6, "start"),
        "BoolQ": (10, -4, "start"),
        "WiC": (10, -2, "start"),
        "CoLA": (10, 14, "start"),
        "ANLI-R2": (10, -12, "start"),
        "SciQ": (10, -10, "start"),
        "PAWS": (-12, 18, "end"),
        "Dream": (-12, 4, "end"),
        "HellaSwag": (10, 16, "start"),
    }
    for row in rows:
        x_key = int(row["x"])
        seen = by_x_seen.get(x_key, 0)
        by_x_seen[x_key] = seen + 1
        jitter = (seen - 2.5) * 8 if x_key == 2 else (seen - 0.5) * 8
        xx = sx(row["x"]) + jitter
        yy = sy(row["y"])
        color = COLORS[row["skew_bucket"]]
        parts.append(circle(xx, yy, 9, color))
        dx, dy, anchor = label_offsets.get(row["display_name"], (10, -10, "start"))
        parts.append(text(xx + dx, yy + dy, row["display_name"], 14, anchor))

    # Legend.
    legend_x, legend_y = 675, 130
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="280" height="130" rx="6" fill="#ffffff" stroke="#d5d5d5"/>')
    parts.append(text(legend_x + 14, legend_y + 26, "Skew bucket", 16, "start", "700"))
    for i, (label, color) in enumerate(COLORS.items()):
        yy = legend_y + 52 + i * 24
        parts.append(circle(legend_x + 18, yy - 5, 6, color))
        parts.append(text(legend_x + 34, yy, label, 13, "start"))

    # Takeaway callout.
    parts.append('<rect x="116" y="625" width="762" height="34" rx="6" fill="#f5f7f7" stroke="#d7dddd"/>')
    parts.append(
        text(
            497,
            648,
            "Takeaway: answer count matters, but PAWS shows binary tasks do not automatically become highly skewed.",
            16,
        )
    )

    parts.append("</svg>")
    OUTPUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
