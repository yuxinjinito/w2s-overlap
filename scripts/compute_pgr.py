#!/usr/bin/env python3
"""Compute Performance Gap Recovered (PGR) for a paper-style sweep output directory.

PGR(method) = (method - weak) / (strong_GT - weak), the standard weak-to-strong
metric: 0 means no better than the weak model, 1 means it matches a strong model
trained on ground-truth labels.

Anchors:
  weak     = weak probe on the test set
  strong_GT= the ground_truth-trained run (strong model on true labels)

Reported on three metrics, averaged over LoRA seeds:
  accuracy       (paper-comparable, but threshold-brittle here)
  prior_matched  (accuracy at the balanced operating point; de-brittled)
  auroc          (threshold-free ranking)

Usage:
  PYTHONPATH=scripts .venv/bin/python scripts/compute_pgr.py <run_dir>
where <run_dir>/baseline_seed42/eval_predictions.csv has weak + ground_truth,
and <run_dir>/filtering/trainseed_*/eval_predictions.csv has the filter methods.
"""
import sys
import csv
import glob
import os
import statistics

import numpy as np


def acc(scores, y, t=0.5):
    return float(((np.asarray(scores) >= t).astype(int) == np.asarray(y)).mean())


def prior_matched(scores, y):
    s = np.asarray(scores, float)
    y = np.asarray(y, int)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    preds = np.zeros(len(y), int)
    preds[order[:n_pos]] = 1
    return float((preds == y).mean())


def auroc(scores, y):
    s = np.asarray(scores, float)
    y = np.asarray(y, int)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks_sorted = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks_sorted[i : j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks = np.empty(len(s))
    ranks[order] = ranks_sorted
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


METRICS = {"accuracy": acc, "prior_matched": prior_matched, "auroc": auroc}


def load(fp):
    rows = list(csv.DictReader(open(fp)))
    y = [int(float(r["label"])) for r in rows]
    return rows, y


def col(rows, name):
    return [float(r[name]) for r in rows]


def main(run_dir):
    brows, by = load(os.path.join(run_dir, "baseline_seed42", "eval_predictions.csv"))
    if "ground_truth_prob_label1" not in brows[0]:
        raise SystemExit("baseline eval_predictions.csv has no ground_truth column; PGR needs the GT anchor.")
    anchors = {}
    for name, fn in METRICS.items():
        anchors[name] = {
            "weak": fn(col(brows, "weak_prob_label1"), by),
            "gt": fn(col(brows, "ground_truth_prob_label1"), by),
        }

    seed_files = sorted(glob.glob(os.path.join(run_dir, "filtering", "trainseed_*", "eval_predictions.csv")))
    method_vals = {name: {} for name in METRICS}
    for fp in seed_files:
        rows, y = load(fp)
        methods = [
            c[: -len("_prob_label1")]
            for c in rows[0]
            if c.endswith("_prob_label1") and c != "weak_prob_label1"
        ]
        for m in methods:
            for name, fn in METRICS.items():
                method_vals[name].setdefault(m, []).append(fn(col(rows, f"{m}_prob_label1"), y))

    print(f"PGR for {run_dir}  ({len(seed_files)} LoRA seed(s))")
    for name in METRICS:
        w, g = anchors[name]["weak"], anchors[name]["gt"]
        print(f"\n=== {name}  (weak={w:.3f}, strong_GT={g:.3f}, gap={g - w:.3f}) ===")
        print(f"{'method':32} {'value':>7} {'PGR':>7}")
        out = []
        for m, vals in method_vals[name].items():
            v = statistics.mean(vals)
            pgr = (v - w) / (g - w) if g != w else float("nan")
            out.append((m, v, pgr))
        for m, v, pgr in sorted(out, key=lambda r: -(r[2] if r[2] == r[2] else -9)):
            print(f"{m:32} {v:>7.3f} {pgr:>7.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: compute_pgr.py <run_dir>")
    main(sys.argv[1])
