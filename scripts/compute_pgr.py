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
import re
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


def spearman(x, y):
    """Rank correlation with midranks, so ties do not depend on input order."""
    def _midranks(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind="mergesort")
        ranks_sorted = np.arange(1, len(v) + 1, dtype=float)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            ranks_sorted[i : j + 1] = (i + 1 + j + 1) / 2.0
            i = j + 1
        ranks = np.empty(len(v))
        ranks[order] = ranks_sorted
        return ranks
    return float(np.corrcoef(_midranks(x), _midranks(y))[0, 1])


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
            group = re.sub(r"_\d+$", "", m)  # pool random_balanced_0/1/2 -> random_balanced
            for name, fn in METRICS.items():
                method_vals[name].setdefault(group, []).append(fn(col(rows, f"{m}_prob_label1"), y))

    print(f"PGR for {run_dir}  ({len(seed_files)} LoRA seed(s); +/- = std across seeds x controls)")
    for name in METRICS:
        w, g = anchors[name]["weak"], anchors[name]["gt"]
        print(f"\n=== {name}  (weak={w:.3f}, strong_GT={g:.3f}, gap={g - w:.3f}) ===")
        print(f"{'method':30} {'value (mean±sd)':>17} {'PGR (mean±sd)':>20} {'n':>3}")
        out = []
        for m, vals in method_vals[name].items():
            v = statistics.mean(vals)
            vsd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            pgrs = [(x - w) / (g - w) for x in vals] if g != w else [float("nan")]
            pgr = statistics.mean(pgrs)
            psd = statistics.stdev(pgrs) if len(pgrs) > 1 else 0.0
            out.append((m, v, vsd, pgr, psd, len(vals)))
        for m, v, vsd, pgr, psd, n in sorted(out, key=lambda r: -(r[3] if r[3] == r[3] else -9)):
            print(f"{m:30} {v:>8.3f} ± {vsd:5.3f} {pgr:>+11.3f} ± {psd:5.3f} {n:>3}")


def _self_test() -> None:
    rng = np.random.default_rng(0)
    s = rng.standard_normal(400)
    y = (rng.random(400) < 0.4).astype(int)
    # constant scores carry no ranking information: AUROC must be exactly 0.5
    assert auroc(np.zeros(400), y) == 0.5
    # a score equal to the label separates perfectly
    assert auroc(y.astype(float), y) == 1.0
    assert abs(spearman(s, s) - 1.0) < 1e-12
    assert abs(spearman(s, -s) + 1.0) < 1e-12
    # single-class labels have no negative (or positive) set to rank against
    assert np.isnan(auroc(s, np.ones(400, int)))
    print("PGR METRICS: SELF-TEST PASSED")


if __name__ == "__main__" and len(sys.argv) == 2 and sys.argv[1] == "--self-test":
    _self_test()
elif __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: compute_pgr.py <run_dir>")
    main(sys.argv[1])
