#!/usr/bin/env python3
"""Why does rp_high work? Per-example analysis of the RP selection score.

Reads a run's strong_train_labels.csv (needs the rp_score column) and asks what the top-50%
rp_high band actually selects, compared with the confidence_high band:
  - Spearman correlation of rp_score with label-correctness / weak-confidence / L2-residual / kNN.
  - rp_high vs confidence_high: label cleanliness in each band + how much the two bands overlap.

Reading: a LOW rp<->confidence correlation and LOW band overlap means rp is a DIFFERENT signal --
it keeps learnable / overlap points that confidence_high misses, rather than just cleaner/easier
ones. That is the overlap-density story (and explains why rp beats confidence despite confidence
keeping cleaner labels).

Usage:
  python3 scripts/analyze_rp.py <run>/filtering/trainseed_42/strong_train_labels.csv
"""
import csv
import sys

import numpy as np


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: analyze_rp.py <strong_train_labels.csv>")
        sys.exit(1)
    rows = [r for r in csv.DictReader(open(sys.argv[1])) if r.get("rp_score", "") != ""]
    if not rows:
        print("No rows with an rp_score column -- re-run with the updated pipeline.")
        sys.exit(1)

    rp = np.array([float(r["rp_score"]) for r in rows])
    wc = np.array([int(r["weak_correct"]) for r in rows])          # 1 if weak label == gold
    wconf = np.array([float(r["weak_confidence"]) for r in rows])
    res = np.array([float(r["residual_l2"]) for r in rows])
    has_knn = rows[0].get("knn_correct_rate", "") != ""
    knn = np.array([float(r["knn_correct_rate"]) for r in rows]) if has_knn else None
    label = np.array([int(r["label"]) for r in rows])
    has_base = rows[0].get("base_confidence", "") != ""
    bconf = np.array([float(r["base_confidence"]) for r in rows]) if has_base else None
    bprob = np.array([float(r["base_prob_correct"]) for r in rows]) if has_base else None
    # base_correct = does the untuned base classify this (question, candidate) right
    bcorr = ((bprob >= 0.5).astype(int) == label).astype(int) if has_base else None
    n = len(rp)

    print(f"n = {n} strong_train examples;  overall weak-label accuracy = {wc.mean():.3f}")

    print("\n== Spearman corr of rp_score with ==")
    print(f"  weak_correct (label cleanliness) : {spearman(rp, wc):+.3f}")
    print(f"  weak_confidence                  : {spearman(rp, wconf):+.3f}   <- low => rp is a DIFFERENT signal")
    print(f"  residual_l2 (L2-map score)       : {spearman(rp, res):+.3f}")
    if knn is not None:
        print(f"  knn_correct_rate                 : {spearman(rp, knn):+.3f}")
    if has_base:
        print(f"  base_correct (base gets it right): {spearman(rp, bcorr):+.3f}   (overall base acc {bcorr.mean():.3f})")
        print(f"  base_prob_correct (raw P)        : {spearman(rp, bprob):+.3f}")
        print(f"  base_confidence (strong base)    : {spearman(rp, bconf):+.3f}")

    k = n // 2
    rp_hi = set(np.argsort(-rp)[:k].tolist())
    cf_hi = set(np.argsort(-wconf)[:k].tolist())
    inter = len(rp_hi & cf_hi)
    jacc = inter / len(rp_hi | cf_hi)
    rp_idx = np.array(sorted(rp_hi))
    cf_idx = np.array(sorted(cf_hi))
    print("\n== top-50% bands ==")
    print(f"  rp_high          label-cleanliness (mean weak_correct): {wc[rp_idx].mean():.3f}")
    print(f"  confidence_high  label-cleanliness                   : {wc[cf_idx].mean():.3f}")
    print(f"  overlap rp_high & confidence_high (Jaccard)          : {jacc:.3f}   ({inter}/{k} shared)")

    only_rp = np.array(sorted(rp_hi - cf_hi))
    if len(only_rp):
        print(f"\n== points rp_high keeps but confidence_high drops (n={len(only_rp)}) ==")
        print(f"  weak_correct (cleanliness)        : {wc[only_rp].mean():.3f}")
        print(f"  weak_confidence (low = uncertain) : {wconf[only_rp].mean():.3f}")
        if has_base:
            print(f"  base_correct (these vs all)       : {bcorr[only_rp].mean():.3f}  (all: {bcorr.mean():.3f})"
                  "  <- higher => rp's distinctive picks are ones the base already gets right")
        print("  (these are rp's distinctive picks -- the points confidence would have thrown out)")


if __name__ == "__main__":
    main()
