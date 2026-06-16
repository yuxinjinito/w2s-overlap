#!/usr/bin/env python3
"""Partial / final results table for a band-map sweep, with training-set size and accuracy.

Run ON the box where results/ lives; reads each completed seed's summary.json and
aggregates per method. For every run reports, alongside the fine-tuned eval accuracy:
  - training set SIZE                (train_subset.n)
  - training set ACCURACY            (train_subset.train_label_accuracy = weak-label
                                      accuracy on the selected subset)
plus the data generator / fine-tuned model identities. Size and train accuracy are the
same across train-seeds (the subset is fixed; only LoRA training varies), so they are
shown once; the eval accuracy is aggregated mean +/- std across seeds.

Usage:
  python3 scripts/peek_results.py results/hellaswag_bounds_epochs_0616 [weak_model] [strong_model]
  # POSIX sh (no conda init): micromamba run -n <env> python scripts/peek_results.py <root>
"""
import glob
import json
import os
import statistics as st
import sys


def _get(o, *keys):
    for k in keys:
        o = o.get(k) if isinstance(o, dict) else None
        if o is None:
            return None
    return o


def _acc3(rv):
    def rec(o):
        if isinstance(o, dict):
            if "accuracy_3class" in o:
                return o["accuracy_3class"]
            for v in o.values():
                r = rec(v)
                if r is not None:
                    return r
        return None
    return rec(rv.get("eval", rv) if isinstance(rv, dict) else rv)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: peek_results.py <output_root> [weak_model] [strong_model]")
        sys.exit(1)
    root = sys.argv[1].rstrip("/")
    weak_model = sys.argv[2] if len(sys.argv) > 2 else "Qwen1.5-0.5B"
    strong_model = sys.argv[3] if len(sys.argv) > 3 else "Llama-3.1-8B"

    base = gt = weak = None
    for f in glob.glob(os.path.join(root, "*baseline*", "summary.json")):
        d = json.load(open(f))
        runs = d.get("runs", {})
        if "base" in runs:
            base = _acc3(runs["base"])
        if "ground_truth" in runs:
            gt = _acc3(runs["ground_truth"])
        weak = d.get("weak_label_diagnostics", {}).get("accuracy_3class", weak)

    seeds = sorted(
        glob.glob(os.path.join(root, "filtering", "trainseed_*", "summary.json")),
        key=os.path.getmtime,
    )
    agg, meta = {}, {}
    for f in seeds:
        for m, rv in json.load(open(f)).get("runs", {}).items():
            a = _acc3(rv)
            if a is not None:
                agg.setdefault(m, []).append(a)
            meta[m] = (_get(rv, "train_subset", "n"), _get(rv, "train_subset", "train_label_accuracy"))

    print(f"=== {root}  (filter seeds: {len(seeds)}) ===")
    print(f"data generator: {weak_model}   ->   fine-tuned: {strong_model}")
    print(f"anchors:  weak {weak}   base {base}   GT {gt}")
    print(f"{'method':30} {'size':>5} {'tr_acc':>7} {'eval_acc3 (mean+/-std)':>23} {'vs base':>8}")
    rows = [(st.mean(v), m, (st.stdev(v) if len(v) > 1 else 0.0), len(v)) for m, v in agg.items()]
    for mean, m, sd, n in sorted(rows, reverse=True):
        sz, ta = meta.get(m, (None, None))
        ta_s = f"{ta:.3f}" if ta is not None else "  -  "
        db = f"{mean - base:+.3f}" if base is not None else "  -  "
        print(f"{m:30} {str(sz):>5} {ta_s:>7} {mean:>11.3f} +/-{sd:.3f} (n={n})   {db}")


if __name__ == "__main__":
    main()
