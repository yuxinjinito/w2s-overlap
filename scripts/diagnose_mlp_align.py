#!/usr/bin/env python3
"""Training diagnostics for the MLP alignment score (asked for at the 07/22 meeting).

Extracts the same weak/strong representations the pipeline uses, then fits the alignment MLP
under several settings and reports per-epoch train vs held-out MSE, so we can see whether the
model has converged, is underfitting, or is overfitting the fold-train points.

Usage (GPU box):
  PYTHONPATH=scripts python3 scripts/diagnose_mlp_align.py --dataset anli --anli-round r1
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mlp_alignment import mlp_alignment_scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-npz", required=True,
                    help="npz with weak/strong strong_train activations (see --dump-acts run)")
    ap.add_argument("--out", default="mlp_align_diagnostics.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    d = np.load(args.acts_npz)
    Xw, Xs = d["weak"], d["strong"]
    print(f"weak {Xw.shape}  strong {Xs.shape}")

    # (name, hidden, bottleneck, epochs, weight_decay)
    settings = [
        ("deployed", 1024, None, 150, 1e-4),
        ("long", 1024, None, 600, 1e-4),
        ("bottleneck128", 1024, 128, 300, 1e-4),
        ("small_wd", 256, None, 300, 1e-2),
    ]
    report = {}
    for name, hidden, bneck, epochs, wd in settings:
        hist: list = []
        s = mlp_alignment_scores(Xw, Xs, n_folds=5, hidden=hidden, epochs=epochs,
                                 weight_decay=wd, seed=42, device=args.device,
                                 bottleneck=bneck, history=hist)
        f0 = [h for h in hist if h["fold"] == 0]
        first, last = f0[0], f0[-1]
        best = min(f0, key=lambda h: h["heldout_mse"])
        report[name] = {
            "hidden": hidden, "bottleneck": bneck, "epochs": epochs, "weight_decay": wd,
            "fold0_train_first": first["train_mse"], "fold0_train_last": last["train_mse"],
            "fold0_heldout_first": first["heldout_mse"], "fold0_heldout_last": last["heldout_mse"],
            "fold0_heldout_best": best["heldout_mse"], "fold0_heldout_best_epoch": best["epoch"],
            "score_mean": float(s.mean()), "score_std": float(s.std()),
            "history": hist,
        }
        print(f"[{name}] train {first['train_mse']:.4f} -> {last['train_mse']:.4f} | "
              f"heldout {first['heldout_mse']:.4f} -> {last['heldout_mse']:.4f} "
              f"(best {best['heldout_mse']:.4f} @ep{best['epoch']})")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
