#!/usr/bin/env python3
"""Weight-decay sweep for an MLP replacing rp's weak-side (step-1) ridge solve.

Follow-up to a 07/22 suggestion: rp is two ridge solves; regularize the MLP
parameters too and see whether the nonlinear version approaches the ridge behavior. The
strong-side MLP was already tested (nonlinear rp, report sec 4); this sweeps the weak side,
which previously failed with the default weight decay (the MLP fits the binary labels and
the residual turns into noise).

For each weight decay we fit a cross-fitted MLP ensemble regressing yc from the weak reps,
form the residual a_mlp = yc - oof_pred, push it through the SAME strong-side ridge solve as
rp, and compare against the all-ridge reference: residual scale, residual correlation, final
score correlation, and top-50% selection overlap.

Usage (GPU box, after a --dump-acts run):
  PYTHONPATH=scripts python3 scripts/diagnose_step1_mlp.py --acts-npz acts_r1_seed42.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from compute_pgr import spearman
import torch

from mlp_rp import _oof_regress


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return spearman(a, b)


def _center_cols(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def _ridge_residual(Xw: np.ndarray, yc: np.ndarray, reg: float) -> np.ndarray:
    n = Xw.shape[0]
    Kw = Xw @ Xw.T / n
    rw = reg * (np.trace(Kw) / n) + 1e-12
    return yc - Kw @ np.linalg.solve(Kw + rw * np.eye(n), yc)


def _strong_project(Xs: np.ndarray, a: np.ndarray, reg: float) -> np.ndarray:
    n = Xs.shape[0]
    Ks = Xs @ Xs.T / n
    rs = reg * (np.trace(Ks) / n) + 1e-12
    return Ks @ np.linalg.solve(Ks + rs * np.eye(n), a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-npz", required=True)
    ap.add_argument("--out", default="step1_mlp_wd_sweep.json")
    ap.add_argument("--reg", type=float, default=0.1)
    ap.add_argument("--wd-grid", default="1e-3,1e-2,1e-1,1,3,10")
    ap.add_argument("--n-ensemble", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    d = np.load(args.acts_npz)
    Xw = _center_cols(np.asarray(d["weak"], dtype=np.float64))
    Xs = _center_cols(np.asarray(d["strong"], dtype=np.float64))
    yhat = np.asarray(d["weak_preds"], dtype=np.float64)
    yc = yhat - yhat.mean()
    n = Xw.shape[0]
    print(f"n={n}  d_w={Xw.shape[1]}  d_s={Xs.shape[1]}")

    a_ridge = _ridge_residual(Xw, yc, args.reg)
    rp_ridge = np.abs(_strong_project(Xs, a_ridge, args.reg))
    top_ridge = set(np.argsort(-rp_ridge)[: n // 2].tolist())
    print(f"[ridge reference] resid std {a_ridge.std():.4f}  (yc std {yc.std():.4f})")

    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    Xw32 = Xw.astype(np.float32)
    report = {"n": n, "reg": args.reg,
              "ridge_resid_std": float(a_ridge.std()), "yc_std": float(yc.std()), "rows": []}
    for wd in [float(w) for w in args.wd_grid.split(",")]:
        preds = np.zeros(n)
        for e in range(args.n_ensemble):
            preds += _oof_regress(Xw32, yc, n_folds=5, hidden=256, epochs=60, lr=1e-3,
                                  batch_size=256, weight_decay=wd, seed=1000 + e, dev=dev)
        preds /= args.n_ensemble
        a_mlp = yc - preds
        rp_mlp = np.abs(_strong_project(Xs, a_mlp, args.reg))
        top_mlp = set(np.argsort(-rp_mlp)[: n // 2].tolist())
        row = {
            "wd": wd,
            "resid_std": float(a_mlp.std()),
            "resid_pearson_vs_ridge": float(np.corrcoef(a_mlp, a_ridge)[0, 1]),
            "resid_spearman_vs_ridge": _spearman(a_mlp, a_ridge),
            "score_spearman_vs_rp": _spearman(rp_mlp, rp_ridge),
            "top50_overlap": len(top_mlp & top_ridge) / max(1, len(top_ridge)),
        }
        report["rows"].append(row)
        print(f"wd={wd:<8g} resid_std {row['resid_std']:.4f}  "
              f"resid_corr {row['resid_pearson_vs_ridge']:.3f}  "
              f"score_spearman {row['score_spearman_vs_rp']:.3f}  "
              f"top50_overlap {row['top50_overlap']:.3f}")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
