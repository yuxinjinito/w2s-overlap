#!/usr/bin/env python3
"""Two controls for the 07/22 step-1 weight-decay sweep.

A. Strong-side MLP sweep: the report's nonlinear rp used wd=1e-2 (the flat region of the
   step-1 sweep), so the 07/22 "regularize harder" idea was never tested on the strong-side
   solve. Same protocol as the step-1 sweep, applied to step 2: regress the ridge step-1
   residual onto the strong reps with the deployed MLP harness across a wd grid, compare
   the projection against the all-ridge reference.

B. Linear-in-same-harness control: the step-1 MLP path differs from ridge not only in
   nonlinearity + weight decay but also in cross-fitting (oof) and per-fold z-scoring. A
   linear model trained in the IDENTICAL harness separates those factors: if linear-oof
   also caps well below 1.0 agreement, the ceiling belongs to the harness, not to
   nonlinearity. Run on both sides.

Usage:
  PYTHONPATH=scripts python3 scripts/diagnose_mlp_controls.py --acts-npz acts_r1_seed42.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn

from mlp_rp import _oof_regress, _zfit


def _rank(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r


def _spearman(a, b):
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def _center(X):
    return X - X.mean(axis=0, keepdims=True)


def _ridge_solve(X, y, reg):
    n = X.shape[0]
    K = X @ X.T / n
    r = reg * (np.trace(K) / n) + 1e-12
    return K @ np.linalg.solve(K + r * np.eye(n), y)


def _oof_linear(X, y, n_folds, epochs, lr, batch_size, weight_decay, seed, dev):
    """nn.Linear trained in the exact _oof_regress harness (z-score, AdamW, oof)."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    preds = np.zeros(n, dtype=np.float64)
    for fi, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(perm, test_idx, assume_unique=False)
        mu, sd = _zfit(X[train_idx])
        Xtr = torch.tensor((X[train_idx] - mu) / sd, dtype=torch.float32, device=dev)
        ytr = torch.tensor(y[train_idx], dtype=torch.float32, device=dev)
        Xte = torch.tensor((X[test_idx] - mu) / sd, dtype=torch.float32, device=dev)
        torch.manual_seed(seed * 1000 + fi)
        model = nn.Linear(X.shape[1], 1).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        m = Xtr.shape[0]
        model.train()
        for _ in range(epochs):
            order = torch.randperm(m, device=dev)
            for b0 in range(0, m, batch_size):
                idx = order[b0:b0 + batch_size]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(Xtr[idx]).squeeze(-1), ytr[idx])
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            preds[test_idx] = model(Xte).squeeze(-1).double().cpu().numpy()
        del model
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-npz", required=True)
    ap.add_argument("--out", default="mlp_controls.json")
    ap.add_argument("--reg", type=float, default=0.1)
    ap.add_argument("--wd-grid", default="1e-3,1e-2,1e-1,1,3,10,30,100,300")
    ap.add_argument("--n-ensemble", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    d = np.load(args.acts_npz)
    Xw = _center(np.asarray(d["weak"], dtype=np.float64))
    Xs = _center(np.asarray(d["strong"], dtype=np.float64))
    yhat = np.asarray(d["weak_preds"], dtype=np.float64)
    yc = yhat - yhat.mean()
    n = len(yc)
    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    wds = [float(w) for w in args.wd_grid.split(",")]
    print(f"n={n}  d_w={Xw.shape[1]}  d_s={Xs.shape[1]}  grid={wds}")

    # all-ridge reference
    a_ridge = yc - _ridge_solve(Xw, yc, args.reg)
    v_ridge = _ridge_solve(Xs, a_ridge, args.reg)
    rp_ridge = np.abs(v_ridge)
    top_ridge = set(np.argsort(-rp_ridge)[: n // 2].tolist())
    Xw32, Xs32 = Xw.astype(np.float32), Xs.astype(np.float32)

    def eval_scores(v_pred, name, wd, extra):
        s = np.abs(v_pred)
        top = set(np.argsort(-s)[: n // 2].tolist())
        row = {"wd": wd,
               "proj_pearson_vs_ridge": float(np.corrcoef(v_pred, v_ridge)[0, 1]) if extra else None,
               "score_spearman_vs_rp": _spearman(s, rp_ridge),
               "top50_overlap": len(top & top_ridge) / len(top_ridge)}
        print(f"[{name}] wd={wd:<8g} score_spearman {row['score_spearman_vs_rp']:.3f}  "
              f"top50_overlap {row['top50_overlap']:.3f}")
        return row

    report = {"n": n, "reg": args.reg, "strong_mlp": [], "weak_linear": [], "strong_linear": []}

    # A: strong-side MLP sweep (step 1 stays ridge; deployed harness = _oof_regress)
    for wd in wds:
        v = np.zeros(n)
        for e in range(args.n_ensemble):
            v += _oof_regress(Xs32, a_ridge, 5, 256, 60, 1e-3, 256, wd, 1000 + e, dev)
        v /= args.n_ensemble
        report["strong_mlp"].append(eval_scores(v, "A strong-MLP", wd, extra=True))

    # B1: weak-side LINEAR control (same harness; step 2 stays ridge)
    for wd in wds:
        p = np.zeros(n)
        for e in range(args.n_ensemble):
            p += _oof_linear(Xw32, yc, 5, 60, 1e-3, 256, wd, 1000 + e, dev)
        p /= args.n_ensemble
        a_lin = yc - p
        v = _ridge_solve(Xs, a_lin, args.reg)
        row = eval_scores(v, "B1 weak-linear", wd, extra=True)
        row["resid_pearson_vs_ridge"] = float(np.corrcoef(a_lin, a_ridge)[0, 1])
        report["weak_linear"].append(row)

    # B2: strong-side LINEAR control (step 1 stays ridge)
    for wd in wds:
        v = np.zeros(n)
        for e in range(args.n_ensemble):
            v += _oof_linear(Xs32, a_ridge, 5, 60, 1e-3, 256, wd, 1000 + e, dev)
        v /= args.n_ensemble
        report["strong_linear"].append(eval_scores(v, "B2 strong-linear", wd, extra=True))

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
