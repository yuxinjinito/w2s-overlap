#!/usr/bin/env python3
"""Screen for the open-book (in-sample, capped) MLP step 2 -- the one legal cell the
step-2 story left untested.

Design: step 1 stays rp's own in-sample capped kernel solve (byte-identical), so this
is a one-factor test of the step-2 estimator. Step 2 candidates fit a = (I - P_w) yc
from the strong reps ON THE SAME ROWS, with the cap swept: hidden width {256, 64} x
weight decay {0.01, 3, 30}. Score = |prediction|.

Screen metrics per candidate (screen-level only; tickets go downstream, conclusions
do not happen here):
  spearman vs rp        how much it behaves like the deployed kernel step 2
  spearman vs |a|       how much it collapses onto step-1-only ranking
  corr(v, a)            memorization dial: 1.0 means it just returned its target
  top-50% overlap vs rp band, errAUROC vs row weak-label correctness, score std
Reference rows: rp itself and the strong-side-off null (score = |a|).

Usage: python scripts/mlpstep2_screen.py --acts acts_r1_seed42.npz --out custom_scores_mlpstep2_r1.npz
Ticket slots are chosen by the preregistered rule printed at the end.
"""
from __future__ import annotations

import argparse

import numpy as np

from compute_pgr import auroc
import torch
from torch import nn


def _spearman(x, y):
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    return float(np.corrcoef(rk(x), rk(y))[0, 1])


def _auroc(v, bad):
    # the L1 original tie-averages; my first version here did not, and ties
    # made the number depend on row order
    return auroc(v, bad.astype(int))


def _band_overlap(s1, s2, frac=0.5):
    n = len(s1)
    k = int(n * frac)
    b1 = set(np.argsort(s1)[-k:]); b2 = set(np.argsort(s2)[-k:])
    return len(b1 & b2) / k


class _Head(nn.Module):
    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # noqa: D102
        return self.net(x).squeeze(-1)


def insample_mlp_fit_read(Xs, a, hidden, wd, seed, epochs=60, lr=1e-3, bs=256, n_ens=3):
    """Fit MLPs to a on ALL rows and read predictions on the SAME rows (open book)."""
    dev = torch.device("cpu")
    mu = Xs.mean(0, keepdims=True); sd = Xs.std(0, keepdims=True) + 1e-8
    X = torch.tensor((Xs - mu) / sd, dtype=torch.float32, device=dev)
    y = torch.tensor(a, dtype=torch.float32, device=dev)
    n = X.shape[0]
    v = np.zeros(n, dtype=np.float64)
    for e in range(n_ens):
        torch.manual_seed(seed + 1000 * e)
        model = _Head(X.shape[1], hidden).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(epochs):
            order = torch.randperm(n)
            for b0 in range(0, n, bs):
                idx = order[b0:b0 + bs]
                opt.zero_grad(set_to_none=True)
                loss_fn(model(X[idx]), y[idx]).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            v += model(X).double().numpy()
    return v / n_ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--reg", type=float, default=0.1)
    # the corridor peaks around wd=10; my first pass swept {.01, 3, 30} and stepped
    # right over it, so the default is denser now, orz
    ap.add_argument("--wd-grid", default="0.01,0.1,1,3,10,30",
                    help="comma-separated weight decays to sweep")
    ap.add_argument("--hidden-grid", default="256,64")
    ap.add_argument("--csv", default="", help="optional path to dump the screen table")
    args = ap.parse_args()

    d = np.load(args.acts)
    Xw = np.asarray(d["weak"], np.float64)
    Xs = np.asarray(d["strong"], np.float64)
    wp = np.asarray(d["weak_preds"]).astype(int)
    gt = np.asarray(d["gt"]).astype(int)
    n = len(wp)
    bad = wp != gt

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        # step 1: rp's in-sample capped kernel solve, unchanged
        Xwc = Xw - Xw.mean(0, keepdims=True)
        Kw = Xwc @ Xwc.T / n
        rw = args.reg * (np.trace(Kw) / n) + 1e-12
        yc = wp.astype(np.float64); yc -= yc.mean()
        a = yc - Kw @ np.linalg.solve(Kw + rw * np.eye(n), yc)
        # deployed step 2 (kernel) for reference
        Xsc = Xs - Xs.mean(0, keepdims=True)
        Ks = Xsc @ Xsc.T / n
        rs = args.reg * (np.trace(Ks) / n) + 1e-12
        v_rp = Ks @ np.linalg.solve(Ks + rs * np.eye(n), a)
    s_rp = np.abs(v_rp)
    s_a = np.abs(a)

    print(f"rows {n} | weak-correct {(~bad).mean():.3f}")
    print(f"{'config':16s} {'sp(rp)':>7s} {'sp(|a|)':>8s} {'corr(v,a)':>10s} {'ovl50':>6s} {'errAUC':>7s} {'std':>6s}")
    ref_auc_rp, ref_auc_a = _auroc(s_rp, bad), _auroc(s_a, bad)
    print(f"{'rp (kernel)':16s} {1.0:7.3f} {_spearman(s_rp, s_a):8.3f} {'--':>10s} {1.0:6.2f} {ref_auc_rp:7.3f} {s_rp.std():6.3f}")
    print(f"{'|a| (no step2)':16s} {_spearman(s_a, s_rp):7.3f} {1.0:8.3f} {'--':>10s} {_band_overlap(s_a, s_rp):6.2f} {ref_auc_a:7.3f} {s_a.std():6.3f}")

    results = {}
    for hidden in [int(h) for h in args.hidden_grid.split(",")]:
        for wd in [float(w) for w in args.wd_grid.split(",")]:
            v = insample_mlp_fit_read(np.asarray(Xs, np.float32), a, hidden, wd, seed=7)
            s = np.abs(v)
            row = dict(hidden=hidden, wd=wd,
                       sp_rp=_spearman(s, s_rp), sp_a=_spearman(s, s_a),
                       corr_va=float(np.corrcoef(v, a)[0, 1]),
                       ovl=_band_overlap(s, s_rp), auc=_auroc(s, bad), std=float(s.std()))
            results[f"h{hidden}_wd{wd:g}"] = (s, row)
            print(f"{f'h{hidden}_wd{wd:g}':16s} {row['sp_rp']:7.3f} {row['sp_a']:8.3f} "
                  f"{row['corr_va']:10.3f} {row['ovl']:6.2f} {row['auc']:7.3f} {row['std']:6.3f}",
                  flush=True)
    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(next(iter(results.values()))[1].keys()))
            w.writeheader()
            for _, row in results.values():
                w.writerow(row)
        print(f"wrote {args.csv}")

    # preregistered ticket rule: among non-degenerate configs (corr(v,a) < 0.95),
    # cs1 = highest spearman vs rp (the best-behaved open-book MLP step 2);
    # cs2 = the non-degenerate config most different from rp (lowest 50% overlap)
    # with errAUROC at least |a|'s (so "different" is not just "broken").
    ok = {k: v for k, v in results.items() if v[1]["corr_va"] < 0.95}
    if not ok:
        print("all configs degenerate (corr(v,a) >= .95); no tickets")
        return
    cs1 = max(ok, key=lambda k: ok[k][1]["sp_rp"])
    div = {k: v for k, v in ok.items() if v[1]["auc"] >= ref_auc_a and k != cs1}
    cs2 = min(div, key=lambda k: div[k][1]["ovl"]) if div else None
    print(f"\ntickets: cs1 = {cs1}" + (f", cs2 = {cs2}" if cs2 else " (no diverse ticket)"))

    if args.out:
        out = {"weak_preds": wp, "cs1": results[cs1][0]}
        if cs2:
            out["cs2"] = results[cs2][0]
        np.savez(args.out, **out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
