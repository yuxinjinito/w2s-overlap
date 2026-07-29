#!/usr/bin/env python3
"""LDA package: mentor-expectation audit remediation screen.

Covers the three items the first LDA implementation left open:
  A. fair shake for the deployed construction (residual onto the strong LDA axis):
     shrinkage gamma and the underlying map's reg swept instead of fixed;
  B. the literal reading of ``alignment between the LDA projections'': project each side
     onto its own LDA axis, align the two 1-D label coordinates out-of-fold, score by the
     disagreement residual;
  C. the meeting's second ask, never implemented: MLP-align trained with an LDA-style loss
     term on the outputs (penalize low between-cluster variance), weak labels only.

Tickets (preregistered): cs1 = best-A, cs2 = best-B, cs3/cs4 = the two C variants.
Written to a custom-scores npz for the downstream stage.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from lda_alignment import _fit_ridge_map, _lda_axis
from mlp_alignment import mlp_alignment_scores


def rankv(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="acts_r1_seed42.npz")
    ap.add_argument("--out-scores", default="custom_scores_lda_r1.npz")
    ap.add_argument("--out-report", default="lda_screen_report.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    lab = np.load(args.labeled)
    Xw = lab["weak"].astype(np.float64)
    Xs = lab["strong"].astype(np.float64)
    y = lab["weak_preds"].astype(int)
    wrong = y != lab["gt"].astype(int)
    n = len(y)

    def sep_of(sc):
        rr = rankv(sc) + 1; npos = wrong.sum()
        au = float((rr[wrong].sum() - npos * (npos + 1) / 2) / (npos * (n - npos)))
        return au, abs(au - 0.5)

    # rp reference for overlap
    Xwc = Xw - Xw.mean(0, keepdims=True); Xsc = Xs - Xs.mean(0, keepdims=True)
    def smooth(K, t, reg=0.1):
        r = reg * np.trace(K) / n + 1e-12
        return K @ np.linalg.solve(K + r * np.eye(n), t)
    yc = y.astype(np.float64) - y.mean()
    rp = np.abs(smooth(Xsc @ Xsc.T / n, yc - smooth(Xwc @ Xwc.T / n, yc)))
    top_rp = set(np.argsort(-rp)[: n // 2].tolist())

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    folds = np.array_split(perm, 3 if args.smoke else 5)

    rows, vectors = [], {}

    def add(name, sc, extra=None):
        au, sp = sep_of(sc)
        ov = len(set(np.argsort(-sc)[: n // 2].tolist()) & top_rp) / (n // 2)
        rows.append({"name": name, "auroc": au, "sep": sp, "overlap": ov, **(extra or {})})
        vectors[len(rows) - 1] = sc
        print(f"  {name:38s} auroc={au:.3f} sep={sp:.3f} overlap={ov:.3f}", flush=True)

    GAMMAS = (0.03, 0.1, 0.3) if args.smoke else (0.01, 0.03, 0.1, 0.3, 1.0)
    REGS = (0.1,) if args.smoke else (0.03, 0.1, 0.3)

    # A. deployed construction, fair-shake grid
    sd = Xs.std(0, keepdims=True) + 1e-8
    Xs_n = Xs / sd
    for reg in REGS:
        r_vec = np.zeros((n, Xs.shape[1]))
        axes_cache = {}
        for te in folds:
            tr = np.setdiff1d(perm, te)
            muw, mus, W = _fit_ridge_map(Xw[tr], Xs_n[tr], reg)
            r_vec[te] = (Xs_n[te] - mus) - (Xw[te] - muw) @ W
            axes_cache[te.tobytes()] = (tr, mus)
        for g in GAMMAS:
            sc = np.zeros(n)
            for te in folds:
                tr, mus = axes_cache[te.tobytes()]
                w = _lda_axis(Xs_n[tr] - mus, y[tr], g)
                sc[te] = np.abs(r_vec[te] @ w)
            add(f"A strong-axis g={g} reg={reg}", sc, {"family": "A"})

    # B. literal reading: align the two 1-D LDA coordinates, score the disagreement
    for g in GAMMAS:
        sc = np.zeros(n)
        for te in folds:
            tr = np.setdiff1d(perm, te)
            ww = _lda_axis(Xw[tr] - Xw[tr].mean(0), y[tr], g)
            ws = _lda_axis(Xs[tr] - Xs[tr].mean(0), y[tr], g)
            mw = (Xw - Xw[tr].mean(0)) @ ww
            ms = (Xs - Xs[tr].mean(0)) @ ws
            mw_z = (mw - mw[tr].mean()) / (mw[tr].std() + 1e-9)
            ms_z = (ms - ms[tr].mean()) / (ms[tr].std() + 1e-9)
            beta = float((mw_z[tr] * ms_z[tr]).mean())
            sc[te] = np.abs(ms_z[te] - beta * mw_z[te])
        add(f"B both-axes g={g}", sc, {"family": "B"})

    # C. LDA-loss MLP-align (the meeting's second ask), two lambdas + plain control
    ep = 5 if args.smoke else 150
    for lam in (0.0, 0.3, 3.0):
        sc = mlp_alignment_scores(Xw.astype(np.float32), Xs.astype(np.float32),
                                  n_folds=3 if args.smoke else 5, epochs=ep, seed=42,
                                  device=args.device, lda_weight=lam, lda_labels=y)
        add(f"C lda-loss mlpalign lam={lam}", sc, {"family": "C"})

    order = sorted(range(len(rows)), key=lambda i: -rows[i]["sep"])
    def best(fam):
        c = [i for i in order if rows[i]["family"] == fam]
        return c[0] if c else None
    tickets = [best("A"), best("B")]
    c_idx = sorted([i for i in range(len(rows)) if rows[i]["family"] == "C" and "lam=0.0" not in rows[i]["name"]],
                   key=lambda i: -rows[i]["sep"])
    tickets += c_idx[:2]
    tickets = [t for t in tickets if t is not None]

    out = {"weak_preds": lab["weak_preds"]}
    tinfo = []
    for slot, i in enumerate(tickets, 1):
        out[f"cs{slot}"] = vectors[i]
        tinfo.append({"slot": f"cs{slot}", **rows[i]})
        print(f"TICKET cs{slot}: {rows[i]}")
    np.savez_compressed(args.out_scores, **out)
    with open(args.out_report, "w") as f:
        json.dump({"rows": rows, "tickets": tinfo}, f, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
