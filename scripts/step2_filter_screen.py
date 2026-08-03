#!/usr/bin/env python3
"""Screen for kernel-side step-2 variants that are not reparameterizations of
the ridge smoother: label-aware spectral filters and a robust-loss solve.

The ridge step 2 applies the fixed filter lam/(lam+rs) in the strong
eigenbasis, blind to where a's energy sits. The candidates here let a in:
  eb      empirical-Bayes filter c^2/(c^2+s^2) per eigendirection, s^2 from
          the median |c|^2 of the bottom half of the spectrum (not a knob)
  sure-q  soft-threshold on the coefficients c at the q-quantile of |c|
  huber-q IRLS kernel ridge with Huber weights, delta at the q-quantile of
          the rp residual |a - v_rp| (nests rp as delta -> inf)
Same table, gate, and ticket rule as mlpstep2_screen; screen-level only.

Usage: python scripts/step2_filter_screen.py --acts acts_r1_seed42.npz --csv table.csv
"""
from __future__ import annotations

import argparse

import numpy as np

from compute_pgr import auroc, spearman


def _band_overlap(s1, s2, frac=0.5):
    n = len(s1)
    k = int(n * frac)
    b1 = set(np.argsort(s1)[-k:]); b2 = set(np.argsort(s2)[-k:])
    return len(b1 & b2) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--reg", type=float, default=0.1)
    ap.add_argument("--sure-quantiles", default="0.5,0.7,0.9")
    ap.add_argument("--huber-quantiles", default="0.5,0.7,0.9")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    d = np.load(args.acts)
    Xw = np.asarray(d["weak"], np.float64)
    Xs = np.asarray(d["strong"], np.float64)
    wp = np.asarray(d["weak_preds"]).astype(int)
    gt = np.asarray(d["gt"]).astype(int)
    n = len(wp)
    bad = wp != gt

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Xwc = Xw - Xw.mean(0, keepdims=True)
        Kw = Xwc @ Xwc.T / n
        rw = args.reg * (np.trace(Kw) / n) + 1e-12
        yc = wp.astype(np.float64); yc -= yc.mean()
        a = yc - Kw @ np.linalg.solve(Kw + rw * np.eye(n), yc)
        Xsc = Xs - Xs.mean(0, keepdims=True)
        Ks = Xsc @ Xsc.T / n
        rs = args.reg * (np.trace(Ks) / n) + 1e-12
        v_rp = Ks @ np.linalg.solve(Ks + rs * np.eye(n), a)
        lam, U = np.linalg.eigh(Ks)
    if not np.isfinite(a).all() or not np.isfinite(v_rp).all():
        raise ValueError("non-finite step-1 residual or rp read; check the acts file")
    s_rp, s_a = np.abs(v_rp), np.abs(a)
    c = U.T @ a

    candidates = {}
    # empirical Bayes: shrink each coefficient by its own evidence
    bottom = np.sort(c ** 2)[: n // 2]
    s2 = float(np.median(bottom))
    candidates["eb"] = U @ (c * (c ** 2 / (c ** 2 + s2)))
    # SURE-style soft threshold on the coefficients
    for q in [float(x) for x in args.sure_quantiles.split(",")]:
        t = float(np.quantile(np.abs(c), q))
        candidates[f"sure-q{q:g}"] = U @ (np.sign(c) * np.maximum(np.abs(c) - t, 0.0))
    # Huber IRLS: bounded loss on the fit residual, kernel form
    r0 = np.abs(a - v_rp)
    for q in [float(x) for x in args.huber_quantiles.split(",")]:
        delta = float(np.quantile(r0, q)) + 1e-12
        w = np.ones(n)
        v = v_rp.copy()
        for _ in range(8):
            W_inv = 1.0 / np.maximum(w, 1e-8)
            v = Ks @ np.linalg.solve(Ks + rs * np.diag(W_inv), a)
            r = np.abs(a - v)
            w = np.minimum(1.0, delta / np.maximum(r, 1e-12))
        candidates[f"huber-q{q:g}"] = v

    print(f"rows {n} | weak-correct {(~bad).mean():.3f}")
    print(f"{'config':12s} {'sp(rp)':>7s} {'sp(|a|)':>8s} {'corr(v,a)':>10s} {'ovl50':>6s} {'errAUC':>7s} {'std':>6s}")
    ref_auc_rp, ref_auc_a = auroc(s_rp, bad.astype(int)), auroc(s_a, bad.astype(int))
    print(f"{'rp (kernel)':12s} {1.0:7.3f} {spearman(s_rp, s_a):8.3f} {'--':>10s} {1.0:6.2f} {ref_auc_rp:7.3f} {s_rp.std():6.3f}")
    print(f"{'|a|':12s} {spearman(s_a, s_rp):7.3f} {1.0:8.3f} {'--':>10s} {_band_overlap(s_a, s_rp):6.2f} {ref_auc_a:7.3f} {s_a.std():6.3f}")

    results = {}
    for name, v in candidates.items():
        s = np.abs(v)
        row = dict(config=name,
                   sp_rp=spearman(s, s_rp), sp_a=spearman(s, s_a),
                   corr_va=float(np.corrcoef(v, a)[0, 1]),
                   ovl=_band_overlap(s, s_rp), auc=auroc(s, bad.astype(int)),
                   std=float(s.std()))
        results[name] = (s, row)
        print(f"{name:12s} {row['sp_rp']:7.3f} {row['sp_a']:8.3f} {row['corr_va']:10.3f} "
              f"{row['ovl']:6.2f} {row['auc']:7.3f} {row['std']:6.3f}", flush=True)

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(next(iter(results.values()))[1].keys()))
            w.writeheader()
            for _, row in results.values():
                w.writerow(row)
        print(f"wrote {args.csv}")

    # ticket rule for filter candidates: non-degenerate, meaningfully different
    # from rp (sp_rp < .95), and an error direction no worse than rp's. The first
    # version of this gate demanded auc >= |a|'s, which is the MLP screen's
    # diverse-ticket test transplanted with its sign backwards for this family (._.)
    ok = {k: v for k, v in results.items()
          if v[1]["corr_va"] < 0.95 and v[1]["std"] > 1e-6
          and v[1]["sp_rp"] < 0.95 and v[1]["auc"] <= ref_auc_rp}
    if not ok:
        print("\nno candidate clears the gate; no tickets")
        return
    cs1 = min(ok, key=lambda k: ok[k][1]["sp_rp"])
    print(f"\nticket: cs1 = {cs1}")
    if args.out:
        np.savez(args.out, weak_preds=wp, cs1=results[cs1][0])
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
