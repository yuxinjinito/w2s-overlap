#!/usr/bin/env python3
"""Screen for the open-book (in-sample, capped) MLP step 2, and the attribution
controls around it.

Design: step 1 is selectable (--step1 kernel = rp's own in-sample capped solve,
--step1 mlp = mlpstep1's cross-fitted wd-30 ensemble), and only the step-2
estimator varies within a run. Step-2 candidates fit a from the strong reps ON
THE SAME ROWS, cap swept over weight decay and width; "lin" in the hidden grid
runs a linear head at matched optimizer and geometry, which is the control that
separates the cap position from nonlinearity. The target is standardized inside
the fit, so the wd axis no longer depends on the residual's scale (tables made
before this change are not comparable row-for-row).

Screen metrics per candidate (screen-level only; tickets go downstream,
conclusions do not happen here):
  spearman vs ref       how much it behaves like the deployed kernel step 2
  spearman vs |a|       how much it collapses onto step-1-only ranking
  corr(v, a)            memorization dial: 1.0 means it just returned its target
  top-50% overlap vs the reference band, errAUROC, score std
Reference rows: the kernel step 2 itself, |a| (step 2 off), the same kernel on
z-scored features (the geometry the MLP sees), and the kernel's leave-one-out
read (what in-sample buys beyond self-inclusion).

Usage: python scripts/mlpstep2_screen.py --acts acts_r1_seed42.npz --csv table.csv
Ticket slots are chosen by the preregistered rule printed at the end.
"""
from __future__ import annotations

import argparse

import numpy as np

from compute_pgr import auroc, spearman
import torch
from torch import nn

from mlp_rp import _oof_regress


def _spearman(x, y):
    return spearman(x, y)


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
    def __init__(self, d_in: int, hidden):
        super().__init__()
        if hidden == "lin":
            self.net = nn.Linear(d_in, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, 1),
            )

    def forward(self, x):  # noqa: D102
        return self.net(x).squeeze(-1)


def insample_mlp_fit_read(Xs, a, hidden, wd, seed, epochs=60, lr=1e-3, bs=256, n_ens=3):
    """Fit heads to a on ALL rows and read predictions on the SAME rows (open book).

    The target is scaled to unit std before fitting and the predictions are
    scaled back, so the same wd means the same cap whatever the residual's
    scale. hidden may be an int or "lin" for the linear-head control."""
    dev = torch.device("cpu")
    mu = Xs.mean(0, keepdims=True); sd = Xs.std(0, keepdims=True) + 1e-8
    X = torch.tensor((Xs - mu) / sd, dtype=torch.float32, device=dev)
    scale = float(a.std()) + 1e-12
    y = torch.tensor(a / scale, dtype=torch.float32, device=dev)
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
    return (v / n_ens) * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--reg", type=float, default=0.1)
    # the corridor peaks around wd=10 on the pre-standardization instrument; my
    # first pass swept {.01, 3, 30} and stepped right over it, so the default is
    # denser now, orz
    ap.add_argument("--wd-grid", default="0.01,0.1,1,3,10,30",
                    help="comma-separated weight decays to sweep")
    ap.add_argument("--hidden-grid", default="256,64",
                    help="widths to sweep; 'lin' runs the linear-head control")
    ap.add_argument("--step1", choices=["kernel", "mlp"], default="kernel",
                    help="whose residual the step-2 candidates fit: rp's kernel "
                         "solve or mlpstep1's cross-fitted wd-30 ensemble")
    ap.add_argument("--step1-target", choices=["hard", "logit", "delta"], default="hard",
                    help="step-1 target: hard 0/1 weak labels; the probe's logit "
                         "difference (weak_margin if the npz has it, else "
                         "logit(clip(weak_probs))); or delta, standardized probe "
                         "evidence minus standardized zero-shot weak evidence "
                         "(needs --weak-prior-npz)")
    ap.add_argument("--weak-prior-npz", default="",
                    help="npz with weak_prior_margin (+ weak_preds guard) for the delta target")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--csv", default="", help="optional path to dump the screen table")
    args = ap.parse_args()

    d = np.load(args.acts)
    Xw = np.asarray(d["weak"], np.float64)
    Xs = np.asarray(d["strong"], np.float64)
    wp = np.asarray(d["weak_preds"]).astype(int)
    gt = np.asarray(d["gt"]).astype(int)
    n = len(wp)
    bad = wp != gt

    def probe_margin():
        if "weak_margin" in d.files:
            return np.asarray(d["weak_margin"], np.float64)
        probs = np.clip(np.asarray(d["weak_probs"], np.float64), 1e-3, 1 - 1e-3)
        return np.log(probs) - np.log1p(-probs)

    if args.step1_target == "logit":
        y = probe_margin()
    elif args.step1_target == "delta":
        # anti-expert construction: what supervision taught the probe, beyond
        # what the weak model already believed zero-shot. The two evidence
        # channels differ in scale, so both are standardized before subtracting.
        prior = np.load(args.weak_prior_npz)
        pw = np.asarray(prior["weak_preds"]).astype(int)
        if (pw == wp).mean() < 0.95:
            raise ValueError("weak-prior npz misaligned with the acts file")
        z_probe = probe_margin()
        z0 = np.asarray(prior["weak_prior_margin"], np.float64)
        zp = (z_probe - z_probe.mean()) / (z_probe.std() + 1e-12)
        z0 = (z0 - z0.mean()) / (z0.std() + 1e-12)
        y = zp - z0
    else:
        y = wp.astype(np.float64)
    yc = y - y.mean()

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        if args.step1 == "mlp":
            # mlpstep1's step 1, replicated with its shipped hyperparameters
            p = np.zeros(n, dtype=np.float64)
            for e in range(3):
                p += _oof_regress(np.asarray(Xw, np.float32), yc, 5, 256, 60, 1e-3, 256,
                                  30.0, args.seed + 77 + 131 * e, torch.device("cpu"))
            a = yc - p / 3
            ref_name = "ms1 (kernel)"
        else:
            Xwc = Xw - Xw.mean(0, keepdims=True)
            Kw = Xwc @ Xwc.T / n
            rw = args.reg * (np.trace(Kw) / n) + 1e-12
            a = yc - Kw @ np.linalg.solve(Kw + rw * np.eye(n), yc)
            ref_name = "rp (kernel)"
        # deployed step 2 (kernel) on this residual, for reference
        Xsc = Xs - Xs.mean(0, keepdims=True)
        Ks = Xsc @ Xsc.T / n
        rs = args.reg * (np.trace(Ks) / n) + 1e-12
        Kinv = np.linalg.inv(Ks + rs * np.eye(n))
        v_ref = Ks @ (Kinv @ a)
        # the same solve in the geometry the MLP sees (z-scored features)
        Xz = (Xs - Xs.mean(0, keepdims=True)) / (Xs.std(0, keepdims=True) + 1e-8)
        Kz = Xz @ Xz.T / n
        rz = args.reg * (np.trace(Kz) / n) + 1e-12
        v_z = Kz @ np.linalg.solve(Kz + rz * np.eye(n), a)
        # leave-one-out read of the kernel: in-sample minus self-inclusion
        H = Ks @ Kinv
        hd = np.clip(np.diag(H), None, 1 - 1e-8)
        v_loo = (v_ref - hd * a) / (1 - hd)
    if not np.isfinite(a).all() or not np.isfinite(v_ref).all():
        raise ValueError("non-finite step-1 residual or reference read; check the acts file")
    s_ref = np.abs(v_ref)
    s_a = np.abs(a)

    print(f"rows {n} | weak-correct {(~bad).mean():.3f} | step1 {args.step1} | target {args.step1_target}")
    print(f"{'config':16s} {'sp(ref)':>7s} {'sp(|a|)':>8s} {'corr(v,a)':>10s} {'ovl50':>6s} {'errAUC':>7s} {'std':>6s}")
    ref_auc, ref_auc_a = _auroc(s_ref, bad), _auroc(s_a, bad)
    print(f"{ref_name:16s} {1.0:7.3f} {_spearman(s_ref, s_a):8.3f} {'--':>10s} {1.0:6.2f} {ref_auc:7.3f} {s_ref.std():6.3f}")
    print(f"{'|a| (no step2)':16s} {_spearman(s_a, s_ref):7.3f} {1.0:8.3f} {'--':>10s} {_band_overlap(s_a, s_ref):6.2f} {ref_auc_a:7.3f} {s_a.std():6.3f}")
    for nm, vv in (("ridge z-scored", v_z), ("kernel LOO", v_loo)):
        ss = np.abs(vv)
        print(f"{nm:16s} {_spearman(ss, s_ref):7.3f} {_spearman(ss, s_a):8.3f} "
              f"{float(np.corrcoef(vv, a)[0, 1]):10.3f} {_band_overlap(ss, s_ref):6.2f} "
              f"{_auroc(ss, bad):7.3f} {ss.std():6.3f}")

    results = {}
    hiddens = [h.strip() for h in args.hidden_grid.split(",")]
    for hidden in [("lin" if h == "lin" else int(h)) for h in hiddens]:
        for wd in [float(w) for w in args.wd_grid.split(",")]:
            v = insample_mlp_fit_read(np.asarray(Xs, np.float32), a, hidden, wd,
                                      seed=args.seed, epochs=args.epochs)
            s = np.abs(v)
            row = dict(hidden=hidden, wd=wd,
                       sp_rp=_spearman(s, s_ref), sp_a=_spearman(s, s_a),
                       corr_va=float(np.corrcoef(v, a)[0, 1]),
                       ovl=_band_overlap(s, s_ref), auc=_auroc(s, bad), std=float(s.std()))
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

    # preregistered ticket rule: among non-degenerate configs (corr(v,a) < 0.95
    # and a score std that is not float dust), cs1 = highest spearman vs the
    # reference; cs2 = the non-degenerate config most different from it (lowest
    # 50% overlap) with errAUROC at least |a|'s.
    ok = {k: v for k, v in results.items()
          if v[1]["corr_va"] < 0.95 and v[1]["std"] > 1e-6}
    if not ok:
        print("all configs degenerate; no tickets")
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
