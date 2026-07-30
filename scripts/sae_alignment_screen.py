#!/usr/bin/env python3
"""Proper-fidelity screen of the SAE-basis proposal (07/24): train SAEs on bulk acts
(interp-style: overcomplete dicts, dead-feature resampling, two L1s), encode the LABELED
acts through them, then run the alignment suite in code space:

  linear residual | PCA-k + linear residual | CCA on PCA-256 | sae-rp (label-conditioned)

against the raw-acts references (rp, l2resid family baseline). Metrics: weak-error
separation |AUROC - 0.5| and agreement with raw rp. Everything offline; downstream arms
only if a variant clears the preregistered bar.

Usage:
  PYTHONPATH=scripts python3 scripts/sae_alignment_screen.py \
      --bulk bulk_acts_r1.npz --labeled acts_r1_seed42.npz --out sae_screen_proper.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from compute_pgr import auroc as pgr_auroc
import torch
from torch import nn


class SAE(nn.Module):
    def __init__(self, dim: int, m: int):
        super().__init__()
        self.enc = nn.Linear(dim, m)
        self.dec = nn.Linear(m, dim)

    def forward(self, x):  # noqa: D102
        z = torch.relu(self.enc(x))
        return z, self.dec(z)


def train_sae(X: np.ndarray, m: int, l1: float, device, epochs=60, bs=1024, lr=1e-3,
              seed=0, resample_every=20):
    n, dim = X.shape
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    sae = SAE(dim, m).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for b0 in range(0, n, bs):
            idx = perm[b0:b0 + bs]
            opt.zero_grad(set_to_none=True)
            z, xh = sae(Xt[idx])
            loss = ((xh - Xt[idx]) ** 2).mean() + l1 * z.abs().mean()
            loss.backward(); opt.step()
        if resample_every and (ep + 1) % resample_every == 0 and ep + 1 < epochs:
            with torch.no_grad():
                zp, _ = sae(Xt[:4096])
                dead = (zp > 0).float().mean(0) < 1e-4
                if dead.any():
                    ndead = int(dead.sum())
                    nn.init.kaiming_uniform_(sae.enc.weight[dead])
                    sae.enc.bias[dead] = 0.0
                    sae.dec.weight[:, dead] = torch.randn_like(sae.dec.weight[:, dead]) * 0.01
                    print(f"    ep{ep+1}: resampled {ndead} dead features", flush=True)
    sae.eval()
    with torch.no_grad():
        z, xh = sae(Xt[:4096])
        stats = {"rec_mse": float(((xh - Xt[:4096]) ** 2).mean()),
                 "sparsity": float((z == 0).float().mean()),
                 "dead_frac": float(((z > 0).float().mean(0) < 1e-4).float().mean())}
    return sae, mu, sd, stats


@torch.no_grad()
def encode(sae: SAE, X: np.ndarray, mu, sd, device) -> np.ndarray:
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
    z, _ = sae(Xt)
    return z.double().cpu().numpy()


def rank(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r


def smooth(K, t, reg=0.1):
    n = K.shape[0]
    r = reg * (np.trace(K) / n) + 1e-12
    return K @ np.linalg.solve(K + r * np.eye(n), t)


def oof_linear_residual(Zw, Zs, n_folds=5, seed=0, lam_frac=0.1):
    n = Zw.shape[0]
    sd = Zs.std(0, keepdims=True) + 1e-8
    Zs = Zs / sd
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    resid = np.zeros(n)
    for te in np.array_split(perm, n_folds):
        tr = np.setdiff1d(perm, te)
        muw, mus = Zw[tr].mean(0, keepdims=True), Zs[tr].mean(0, keepdims=True)
        G = (Zw[tr] - muw).T @ (Zw[tr] - muw) / len(tr)
        lam = lam_frac * np.trace(G) / G.shape[0] + 1e-12
        W = np.linalg.solve(G + lam * np.eye(G.shape[0]),
                            (Zw[tr] - muw).T @ (Zs[tr] - mus) / len(tr))
        resid[te] = np.linalg.norm((Zw[te] - muw) @ W - (Zs[te] - mus), axis=1)
    return resid


def pca_reduce(Z, k):
    Zc = Z - Z.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Zc, full_matrices=False)
    return U[:, :k] * S[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bulk", required=True)
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--out", default="sae_screen_proper.json")
    ap.add_argument("--l1-grid", default="1e-3,3e-3")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    bulk = np.load(args.bulk)
    lab = np.load(args.labeled)
    Xw_l = lab["weak"].astype(np.float64); Xs_l = lab["strong"].astype(np.float64)
    y = lab["weak_preds"].astype(np.float64); yc = y - y.mean()
    wrong = lab["weak_preds"].astype(int) != lab["gt"].astype(int)
    n = len(y)

    def _au(sc):
        return pgr_auroc(sc, wrong.astype(int))

    # raw references
    Xwc = Xw_l - Xw_l.mean(0, keepdims=True); Xsc = Xs_l - Xs_l.mean(0, keepdims=True)
    rp_raw = np.abs(smooth(Xsc @ Xsc.T / n, yc - smooth(Xwc @ Xwc.T / n, yc)))
    report = {"raw_rp_auroc": _au(rp_raw),
              "raw_l2resid_auroc": _au(oof_linear_residual(Xw_l, Xs_l)), "rows": []}
    print(f"raw rp AUROC={report['raw_rp_auroc']:.3f}  raw l2resid AUROC={report['raw_l2resid_auroc']:.3f}")

    def add(name, s):
        sp = float(np.corrcoef(rank(s), rank(rp_raw))[0, 1])
        ov = len(set(np.argsort(-s)[:n//2]) & set(np.argsort(-rp_raw)[:n//2])) / (n // 2)
        row = {"name": name, "auroc": _au(s), "spearman_vs_rp": sp, "top50_overlap": ov}
        report["rows"].append(row)
        print(f"{name:34s} AUROC={row['auroc']:.3f} spearman={sp:.3f} overlap={ov:.3f}")

    for l1 in [float(x) for x in args.l1_grid.split(",")]:
        print(f"== L1={l1}")
        sae_w, muw, sdw, st_w = train_sae(bulk["weak"].astype(np.float32), 8 * 896, l1, device, seed=1)
        sae_s, mus, sds, st_s = train_sae(bulk["strong"].astype(np.float32), 4 * 3584, l1, device, seed=2)
        print(f"  SAE weak {st_w} | strong {st_s}")
        Zw = encode(sae_w, Xw_l.astype(np.float32), muw, sdw, device)
        Zs = encode(sae_s, Xs_l.astype(np.float32), mus, sds, device)
        Zw -= Zw.mean(0, keepdims=True); Zs -= Zs.mean(0, keepdims=True)

        add(f"linear-resid l1={l1}", oof_linear_residual(Zw, Zs))
        for k in (64, 256):
            add(f"pca{k}-resid l1={l1}", oof_linear_residual(pca_reduce(Zw, k), pca_reduce(Zs, k)))
        try:
            from cca_alignment import cca_alignment_scores
            add(f"cca-pca256 l1={l1}",
                cca_alignment_scores(pca_reduce(Zw, 256), pca_reduce(Zs, 256), n_folds=5, seed=0))
        except Exception as e:  # noqa: BLE001
            print(f"  cca skipped: {e}")
        Kw = Zw @ Zw.T / n; Ks = Zs @ Zs.T / n
        add(f"sae-rp l1={l1}", np.abs(smooth(Ks, yc - smooth(Kw, yc))))

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
