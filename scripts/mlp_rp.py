#!/usr/bin/env python3
# 2026-07-09, World Cup quarterfinal day: France 2-0 Morocco in Boston.
"""MLP twists on the rp score. Same two-solve skeleton, I swap one solve at a time
for a small MLP and let the downstream runs judge.

The skeleton (the original lives in representation_projection.py):

  step 1:  a = (I - P_w) yc       # what the weak rep cannot explain about its labels
  step 2:  v = P_s a              # how much of that leftover the strong rep claims
  score_i = |v_i|

Three variants in this file:

  mlp_rp_scores        step 2 becomes a cross-fitted MLP ensemble. The first thing
                       we tried. Spoiler: loses to rp, because the strong side is
                       not a prediction problem (step-2 note has the full story).
  mlp_step1_rp_scores  step 1 becomes the MLP, with weight decay 30: I need a
                       smoother there, not a memorizer, and the wd sweep picked
                       it. This is the variant that beats rp on ANLI-R1 and WANLI.
  linstep1_rp_scores   the referee. Cross-fitted but linear. If this ties rp, the
                       credit of mlpstep1 is the nonlinearity, not the cross-fitting.
                       (It ties rp. Credit assigned.)

House rule I keep relearning the hard way: whatever fits where it predicts needs a
capacity cap, and the strong side must stay in-sample. Every variant here obeys it
except mlp_rp_scores, which is exactly why mlp_rp_scores lost.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _zfit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return mu, sd


class _RegMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # noqa: D102
        return self.net(x).squeeze(-1)


def _oof_regress(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    seed: int,
    dev: torch.device,
) -> np.ndarray:
    """Out-of-fold MLP regression: every row gets predicted by a model that never saw it.
    Inputs are z-scored per fold (stats from the train folds only, no peeking)."""
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
        # global torch seeding, kept: a per-call Generator would change the draw
        # stream and with it every recorded mlpstep1 number
        torch.manual_seed(seed * 1000 + fi)
        model = _RegMLP(X.shape[1], hidden).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        m = Xtr.shape[0]
        model.train()
        for _ in range(epochs):
            order = torch.randperm(m, device=dev)
            for b0 in range(0, m, batch_size):
                idx = order[b0:b0 + batch_size]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            preds[test_idx] = model(Xte).double().cpu().numpy()
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return preds


def mlp_rp_scores(
    weak_acts: np.ndarray,
    strong_acts: np.ndarray,
    weak_labels: np.ndarray,
    n_folds: int = 5,
    hidden: int = 256,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 1e-2,
    seed: int = 0,
    device: str | None = None,
    ridge_reg: float = 0.1,
    n_ensemble: int = 3,
) -> np.ndarray:
    """rp with an MLP STEP 2: |g_s^oof(h_s)| where the ensemble g_s fits the weak
    residual a out-of-fold. Kept for the record; the in-sample rule says this one
    should lose, and it does. Returns [n] non-negative scores."""
    Xw = np.asarray(weak_acts, dtype=np.float32)
    Xs = np.asarray(strong_acts, dtype=np.float32)
    y = np.asarray(weak_labels, dtype=np.float64)
    n = Xw.shape[0]
    if Xs.shape[0] != n or y.shape[0] != n:
        raise ValueError("weak_acts, strong_acts, weak_labels must share the first dim n")
    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    yc = y - y.mean()
    # step 1: kernel-ridge weak-side smoother, byte-identical to rp's
    Xw64 = np.asarray(Xw, dtype=np.float64)
    Xw64 = Xw64 - Xw64.mean(axis=0, keepdims=True)
    Kw = Xw64 @ Xw64.T / n
    rw = ridge_reg * (np.trace(Kw) / n) + 1e-12
    a = yc - Kw @ np.linalg.solve(Kw + rw * np.eye(n), yc)
    # step 2: cross-fitted MLP ensemble on the residual (3 nets, single MLPs are moody)
    v = np.zeros(n, dtype=np.float64)
    for e in range(n_ensemble):
        v += _oof_regress(Xs, a, n_folds, hidden, epochs, lr, batch_size, weight_decay,
                          seed + 77 + 131 * e, dev)
    v /= n_ensemble
    scores = np.abs(v)
    if not np.isfinite(scores).all():
        raise ValueError("mlp_rp_scores produced non-finite values (diverged fold)")
    return scores


def mlp_step1_rp_scores(
    weak_acts: np.ndarray,
    strong_acts: np.ndarray,
    weak_labels: np.ndarray,
    n_folds: int = 5,
    hidden: int = 256,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 30.0,
    seed: int = 0,
    device: str | None = None,
    ridge_reg: float = 0.1,
    n_ensemble: int = 3,
) -> np.ndarray:
    """rp with an MLP STEP 1 (the good one). The weak-side smoother becomes a heavily
    weight-decayed MLP ensemble, still cross-fitted; the strong-side projection stays
    rp's in-sample kernel ridge. wd=30 comes from the 07/22 sweep, and in ridge language
    it is just the lambda knob turned way up so the net smooths instead of memorizing.
    Returns [n] scores."""
    Xw = np.asarray(weak_acts, dtype=np.float32)
    y = np.asarray(weak_labels, dtype=np.float64)
    n = Xw.shape[0]
    if strong_acts.shape[0] != n or y.shape[0] != n:
        raise ValueError("weak_acts, strong_acts, weak_labels must share the first dim n")
    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    yc = y - y.mean()
    # step 1: cross-fitted MLP ensemble instead of the weak-side ridge
    p = np.zeros(n, dtype=np.float64)
    for e in range(n_ensemble):
        p += _oof_regress(Xw, yc, n_folds, hidden, epochs, lr, batch_size, weight_decay,
                          seed + 77 + 131 * e, dev)
    p /= n_ensemble
    a = yc - p
    # step 2: rp's strong-side kernel ridge, in-sample
    Xs64 = np.asarray(strong_acts, dtype=np.float64)
    Xs64 = Xs64 - Xs64.mean(axis=0, keepdims=True)
    Ks = Xs64 @ Xs64.T / n
    rs = ridge_reg * (np.trace(Ks) / n) + 1e-12
    v = Ks @ np.linalg.solve(Ks + rs * np.eye(n), a)
    scores = np.abs(v)
    if not np.isfinite(scores).all():
        raise ValueError("mlp_step1_rp_scores produced non-finite values (diverged fold)")
    return scores


def linstep1_rp_scores(
    weak_acts: np.ndarray,
    strong_acts: np.ndarray,
    weak_labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 0,
    ridge_reg: float = 0.1,
) -> np.ndarray:
    """The referee: rp's own step-1 kernel ridge, just fitted out-of-fold like mlpstep1;
    strong side unchanged. Differs from rp only in the cross-fitting, differs from
    mlpstep1 only in linearity, so whoever it matches gets the blame or the credit.
    Closed-form, no optimizer, runs in seconds."""
    Xw = np.asarray(weak_acts, dtype=np.float64)
    y = np.asarray(weak_labels, dtype=np.float64)
    n = len(y)
    if Xw.shape[0] != n or strong_acts.shape[0] != n:
        raise ValueError("weak_acts, strong_acts, weak_labels must share the first dim n")
    yc = y - y.mean()
    Xwc = Xw - Xw.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    preds = np.zeros(n, dtype=np.float64)
    for te in np.array_split(perm, n_folds):
        tr = np.setdiff1d(perm, te, assume_unique=False)
        m = len(tr)
        Ktr = Xwc[tr] @ Xwc[tr].T / m
        r = ridge_reg * (np.trace(Ktr) / m) + 1e-12
        mu_tr = yc[tr].mean()
        alpha = np.linalg.solve(Ktr + r * np.eye(m), yc[tr] - mu_tr)
        preds[te] = (Xwc[te] @ Xwc[tr].T / m) @ alpha + mu_tr
    a = yc - preds
    Xsc = np.asarray(strong_acts, dtype=np.float64)
    Xsc = Xsc - Xsc.mean(axis=0, keepdims=True)
    Ks = Xsc @ Xsc.T / n
    rs = ridge_reg * (np.trace(Ks) / n) + 1e-12
    v = Ks @ np.linalg.solve(Ks + rs * np.eye(n), a)
    scores = np.abs(v)
    if not np.isfinite(scores).all():
        raise ValueError("linstep1_rp_scores produced non-finite values")
    return scores


def _self_test() -> None:
    rng = np.random.default_rng(0)
    n, dw, ds = 900, 24, 32

    # Toy world: factor z visible to BOTH reps, factor q only to the STRONG rep.
    # Label y = sign(0.3 z + q). Points where q dominates are unexplainable by the
    # weak rep but expressible by the strong one, so they should score HIGH.
    from scipy.special import erf as _erf

    z = rng.standard_normal(n)
    q = rng.standard_normal(n)
    y = ((0.3 * z + q) > 0).astype(float)

    hw = np.column_stack([z, rng.standard_normal((n, dw - 1))]).astype(np.float32)
    hs = np.column_stack([z, q, rng.standard_normal((n, ds - 2))]).astype(np.float32)

    # ground truth for the toy: best weak-side predictor is E[y|z] = Phi(0.3 z),
    # so the true weak-unexplainable residual is computable exactly \o/
    e_y_z = 0.5 * (1.0 + _erf(0.3 * z / np.sqrt(2.0)))
    a_true = (y - y.mean()) - (e_y_z - e_y_z.mean())
    target_hi = np.abs(a_true) > np.median(np.abs(a_true))

    def _auroc(sc, pos_mask):
        return auroc(sc, pos_mask.astype(int))

    from compute_pgr import auroc, spearman
    from representation_projection import representation_projection_scores

    s = mlp_rp_scores(hw, hs, y, n_folds=4, seed=0, device="cpu")
    assert s.shape == (n,), "shape failed"
    s_lin = representation_projection_scores(hw, hs, y)
    auc_mlp, auc_lin = _auroc(s, target_hi), _auroc(s_lin, target_hi)
    print(f"  AUROC vs |a_true|: mlprp {auc_mlp:.3f} | linear rp {auc_lin:.3f} (informational)")
    # sanity gates only; whether a ranking helps TRAINING is for the real runs to say
    corr = spearman(s, s_lin)
    print(f"  spearman(mlprp, linear rp) = {corr:+.3f}  score std = {s.std():.4f}")
    assert s.std() > 1e-4, "degenerate (constant) scores"
    assert corr > 0.15, f"mlprp should broadly track the linear rp signal, got {corr:+.3f}"

    # control: constant labels -> nothing to explain -> nothing to score
    s0 = mlp_rp_scores(hw, hs, np.ones(n), n_folds=4, hidden=128, epochs=60, seed=1, device="cpu")
    print(f"  constant-y score mean (should be ~0): {s0.mean():.4f}")
    assert s0.mean() < 0.15, "constant labels should give near-zero scores"

    print("MLP-RP SCORE: SELF-TEST PASSED")


if __name__ == "__main__":
    _self_test()
