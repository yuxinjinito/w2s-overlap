#!/usr/bin/env python3
"""Robust linear (ridge) alignment residual -- the whiteboard's "ridge that throws out far
outliers", cross-fitted.

Per fold: closed-form ridge map W: weak -> strong on fold-train; drop the trim_frac train
points with the largest residual; REFIT on the kept points; score fold-TEST points by
||W_robust(h_w) - h_s|| (standardized target space). Label-free, like l2resid; the robust
refit removes the outlier-chasing that made the plain high-residual end poisonous.
"""
from __future__ import annotations

import numpy as np


def _fit_ridge(Xw: np.ndarray, Xs: np.ndarray, lam_frac: float):
    muw, mus = Xw.mean(0, keepdims=True), Xs.mean(0, keepdims=True)
    Xwc, Xsc = Xw - muw, Xs - mus
    G = Xwc.T @ Xwc / Xwc.shape[0]
    lam = lam_frac * (np.trace(G) / G.shape[0]) + 1e-12
    W = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xwc.T @ Xsc / Xwc.shape[0])
    return muw, mus, W


def robust_linear_residual_scores(
    weak_acts: np.ndarray,
    strong_acts: np.ndarray,
    n_folds: int = 5,
    lam_frac: float = 0.1,
    trim_frac: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Out-of-fold robust-ridge alignment residuals. Returns [n] non-negative scores."""
    Xw = np.asarray(weak_acts, dtype=np.float64)
    Xs = np.asarray(strong_acts, dtype=np.float64)
    n = Xw.shape[0]
    if Xs.shape[0] != n:
        raise ValueError("weak_acts and strong_acts must share the first dim n")
    sd = Xs.std(0, keepdims=True) + 1e-8
    Xs = Xs / sd  # standardize target so residual norms are scale-free

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    scores = np.zeros(n, dtype=np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for test_idx in folds:
            tr = np.setdiff1d(perm, test_idx, assume_unique=False)
            muw, mus, W = _fit_ridge(Xw[tr], Xs[tr], lam_frac)
            r_tr = np.linalg.norm((Xw[tr] - muw) @ W - (Xs[tr] - mus), axis=1)
            keep = r_tr <= np.quantile(r_tr, 1.0 - trim_frac)
            muw, mus, W = _fit_ridge(Xw[tr][keep], Xs[tr][keep], lam_frac)
            scores[test_idx] = np.linalg.norm((Xw[test_idx] - muw) @ W - (Xs[test_idx] - mus), axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("robust_linear_residual_scores produced non-finite values")
    return scores


def _self_test() -> None:
    rng = np.random.default_rng(0)
    n, dw, ds = 500, 24, 32
    hw = rng.standard_normal((n, dw))
    Wm = rng.standard_normal((dw, ds))
    sigma = np.where(np.arange(n) < n // 2, 0.2, 1.5)[:, None]
    hs = hw @ Wm + sigma * rng.standard_normal((n, ds))
    s = robust_linear_residual_scores(hw, hs, n_folds=4, seed=0)
    assert s.shape == (n,)
    # degenerate control: the score is scale-normalized, so "near zero" is not a
    # property it has. What IS analytic: under an exact linear map with exactly
    # ten corrupted rows, those ten must be the ten largest scores.
    hs_c = (hw @ Wm).copy()
    hs_c[:10] += 25.0 * rng.standard_normal((10, ds))
    s_c = robust_linear_residual_scores(hw, hs_c, n_folds=4, seed=0)
    assert set(np.argsort(-s_c)[:10].tolist()) == set(range(10)), "corrupted rows must top the ranking"
    lo, hi = s[: n // 2].mean(), s[n // 2:].mean()
    print(f"  low-noise mean {lo:.3f} vs high-noise mean {hi:.3f}")
    assert lo < hi, "robust residuals should still rank noisy points higher"
    print("ROBUST LINEAR RESIDUAL: SELF-TEST PASSED")


if __name__ == "__main__":
    _self_test()
