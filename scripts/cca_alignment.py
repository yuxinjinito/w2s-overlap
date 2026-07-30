#!/usr/bin/env python3
"""CCA alignment-residual selection score (the "CCA" alternative alignment method).

Regularized linear CCA between weak and strong representations, cross-fitted:
per fold, center + (shrinkage-regularized, SVD-truncated) whiten both views on the
fold-train points, take the top-k canonical directions from SVD(Zw^T Zs / m), then score
each fold-TEST point by its distance in the shared canonical space,

    score_i = || U_k^T zw_i  -  V_k^T zs_i ||_2 ,

i.e. how far the point's weak and strong canonical coordinates disagree. Small = the two
views agree on this point (well-aligned); large = the strong view carries structure the
weak view cannot express here. keep-high vs keep-low decided empirically (two bands).

Label-free, like the MLP alignment score. Cross-fit keeps the directions honest.
"""
from __future__ import annotations

import numpy as np


def _whitener(X: np.ndarray, shrink: float, energy: float = 0.999):
    """Return (mean, W) with W built on the fold-train points: PCA basis truncated to
    `energy` variance, variances shrunk toward their mean by `shrink` for stability."""
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(1, Xc.shape[0] - 1)
    keep = np.cumsum(var) / var.sum() <= energy
    keep[0] = True
    Vt, var = Vt[keep], var[keep]
    var_s = (1.0 - shrink) * var + shrink * var.mean()
    W = Vt.T / np.sqrt(var_s + 1e-12)
    return mu, W  # z = (x - mu) @ W


def cca_alignment_scores(
    weak_acts: np.ndarray,
    strong_acts: np.ndarray,
    n_folds: int = 5,
    k_components: int = 64,
    shrink: float = 0.1,
    seed: int = 0,
    trim_frac: float = 0.0,
) -> np.ndarray:
    """Per-example out-of-fold CCA disagreement score. Returns [n] non-negative scores.

    trim_frac > 0 = ROBUST variant: fit once, drop the trim_frac fold-train points with the
    largest canonical disagreement, refit directions on the kept points, then score."""
    Xw = np.asarray(weak_acts, dtype=np.float64)
    Xs = np.asarray(strong_acts, dtype=np.float64)
    n = Xw.shape[0]
    if Xs.shape[0] != n:
        raise ValueError("weak_acts and strong_acts must share the first dim n")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    scores = np.zeros(n, dtype=np.float64)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for test_idx in folds:
            train_idx = np.setdiff1d(perm, test_idx, assume_unique=False)
            muw, Ww = _whitener(Xw[train_idx], shrink)
            mus, Ws = _whitener(Xs[train_idx], shrink)
            Zw_tr = (Xw[train_idx] - muw) @ Ww
            Zs_tr = (Xs[train_idx] - mus) @ Ws
            C = Zw_tr.T @ Zs_tr / Zw_tr.shape[0]
            U, rho, Vt = np.linalg.svd(C, full_matrices=False)
            k = min(k_components, U.shape[1])
            Uk, Vk = U[:, :k], Vt[:k].T
            if trim_frac > 0.0:
                d_tr = np.linalg.norm(Zw_tr @ Uk - Zs_tr @ Vk, axis=1)
                keep = d_tr <= np.quantile(d_tr, 1.0 - trim_frac)
                kt = train_idx[keep]
                muw, Ww = _whitener(Xw[kt], shrink)
                mus, Ws = _whitener(Xs[kt], shrink)
                Zw_k = (Xw[kt] - muw) @ Ww
                Zs_k = (Xs[kt] - mus) @ Ws
                C = Zw_k.T @ Zs_k / Zw_k.shape[0]
                U, rho, Vt = np.linalg.svd(C, full_matrices=False)
                k = min(k_components, U.shape[1])
                Uk, Vk = U[:, :k], Vt[:k].T
            zw = ((Xw[test_idx] - muw) @ Ww) @ Uk
            zs = ((Xs[test_idx] - mus) @ Ws) @ Vk
            scores[test_idx] = np.linalg.norm(zw - zs, axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("cca_alignment_scores produced non-finite values (degenerate fold/whitening)")
    return scores


def _self_test() -> None:
    from compute_pgr import auroc

    rng = np.random.default_rng(0)
    n, dw, ds = 600, 24, 32

    # strong = LINEAR map of weak + per-point noise of varying magnitude (CCA is linear):
    # low-noise points alignable (small score), high-noise points not (large score).
    hw = rng.standard_normal((n, dw))
    W = rng.standard_normal((dw, ds))
    alignable = np.zeros(n, dtype=bool); alignable[: n // 2] = True
    sigma = np.where(alignable, 0.2, 1.5)[:, None]
    hs = hw @ W + sigma * rng.standard_normal((n, ds))

    def _auroc(sc, pos_mask):
        return auroc(sc, pos_mask.astype(int))

    s = cca_alignment_scores(hw, hs, n_folds=4, k_components=16, seed=0)
    assert s.shape == (n,), "shape failed"
    auc = _auroc(s, ~alignable)
    print(f"  alignable mean {s[alignable].mean():.3f} vs noise mean {s[~alignable].mean():.3f}; "
          f"AUROC(noise higher) = {auc:.3f}")
    assert auc > 0.85, f"CCA score should rank noisy points higher, AUROC={auc:.3f}"

    # permutation control
    sp = cca_alignment_scores(hw, hs[rng.permutation(n)], n_folds=4, k_components=16, seed=1)
    auc_p = _auroc(sp, ~alignable)
    print(f"  permuted-strong AUROC (should be ~0.5): {auc_p:.3f}")
    assert abs(auc_p - 0.5) < 0.1, "permutation control failed"

    print("CCA ALIGNMENT SCORE: SELF-TEST PASSED")


if __name__ == "__main__":
    _self_test()
