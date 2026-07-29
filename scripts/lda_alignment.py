#!/usr/bin/env python3
"""LDA-projected alignment residual -- the label-aware twin of l2resid.

Same cross-fit closed-form ridge map W: weak -> strong as the linear-residual arm, but the
held-out residual VECTOR is projected onto the strong side's LDA axis (Fisher discriminant
of the strong representations against the HARD WEAK labels, shrinkage-regularized) instead
of being collapsed to a norm:

    score_i = | < r_i , w_lda > |,   r_i = (h_s_i - mu_s) - (h_w_i - mu_w) W

One factor away from l2resid (which is ||r_i||, all directions weighted equally): the only
added ingredient is the weak-label axis. If this arm works where l2resid failed, the label
component is the active ingredient (the disentangling experiment proposed at the 07/22
meeting). Weak labels only; gold never enters.
"""
from __future__ import annotations

import numpy as np


def _fit_ridge_map(Xw: np.ndarray, Xs: np.ndarray, lam_frac: float):
    muw, mus = Xw.mean(0, keepdims=True), Xs.mean(0, keepdims=True)
    Xwc, Xsc = Xw - muw, Xs - mus
    G = Xwc.T @ Xwc / Xwc.shape[0]
    lam = lam_frac * (np.trace(G) / G.shape[0]) + 1e-12
    W = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xwc.T @ Xsc / Xwc.shape[0])
    return muw, mus, W


def _lda_axis(Xs: np.ndarray, y: np.ndarray, shrink_frac: float) -> np.ndarray:
    """Shrinkage Fisher axis on (strong reps, hard weak labels); unit norm."""
    X0, X1 = Xs[y == 0], Xs[y == 1]
    if len(X0) < 2 or len(X1) < 2:
        raise ValueError("_lda_axis needs both classes present in the fold")
    mu0, mu1 = X0.mean(0), X1.mean(0)
    Xc = np.concatenate([X0 - mu0, X1 - mu1], axis=0)
    Sw = Xc.T @ Xc / Xc.shape[0]
    lam = shrink_frac * (np.trace(Sw) / Sw.shape[0]) + 1e-12
    w = np.linalg.solve(Sw + lam * np.eye(Sw.shape[0]), mu1 - mu0)
    return w / (np.linalg.norm(w) + 1e-12)


def lda_alignment_scores(
    weak_acts: np.ndarray,
    strong_acts: np.ndarray,
    weak_labels: np.ndarray,
    n_folds: int = 5,
    lam_frac: float = 0.1,
    shrink_frac: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Out-of-fold |<ridge-map residual, strong LDA axis>|. Returns [n] non-negative scores."""
    Xw = np.asarray(weak_acts, dtype=np.float64)
    Xs = np.asarray(strong_acts, dtype=np.float64)
    y = np.asarray(weak_labels).astype(int).reshape(-1)
    n = Xw.shape[0]
    if Xs.shape[0] != n or y.shape[0] != n:
        raise ValueError("weak_acts, strong_acts, weak_labels must share the first dim n")
    sd = Xs.std(0, keepdims=True) + 1e-8
    Xs = Xs / sd  # same target standardization as the linear-residual arms

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    scores = np.zeros(n, dtype=np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for test_idx in folds:
            tr = np.setdiff1d(perm, test_idx, assume_unique=False)
            muw, mus, W = _fit_ridge_map(Xw[tr], Xs[tr], lam_frac)
            w = _lda_axis(Xs[tr] - mus, y[tr], shrink_frac)
            r = (Xs[test_idx] - mus) - (Xw[test_idx] - muw) @ W
            scores[test_idx] = np.abs(r @ w)
    if not np.isfinite(scores).all():
        raise ValueError("lda_alignment_scores produced non-finite values")
    return scores


def _self_test() -> None:
    rng = np.random.default_rng(0)
    n, dw, ds = 600, 24, 32

    # Weak reps + a linear map to strong reps. Labels depend on hw[:,0]. Half the points get
    # residual noise ALONG the strong-space class-mean axis (label-relevant unexplained
    # content), half get the SAME NORM of noise orthogonal to it. l2resid (norm) cannot tell
    # them apart; the LDA projection must rank the along-axis group higher.
    hw = rng.standard_normal((n, dw))
    Wm = rng.standard_normal((dw, ds))
    y = (hw[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(int)
    base = hw @ Wm
    mu_diff = base[y == 1].mean(0) - base[y == 0].mean(0)
    u = mu_diff / np.linalg.norm(mu_diff)
    v = rng.standard_normal(ds); v -= (v @ u) * u; v /= np.linalg.norm(v)
    along = np.zeros(n, dtype=bool); along[: n // 2] = True
    rng.shuffle(along)
    noise_dir = np.where(along[:, None], u[None, :], v[None, :])
    hs = base + 3.0 * noise_dir * rng.standard_normal((n, 1))

    def _auroc(sc, pos):
        order = np.argsort(sc); ranks = np.empty(n); ranks[order] = np.arange(1, n + 1)
        np_, nn = pos.sum(), n - pos.sum()
        return (ranks[pos].sum() - np_ * (np_ + 1) / 2) / (np_ * nn)

    s = lda_alignment_scores(hw, hs, y, n_folds=4, seed=0)
    auc = _auroc(s, along)
    print(f"  AUROC(label-axis noise ranked higher) = {auc:.3f}")
    assert auc > 0.80, f"LDA projection should find label-relevant residuals, AUROC={auc:.3f}"

    y_perm = np.array(y); rng.shuffle(y_perm)
    sp = lda_alignment_scores(hw, hs, y_perm, n_folds=4, seed=1)
    auc_p = _auroc(sp, along)
    print(f"  permuted-label AUROC (should drop toward 0.5): {auc_p:.3f}")
    assert auc_p < auc - 0.15, "permuting labels should destroy most of the separation"

    print("LDA ALIGNMENT SCORE: SELF-TEST PASSED")


if __name__ == "__main__":
    _self_test()
