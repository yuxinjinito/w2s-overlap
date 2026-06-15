#!/usr/bin/env python3
"""Representation-projection selection score (Xue, Li & Mirzasoleiman, ICML 2025,
"Representations Shape Weak-to-Strong Generalization", arXiv:2502.00620).

The paper has NO public code; this is our own implementation of the per-example
quantity that drives the weak-to-strong prediction gap.

Idea: from the (optionally PCA-reduced) weak/strong representations build regularized
kernel projection operators

    P = K (K + reg*I)^{-1},     K = X X^T / n   (Gram kernel of the representations)

P_s(I - P_w) y_hat is the part of the weak labels that lies in the STRONG model's
expressible space but NOT the weak model's -- "what the strong model can learn here
that the weak one cannot." Its squared norm approximates the weak-vs-GT prediction gap
(Thm 3.8). We expose the per-example contribution |[P_s(I - P_w) y_hat]_i| as a
selection score; keep-high vs keep-low is to be decided empirically (a new band, like
confidence / kNN / L2-residual).
"""
from __future__ import annotations

import numpy as np


def _pca(X: np.ndarray, k: int | None) -> np.ndarray:
    if not k or k >= X.shape[1]:
        return X
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :k] * S[:k]


def _center(X: np.ndarray, k: int | None) -> np.ndarray:
    return _pca(X - X.mean(axis=0, keepdims=True), k)


def representation_projection_scores(
    weak_acts: np.ndarray,
    strong_acts: np.ndarray,
    weak_labels: np.ndarray,
    reg: float = 0.1,
    n_components: int | None = None,
) -> np.ndarray:
    """Per-example score |[P_s (I - P_w) y_hat]_i| for selection.

    weak_acts/strong_acts: [n, d] representations (e.g. pooled pre-lm_head activations).
    weak_labels: [n] weak labels y_hat (0/1 or soft).
    reg: kernel ridge regularization (fraction of the avg kernel eigenvalue).
    n_components: optional PCA truncation to the top components ("principal" reps).
    Returns: [n] non-negative scores.
    """
    Xw = np.asarray(weak_acts, dtype=np.float64)
    Xs = np.asarray(strong_acts, dtype=np.float64)
    y = np.asarray(weak_labels, dtype=np.float64)
    n = Xw.shape[0]
    if Xs.shape[0] != n or y.shape[0] != n:
        raise ValueError("weak_acts, strong_acts, weak_labels must share the first dim n")

    # macOS Accelerate BLAS (numpy>=2.0) raises spurious FP-exception RuntimeWarnings on
    # large matmuls even when the result is finite/correct (does not happen on Linux);
    # suppress them around the (validated) linear algebra.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Xw = _center(Xw, n_components)
        Xs = _center(Xs, n_components)
        Kw = Xw @ Xw.T / n
        Ks = Xs @ Xs.T / n
        eye = np.eye(n)
        # reg as a fraction of each kernel's average eigenvalue (= trace/n), so it is
        # scale-invariant and never blows up the representations.
        rw = reg * (np.trace(Kw) / n) + 1e-12
        rs = reg * (np.trace(Ks) / n) + 1e-12

        yc = y - y.mean()
        # (I - P_w) y = y - K_w (K_w + rw I)^{-1} y   -- solve, don't invert
        pw_y = Kw @ np.linalg.solve(Kw + rw * eye, yc)
        a = yc - pw_y
        # P_s a = K_s (K_s + rs I)^{-1} a
        v = Ks @ np.linalg.solve(Ks + rs * eye, a)
    return np.abs(v)


def _self_test() -> None:
    rng = np.random.default_rng(0)
    n, dw, ds = 300, 32, 48

    # 1) shapes
    hw, hs = rng.standard_normal((n, dw)), rng.standard_normal((n, ds))
    y = (rng.random(n) < 0.5).astype(float)
    s = representation_projection_scores(hw, hs, y)
    assert s.shape == (n,) and np.all(s >= 0), "shape/non-neg failed"

    # 2) constant labels -> zero signal
    s_const = representation_projection_scores(hw, hs, np.ones(n))
    assert s_const.max() < 1e-9, f"constant-y should give 0, got max {s_const.max()}"

    # 3) h_w == h_s (strong adds nothing) -> small, with small reg
    s_same = representation_projection_scores(hw, hw, y, reg=1e-3)
    s_diff = representation_projection_scores(hw, hs, y, reg=1e-3)
    assert s_same.mean() < s_diff.mean(), "h_w==h_s should give smaller scores than independent h_s"

    # 4) informative: strong aligned with y, weak is noise -> non-trivial signal on the right points
    signal = (y[:, None] * rng.standard_normal((1, ds)))  # y-correlated direction
    hs_inf = signal + 0.1 * rng.standard_normal((n, ds))
    hw_noise = rng.standard_normal((n, dw))
    s_inf = representation_projection_scores(hw_noise, hs_inf, y)
    # points where the strong rep carries the label signal should score higher on average
    hi = s_inf[y == 1].mean()
    lo = s_inf[y == 0].mean()
    print(f"  informative case: score(y=1)={hi:.4f}  score(y=0)={lo:.4f}")

    print("  shapes OK | constant-y=0 OK | h_w==h_s < independent OK")
    print("REPRESENTATION-PROJECTION SCORE: SELF-TEST PASSED")


if __name__ == "__main__":
    _self_test()
