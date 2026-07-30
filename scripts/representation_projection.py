#!/usr/bin/env python3
# 2026-06-15, World Cup day 5: Spain 0-0 Cape Verde. The favorites, held. Vamos Cabo Verde!!
"""rp: the representation-projection selection score.

Comes from "Representations Shape Weak-to-Strong Generalization" (Xue, Li &
Mirzasoleiman, ICML 2025, arXiv:2502.00620). Their theorem has this quantity
P_s(I - P_w) y_hat sitting in the middle of the prediction-gap formula, but the
paper never uses it to pick data (and has no public code), so I implemented it
and turned the per-example entries into a selection score.

How I read the math, in one breath: P_w and P_s are kernel-ridge smoothers

    P = K (K + reg*I)^{-1},   K = X X^T / n   (Gram kernel of the reps)

so (I - P_w) y_hat = the part of the weak labels the weak rep cannot explain,
and P_s(...) = how much of that leftover the strong rep CAN express. Big value
= "the weak model is confused here but the strong model gets it", which is
exactly the kind of point you want to train on. Score = |per-example entry|.
Which end to keep (high or low) is an empirical question; downstream says high.
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
    """Per-example score |[P_s (I - P_w) y_hat]_i|.

    weak_acts/strong_acts: [n, d] reps (pooled pre-lm_head activations in our runs).
    weak_labels: [n] weak labels y_hat (0/1; soft also accepted).
    reg: ridge knob, as a fraction of the average kernel eigenvalue (see below).
    n_components: optional PCA cut to imitate the paper's "principal" reps.
    Returns: [n] non-negative scores.
    """
    Xw = np.asarray(weak_acts, dtype=np.float64)
    Xs = np.asarray(strong_acts, dtype=np.float64)
    y = np.asarray(weak_labels, dtype=np.float64)
    n = Xw.shape[0]
    if Xs.shape[0] != n or y.shape[0] != n:
        raise ValueError("weak_acts, strong_acts, weak_labels must share the first dim n")

    # macOS + numpy>=2.0 fires fake overflow warnings on big matmuls (Accelerate
    # BLAS thing, results are fine, Linux is silent). I checked the outputs are
    # finite, so I just mute it here instead of scaring everyone in the logs (._.)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Xw = _center(Xw, n_components)
        Xs = _center(Xs, n_components)
        Kw = Xw @ Xw.T / n
        Ks = Xs @ Xs.T / n
        eye = np.eye(n)
        # reg scales with trace/n (= average eigenvalue), so the same reg=0.1
        # means the same thing on a 896-dim weak rep and a 3584-dim strong rep.
        rw = reg * (np.trace(Kw) / n) + 1e-12
        rs = reg * (np.trace(Ks) / n) + 1e-12

        # center y first, otherwise the class prior walks in through the constant
        # direction and the score silently becomes "rank by the majority class"
        yc = y - y.mean()
        # step 1: (I - P_w) y = y - K_w (K_w + rw I)^{-1} y. solve() not inv(),
        # my laptop and I both prefer to keep the O(n^3) as clean as possible
        pw_y = Kw @ np.linalg.solve(Kw + rw * eye, yc)
        a = yc - pw_y
        # step 2: P_s a. same smoother, strong side, fitted right here in-sample.
        # this being in-sample is load-bearing, not laziness: see the step-2 note
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

    # 2) constant labels -> zero signal (if this fails something is very wrong)
    s_const = representation_projection_scores(hw, hs, np.ones(n))
    assert s_const.max() < 1e-9, f"constant-y should give 0, got max {s_const.max()}"

    # 3) h_w == h_s (strong adds nothing new) -> smaller scores than independent reps
    s_same = representation_projection_scores(hw, hw, y, reg=1e-3)
    s_diff = representation_projection_scores(hw, hs, y, reg=1e-3)
    assert s_same.mean() < s_diff.mean(), "h_w==h_s should give smaller scores than independent h_s"

    # 4) informative case: strong rep carries the label, weak rep is pure noise
    signal = (y[:, None] * rng.standard_normal((1, ds)))  # y-correlated direction
    hs_inf = signal + 0.1 * rng.standard_normal((n, ds))
    hw_noise = rng.standard_normal((n, dw))
    s_inf = representation_projection_scores(hw_noise, hs_inf, y)
    hi = s_inf[y == 1].mean()
    lo = s_inf[y == 0].mean()
    print(f"  informative case: score(y=1)={hi:.4f}  score(y=0)={lo:.4f}")

    print("  shapes OK | constant-y=0 OK | h_w==h_s < independent OK")
    print("REPRESENTATION-PROJECTION SCORE: SELF-TEST PASSED")


if __name__ == "__main__":
    _self_test()
