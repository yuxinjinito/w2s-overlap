#!/usr/bin/env python3
"""Excess-loss / learnability selection score.

For each input x, measure the loss on the weak model and on the UNTUNED strong model
and subtract them,

    score(x) = loss_weak(x) - loss_untuned_strong(x)      "how much can we learn from x".

The loss can vary; we expose two:
  - "entropy": each model's predictive class entropy H(p) (label-free uncertainty).
               s = H(p_weak) - H(p_strong).   <- label-free; the default flavor.
  - "nll":     the negative log-likelihood of the (weak) label yhat under each model.
               s = nll(p_weak, yhat) - nll(p_strong, yhat).

p_weak / p_strong are P(label=1) from the weak and the UNTUNED-strong correctness
predictors. In our pipeline both are linear probes on the weak / base-strong
representations -- exactly how the weak labeler is already defined -- so "untuned strong
loss" is the natural symmetric counterpart to the weak loss. Keep-high vs keep-low is
decided empirically (a new band, like confidence / kNN / L2-residual / rp).
"""
from __future__ import annotations

import numpy as np


def _binary_entropy(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def _binary_nll(p: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def excess_loss_scores(
    weak_probs,
    strong_probs,
    weak_labels=None,
    loss: str = "entropy",
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-example score loss_weak(x) - loss_untuned_strong(x).

    weak_probs / strong_probs: [n] P(label=1) from the weak and untuned-strong predictors.
    weak_labels: [n] weak labels yhat (0/1), required only for loss="nll".
    loss: "entropy" (label-free class uncertainty) or "nll" (NLL of yhat).
    Returns: [n] signed scores (can be +/-); band-select high vs low empirically.
    """
    pw = np.asarray(weak_probs, dtype=np.float64).ravel()
    ps = np.asarray(strong_probs, dtype=np.float64).ravel()
    if pw.shape != ps.shape:
        raise ValueError("weak_probs and strong_probs must share shape")
    if loss == "entropy":
        return _binary_entropy(pw, eps) - _binary_entropy(ps, eps)
    if loss == "nll":
        if weak_labels is None:
            raise ValueError('loss="nll" requires weak_labels')
        y = np.asarray(weak_labels, dtype=np.float64).ravel()
        if y.shape != pw.shape:
            raise ValueError("weak_labels must match weak_probs shape")
        return _binary_nll(pw, y, eps) - _binary_nll(ps, y, eps)
    raise ValueError(f"unknown loss {loss!r} (expected 'entropy' or 'nll')")


def _self_test() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    pw, ps = rng.random(n), rng.random(n)
    y = (rng.random(n) < 0.5).astype(int)

    s_ent = excess_loss_scores(pw, ps, loss="entropy")
    s_nll = excess_loss_scores(pw, ps, y, loss="nll")
    assert s_ent.shape == (n,) and np.all(np.isfinite(s_ent)), "entropy shape/finite failed"
    assert s_nll.shape == (n,) and np.all(np.isfinite(s_nll)), "nll shape/finite failed"

    # entropy: weak uncertain (0.5) vs strong confident (0.99) -> strong less uncertain -> +
    assert excess_loss_scores([0.5], [0.99], loss="entropy")[0] > 0
    # identical predictors -> exactly 0
    assert abs(excess_loss_scores([0.7], [0.7], loss="entropy")[0]) < 1e-12
    # nll on y=1: weak wrong-confident (0.01) vs strong right-confident (0.99) -> +
    assert excess_loss_scores([0.01], [0.99], [1], loss="nll")[0] > 0
    # boundary probs do not blow up (clipped)
    assert np.all(np.isfinite(excess_loss_scores([0.0, 1.0], [1.0, 0.0], [1, 0], loss="nll")))

    print("  entropy/nll shapes+finite OK | sign OK | identical=0 OK | boundary-safe OK")
    print("EXCESS-LOSS SCORE: SELF-TEST PASSED")


if __name__ == "__main__":
    _self_test()
