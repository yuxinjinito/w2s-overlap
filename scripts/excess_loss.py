#!/usr/bin/env python3
"""Excess-loss / learnability selection score -- K-way class-entropy variant.

For each question we score its K answer options with a model's yes/no LM head, take
P(correct | question, option_k) for each option, normalise the K scores into a
distribution over the options, and take its entropy -- the model's uncertainty
"across the class labels". The selection score is the difference between the weak and
the UNTUNED-strong model's option entropies,

    score(question) = H_weak(options) - H_strong(options),

both from the discrimination (yes/no) head, no probe, keeping the existing coin-flip
training format. Keep-high vs keep-low is a band decided empirically (like confidence /
kNN / L2-residual / rp).
"""
from __future__ import annotations

import numpy as np


def option_entropy(p_correct, eps: float = 1e-12) -> float:
    """Entropy (nats) of the option distribution from per-option P(correct).

    p_correct: the K per-option P(correct) scores for ONE question. They need not sum
    to 1 -- we normalise (so this is the entropy of "which option is the answer").
    """
    p = np.asarray(p_correct, dtype=np.float64).ravel()
    p = np.clip(p, eps, None)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    return float(-(p * np.log(p)).sum())


def excess_loss_kway_scores(weak_probs_by_q: dict, strong_probs_by_q: dict) -> dict:
    """Per-question H_weak - H_strong.

    The one deliberate exception to the score-function contract: it maps
    question dicts to signed values, because the entropy gap is a question-level
    quantity with a meaningful sign, and the pipeline broadcasts it to rows and
    bands it in both directions downstream.

    weak_probs_by_q / strong_probs_by_q: {question_id: [P(correct) for each of the K
    options]} from the weak and untuned-strong models. Returns {question_id: score}.
    """
    out = {}
    for q, wp in weak_probs_by_q.items():
        sp = strong_probs_by_q.get(q)
        if sp is None:
            continue
        out[q] = option_entropy(wp) - option_entropy(sp)
    return out


def _self_test() -> None:
    # uniform weak (max entropy) vs peaked strong (low entropy) -> el > 0
    weak = {"q": [0.5, 0.5, 0.5, 0.5]}
    strong = {"q": [0.95, 0.02, 0.02, 0.01]}
    el = excess_loss_kway_scores(weak, strong)["q"]
    assert el > 0, f"uniform-weak vs peaked-strong should be +, got {el}"

    # identical distributions -> 0
    same = excess_loss_kway_scores({"q": [0.6, 0.3, 0.1]}, {"q": [0.6, 0.3, 0.1]})["q"]
    assert abs(same) < 1e-12, f"identical should be 0, got {same}"

    # entropy bounds: uniform over K -> log K; one-hot -> 0
    K = 4
    assert abs(option_entropy([1.0] * K) - np.log(K)) < 1e-9
    assert option_entropy([1.0, 0.0, 0.0, 0.0]) < 1e-9

    # all-zero option scores do not blow up
    assert np.isfinite(option_entropy([0.0, 0.0, 0.0]))

    # missing strong question is skipped, not crashed
    assert excess_loss_kway_scores({"a": [0.5, 0.5]}, {}) == {}

    print("  option_entropy bounds OK | sign OK | identical=0 OK | robust to zeros/missing")
    print("EXCESS-LOSS K-WAY SCORE: SELF-TEST PASSED")


if __name__ == "__main__":
    _self_test()
