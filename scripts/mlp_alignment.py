#!/usr/bin/env python3
"""MLP alignment-residual selection score (the "MLP projection" alternative to the
ridge/PCA-style alignment; same family as CCA/CKA alignment methods).

Fit a small MLP f: weak_act -> strong_act and score each example by its OUT-OF-FOLD
alignment residual ||f(h_w_i) - h_s_i||_2. Small residual = the strong representation of
this point is predictable from the weak one (well-aligned / overlap region); large residual
= the strong rep carries structure the weak rep cannot express. keep-high vs keep-low is
decided empirically (two bands, like rp_high/rp_low).

Design notes:
  * K-fold CROSS-FIT is mandatory: an MLP is expressive enough to drive in-sample residuals
    to ~0, which would destroy the signal. Every example is scored by a fold model that
    never saw it.
  * Inputs and targets are z-scored per dimension (fold-train statistics); the residual is
    computed in the standardized target space so the score is scale-invariant.
  * Label-free: unlike rp (which projects the weak-label residual), this score uses only
    the two representation sets.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _zfit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return mu, sd


class _AlignMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int, bottleneck: int | None = None):
        super().__init__()
        mid = bottleneck or hidden
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, mid), nn.GELU(),
            nn.Linear(mid, d_out),
        )

    def forward(self, x):  # noqa: D102
        return self.net(x)


def mlp_alignment_scores(
    weak_acts: np.ndarray,
    strong_acts: np.ndarray,
    n_folds: int = 5,
    hidden: int = 1024,
    epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str | None = None,
    bottleneck: int | None = None,
    history: list | None = None,
    insample: bool = False,
    lda_weight: float = 0.0,
    lda_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Per-example alignment residual ||f(h_w_i) - h_s_i||_2 (standardized target space).

    Default: out-of-fold (cross-fit). insample=True is the 06/23 sketch protocol --
    train = test, no folds -- only meaningful with a narrow bottleneck acting as the
    capacity cap (a no-bottleneck net interpolates and the residual degenerates).
    weak_acts: [n, d_w], strong_acts: [n, d_s]. Returns [n] non-negative scores.
    """
    Xw = np.asarray(weak_acts, dtype=np.float32)
    Xs = np.asarray(strong_acts, dtype=np.float32)
    n = Xw.shape[0]
    if Xs.shape[0] != n:
        raise ValueError("weak_acts and strong_acts must share the first dim n")
    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    if insample:
        muw, sdw = _zfit(Xw); mus, sds = _zfit(Xs)
        Xtr = torch.tensor((Xw - muw) / sdw, device=dev)
        Ytr = torch.tensor((Xs - mus) / sds, device=dev)
        torch.manual_seed(seed)
        model = _AlignMLP(Xw.shape[1], Xs.shape[1], hidden, bottleneck).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        for ep in range(epochs):
            model.train()
            order = torch.randperm(n, device=dev)
            ep_loss, nb = 0.0, 0
            for b0 in range(0, n, batch_size):
                idx = order[b0:b0 + batch_size]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(Xtr[idx]), Ytr[idx])
                loss.backward()
                opt.step()
                ep_loss += float(loss.detach()); nb += 1
            if history is not None and (ep % 10 == 0 or ep == epochs - 1):
                history.append({"fold": -1, "epoch": ep, "train_mse": ep_loss / max(1, nb)})
        model.eval()
        with torch.no_grad():
            scores = (model(Xtr) - Ytr).norm(dim=1).double().cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError("mlp_alignment_scores (insample) produced non-finite values")
        return scores

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    scores = np.zeros(n, dtype=np.float64)

    for fi, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(perm, test_idx, assume_unique=False)
        muw, sdw = _zfit(Xw[train_idx]); mus, sds = _zfit(Xs[train_idx])
        Xtr = torch.tensor((Xw[train_idx] - muw) / sdw, device=dev)
        Ytr = torch.tensor((Xs[train_idx] - mus) / sds, device=dev)
        Xte = torch.tensor((Xw[test_idx] - muw) / sdw, device=dev)
        Yte = torch.tensor((Xs[test_idx] - mus) / sds, device=dev)

        torch.manual_seed(seed * 1000 + fi)
        model = _AlignMLP(Xw.shape[1], Xs.shape[1], hidden, bottleneck).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        m = Xtr.shape[0]
        ytr_lda = (torch.tensor(np.asarray(lda_labels)[train_idx].astype(np.float32), device=dev)
                   if lda_weight > 0 and lda_labels is not None else None)
        for ep in range(epochs):
            model.train()
            order = torch.randperm(m, device=dev)
            ep_loss, nb = 0.0, 0
            for b0 in range(0, m, batch_size):
                idx = order[b0:b0 + batch_size]
                opt.zero_grad(set_to_none=True)
                out = model(Xtr[idx])
                loss = loss_fn(out, Ytr[idx])
                if ytr_lda is not None:
                    yb = ytr_lda[idx]
                    if yb.min() < 0.5 < yb.max():
                        mu1 = out[yb >= 0.5].mean(0); mu0 = out[yb < 0.5].mean(0)
                        between = ((mu1 - mu0) ** 2).sum()
                        within = (((out[yb >= 0.5] - mu1) ** 2).sum(1).mean()
                                  + ((out[yb < 0.5] - mu0) ** 2).sum(1).mean()) / 2
                        # 07/22 construction: penalize low between-cluster variance of outputs
                        loss = loss + lda_weight * within / (between + 1e-6)
                loss.backward()
                opt.step()
                ep_loss += float(loss); nb += 1
            if history is not None and (ep % 5 == 0 or ep == epochs - 1):
                model.eval()
                with torch.no_grad():
                    te = float(loss_fn(model(Xte), Yte))
                history.append({"fold": fi, "epoch": ep, "train_mse": ep_loss / max(1, nb),
                                "heldout_mse": te})
        model.eval()
        with torch.no_grad():
            resid = (model(Xte) - Yte).norm(dim=1)
        scores[test_idx] = resid.double().cpu().numpy()
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if not np.isfinite(scores).all():
        raise ValueError("mlp_alignment_scores produced non-finite values (diverged fold)")
    return scores


def _self_test() -> None:
    rng = np.random.default_rng(0)
    n, dw, ds = 600, 24, 32

    # strong rep = nonlinear function of the weak rep + per-point noise of varying magnitude:
    # low-noise points are alignable (residual should be SMALL), high-noise points carry
    # strong-specific structure the weak rep cannot express (residual should be LARGE).
    hw = rng.standard_normal((n, dw)).astype(np.float32)
    W1 = rng.standard_normal((dw, ds)).astype(np.float32)
    W2 = rng.standard_normal((dw, ds)).astype(np.float32)
    mapped = np.tanh(hw @ W1) + 0.3 * (hw @ W2)          # deterministic nonlinear map
    alignable = np.zeros(n, dtype=bool); alignable[: n // 2] = True
    sigma = np.where(alignable, 0.2, 1.5)[:, None].astype(np.float32)
    hs = mapped + sigma * rng.standard_normal((n, ds)).astype(np.float32)

    def _auroc(sc, pos_mask):
        # rank-based AUROC of sc separating pos (higher) from neg
        order = np.argsort(sc)
        ranks = np.empty(n); ranks[order] = np.arange(1, n + 1)
        npos = pos_mask.sum(); nneg = n - npos
        return (ranks[pos_mask].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)

    s = mlp_alignment_scores(hw, hs, n_folds=4, hidden=256, epochs=250, seed=0, device="cpu")
    assert s.shape == (n,) and np.all(s >= 0), "shape/non-neg failed"
    lo, hi = s[alignable].mean(), s[~alignable].mean()
    auc = _auroc(s, ~alignable)  # noise points should rank HIGHER (bigger residual)
    print(f"  alignable mean {lo:.3f} vs noise mean {hi:.3f};  AUROC(noise ranked higher) = {auc:.3f}")
    assert auc > 0.80, f"score should rank noise points above alignable ones, AUROC={auc:.3f}"

    # permutation control: shuffle strong reps -> ranking signal must vanish
    hs_perm = hs[rng.permutation(n)]
    sp = mlp_alignment_scores(hw, hs_perm, n_folds=4, hidden=128, epochs=60, seed=1, device="cpu")
    auc_p = _auroc(sp, ~alignable)
    print(f"  permuted-strong AUROC (should be ~0.5): {auc_p:.3f}")
    assert abs(auc_p - 0.5) < 0.1, "permutation control failed -- signal without structure"

    print("MLP ALIGNMENT SCORE: SELF-TEST PASSED")


if __name__ == "__main__":
    _self_test()
