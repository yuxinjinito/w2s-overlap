#!/usr/bin/env python3
"""A small sparse autoencoder over activation rows, shared by the screens.

Tied-ish two-layer SAE with an L1 code penalty and dead-unit resampling. Both
sae_alignment_screen and joint_screen train these; the class lived inside the
former until the latter started importing it sideways, which put shared math
inside one driver. This module may import nothing else in the project.
"""
import numpy as np
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
