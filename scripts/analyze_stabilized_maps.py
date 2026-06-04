#!/usr/bin/env python3
"""Fit more stable weak-to-strong activation maps from saved embeddings.

This script is for the "two objects" John asked for:

1. a usable optimization matrix A from weak activations to strong activations;
2. a per-sample residual measuring how much each example changes after mapping.

The raw high-dimensional linear map can be unstable when the number of mapping
training examples is close to the weak hidden dimension. This script compares
centered ridge maps and PCA-reduced ridge maps without re-extracting model
activations.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@dataclass
class MapResult:
    name: str
    kind: str
    ridge: float
    pca_dim: Optional[int]
    map_matrix: torch.Tensor
    predictions: torch.Tensor
    residuals: torch.Tensor
    x_mean: Optional[torch.Tensor] = None
    y_mean: Optional[torch.Tensor] = None
    weak_pca_basis: Optional[torch.Tensor] = None
    strong_pca_basis: Optional[torch.Tensor] = None
    original_space_map: Optional[torch.Tensor] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_artifact", help=".pt with weak_embeddings, strong_embeddings, map_train_mask")
    parser.add_argument("--mapping-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ridge-values", default="0.1,1.0,10.0,100.0")
    parser.add_argument("--pca-dims", default="64,128,256,512")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--best-by", choices=["heldout_mean", "heldout_median"], default="heldout_median")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_artifact(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def require_keys(artifact: dict, keys: list[str]) -> None:
    missing = [key for key in keys if key not in artifact]
    if missing:
        raise KeyError(f"Embedding artifact is missing required keys: {missing}")


def fit_ridge_map(x_train: torch.Tensor, y_train: torch.Tensor, ridge: float) -> torch.Tensor:
    n = x_train.shape[0]
    gram = x_train @ x_train.T
    gram = gram + ridge * torch.eye(n, dtype=x_train.dtype, device=x_train.device)
    coeff = torch.linalg.solve(gram, y_train)
    return x_train.T @ coeff


def pca_basis(x_centered_train: torch.Tensor, dim: int) -> torch.Tensor:
    _, _, vh = torch.linalg.svd(x_centered_train, full_matrices=False)
    dim = min(dim, vh.shape[0])
    return vh[:dim].T.contiguous()


def fit_raw_ridge(
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    ridge: float,
) -> MapResult:
    a = fit_ridge_map(x[train_mask], y[train_mask], ridge)
    pred = x @ a
    return MapResult(
        name=f"raw_ridge_{safe_float(ridge)}",
        kind="raw_ridge",
        ridge=ridge,
        pca_dim=None,
        map_matrix=a,
        predictions=pred,
        residuals=pred - y,
        original_space_map=a,
    )


def fit_centered_ridge(
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    ridge: float,
) -> MapResult:
    x_train = x[train_mask]
    y_train = y[train_mask]
    x_mean = x_train.mean(dim=0, keepdim=True)
    y_mean = y_train.mean(dim=0, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    a = fit_ridge_map(x_centered[train_mask], y_centered[train_mask], ridge)
    pred = x_centered @ a + y_mean
    return MapResult(
        name=f"centered_ridge_{safe_float(ridge)}",
        kind="centered_ridge",
        ridge=ridge,
        pca_dim=None,
        map_matrix=a,
        predictions=pred,
        residuals=pred - y,
        x_mean=x_mean.squeeze(0),
        y_mean=y_mean.squeeze(0),
        original_space_map=a,
    )


def fit_pca_ridge(
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    weak_basis: torch.Tensor,
    strong_basis: torch.Tensor,
    ridge: float,
    dim: int,
) -> MapResult:
    x_train = x[train_mask]
    y_train = y[train_mask]
    x_mean = x_train.mean(dim=0, keepdim=True)
    y_mean = y_train.mean(dim=0, keepdim=True)

    bx = weak_basis[:, :dim]
    by = strong_basis[:, :dim]
    x_scores = (x - x_mean) @ bx
    y_scores = (y - y_mean) @ by
    a = fit_ridge_map(x_scores[train_mask], y_scores[train_mask], ridge)
    pred = (x_scores @ a) @ by.T + y_mean
    original = bx @ a @ by.T

    return MapResult(
        name=f"pca{dim}_ridge_{safe_float(ridge)}",
        kind="pca_ridge",
        ridge=ridge,
        pca_dim=dim,
        map_matrix=a,
        predictions=pred,
        residuals=pred - y,
        x_mean=x_mean.squeeze(0),
        y_mean=y_mean.squeeze(0),
        weak_pca_basis=bx,
        strong_pca_basis=by,
        original_space_map=original,
    )


def safe_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def summarize_series(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def metric_summary(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    pred = pred[mask]
    target = target[mask]
    residual = pred - target
    l2 = torch.linalg.norm(residual, dim=1)
    mse = residual.square().mean(dim=1)
    cosine = F.cosine_similarity(pred, target, dim=1)
    out = {}
    for prefix, values in [("l2", l2), ("mse", mse), ("cosine", cosine)]:
        stats = summarize_series(values)
        for key, value in stats.items():
            out[f"{prefix}_{key}"] = value
    return out


def singular_summary(matrix: torch.Tensor, top_k: int) -> dict[str, object]:
    s = torch.linalg.svdvals(matrix.float())
    top = s[:top_k]
    energy = s.square()
    total_energy = energy.sum().item()
    cumulative = (energy[:top_k].cumsum(dim=0) / total_energy).cpu().tolist() if total_energy > 0 else []
    tol = torch.finfo(s.dtype).eps * max(matrix.shape) * s.max()
    positive = s[s > 1e-8]
    condition = float((positive.max() / positive.min()).item()) if len(positive) else None
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "rank_estimate": int((s > tol).sum().item()),
        "frobenius_norm": float(torch.linalg.norm(matrix.float()).item()),
        "spectral_norm": float(s.max().item()) if len(s) else 0.0,
        "condition_estimate": condition,
        "top_singular_values": [float(v) for v in top.cpu().tolist()],
        "top_singular_cumulative_energy": [float(v) for v in cumulative],
    }


def result_row(result: MapResult, y: torch.Tensor, train_mask: torch.Tensor, top_k: int) -> dict[str, object]:
    heldout_mask = ~train_mask
    train_metrics = metric_summary(result.predictions, y, train_mask)
    heldout_metrics = metric_summary(result.predictions, y, heldout_mask)
    sv = singular_summary(result.map_matrix, top_k)
    return {
        "name": result.name,
        "kind": result.kind,
        "ridge": result.ridge,
        "pca_dim": result.pca_dim,
        "map_shape": "x".join(str(v) for v in sv["shape"]),
        "rank_estimate": sv["rank_estimate"],
        "spectral_norm": sv["spectral_norm"],
        "frobenius_norm": sv["frobenius_norm"],
        "top1_energy": sv["top_singular_cumulative_energy"][0] if sv["top_singular_cumulative_energy"] else None,
        "top20_energy": sv["top_singular_cumulative_energy"][-1] if sv["top_singular_cumulative_energy"] else None,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"heldout_{key}": value for key, value in heldout_metrics.items()},
    }


def residual_dataframe(result: MapResult, artifact: dict, mapping_csv: Optional[str]) -> pd.DataFrame:
    residuals = result.residuals.detach().cpu().float()
    pred = result.predictions.detach().cpu().float()
    strong = artifact["strong_embeddings"].detach().cpu().float()
    weak = artifact["weak_embeddings"].detach().cpu().float()
    train_mask = artifact["map_train_mask"].detach().cpu().bool()
    df = pd.DataFrame(
        {
            "row_index": np.arange(len(train_mask)),
            "map_train": train_mask.numpy().astype(int),
            "weak_norm": torch.linalg.norm(weak, dim=1).numpy(),
            "strong_norm": torch.linalg.norm(strong, dim=1).numpy(),
            "mapped_weak_norm": torch.linalg.norm(pred, dim=1).numpy(),
            "residual_l2": torch.linalg.norm(residuals, dim=1).numpy(),
            "residual_mse": residuals.square().mean(dim=1).numpy(),
            "map_name": result.name,
            "map_kind": result.kind,
            "ridge": result.ridge,
            "pca_dim": result.pca_dim,
        }
    )
    if mapping_csv:
        meta = pd.read_csv(mapping_csv)
        if len(meta) != len(df):
            raise ValueError(f"mapping CSV length ({len(meta)}) does not match artifact length ({len(df)})")
        keep_cols = [col for col in ["id", "dataset", "split", "label", "text"] if col in meta.columns]
        df = pd.concat([meta[keep_cols].reset_index(drop=True), df], axis=1)
    return df


def write_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_best_artifact(result: MapResult, artifact: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": result.name,
        "kind": result.kind,
        "ridge": result.ridge,
        "pca_dim": result.pca_dim,
        "map_matrix": result.map_matrix.detach().cpu(),
        "original_space_map": result.original_space_map.detach().cpu() if result.original_space_map is not None else None,
        "predictions": result.predictions.detach().cpu(),
        "residuals": result.residuals.detach().cpu(),
        "map_train_mask": artifact["map_train_mask"].detach().cpu(),
        "weak_model": artifact.get("weak_model"),
        "strong_model": artifact.get("strong_model"),
        "dataset": artifact.get("dataset"),
        "target_split": artifact.get("target_split"),
        "pooling": artifact.get("pooling"),
    }
    for key, value in [
        ("x_mean", result.x_mean),
        ("y_mean", result.y_mean),
        ("weak_pca_basis", result.weak_pca_basis),
        ("strong_pca_basis", result.strong_pca_basis),
    ]:
        payload[key] = value.detach().cpu() if value is not None else None
    torch.save(payload, path)


def write_report(best: MapResult, rows: list[dict[str, object]], output: Path, top_k: int) -> None:
    best_row = next(row for row in rows if row["name"] == best.name)
    sorted_rows = sorted(rows, key=lambda row: row["heldout_l2_median"])
    lines = [
        "Stabilized Weak-to-Strong Map Analysis",
        "======================================",
        "",
        "Best Config",
        "-----------",
        f"name: {best.name}",
        f"kind: {best.kind}",
        f"ridge: {best.ridge}",
        f"pca_dim: {best.pca_dim}",
        f"heldout_l2_mean: {best_row['heldout_l2_mean']:.6f}",
        f"heldout_l2_median: {best_row['heldout_l2_median']:.6f}",
        f"heldout_cosine_mean: {best_row['heldout_cosine_mean']:.6f}",
        f"spectral_norm: {best_row['spectral_norm']:.6f}",
        f"top1_energy: {best_row['top1_energy']:.6f}",
        f"top20_energy: {best_row['top20_energy']:.6f}",
        "",
        "Top Configs By Heldout Median L2",
        "--------------------------------",
    ]
    for rank, row in enumerate(sorted_rows[:10], start=1):
        lines.append(
            f"{rank:02d}. {row['name']} | heldout median={row['heldout_l2_median']:.6f} "
            f"mean={row['heldout_l2_mean']:.6f} cosine={row['heldout_cosine_mean']:.6f} "
            f"spectral={row['spectral_norm']:.6f} top1_energy={row['top1_energy']:.6f}"
        )
    lines.extend(
        [
            "",
            "Interpretation",
            "--------------",
            "- Prefer configs with low heldout residual and non-exploding singular values.",
            "- Raw/weakly-regularized maps can look impressive on mapping-train data while producing unstable heldout residuals.",
            "- PCA maps are optimization matrices in reduced weak/strong principal-component coordinates; the saved artifact also includes the equivalent original-space map.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def maybe_plot(rows: list[dict[str, object]], best: MapResult, artifact: dict, output_dir: Path, top_k: int) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = df.sort_values("heldout_l2_median")
    ax.bar(np.arange(len(ordered)), ordered["heldout_l2_median"])
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels(ordered["name"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("heldout median residual L2")
    ax.set_title("Stabilized Map Config Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "config_heldout_median_l2.png", dpi=200)
    plt.close(fig)

    s = torch.linalg.svdvals(best.map_matrix.float()).cpu().numpy()
    energy = np.cumsum(s**2) / np.sum(s**2) if np.sum(s**2) > 0 else np.zeros_like(s)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    n = min(top_k, len(s))
    ax1.plot(np.arange(1, n + 1), s[:n], marker="o")
    ax1.set_xlabel("singular direction")
    ax1.set_ylabel("singular value")
    ax1.set_title(f"Best Map Direction Strength: {best.name}")
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, n + 1), energy[:n], color="tab:orange", marker="x")
    ax2.set_ylabel("cumulative energy")
    fig.tight_layout()
    fig.savefig(output_dir / "best_map_singular_values.png", dpi=200)
    plt.close(fig)

    heldout = ~artifact["map_train_mask"].detach().cpu().bool()
    l2 = torch.linalg.norm(best.residuals.detach().cpu().float(), dim=1)[heldout].numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(l2, bins=40, color="#4c78a8", edgecolor="white")
    ax.set_xlabel("heldout residual L2")
    ax.set_ylabel("number of examples")
    ax.set_title(f"Best Map Heldout Residuals: {best.name}")
    fig.tight_layout()
    fig.savefig(output_dir / "best_map_heldout_residual_hist.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ridge_values = parse_float_list(args.ridge_values)
    pca_dims = parse_int_list(args.pca_dims)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_artifact(args.embedding_artifact)
    require_keys(artifact, ["weak_embeddings", "strong_embeddings", "map_train_mask"])
    x = artifact["weak_embeddings"].float().to(device)
    y = artifact["strong_embeddings"].float().to(device)
    train_mask = artifact["map_train_mask"].bool().to(device)

    results: list[MapResult] = []
    for ridge in ridge_values:
        results.append(fit_raw_ridge(x, y, train_mask, ridge))
        results.append(fit_centered_ridge(x, y, train_mask, ridge))

    max_pca = max(pca_dims) if pca_dims else 0
    if max_pca > 0:
        x_train = x[train_mask]
        y_train = y[train_mask]
        weak_basis = pca_basis(x_train - x_train.mean(dim=0, keepdim=True), max_pca)
        strong_basis = pca_basis(y_train - y_train.mean(dim=0, keepdim=True), max_pca)
        for dim in pca_dims:
            for ridge in ridge_values:
                results.append(fit_pca_ridge(x, y, train_mask, weak_basis, strong_basis, ridge, dim))

    rows = [result_row(result, y, train_mask, args.top_k) for result in results]
    best_key = "heldout_l2_mean" if args.best_by == "heldout_mean" else "heldout_l2_median"
    best_name = min(rows, key=lambda row: row[best_key])["name"]
    best = next(result for result in results if result.name == best_name)

    write_summary_csv(rows, output_dir / "stabilized_map_summary.csv")
    (output_dir / "stabilized_map_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_report(best, rows, output_dir / "stabilized_map_report.txt", args.top_k)
    residual_dataframe(best, artifact, args.mapping_csv).to_csv(output_dir / "best_map_residuals.csv", index=False)
    save_best_artifact(best, artifact, output_dir / "best_map_artifact.pt")
    maybe_plot(rows, best, artifact, output_dir / "plots", args.top_k)

    print(json.dumps({"best": best_name, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
