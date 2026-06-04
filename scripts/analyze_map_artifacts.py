#!/usr/bin/env python3
"""Analyze learned weak-to-strong representation maps.

This reads the optional ``--map-output`` artifact produced by
``run_representation_mapping.py``. It is meant to inspect the two objects John
asked for:

- the learned linear map A from weak activations to strong activations;
- the per-sample residuals after applying that map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "map_artifact",
        help=".pt file from run_representation_mapping.py --map-output; old --embedding-output files also work",
    )
    parser.add_argument("--mapping-csv", default=None, help="Optional CSV from run_representation_mapping.py")
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--residual-output", default=None, help="Optional per-sample residual diagnostic CSV")
    parser.add_argument("--prepared-map-output", default=None, help="Optional .pt file after fitting/loading linear A")
    parser.add_argument("--plot-dir", default=None, help="Optional directory for diagnostic PNG plots")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-examples", type=int, default=8)
    parser.add_argument("--pca-lines", type=int, default=120)
    parser.add_argument("--ridge", type=float, default=1e-3, help="Ridge used if fitting A from an old embedding artifact")
    parser.add_argument("--heldout-only", action="store_true", help="Use only map_train=0 examples in plots/examples")
    return parser.parse_args()


def load_artifact(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def require_keys(artifact: dict, keys: list[str]) -> None:
    missing = [key for key in keys if key not in artifact]
    if missing:
        raise KeyError(f"Map artifact is missing required keys: {missing}")


def fit_linear_map(x_train: torch.Tensor, y_train: torch.Tensor, ridge: float) -> torch.Tensor:
    if ridge > 0:
        n = x_train.shape[0]
        gram = x_train @ x_train.T
        gram = gram + ridge * torch.eye(n, dtype=x_train.dtype, device=x_train.device)
        coeff = torch.linalg.solve(gram, y_train)
        return x_train.T @ coeff
    return torch.linalg.pinv(x_train) @ y_train


def prepare_linear_artifact(artifact: dict, ridge: float) -> dict:
    require_keys(artifact, ["weak_embeddings", "strong_embeddings", "map_train_mask"])
    weak = artifact["weak_embeddings"].float()
    strong = artifact["strong_embeddings"].float()
    train_mask = artifact["map_train_mask"].bool()

    if "linear_map" not in artifact:
        linear_map = fit_linear_map(weak[train_mask], strong[train_mask], ridge)
        artifact["linear_map"] = linear_map
        artifact["linear_map_source"] = "fit_from_embeddings"
        artifact["ridge"] = ridge
    else:
        artifact["linear_map"] = artifact["linear_map"].float()
        artifact["linear_map_source"] = artifact.get("linear_map_source", "loaded_from_artifact")

    if "linear_predictions" not in artifact:
        artifact["linear_predictions"] = weak @ artifact["linear_map"]
    else:
        artifact["linear_predictions"] = artifact["linear_predictions"].float()

    if "linear_residuals" not in artifact:
        artifact["linear_residuals"] = artifact["linear_predictions"] - strong
    else:
        artifact["linear_residuals"] = artifact["linear_residuals"].float()

    return artifact


def summarize_values(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().cpu().float()
    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def svd_diagnostics(matrix: torch.Tensor, top_k: int) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    u, singular_values, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    total_energy = singular_values.square().sum()
    top = singular_values[:top_k]
    if total_energy > 0:
        cumulative_energy = top.square().cumsum(dim=0) / total_energy
    else:
        cumulative_energy = torch.zeros_like(top)

    tol = torch.finfo(singular_values.dtype).eps * max(matrix.shape) * singular_values.max()
    rank = int((singular_values > tol).sum().item())
    positive = singular_values[singular_values > 1e-8]
    condition = float((positive.max() / positive.min()).item()) if len(positive) else None

    summary = {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "rank_estimate": rank,
        "frobenius_norm": float(torch.linalg.norm(matrix.float()).item()),
        "spectral_norm": float(singular_values.max().item()) if len(singular_values) else 0.0,
        "condition_estimate": condition,
        "top_singular_values": [float(x) for x in top.tolist()],
        "top_singular_cumulative_energy": [float(x) for x in cumulative_energy.tolist()],
    }
    return summary, u, singular_values, vh


def residual_dataframe(artifact: dict, mapping_csv: Optional[str], vh: torch.Tensor, rank: int) -> pd.DataFrame:
    residuals = artifact["linear_residuals"].float()
    linear_pred = artifact["linear_predictions"].float()
    strong = artifact["strong_embeddings"].float()
    weak = artifact["weak_embeddings"].float()
    map_train = artifact["map_train_mask"].bool()

    basis = vh[:rank].T.contiguous()
    if basis.numel() > 0:
        residual_in_map_space = residuals @ basis @ basis.T
    else:
        residual_in_map_space = torch.zeros_like(residuals)
    residual_outside_map_space = residuals - residual_in_map_space

    df = pd.DataFrame(
        {
            "row_index": np.arange(residuals.shape[0]),
            "map_train": map_train.numpy().astype(int),
            "weak_norm": torch.linalg.norm(weak, dim=1).numpy(),
            "strong_norm": torch.linalg.norm(strong, dim=1).numpy(),
            "mapped_weak_norm": torch.linalg.norm(linear_pred, dim=1).numpy(),
            "linear_residual_l2": torch.linalg.norm(residuals, dim=1).numpy(),
            "linear_residual_mse": residuals.square().mean(dim=1).numpy(),
            "linear_residual_in_map_space_l2": torch.linalg.norm(residual_in_map_space, dim=1).numpy(),
            "linear_residual_outside_map_space_l2": torch.linalg.norm(residual_outside_map_space, dim=1).numpy(),
        }
    )
    denom = df["linear_residual_l2"].replace(0, np.nan)
    df["outside_map_space_ratio"] = df["linear_residual_outside_map_space_l2"] / denom

    if mapping_csv:
        meta = pd.read_csv(mapping_csv)
        if len(meta) != len(df):
            raise ValueError(
                f"mapping CSV length ({len(meta)}) does not match artifact length ({len(df)})."
            )
        keep_cols = [col for col in ["id", "dataset", "split", "label", "text"] if col in meta.columns]
        df = pd.concat([meta[keep_cols].reset_index(drop=True), df], axis=1)

    return df


def summarize_residual_groups(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out = {
        "overall": {
            "n": int(len(df)),
            **{f"linear_residual_l2_{k}": v for k, v in summarize_series(df["linear_residual_l2"]).items()},
            **{
                f"outside_map_space_ratio_{k}": v
                for k, v in summarize_series(df["outside_map_space_ratio"].dropna()).items()
            },
        }
    }
    if "map_train" in df.columns:
        for value, name in [(1, "mapping_train"), (0, "heldout")]:
            sub = df[df["map_train"] == value]
            if len(sub) == 0:
                continue
            out[name] = {
                "n": int(len(sub)),
                **{f"linear_residual_l2_{k}": v for k, v in summarize_series(sub["linear_residual_l2"]).items()},
                **{
                    f"outside_map_space_ratio_{k}": v
                    for k, v in summarize_series(sub["outside_map_space_ratio"].dropna()).items()
                },
            }
    return out


def summarize_series(series: pd.Series) -> dict[str, float]:
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def trim_text(value: object, max_chars: int = 180) -> str:
    if not isinstance(value, str):
        return ""
    compact = " ".join(value.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def example_rows(df: pd.DataFrame, top_examples: int, heldout_only: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    view = df[df["map_train"] == 0] if heldout_only and "map_train" in df.columns else df
    high = view.sort_values("linear_residual_l2", ascending=False).head(top_examples)
    low = view.sort_values("linear_residual_l2", ascending=True).head(top_examples)
    return high, low


def render_examples(title: str, rows: pd.DataFrame) -> list[str]:
    lines = [title, "-" * len(title)]
    for rank, (_, row) in enumerate(rows.iterrows(), start=1):
        bits = [
            f"[{rank}]",
            f"row={row.get('row_index')}",
            f"map_train={row.get('map_train')}",
            f"residual_l2={row.get('linear_residual_l2'):.6f}",
            f"outside_ratio={row.get('outside_map_space_ratio'):.6f}",
        ]
        if "id" in row:
            bits.append(f"id={row.get('id')}")
        if "label" in row:
            bits.append(f"label={row.get('label')}")
        lines.append(" | ".join(bits))
        text = trim_text(row.get("text"))
        if text:
            lines.append(f"text: {text}")
    lines.append("")
    return lines


def write_report(
    args: argparse.Namespace,
    artifact: dict,
    summary: dict,
    residual_df: pd.DataFrame,
    singular_values: torch.Tensor,
) -> None:
    high, low = example_rows(residual_df, args.top_examples, args.heldout_only)
    lines = [
        "Weak-to-Strong Map Artifact Analysis",
        "====================================",
        f"Map artifact: {args.map_artifact}",
        f"Mapping CSV: {args.mapping_csv}",
        f"Weak model: {artifact.get('weak_model')}",
        f"Strong model: {artifact.get('strong_model')}",
        f"Dataset / split: {artifact.get('dataset')} / {artifact.get('target_split')}",
        f"Pooling: {artifact.get('pooling')}",
        f"Linear map source: {artifact.get('linear_map_source')}",
        f"Ridge: {artifact.get('ridge')}",
        "",
        "Linear Map A",
        "------------",
        f"shape: {summary['linear_map']['shape']}  # weak_dim x strong_dim",
        f"rank_estimate: {summary['linear_map']['rank_estimate']}",
        f"frobenius_norm: {summary['linear_map']['frobenius_norm']:.6f}",
        f"spectral_norm: {summary['linear_map']['spectral_norm']:.6f}",
        f"condition_estimate: {summary['linear_map']['condition_estimate']}",
        "",
        "Top Singular Values",
        "-------------------",
    ]
    for idx, value in enumerate(singular_values[: args.top_k].tolist(), start=1):
        lines.append(f"{idx:02d}. {value:.6f}")

    lines.extend(
        [
            "",
            "Residual Summary",
            "----------------",
        ]
    )
    for group, stats in summary["residuals"].items():
        rendered = ", ".join(
            f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in stats.items()
        )
        lines.append(f"{group}: {rendered}")

    lines.extend(
        [
            "",
            "Interpretation Notes",
            "--------------------",
            "- Large singular values are weak-space directions that the learned map strongly uses to predict strong-space directions.",
            "- Per-sample residuals measure how much strong activation remains unmatched after mapping weak activation into strong space.",
            "- Residual energy outside the learned output subspace is a rough signal for strong-space structure not captured by the linear weak-to-strong map.",
            "",
        ]
    )
    lines.extend(render_examples("Highest Residual Examples", high))
    lines.extend(render_examples("Lowest Residual Examples", low))

    output = Path(args.report_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def maybe_write_plots(
    artifact: dict,
    residual_df: pd.DataFrame,
    singular_values: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    if not args.plot_dir:
        return
    import matplotlib.pyplot as plt

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    values = singular_values.detach().cpu().numpy()
    cumulative = np.cumsum(values**2) / np.sum(values**2) if np.sum(values**2) > 0 else np.zeros_like(values)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(np.arange(1, min(len(values), args.top_k) + 1), values[: args.top_k], marker="o")
    ax1.set_xlabel("singular direction")
    ax1.set_ylabel("singular value")
    ax1.set_title("Linear Map Direction Strength")
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, min(len(values), args.top_k) + 1), cumulative[: args.top_k], color="tab:orange", marker="x")
    ax2.set_ylabel("cumulative energy")
    fig.tight_layout()
    fig.savefig(plot_dir / "linear_map_singular_values.png", dpi=200)
    plt.close(fig)

    view = residual_df
    if args.heldout_only and "map_train" in view.columns:
        view = view[view["map_train"] == 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(view["linear_residual_l2"], bins=30, color="#4c78a8", edgecolor="white")
    ax.set_xlabel("linear residual L2")
    ax.set_ylabel("number of examples")
    ax.set_title("Per-Sample Weak-to-Strong Residuals")
    fig.tight_layout()
    fig.savefig(plot_dir / "linear_residual_hist.png", dpi=200)
    plt.close(fig)

    save_pca_alignment_plot(artifact, residual_df, plot_dir / "mapped_weak_vs_strong_pca.png", args)


def save_pca_alignment_plot(artifact: dict, residual_df: pd.DataFrame, output: Path, args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    mapped = artifact["linear_predictions"].float()
    strong = artifact["strong_embeddings"].float()
    if args.heldout_only and "map_train" in residual_df.columns:
        mask = torch.tensor((residual_df["map_train"].to_numpy() == 0), dtype=torch.bool)
        mapped = mapped[mask]
        strong = strong[mask]
        residual_view = residual_df[residual_df["map_train"] == 0].reset_index(drop=True)
    else:
        residual_view = residual_df.reset_index(drop=True)

    combined = torch.cat([mapped, strong], dim=0)
    centered = combined - combined.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T
    n = mapped.shape[0]
    mapped_xy = coords[:n].numpy()
    strong_xy = coords[n:].numpy()

    residual_l2 = residual_view["linear_residual_l2"].to_numpy()
    if n > args.pca_lines:
        line_idx = np.linspace(0, n - 1, args.pca_lines).astype(int)
    else:
        line_idx = np.arange(n)

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx in line_idx:
        ax.plot(
            [mapped_xy[idx, 0], strong_xy[idx, 0]],
            [mapped_xy[idx, 1], strong_xy[idx, 1]],
            color="#999999",
            alpha=0.25,
            linewidth=0.8,
        )
    scatter = ax.scatter(strong_xy[:, 0], strong_xy[:, 1], c=residual_l2, cmap="magma", s=26, label="strong")
    ax.scatter(mapped_xy[:, 0], mapped_xy[:, 1], color="#4c78a8", s=14, alpha=0.65, label="mapped weak")
    ax.set_title("Mapped Weak vs Strong Activations in PCA Space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="linear residual L2")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    artifact = load_artifact(args.map_artifact)
    artifact = prepare_linear_artifact(artifact, args.ridge)

    linear_summary, _, singular_values, vh = svd_diagnostics(artifact["linear_map"], args.top_k)
    rank = linear_summary["rank_estimate"]
    residual_df = residual_dataframe(artifact, args.mapping_csv, vh, rank)

    summary = {
        "map_artifact": args.map_artifact,
        "mapping_csv": args.mapping_csv,
        "weak_model": artifact.get("weak_model"),
        "strong_model": artifact.get("strong_model"),
        "dataset": artifact.get("dataset"),
        "target_split": artifact.get("target_split"),
        "pooling": artifact.get("pooling"),
        "linear_map_source": artifact.get("linear_map_source"),
        "ridge": artifact.get("ridge"),
        "linear_map": linear_summary,
        "residuals": summarize_residual_groups(residual_df),
    }

    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.residual_output:
        residual_output = Path(args.residual_output)
        residual_output.parent.mkdir(parents=True, exist_ok=True)
        residual_df.to_csv(residual_output, index=False)

    if args.prepared_map_output:
        prepared_output = Path(args.prepared_map_output)
        prepared_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, prepared_output)

    write_report(args, artifact, summary, residual_df, singular_values)
    maybe_write_plots(artifact, residual_df, singular_values, args)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
