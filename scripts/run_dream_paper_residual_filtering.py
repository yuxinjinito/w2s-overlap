#!/usr/bin/env python3
"""Paper-style Dream residual filtering for W2S linear probes.

This is the follow-up to `run_dream_paper_linear_probe.py`. It keeps the
paper-style Dream setup, but adds the residual-filtering question:

- fit a weak-to-strong representation map on weak_train examples;
- score residuals on strong_train examples;
- keep the middle residual band as an overlap-like subset;
- compare it with same-size random weak-label-balanced controls.

The training/evaluation here is linear probing, not LoRA and not yes/no
generation scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from analyze_stabilized_maps import (
    fit_centered_ridge,
    fit_pca_ridge,
    fit_raw_ridge,
    maybe_plot,
    pca_basis,
    residual_dataframe,
    result_row,
    save_best_artifact,
    write_report,
    write_summary_csv,
)
from run_dream_paper_linear_probe import (
    SplitBundle,
    binary_metrics,
    extract_final_token_activations,
    fit_probe,
    load_paper_dream_splits,
    predict_probe,
    resolve_device,
    write_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak-model", default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--strong-model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-val", type=int, default=1_000)
    parser.add_argument("--n-test", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--l2-penalty", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="results/dream_paper_residual_filtering/dream_seed42")
    parser.add_argument("--residual-keep-middle-frac", type=float, default=0.5)
    parser.add_argument("--random-control-count", type=int, default=10)
    parser.add_argument("--random-control-size", type=int, default=None)
    parser.add_argument("--ridge-values", default="100.0")
    parser.add_argument("--pca-dims", default="512")
    parser.add_argument("--best-by", choices=["heldout_mean", "heldout_median"], default="heldout_median")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def split_labels(split) -> np.ndarray:
    return np.array(split["labels"], dtype=int)


def split_texts(splits: SplitBundle) -> dict[str, list[str]]:
    return {
        "weak_train": list(splits.weak_train["txt"]),
        "strong_train": list(splits.strong_train["txt"]),
        "test": list(splits.test["txt"]),
    }


def slice_activations(all_activations: torch.Tensor, sizes: dict[str, int]) -> dict[str, torch.Tensor]:
    out = {}
    start = 0
    for name in ["weak_train", "strong_train", "test"]:
        end = start + sizes[name]
        out[name] = all_activations[start:end]
        start = end
    return out


def extract_all_activations(
    model_name: str,
    texts_by_split: dict[str, list[str]],
    device: torch.device,
    dtype_arg: str,
    batch_size: int,
    max_length: int | None,
    desc: str,
    layer: str | int = "end",
    pooling: str = "last",
    answer_span: bool = False,
    answer_suffix: str = "",
    span_kind: str = "answer",
) -> dict[str, torch.Tensor]:
    sizes = {name: len(texts_by_split[name]) for name in ["weak_train", "strong_train", "test"]}
    all_texts = texts_by_split["weak_train"] + texts_by_split["strong_train"] + texts_by_split["test"]
    acts = extract_final_token_activations(
        model_name,
        all_texts,
        device,
        dtype_arg,
        batch_size,
        max_length,
        desc,
        layer=layer,
        pooling=pooling,
        answer_span=answer_span,
        answer_suffix=answer_suffix,
        span_kind=span_kind,
    )
    return slice_activations(acts, sizes)


def fit_maps(
    weak_train_acts: torch.Tensor,
    weak_strong_train_acts: torch.Tensor,
    strong_train_acts_for_map: torch.Tensor,
    strong_strong_train_acts: torch.Tensor,
    splits: SplitBundle,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
):
    x = torch.cat([weak_train_acts, weak_strong_train_acts], dim=0).float().to(device)
    y = torch.cat([strong_train_acts_for_map, strong_strong_train_acts], dim=0).float().to(device)
    train_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=device)
    train_mask[: weak_train_acts.shape[0]] = True

    ridge_values = parse_float_list(args.ridge_values)
    pca_dims = parse_int_list(args.pca_dims)

    results = []
    for ridge in ridge_values:
        results.append(fit_raw_ridge(x, y, train_mask, ridge))
        results.append(fit_centered_ridge(x, y, train_mask, ridge))

    if pca_dims:
        max_pca = max(pca_dims)
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

    map_dir = output_dir / "map"
    map_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = getattr(args, "dataset", "dream")
    artifact = {
        "weak_embeddings": x.detach().cpu(),
        "strong_embeddings": y.detach().cpu(),
        "map_train_mask": train_mask.detach().cpu(),
        "weak_model": args.weak_model,
        "strong_model": args.strong_model,
        "dataset": dataset_name,
        "target_split": "paper_style_weak_train_to_strong_train",
        "pooling": "final_token",
    }
    metadata_csv = map_dir / "mapping_metadata.csv"
    write_mapping_metadata(metadata_csv, splits, dataset_name)
    write_summary_csv(rows, map_dir / "stabilized_map_summary.csv")
    (map_dir / "stabilized_map_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_report(best, rows, map_dir / "stabilized_map_report.txt", args.top_k)
    residual_dataframe(best, artifact, str(metadata_csv)).to_csv(map_dir / "best_map_residuals.csv", index=False)
    save_best_artifact(best, artifact, map_dir / "best_map_artifact.pt")
    if not args.no_plots:
        maybe_plot(rows, best, artifact, map_dir / "plots", args.top_k)

    heldout_residuals = torch.linalg.norm(best.residuals.detach().cpu().float(), dim=1)[
        ~artifact["map_train_mask"].bool()
    ].numpy()
    return best, rows, heldout_residuals


def write_mapping_metadata(path: Path, splits: SplitBundle, dataset_name: str = "dream") -> None:
    fieldnames = ["id", "dataset", "split", "label", "text", "map_train"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, split, map_train in [
            ("weak_train", splits.weak_train, 1),
            ("strong_train", splits.strong_train, 0),
        ]:
            for idx in range(len(split)):
                writer.writerow(
                    {
                        "id": split[idx]["id"],
                        "dataset": dataset_name,
                        "split": split_name,
                        "label": int(split[idx]["labels"]),
                        "text": split[idx]["txt"],
                        "map_train": map_train,
                    }
                )


def hard_weak_label_balance(indices: np.ndarray, weak_preds: np.ndarray, seed: int, size: int | None = None) -> np.ndarray:
    by_label = {
        0: indices[weak_preds[indices] == 0],
        1: indices[weak_preds[indices] == 1],
    }
    if len(by_label[0]) == 0 or len(by_label[1]) == 0:
        return indices
    if size is None:
        per_label = min(len(by_label[0]), len(by_label[1]))
    else:
        per_label = max(1, size // 2)
        per_label = min(per_label, len(by_label[0]), len(by_label[1]))
    rng = np.random.default_rng(seed)
    selected = np.concatenate(
        [
            rng.choice(by_label[0], size=per_label, replace=False),
            rng.choice(by_label[1], size=per_label, replace=False),
        ]
    )
    return selected[rng.permutation(len(selected))]


def middle_residual_indices(residuals: np.ndarray, keep_middle_frac: float) -> tuple[np.ndarray, dict]:
    if not 0.0 < keep_middle_frac <= 1.0:
        raise ValueError("--residual-keep-middle-frac must be in (0, 1].")
    order = np.argsort(residuals)
    n_keep = max(1, int(round(len(order) * keep_middle_frac)))
    start = (len(order) - n_keep) // 2
    end = start + n_keep
    kept = order[start:end]
    kept_scores = residuals[kept]
    summary = {
        "matched_examples": int(len(order)),
        "kept_examples": int(len(kept)),
        "keep_middle_frac": float(keep_middle_frac),
        "dropped_low_examples": int(start),
        "dropped_high_examples": int(len(order) - end),
        "kept_residual_min": float(np.min(kept_scores)),
        "kept_residual_max": float(np.max(kept_scores)),
        "kept_residual_mean": float(np.mean(kept_scores)),
        "kept_residual_median": float(np.median(kept_scores)),
    }
    return kept, summary


def random_balanced_indices(weak_preds: np.ndarray, size: int, seed: int) -> np.ndarray:
    all_indices = np.arange(len(weak_preds))
    return hard_weak_label_balance(all_indices, weak_preds, seed, size=size)


def subset_summary(
    indices: np.ndarray,
    labels: np.ndarray,
    weak_probs: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, float]:
    weak_preds = (weak_probs >= 0.5).astype(int)
    return {
        "n": int(len(indices)),
        "true_label_mean": float(np.mean(labels[indices])) if len(indices) else math.nan,
        "weak_label_mean": float(np.mean(weak_preds[indices])) if len(indices) else math.nan,
        "weak_label_accuracy": float(np.mean(weak_preds[indices] == labels[indices])) if len(indices) else math.nan,
        "weak_soft_prob_mean": float(np.mean(weak_probs[indices])) if len(indices) else math.nan,
        "residual_l2_mean": float(np.mean(residuals[indices])) if len(indices) else math.nan,
        "residual_l2_median": float(np.median(residuals[indices])) if len(indices) else math.nan,
    }


def train_and_eval_w2s_probe(
    name: str,
    train_indices: np.ndarray,
    strong_train_acts: torch.Tensor,
    weak_probs_strong_train: np.ndarray,
    strong_test_acts: torch.Tensor,
    test_labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, np.ndarray]:
    targets = torch.tensor(weak_probs_strong_train[train_indices], dtype=torch.float32)
    probe = fit_probe(
        strong_train_acts[torch.tensor(train_indices, dtype=torch.long)],
        targets,
        args.l2_penalty,
        args.max_iter,
        device,
    )
    probs = predict_probe(probe, strong_test_acts, device)
    metrics = binary_metrics(probs, test_labels)
    metrics["train_examples"] = int(len(train_indices))
    metrics["run_name"] = name
    return metrics, probs


def write_selection_csv(
    path: Path,
    strong_train,
    weak_probs: np.ndarray,
    residuals: np.ndarray,
    middle_indices: np.ndarray,
    middle_balanced_indices: np.ndarray,
) -> None:
    middle_set = set(int(i) for i in middle_indices)
    middle_balanced_set = set(int(i) for i in middle_balanced_indices)
    weak_preds = (weak_probs >= 0.5).astype(int)
    ranks = np.empty_like(np.argsort(residuals))
    ranks[np.argsort(residuals)] = np.arange(len(residuals))
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "id",
        "source_id",
        "label",
        "weak_prob_label1",
        "weak_label",
        "weak_correct",
        "residual_l2",
        "residual_rank",
        "selected_middle",
        "selected_middle_balanced",
        "text",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(len(strong_train)):
            label = int(strong_train[idx]["labels"])
            writer.writerow(
                {
                    "row_index": idx,
                    "id": strong_train[idx]["id"],
                    "source_id": strong_train[idx]["source_id"],
                    "label": label,
                    "weak_prob_label1": float(weak_probs[idx]),
                    "weak_label": int(weak_preds[idx]),
                    "weak_correct": int(weak_preds[idx] == label),
                    "residual_l2": float(residuals[idx]),
                    "residual_rank": int(ranks[idx]),
                    "selected_middle": int(idx in middle_set),
                    "selected_middle_balanced": int(idx in middle_balanced_set),
                    "text": strong_train[idx]["txt"],
                }
            )


def random_control_summary(metrics: dict[str, dict]) -> dict[str, float]:
    accs = np.array([value["accuracy"] for value in metrics.values()], dtype=float)
    return {
        "count": int(len(accs)),
        "accuracy_mean": float(np.mean(accs)) if len(accs) else math.nan,
        "accuracy_std": float(np.std(accs)) if len(accs) else math.nan,
        "accuracy_min": float(np.min(accs)) if len(accs) else math.nan,
        "accuracy_max": float(np.max(accs)) if len(accs) else math.nan,
    }


def write_txt_report(path: Path, summary: dict) -> None:
    random_summary = summary["random_balanced_controls_summary"]
    lines = [
        "Dream paper-style residual filtering rerun",
        "---",
        "Setup:",
        "- dataset: Dream, paper-style binary candidate-answer correctness",
        '- prompt format: dialogue + "Q: ... A: candidate"',
        f"- weak model: {summary['weak_model']}",
        f"- strong model: {summary['strong_model']}",
        f"- weak_train: {summary['actual_sizes']['weak_train']}",
        f"- strong_train: {summary['actual_sizes']['strong_train']}",
        f"- val: {summary['actual_sizes']['val']}",
        f"- test: {summary['actual_sizes']['test']}",
        "- scoring/training: logistic linear probe",
        "- residual filtering: drop lowest/highest residuals, keep middle 50%",
        "---",
        "",
        "Map:",
        f"- best map: {summary['map']['best_name']}",
        f"- heldout L2 median: {summary['map']['heldout_l2_median']:.3f}",
        f"- heldout cosine mean: {summary['map']['heldout_cosine_mean']:.3f}",
        "---",
        "",
        "Baseline probe results:",
        f"- weak probe: {summary['metrics']['weak_probe_on_test']['accuracy']:.3f}",
        f"- strong ground-truth probe: {summary['metrics']['strong_gt_probe_on_test']['accuracy']:.3f}",
        f"- Full W2S probe: {summary['metrics']['full_w2s_probe_on_test']['accuracy']:.3f}",
        "---",
        "",
        "Residual filtering results:",
        f"- middle residual, unbalanced: {summary['metrics']['middle_residual_unbalanced']['accuracy']:.3f}",
        f"- middle residual, weak-label balanced: {summary['metrics']['middle_residual_balanced']['accuracy']:.3f}",
        (
            "- random weak-label balanced controls: "
            f"mean={random_summary['accuracy_mean']:.3f}, "
            f"std={random_summary['accuracy_std']:.3f}, "
            f"min={random_summary['accuracy_min']:.3f}, "
            f"max={random_summary['accuracy_max']:.3f}"
        ),
        "---",
        "",
        "Subset diagnostics:",
        f"- middle kept examples: {summary['middle_filter']['kept_examples']}",
        f"- middle balanced examples: {summary['subset_summaries']['middle_residual_balanced']['n']}",
        (
            "- middle balanced weak-label accuracy: "
            f"{summary['subset_summaries']['middle_residual_balanced']['weak_label_accuracy']:.3f}"
        ),
        (
            "- middle balanced weak-label positive rate: "
            f"{summary['subset_summaries']['middle_residual_balanced']['weak_label_mean']:.3f}"
        ),
        "---",
        "",
        "Takeaway:",
        "- This is the paper-style version of the earlier residual-filtering check.",
        "- It should replace the old LoRA/yes-no residual-filter values when discussing the paper-style rerun.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits = load_paper_dream_splits(args.n_train, args.n_val, args.n_test, args.seed)
    texts = split_texts(splits)
    weak_acts = extract_all_activations(
        args.weak_model,
        texts,
        device,
        args.torch_dtype,
        args.batch_size,
        args.max_length,
        "extract weak activations",
    )
    strong_acts = extract_all_activations(
        args.strong_model,
        texts,
        device,
        args.torch_dtype,
        args.batch_size,
        args.max_length,
        "extract strong activations",
    )

    y_weak_train = torch.tensor(splits.weak_train["labels"], dtype=torch.float32)
    y_strong_train_np = split_labels(splits.strong_train)
    y_test_np = split_labels(splits.test)

    weak_probe = fit_probe(weak_acts["weak_train"], y_weak_train, args.l2_penalty, args.max_iter, device)
    strong_probe = fit_probe(
        strong_acts["strong_train"],
        torch.tensor(y_strong_train_np, dtype=torch.float32),
        args.l2_penalty,
        args.max_iter,
        device,
    )
    weak_probs_strong_train = predict_probe(weak_probe, weak_acts["strong_train"], device)
    weak_probs_test = predict_probe(weak_probe, weak_acts["test"], device)
    strong_probs_test = predict_probe(strong_probe, strong_acts["test"], device)

    all_indices = np.arange(len(splits.strong_train))
    full_metrics, full_probs_test = train_and_eval_w2s_probe(
        "full_w2s",
        all_indices,
        strong_acts["strong_train"],
        weak_probs_strong_train,
        strong_acts["test"],
        y_test_np,
        args,
        device,
    )

    best_map, map_rows, residuals = fit_maps(
        weak_acts["weak_train"],
        weak_acts["strong_train"],
        strong_acts["weak_train"],
        strong_acts["strong_train"],
        splits,
        args,
        device,
        output_dir,
    )
    best_row = next(row for row in map_rows if row["name"] == best_map.name)
    middle_indices, middle_summary = middle_residual_indices(residuals, args.residual_keep_middle_frac)
    weak_preds_strong = (weak_probs_strong_train >= 0.5).astype(int)
    middle_balanced_indices = hard_weak_label_balance(
        middle_indices,
        weak_preds_strong,
        args.seed,
    )
    random_control_size = args.random_control_size or len(middle_balanced_indices)

    middle_metrics, middle_probs_test = train_and_eval_w2s_probe(
        "middle_residual_unbalanced",
        middle_indices,
        strong_acts["strong_train"],
        weak_probs_strong_train,
        strong_acts["test"],
        y_test_np,
        args,
        device,
    )
    middle_balanced_metrics, middle_balanced_probs_test = train_and_eval_w2s_probe(
        "middle_residual_balanced",
        middle_balanced_indices,
        strong_acts["strong_train"],
        weak_probs_strong_train,
        strong_acts["test"],
        y_test_np,
        args,
        device,
    )

    random_metrics: dict[str, dict] = {}
    random_prediction_columns: dict[str, np.ndarray] = {}
    random_subset_summaries: dict[str, dict] = {}
    for idx in range(args.random_control_count):
        run_name = f"random_balanced_{idx}"
        random_indices = random_balanced_indices(weak_preds_strong, random_control_size, args.seed + 1000 + idx)
        metrics, probs = train_and_eval_w2s_probe(
            run_name,
            random_indices,
            strong_acts["strong_train"],
            weak_probs_strong_train,
            strong_acts["test"],
            y_test_np,
            args,
            device,
        )
        random_metrics[run_name] = metrics
        random_prediction_columns[run_name] = probs
        random_subset_summaries[run_name] = subset_summary(
            random_indices,
            y_strong_train_np,
            weak_probs_strong_train,
            residuals,
        )

    subset_summaries = {
        "full_w2s": subset_summary(all_indices, y_strong_train_np, weak_probs_strong_train, residuals),
        "middle_residual_unbalanced": subset_summary(
            middle_indices,
            y_strong_train_np,
            weak_probs_strong_train,
            residuals,
        ),
        "middle_residual_balanced": subset_summary(
            middle_balanced_indices,
            y_strong_train_np,
            weak_probs_strong_train,
            residuals,
        ),
        **random_subset_summaries,
    }

    write_predictions(
        output_dir / "test_predictions.csv",
        splits.test,
        {
            "weak": weak_probs_test,
            "strong_gt": strong_probs_test,
            "full_w2s": full_probs_test,
            "middle_residual_unbalanced": middle_probs_test,
            "middle_residual_balanced": middle_balanced_probs_test,
            **random_prediction_columns,
        },
    )
    write_selection_csv(
        output_dir / "strong_train_residual_selection.csv",
        splits.strong_train,
        weak_probs_strong_train,
        residuals,
        middle_indices,
        middle_balanced_indices,
    )

    summary = {
        "dataset": "dream",
        "source": "paper_style_residual_filtering",
        "weak_model": args.weak_model,
        "strong_model": args.strong_model,
        "seed": args.seed,
        "requested_sizes": {
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
        },
        "actual_sizes": {
            "weak_train": len(splits.weak_train),
            "strong_train": len(splits.strong_train),
            "val": len(splits.val),
            "test": len(splits.test),
        },
        "format": {
            "task": "binary candidate-answer correctness",
            "prompt": "dialogue + 'Q: {question} A: {candidate}'",
            "label": "1 if candidate is the original Dream answer, else 0",
        },
        "activation": {
            "model_class": "AutoModelForSequenceClassification",
            "layer": "final hidden layer",
            "token": "final non-padding token",
            "pooling": "none",
            "max_length": args.max_length,
        },
        "probe": {
            "type": "LBFGS logistic probe",
            "l2_penalty": args.l2_penalty,
            "max_iter": args.max_iter,
            "w2s_target": "weak-probe soft probabilities on strong_train",
        },
        "map": {
            "best_name": best_map.name,
            "best_by": args.best_by,
            "heldout_l2_mean": float(best_row["heldout_l2_mean"]),
            "heldout_l2_median": float(best_row["heldout_l2_median"]),
            "heldout_cosine_mean": float(best_row["heldout_cosine_mean"]),
            "spectral_norm": float(best_row["spectral_norm"]),
            "top20_energy": float(best_row["top20_energy"]),
            "ridge_values": parse_float_list(args.ridge_values),
            "pca_dims": parse_int_list(args.pca_dims),
        },
        "middle_filter": middle_summary,
        "metrics": {
            "weak_probe_on_test": binary_metrics(weak_probs_test, y_test_np),
            "strong_gt_probe_on_test": binary_metrics(strong_probs_test, y_test_np),
            "full_w2s_probe_on_test": full_metrics,
            "middle_residual_unbalanced": middle_metrics,
            "middle_residual_balanced": middle_balanced_metrics,
            "random_balanced_controls": random_metrics,
        },
        "random_balanced_controls_summary": random_control_summary(random_metrics),
        "subset_summaries": subset_summaries,
        "outputs": {
            "summary": str(output_dir / "summary.json"),
            "text_report": str(output_dir / "paper_residual_filtering_report.txt"),
            "test_predictions": str(output_dir / "test_predictions.csv"),
            "strong_train_residual_selection": str(output_dir / "strong_train_residual_selection.csv"),
            "map_dir": str(output_dir / "map"),
        },
        "paper_alignment_audit": {
            "matches_original_dream_formatter": True,
            "matches_original_split_logic": True,
            "matches_original_final_token_activation": True,
            "matches_original_logistic_probe": True,
            "uses_linear_probe_not_lora": True,
            "extension_beyond_figure_a1": (
                "Residual-middle filtering and random balanced controls are targeted "
                "selection diagnostics. They are not part of the original Figure A1 baseline."
            ),
        },
        "elapsed_sec": time.time() - start,
    }

    if args.save_activations:
        torch.save(
            {
                "weak_activations": weak_acts,
                "strong_activations": strong_acts,
                "splits": {
                    name: getattr(splits, name).to_pandas()
                    for name in ["weak_train", "strong_train", "val", "test"]
                },
            },
            output_dir / "activations.pt",
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_txt_report(output_dir / "paper_residual_filtering_report.txt", summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
