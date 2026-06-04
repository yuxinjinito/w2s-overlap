#!/usr/bin/env python3
"""Dream LoRA W2S experiment with paper-style data formatting.

This is the downstream-training counterpart to the paper-style linear-probe
rerun. It keeps the Dream split and candidate-answer formatting close to the
original repo, but trains/evaluates a causal LM with LoRA on yes/no targets.

The input text is the original-style candidate statement:

    dialogue

    Q: question A: candidate

For causal-LM scoring/training, we append a short answer slot asking whether the
candidate is correct. This keeps the task readable for a generative model while
avoiding the older verbose "Dialogue:/Candidate answer:" format.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset

from run_dream_paper_linear_probe import (
    SplitBundle,
    balance_binary_dataset,
    extract_final_token_activations,
    fit_probe,
    load_paper_dream_splits,
    predict_probe,
    resolve_device,
)
from run_dream_paper_residual_filtering import (
    fit_maps,
    hard_weak_label_balance,
    middle_residual_indices,
    random_balanced_indices,
    subset_summary,
)
from run_dream_w2s_baselines import (
    clear_memory,
    evaluate_yes_no,
    train_lora_model,
    write_predictions,
)


@dataclass
class LoraExample:
    id: str
    source_id: str
    text: str
    label: int
    answer_suffix: str

    @property
    def prompt(self) -> str:
        return f"{self.text}{self.answer_suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dream", "sciq", "paws"], default="dream")
    parser.add_argument(
        "--sciq-use-support",
        action="store_true",
        help=(
            "For SciQ only, include the support passage in the prompt. "
            "Default false matches the original datacentric_w2s SciQ formatter."
        ),
    )
    parser.add_argument("--weak-model", default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--strong-model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--output-dir", default="results/dream_paper_style_lora/dream_seed42")
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-val", type=int, default=1_000)
    parser.add_argument("--n-test", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="Optional seed for LoRA initialization/training only. Keeps data splits and filters fixed.",
    )
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--activation-max-length", type=int, default=None)
    parser.add_argument("--answer-suffix", default="\nIs the candidate answer correct? Answer:")
    parser.add_argument("--weak-batch-size", type=int, default=4)
    parser.add_argument("--activation-batch-size", type=int, default=1)
    parser.add_argument("--l2-penalty", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--strong-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-adapters", action="store_true")
    parser.add_argument(
        "--runs",
        default="base,ground_truth,weak_label,middle_balanced,random_balanced",
        help=(
            "Comma-separated runs. Options: base, ground_truth, weak_label, "
            "middle_unbalanced, middle_balanced, confidence_middle_unbalanced, "
            "confidence_middle_balanced, confidence_high_unbalanced, "
            "confidence_high_balanced, knn_middle_unbalanced, "
            "knn_middle_balanced, knn_mixed_unbalanced, "
            "knn_mixed_balanced, random_unbalanced, random_balanced."
        ),
    )
    parser.add_argument("--residual-keep-middle-frac", type=float, default=0.5)
    parser.add_argument("--confidence-keep-frac", type=float, default=0.5)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-keep-middle-frac", type=float, default=0.5)
    parser.add_argument("--knn-mixed-center", type=float, default=0.5)
    parser.add_argument(
        "--knn-reference-cross-fit",
        action="store_true",
        help=(
            "Label weak_train reference points weak-correct/weak-wrong using "
            "cross-fitted (out-of-fold) weak-probe predictions instead of "
            "in-sample predictions. Avoids the kNN saturation that happens when "
            "the probe is near-perfect on its own training set."
        ),
    )
    parser.add_argument("--cross-fit-folds", type=int, default=5)
    parser.add_argument("--random-control-count", type=int, default=3)
    parser.add_argument("--random-control-size", type=int, default=None)
    parser.add_argument("--random-unbalanced-size", type=int, default=None)
    parser.add_argument("--ridge-values", default="100.0")
    parser.add_argument("--pca-dims", default="")
    parser.add_argument("--best-by", choices=["heldout_mean", "heldout_median"], default="heldout_median")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-activations", action="store_true")
    return parser.parse_args()


def requested_runs(args: argparse.Namespace) -> list[str]:
    runs = [item.strip() for item in args.runs.split(",") if item.strip()]
    allowed = {
        "base",
        "ground_truth",
        "weak_label",
        "middle_unbalanced",
        "middle_balanced",
        "confidence_middle_unbalanced",
        "confidence_middle_balanced",
        "confidence_high_unbalanced",
        "confidence_high_balanced",
        "knn_middle_unbalanced",
        "knn_middle_balanced",
        "knn_mixed_unbalanced",
        "knn_mixed_balanced",
        "random_unbalanced",
        "random_balanced",
    }
    unknown = sorted(set(runs) - allowed)
    if unknown:
        raise SystemExit(f"Unknown run(s): {', '.join(unknown)}")
    return runs


def to_lora_examples(split, answer_suffix: str) -> list[LoraExample]:
    examples = []
    for idx in range(len(split)):
        examples.append(
            LoraExample(
                id=split[idx]["id"],
                source_id=split[idx]["source_id"],
                text=split[idx]["txt"],
                label=int(split[idx]["labels"]),
                answer_suffix=answer_suffix,
            )
        )
    return examples


def format_sciq_paper_style(ex: dict, row_id: int, rng: random.Random, use_support: bool) -> dict:
    """Format SciQ as binary candidate-answer correctness.

    The no-support default matches the original datacentric_w2s `sciq`
    formatter: `Q: {question} A: {candidate}`. The support-passage variant is
    kept as an explicit ablation because it can make the binary task easier.
    """

    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        distractors = [ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        ans = rng.choice(distractors)

    if use_support:
        support = (ex.get("support") or "").strip()
        support_block = f"Support: {support}\n\n" if support else ""
    else:
        support_block = ""
    txt = f"{support_block}Q: {ex['question']} A: {ans}"
    return {
        "id": f"sciq-{row_id}",
        "source_id": f"sciq-{row_id}",
        "txt": txt,
        "labels": hard_label,
        "gt_labels": hard_label,
    }


def load_and_process_sciq_split(split: str, n_docs: int, seed: int, use_support: bool) -> Dataset:
    raw = load_dataset("sciq", split=split).shuffle(seed=seed)
    if len(raw) < n_docs:
        print(f"sciq/{split} has < {n_docs} raw docs, using all {len(raw)}")
    raw = raw.select(range(min(n_docs, len(raw))))
    rng = random.Random(seed)
    formatted_rows = [
        format_sciq_paper_style(ex, row_id=i, rng=rng, use_support=use_support)
        for i, ex in enumerate(raw)
    ]
    ds = Dataset.from_list(formatted_rows)
    ds = ds.filter(lambda ex: ex["txt"] != "")
    ds = balance_binary_dataset(ds, seed)
    return ds


def load_paper_sciq_splits(
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    use_support: bool,
) -> SplitBundle:
    train_pool = load_and_process_sciq_split("train", n_train + n_val, seed, use_support)
    test = load_and_process_sciq_split("test", n_test, seed + 2, use_support)

    val_count = min(n_val, len(train_pool))
    val = train_pool.select(range(val_count))
    train = train_pool.select(range(val_count, len(train_pool)))
    train_halves = train.train_test_split(test_size=0.5, seed=seed)

    return SplitBundle(
        weak_train=train_halves["train"],
        strong_train=train_halves["test"],
        val=val,
        test=test,
    )


def format_paws_paper_style(ex: dict, row_id: int) -> dict:
    txt = (
        f"Sent 1: {ex['sentence1']}\n"
        f"Sent 2: {ex['sentence2']}\n\n"
        "Q: Are these sentences semantically equivalent?"
    )
    hard_label = int(ex["label"])
    return {
        "id": f"paws-{row_id}",
        "source_id": f"paws-{row_id}",
        "txt": txt,
        "labels": hard_label,
        "gt_labels": hard_label,
    }


def load_and_process_paws_split(split: str, n_docs: int, seed: int) -> Dataset:
    raw = load_dataset("paws", "labeled_final", split=split).shuffle(seed=seed)
    formatted_rows = [
        format_paws_paper_style(ex, row_id=i)
        for i, ex in enumerate(raw)
    ]
    ds = Dataset.from_list(formatted_rows)
    ds = ds.filter(lambda ex: ex["txt"] != "")
    ds = balance_binary_dataset(ds, seed)
    if len(ds) < n_docs:
        print(f"paws/{split} has < {n_docs} docs after balancing, using all {len(ds)}")
    return ds.select(range(min(n_docs, len(ds))))


def load_paper_paws_splits(n_train: int, n_val: int, n_test: int, seed: int) -> SplitBundle:
    train_pool = load_and_process_paws_split("train", n_train + n_val, seed)
    test = load_and_process_paws_split("test", n_test, seed + 2)

    val_count = min(n_val, len(train_pool))
    val = train_pool.select(range(val_count))
    train = train_pool.select(range(val_count, len(train_pool)))
    train_halves = train.train_test_split(test_size=0.5, seed=seed)

    return SplitBundle(
        weak_train=train_halves["train"],
        strong_train=train_halves["test"],
        val=val,
        test=test,
    )


def load_paper_style_splits(args: argparse.Namespace) -> SplitBundle:
    if args.dataset == "dream":
        return load_paper_dream_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "sciq":
        return load_paper_sciq_splits(
            args.n_train,
            args.n_val,
            args.n_test,
            args.seed,
            args.sciq_use_support,
        )
    if args.dataset == "paws":
        return load_paper_paws_splits(args.n_train, args.n_val, args.n_test, args.seed)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def format_summary(args: argparse.Namespace) -> dict[str, str]:
    if args.dataset == "dream":
        return {
            "task": "paper-style binary candidate-answer correctness",
            "candidate_text": "dialogue + 'Q: {question} A: {candidate}'",
            "label": "1 if candidate is the original Dream answer, else 0",
            "takeaway": "This run uses LoRA with the paper-style Dream candidate-answer setup.",
        }
    if args.dataset == "sciq":
        candidate_text = (
            "support + 'Q: {question} A: {candidate}'"
            if args.sciq_use_support
            else "'Q: {question} A: {candidate}'"
        )
        takeaway = (
            "This run uses the SciQ support-passage ablation."
            if args.sciq_use_support
            else "This run uses the original datacentric_w2s SciQ no-support candidate-answer setup."
        )
        return {
            "task": "paper-style binary candidate-answer correctness",
            "candidate_text": candidate_text,
            "label": "1 if candidate is the SciQ correct answer, else 0",
            "takeaway": takeaway,
        }
    if args.dataset == "paws":
        return {
            "task": "paper-style binary semantic-equivalence classification",
            "candidate_text": "'Sent 1: {sentence1}\\nSent 2: {sentence2}\\n\\nQ: Are these sentences semantically equivalent?'",
            "label": "1 if the two sentences are semantically equivalent, else 0",
            "takeaway": "This run uses the original datacentric_w2s PAWS semantic-equivalence prompt.",
        }
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def split_texts(splits: SplitBundle) -> dict[str, list[str]]:
    return {
        "weak_train": list(splits.weak_train["txt"]),
        "strong_train": list(splits.strong_train["txt"]),
        "test": list(splits.test["txt"]),
    }


def split_labels(split) -> np.ndarray:
    return np.array(split["labels"], dtype=int)


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
    )
    return slice_activations(acts, sizes)


def eval_rows_from_probs(examples: list[LoraExample], probs: np.ndarray) -> list[dict]:
    rows = []
    preds = (probs >= 0.5).astype(int)
    for ex, prob, pred in zip(examples, probs, preds):
        rows.append(
            {
                "id": ex.id,
                "label": ex.label,
                "prob_label1": float(prob),
                "pred": int(pred),
                "correct": int(pred == ex.label),
            }
        )
    return rows


def metric_from_rows(rows: list[dict]) -> dict[str, float]:
    confidence = [2.0 * abs(float(row["prob_label1"]) - 0.5) for row in rows]
    return {
        "n": len(rows),
        "accuracy": float(np.mean([int(row["correct"]) for row in rows])) if rows else math.nan,
        "pred_label1_rate": float(np.mean([int(row["pred"]) for row in rows])) if rows else math.nan,
        "prob_label1_mean": float(np.mean([float(row["prob_label1"]) for row in rows])) if rows else math.nan,
        "confidence_mean": float(np.mean(confidence)) if confidence else math.nan,
        "confidence_median": float(np.median(confidence)) if confidence else math.nan,
    }


def weak_label_balance(
    examples: list[LoraExample],
    labels: list[int],
    seed: int,
) -> tuple[list[LoraExample], list[int], np.ndarray]:
    indices = np.arange(len(examples))
    weak_preds = np.array(labels, dtype=int)
    selected = hard_weak_label_balance(indices, weak_preds, seed)
    return [examples[int(i)] for i in selected], [int(labels[int(i)]) for i in selected], selected


def score_band_indices(scores: np.ndarray, keep_frac: float, mode: str) -> tuple[np.ndarray, dict[str, float | str]]:
    if not 0.0 < keep_frac <= 1.0:
        raise ValueError("keep_frac must be in (0, 1].")
    if mode not in {"middle", "high"}:
        raise ValueError(f"Unknown score band mode: {mode}")

    order = np.argsort(scores)
    n_keep = max(1, int(round(len(order) * keep_frac)))
    if mode == "middle":
        start = (len(order) - n_keep) // 2
        end = start + n_keep
        kept = order[start:end]
        dropped_low = start
        dropped_high = len(order) - end
    else:
        kept = order[-n_keep:]
        dropped_low = len(order) - n_keep
        dropped_high = 0

    kept_scores = scores[kept]
    return kept, {
        "mode": mode,
        "matched_examples": int(len(order)),
        "kept_examples": int(len(kept)),
        "keep_frac": float(keep_frac),
        "dropped_low_examples": int(dropped_low),
        "dropped_high_examples": int(dropped_high),
        "kept_score_min": float(np.min(kept_scores)),
        "kept_score_max": float(np.max(kept_scores)),
        "kept_score_mean": float(np.mean(kept_scores)),
        "kept_score_median": float(np.median(kept_scores)),
    }


def score_closest_indices(
    scores: np.ndarray,
    keep_frac: float,
    center: float,
    mode_name: str,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Keep examples whose score is closest to a target value.

    For kNN filtering, John's intended "overlap-like" points are not
    necessarily the middle quantile of the score distribution. They are points
    whose neighborhoods are mixed between weak-correct and weak-wrong reference
    examples. With a correct-neighbor-rate score, that means closest to 0.5.
    """

    if not 0.0 < keep_frac <= 1.0:
        raise ValueError("keep_frac must be in (0, 1].")

    order = np.argsort(np.abs(scores - center))
    n_keep = max(1, int(round(len(order) * keep_frac)))
    kept = order[:n_keep]
    kept_scores = scores[kept]
    return kept, {
        "mode": mode_name,
        "matched_examples": int(len(order)),
        "kept_examples": int(len(kept)),
        "keep_frac": float(keep_frac),
        "center": float(center),
        "kept_score_min": float(np.min(kept_scores)),
        "kept_score_max": float(np.max(kept_scores)),
        "kept_score_mean": float(np.mean(kept_scores)),
        "kept_score_median": float(np.median(kept_scores)),
        "kept_abs_distance_to_center_mean": float(np.mean(np.abs(kept_scores - center))),
        "kept_abs_distance_to_center_median": float(np.median(np.abs(kept_scores - center))),
    }


def cross_fitted_weak_probs(
    activations: torch.Tensor,
    labels: np.ndarray,
    folds: int,
    l2_penalty: float,
    max_iter: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    """Out-of-fold weak-probe probabilities on the weak_train reference set.

    Each weak_train point is scored by a probe that was NOT trained on it: the
    set is split into ``folds`` folds, and for each fold a probe is fit on the
    other folds and used to predict the held-out fold. This gives honest
    weak-correct / weak-wrong reference labels for the kNN filter and avoids the
    in-sample saturation that occurs when the probe predicts its own training
    data (an overfit probe labels almost every reference point weak-correct,
    leaving no weak-wrong neighbors for the mixed-neighborhood signal).
    """
    n = activations.shape[0]
    folds = max(2, min(folds, n))
    order = np.random.default_rng(seed).permutation(n)
    labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
    probs = np.zeros(n, dtype=np.float32)
    for held_out in np.array_split(order, folds):
        held_out_t = torch.as_tensor(held_out, dtype=torch.long)
        train_mask = torch.ones(n, dtype=torch.bool)
        train_mask[held_out_t] = False
        train_idx_t = torch.nonzero(train_mask, as_tuple=False).squeeze(1)
        probe = fit_probe(
            activations[train_idx_t],
            labels_t[train_idx_t],
            l2_penalty,
            max_iter,
            device,
        )
        probs[held_out] = predict_probe(probe, activations[held_out_t], device)
    return probs


def compute_weak_train_knn_stats(
    reference_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    reference_weak_correct: np.ndarray,
    k: int,
) -> dict[str, np.ndarray]:
    if k <= 0:
        raise ValueError("--knn-k must be positive.")
    if len(reference_embeddings) == 0:
        raise ValueError("Need at least one weak_train reference embedding.")
    k = min(k, len(reference_embeddings))

    ref = torch.nn.functional.normalize(reference_embeddings.to(torch.float32), p=2, dim=1)
    query = torch.nn.functional.normalize(query_embeddings.to(torch.float32), p=2, dim=1)
    sims = query @ ref.T
    top_values, top_indices = torch.topk(sims, k=k, dim=1)

    correct = torch.tensor(reference_weak_correct.astype(np.float32))
    neighbor_correct = correct[top_indices]
    correct_count = neighbor_correct.sum(dim=1)
    correct_rate = correct_count / float(k)
    return {
        "neighbor_indices": top_indices.cpu().numpy(),
        "neighbor_cosine_mean": top_values.mean(dim=1).cpu().numpy(),
        "neighbor_cosine_min": top_values.min(dim=1).values.cpu().numpy(),
        "knn_correct_count": correct_count.cpu().numpy(),
        "knn_correct_rate": correct_rate.cpu().numpy(),
    }


def write_strong_train_labels(
    path: Path,
    examples: list[LoraExample],
    weak_probs: np.ndarray,
    residuals: np.ndarray,
    knn_stats=None,
) -> None:
    weak_preds = (weak_probs >= 0.5).astype(int)
    ranks = np.empty_like(np.argsort(residuals))
    ranks[np.argsort(residuals)] = np.arange(len(residuals))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "source_id",
                "label",
                "weak_prob_label1",
                "weak_confidence",
                "weak_label",
                "weak_correct",
                "residual_l2",
                "residual_rank",
                "knn_correct_count",
                "knn_correct_rate",
                "knn_neighbor_cosine_mean",
                "knn_neighbor_cosine_min",
                "text",
                "prompt",
            ],
        )
        writer.writeheader()
        for idx, ex in enumerate(examples):
            writer.writerow(
                {
                    "id": ex.id,
                    "source_id": ex.source_id,
                    "label": ex.label,
                    "weak_prob_label1": float(weak_probs[idx]),
                    "weak_confidence": float(2.0 * abs(float(weak_probs[idx]) - 0.5)),
                    "weak_label": int(weak_preds[idx]),
                    "weak_correct": int(weak_preds[idx] == ex.label),
                    "residual_l2": float(residuals[idx]),
                    "residual_rank": int(ranks[idx]),
                    "knn_correct_count": ""
                    if knn_stats is None
                    else int(knn_stats["knn_correct_count"][idx]),
                    "knn_correct_rate": ""
                    if knn_stats is None
                    else float(knn_stats["knn_correct_rate"][idx]),
                    "knn_neighbor_cosine_mean": ""
                    if knn_stats is None
                    else float(knn_stats["neighbor_cosine_mean"][idx]),
                    "knn_neighbor_cosine_min": ""
                    if knn_stats is None
                    else float(knn_stats["neighbor_cosine_min"][idx]),
                    "text": ex.text,
                    "prompt": ex.prompt,
                }
            )


def write_knn_diagnostics(
    path: Path,
    examples: list[LoraExample],
    labels: np.ndarray,
    weak_probs: np.ndarray,
    residuals: np.ndarray,
    knn_stats: dict[str, np.ndarray],
) -> None:
    weak_preds = (weak_probs >= 0.5).astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "source_id",
                "label",
                "weak_label",
                "weak_correct",
                "weak_confidence",
                "residual_l2",
                "knn_correct_count",
                "knn_correct_rate",
                "knn_neighbor_cosine_mean",
                "knn_neighbor_cosine_min",
                "text",
            ],
        )
        writer.writeheader()
        for idx, ex in enumerate(examples):
            writer.writerow(
                {
                    "id": ex.id,
                    "source_id": ex.source_id,
                    "label": int(labels[idx]),
                    "weak_label": int(weak_preds[idx]),
                    "weak_correct": int(weak_preds[idx] == labels[idx]),
                    "weak_confidence": float(2.0 * abs(float(weak_probs[idx]) - 0.5)),
                    "residual_l2": float(residuals[idx]),
                    "knn_correct_count": int(knn_stats["knn_correct_count"][idx]),
                    "knn_correct_rate": float(knn_stats["knn_correct_rate"][idx]),
                    "knn_neighbor_cosine_mean": float(knn_stats["neighbor_cosine_mean"][idx]),
                    "knn_neighbor_cosine_min": float(knn_stats["neighbor_cosine_min"][idx]),
                    "text": ex.text,
                }
            )


def write_subset_csv(
    path: Path,
    subsets: dict[str, tuple[list[LoraExample], list[int]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_name", "id", "source_id", "label", "train_label", "train_correct", "text", "prompt"],
        )
        writer.writeheader()
        for run_name, (examples, labels) in subsets.items():
            for ex, label in zip(examples, labels):
                writer.writerow(
                    {
                        "run_name": run_name,
                        "id": ex.id,
                        "source_id": ex.source_id,
                        "label": ex.label,
                        "train_label": int(label),
                        "train_correct": int(int(label) == ex.label),
                        "text": ex.text,
                        "prompt": ex.prompt,
                    }
                )


def summarize_train_subset(examples: list[LoraExample], labels: list[int]) -> dict[str, float]:
    return {
        "n": len(examples),
        "true_label_mean": float(np.mean([ex.label for ex in examples])) if examples else math.nan,
        "train_label_mean": float(np.mean(labels)) if labels else math.nan,
        "train_label_accuracy": float(np.mean([int(label == ex.label) for ex, label in zip(examples, labels)]))
        if examples
        else math.nan,
    }


def random_unbalanced_indices(n_examples: int, size: int, seed: int) -> np.ndarray:
    size = min(size, n_examples)
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(n_examples), size=size, replace=False)


def run_lora_eval(
    args: argparse.Namespace,
    run_name: str,
    train_examples: list[LoraExample],
    train_labels: list[int],
    eval_examples: list[LoraExample],
    output_dir: Path,
) -> tuple[dict, list[dict]]:
    if args.train_seed is not None:
        torch.manual_seed(args.train_seed)
        np.random.seed(args.train_seed)
        random.seed(args.train_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.train_seed)
    model, tokenizer, report = train_lora_model(args, train_examples, train_labels, run_name, output_dir)
    eval_summary, rows = evaluate_yes_no(
        model,
        tokenizer,
        eval_examples,
        args.strong_batch_size,
        args.device,
        args.max_length,
        f"eval {run_name}",
    )
    del model
    del tokenizer
    clear_memory()
    return {
        **report,
        "train_seed": args.train_seed,
        "train_subset": summarize_train_subset(train_examples, train_labels),
        "eval": eval_summary,
    }, rows


def write_text_report(path: Path, summary: dict) -> None:
    runs = summary["runs"]
    lines = [
        f"{summary['dataset']} paper-style prompt LoRA rerun",
        "---",
        "Setup:",
        f"- dataset: {summary['dataset']}, {summary['format']['task']}",
        f"- candidate text: {summary['format']['candidate_text']}",
        f"- LoRA prompt suffix: {summary['answer_suffix']!r}",
        f"- weak model: {summary['weak_model']}",
        f"- strong model: {summary['strong_model']}",
        f"- weak_train: {summary['actual_sizes']['weak_train']}",
        f"- strong_train: {summary['actual_sizes']['strong_train']}",
        f"- val: {summary['actual_sizes']['val']}",
        f"- test: {summary['actual_sizes']['test']}",
        f"- max_train_steps: {summary['lora']['max_train_steps']}",
        f"- lr: {summary['lora']['lr']}",
        f"- weight_decay: {summary['lora']['weight_decay']}",
        f"- warmup_steps: {summary['lora']['warmup_steps']}",
        f"- max_grad_norm: {summary['lora']['max_grad_norm']}",
        f"- lora_target_modules: {', '.join(summary['lora']['lora_target_modules'])}",
        "---",
        "",
        "Weak-label diagnostics:",
        f"- weak-label accuracy on strong_train: {summary['weak_label_diagnostics']['accuracy']:.3f}",
        f"- weak-label positive rate on strong_train: {summary['weak_label_diagnostics']['positive_rate']:.3f}",
        "---",
        "",
        "Map:",
        f"- best map: {summary['map']['best_name']}",
        f"- heldout L2 median: {summary['map']['heldout_l2_median']:.3f}",
        f"- heldout cosine mean: {summary['map']['heldout_cosine_mean']:.3f}",
        "---",
        "",
        "kNN diagnostic:",
        f"- reference/query: {summary['knn_filter']['reference_split']} -> {summary['knn_filter']['query_split']}",
        f"- k: {summary['knn_filter']['k']}",
        f"- correct-neighbor-rate mean: {summary['knn_filter']['knn_correct_rate_mean']:.3f}",
        f"- correct-neighbor-rate median: {summary['knn_filter']['knn_correct_rate_median']:.3f}",
    ]
    if "mixed" in summary["knn_filter"]:
        mixed = summary["knn_filter"]["mixed"]
        lines.extend(
            [
                f"- mixed filter center: {mixed['center']:.3f}",
                f"- mixed kept score range: {mixed['kept_score_min']:.3f}-{mixed['kept_score_max']:.3f}",
                f"- mixed kept score median: {mixed['kept_score_median']:.3f}",
            ]
        )
    lines.extend(["---", "", "LoRA results:"])
    for name in [
        "base",
        "ground_truth",
        "weak_label",
        "middle_unbalanced",
        "middle_balanced",
        "confidence_middle_unbalanced",
        "confidence_middle_balanced",
        "confidence_high_unbalanced",
        "confidence_high_balanced",
        "knn_middle_unbalanced",
        "knn_middle_balanced",
        "knn_mixed_unbalanced",
        "knn_mixed_balanced",
        "random_unbalanced",
    ]:
        if name in runs:
            acc = runs[name]["eval"]["accuracy"]
            subset = summary.get("subset_summaries", {}).get(name)
            strong_n = summary["actual_sizes"].get("strong_train")
            if subset and subset.get("n") and strong_n:
                kept = subset["n"]
                lines.append(
                    f"- {name}: {acc:.3f} "
                    f"(kept {kept}/{strong_n} = {100.0 * kept / strong_n:.1f}% not filtered)"
                )
            else:
                lines.append(f"- {name}: {acc:.3f}")
    random_unbalanced = summary.get("random_unbalanced_controls_summary")
    if random_unbalanced:
        lines.append(
            "- random_unbalanced controls: "
            f"mean={random_unbalanced['accuracy_mean']:.3f}, "
            f"std={random_unbalanced['accuracy_std']:.3f}, "
            f"min={random_unbalanced['accuracy_min']:.3f}, "
            f"max={random_unbalanced['accuracy_max']:.3f}"
        )
    random = summary.get("random_balanced_controls_summary")
    if random:
        lines.append(
            "- random_balanced controls: "
            f"mean={random['accuracy_mean']:.3f}, "
            f"std={random['accuracy_std']:.3f}, "
            f"min={random['accuracy_min']:.3f}, "
            f"max={random['accuracy_max']:.3f}"
        )
    lines.extend(
        [
            "---",
            "",
            "Takeaway:",
            f"- {summary['format']['takeaway']}",
            "- Compare filtering methods against weak-label full training and random controls before treating a selection score as useful.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = requested_runs(args)
    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this script on a CUDA GPU machine.")
    args.device = str(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits = load_paper_style_splits(args)
    lora_examples = {
        "weak_train": to_lora_examples(splits.weak_train, args.answer_suffix),
        "strong_train": to_lora_examples(splits.strong_train, args.answer_suffix),
        "test": to_lora_examples(splits.test, args.answer_suffix),
    }
    texts = split_texts(splits)

    weak_acts = extract_all_activations(
        args.weak_model,
        texts,
        device,
        args.torch_dtype,
        args.activation_batch_size,
        args.activation_max_length,
        "extract weak activations",
    )
    weak_probe = fit_probe(
        weak_acts["weak_train"],
        torch.tensor(split_labels(splits.weak_train), dtype=torch.float32),
        args.l2_penalty,
        args.max_iter,
        device,
    )
    weak_probs_weak_train = predict_probe(weak_probe, weak_acts["weak_train"], device)
    weak_probs_strong = predict_probe(weak_probe, weak_acts["strong_train"], device)
    weak_probs_test = predict_probe(weak_probe, weak_acts["test"], device)
    weak_train_labels = split_labels(splits.weak_train)
    weak_preds_weak_train = (weak_probs_weak_train >= 0.5).astype(int)
    if args.knn_reference_cross_fit:
        cross_fit_probs_weak_train = cross_fitted_weak_probs(
            weak_acts["weak_train"],
            weak_train_labels,
            args.cross_fit_folds,
            args.l2_penalty,
            args.max_iter,
            device,
            args.seed,
        )
        reference_preds_weak_train = (cross_fit_probs_weak_train >= 0.5).astype(int)
        weak_correct_weak_train = (reference_preds_weak_train == weak_train_labels).astype(int)
        print(
            "[cross-fit] weak_train reference accuracy: "
            f"in-sample={float(np.mean(weak_preds_weak_train == weak_train_labels)):.3f}, "
            f"cross-fitted={float(np.mean(weak_correct_weak_train)):.3f} "
            f"(folds={max(2, min(args.cross_fit_folds, len(weak_train_labels)))})"
        )
    else:
        weak_correct_weak_train = (weak_preds_weak_train == weak_train_labels).astype(int)
    weak_preds_strong = (weak_probs_strong >= 0.5).astype(int)
    weak_preds_test = (weak_probs_test >= 0.5).astype(int)

    strong_train_labels = split_labels(splits.strong_train)
    test_labels = split_labels(splits.test)
    weak_eval_rows = eval_rows_from_probs(lora_examples["test"], weak_probs_test)

    strong_acts = extract_all_activations(
        args.strong_model,
        texts,
        device,
        args.torch_dtype,
        args.activation_batch_size,
        args.activation_max_length,
        "extract strong activations",
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
    knn_stats = compute_weak_train_knn_stats(
        strong_acts["weak_train"],
        strong_acts["strong_train"],
        weak_correct_weak_train,
        args.knn_k,
    )
    del strong_acts
    clear_memory()

    best_row = next(row for row in map_rows if row["name"] == best_map.name)
    middle_indices, middle_filter = middle_residual_indices(residuals, args.residual_keep_middle_frac)
    middle_balanced_indices = hard_weak_label_balance(
        middle_indices,
        weak_preds_strong,
        args.seed,
    )
    weak_confidences_strong = 2.0 * np.abs(weak_probs_strong - 0.5)
    confidence_middle_indices, confidence_middle_filter = score_band_indices(
        weak_confidences_strong,
        args.confidence_keep_frac,
        "middle",
    )
    confidence_middle_balanced_indices = hard_weak_label_balance(
        confidence_middle_indices,
        weak_preds_strong,
        args.seed + 2000,
    )
    confidence_high_indices, confidence_high_filter = score_band_indices(
        weak_confidences_strong,
        args.confidence_keep_frac,
        "high",
    )
    confidence_high_balanced_indices = hard_weak_label_balance(
        confidence_high_indices,
        weak_preds_strong,
        args.seed + 3000,
    )
    knn_middle_indices, knn_middle_filter = score_band_indices(
        knn_stats["knn_correct_rate"],
        args.knn_keep_middle_frac,
        "middle",
    )
    knn_middle_balanced_indices = hard_weak_label_balance(
        knn_middle_indices,
        weak_preds_strong,
        args.seed + 4000,
    )
    knn_mixed_indices, knn_mixed_filter = score_closest_indices(
        knn_stats["knn_correct_rate"],
        args.knn_keep_middle_frac,
        args.knn_mixed_center,
        "mixed",
    )
    knn_mixed_balanced_indices = hard_weak_label_balance(
        knn_mixed_indices,
        weak_preds_strong,
        args.seed + 5000,
    )
    random_control_size = args.random_control_size or len(middle_balanced_indices)

    strong_examples = lora_examples["strong_train"]
    eval_examples = lora_examples["test"]
    weak_labels = weak_preds_strong.tolist()
    run_subsets: dict[str, tuple[list[LoraExample], list[int]]] = {
        "ground_truth": (strong_examples, strong_train_labels.tolist()),
        "weak_label": (strong_examples, weak_labels),
        "middle_unbalanced": (
            [strong_examples[int(i)] for i in middle_indices],
            [weak_labels[int(i)] for i in middle_indices],
        ),
        "middle_balanced": (
            [strong_examples[int(i)] for i in middle_balanced_indices],
            [weak_labels[int(i)] for i in middle_balanced_indices],
        ),
        "confidence_middle_unbalanced": (
            [strong_examples[int(i)] for i in confidence_middle_indices],
            [weak_labels[int(i)] for i in confidence_middle_indices],
        ),
        "confidence_middle_balanced": (
            [strong_examples[int(i)] for i in confidence_middle_balanced_indices],
            [weak_labels[int(i)] for i in confidence_middle_balanced_indices],
        ),
        "confidence_high_unbalanced": (
            [strong_examples[int(i)] for i in confidence_high_indices],
            [weak_labels[int(i)] for i in confidence_high_indices],
        ),
        "confidence_high_balanced": (
            [strong_examples[int(i)] for i in confidence_high_balanced_indices],
            [weak_labels[int(i)] for i in confidence_high_balanced_indices],
        ),
        "knn_middle_unbalanced": (
            [strong_examples[int(i)] for i in knn_middle_indices],
            [weak_labels[int(i)] for i in knn_middle_indices],
        ),
        "knn_middle_balanced": (
            [strong_examples[int(i)] for i in knn_middle_balanced_indices],
            [weak_labels[int(i)] for i in knn_middle_balanced_indices],
        ),
        "knn_mixed_unbalanced": (
            [strong_examples[int(i)] for i in knn_mixed_indices],
            [weak_labels[int(i)] for i in knn_mixed_indices],
        ),
        "knn_mixed_balanced": (
            [strong_examples[int(i)] for i in knn_mixed_balanced_indices],
            [weak_labels[int(i)] for i in knn_mixed_balanced_indices],
        ),
    }
    random_run_names: list[str] = []
    random_unbalanced_run_names: list[str] = []
    if "random_unbalanced" in runs:
        runs = [name for name in runs if name != "random_unbalanced"]
        random_unbalanced_size = args.random_unbalanced_size or len(middle_indices)
        for idx in range(args.random_control_count):
            run_name = f"random_unbalanced_{idx}"
            selected = random_unbalanced_indices(
                len(strong_examples),
                random_unbalanced_size,
                args.seed + 6000 + idx,
            )
            run_subsets[run_name] = (
                [strong_examples[int(i)] for i in selected],
                [weak_labels[int(i)] for i in selected],
            )
            random_unbalanced_run_names.append(run_name)
            runs.append(run_name)

    if "random_balanced" in runs:
        runs = [name for name in runs if name != "random_balanced"]
        for idx in range(args.random_control_count):
            run_name = f"random_balanced_{idx}"
            selected = random_balanced_indices(weak_preds_strong, random_control_size, args.seed + 1000 + idx)
            run_subsets[run_name] = (
                [strong_examples[int(i)] for i in selected],
                [weak_labels[int(i)] for i in selected],
            )
            random_run_names.append(run_name)
            runs.append(run_name)

    prediction_columns: dict[str, list[dict]] = {}
    run_reports: dict[str, dict] = {}

    if "base" in runs:
        from run_dream_w2s_baselines import load_strong_model_and_tokenizer

        model, tokenizer = load_strong_model_and_tokenizer(args, trainable_lora=False)
        base_summary, base_rows = evaluate_yes_no(
            model,
            tokenizer,
            eval_examples,
            args.strong_batch_size,
            args.device,
            args.max_length,
            "eval base strong",
        )
        prediction_columns["base"] = base_rows
        run_reports["base"] = {"eval": base_summary}
        del model
        del tokenizer
        clear_memory()

    for run_name in runs:
        if run_name == "base":
            continue
        train_examples, train_labels = run_subsets[run_name]
        report, rows = run_lora_eval(args, run_name, train_examples, train_labels, eval_examples, output_dir)
        prediction_columns[run_name] = rows
        run_reports[run_name] = report

    write_predictions(output_dir / "eval_predictions.csv", eval_examples, prediction_columns, weak_eval_rows)
    write_strong_train_labels(output_dir / "strong_train_labels.csv", strong_examples, weak_probs_strong, residuals, knn_stats)
    write_knn_diagnostics(
        output_dir / "knn_diagnostics.csv",
        strong_examples,
        strong_train_labels,
        weak_probs_strong,
        residuals,
        knn_stats,
    )
    write_subset_csv(output_dir / "train_subsets.csv", {name: run_subsets[name] for name in run_subsets if name in run_reports})

    random_accs = [
        run_reports[name]["eval"]["accuracy"]
        for name in random_run_names
        if name in run_reports
    ]
    random_unbalanced_accs = [
        run_reports[name]["eval"]["accuracy"]
        for name in random_unbalanced_run_names
        if name in run_reports
    ]
    random_unbalanced_summary = None
    if random_unbalanced_accs:
        random_unbalanced_summary = {
            "count": len(random_unbalanced_accs),
            "accuracy_mean": float(np.mean(random_unbalanced_accs)),
            "accuracy_std": float(np.std(random_unbalanced_accs)),
            "accuracy_min": float(np.min(random_unbalanced_accs)),
            "accuracy_max": float(np.max(random_unbalanced_accs)),
        }
    random_summary = None
    if random_accs:
        random_summary = {
            "count": len(random_accs),
            "accuracy_mean": float(np.mean(random_accs)),
            "accuracy_std": float(np.std(random_accs)),
            "accuracy_min": float(np.min(random_accs)),
            "accuracy_max": float(np.max(random_accs)),
        }

    subset_summaries = {
        name: summarize_train_subset(*subset)
        for name, subset in run_subsets.items()
    }
    subset_summaries["middle_residual_diagnostics"] = subset_summary(
        middle_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["confidence_middle_diagnostics"] = subset_summary(
        confidence_middle_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["confidence_high_diagnostics"] = subset_summary(
        confidence_high_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["knn_middle_diagnostics"] = subset_summary(
        knn_middle_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["knn_mixed_diagnostics"] = subset_summary(
        knn_mixed_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )

    fmt = format_summary(args)
    summary = {
        "dataset": args.dataset,
        "source": "paper_style_lora",
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
            "task": fmt["task"],
            "candidate_text": fmt["candidate_text"],
            "lora_prompt": "candidate_text + answer_suffix",
            "label": fmt["label"],
            "takeaway": fmt["takeaway"],
        },
        "answer_suffix": args.answer_suffix,
        "activation": {
            "weak_label_probe": "LBFGS logistic probe on final-layer final-token weak activations",
            "mapping": "weak/strong final-layer final-token activations",
            "activation_max_length": args.activation_max_length,
        },
        "lora": {
            "max_length": args.max_length,
            "max_train_steps": args.max_train_steps,
            "batch_size": args.strong_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "max_grad_norm": args.max_grad_norm,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": [item.strip() for item in args.lora_target_modules.split(",") if item.strip()],
            "train_seed": args.train_seed,
        },
        "weak_label_diagnostics": {
            "accuracy": float(np.mean(weak_preds_strong == strong_train_labels)),
            "positive_rate": float(np.mean(weak_preds_strong)),
            "soft_prob_label1_mean": float(np.mean(weak_probs_strong)),
            "eval_accuracy": float(np.mean(weak_preds_test == test_labels)),
            "weak_train_accuracy": float(np.mean(weak_correct_weak_train)),
        },
        "map": {
            "best_name": best_map.name,
            "heldout_l2_mean": float(best_row["heldout_l2_mean"]),
            "heldout_l2_median": float(best_row["heldout_l2_median"]),
            "heldout_cosine_mean": float(best_row["heldout_cosine_mean"]),
            "spectral_norm": float(best_row["spectral_norm"]),
            "top20_energy": float(best_row["top20_energy"]),
        },
        "middle_filter": middle_filter,
        "confidence_filters": {
            "middle": confidence_middle_filter,
            "high": confidence_high_filter,
        },
        "knn_filter": {
            "k": args.knn_k,
            "reference_split": "weak_train",
            "query_split": "strong_train",
            "embedding": "strong model final-layer final-token activations",
            "distance": "cosine similarity",
            "middle": knn_middle_filter,
            "mixed": knn_mixed_filter,
            "knn_correct_rate_mean": float(np.mean(knn_stats["knn_correct_rate"])),
            "knn_correct_rate_median": float(np.median(knn_stats["knn_correct_rate"])),
            "knn_correct_rate_min": float(np.min(knn_stats["knn_correct_rate"])),
            "knn_correct_rate_max": float(np.max(knn_stats["knn_correct_rate"])),
        },
        "runs": run_reports,
        "random_unbalanced_controls_summary": random_unbalanced_summary,
        "random_balanced_controls_summary": random_summary,
        "subset_summaries": subset_summaries,
        "outputs": {
            "summary": str(output_dir / "summary.json"),
            "text_report": str(output_dir / "paper_style_lora_report.txt"),
            "eval_predictions": str(output_dir / "eval_predictions.csv"),
            "strong_train_labels": str(output_dir / "strong_train_labels.csv"),
            "knn_diagnostics": str(output_dir / "knn_diagnostics.csv"),
            "train_subsets": str(output_dir / "train_subsets.csv"),
            "map_dir": str(output_dir / "map"),
        },
        "elapsed_sec": time.time() - start,
    }

    if args.save_activations:
        torch.save(
            {
                "weak_activations": weak_acts,
                "splits": {
                    name: getattr(splits, name).to_pandas()
                    for name in ["weak_train", "strong_train", "val", "test"]
                },
            },
            output_dir / "activations.pt",
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_text_report(output_dir / "paper_style_lora_report.txt", summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
