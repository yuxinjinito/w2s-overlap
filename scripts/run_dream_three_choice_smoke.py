#!/usr/bin/env python3
"""Dream 3-choice / open-answer-format smoke test.

This is a fast diagnostic before running a full LoRA W2S experiment.  The
current main Dream pipeline follows the original paper-style binary
candidate-correctness task:

    dialogue + Q: question A: candidate -> is the candidate correct?

This script instead keeps each Dream question as one 3-choice example:

    dialogue + question + choices A/B/C -> answer letter

It measures whether the weak model's correctness signal and the strong-space
kNN mixed-neighborhood signal remain informative under this more natural
multiple-choice format.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_dream_paper_linear_probe import (
    SplitBundle,
    extract_final_token_activations,
    flatten_text,
    iter_dream_question_rows,
    load_dream_raw,
    resolve_device,
    resolve_dtype,
)
from run_dream_w2s_baselines import score_candidates


LETTERS = ["A", "B", "C"]


@dataclass(frozen=True)
class ChoiceSplitBundle:
    weak_train: Dataset
    strong_train: Dataset
    val: Dataset
    test: Dataset


class MulticlassProbe(torch.nn.Module):
    """Small LBFGS linear probe for a 3-class label."""

    def __init__(self, input_dim: int, n_classes: int, device: torch.device):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, n_classes, device=device)
        self.linear.bias.data.zero_()
        self.linear.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @torch.enable_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor, l2_penalty: float, max_iter: int) -> float:
        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
        )
        loss = torch.inf

        def closure():
            nonlocal loss
            optimizer.zero_grad()
            logits = self(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            reg_loss = loss + l2_penalty * self.linear.weight.square().sum()
            reg_loss.backward()
            return float(reg_loss)

        optimizer.step(closure)
        return float(loss)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(x), dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak-model", default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--strong-model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--output-dir", default="results/dream_three_choice_smoke_0604")
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--weak-logprob-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--l2-penalty", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument(
        "--weak-correct-source",
        choices=["logprob", "probe"],
        default="logprob",
        help=(
            "Which weak prediction defines weak-correct labels for the kNN "
            "reference set. logprob is closer to direct 3-choice answering; "
            "probe is closer to the previous activation-probe pipeline."
        ),
    )
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-shuffle-choices",
        action="store_true",
        help="Keep the dataset's answer order instead of deterministically shuffling A/B/C.",
    )
    parser.add_argument("--dry-run-splits", action="store_true")
    parser.add_argument("--save-activations", action="store_true")
    return parser.parse_args()


def batched_indices(n_items: int, batch_size: int):
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)


def label_distribution(labels: list[int] | np.ndarray, n_classes: int = 3) -> dict[str, float]:
    arr = np.array(labels, dtype=int)
    if len(arr) == 0:
        return {LETTERS[idx]: float("nan") for idx in range(n_classes)}
    return {LETTERS[idx]: float(np.mean(arr == idx)) for idx in range(n_classes)}


def count_distribution(values: np.ndarray, max_value: int) -> dict[str, int]:
    arr = np.array(values, dtype=int)
    return {str(idx): int(np.sum(arr == idx)) for idx in range(max_value + 1)}


def format_three_choice_example(ex: dict[str, Any], rng: random.Random, shuffle_choices: bool) -> dict[str, Any] | None:
    choices = [str(choice) for choice in ex["choice"]]
    answer = str(ex["answer"])
    if answer not in choices or len(choices) != 3:
        return None

    indexed_choices = list(enumerate(choices))
    if shuffle_choices:
        rng.shuffle(indexed_choices)

    shuffled_choices = [choice for _, choice in indexed_choices]
    label = shuffled_choices.index(answer)
    joined = "\n".join(ex["dialogue"]) if isinstance(ex["dialogue"], list) else flatten_text(ex["dialogue"])
    choice_lines = "\n".join(f"{LETTERS[idx]}. {choice}" for idx, choice in enumerate(shuffled_choices))
    txt = f"{joined}\n\nQuestion: {ex['question']}\nChoices:\n{choice_lines}\nAnswer:"
    stable = f"{ex['source_id']}|{ex['question']}|{'|'.join(shuffled_choices)}"
    return {
        "id": hashlib.sha1(stable.encode()).hexdigest()[:12],
        "source_id": ex["source_id"],
        "txt": txt,
        "labels": int(label),
        "label_letter": LETTERS[label],
        "answer": answer,
        "choice_a": shuffled_choices[0],
        "choice_b": shuffled_choices[1],
        "choice_c": shuffled_choices[2],
    }


def load_three_choice_split(split: str, n_docs: int, seed: int, shuffle_choices: bool) -> Dataset:
    raw_rows = list(iter_dream_question_rows(load_dream_raw(split)))
    rng = random.Random(seed)
    rng.shuffle(raw_rows)
    formatted_rows = [
        row
        for ex in raw_rows
        if (row := format_three_choice_example(ex, rng, shuffle_choices)) is not None
    ]
    ds = Dataset.from_list(formatted_rows)
    if len(ds) < n_docs:
        print(f"dream/{split} has < {n_docs} 3-choice examples, using all {len(ds)}")
        return ds
    return ds.select(range(n_docs))


def load_three_choice_splits(n_train: int, n_val: int, n_test: int, seed: int, shuffle_choices: bool) -> ChoiceSplitBundle:
    train_pool = load_three_choice_split("train", n_train + n_val, seed, shuffle_choices)
    test = load_three_choice_split("test", n_test, seed + 1, shuffle_choices)

    val_count = min(n_val, len(train_pool))
    val = train_pool.select(range(val_count))
    train = train_pool.select(range(val_count, len(train_pool)))
    train_halves = train.train_test_split(test_size=0.5, seed=seed)

    return ChoiceSplitBundle(
        weak_train=train_halves["train"],
        strong_train=train_halves["test"],
        val=val,
        test=test,
    )


def split_texts(splits: ChoiceSplitBundle) -> dict[str, list[str]]:
    return {
        "weak_train": list(splits.weak_train["txt"]),
        "strong_train": list(splits.strong_train["txt"]),
        "test": list(splits.test["txt"]),
    }


def split_labels(split: Dataset) -> np.ndarray:
    return np.array(split["labels"], dtype=int)


def fit_multiclass_probe(
    x: torch.Tensor,
    y: np.ndarray,
    l2_penalty: float,
    max_iter: int,
    device: torch.device,
    n_classes: int = 3,
) -> MulticlassProbe:
    probe = MulticlassProbe(x.shape[1], n_classes, device)
    probe.fit(x.to(device), torch.tensor(y, dtype=torch.long, device=device), l2_penalty, max_iter)
    return probe


@torch.no_grad()
def predict_probe_probs(probe: MulticlassProbe, x: torch.Tensor, device: torch.device) -> np.ndarray:
    return probe.predict_proba(x.to(device)).detach().cpu().numpy()


def multiclass_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    preds = probs.argmax(axis=1)
    max_probs = probs.max(axis=1)
    return {
        "accuracy": float(np.mean(preds == labels)),
        "confidence_mean": float(np.mean(max_probs)),
        "confidence_median": float(np.median(max_probs)),
        "label_distribution": label_distribution(labels),
        "pred_distribution": label_distribution(preds),
    }


def resolve_causal_dtype(dtype_arg: str):
    if dtype_arg == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_arg]


@torch.no_grad()
def score_choice_letters(
    model_name: str,
    texts: dict[str, list[str]],
    device: torch.device,
    dtype_arg: str,
    batch_size: int,
    max_length: int,
) -> dict[str, np.ndarray]:
    dtype = resolve_causal_dtype(dtype_arg)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    out: dict[str, np.ndarray] = {}
    for split_name, split_texts_list in texts.items():
        rows = []
        for start, end in tqdm(
            list(batched_indices(len(split_texts_list), batch_size)),
            desc=f"score weak choice logprobs: {split_name}",
        ):
            batch_texts = split_texts_list[start:end]
            flat_prompts = []
            flat_candidates = []
            for text in batch_texts:
                for letter in LETTERS:
                    flat_prompts.append(text)
                    flat_candidates.append(f" {letter}")
            scores = score_candidates(model, tokenizer, flat_prompts, flat_candidates, str(device), max_length)
            rows.append(scores.reshape(len(batch_texts), len(LETTERS)))
        logits = torch.cat(rows, dim=0) if rows else torch.empty((0, len(LETTERS)))
        out[split_name] = torch.softmax(logits, dim=1).cpu().numpy()

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return out


def compute_knn_stats(
    reference_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    reference_weak_correct: np.ndarray,
    k: int,
) -> dict[str, np.ndarray]:
    if k <= 0:
        raise ValueError("--knn-k must be positive.")
    k = min(k, len(reference_embeddings))
    ref = torch.nn.functional.normalize(reference_embeddings.to(torch.float32), p=2, dim=1)
    query = torch.nn.functional.normalize(query_embeddings.to(torch.float32), p=2, dim=1)
    sims = query @ ref.T
    top_values, top_indices = torch.topk(sims, k=k, dim=1)

    correct = torch.tensor(reference_weak_correct.astype(np.float32))
    neighbor_correct = correct[top_indices]
    correct_count = neighbor_correct.sum(dim=1)
    return {
        "neighbor_indices": top_indices.cpu().numpy(),
        "neighbor_cosine_mean": top_values.mean(dim=1).cpu().numpy(),
        "neighbor_cosine_min": top_values.min(dim=1).values.cpu().numpy(),
        "knn_correct_count": correct_count.cpu().numpy(),
        "knn_correct_rate": (correct_count / float(k)).cpu().numpy(),
    }


def write_split_preview(path: Path, splits: ChoiceSplitBundle) -> None:
    lines = []
    for name in ["weak_train", "strong_train", "val", "test"]:
        split = getattr(splits, name)
        labels = split_labels(split)
        lines.append(f"{name}: n={len(split)} label_distribution={label_distribution(labels)}")
        if len(split):
            lines.append("example:")
            lines.append(split[0]["txt"])
            lines.append(f"label: {split[0]['label_letter']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_predictions_csv(
    path: Path,
    splits: ChoiceSplitBundle,
    probe_probs: dict[str, np.ndarray],
    logprob_probs: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "id",
            "source_id",
            "label",
            "label_letter",
            "probe_pred",
            "probe_correct",
            "probe_confidence",
            "logprob_pred",
            "logprob_correct",
            "logprob_confidence",
            "text",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name in ["weak_train", "strong_train", "test"]:
            split = getattr(splits, split_name)
            labels = split_labels(split)
            probe_preds = probe_probs[split_name].argmax(axis=1)
            logprob_preds = logprob_probs[split_name].argmax(axis=1)
            for idx in range(len(split)):
                writer.writerow(
                    {
                        "split": split_name,
                        "id": split[idx]["id"],
                        "source_id": split[idx]["source_id"],
                        "label": int(labels[idx]),
                        "label_letter": LETTERS[int(labels[idx])],
                        "probe_pred": LETTERS[int(probe_preds[idx])],
                        "probe_correct": int(probe_preds[idx] == labels[idx]),
                        "probe_confidence": float(probe_probs[split_name][idx].max()),
                        "logprob_pred": LETTERS[int(logprob_preds[idx])],
                        "logprob_correct": int(logprob_preds[idx] == labels[idx]),
                        "logprob_confidence": float(logprob_probs[split_name][idx].max()),
                        "text": split[idx]["txt"],
                    }
                )


def write_knn_csv(
    path: Path,
    split: Dataset,
    source_probs: np.ndarray,
    labels: np.ndarray,
    knn_stats: dict[str, np.ndarray],
    source_name: str,
) -> None:
    preds = source_probs.argmax(axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "source_id",
                "label",
                "weak_pred",
                "weak_correct",
                "weak_confidence",
                "weak_correct_source",
                "knn_correct_count",
                "knn_correct_rate",
                "knn_neighbor_cosine_mean",
                "knn_neighbor_cosine_min",
                "text",
            ],
        )
        writer.writeheader()
        for idx in range(len(split)):
            writer.writerow(
                {
                    "id": split[idx]["id"],
                    "source_id": split[idx]["source_id"],
                    "label": LETTERS[int(labels[idx])],
                    "weak_pred": LETTERS[int(preds[idx])],
                    "weak_correct": int(preds[idx] == labels[idx]),
                    "weak_confidence": float(source_probs[idx].max()),
                    "weak_correct_source": source_name,
                    "knn_correct_count": int(knn_stats["knn_correct_count"][idx]),
                    "knn_correct_rate": float(knn_stats["knn_correct_rate"][idx]),
                    "knn_neighbor_cosine_mean": float(knn_stats["neighbor_cosine_mean"][idx]),
                    "knn_neighbor_cosine_min": float(knn_stats["neighbor_cosine_min"][idx]),
                    "text": split[idx]["txt"],
                }
            )


def write_text_report(path: Path, summary: dict[str, Any]) -> None:
    k = summary["knn"]["k"]
    lines = [
        "Dream 3-choice smoke test",
        "========================",
        "",
        "Setup:",
        f"- weak model: {summary['weak_model']}",
        f"- strong model: {summary['strong_model']}",
        f"- weak_correct_source for kNN: {summary['weak_correct_source']}",
        f"- prompt: dialogue + question + A/B/C choices + Answer:",
        f"- choice order: {summary['format']['choice_order']}",
        f"- sizes: weak_train={summary['sizes']['weak_train']}, strong_train={summary['sizes']['strong_train']}, test={summary['sizes']['test']}",
        "",
        "Weak 3-choice accuracy:",
        f"- logprob weak_train: {summary['metrics']['logprob']['weak_train']['accuracy']:.3f}",
        f"- logprob strong_train: {summary['metrics']['logprob']['strong_train']['accuracy']:.3f}",
        f"- logprob test: {summary['metrics']['logprob']['test']['accuracy']:.3f}",
        f"- probe weak_train: {summary['metrics']['probe']['weak_train']['accuracy']:.3f}",
        f"- probe strong_train: {summary['metrics']['probe']['strong_train']['accuracy']:.3f}",
        f"- probe test: {summary['metrics']['probe']['test']['accuracy']:.3f}",
        "",
        "kNN signal on strong_train:",
        f"- k: {k}",
        f"- correct-neighbor-rate mean: {summary['knn']['correct_rate_mean']:.3f}",
        f"- correct-neighbor-rate median: {summary['knn']['correct_rate_median']:.3f}",
        f"- correct-neighbor-rate min-max: {summary['knn']['correct_rate_min']:.3f} - {summary['knn']['correct_rate_max']:.3f}",
        f"- all-correct-neighbor fraction: {summary['knn']['all_correct_fraction']:.3f}",
        f"- 0/{k} neighbor fraction: {summary['knn']['all_wrong_fraction']:.3f}",
        f"- correct-count histogram: {summary['knn']['correct_count_histogram']}",
        "",
        "Interpretation guide:",
        "- If weak_train accuracy is near 1.0 and most points have k/k correct neighbors, kNN is saturated.",
        "- If correct-neighbor counts spread across the range, the 3-choice format may be useful for kNN filtering.",
        "- This is a smoke test only; it does not train the strong model with LoRA.",
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

    shuffle_choices = not args.no_shuffle_choices
    splits = load_three_choice_splits(args.n_train, args.n_val, args.n_test, args.seed, shuffle_choices)
    write_split_preview(output_dir / "split_preview.txt", splits)

    if args.dry_run_splits:
        summary = {
            "dataset": "dream",
            "task": "3-choice answer selection",
            "dry_run_splits": True,
            "sizes": {name: len(getattr(splits, name)) for name in ["weak_train", "strong_train", "val", "test"]},
            "label_distributions": {
                name: label_distribution(split_labels(getattr(splits, name)))
                for name in ["weak_train", "strong_train", "val", "test"]
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    texts = split_texts(splits)
    labels = {name: split_labels(getattr(splits, name)) for name in ["weak_train", "strong_train", "test"]}

    weak_acts = {
        name: extract_final_token_activations(
            args.weak_model,
            split_texts_list,
            device,
            args.torch_dtype,
            args.batch_size,
            args.max_length,
            f"extract weak activations: {name}",
        )
        for name, split_texts_list in texts.items()
    }
    probe = fit_multiclass_probe(
        weak_acts["weak_train"],
        labels["weak_train"],
        args.l2_penalty,
        args.max_iter,
        device,
        n_classes=len(LETTERS),
    )
    probe_probs = {name: predict_probe_probs(probe, weak_acts[name], device) for name in texts}

    logprob_probs = score_choice_letters(
        args.weak_model,
        texts,
        device,
        args.torch_dtype,
        args.weak_logprob_batch_size,
        args.max_length,
    )

    source_probs_by_name = {"logprob": logprob_probs, "probe": probe_probs}
    source_probs = source_probs_by_name[args.weak_correct_source]
    reference_preds = source_probs["weak_train"].argmax(axis=1)
    reference_weak_correct = (reference_preds == labels["weak_train"]).astype(int)

    strong_texts = {
        "weak_train": texts["weak_train"],
        "strong_train": texts["strong_train"],
    }
    strong_acts = {
        name: extract_final_token_activations(
            args.strong_model,
            split_texts_list,
            device,
            args.torch_dtype,
            args.batch_size,
            args.max_length,
            f"extract strong activations: {name}",
        )
        for name, split_texts_list in strong_texts.items()
    }
    knn_stats = compute_knn_stats(
        strong_acts["weak_train"],
        strong_acts["strong_train"],
        reference_weak_correct,
        args.knn_k,
    )
    k = min(args.knn_k, len(strong_acts["weak_train"]))
    correct_counts = knn_stats["knn_correct_count"].astype(int)
    correct_rates = knn_stats["knn_correct_rate"]

    metrics = {
        "probe": {
            name: multiclass_metrics(probe_probs[name], labels[name])
            for name in ["weak_train", "strong_train", "test"]
        },
        "logprob": {
            name: multiclass_metrics(logprob_probs[name], labels[name])
            for name in ["weak_train", "strong_train", "test"]
        },
    }
    summary: dict[str, Any] = {
        "dataset": "dream",
        "task": "3-choice answer selection",
        "source": "dream_three_choice_smoke",
        "weak_model": args.weak_model,
        "strong_model": args.strong_model,
        "seed": args.seed,
        "weak_correct_source": args.weak_correct_source,
        "sizes": {name: len(getattr(splits, name)) for name in ["weak_train", "strong_train", "val", "test"]},
        "requested_sizes": {"n_train": args.n_train, "n_val": args.n_val, "n_test": args.n_test},
        "format": {
            "prompt": "dialogue + Question + Choices A/B/C + Answer:",
            "label": "0/1/2 for the correct answer letter",
            "choice_order": "deterministically shuffled" if shuffle_choices else "original dataset order",
            "note": (
                "This moves away from binary candidate-answer correctness. "
                "It is a smoke test, not the original paper setup."
            ),
        },
        "activation": {
            "model_class": "AutoModelForSequenceClassification",
            "layer": "final hidden layer",
            "token": "final non-padding token",
            "max_length": args.max_length,
        },
        "metrics": metrics,
        "knn": {
            "k": int(k),
            "reference_split": "weak_train",
            "query_split": "strong_train",
            "embedding_space": "strong model final-token activations",
            "distance": "cosine similarity",
            "correct_source": args.weak_correct_source,
            "correct_rate_mean": float(np.mean(correct_rates)),
            "correct_rate_median": float(np.median(correct_rates)),
            "correct_rate_min": float(np.min(correct_rates)),
            "correct_rate_max": float(np.max(correct_rates)),
            "all_correct_fraction": float(np.mean(correct_counts == k)),
            "all_wrong_fraction": float(np.mean(correct_counts == 0)),
            "correct_count_histogram": count_distribution(correct_counts, k),
            "neighbor_cosine_mean": float(np.mean(knn_stats["neighbor_cosine_mean"])),
            "neighbor_cosine_min_mean": float(np.mean(knn_stats["neighbor_cosine_min"])),
        },
        "paper_alignment_audit": {
            "matches_original_binary_candidate_correctness": False,
            "reason_for_change": "John/Fred asked about moving beyond binary candidate-correctness toward true 3-choice/open-ended formats.",
            "uses_lora": False,
            "uses_generation_scoring": True,
            "uses_activation_probe": True,
            "smoke_test_only": True,
        },
        "elapsed_sec": time.time() - start,
    }

    write_predictions_csv(output_dir / "weak_predictions.csv", splits, probe_probs, logprob_probs)
    write_knn_csv(
        output_dir / "strong_train_knn_diagnostics.csv",
        splits.strong_train,
        source_probs["strong_train"],
        labels["strong_train"],
        knn_stats,
        args.weak_correct_source,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_text_report(output_dir / "three_choice_smoke_report.txt", summary)

    if args.save_activations:
        torch.save(
            {
                "weak_activations": weak_acts,
                "strong_activations": strong_acts,
                "splits": {
                    name: getattr(splits, name).to_pandas()
                    for name in ["weak_train", "strong_train", "val", "test"]
                },
                "probe_probs": probe_probs,
                "logprob_probs": logprob_probs,
                "knn_stats": knn_stats,
            },
            output_dir / "three_choice_smoke_artifact.pt",
        )

    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
