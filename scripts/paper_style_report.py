#!/usr/bin/env python3
"""Metrics and output files for the paper-style pipeline.

Everything a run writes down: the per-arm metrics (accuracy, prior-matched
accuracy, AUROC, multiple-choice accuracy), the kept-set summaries, the CSV
dumps, and the human-readable report next to summary.json. Split out of
run_dream_paper_style_lora.py; like the dataset layer it is a leaf and calls
nothing else in the pipeline.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from paper_style_datasets import LoraExample


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
    if args.dataset == "anli":
        return {
            "task": "paper-style binary candidate-relation correctness (ANLI)",
            "candidate_text": "'Premise: ...\\nHypothesis: ...\\nQ: What is the relationship ...? A: {candidate}'",
            "label": "1 if the candidate relation (entailment/neutral/contradiction) is the gold ANLI label, else 0",
            "takeaway": f"This run uses ANLI round {args.anli_round} as binary candidate-relation correctness (adversarial NLI -> weak model near chance).",
        }
    if args.dataset == "hellaswag":
        return {
            "task": "paper-style binary candidate-ending correctness (HellaSwag)",
            "candidate_text": "'{context} {ending}'",
            "label": "1 if the ending is the gold HellaSwag continuation, else 0",
            "takeaway": "This run uses HellaSwag (4-way commonsense ending) as binary candidate-ending correctness -> weak model near chance.",
        }
    if args.dataset in ("reclor", "logiqa2"):
        nice = "ReClor" if args.dataset == "reclor" else "LogiQA 2.0"
        return {
            "task": f"paper-style binary candidate-answer correctness ({nice})",
            "candidate_text": "'{context}\\nQ: {question} A: {candidate}'",
            "label": f"1 if the candidate option is the gold {nice} answer, else 0",
            "takeaway": f"This run uses {nice} (logical-reasoning MC) as binary candidate-answer correctness -> base near chance in the eval format.",
        }
    if args.dataset == "wanli":
        return {
            "task": "paper-style binary candidate-relation correctness (WANLI)",
            "candidate_text": "'Premise: ...\\nHypothesis: ...\\nQ: What is the relationship ...? A: {candidate}'",
            "label": "1 if the candidate relation (entailment/neutral/contradiction) is the gold WANLI label, else 0",
            "takeaway": "This run uses WANLI (worker-and-AI adversarial NLI) as binary candidate-relation correctness -> ANLI-like noisy-weak regime.",
        }
    if args.dataset == "control":
        return {
            "task": "paper-style binary candidate-relation correctness (ConTRoL)",
            "candidate_text": "'Premise: ...\\nHypothesis: ...\\nQ: What is the relationship ...? A: {candidate}'",
            "label": "1 if the candidate relation (entailment/neutral/contradiction) is the gold ConTRoL label, else 0",
            "takeaway": "This run uses ConTRoL (contextual-reasoning adversarial NLI) as binary candidate-relation correctness -> low-base testbed search.",
        }
    if args.dataset == "fevernli":
        return {
            "task": "paper-style binary candidate-relation correctness (FEVER-NLI)",
            "candidate_text": "'Evidence: ...\\nClaim: ...\\nQ: Based on the evidence, the claim is? A: {candidate}'",
            "label": "1 if the candidate verdict (supported/not enough info/refuted) is the gold FEVER label, else 0",
            "takeaway": "This run uses FEVER recast as 3-way fact-verification NLI (candidate-verdict correctness) -> low-base testbed search.",
        }
    if args.dataset == "conjnli":
        return {
            "task": "paper-style binary candidate-relation correctness (ConjNLI)",
            "candidate_text": "'Premise: ...\\nHypothesis: ...\\nQ: What is the relationship ...? A: {candidate}'",
            "label": "1 if the candidate relation is the ConjNLI label, else 0",
            "takeaway": "This run uses ConjNLI (adversarial conjunction NLI; silver 15k train pool, human dev as test) -> low-base testbed search.",
        }
    if args.dataset == "imppres":
        return {
            "task": "paper-style binary candidate-relation correctness (ImpPres)",
            "candidate_text": "'Premise: ...\\nHypothesis: ...\\nQ: What is the relationship ...? A: {candidate}'",
            "label": "1 if the candidate relation is the ImpPres gold label, else 0",
            "takeaway": "This run uses ImpPres presupposition NLI (templated, structurally adversarial) -> low-base testbed search; watch template memorization.",
        }
    if args.dataset == "aquarat":
        return {
            "task": "paper-style binary candidate-answer correctness (AQuA-RAT)",
            "candidate_text": "'Q: {math word problem} A: {candidate option}'",
            "label": "1 if the candidate is the correct option, else 0",
            "takeaway": "This run uses AQuA-RAT (5-option math word problems; crowd-amplified pool) -> trio-PASS(CAUTION) validation; preregistered expectation is the ReClor outcome.",
        }
    if args.dataset == "quail":
        return {
            "task": "paper-style binary candidate-answer correctness (QuAIL)",
            "candidate_text": "'{context}\\nQ: {question} A: {candidate}'",
            "label": "1 if the candidate is the correct option, else 0",
            "takeaway": "This run uses QuAIL (4-option multi-domain RC) under the relaxed base<0.70 bar (discrim .698) -> needs GT>=.8 headroom.",
        }
    if args.dataset == "riddlesense":
        return {
            "task": "paper-style binary candidate-answer correctness (RiddleSense)",
            "candidate_text": "'Q: {riddle} A: {candidate}'",
            "label": "1 if the candidate is the correct option, else 0",
            "takeaway": "This run uses RiddleSense (5-option human riddles) under the relaxed base<0.70 bar (discrim .657).",
        }
    if args.dataset == "scinli":
        return {
            "task": "paper-style binary candidate-relation correctness (SciNLI)",
            "candidate_text": "'Sentence 1: ...\\nSentence 2: ...\\nQ: What is the relationship ...? A: {candidate}'",
            "label": "1 if the candidate relation (entailment/contrasting/reasoning/neutral) is the SciNLI label, else 0",
            "takeaway": "This run uses SciNLI (4-way scientific-paper NLI, 107k, human-written) -> best trio profile of round 4 (probe .588, base .425).",
        }
    raise ValueError(f"Unsupported dataset: {args.dataset}")


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


def auroc_from_rows(rows: list[dict]) -> float:
    """Threshold-free AUROC of prob_label1 vs the true label (rank-based, tie-averaged).

    More stable than thresholded accuracy when the model sits near the 0.5 boundary:
    it measures whether truly-positive examples are scored above truly-negative ones,
    regardless of where the decision threshold falls.
    """
    if not rows:
        return float("nan")
    scores = np.array([r["prob_label1"] for r in rows], dtype=float)
    labels = np.array([int(r["label"]) for r in rows], dtype=int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=float)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks_sorted[i : j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = ranks_sorted
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def prior_matched_accuracy(rows: list[dict]) -> float:
    """Accuracy at the balanced operating point: predict the highest-scored points as
    positive, matching the number of true positives (the test set is class-balanced by
    construction, so this uses only the known prior, not per-example labels). Removes the
    0.5-threshold brittleness when the model ranks well but its probabilities are bunched
    near 0.5 / miscalibrated -- equivalent to a val-tuned threshold for a balanced test.
    """
    if not rows:
        return float("nan")
    probs = np.array([r["prob_label1"] for r in rows], dtype=float)
    labels = np.array([int(r["label"]) for r in rows], dtype=int)
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return float("nan")
    order = np.argsort(-probs, kind="mergesort")
    preds = np.zeros(len(labels), dtype=int)
    preds[order[:n_pos]] = 1
    return float((preds == labels).mean())


def accuracy_3class(examples: list[LoraExample], rows: list[dict]) -> float:
    """Per-question multi-class accuracy. Group candidate rows by source_id (rows must be
    aligned with examples in order), pick the candidate with the highest P(correct)
    (prob_label1), and count the question correct iff that candidate is the true answer
    (label == 1). This converts the binary 'is this candidate correct?' model into a true
    multiple-choice prediction (argmax over a question's candidates)."""
    from collections import defaultdict

    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for ex, row in zip(examples, rows):
        groups[ex.source_id].append((float(row["prob_label1"]), int(ex.label)))
    if not groups:
        return float("nan")
    correct = 0
    for cands in groups.values():
        best = max(range(len(cands)), key=lambda i: cands[i][0])
        correct += int(cands[best][1] == 1)
    return correct / len(groups)


def write_eval3_rows(path: Path, examples: list[LoraExample], rows: list[dict]) -> None:
    """Persist per-candidate multichoice predictions so post-hoc analysis (tie rates,
    per-question breakdowns, per-seed multichoice PGR) does not require a re-run."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "source_id", "label", "prob_label1"])
        for ex, row in zip(examples, rows):
            writer.writerow([ex.id, ex.source_id, ex.label, f"{float(row['prob_label1']):.6f}"])


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


def write_strong_train_labels(
    path: Path,
    examples: list[LoraExample],
    weak_probs: np.ndarray,
    residuals: np.ndarray,
    knn_stats=None,
    rp_scores=None,
    base_probs=None,
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
                "rp_score",
                "base_prob_correct",
                "base_confidence",
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
                    "rp_score": "" if rp_scores is None else float(rp_scores[idx]),
                    "base_prob_correct": "" if base_probs is None else float(base_probs[idx]),
                    "base_confidence": ""
                    if base_probs is None
                    else float(2.0 * abs(float(base_probs[idx]) - 0.5)),
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


def build_summary(
    *,
    args,
    best_map,
    best_row,
    committee_agree_filter,
    committee_disagree_filter,
    committee_disagreement_strong,
    confidence_high_filter,
    confidence_middle_filter,
    fmt,
    knn_middle_filter,
    knn_mixed_filter,
    knn_stats,
    middle_filter,
    output_dir,
    random_summary,
    random_unbalanced_summary,
    run_reports,
    splits,
    start,
    strong_train_labels,
    subset_summaries,
    test_labels,
    weak_correct_weak_train,
    weak_eval3_acc,
    weak_preds_strong,
    weak_preds_test,
    weak_probs_strong,
) -> dict:
    """Assemble the summary.json payload.

    One flat record per run: the configuration that produced it, the weak-label
    diagnostics, the per-arm evaluation, and the kept-set diagnostics. Anything
    reported anywhere else has to be traceable back to a field here.
    """
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
            "accuracy_3class": weak_eval3_acc,
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
            "embedding": "strong model activations",
            "activation_layer": args.representation_layer,
            "activation_pooling": args.representation_pooling,
            "distance": "cosine similarity",
            "middle": knn_middle_filter,
            "mixed": knn_mixed_filter,
            "knn_correct_rate_mean": float(np.mean(knn_stats["knn_correct_rate"])),
            "knn_correct_rate_median": float(np.median(knn_stats["knn_correct_rate"])),
            "knn_correct_rate_min": float(np.min(knn_stats["knn_correct_rate"])),
            "knn_correct_rate_max": float(np.max(knn_stats["knn_correct_rate"])),
        },
        "committee_filter": {
            "members": args.committee_members,
            "score": "std of bootstrap committee weak-probe probabilities on strong_train",
            "reference_split": "weak_train (bootstrap resamples)",
            "query_split": "strong_train",
            "agree": committee_agree_filter,
            "disagree": committee_disagree_filter,
            "disagreement_mean": float(np.mean(committee_disagreement_strong)),
            "disagreement_median": float(np.median(committee_disagreement_strong)),
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
    return summary
