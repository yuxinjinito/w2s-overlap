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
    load_dream_3class_eval,
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
from representation_projection import representation_projection_scores
from excess_loss import excess_loss_kway_scores


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
    parser.add_argument("--dataset", choices=["dream", "sciq", "paws", "anli", "hellaswag"], default="dream")
    parser.add_argument(
        "--anli-round",
        choices=["r1", "r2", "r3"],
        default="r2",
        help="For ANLI only: which adversarial round to use (default r2, matching the survey).",
    )
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="If >0, train each run for this many FULL passes over its actual subset "
        "(no-filter -> all data, f50 -> the real 50%), setting max_train_steps per run = "
        "epochs * ceil(len(subset)/(batch*grad_accum)). Overrides the fixed compute-matched "
        "--max-train-steps cap.",
    )
    parser.add_argument("--rp-reg", type=float, default=0.1,
                        help="Kernel ridge reg for the representation-projection score (rp_high/rp_low).")
    parser.add_argument("--rp-components", type=int, default=0,
                        help="PCA components for the representation-projection score (0 = use all).")
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
            "knn_mixed_balanced, committee_agree_unbalanced, committee_agree_balanced, "
            "committee_disagree_unbalanced, committee_disagree_balanced, "
            "random_unbalanced, random_balanced."
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
    parser.add_argument(
        "--committee-members",
        type=int,
        default=8,
        help=(
            "Number of bootstrap weak probes in the query-by-committee disagreement "
            "selector (committee_agree / committee_disagree)."
        ),
    )
    parser.add_argument("--committee-keep-frac", type=float, default=0.5)
    parser.add_argument(
        "--committee-keep-fracs",
        default="",
        help=(
            "Comma-separated keep fractions for a committee selection sweep, e.g. "
            "'0.1,0.2,0.3,0.45,0.6,0.8'. For each fraction f (pct=round(100f)) it adds runs "
            "committee_agree_balanced_f{pct}, committee_agree_unbalanced_f{pct}, "
            "committee_disagree_balanced_f{pct}, and a matched random_balanced_f{pct}. "
            "Empty = no sweep."
        ),
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help=(
            "Compute weak_train reference accuracy and kNN saturation (fraction "
            "of strong_train points whose k nearest neighbors are all weak-correct) "
            "under in-sample vs cross-fitted reference labels, then exit before any "
            "LoRA training. Cheap check for whether cross-fitting de-saturates kNN."
        ),
    )
    parser.add_argument("--random-control-count", type=int, default=3)
    parser.add_argument("--random-control-size", type=int, default=None)
    parser.add_argument("--random-unbalanced-size", type=int, default=None)
    parser.add_argument("--ridge-values", default="100.0")
    parser.add_argument("--pca-dims", default="")
    parser.add_argument("--best-by", choices=["heldout_mean", "heldout_median"], default="heldout_median")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument(
        "--eval-3class",
        action="store_true",
        help="(Dream) also report per-question multi-class accuracy: score every candidate "
        "answer of each test question and argmax P(correct).",
    )
    parser.add_argument(
        "--n-eval-questions",
        type=int,
        default=2041,
        help="Number of Dream test questions for --eval-3class (each expands to its candidate answers). "
        "Default 2041 = the full Dream test split.",
    )
    return parser.parse_args()


def requested_runs(args: argparse.Namespace) -> list[str]:
    runs = [item.strip() for item in args.runs.split(",") if item.strip()]
    allowed = {
        "base",
        "ground_truth",
        "weak_label",
        "curriculum_easy",
        "curriculum_hard",
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
        "knn_high_unbalanced",
        "knn_high_balanced",
        "committee_agree_unbalanced",
        "committee_agree_balanced",
        "committee_disagree_unbalanced",
        "committee_disagree_balanced",
        "random_unbalanced",
        "random_balanced",
    }
    def _is_frac_run(name: str) -> bool:
        for pref in (
            "committee_agree_balanced_f",
            "committee_agree_unbalanced_f",
            "committee_disagree_balanced_f",
            "knn_high_balanced_f",
            "knn_low_balanced_f",
            "knn_mixed_balanced_f",
            "confidence_high_balanced_f",
            "confidence_low_balanced_f",
            "rp_high_balanced_f",
            "rp_low_balanced_f",
            "el_high_balanced_f",
            "el_low_balanced_f",
            "random_balanced_f",
        ):
            if name.startswith(pref) and name[len(pref):].isdigit():
                return True
        return False

    unknown = sorted(r for r in set(runs) - allowed if not _is_frac_run(r))
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
    options = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
    return {
        "id": f"sciq-{row_id}",
        "source_id": f"sciq-{row_id}",
        "txt": txt,
        "labels": hard_label,
        "gt_labels": hard_label,
        "mc_options": [f"{support_block}Q: {ex['question']} A: {o}" for o in options],
        "mc_correct": 0,
    }


def load_and_process_sciq_split(split: str, n_docs: int, seed: int, use_support: bool) -> Dataset:
    raw = load_dataset("allenai/sciq", split=split).shuffle(seed=seed)
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


def load_sciq_multichoice_eval(n_questions: int, seed: int, use_support: bool) -> list[dict]:
    """Per-question multiple-choice eval for SciQ (4 options).

    For each test question, emit all 4 candidates (correct_answer + 3 distractors),
    sharing source_id, label=1 on the correct one, same prompt format as training
    (no support by default). argmax of P(correct) over a question's 4 candidates gives
    a true 4-way multiple-choice prediction (scored by accuracy_3class's argmax)."""
    raw = load_dataset("allenai/sciq", split="test").shuffle(seed=seed)
    rng = random.Random(seed + 7)
    out: list[dict] = []
    n = 0
    for ex in raw:
        if n >= n_questions:
            break
        if use_support:
            support = (ex.get("support") or "").strip()
            support_block = f"Support: {support}\n\n" if support else ""
        else:
            support_block = ""
        options = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        # Shuffle the emission order: argmax tie-breaks pick the FIRST max, so a fixed
        # correct-first order would systematically resolve fp16 probability ties in favor
        # of the correct answer and inflate accuracy_3class.
        order = rng.sample(range(4), 4)
        for slot, oi in enumerate(order):
            txt = f"{support_block}Q: {ex['question']} A: {options[oi]}"
            out.append(
                {
                    "id": f"sciq-mc-{n}-{slot}",
                    "source_id": f"sciq-mc-{n}",
                    "txt": txt,
                    "labels": int(oi == 0),  # options[0] is the correct_answer
                }
            )
        n += 1
    return out


def format_hellaswag_paper_style(ex: dict, row_id: int, rng: random.Random) -> dict:
    """Format HellaSwag as binary candidate-ending correctness: '{context} {ending}'."""
    endings = list(ex["endings"])
    correct = int(ex["label"])
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = endings[correct]
    else:
        wrong = [e for i, e in enumerate(endings) if i != correct]
        ans = rng.choice(wrong)
    ctx = (ex.get("ctx") or "").strip()
    txt = f"{ctx} {ans}".strip()
    return {
        "id": f"hellaswag-{row_id}",
        "source_id": f"hellaswag-{row_id}",
        "txt": txt,
        "labels": hard_label,
        "gt_labels": hard_label,
        "mc_options": [f"{ctx} {e}".strip() for e in endings],
        "mc_correct": correct,
    }


def load_and_process_hellaswag_split(split: str, n_docs: int, seed: int) -> Dataset:
    raw = load_dataset("Rowan/hellaswag", split=split).shuffle(seed=seed)
    raw = raw.filter(lambda ex: ex["label"] != "")  # test labels are hidden -> drop them
    if len(raw) < n_docs:
        print(f"hellaswag/{split} has < {n_docs} labeled docs, using all {len(raw)}")
    raw = raw.select(range(min(n_docs, len(raw))))
    rng = random.Random(seed)
    rows = [format_hellaswag_paper_style(ex, row_id=i, rng=rng) for i, ex in enumerate(raw)]
    ds = Dataset.from_list(rows)
    ds = ds.filter(lambda ex: ex["txt"] != "")
    ds = balance_binary_dataset(ds, seed)
    return ds


def load_paper_hellaswag_splits(n_train: int, n_val: int, n_test: int, seed: int) -> SplitBundle:
    # HellaSwag test labels are hidden -> train pool from "train", eval from "validation".
    train_pool = load_and_process_hellaswag_split("train", n_train + n_val, seed)
    test = load_and_process_hellaswag_split("validation", n_test, seed + 2)
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


def load_hellaswag_multichoice_eval(n_questions: int, seed: int) -> list[dict]:
    """Per-question 4-way multiple-choice eval for HellaSwag (pick the best of 4 endings)."""
    raw = load_dataset("Rowan/hellaswag", split="validation").shuffle(seed=seed)
    rng = random.Random(seed + 7)
    out: list[dict] = []
    n = 0
    for ex in raw:
        if n >= n_questions:
            break
        if ex["label"] == "":
            continue
        ctx = (ex.get("ctx") or "").strip()
        endings = list(ex["endings"])
        correct = int(ex["label"])
        order = rng.sample(range(len(endings)), len(endings))
        for slot, oi in enumerate(order):
            out.append(
                {
                    "id": f"hellaswag-mc-{n}-{slot}",
                    "source_id": f"hellaswag-mc-{n}",
                    "txt": f"{ctx} {endings[oi]}".strip(),
                    "labels": int(oi == correct),
                }
            )
        n += 1
    return out


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


ANLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def load_anli_multichoice_eval(n_questions: int, seed: int, anli_round: str) -> list[dict]:
    """Per-question multiple-choice eval for ANLI (3 relations).

    For each test example, emit all 3 candidate relations (entailment / neutral /
    contradiction), sharing source_id, label=1 on the gold relation, same prompt
    format as training. Option order shuffled (seeded) so argmax tie-breaks are not
    biased toward a fixed position. argmax of P(correct) over the 3 -> 3-way NLI."""
    raw = load_dataset("facebook/anli", split=f"test_{anli_round}").shuffle(seed=seed)
    rng = random.Random(seed + 7)
    out: list[dict] = []
    n = 0
    for ex in raw:
        if n >= n_questions:
            break
        gold = int(ex["label"])
        if gold not in (0, 1, 2):
            continue
        order = rng.sample(range(3), 3)
        for slot, ri in enumerate(order):
            txt = (
                f"Premise: {ex['premise']}\n"
                f"Hypothesis: {ex['hypothesis']}\n"
                f"Q: What is the relationship from the premise to the hypothesis? A: {ANLI_LABELS[ri]}"
            )
            out.append(
                {
                    "id": f"anli-mc-{n}-{slot}",
                    "source_id": f"anli-mc-{n}",
                    "txt": txt,
                    "labels": int(ri == gold),
                }
            )
        n += 1
    return out


def format_anli_paper_style(ex: dict, row_id: int, rng: random.Random) -> dict:
    """Format ANLI (3-class NLI) as binary candidate-relation correctness.

    A candidate relation is sampled (50% the gold relation, 50% a wrong one) and
    the model judges whether it is correct, mirroring the SciQ candidate-answer
    reduction. ANLI is adversarial, so the weak model is near chance here -- the
    noisy-weak-label regime where overlap selection should matter if it is real.
    """
    gold = int(ex["label"])
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        candidate = ANLI_LABELS[gold]
    else:
        wrong = [name for key, name in ANLI_LABELS.items() if key != gold]
        candidate = rng.choice(wrong)
    txt = (
        f"Premise: {ex['premise']}\n"
        f"Hypothesis: {ex['hypothesis']}\n"
        f"Q: What is the relationship from the premise to the hypothesis? A: {candidate}"
    )
    return {
        "id": f"anli-{row_id}",
        "source_id": f"anli-{row_id}",
        "txt": txt,
        "labels": hard_label,
        "gt_labels": hard_label,
        "mc_options": [
            f"Premise: {ex['premise']}\n"
            f"Hypothesis: {ex['hypothesis']}\n"
            f"Q: What is the relationship from the premise to the hypothesis? A: {ANLI_LABELS[k]}"
            for k in (0, 1, 2)
        ],
        "mc_correct": gold,
    }


def load_and_process_anli_split(split: str, n_docs: int, seed: int) -> Dataset:
    raw = load_dataset("facebook/anli", split=split).shuffle(seed=seed)
    raw = raw.filter(lambda ex: int(ex["label"]) in (0, 1, 2))
    rng = random.Random(seed)
    formatted_rows = [
        format_anli_paper_style(ex, row_id=i, rng=rng)
        for i, ex in enumerate(raw)
    ]
    ds = Dataset.from_list(formatted_rows)
    ds = ds.filter(lambda ex: ex["txt"] != "")
    ds = balance_binary_dataset(ds, seed)
    if len(ds) < n_docs:
        print(f"anli/{split} has < {n_docs} docs after balancing, using all {len(ds)}")
    return ds.select(range(min(n_docs, len(ds))))


def load_paper_anli_splits(
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    anli_round: str,
) -> SplitBundle:
    train_pool = load_and_process_anli_split(f"train_{anli_round}", n_train + n_val, seed)
    test = load_and_process_anli_split(f"test_{anli_round}", n_test, seed + 2)

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
    if args.dataset == "anli":
        return load_paper_anli_splits(
            args.n_train, args.n_val, args.n_test, args.seed, args.anli_round
        )
    if args.dataset == "hellaswag":
        return load_paper_hellaswag_splits(args.n_train, args.n_val, args.n_test, args.seed)
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
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def split_texts(splits: SplitBundle) -> dict[str, list[str]]:
    return {
        "weak_train": list(splits.weak_train["txt"]),
        "strong_train": list(splits.strong_train["txt"]),
        "test": list(splits.test["txt"]),
    }


def split_labels(split) -> np.ndarray:
    return np.array(split["labels"], dtype=int)


def run_knn_saturation_diagnostics(
    args: argparse.Namespace,
    splits: SplitBundle,
    texts: dict[str, list[str]],
    device: torch.device,
    output_dir: Path,
) -> None:
    """Run the lightweight kNN saturation check and exit before mapping/LoRA.

    This intentionally does less than the full experiment: weak activations are
    needed only on weak_train, and strong activations are needed only on
    weak_train + strong_train. It does not extract test activations, fit
    weak-to-strong maps, or train LoRA adapters.
    """

    weak_train_labels = split_labels(splits.weak_train)
    weak_train_acts = extract_final_token_activations(
        args.weak_model,
        texts["weak_train"],
        device,
        args.torch_dtype,
        args.activation_batch_size,
        args.activation_max_length,
        "extract weak_train activations",
    )
    weak_probe = fit_probe(
        weak_train_acts,
        torch.tensor(weak_train_labels, dtype=torch.float32),
        args.l2_penalty,
        args.max_iter,
        device,
    )
    weak_probs_weak_train = predict_probe(weak_probe, weak_train_acts, device)
    insample_ref_correct = ((weak_probs_weak_train >= 0.5).astype(int) == weak_train_labels).astype(int)

    cross_fit_probs = cross_fitted_weak_probs(
        weak_train_acts,
        weak_train_labels,
        args.cross_fit_folds,
        args.l2_penalty,
        args.max_iter,
        device,
        args.seed,
    )
    crossfit_ref_correct = ((cross_fit_probs >= 0.5).astype(int) == weak_train_labels).astype(int)

    strong_diag_texts = texts["weak_train"] + texts["strong_train"]
    strong_diag_acts = extract_final_token_activations(
        args.strong_model,
        strong_diag_texts,
        device,
        args.torch_dtype,
        args.activation_batch_size,
        args.activation_max_length,
        "extract strong weak_train+strong_train activations",
    )
    n_weak_train = len(texts["weak_train"])
    strong_weak_train_acts = strong_diag_acts[:n_weak_train]
    strong_strong_train_acts = strong_diag_acts[n_weak_train:]

    def _knn_saturation(reference_correct: np.ndarray) -> dict[str, float]:
        stats = compute_weak_train_knn_stats(
            strong_weak_train_acts,
            strong_strong_train_acts,
            reference_correct,
            args.knn_k,
        )
        rates = stats["knn_correct_rate"]
        return {
            "reference_accuracy": float(np.mean(reference_correct)),
            "all_neighbors_weak_correct_fraction": float(np.mean(rates >= 1.0 - 1e-9)),
            "correct_neighbor_rate_mean": float(np.mean(rates)),
            "correct_neighbor_rate_median": float(np.median(rates)),
            "correct_neighbor_rate_min": float(np.min(rates)),
            "correct_neighbor_rate_max": float(np.max(rates)),
        }

    diagnostics = {
        "dataset": args.dataset,
        "knn_k": args.knn_k,
        "cross_fit_folds": args.cross_fit_folds,
        "sizes": {
            "weak_train": len(splits.weak_train),
            "strong_train": len(splits.strong_train),
            "test": len(splits.test),
        },
        "in_sample": _knn_saturation(insample_ref_correct),
        "cross_fitted": _knn_saturation(crossfit_ref_correct),
        "note": "diagnostics-only skips test activations, representation mapping, and LoRA training",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "knn_saturation_diagnostic.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(
        f"[diagnostics-only] {args.dataset}: "
        f"reference acc in-sample={diagnostics['in_sample']['reference_accuracy']:.3f} "
        f"-> cross-fit={diagnostics['cross_fitted']['reference_accuracy']:.3f}; "
        f"kNN all-{args.knn_k}-correct fraction "
        f"in-sample={diagnostics['in_sample']['all_neighbors_weak_correct_fraction']:.3f} "
        f"-> cross-fit={diagnostics['cross_fitted']['all_neighbors_weak_correct_fraction']:.3f}"
    )
    print(f"  wrote {output_dir / 'knn_saturation_diagnostic.json'}")

    del weak_train_acts, strong_diag_acts, strong_weak_train_acts, strong_strong_train_acts
    clear_memory()


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
    if mode not in {"middle", "high", "low"}:
        raise ValueError(f"Unknown score band mode: {mode}")

    order = np.argsort(scores)
    n_keep = max(1, int(round(len(order) * keep_frac)))
    if mode == "middle":
        start = (len(order) - n_keep) // 2
        end = start + n_keep
        kept = order[start:end]
        dropped_low = start
        dropped_high = len(order) - end
    elif mode == "low":
        kept = order[:n_keep]
        dropped_low = 0
        dropped_high = len(order) - n_keep
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

    For kNN filtering, the intended "overlap-like" points are not
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


def committee_disagreement_on_strong_train(
    weak_train_activations: torch.Tensor,
    weak_train_labels: np.ndarray,
    strong_train_activations: torch.Tensor,
    members: int,
    l2_penalty: float,
    max_iter: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Query-by-committee disagreement for each strong_train point.

    Trains ``members`` weak probes on bootstrap resamples of weak_train and
    predicts every strong_train point with each one. The committee's per-point
    disagreement (standard deviation of the member probabilities) operationalizes
    the active-learning "region of disagreement": high disagreement means the
    plausible weak probes cannot agree on the label (boundary / unreliable weak
    label), low disagreement means they concur (reliable weak label). strong_train
    is out-of-sample for every member, so the estimate is honest by construction.
    """
    n = weak_train_activations.shape[0]
    members = max(2, members)
    rng = np.random.default_rng(seed)
    labels_t = torch.as_tensor(np.asarray(weak_train_labels), dtype=torch.float32)
    member_probs = []
    for _ in range(members):
        boot = rng.integers(0, n, size=n)
        boot_t = torch.as_tensor(boot, dtype=torch.long)
        probe = fit_probe(
            weak_train_activations[boot_t],
            labels_t[boot_t],
            l2_penalty,
            max_iter,
            device,
        )
        member_probs.append(predict_probe(probe, strong_train_activations, device))
    stacked = np.stack(member_probs, axis=0)
    mean_prob = stacked.mean(axis=0)
    disagreement = stacked.std(axis=0)
    return mean_prob, disagreement


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
    eval3_examples: list[LoraExample] | None = None,
    curriculum_order=None,
) -> tuple[dict, list[dict]]:
    if args.train_seed is not None:
        torch.manual_seed(args.train_seed)
        np.random.seed(args.train_seed)
        random.seed(args.train_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.train_seed)
    model, tokenizer, report = train_lora_model(
        args, train_examples, train_labels, run_name, output_dir, curriculum_order=curriculum_order
    )
    eval_summary, rows = evaluate_yes_no(
        model,
        tokenizer,
        eval_examples,
        args.strong_batch_size,
        args.device,
        args.max_length,
        f"eval {run_name}",
    )
    eval_summary["auroc"] = auroc_from_rows(rows)
    eval_summary["accuracy_prior_matched"] = prior_matched_accuracy(rows)
    if eval3_examples:
        _, rows3 = evaluate_yes_no(
            model,
            tokenizer,
            eval3_examples,
            args.strong_batch_size,
            args.device,
            args.max_length,
            f"eval3 {run_name}",
        )
        eval_summary["accuracy_3class"] = accuracy_3class(eval3_examples, rows3)
        write_eval3_rows(output_dir / f"eval3_{run_name}.csv", eval3_examples, rows3)
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


def _load_causal_lm_for_scoring(model_name: str, args: argparse.Namespace, device):
    """Load an arbitrary model name as a causal LM + tokenizer for yes/no scoring."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run_dream_w2s_baselines import resolve_dtype

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=resolve_dtype(args.torch_dtype), low_cpu_mem_usage=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.to(device)
    return model, tokenizer


def compute_el_kway_scores(args: argparse.Namespace, strong_train_ds, device) -> np.ndarray:
    """Excess-loss / learnability score per strong_train question (in row order):
    H_weak(options) - H_strong(options), where each model's option distribution is the
    softmax of its yes/no P(correct) over the K answer options. Loads the weak and the
    untuned-strong model once each (no probe); independent of LoRA training, so computed
    once per data seed and reused across train seeds and bands.
    """
    from run_dream_w2s_baselines import load_strong_model_and_tokenizer

    if "mc_options" not in strong_train_ds.column_names:
        raise SystemExit(
            "el_* selectors need per-question options; this dataset's formatter does not "
            "emit 'mc_options'. Add it to the format_* function before requesting el runs."
        )
    source_ids = list(strong_train_ds["source_id"])
    mc_options = list(strong_train_ds["mc_options"])
    opt_examples = [
        LoraExample(
            id=f"{src}::opt{k}",
            source_id=src,
            text=str(otext),
            label=0,
            answer_suffix=args.answer_suffix,
        )
        for src, opts in zip(source_ids, mc_options)
        for k, otext in enumerate(opts)
    ]
    el_batch = max(int(getattr(args, "strong_batch_size", 1)), 8)

    def _option_probs_by_q(model, tokenizer, desc):
        _, rows = evaluate_yes_no(
            model, tokenizer, opt_examples, el_batch, device, args.max_length, desc
        )
        by_q: dict[str, list[float]] = {}
        for ex, row in zip(opt_examples, rows):
            by_q.setdefault(ex.source_id, []).append(float(row["prob_label1"]))
        return by_q

    weak_model, weak_tok = _load_causal_lm_for_scoring(args.weak_model, args, device)
    weak_by_q = _option_probs_by_q(weak_model, weak_tok, "el K-option (weak)")
    del weak_model
    clear_memory()

    strong_model, strong_tok = load_strong_model_and_tokenizer(args, trainable_lora=False)
    strong_by_q = _option_probs_by_q(strong_model, strong_tok, "el K-option (base strong)")
    del strong_model
    clear_memory()

    el_by_q = excess_loss_kway_scores(weak_by_q, strong_by_q)
    return np.asarray([el_by_q.get(src, 0.0) for src in source_ids], dtype=np.float64)


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
    texts = split_texts(splits)
    if args.diagnostics_only:
        run_knn_saturation_diagnostics(args, splits, texts, device, output_dir)
        return

    lora_examples = {
        "weak_train": to_lora_examples(splits.weak_train, args.answer_suffix),
        "strong_train": to_lora_examples(splits.strong_train, args.answer_suffix),
        "test": to_lora_examples(splits.test, args.answer_suffix),
    }

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

    committee_mean_strong, committee_disagreement_strong = committee_disagreement_on_strong_train(
        weak_acts["weak_train"],
        weak_train_labels,
        weak_acts["strong_train"],
        args.committee_members,
        args.l2_penalty,
        args.max_iter,
        device,
        args.seed + 9000,
    )

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
    rp_scores = representation_projection_scores(
        np.asarray(weak_acts["strong_train"], dtype=np.float64),
        np.asarray(strong_acts["strong_train"], dtype=np.float64),
        weak_preds_strong,
        reg=args.rp_reg,
        n_components=(args.rp_components or None),
    )
    del strong_acts
    clear_memory()
    # excess-loss / learnability (K-way option entropy): H_weak(options) - H_strong(options),
    # each model's yes/no head over the K options (no probe). Expensive (loads the weak +
    # untuned-strong models once), so only when an el_* run is requested; independent of LoRA
    # training -> computed once per data seed.
    el_scores = None
    if any(r.strip().startswith("el_") for r in args.runs.split(",")):
        el_scores = compute_el_kway_scores(args, splits.strong_train, args.device)

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
    knn_high_indices, knn_high_filter = score_band_indices(
        knn_stats["knn_correct_rate"],
        args.knn_keep_middle_frac,
        "high",
    )
    knn_high_balanced_indices = hard_weak_label_balance(
        knn_high_indices,
        weak_preds_strong,
        args.seed + 5500,
    )
    committee_agree_indices, committee_agree_filter = score_band_indices(
        committee_disagreement_strong,
        args.committee_keep_frac,
        "low",
    )
    committee_agree_balanced_indices = hard_weak_label_balance(
        committee_agree_indices,
        weak_preds_strong,
        args.seed + 7000,
    )
    committee_disagree_indices, committee_disagree_filter = score_band_indices(
        committee_disagreement_strong,
        args.committee_keep_frac,
        "high",
    )
    committee_disagree_balanced_indices = hard_weak_label_balance(
        committee_disagree_indices,
        weak_preds_strong,
        args.seed + 8000,
    )
    random_control_size = args.random_control_size or len(middle_balanced_indices)

    strong_examples = lora_examples["strong_train"]
    eval_examples = lora_examples["test"]
    eval3_examples = None
    if args.eval_3class:
        if args.dataset == "dream":
            multichoice_rows = load_dream_3class_eval(args.n_eval_questions, args.seed)
        elif args.dataset == "sciq":
            multichoice_rows = load_sciq_multichoice_eval(
                args.n_eval_questions, args.seed + 2, args.sciq_use_support
            )
        elif args.dataset == "anli":
            multichoice_rows = load_anli_multichoice_eval(
                args.n_eval_questions, args.seed + 2, args.anli_round
            )
        elif args.dataset == "hellaswag":
            multichoice_rows = load_hellaswag_multichoice_eval(args.n_eval_questions, args.seed + 2)
        else:
            multichoice_rows = []
        if multichoice_rows:
            eval3_examples = [
                LoraExample(
                    id=r["id"],
                    source_id=r["source_id"],
                    text=r["txt"],
                    label=int(r["labels"]),
                    answer_suffix=args.answer_suffix,
                )
                for r in multichoice_rows
            ]
            n_questions = len({r["source_id"] for r in multichoice_rows})
            print(
                f"[multichoice] built {len(eval3_examples)} candidate rows over "
                f"{n_questions} {args.dataset} questions"
            )
    # Weak-model multiple-choice accuracy = the LOWER BOUND for the bounds table /
    # multichoice PGR. Score the candidate set with the (already-fitted) weak probe and
    # argmax over each question's candidates, same as the strong-side multichoice metric.
    weak_eval3_acc = None
    if eval3_examples:
        weak_eval3_acts = extract_final_token_activations(
            args.weak_model,
            [ex.text for ex in eval3_examples],
            device,
            args.torch_dtype,
            args.activation_batch_size,
            args.activation_max_length,
            "extract weak eval3 activations",
        )
        weak_eval3_rows = [
            {"prob_label1": float(p)} for p in predict_probe(weak_probe, weak_eval3_acts, device)
        ]
        weak_eval3_acc = accuracy_3class(eval3_examples, weak_eval3_rows)
        write_eval3_rows(output_dir / "eval3_weak.csv", eval3_examples, weak_eval3_rows)
        del weak_eval3_acts
        clear_memory()
        print(f"[multichoice] weak-probe lower bound accuracy_3class = {weak_eval3_acc:.3f}")
    weak_labels = weak_preds_strong.tolist()
    # Curriculum: same full no-filter set as weak_label, trained in a fixed order by weak
    # confidence (easy = most-confident first, hard = least-confident first). Control is
    # weak_label (random order); see train_lora_model's curriculum_order.
    curriculum_orders = {
        "curriculum_easy": list(np.argsort(-weak_confidences_strong)),
        "curriculum_hard": list(np.argsort(weak_confidences_strong)),
    }
    run_subsets: dict[str, tuple[list[LoraExample], list[int]]] = {
        "ground_truth": (strong_examples, strong_train_labels.tolist()),
        "weak_label": (strong_examples, weak_labels),
        "curriculum_easy": (strong_examples, weak_labels),
        "curriculum_hard": (strong_examples, weak_labels),
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
        "knn_high_unbalanced": (
            [strong_examples[int(i)] for i in knn_high_indices],
            [weak_labels[int(i)] for i in knn_high_indices],
        ),
        "knn_high_balanced": (
            [strong_examples[int(i)] for i in knn_high_balanced_indices],
            [weak_labels[int(i)] for i in knn_high_balanced_indices],
        ),
        "committee_agree_unbalanced": (
            [strong_examples[int(i)] for i in committee_agree_indices],
            [weak_labels[int(i)] for i in committee_agree_indices],
        ),
        "committee_agree_balanced": (
            [strong_examples[int(i)] for i in committee_agree_balanced_indices],
            [weak_labels[int(i)] for i in committee_agree_balanced_indices],
        ),
        "committee_disagree_unbalanced": (
            [strong_examples[int(i)] for i in committee_disagree_indices],
            [weak_labels[int(i)] for i in committee_disagree_indices],
        ),
        "committee_disagree_balanced": (
            [strong_examples[int(i)] for i in committee_disagree_balanced_indices],
            [weak_labels[int(i)] for i in committee_disagree_balanced_indices],
        ),
    }
    # Optional committee selection sweep over keep fractions (label-complexity /
    # data-efficiency curve): for each f, keep the most-reliable (low-disagreement)
    # or most-boundary (high-disagreement) f-fraction, plus a matched random_balanced.
    committee_keep_fracs = [
        float(x) for x in (args.committee_keep_fracs or "").split(",") if x.strip()
    ]
    n_strong_examples = len(strong_examples)
    for fi, frac in enumerate(committee_keep_fracs):
        pct = int(round(frac * 100))
        agree_idx, _ = score_band_indices(committee_disagreement_strong, frac, "low")
        disagree_idx, _ = score_band_indices(committee_disagreement_strong, frac, "high")
        agree_bal = hard_weak_label_balance(agree_idx, weak_preds_strong, args.seed + 10000 + fi)
        disagree_bal = hard_weak_label_balance(disagree_idx, weak_preds_strong, args.seed + 11000 + fi)
        rand_bal = random_balanced_indices(
            weak_preds_strong, int(round(frac * n_strong_examples)), args.seed + 12000 + fi
        )
        run_subsets[f"committee_agree_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in agree_bal],
            [weak_labels[int(i)] for i in agree_bal],
        )
        run_subsets[f"committee_agree_unbalanced_f{pct}"] = (
            [strong_examples[int(i)] for i in agree_idx],
            [weak_labels[int(i)] for i in agree_idx],
        )
        run_subsets[f"committee_disagree_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in disagree_bal],
            [weak_labels[int(i)] for i in disagree_bal],
        )
        run_subsets[f"random_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in rand_bal],
            [weak_labels[int(i)] for i in rand_bal],
        )
        knn_high_idx, _ = score_band_indices(knn_stats["knn_correct_rate"], frac, "high")
        knn_high_bal = hard_weak_label_balance(knn_high_idx, weak_preds_strong, args.seed + 13000 + fi)
        run_subsets[f"knn_high_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in knn_high_bal],
            [weak_labels[int(i)] for i in knn_high_bal],
        )
        conf_high_idx, _ = score_band_indices(weak_confidences_strong, frac, "high")
        conf_high_bal = hard_weak_label_balance(conf_high_idx, weak_preds_strong, args.seed + 14000 + fi)
        run_subsets[f"confidence_high_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in conf_high_bal],
            [weak_labels[int(i)] for i in conf_high_bal],
        )
        conf_low_idx, _ = score_band_indices(weak_confidences_strong, frac, "low")
        conf_low_bal = hard_weak_label_balance(conf_low_idx, weak_preds_strong, args.seed + 14500 + fi)
        run_subsets[f"confidence_low_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in conf_low_bal],
            [weak_labels[int(i)] for i in conf_low_bal],
        )
        knn_low_idx, _ = score_band_indices(knn_stats["knn_correct_rate"], frac, "low")
        knn_low_bal = hard_weak_label_balance(knn_low_idx, weak_preds_strong, args.seed + 13500 + fi)
        run_subsets[f"knn_low_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in knn_low_bal],
            [weak_labels[int(i)] for i in knn_low_bal],
        )
        rp_high_idx, _ = score_band_indices(rp_scores, frac, "high")
        rp_high_bal = hard_weak_label_balance(rp_high_idx, weak_preds_strong, args.seed + 16000 + fi)
        run_subsets[f"rp_high_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in rp_high_bal],
            [weak_labels[int(i)] for i in rp_high_bal],
        )
        rp_low_idx, _ = score_band_indices(rp_scores, frac, "low")
        rp_low_bal = hard_weak_label_balance(rp_low_idx, weak_preds_strong, args.seed + 16500 + fi)
        run_subsets[f"rp_low_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in rp_low_bal],
            [weak_labels[int(i)] for i in rp_low_bal],
        )
        # excess-loss / learnability (K-way option entropy): H_weak(options) - H_strong(options)
        if el_scores is not None:
            el_high_idx, _ = score_band_indices(el_scores, frac, "high")
            el_high_bal = hard_weak_label_balance(el_high_idx, weak_preds_strong, args.seed + 17000 + fi)
            run_subsets[f"el_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in el_high_bal],
                [weak_labels[int(i)] for i in el_high_bal],
            )
            el_low_idx, _ = score_band_indices(el_scores, frac, "low")
            el_low_bal = hard_weak_label_balance(el_low_idx, weak_preds_strong, args.seed + 17500 + fi)
            run_subsets[f"el_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in el_low_bal],
                [weak_labels[int(i)] for i in el_low_bal],
            )
        knn_mixed_idx, _ = score_closest_indices(
            knn_stats["knn_correct_rate"], frac, args.knn_mixed_center, "mixed"
        )
        knn_mixed_bal = hard_weak_label_balance(knn_mixed_idx, weak_preds_strong, args.seed + 15000 + fi)
        run_subsets[f"knn_mixed_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in knn_mixed_bal],
            [weak_labels[int(i)] for i in knn_mixed_bal],
        )
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
        base_summary["auroc"] = auroc_from_rows(base_rows)
        base_summary["accuracy_prior_matched"] = prior_matched_accuracy(base_rows)
        if eval3_examples:
            _, base_rows3 = evaluate_yes_no(
                model,
                tokenizer,
                eval3_examples,
                args.strong_batch_size,
                args.device,
                args.max_length,
                "eval3 base",
            )
            base_summary["accuracy_3class"] = accuracy_3class(eval3_examples, base_rows3)
            write_eval3_rows(output_dir / "eval3_base.csv", eval3_examples, base_rows3)
        prediction_columns["base"] = base_rows
        run_reports["base"] = {"eval": base_summary}
        del model
        del tokenizer
        clear_memory()

    eff_batch = max(1, args.strong_batch_size * args.gradient_accumulation_steps)
    fixed_steps = args.max_train_steps
    for run_name in runs:
        if run_name == "base":
            continue
        train_examples, train_labels = run_subsets[run_name]
        if args.epochs > 0:
            # Train this run for `epochs` full passes over its ACTUAL subset
            # (no-filter -> all data; f50 -> the real 50%), instead of the fixed
            # compute-matched cap. no-filter therefore gets proportionally more steps.
            args.max_train_steps = args.epochs * max(1, math.ceil(len(train_examples) / eff_batch))
        report, rows = run_lora_eval(
            args, run_name, train_examples, train_labels, eval_examples, output_dir, eval3_examples,
            curriculum_order=curriculum_orders.get(run_name),
        )
        prediction_columns[run_name] = rows
        run_reports[run_name] = report
    args.max_train_steps = fixed_steps

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
    subset_summaries["committee_agree_diagnostics"] = subset_summary(
        committee_agree_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["committee_disagree_diagnostics"] = subset_summary(
        committee_disagree_balanced_indices,
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
            "embedding": "strong model final-layer final-token activations",
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
