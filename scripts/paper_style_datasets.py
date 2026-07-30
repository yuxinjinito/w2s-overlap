#!/usr/bin/env python3
"""Dataset loading and prompt formatting for the paper-style pipeline.

Sixteen testbeds, each reduced to the same binary shape: one row per candidate
answer, with the yes/no suffix appended at prompt time. Split out of
run_dream_paper_style_lora.py so the pipeline file is about the method rather
than about sixteen dataset schemas. This module may import contracts and
nothing else in the project: add a dataset here and nothing upstream changes
except the dispatch in load_paper_style_splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download

from contracts import LoraExample, SplitBundle


# --- Dream: the original bed. These lived in the probe script from the first
# phase; every other testbed already lived here, so Dream now does too.
def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return "\n".join(flatten_text(item) for item in value)
    return str(value)


def load_dream_raw(split: str) -> list[Any]:
    split_file = {
        "train": "train.json",
        "validation": "dev.json",
        "dev": "dev.json",
        "test": "test.json",
    }.get(split, f"{split}.json")
    data_path = hf_hub_download(
        repo_id="onionmonster/dream",
        filename=f"raw/{split_file}",
        repo_type="dataset",
    )
    return json.loads(Path(data_path).read_text(encoding="utf-8"))


def iter_dream_question_rows(raw_data: list[Any]):
    """Yield rows shaped like the original HF Dream builder examples.

    The current `datasets` package no longer reliably loads the old Dream script
    on our machine, so we read the raw JSON and construct the same fields the
    original repo's formatter expects: dialogue, question, answer, choice.
    """

    for row_index, ex in enumerate(raw_data):
        if isinstance(ex, dict):
            dialogue = ex.get("dialogue") or ex.get("article") or ex.get("context") or ex.get("text") or []
            questions = ex.get("questions") or ex.get("question") or ex.get("qas") or []
            source_id = ex.get("id", row_index)
        elif isinstance(ex, (list, tuple)) and len(ex) >= 2:
            dialogue = ex[0]
            questions = ex[1]
            source_id = ex[2] if len(ex) > 2 else row_index
        else:
            continue

        if isinstance(questions, dict):
            questions = [questions]
        for question_index, qa in enumerate(questions):
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question", ""))
            answer = str(qa.get("answer", ""))
            choices = qa.get("choice") or qa.get("choices") or []
            if not question or not answer or not choices:
                continue
            yield {
                "source_id": f"{source_id}-{question_index}",
                "dialogue": dialogue,
                "question": question,
                "answer": answer,
                "choice": [str(choice) for choice in choices],
            }


def format_dream_paper_style(ex: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Match `format_dream` in the original repo."""

    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["answer"]
    else:
        distractors = list(ex["choice"])
        distractors.remove(ex["answer"])
        ans = rng.choice(distractors)

    joined = "\n".join(ex["dialogue"]) if isinstance(ex["dialogue"], list) else flatten_text(ex["dialogue"])
    txt = f"{joined}\n\nQ: {ex['question']} A: {ans}"
    choices = list(ex["choice"])
    return {
        "id": hashlib.sha1(txt.encode()).hexdigest()[:8],
        "source_id": ex["source_id"],
        "txt": txt,
        "labels": hard_label,
        "gt_labels": hard_label,
        "mc_options": [f"{joined}\n\nQ: {ex['question']} A: {c}" for c in choices],
        "mc_correct": choices.index(ex["answer"]),
    }


def balance_binary_dataset(ds: Dataset, seed: int) -> Dataset:
    labels = [int(label) for label in ds["labels"]]
    counts = {0: labels.count(0), 1: labels.count(1)}
    if not counts[0] or not counts[1]:
        return ds

    majority_label = 0 if counts[0] > counts[1] else 1
    minority_label = 1 - majority_label
    minority_count = counts[minority_label]
    minority_ds = ds.filter(lambda ex: int(ex["labels"]) == minority_label)
    majority_ds = (
        ds.filter(lambda ex: int(ex["labels"]) == majority_label)
        .shuffle(seed=seed)
        .select(range(minority_count))
    )
    return concatenate_datasets([minority_ds, majority_ds]).shuffle(seed=seed)


def load_and_process_dream_split(split: str, n_docs: int, seed: int) -> Dataset:
    raw_rows = list(iter_dream_question_rows(load_dream_raw(split)))
    raw_ds = Dataset.from_list(raw_rows).shuffle(seed=seed)
    rng = random.Random(seed)
    formatted_rows = [format_dream_paper_style(ex, rng) for ex in raw_ds]
    ds = Dataset.from_list(formatted_rows)
    ds = ds.filter(lambda ex: ex["txt"] != "")
    ds = balance_binary_dataset(ds, seed)
    if len(ds) < n_docs:
        print(f"dream/{split} has < {n_docs} docs after balancing, using all {len(ds)}")
        return ds
    return ds.select(range(n_docs))


def load_paper_dream_splits(n_train: int, n_val: int, n_test: int, seed: int) -> SplitBundle:
    train_pool = load_and_process_dream_split("train", n_train + n_val, seed)
    test = load_and_process_dream_split("test", n_test, seed)

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


def _mc_candidate_row(prefix: str, row_id: int, ctx: str, options: list[str], gold: int,
                      rng: random.Random) -> dict:
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        cand = gold
    else:
        cand = rng.choice([k for k in range(len(options)) if k != gold])
    return {
        "id": f"{prefix}-{row_id}",
        "source_id": f"{prefix}-{row_id}",
        "txt": f"{ctx} {options[cand]}",
        "labels": hard_label,
        "gt_labels": hard_label,
        "mc_options": [f"{ctx} {o}" for o in options],
        "mc_correct": gold,
    }


def _reclor_ctx(ex: dict) -> str:
    return f"{ex['context'].strip()}\nQ: {ex['question'].strip()} A:"


def format_reclor_paper_style(ex: dict, row_id: int, rng: random.Random) -> dict:
    options = [str(a).strip() for a in ex["answers"]]
    return _mc_candidate_row("reclor", row_id, _reclor_ctx(ex), options, int(ex["label"]), rng)


# --- ReClor / LogiQA2: logical-reasoning multiple-choice candidate-correctness testbeds.
# Same candidate-correctness reduction as ANLI/SciQ: sample a candidate option (50% gold,
# 50% a wrong one), the model judges yes/no; mc_options keeps all options for the MC eval.
def load_and_process_reclor_split(split: str, n_docs: int, seed: int) -> Dataset:
    raw = load_dataset("metaeval/reclor", split=split).shuffle(seed=seed)
    rng = random.Random(seed)
    rows = [format_reclor_paper_style(ex, i, rng) for i, ex in enumerate(raw)]
    ds = Dataset.from_list(rows).filter(lambda ex: ex["txt"] != "")
    ds = balance_binary_dataset(ds, seed)
    if len(ds) < n_docs:
        print(f"reclor/{split} has < {n_docs} docs after balancing, using all {len(ds)}")
    return ds.select(range(min(n_docs, len(ds))))


def load_paper_reclor_splits(n_train: int, n_val: int, n_test: int, seed: int) -> SplitBundle:
    # ReClor test labels are hidden -> train pool from "train", eval from labelled "validation".
    train_pool = load_and_process_reclor_split("train", n_train + n_val, seed)
    test = load_and_process_reclor_split("validation", n_test, seed + 2)
    val_count = min(n_val, len(train_pool))
    val = train_pool.select(range(val_count))
    train = train_pool.select(range(val_count, len(train_pool)))
    halves = train.train_test_split(test_size=0.5, seed=seed)
    return SplitBundle(weak_train=halves["train"], strong_train=halves["test"], val=val, test=test)


def load_reclor_multichoice_eval(n_questions: int, seed: int) -> list[dict]:
    raw = load_dataset("metaeval/reclor", split="validation").shuffle(seed=seed)
    rng = random.Random(seed + 7)
    out: list[dict] = []
    n = 0
    for ex in raw:
        if n >= n_questions:
            break
        options = [str(a).strip() for a in ex["answers"]]
        gold = int(ex["label"])
        ctx = _reclor_ctx(ex)
        for slot, o in enumerate(options):
            out.append({"id": f"reclor-mc-{n}-{slot}", "source_id": f"reclor-mc-{n}",
                        "txt": f"{ctx} {o}", "labels": int(slot == gold)})
        n += 1
    return out


def _iter_logiqa2_mc(split: str):
    """LogiQA2.0 HF mirror mixes NLI-style rows in; yield only the MC rows (parsed JSON)."""
    raw = load_dataset("datatune/LogiQA2.0", split=split)
    for row in raw:
        try:
            ex = json.loads(row["text"])
        except Exception:
            continue
        if not all(k in ex for k in ("options", "answer", "question", "text")):
            continue
        try:
            gold = int(ex["answer"])
        except (ValueError, TypeError):
            continue
        options = [str(o).strip() for o in ex["options"]]
        if 0 <= gold < len(options):
            yield ex, options, gold


def _logiqa2_ctx(ex: dict) -> str:
    return f"{ex['text'].strip()}\nQ: {ex['question'].strip()} A:"


def load_and_process_logiqa2_split(split: str, n_docs: int, seed: int) -> Dataset:
    rng = random.Random(seed)
    rows = [
        _mc_candidate_row("logiqa2", i, _logiqa2_ctx(ex), options, gold, rng)
        for i, (ex, options, gold) in enumerate(_iter_logiqa2_mc(split))
    ]
    random.Random(seed + 1).shuffle(rows)
    ds = Dataset.from_list(rows).filter(lambda ex: ex["txt"] != "")
    ds = balance_binary_dataset(ds, seed)
    if len(ds) < n_docs:
        print(f"logiqa2/{split} has < {n_docs} MC docs after balancing, using all {len(ds)}")
    return ds.select(range(min(n_docs, len(ds))))


def load_paper_logiqa2_splits(n_train: int, n_val: int, n_test: int, seed: int) -> SplitBundle:
    train_pool = load_and_process_logiqa2_split("train", n_train + n_val, seed)
    test = load_and_process_logiqa2_split("test", n_test, seed + 2)
    val_count = min(n_val, len(train_pool))
    val = train_pool.select(range(val_count))
    train = train_pool.select(range(val_count, len(train_pool)))
    halves = train.train_test_split(test_size=0.5, seed=seed)
    return SplitBundle(weak_train=halves["train"], strong_train=halves["test"], val=val, test=test)


def load_logiqa2_multichoice_eval(n_questions: int, seed: int) -> list[dict]:
    rows = list(_iter_logiqa2_mc("test"))
    random.Random(seed).shuffle(rows)
    out: list[dict] = []
    for n, (ex, options, gold) in enumerate(rows[:n_questions]):
        ctx = _logiqa2_ctx(ex)
        for slot, o in enumerate(options):
            out.append({"id": f"logiqa2-mc-{n}-{slot}", "source_id": f"logiqa2-mc-{n}",
                        "txt": f"{ctx} {o}", "labels": int(slot == gold)})
    return out


WANLI_REL2IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}


def format_wanli_paper_style(ex: dict, row_id: int, rng: random.Random) -> dict:
    gold = WANLI_REL2IDX.get(ex["gold"])
    if gold is None:
        return {"id": "", "source_id": "", "txt": "", "labels": 0, "gt_labels": 0,
                "mc_options": [], "mc_correct": 0}
    hard_label = int(rng.random() < 0.5)
    candidate = ANLI_LABELS[gold] if hard_label else rng.choice(
        [name for key, name in ANLI_LABELS.items() if key != gold]
    )
    base = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\n"
    q = "Q: What is the relationship from the premise to the hypothesis? A:"
    return {
        "id": f"wanli-{row_id}", "source_id": f"wanli-{row_id}",
        "txt": f"{base}{q} {candidate}",
        "labels": hard_label, "gt_labels": hard_label,
        "mc_options": [f"{base}{q} {ANLI_LABELS[k]}" for k in (0, 1, 2)],
        "mc_correct": gold,
    }


# --- WANLI: worker-and-AI adversarial NLI (3-way), same candidate-relation reduction as ANLI.
def load_and_process_wanli_split(split: str, n_docs: int, seed: int) -> Dataset:
    raw = load_dataset("alisawuffles/WANLI", split=split).shuffle(seed=seed)
    raw = raw.filter(lambda ex: ex["gold"] in WANLI_REL2IDX)
    rng = random.Random(seed)
    rows = [format_wanli_paper_style(ex, i, rng) for i, ex in enumerate(raw)]
    ds = Dataset.from_list(rows).filter(lambda ex: ex["txt"] != "")
    ds = balance_binary_dataset(ds, seed)
    if len(ds) < n_docs:
        print(f"wanli/{split} has < {n_docs} docs after balancing, using all {len(ds)}")
    return ds.select(range(min(n_docs, len(ds))))


def load_paper_wanli_splits(n_train: int, n_val: int, n_test: int, seed: int) -> SplitBundle:
    train_pool = load_and_process_wanli_split("train", n_train + n_val, seed)
    test = load_and_process_wanli_split("test", n_test, seed + 2)
    val_count = min(n_val, len(train_pool))
    val = train_pool.select(range(val_count))
    train = train_pool.select(range(val_count, len(train_pool)))
    halves = train.train_test_split(test_size=0.5, seed=seed)
    return SplitBundle(weak_train=halves["train"], strong_train=halves["test"], val=val, test=test)


def load_wanli_multichoice_eval(n_questions: int, seed: int) -> list[dict]:
    raw = load_dataset("alisawuffles/WANLI", split="test").shuffle(seed=seed)
    rng = random.Random(seed + 7)
    out: list[dict] = []
    n = 0
    for ex in raw:
        if n >= n_questions:
            break
        gold = WANLI_REL2IDX.get(ex["gold"])
        if gold is None:
            continue
        base = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\n"
        q = "Q: What is the relationship from the premise to the hypothesis? A:"
        for slot, ri in enumerate(rng.sample(range(3), 3)):
            out.append({"id": f"wanli-mc-{n}-{slot}", "source_id": f"wanli-mc-{n}",
                        "txt": f"{base}{q} {ANLI_LABELS[ri]}", "labels": int(ri == gold)})
        n += 1
    return out


CONTROL_REL2IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}


def format_control_paper_style(ex: dict, row_id: int, rng: random.Random) -> dict:
    gold = CONTROL_REL2IDX.get(ex["label"])
    if gold is None:
        return {"id": "", "source_id": "", "txt": "", "labels": 0, "gt_labels": 0,
                "mc_options": [], "mc_correct": 0}
    hard_label = int(rng.random() < 0.5)
    candidate = ANLI_LABELS[gold] if hard_label else rng.choice(
        [name for key, name in ANLI_LABELS.items() if key != gold]
    )
    base = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\n"
    q = "Q: What is the relationship from the premise to the hypothesis? A:"
    return {
        "id": f"control-{row_id}", "source_id": f"control-{row_id}",
        "txt": f"{base}{q} {candidate}",
        "labels": hard_label, "gt_labels": hard_label,
        "mc_options": [f"{base}{q} {ANLI_LABELS[k]}" for k in (0, 1, 2)],
        "mc_correct": gold,
    }


# --- ConTRoL: contextual-reasoning adversarial NLI (3-way), same reduction as ANLI/WANLI.
def load_and_process_control_split(split: str, n_docs: int, seed: int) -> Dataset:
    raw = load_dataset("tasksource/ConTRoL-nli", split=split).shuffle(seed=seed)
    raw = raw.filter(lambda ex: ex["label"] in CONTROL_REL2IDX)
    rng = random.Random(seed)
    rows = [format_control_paper_style(ex, i, rng) for i, ex in enumerate(raw)]
    ds = Dataset.from_list(rows).filter(lambda ex: ex["txt"] != "")
    ds = balance_binary_dataset(ds, seed)
    if len(ds) < n_docs:
        print(f"control/{split} has < {n_docs} docs after balancing, using all {len(ds)}")
    return ds.select(range(min(n_docs, len(ds))))


def load_paper_control_splits(n_train: int, n_val: int, n_test: int, seed: int) -> SplitBundle:
    train_pool = load_and_process_control_split("train", n_train + n_val, seed)
    test = load_and_process_control_split("test", n_test, seed + 2)
    val_count = min(n_val, len(train_pool))
    val = train_pool.select(range(val_count))
    train = train_pool.select(range(val_count, len(train_pool)))
    halves = train.train_test_split(test_size=0.5, seed=seed)
    return SplitBundle(weak_train=halves["train"], strong_train=halves["test"], val=val, test=test)


def load_control_multichoice_eval(n_questions: int, seed: int) -> list[dict]:
    raw = load_dataset("tasksource/ConTRoL-nli", split="test").shuffle(seed=seed)
    rng = random.Random(seed + 7)
    out: list[dict] = []
    n = 0
    for ex in raw:
        if n >= n_questions:
            break
        gold = CONTROL_REL2IDX.get(ex["label"])
        if gold is None:
            continue
        base = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\n"
        q = "Q: What is the relationship from the premise to the hypothesis? A:"
        for slot, ri in enumerate(rng.sample(range(3), 3)):
            out.append({"id": f"control-mc-{n}-{slot}", "source_id": f"control-mc-{n}",
                        "txt": f"{base}{q} {ANLI_LABELS[ri]}", "labels": int(ri == gold)})
        n += 1
    return out


FEVER_REL2IDX = {"SUPPORTS": 0, "NOT ENOUGH INFO": 1, "REFUTES": 2}


FEVER_LABELS = ["supported", "not enough info", "refuted"]


def format_fevernli_paper_style(ex: dict, row_id: int, rng: random.Random) -> dict:
    gold = FEVER_REL2IDX.get(str(ex["fever_gold_label"]))
    ev = str(ex.get("premise", "")).strip()
    cl = str(ex.get("hypothesis", "")).strip()
    if gold is None or not ev or not cl:
        return {"id": "", "source_id": "", "txt": "", "labels": 0, "gt_labels": 0,
                "mc_options": [], "mc_correct": 0}
    hard_label = int(rng.random() < 0.5)
    candidate = FEVER_LABELS[gold] if hard_label else rng.choice(
        [FEVER_LABELS[k] for k in range(3) if k != gold]
    )
    base = f"Evidence: {ev}\nClaim: {cl}\n"
    q = "Q: Based on the evidence, the claim is? A:"
    return {
        "id": f"fevernli-{row_id}", "source_id": f"fevernli-{row_id}",
        "txt": f"{base}{q} {candidate}",
        "labels": hard_label, "gt_labels": hard_label,
        "mc_options": [f"{base}{q} {FEVER_LABELS[k]}" for k in range(3)],
        "mc_correct": gold,
    }


def load_and_process_fevernli_split(split: str, n_docs: int, seed: int) -> Dataset:
    raw = load_dataset("pietrolesci/nli_fever", split=split).shuffle(seed=seed)
    raw = raw.filter(lambda ex: str(ex["fever_gold_label"]) in FEVER_REL2IDX)
    rng = random.Random(seed)
    rows = [format_fevernli_paper_style(ex, i, rng) for i, ex in enumerate(raw)]
    ds = Dataset.from_list(rows).filter(lambda ex: ex["txt"] != "")
    ds = balance_binary_dataset(ds, seed)
    if len(ds) < n_docs:
        print(f"fevernli/{split} has < {n_docs} docs after balancing, using all {len(ds)}")
    return ds.select(range(min(n_docs, len(ds))))


def load_paper_fevernli_splits(n_train: int, n_val: int, n_test: int, seed: int) -> SplitBundle:
    # nli_fever: labeled 'train' + 'dev' (test is unlabeled). Carve test from dev.
    train_pool = load_and_process_fevernli_split("train", n_train + n_val, seed)
    test = load_and_process_fevernli_split("dev", n_test, seed + 2)
    val_count = min(n_val, len(train_pool))
    val = train_pool.select(range(val_count))
    train = train_pool.select(range(val_count, len(train_pool)))
    halves = train.train_test_split(test_size=0.5, seed=seed)
    return SplitBundle(weak_train=halves["train"], strong_train=halves["test"], val=val, test=test)


def load_fevernli_multichoice_eval(n_questions: int, seed: int) -> list[dict]:
    raw = load_dataset("pietrolesci/nli_fever", split="dev").shuffle(seed=seed)
    rng = random.Random(seed + 7)
    out: list[dict] = []
    n = 0
    for ex in raw:
        if n >= n_questions:
            break
        gold = FEVER_REL2IDX.get(str(ex["fever_gold_label"]))
        ev = str(ex.get("premise", "")).strip()
        cl = str(ex.get("hypothesis", "")).strip()
        if gold is None or not ev or not cl:
            continue
        base = f"Evidence: {ev}\nClaim: {cl}\n"
        q = "Q: Based on the evidence, the claim is? A:"
        for slot, ri in enumerate(rng.sample(range(3), 3)):
            out.append({"id": f"fevernli-mc-{n}-{slot}", "source_id": f"fevernli-mc-{n}",
                        "txt": f"{base}{q} {FEVER_LABELS[ri]}", "labels": int(ri == gold)})
        n += 1
    return out


CONJNLI_TRAIN_URL = "https://raw.githubusercontent.com/swarnaHub/ConjNLI/master/data/NLI/adversarial_train_15k.tsv"


CONJNLI_DEV_URL = "https://raw.githubusercontent.com/swarnaHub/ConjNLI/master/data/NLI/conj_dev.tsv"


CONJNLI_REL2IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}


def _conjnli_rows(url: str):
    ds = load_dataset("csv", data_files={"d": url}, delimiter="\t")["d"]
    for ex in ds:
        keys = list(ex.keys())
        prem = str(ex.get("Premise", ex.get(keys[0], "")) or "").strip()
        hyp = str(ex.get("Hypothesis", ex.get(keys[1], "")) or "").strip()
        lab = str(ex.get("Label", ex.get(keys[2], "")) or "").strip().lower()
        gold = CONJNLI_REL2IDX.get(lab)
        if gold is None or not prem or not hyp:
            continue
        yield prem, hyp, gold


def format_conjnli_paper_style(prem: str, hyp: str, gold: int, row_id: int, rng: random.Random) -> dict:
    hard_label = int(rng.random() < 0.5)
    candidate = ANLI_LABELS[gold] if hard_label else rng.choice(
        [name for key, name in ANLI_LABELS.items() if key != gold]
    )
    base = f"Premise: {prem}\nHypothesis: {hyp}\n"
    q = "Q: What is the relationship from the premise to the hypothesis? A:"
    return {
        "id": f"conjnli-{row_id}", "source_id": f"conjnli-{row_id}",
        "txt": f"{base}{q} {candidate}",
        "labels": hard_label, "gt_labels": hard_label,
        "mc_options": [f"{base}{q} {ANLI_LABELS[k]}" for k in (0, 1, 2)],
        "mc_correct": gold,
    }


def load_and_process_conjnli_split(url: str, n_docs: int, seed: int) -> Dataset:
    rows = list(_conjnli_rows(url))
    rng0 = random.Random(seed)
    rng0.shuffle(rows)
    rng = random.Random(seed)
    formatted = [format_conjnli_paper_style(p, h, g, i, rng) for i, (p, h, g) in enumerate(rows)]
    ds = Dataset.from_list(formatted).filter(lambda ex: ex["txt"] != "")
    ds = balance_binary_dataset(ds, seed)
    if len(ds) < n_docs:
        print(f"conjnli has < {n_docs} docs after balancing, using all {len(ds)}")
    return ds.select(range(min(n_docs, len(ds))))


def load_paper_conjnli_splits(n_train: int, n_val: int, n_test: int, seed: int) -> SplitBundle:
    # Train pool = the authors' 15k adversarial (silver-label) train file; test carved from
    # the human-labeled dev (623 rows). NOTE: silver train labels -> the GT/oracle arms are
    # only as good as the silver labels on this bed.
    train_pool = load_and_process_conjnli_split(CONJNLI_TRAIN_URL, n_train + n_val, seed)
    test = load_and_process_conjnli_split(CONJNLI_DEV_URL, n_test, seed + 2)
    val_count = min(n_val, len(train_pool))
    val = train_pool.select(range(val_count))
    train = train_pool.select(range(val_count, len(train_pool)))
    halves = train.train_test_split(test_size=0.5, seed=seed)
    return SplitBundle(weak_train=halves["train"], strong_train=halves["test"], val=val, test=test)


def load_conjnli_multichoice_eval(n_questions: int, seed: int) -> list[dict]:
    rows = list(_conjnli_rows(CONJNLI_DEV_URL))
    random.Random(seed).shuffle(rows)
    rng = random.Random(seed + 7)
    out: list[dict] = []
    for n_i, (prem, hyp, gold) in enumerate(rows[:n_questions]):
        base = f"Premise: {prem}\nHypothesis: {hyp}\n"
        q = "Q: What is the relationship from the premise to the hypothesis? A:"
        for slot, ri in enumerate(rng.sample(range(3), 3)):
            out.append({"id": f"conjnli-mc-{n_i}-{slot}", "source_id": f"conjnli-mc-{n_i}",
                        "txt": f"{base}{q} {ANLI_LABELS[ri]}", "labels": int(ri == gold)})
    return out


IMPPRES_CONFIGS = [
    "presupposition_all_n_presupposition", "presupposition_both_presupposition",
    "presupposition_change_of_state", "presupposition_cleft_existence",
    "presupposition_cleft_uniqueness", "presupposition_only_presupposition",
    "presupposition_possessed_definites_existence", "presupposition_possessed_definites_uniqueness",
    "presupposition_question_presupposition",
]


def _imppres_rows(seed: int):
    """Merged presupposition-section rows -> (premise, hypothesis, gold 0/1/2), shuffled."""
    import random as _random

    rows = []
    for cfg in IMPPRES_CONFIGS:
        dd = load_dataset("facebook/imppres", cfg)
        ds = dd[list(dd.keys())[0]]
        for ex in ds:
            try:
                gold = int(str(ex.get("gold_label", "")).strip())
            except ValueError:
                continue
            prem = str(ex.get("premise", "")).strip()
            hyp = str(ex.get("hypothesis", "")).strip()
            if gold not in (0, 1, 2) or not prem or not hyp:
                continue
            rows.append((prem, hyp, gold))
    _random.Random(seed).shuffle(rows)
    return rows


def format_imppres_paper_style(prem: str, hyp: str, gold: int, row_id: int, rng: random.Random) -> dict:
    hard_label = int(rng.random() < 0.5)
    candidate = ANLI_LABELS[gold] if hard_label else rng.choice(
        [name for key, name in ANLI_LABELS.items() if key != gold]
    )
    base = f"Premise: {prem}\nHypothesis: {hyp}\n"
    q = "Q: What is the relationship from the premise to the hypothesis? A:"
    return {
        "id": f"imppres-{row_id}", "source_id": f"imppres-{row_id}",
        "txt": f"{base}{q} {candidate}",
        "labels": hard_label, "gt_labels": hard_label,
        "mc_options": [f"{base}{q} {ANLI_LABELS[k]}" for k in (0, 1, 2)],
        "mc_correct": gold,
    }


def load_paper_imppres_splits(n_train: int, n_val: int, n_test: int, seed: int) -> SplitBundle:
    # ImpPres is an eval-only templated corpus (~17k presupposition rows); carve
    # disjoint train/test pools from one shuffle. Templated data -> watch for
    # template memorization in Stage-2 (GT arm may saturate).
    rows = _imppres_rows(seed)
    test_rows = rows[:max(1, n_test * 2)]
    pool_rows = rows[max(1, n_test * 2):]
    rng = random.Random(seed)
    fmt_pool = [format_imppres_paper_style(p, h, g, i, rng) for i, (p, h, g) in enumerate(pool_rows)]
    fmt_test = [format_imppres_paper_style(p, h, g, 10_000_000 + i, rng) for i, (p, h, g) in enumerate(test_rows)]
    pool = balance_binary_dataset(Dataset.from_list(fmt_pool).filter(lambda ex: ex["txt"] != ""), seed)
    test = balance_binary_dataset(Dataset.from_list(fmt_test).filter(lambda ex: ex["txt"] != ""), seed + 2)
    pool = pool.select(range(min(n_train + n_val, len(pool))))
    test = test.select(range(min(n_test, len(test))))
    val_count = min(n_val, len(pool))
    val = pool.select(range(val_count))
    train = pool.select(range(val_count, len(pool)))
    halves = train.train_test_split(test_size=0.5, seed=seed)
    return SplitBundle(weak_train=halves["train"], strong_train=halves["test"], val=val, test=test)


def load_imppres_multichoice_eval(n_questions: int, seed: int) -> list[dict]:
    rows = _imppres_rows(seed)[:n_questions]
    rng = random.Random(seed + 7)
    out: list[dict] = []
    for n_i, (prem, hyp, gold) in enumerate(rows):
        base = f"Premise: {prem}\nHypothesis: {hyp}\n"
        q = "Q: What is the relationship from the premise to the hypothesis? A:"
        for slot, ri in enumerate(rng.sample(range(3), 3)):
            out.append({"id": f"imppres-mc-{n_i}-{slot}", "source_id": f"imppres-mc-{n_i}",
                        "txt": f"{base}{q} {ANLI_LABELS[ri]}", "labels": int(ri == gold)})
    return out


def _aquarat_rows(splits, seed: int):
    """AQuA-RAT rows -> (question, [5 option texts], gold_idx), merged over the given splits."""
    import random as _random

    rows = []
    for sp in splits:
        ds = load_dataset("deepmind/aqua_rat", "raw", split=sp)
        for ex in ds:
            opts_raw = [str(o).strip() for o in ex.get("options", [])]
            options = [o.split(")", 1)[1].strip() if ")" in o else o for o in opts_raw]
            gold = "ABCDE".find(str(ex.get("correct", "")).strip())
            q = str(ex.get("question", "")).strip()
            if not (0 <= gold < len(options)) or not q or len(options) != 5:
                continue
            rows.append((q, options, gold))
    _random.Random(seed).shuffle(rows)
    return rows


def format_aquarat_paper_style(q: str, options: list, gold: int, row_id: int, rng: random.Random) -> dict:
    hard_label = int(rng.random() < 0.5)
    cand = options[gold] if hard_label else options[rng.choice([k for k in range(len(options)) if k != gold])]
    base = f"Q: {q} A:"
    return {
        "id": f"aquarat-{row_id}", "source_id": f"aquarat-{row_id}",
        "txt": f"{base} {cand}",
        "labels": hard_label, "gt_labels": hard_label,
        "mc_options": [f"{base} {o}" for o in options],
        "mc_correct": gold,
    }


def load_paper_aquarat_splits(n_train: int, n_val: int, n_test: int, seed: int) -> SplitBundle:
    # 97k crowd-amplified train pool; official test+validation (254+254) merged as the test set.
    train_rows = _aquarat_rows(["train"], seed)[: (n_train + n_val) * 2]
    test_rows = _aquarat_rows(["test", "validation"], seed + 2)
    rng = random.Random(seed)
    pool = [format_aquarat_paper_style(q, o, g, i, rng) for i, (q, o, g) in enumerate(train_rows)]
    tst = [format_aquarat_paper_style(q, o, g, 10_000_000 + i, rng) for i, (q, o, g) in enumerate(test_rows)]
    pool_ds = balance_binary_dataset(Dataset.from_list(pool).filter(lambda ex: ex["txt"] != ""), seed)
    test_ds = balance_binary_dataset(Dataset.from_list(tst).filter(lambda ex: ex["txt"] != ""), seed + 2)
    pool_ds = pool_ds.select(range(min(n_train + n_val, len(pool_ds))))
    test_ds = test_ds.select(range(min(n_test, len(test_ds))))
    val_count = min(n_val, len(pool_ds))
    val = pool_ds.select(range(val_count))
    train = pool_ds.select(range(val_count, len(pool_ds)))
    halves = train.train_test_split(test_size=0.5, seed=seed)
    return SplitBundle(weak_train=halves["train"], strong_train=halves["test"], val=val, test=test_ds)


def load_aquarat_multichoice_eval(n_questions: int, seed: int) -> list[dict]:
    rows = _aquarat_rows(["test", "validation"], seed)[:n_questions]
    rng = random.Random(seed + 7)
    out: list[dict] = []
    for n_i, (q, options, gold) in enumerate(rows):
        base = f"Q: {q} A:"
        for slot, ri in enumerate(rng.sample(range(len(options)), len(options))):
            out.append({"id": f"aquarat-mc-{n_i}-{slot}", "source_id": f"aquarat-mc-{n_i}",
                        "txt": f"{base} {options[ri]}", "labels": int(ri == gold)})
    return out


def _quail_rows(split: str, seed: int):
    import random as _random

    ds = load_dataset("textmachinelab/quail", "default", split=split, revision="refs/convert/parquet")
    rows = []
    for ex in ds:
        options = [str(o).strip() for o in ex.get("answers", [])]
        gold = ex.get("correct_answer_id", -1)
        c = str(ex.get("context", "")).strip()
        q = str(ex.get("question", "")).strip()
        if not isinstance(gold, int) or not (0 <= gold < len(options)) or not c or not q:
            continue
        rows.append((f"{c}\nQ: {q} A:", options, gold))
    _random.Random(seed).shuffle(rows)
    return rows


def _riddlesense_rows(split: str, seed: int):
    import random as _random

    ds = load_dataset("INK-USC/riddle_sense", "default", split=split, revision="refs/convert/parquet")
    rows = []
    for ex in ds:
        labels = list(ex["choices"]["label"])
        options = [str(t).strip() for t in ex["choices"]["text"]]
        key = str(ex["answerKey"]).strip()
        q = str(ex.get("question", "")).strip()
        if key not in labels or not options or not q:
            continue
        rows.append((f"Q: {q} A:", options, labels.index(key)))
    _random.Random(seed).shuffle(rows)
    return rows


def _format_mc_rows_paper_style(rows, prefix: str, rng: random.Random, id_base: int = 0):
    out = []
    for i, (base, options, gold) in enumerate(rows):
        hard_label = int(rng.random() < 0.5)
        cand = options[gold] if hard_label else options[rng.choice([k for k in range(len(options)) if k != gold])]
        out.append({
            "id": f"{prefix}-{id_base + i}", "source_id": f"{prefix}-{id_base + i}",
            "txt": f"{base} {cand}",
            "labels": hard_label, "gt_labels": hard_label,
            "mc_options": [f"{base} {o}" for o in options],
            "mc_correct": gold,
        })
    return out


def _generic_mc_splits(train_rows, test_rows, prefix, n_train, n_val, n_test, seed) -> SplitBundle:
    rng = random.Random(seed)
    pool = _format_mc_rows_paper_style(train_rows, prefix, rng)
    tst = _format_mc_rows_paper_style(test_rows, prefix, rng, id_base=10_000_000)
    pool_ds = balance_binary_dataset(Dataset.from_list(pool).filter(lambda ex: ex["txt"] != ""), seed)
    test_ds = balance_binary_dataset(Dataset.from_list(tst).filter(lambda ex: ex["txt"] != ""), seed + 2)
    pool_ds = pool_ds.select(range(min(n_train + n_val, len(pool_ds))))
    test_ds = test_ds.select(range(min(n_test, len(test_ds))))
    val_count = min(n_val, len(pool_ds))
    val = pool_ds.select(range(val_count))
    train = pool_ds.select(range(val_count, len(pool_ds)))
    halves = train.train_test_split(test_size=0.5, seed=seed)
    return SplitBundle(weak_train=halves["train"], strong_train=halves["test"], val=val, test=test_ds)


def _generic_mc_eval(rows, prefix, n_questions, seed) -> list[dict]:
    rng = random.Random(seed + 7)
    out: list[dict] = []
    for n_i, (base, options, gold) in enumerate(rows[:n_questions]):
        for slot, ri in enumerate(rng.sample(range(len(options)), len(options))):
            out.append({"id": f"{prefix}-mc-{n_i}-{slot}", "source_id": f"{prefix}-mc-{n_i}",
                        "txt": f"{base} {options[ri]}", "labels": int(ri == gold)})
    return out


def _scinli_rows(split: str, seed: int):
    import random as _random

    ds = load_dataset("tasksource/scinli", split=split)
    opts = ["entailment", "contrasting", "reasoning", "neutral"]
    lab2idx = {o: i for i, o in enumerate(opts)}
    rows = []
    for ex in ds:
        gold = lab2idx.get(str(ex.get("label", "")).strip().lower())
        s1 = str(ex.get("sentence1", "")).strip()
        s2 = str(ex.get("sentence2", "")).strip()
        if gold is None or not s1 or not s2:
            continue
        base = (f"Sentence 1: {s1}\nSentence 2: {s2}\n"
                "Q: What is the relationship from sentence 1 to sentence 2? A:")
        rows.append((base, list(opts), gold))
    _random.Random(seed).shuffle(rows)
    return rows


def load_paper_scinli_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(_scinli_rows("train", seed), _scinli_rows("test", seed + 2),
                              "scinli", n_train, n_val, n_test, seed)


def load_paper_quail_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(_quail_rows("train", seed), _quail_rows("validation", seed + 2),
                              "quail", n_train, n_val, n_test, seed)


def load_paper_riddlesense_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(_riddlesense_rows("train", seed), _riddlesense_rows("validation", seed + 2),
                              "riddlesense", n_train, n_val, n_test, seed)


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
    if args.dataset == "reclor":
        return load_paper_reclor_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "logiqa2":
        return load_paper_logiqa2_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "wanli":
        return load_paper_wanli_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "control":
        return load_paper_control_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "fevernli":
        return load_paper_fevernli_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "conjnli":
        return load_paper_conjnli_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "imppres":
        return load_paper_imppres_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "aquarat":
        return load_paper_aquarat_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "quail":
        return load_paper_quail_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "riddlesense":
        return load_paper_riddlesense_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "scinli":
        return load_paper_scinli_splits(args.n_train, args.n_val, args.n_test, args.seed)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def split_texts(splits: SplitBundle) -> dict[str, list[str]]:
    return {
        "weak_train": list(splits.weak_train["txt"]),
        "strong_train": list(splits.strong_train["txt"]),
        "test": list(splits.test["txt"]),
    }


def split_labels(split) -> np.ndarray:
    return np.array(split["labels"], dtype=int)
