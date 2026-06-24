#!/usr/bin/env python3
"""Native (likelihood) vs discrimination (yes/no) multiple-choice accuracy of a base model.

A low discrimination-format accuracy can mean either (a) the model does not know the task,
or (b) the untuned model simply does not follow the yes/no format. This scores each option
two ways and reports both, so the gap tells you which:

  - NATIVE        : length-normalised log P(option | context), argmax over options.
  - DISCRIMINATION: P(" yes") for "{context} {option}{suffix}", argmax of P(yes) over options
                    (this matches how the pipeline evaluates the base model).

If NATIVE >> DISCRIMINATION, the model knows the task and the low discrimination accuracy is
a format limitation -- so a weak-to-strong "lift" from training partly reflects the model
learning the format / re-surfacing pretrained knowledge, not new capability.

Usage (on a GPU box):
  PYTHONPATH=scripts python3 scripts/base_native_accuracy.py \
      --dataset hellaswag --model meta-llama/Llama-3.1-8B --n-questions 500 --device cuda
"""
import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_dream_w2s_baselines import resolve_dtype, score_candidates

DEFAULT_SUFFIX = " Is the candidate answer correct? Answer:"


def hellaswag_questions(n: int, seed: int):
    raw = load_dataset("Rowan/hellaswag", split="validation").shuffle(seed=seed)
    out = []
    for ex in raw:
        if ex["label"] == "":  # hidden test labels
            continue
        out.append(((ex.get("ctx") or "").strip(), [e.strip() for e in ex["endings"]], int(ex["label"])))
        if len(out) >= n:
            break
    return out


def dream_questions(n: int, seed: int):
    import random as _random

    from run_dream_paper_linear_probe import iter_dream_question_rows, load_dream_raw

    rows = list(iter_dream_question_rows(load_dream_raw("test")))
    _random.Random(seed).shuffle(rows)
    out = []
    for ex in rows:
        joined = "\n".join(ex["dialogue"]) if isinstance(ex["dialogue"], list) else str(ex["dialogue"])
        ctx = f"{joined}\n\nQ: {ex['question']} A:"
        choices = list(ex["choice"])
        out.append((ctx, choices, choices.index(ex["answer"])))
        if len(out) >= n:
            break
    return out


ANLI_RELATIONS = ["entailment", "neutral", "contradiction"]


def anli_questions(n: int, seed: int, anli_round: str = "r2"):
    raw = load_dataset("facebook/anli", split=f"test_{anli_round}").shuffle(seed=seed)
    out = []
    for ex in raw:
        gold = int(ex["label"])
        if gold not in (0, 1, 2):
            continue
        ctx = (
            f"Premise: {ex['premise']}\n"
            f"Hypothesis: {ex['hypothesis']}\n"
            "Q: What is the relationship from the premise to the hypothesis? A:"
        )
        out.append((ctx, list(ANLI_RELATIONS), gold))
        if len(out) >= n:
            break
    return out


def sciq_questions(n: int, seed: int):
    raw = load_dataset("allenai/sciq", split="test").shuffle(seed=seed)
    out = []
    for ex in raw:
        ctx = f"Q: {ex['question']} A:"
        options = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        out.append((ctx, options, 0))
        if len(out) >= n:
            break
    return out


# --- candidate W2S testbeds (Stage-1 headroom screen): adversarial NLI + logical-reasoning MC.
# All return (ctx, options, gold_idx), like the loaders above. Chosen because difficulty is from
# adversarial construction / reasoning (latent, elicitable capability), so they are plausibly
# base-near-chance AND robust to pretraining contamination (see dataset_collection_method note).

def wanli_questions(n: int, seed: int):
    """WANLI: worker-and-AI adversarial NLI (3-way relation), same format as ANLI."""
    raw = load_dataset("alisawuffles/WANLI", split="test").shuffle(seed=seed)
    rel2idx = {"entailment": 0, "neutral": 1, "contradiction": 2}
    out = []
    for ex in raw:
        gold = rel2idx.get(ex["gold"])
        if gold is None:
            continue
        ctx = (
            f"Premise: {ex['premise']}\n"
            f"Hypothesis: {ex['hypothesis']}\n"
            "Q: What is the relationship from the premise to the hypothesis? A:"
        )
        out.append((ctx, list(ANLI_RELATIONS), gold))
        if len(out) >= n:
            break
    return out


def reclor_questions(n: int, seed: int):
    """ReClor: LSAT/GMAT logical-reasoning MC (4 options). Test labels hidden -> use validation."""
    raw = load_dataset("metaeval/reclor", split="validation").shuffle(seed=seed)
    out = []
    for ex in raw:
        ctx = f"{ex['context'].strip()}\nQ: {ex['question'].strip()} A:"
        out.append((ctx, [str(a).strip() for a in ex["answers"]], int(ex["label"])))
        if len(out) >= n:
            break
    return out


def art_questions(n: int, seed: int):
    """ART / alphaNLI: abductive NLI -- pick the hypothesis that better explains the observations
    (2 options). label is 1-indexed (1 or 2)."""
    raw = load_dataset("allenai/art", split="validation").shuffle(seed=seed)
    out = []
    for ex in raw:
        ctx = (
            f"Observation 1: {ex['observation_1'].strip()}\n"
            f"Observation 2: {ex['observation_2'].strip()}\n"
            "Q: Which hypothesis better explains the observations? A:"
        )
        options = [ex["hypothesis_1"].strip(), ex["hypothesis_2"].strip()]
        out.append((ctx, options, int(ex["label"]) - 1))
        if len(out) >= n:
            break
    return out


def logiqa2_questions(n: int, seed: int):
    """LogiQA 2.0: logical-reasoning MC (4 options). Rows are a JSON string under 'text'.
    This HF mirror mixes in NLI-style rows (no options); keep only the MC rows."""
    import json as _json

    raw = load_dataset("datatune/LogiQA2.0", split="test").shuffle(seed=seed)
    out = []
    for row in raw:
        try:
            ex = _json.loads(row["text"])
        except Exception:
            continue
        if not all(k in ex for k in ("options", "answer", "question", "text")):
            continue  # skip the mixed-in NLI-style rows
        options = [str(o).strip() for o in ex["options"]]
        try:
            gold = int(ex["answer"])
        except (ValueError, TypeError):
            continue
        if not 0 <= gold < len(options):
            continue
        ctx = f"{ex['text'].strip()}\nQ: {ex['question'].strip()} A:"
        out.append((ctx, options, gold))
        if len(out) >= n:
            break
    return out


CANDIDATE_LOADERS = {
    "wanli": wanli_questions,
    "reclor": reclor_questions,
    "art": art_questions,
    "logiqa2": logiqa2_questions,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hellaswag",
                    choices=["hellaswag", "dream", "anli", "sciq", "wanli", "reclor", "art", "logiqa2"])
    ap.add_argument("--anli-round", default="r2", choices=["r1", "r2", "r3"])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--n-questions", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--torch-dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--answer-suffix", default=DEFAULT_SUFFIX)
    args = ap.parse_args()

    if args.dataset == "hellaswag":
        qs = hellaswag_questions(args.n_questions, args.seed)
    elif args.dataset == "dream":
        qs = dream_questions(args.n_questions, args.seed)
    elif args.dataset == "anli":
        qs = anli_questions(args.n_questions, args.seed, args.anli_round)
    elif args.dataset == "sciq":
        qs = sciq_questions(args.n_questions, args.seed)
    else:  # candidate testbeds: wanli / reclor / art / logiqa2
        qs = CANDIDATE_LOADERS[args.dataset](args.n_questions, args.seed)
    print(f"dataset={args.dataset}  model={args.model}  questions={len(qs)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=resolve_dtype(args.torch_dtype), low_cpu_mem_usage=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.to(args.device)
    model.eval()

    native_correct = 0
    disc_correct = 0
    with torch.no_grad():
        for ctx, options, gold in qs:
            # NATIVE: length-normalised P(option | ctx)
            opt_scores = score_candidates(
                model, tokenizer, [ctx] * len(options), [" " + o for o in options], args.device, args.max_length
            ).tolist()
            native_norm = [
                s / max(1, len(tokenizer(" " + o, add_special_tokens=False)["input_ids"]))
                for o, s in zip(options, opt_scores)
            ]
            native_correct += int(int(np.argmax(native_norm)) == gold)

            # DISCRIMINATION: P(yes) for "{ctx} {option}{suffix}"
            disc_prompts = [f"{ctx} {o}{args.answer_suffix}" for o in options]
            yes = score_candidates(model, tokenizer, disc_prompts, [" yes"] * len(options), args.device, args.max_length)
            no = score_candidates(model, tokenizer, disc_prompts, [" no"] * len(options), args.device, args.max_length)
            p_yes = torch.softmax(torch.stack([no, yes], dim=1), dim=1)[:, 1]
            disc_correct += int(int(torch.argmax(p_yes)) == gold)

    n = len(qs)
    chance = 1.0 / len(qs[0][1]) if qs else float("nan")
    print(f"chance                                        : {chance:.3f}")
    print(f"NATIVE   (likelihood, length-normalised) acc  : {native_correct / n:.3f}")
    print(f"DISCRIM  (yes/no P(correct)) acc              : {disc_correct / n:.3f}")
    print("If NATIVE >> DISCRIM, the low discrimination accuracy is a yes/no-format limitation,")
    print("not a knowledge gap -- so the W2S lift partly reflects format learning / recall.")


if __name__ == "__main__":
    main()
