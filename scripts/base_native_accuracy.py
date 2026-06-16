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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hellaswag", choices=["hellaswag", "dream"])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--n-questions", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--torch-dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--answer-suffix", default=DEFAULT_SUFFIX)
    args = ap.parse_args()

    qs = (hellaswag_questions if args.dataset == "hellaswag" else dream_questions)(args.n_questions, args.seed)
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
