#!/usr/bin/env python3
"""Quick weak-model NATIVE multiple-choice baseline (no training).

Scores the WEAK model's per-option log-likelihood on the native multiple-choice
form of DREAM / SciQ / ANLI, picks argmax, and reports accuracy (raw sum and
length-normalized) vs chance. Tells us which dataset has usable weak signal
(weak comfortably > chance) BEFORE investing in the native multi-class training
pipeline -- the headroom/sweet-spot screen.

Usage (on a GPU box, env active):
  PYTHONPATH=scripts python scripts/weak_native_baseline.py --n 300
  PYTHONPATH=scripts python scripts/weak_native_baseline.py --datasets anli --n 500 --anli-round r3
"""
import argparse
import warnings

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_dream_paper_linear_probe import iter_dream_question_rows, load_dream_raw, flatten_text

warnings.filterwarnings("ignore")


def build_dream(n):
    items = []
    for ex in iter_dream_question_rows(load_dream_raw("test")):
        ch = list(ex["choice"])
        if ex["answer"] not in ch:
            continue
        joined = "\n".join(ex["dialogue"]) if isinstance(ex["dialogue"], list) else flatten_text(ex["dialogue"])
        items.append((f"{joined}\n\nQ: {ex['question']}\nA:", ch, ch.index(ex["answer"])))
        if len(items) >= n:
            break
    return items, 1 / 3


def build_sciq(n):
    items = []
    for ex in load_dataset("allenai/sciq", split="test"):
        opts = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        items.append((f"Q: {ex['question']}\nA:", opts, 0))
        if len(items) >= n:
            break
    return items, 1 / 4


def build_anli(n, anli_round="r2"):
    rel = ["entailment", "neutral", "contradiction"]
    items = []
    for ex in load_dataset("facebook/anli", split=f"test_{anli_round}"):
        if int(ex["label"]) not in (0, 1, 2):
            continue
        ctx = (
            f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\n"
            f"Q: What is the relationship from the premise to the hypothesis?\nA:"
        )
        items.append((ctx, rel, int(ex["label"])))
        if len(items) >= n:
            break
    return items, 1 / 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weak-model", default="Qwen/Qwen1.5-0.5B")
    ap.add_argument("--datasets", default="dream,sciq,anli")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--anli-round", default="r2")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={dev}  weak={args.weak_model}")
    tok = AutoTokenizer.from_pretrained(args.weak_model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.weak_model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32
        )
        .to(dev)
        .eval()
    )

    def opt_logprob(context, option):
        n_ctx = tok(context, return_tensors="pt").input_ids.shape[1]
        full = tok(context + option, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            lp = torch.log_softmax(model(full).logits[0].float(), dim=-1)
        idx = torch.arange(n_ctx, full.shape[1], device=dev)
        s = lp[idx - 1, full[0, idx]].sum().item()
        return s, max(int(idx.numel()), 1)

    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        if name == "anli":
            items, chance = build_anli(args.n, args.anli_round)
        elif name == "dream":
            items, chance = build_dream(args.n)
        elif name == "sciq":
            items, chance = build_sciq(args.n)
        else:
            print(f"unknown dataset {name}, skipping")
            continue
        raw_ok = norm_ok = 0
        for ctx, opts, gold in items:
            sc = [opt_logprob(ctx, " " + o) for o in opts]
            raw_ok += int(np.argmax([s for s, _ in sc]) == gold)
            norm_ok += int(np.argmax([s / nt for s, nt in sc]) == gold)
        n = len(items)
        best = max(raw_ok, norm_ok) / n
        print(
            f"{name:8} n={n}  weak raw={raw_ok / n:.3f}  norm={norm_ok / n:.3f}  "
            f"chance={chance:.2f}  signal={'YES (>chance+0.05)' if best > chance + 0.05 else 'weak/none'}"
        )


if __name__ == "__main__":
    main()
