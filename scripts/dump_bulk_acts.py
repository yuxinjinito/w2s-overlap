#!/usr/bin/env python3
"""Bulk activation dump for representation-space experiments (SAE training etc.).

Builds the same candidate-correctness rows as the pipeline (Premise/Hypothesis/Q/A +
candidate relation, no yes-no suffix) for many source examples, and saves last-layer
last-token activations of both models. Unsupervised consumers only need the acts; gold
candidate labels ride along for bookkeeping.

Usage (GPU box):
  PYTHONPATH=scripts python3 scripts/dump_bulk_acts.py --anli-round r1 \
      --n-examples 15000 --out bulk_acts_r1.npz
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

RELATIONS = ["entailment", "neutral", "contradiction"]


def build_rows(round_name: str, n_examples: int, seed: int):
    ds = load_dataset("facebook/anli", split=f"train_{round_name}")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n_examples]
    texts, gold = [], []
    for i in idx:
        ex = ds[int(i)]
        prem, hyp, lab = ex["premise"].strip(), ex["hypothesis"].strip(), int(ex["label"])
        ctx = (f"Premise: {prem}\nHypothesis: {hyp}\n"
               "Q: What is the relationship from the premise to the hypothesis? A:")
        for k, rel in enumerate(RELATIONS):
            texts.append(f"{ctx} {rel}")
            gold.append(int(k == lab))
    return texts, np.asarray(gold, dtype=np.int8)


@torch.no_grad()
def extract(model_id: str, texts, device: str, batch_size: int, max_length: int):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device).eval()
    out = []
    for b0 in range(0, len(texts), batch_size):
        batch = tok(texts[b0:b0 + batch_size], return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length).to(device)
        h = model(**batch, output_hidden_states=True).hidden_states[-1]
        idx = batch["attention_mask"].sum(dim=1) - 1
        out.append(h[torch.arange(h.shape[0]), idx].half().cpu())
        if (b0 // batch_size) % 100 == 0:
            print(f"  {model_id}: {b0}/{len(texts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anli-round", default="r1")
    ap.add_argument("--n-examples", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weak-model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--strong-model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="bulk_acts.npz")
    args = ap.parse_args()

    texts, gold = build_rows(args.anli_round, args.n_examples, args.seed)
    print(f"{len(texts)} rows from {args.n_examples} examples")
    weak = extract(args.weak_model, texts, args.device, args.batch_size * 4, args.max_length)
    strong = extract(args.strong_model, texts, args.device, args.batch_size, args.max_length)
    # key is "gt" to match the pipeline's --dump-acts output, so the screens can
    # read either file; this writer used to say "gold" and no reader ever did
    np.savez_compressed(args.out, weak=weak, strong=strong, gt=gold)
    print(f"wrote {args.out}: weak {weak.shape} strong {strong.shape}")


if __name__ == "__main__":
    main()
