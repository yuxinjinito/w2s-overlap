#!/usr/bin/env python3
"""Dump the weak model's zero-shot yes/no evidence margin for a bed's
strong_train rows.

The margin is score(" yes") minus score(" no") from the untuned weak model on
the same prompts the pipeline trains on: the anti-expert baseline for the
delta step-1 target. Rows rebuild deterministically from the dataset seed, and
the output carries weak_preds so consumers can verify alignment against an
acts dump before trusting the row order.

Usage (matches the R1 sweep sizes):
  PYTHONPATH=scripts python scripts/weak_prior_margin.py --dataset anli \
    --anli-round r1 --n-train 6000 --n-val 1000 --n-test 1000 \
    --acts acts_r1_seed42.npz --out weak_prior_r1_seed42.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from paper_style_datasets import load_paper_style_splits, to_lora_examples
from run_dream_w2s_baselines import score_candidates
from run_dream_paper_linear_probe import resolve_device
from run_dream_paper_style_lora import _load_causal_lm_for_scoring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="anli")
    ap.add_argument("--anli-round", default="r1")
    ap.add_argument("--sciq-use-support", action="store_true")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--answer-suffix", default="\nIs the candidate answer correct? Answer:")
    ap.add_argument("--weak-model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="")
    ap.add_argument("--acts", default="", help="acts npz to verify row alignment against")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    splits = load_paper_style_splits(args)
    examples = to_lora_examples(splits.strong_train, args.answer_suffix)
    device = resolve_device(args.device)
    model, tokenizer = _load_causal_lm_for_scoring(args.weak_model, "float32", device)
    model.eval()

    margins = np.zeros(len(examples), dtype=np.float64)
    bs = args.batch_size
    with torch.no_grad():
        for b0 in range(0, len(examples), bs):
            batch = examples[b0:b0 + bs]
            prompts = [ex.prompt for ex in batch]
            yes = score_candidates(model, tokenizer, prompts, [" yes"] * len(batch),
                                   device, args.max_length)
            no = score_candidates(model, tokenizer, prompts, [" no"] * len(batch),
                                  device, args.max_length)
            margins[b0:b0 + len(batch)] = (yes - no).double().cpu().numpy()
    if not np.isfinite(margins).all():
        raise ValueError("non-finite zero-shot margins")

    gold = np.array([ex.label for ex in examples], dtype=np.int64)
    out = {"weak_prior_margin": margins, "gt": gold}
    if args.acts:
        d = np.load(args.acts)
        out["weak_preds"] = np.asarray(d["weak_preds"]).astype(np.int64)
        match = (np.asarray(d["gt"]).astype(int) == gold).mean()
        if match < 0.95:
            raise ValueError(f"row alignment against {args.acts} failed (gt match {match:.3f})")
        print(f"alignment vs {args.acts}: gt match {match:.3f}")
    np.savez(args.out, **out)
    zero_acc = ((margins > 0).astype(int) == gold).mean()
    print(f"wrote {args.out} | zero-shot accuracy {zero_acc:.3f} | margin std {margins.std():.3f}")


if __name__ == "__main__":
    main()
