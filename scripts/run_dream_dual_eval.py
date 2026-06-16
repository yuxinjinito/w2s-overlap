#!/usr/bin/env python3
"""DREAM dual-eval: score the SAME models two ways to close the eval-format question.

Trains the strong model with the BINARY discrimination objective (question+option ->
" yes"/" no" -- the existing W2S setup we did NOT change), then evaluates
each trained model BOTH ways on the test questions:

  - DISCRIMINATION (accuracy_3class): per question, pick the option with the highest
    P(correct) from the yes/no head. [what the 2026-06-10 results used]
  - LIKELIHOOD (the literal target): per question, pick the option with the highest
    per-option TEXT log-likelihood. [base model + per-option log-likelihood]

Goal: show whether the literal likelihood eval captures the (binary) W2S training
signal. Expectation: discrimination scoring separates the methods (base < w2s < GT);
likelihood scoring collapses below base for binary-trained models, because the binary
LoRA reshapes the whole next-token distribution, not just a yes/no head.

Methods: base (untrained) / weak_label / random_balanced / confidence_high / ground_truth.

Single seed (~25-35 min): --seed 42
Multi-seed (data fixed at --seed, TRAIN seed varies, matches the bounds methodology):
  PYTHONPATH=scripts python scripts/run_dream_dual_eval.py \
    --output-sub dream_dual_eval_ms_0614 --n-train 800 --n-test 300 --max-train-steps 100 \
    --seed 42 --seeds 42,123,456,7,31
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from run_dream_w2s_baselines import Example, clear_memory, score_candidates, train_lora_model
from run_dream_native_mc_lora import (
    _auroc,
    evaluate_native,
    label_weak,
    load_causal_lm,
    load_dream_native,
    parse_args,
    select_subset,
)

EVAL_KEYS = ("discrimination_acc", "discrimination_auroc",
             "likelihood_acc_norm", "likelihood_acc_raw", "likelihood_auroc")


def _binary_ctx(q) -> str:
    """Reconstruct the paper-style binary candidate context from the native prompt."""
    return q.prompt[:-3] if q.prompt.endswith("\nA:") else q.prompt


def binary_examples(questions, use_gold: bool):
    exs: list[Example] = []
    labels: list[int] = []
    for q in questions:
        tgt = q.gold if use_gold else q.weak_pred
        ctx = _binary_ctx(q)
        for i, opt in enumerate(q.options):
            exs.append(Example(id=f"{q.id}-{i}", text=f"{ctx} A: {opt}", label=int(i == tgt)))
            labels.append(int(i == tgt))
    return exs, labels


@torch.no_grad()
def eval_discrimination(model, tokenizer, questions, device, max_length):
    """Per question, pick the option with the highest P(correct) from the yes/no head."""
    model.eval()
    ok = 0
    auroc_scores: list[float] = []
    auroc_labels: list[int] = []
    for q in tqdm(questions, desc="discrimination eval"):
        ctx = _binary_ctx(q)
        prompts = [f"{ctx} A: {opt}\nAnswer:" for opt in q.options]
        k = len(q.options)
        yes = score_candidates(model, tokenizer, prompts, [" yes"] * k, device, max_length)
        no = score_candidates(model, tokenizer, prompts, [" no"] * k, device, max_length)
        p_yes = torch.softmax(torch.stack([no, yes], dim=1), dim=1)[:, 1].tolist()
        pred = int(np.argmax(p_yes))
        ok += int(pred == q.gold)
        for i, p in enumerate(p_yes):
            auroc_scores.append(p)
            auroc_labels.append(int(i == q.gold))
    n = len(questions)
    return {"accuracy": ok / n if n else float("nan"), "auroc": _auroc(auroc_scores, auroc_labels)}


def main() -> None:
    args = parse_args()
    args.save_adapters = False
    out = Path("results") / args.output_sub
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [args.seed]

    # Data + weak labels + base eval are computed ONCE (data seed fixed at --seed);
    # only the TRAIN seed varies across `seeds`, matching the bounds-table methodology.
    train_qs = load_dream_native("train", args.n_train, args.seed)
    test_qs = load_dream_native("test", args.n_test, args.seed + 1)
    k = len(train_qs[0].options)
    print(f"loaded train={len(train_qs)} test={len(test_qs)} (DREAM, K={k}) seeds={seeds}")

    weak_model, weak_tok = load_causal_lm(args.weak_model, args, trainable_lora=False)
    label_weak(weak_model, weak_tok, train_qs, args.device, args.max_length, args.weak_scoring)
    weak_train_acc = float(np.mean([q.weak_pred == q.gold for q in train_qs]))
    del weak_model
    clear_memory()
    print(f"weak train acc={weak_train_acc:.3f}")

    def dual_eval(model, tok):
        disc = eval_discrimination(model, tok, test_qs, args.device, args.max_length)
        lik, _ = evaluate_native(model, tok, test_qs, args.device, args.max_length, "likelihood eval")
        return {
            "discrimination_acc": disc["accuracy"],
            "discrimination_auroc": disc["auroc"],
            "likelihood_acc_norm": lik["accuracy_norm"],
            "likelihood_acc_raw": lik["accuracy_raw"],
            "likelihood_auroc": lik["auroc"],
        }

    # base (untrained) strong model -- no seed variance, evaluated once
    base_model, base_tok = load_causal_lm(args.strong_model, args, trainable_lora=False)
    base_row = dual_eval(base_model, base_tok)
    del base_model
    clear_memory()

    methods = [m.strip() for m in args.methods.split(",") if m.strip() and m.strip() != "base"]
    collected = {m: {kk: [] for kk in EVAL_KEYS} for m in methods}
    per_seed = []
    for sd in seeds:
        random.seed(sd)
        np.random.seed(sd)
        torch.manual_seed(sd)
        for method in methods:
            subset = select_subset(train_qs, method, args.keep_frac, sd)
            exs, labels = binary_examples(subset, use_gold=(method == "ground_truth"))
            model, tok, _ = train_lora_model(args, exs, labels, f"{method}_s{sd}", out)
            ev = dual_eval(model, tok)
            for kk in EVAL_KEYS:
                collected[method][kk].append(ev[kk])
            per_seed.append({"seed": sd, "method": method, **ev})
            del model
            clear_memory()

    def agg(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    method_summary = {m: {kk: agg(v) for kk, v in collected[m].items()} for m in methods}
    summary = {
        "dataset": "dream_dual_eval_multiseed",
        "config": {kk: vv for kk, vv in vars(args).items() if kk != "device"},
        "seeds": seeds,
        "weak_train_accuracy": weak_train_acc,
        "base": base_row,
        "methods": method_summary,
        "per_seed": per_seed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== DREAM dual eval (mean ± std over {len(seeds)} train seeds) ===")
    print(f"{'model':16} {'DISC acc':>14} {'LIK acc(norm)':>16}")
    print(f"{'base':16} {base_row['discrimination_acc']:>14.3f} {base_row['likelihood_acc_norm']:>16.3f}")
    for m in methods:
        d = method_summary[m]["discrimination_acc"]
        l = method_summary[m]["likelihood_acc_norm"]
        print(f"{m:16} {d['mean']:>8.3f}±{d['std']:.3f} {l['mean']:>9.3f}±{l['std']:.3f}")
    print(f"\nweak train acc={weak_train_acc:.3f}  (DISC=accuracy_3class / LIK=per-option likelihood)")
    print(f"summary -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
