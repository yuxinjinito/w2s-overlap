#!/usr/bin/env python3
"""DREAM dual-eval: score the SAME models two ways to close John's eval-format question.

Trains the strong model with the BINARY discrimination objective (question+option ->
" yes"/" no" -- the existing W2S setup John did NOT ask to change), then evaluates
each trained model BOTH ways on the test questions:

  - DISCRIMINATION (accuracy_3class): per question, pick the option with the highest
    P(correct) from the yes/no head. [what the 2026-06-10 results used]
  - LIKELIHOOD (John's literal ask): per question, pick the option with the highest
    per-option TEXT log-likelihood. [base model + per-option log-likelihood]

Goal: show whether the literal likelihood eval captures the (binary) W2S training
signal. Expectation: discrimination scoring separates the methods (base < w2s < GT);
likelihood scoring is ~flat at base for all binary-trained models, because the binary
LoRA changes the yes/no head, not the option-text likelihood. If so, the discrimination
score is the meaningful instrument and the literal likelihood eval is blind to it.

Methods: base (untrained) / weak_label / random_balanced / confidence_high / ground_truth.
~25-35 min, 1 seed, on a 4090.

  PYTHONPATH=scripts python scripts/run_dream_dual_eval.py \
    --output-sub dream_dual_eval_0614 --n-train 800 --n-test 300 --max-train-steps 100 --seed 42
"""
from __future__ import annotations

import json
import time
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

    train_qs = load_dream_native("train", args.n_train, args.seed)
    test_qs = load_dream_native("test", args.n_test, args.seed + 1)
    k = len(train_qs[0].options)
    print(f"loaded train={len(train_qs)} test={len(test_qs)} (DREAM, K={k})")

    # weak labels (zero-shot Qwen, per-option likelihood -- same as the native smoke)
    weak_model, weak_tok = load_causal_lm(args.weak_model, args, trainable_lora=False)
    label_weak(weak_model, weak_tok, train_qs, args.device, args.max_length, args.weak_scoring)
    weak_train_acc = float(np.mean([q.weak_pred == q.gold for q in train_qs]))
    del weak_model
    clear_memory()
    print(f"weak train acc={weak_train_acc:.3f}")

    rows = {}

    def dual_eval(model, tok, tag):
        disc = eval_discrimination(model, tok, test_qs, args.device, args.max_length)
        lik, _ = evaluate_native(model, tok, test_qs, args.device, args.max_length, f"likelihood {tag}")
        return {
            "discrimination_acc": disc["accuracy"],
            "discrimination_auroc": disc["auroc"],
            "likelihood_acc_norm": lik["accuracy_norm"],
            "likelihood_acc_raw": lik["accuracy_raw"],
            "likelihood_auroc": lik["auroc"],
        }

    # base (untrained) strong model, both scorings
    base_model, base_tok = load_causal_lm(args.strong_model, args, trainable_lora=False)
    rows["base"] = dual_eval(base_model, base_tok, "base")
    del base_model
    clear_memory()

    # each method: BINARY-train the strong model, then both scorings
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        if method == "base":
            continue
        subset = select_subset(train_qs, method, args.keep_frac, args.seed)
        exs, labels = binary_examples(subset, use_gold=(method == "ground_truth"))
        model, tok, rep = train_lora_model(args, exs, labels, method, out)
        ev = dual_eval(model, tok, method)
        ev["n_train_questions"] = len(subset)
        ev["elapsed_sec"] = rep["elapsed_sec"]
        rows[method] = ev
        del model
        clear_memory()

    summary = {
        "dataset": "dream_dual_eval",
        "config": {k2: v for k2, v in vars(args).items() if k2 != "device"},
        "weak_train_accuracy": weak_train_acc,
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== DREAM dual eval (same models, two scorings) ===")
    print(f"{'model':18} {'DISC acc':>9} {'DISC auroc':>11} | {'LIK acc':>8} {'LIK auroc':>10}")
    for name, r in rows.items():
        print(
            f"{name:18} {r['discrimination_acc']:>9.3f} {r['discrimination_auroc']:>11.3f} | "
            f"{r['likelihood_acc_norm']:>8.3f} {r['likelihood_auroc']:>10.3f}"
        )
    print(f"\nweak train acc={weak_train_acc:.3f}  (DISC=accuracy_3class / LIK=John's per-option likelihood)")
    print(f"summary -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
