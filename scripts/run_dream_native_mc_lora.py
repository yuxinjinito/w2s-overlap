#!/usr/bin/env python3
"""Native multiple-choice weak-to-strong pipeline (DREAM).

Generative causal-LM path: base model + per-option log-likelihood (John's
2026-06-09 eval decision), extended to a full RETRAIN so the per-option
likelihood is measured on a model actually trained for it.

Chain per seed:
  1. weak model scores each option's log-likelihood -> weak-chosen option = weak label
     (+ per-option softmax confidence). Optionally fine-tune the weak model on a
     small gold split first (paper-faithful W2S; --weak-train-steps>0).
  2. a selection method filters the train questions.
  3. LoRA-SFT the strong model toward the (weak-chosen, or gold for the upper
     bound) option -- LM loss on the option tokens only (context masked).
  4. evaluate natively: per-option log-likelihood -> argmax -> accuracy
     (raw sum + length-normalized) + one-vs-rest AUROC.

Bounds reported: weak (lower) / base-strong zero-shot / w2s-no-filter / each
method / strong-GT (upper).

Smoke (1 seed, ~30 min on a 4090):
  PYTHONPATH=scripts python scripts/run_dream_native_mc_lora.py \
    --output-sub dream_native_smoke_0614 --n-train 800 --n-test 300 \
    --max-train-steps 100 --seed 42 \
    --methods weak_label,random_balanced,confidence_high,ground_truth

kNN / excess-loss selectors are deferred to the full sweep (see design note).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_dream_paper_linear_probe import iter_dream_question_rows, load_dream_raw, flatten_text
from run_dream_w2s_baselines import (
    Collator,
    clear_memory,
    count_params,
    cuda_memory_report,
    parse_lora_target_modules,
    require_peft,
    resolve_dtype,
    score_candidates,
)

SMOKE_METHODS = ("weak_label", "ground_truth", "random_balanced", "confidence_high", "confidence_low")


@dataclass
class NativeQ:
    id: str
    prompt: str  # context ending in "A:"
    options: list[str]  # raw option texts
    gold: int  # gold option index
    weak_pred: int = -1
    weak_conf: float = 0.0  # max softmax prob over options
    weak_margin: float = 0.0  # top1 - top2 prob


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--strong-model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--weak-model", default="Qwen/Qwen1.5-0.5B")
    p.add_argument("--output-sub", required=True, help="results/<output-sub>")
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", default="", help="comma-separated TRAIN seeds for multi-seed runs (dual-eval); empty = use --seed")
    p.add_argument("--methods", default="weak_label,random_balanced,confidence_high,ground_truth")
    p.add_argument("--keep-frac", type=float, default=0.5)
    p.add_argument("--scoring", choices=["norm", "raw"], default="norm",
                   help="argmax basis for native eval headline + confidence")
    p.add_argument("--weak-scoring", choices=["norm", "raw"], default="norm")
    p.add_argument("--weak-train-steps", type=int, default=0,
                   help=">0 fine-tunes the weak model on a small gold split first (paper-faithful)")
    p.add_argument("--weak-train-n", type=int, default=500)
    # training (defaults match the binary LoRA pipeline)
    p.add_argument("--max-train-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--strong-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_dream_native(split: str, n: int, seed: int) -> list[NativeQ]:
    rng = random.Random(seed)
    qs: list[NativeQ] = []
    for ex in iter_dream_question_rows(load_dream_raw(split)):
        ch = list(ex["choice"])
        if ex["answer"] not in ch:
            continue
        joined = "\n".join(ex["dialogue"]) if isinstance(ex["dialogue"], list) else flatten_text(ex["dialogue"])
        qs.append(
            NativeQ(
                id=str(ex.get("id", len(qs))),
                prompt=f"{joined}\n\nQ: {ex['question']}\nA:",
                options=ch,
                gold=ch.index(ex["answer"]),
            )
        )
    rng.shuffle(qs)
    return qs[:n]


def load_causal_lm(model_name: str, args: argparse.Namespace, trainable_lora: bool):
    dtype = resolve_dtype(args.torch_dtype)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    if trainable_lora:
        LoraConfig, TaskType, get_peft_model = require_peft()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=parse_lora_target_modules(args.lora_target_modules),
        )
        model = get_peft_model(model, lora)
    model.to(args.device)
    return model, tok


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def _auroc(scores: list[float], labels: list[int]) -> float:
    """One-vs-rest rank AUROC (Mann-Whitney) over flattened (question, option) pairs."""
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = 0.0
    i = 0
    rank = 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum_pos += avg_rank
        rank += j - i
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@torch.no_grad()
def _option_scores(model, tokenizer, q: NativeQ, device: str, max_length: int) -> tuple[list[float], list[int]]:
    opts = [" " + o for o in q.options]
    prompts = [q.prompt] * len(opts)
    sums = score_candidates(model, tokenizer, prompts, opts, device, max_length).tolist()
    ntok = [max(len(tokenizer(o, add_special_tokens=False)["input_ids"]), 1) for o in opts]
    return sums, ntok


@torch.no_grad()
def label_weak(model, tokenizer, questions: list[NativeQ], device: str, max_length: int, scoring: str) -> None:
    model.eval()
    for q in tqdm(questions, desc="weak label"):
        sums, ntok = _option_scores(model, tokenizer, q, device, max_length)
        score = sums if scoring == "raw" else [s / n for s, n in zip(sums, ntok)]
        probs = _softmax(score)
        q.weak_pred = int(np.argmax(score))
        sp = sorted(probs, reverse=True)
        q.weak_conf = float(sp[0])
        q.weak_margin = float(sp[0] - sp[1]) if len(sp) > 1 else float(sp[0])


@torch.no_grad()
def evaluate_native(model, tokenizer, questions: list[NativeQ], device: str, max_length: int, desc: str):
    model.eval()
    raw_ok = norm_ok = 0
    auroc_scores: list[float] = []
    auroc_labels: list[int] = []
    rows = []
    for q in tqdm(questions, desc=desc):
        sums, ntok = _option_scores(model, tokenizer, q, device, max_length)
        norm = [s / n for s, n in zip(sums, ntok)]
        raw_pred = int(np.argmax(sums))
        norm_pred = int(np.argmax(norm))
        raw_ok += int(raw_pred == q.gold)
        norm_ok += int(norm_pred == q.gold)
        probs = _softmax(norm)
        for oi, pr in enumerate(probs):
            auroc_scores.append(pr)
            auroc_labels.append(int(oi == q.gold))
        rows.append({"id": q.id, "gold": q.gold, "pred_raw": raw_pred, "pred_norm": norm_pred})
    n = len(questions)
    summary = {
        "n": n,
        "accuracy_raw": raw_ok / n if n else math.nan,
        "accuracy_norm": norm_ok / n if n else math.nan,
        "auroc": _auroc(auroc_scores, auroc_labels),
    }
    return summary, rows


def select_subset(questions: list[NativeQ], method: str, keep_frac: float, seed: int) -> list[NativeQ]:
    if method in ("weak_label", "ground_truth"):
        return list(questions)  # no filter; target differs (gold vs weak) at train time
    k = max(1, int(round(keep_frac * len(questions))))
    if method == "random_balanced":
        qs = list(questions)
        random.Random(seed).shuffle(qs)
        return qs[:k]
    if method == "confidence_high":
        return sorted(questions, key=lambda q: q.weak_conf, reverse=True)[:k]
    if method == "confidence_low":
        return sorted(questions, key=lambda q: q.weak_conf)[:k]
    raise NotImplementedError(f"selector {method!r} not in smoke set {SMOKE_METHODS} (kNN/excess-loss deferred)")


def build_pairs(subset: list[NativeQ], method: str) -> list[tuple[str, str]]:
    pairs = []
    for q in subset:
        idx = q.gold if method == "ground_truth" else q.weak_pred
        pairs.append((q.prompt, " " + q.options[idx]))
    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> tuple[str, str]:
        return self.pairs[i]


def train_native(model_name: str, pairs: list[tuple[str, str]], args: argparse.Namespace, run_name: str):
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, tok = load_causal_lm(model_name, args, trainable_lora=True)
    model.train()
    total_params, trainable_params = count_params(model)
    loader = DataLoader(
        PairDataset(pairs),
        batch_size=args.strong_batch_size,
        shuffle=True,
        collate_fn=Collator(tok, args.max_length),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    data_iter = cycle(loader)
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for _ in tqdm(range(args.max_train_steps), desc=f"train {run_name}"):
        step_losses = []
        for _ in range(args.gradient_accumulation_steps):
            batch = next(data_iter)
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.gradient_accumulation_steps
            loss.backward()
            step_losses.append(float(loss.detach().cpu().item()) * args.gradient_accumulation_steps)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(sum(step_losses) / len(step_losses))
    report = {
        "run_name": run_name,
        "n_train": len(pairs),
        "max_train_steps": args.max_train_steps,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "losses": losses,
        "elapsed_sec": time.time() - start,
        "cuda_memory": cuda_memory_report(),
    }
    return model, tok, report


def main() -> None:
    args = parse_args()
    out = Path("results") / args.output_sub
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_qs = load_dream_native("train", args.n_train, args.seed)
    test_qs = load_dream_native("test", args.n_test, args.seed + 1)
    print(f"loaded train={len(train_qs)} test={len(test_qs)} (DREAM native, K={len(train_qs[0].options)})")

    # 1. weak model: optional SFT on a small gold split, then label the train pool + eval test (lower bound)
    weak_model, weak_tok = load_causal_lm(args.weak_model, args, trainable_lora=args.weak_train_steps > 0)
    if args.weak_train_steps > 0:
        del weak_model
        clear_memory()
        gold_subset = train_qs[: args.weak_train_n]
        weak_model, weak_tok, _ = train_native(
            args.weak_model, build_pairs(gold_subset, "ground_truth"), args, "weak_sft"
        )
    label_weak(weak_model, weak_tok, train_qs, args.device, args.max_length, args.weak_scoring)
    weak_eval, _ = evaluate_native(weak_model, weak_tok, test_qs, args.device, args.max_length, "weak eval")
    weak_train_acc = float(np.mean([q.weak_pred == q.gold for q in train_qs]))
    label_dist = {str(i): int(sum(q.weak_pred == i for q in train_qs)) for i in range(len(train_qs[0].options))}
    del weak_model
    clear_memory()

    # 2. base strong zero-shot (untrained reference)
    base_model, base_tok = load_causal_lm(args.strong_model, args, trainable_lora=False)
    base_eval, _ = evaluate_native(base_model, base_tok, test_qs, args.device, args.max_length, "base strong eval")
    del base_model
    clear_memory()

    # 3. each method: select -> train strong toward (weak or gold) option -> eval
    runs = {}
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        subset = select_subset(train_qs, method, args.keep_frac, args.seed)
        pairs = build_pairs(subset, method)
        model, tok, rep = train_native(args.strong_model, pairs, args, method)
        ev, _ = evaluate_native(model, tok, test_qs, args.device, args.max_length, f"eval {method}")
        rep.update(ev)
        runs[method] = rep
        del model
        clear_memory()

    summary = {
        "dataset": "dream_native",
        "config": vars(args),
        "weak_train_accuracy": weak_train_acc,
        "weak_label_distribution": label_dist,
        "bounds": {"weak": weak_eval, "base_strong": base_eval},
        "runs": runs,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # console bounds table
    print("\n=== DREAM native bounds (accuracy_norm / accuracy_raw / auroc) ===")

    def fmt(name, d):
        print(f"{name:22} norm={d['accuracy_norm']:.3f}  raw={d['accuracy_raw']:.3f}  auroc={d['auroc']:.3f}")

    fmt("weak (lower)", weak_eval)
    fmt("base-strong 0shot", base_eval)
    for m, rep in runs.items():
        fmt(m, rep)
    print(f"\nweak train acc={weak_train_acc:.3f}  weak label dist={label_dist}")
    print(f"summary -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
