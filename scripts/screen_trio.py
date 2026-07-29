#!/usr/bin/env python3
"""Stage-1.5 "trio" screen for W2S testbed candidates.

One GPU pass per candidate produces FOUR numbers and an automatic verdict, catching all
four known failure modes BEFORE the expensive Stage-2 pipeline run:

  1. strong-base DISCRIM MC acc (Qwen2.5-7B, yes/no P(correct))  -> base-too-strong
  2. strong-base NATIVE MC acc (length-normalised likelihood)    -> recall (native >> discrim)
  3. weak-probe heldout acc (Qwen2.5-0.5B last-token acts + logistic probe on the
     candidate-correctness task, gold labels)                    -> weak-noise (~chance)
                                                                 -> teacher-ceiling (too high)
  4. sizes/counts for the record.

Generic 3-way NLI candidates only (premise/hypothesis-style fields, 3 labels), matching the
settled pipeline prompt. Verdict thresholds are calibrated on the 15 previously screened
candidates (ReClor weak-noise, ConjNLI teacher-ceiling, HellaSwag/LingNLI recall, commonsense
base-too-strong all correctly rejected in backtest; ANLI/WANLI/ConTRoL/FEVER pass).

Usage (GPU box):
  PYTHONPATH=scripts python3 scripts/screen_trio.py --hf-id tasksource/LogicNLI \
      --premise-key premise --hypothesis-key hypothesis --label-key label
"""
from __future__ import annotations

import argparse
import itertools
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_dream_w2s_baselines import resolve_dtype, score_candidates

ANLI_RELATIONS = ["entailment", "neutral", "contradiction"]
REL2IDX = {"entailment": 0, "neutral": 1, "contradiction": 2,
           "0": 0, "1": 1, "2": 2}
SUFFIX = "\nIs the candidate answer correct? Answer:"


def load_rows(hf_id, premise_key, hypothesis_key, label_key, n, seed, config=None):
    rows = []
    for split in ("train", "validation", "test", "dev"):
        try:
            ds = load_dataset(hf_id, config, split=split, streaming=True) if config else \
                load_dataset(hf_id, split=split, streaming=True)
            for ex in itertools.islice(ds, n * 4):
                prem = str(ex.get(premise_key, "")).strip()
                hyp = str(ex.get(hypothesis_key, "")).strip()
                gold = REL2IDX.get(str(ex.get(label_key, "")).strip().lower())
                if gold is None or not prem or not hyp:
                    continue
                rows.append((mc_ctx(prem, hyp), list(ANLI_RELATIONS), gold))
            if rows:
                break
        except Exception:
            continue
    random.Random(seed).shuffle(rows)
    return rows[:n]


def mc_ctx(prem, hyp):
    return (f"Premise: {prem}\nHypothesis: {hyp}\n"
            "Q: What is the relationship from the premise to the hypothesis? A:")


@torch.no_grad()
def base_native_discrim(model, tok, rows, device, max_length):
    nat = dis = 0
    for ctx, opts, gold in rows:
        k = len(opts)
        sc = score_candidates(model, tok, [ctx] * k, [" " + o for o in opts], device, max_length)
        norm = [s / max(1, len(tok(" " + o, add_special_tokens=False)["input_ids"]))
                for o, s in zip(opts, sc.tolist())]
        nat += int(int(np.argmax(norm)) == gold)
        prompts = [f"{ctx} {o}{SUFFIX}" for o in opts]
        yes = score_candidates(model, tok, prompts, [" yes"] * k, device, max_length)
        no = score_candidates(model, tok, prompts, [" no"] * k, device, max_length)
        p = torch.softmax(torch.stack([no, yes], dim=1), dim=1)[:, 1]
        dis += int(int(torch.argmax(p)) == gold)
    return nat / len(rows), dis / len(rows)


@torch.no_grad()
def weak_probe_acc(weak_id, rows, device, max_length, seed):
    """Candidate-correctness probe on 0.5B last-token acts: train/heldout split, gold labels."""
    tok = AutoTokenizer.from_pretrained(weak_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(weak_id, torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True).to(device).eval()
    rng = random.Random(seed)
    txts, labs = [], []
    for ctx, opts, gold in rows:
        corr = rng.random() < 0.5
        cand = opts[gold] if corr else opts[rng.choice([k for k in range(len(opts)) if k != gold])]
        txts.append(f"{ctx} {cand}")
        labs.append(int(corr))
    acts = []
    for b0 in range(0, len(txts), 16):
        batch = tok(txts[b0:b0 + 16], return_tensors="pt", padding=True, truncation=True,
                    max_length=max_length).to(device)
        h = model(**batch, output_hidden_states=True).hidden_states[-1]
        idx = batch["attention_mask"].sum(dim=1) - 1
        acts.append(h[torch.arange(h.shape[0]), idx].float().cpu())
    from run_dream_paper_linear_probe import fit_probe, predict_probe

    X = torch.cat(acts)
    y = torch.tensor(labs, dtype=torch.float32)
    n = len(labs)
    n_tr = int(0.7 * n)
    dev_t = torch.device(device)
    probe = fit_probe(X[:n_tr], y[:n_tr], l2_penalty=1e-3, max_iter=500, device=dev_t)
    preds = (predict_probe(probe, X[n_tr:], dev_t) >= 0.5).astype(int)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return float((preds == np.asarray(labs[n_tr:])).mean())


def verdict(native, discrim, probe, chance=1.0 / 3.0):
    notes = []
    if native - discrim >= 0.15:
        return "REJECT: recall-confounded (native %.3f >> discrim %.3f)" % (native, discrim)
    if discrim >= chance + 0.22:
        return "REJECT: base-too-strong (discrim %.3f vs chance %.3f)" % (discrim, chance)
    if discrim >= chance + 0.12:
        notes.append("moderate base %.3f (WANLI-band; needs large GT headroom)" % discrim)
    if probe < 0.54:
        return "REJECT: weak-noise (probe %.3f ~ chance 0.5, ReClor mode)" % probe
    if probe > 0.82:
        notes.append("teacher-ceiling risk (probe %.3f, ConjNLI mode)" % probe)
    tag = "PASS -> Stage-2" if not notes else "PASS (CAUTION) -> Stage-2: " + "; ".join(notes)
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default=None)
    ap.add_argument("--from-registry", default=None,
                    help="load rows via base_native_accuracy.CANDIDATE_LOADERS[name] instead of --hf-id")
    ap.add_argument("--config", default=None)
    ap.add_argument("--premise-key", default="premise")
    ap.add_argument("--hypothesis-key", default="hypothesis")
    ap.add_argument("--label-key", default="label")
    ap.add_argument("--strong-model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--weak-model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n-mc", type=int, default=400)
    ap.add_argument("--n-probe", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.from_registry:
        from base_native_accuracy import CANDIDATE_LOADERS

        rows = CANDIDATE_LOADERS[args.from_registry](max(args.n_mc, args.n_probe), args.seed)
        name = args.from_registry
    else:
        rows = load_rows(args.hf_id, args.premise_key, args.hypothesis_key, args.label_key,
                         max(args.n_mc, args.n_probe), args.seed, args.config)
        name = args.hf_id
    print(f"{name}: {len(rows)} usable rows")
    if len(rows) < 300:
        print("VERDICT REJECT: too few usable rows")
        return

    probe = weak_probe_acc(args.weak_model, rows[: args.n_probe], args.device, args.max_length, args.seed)
    print(f"weak-probe (0.5B, candidate-correctness, gold) heldout acc: {probe:.3f}")

    tok = AutoTokenizer.from_pretrained(args.strong_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.strong_model, torch_dtype=resolve_dtype("float16"), low_cpu_mem_usage=True
    ).to(args.device).eval()
    native, discrim = base_native_discrim(model, tok, rows[: args.n_mc], args.device, args.max_length)
    chance = 1.0 / float(np.mean([len(r[1]) for r in rows[: args.n_mc]]))
    print(f"strong-base NATIVE {native:.3f}   DISCRIM {discrim:.3f}   (chance {chance:.3f})")
    print("VERDICT", verdict(native, discrim, probe, chance))


if __name__ == "__main__":
    main()
