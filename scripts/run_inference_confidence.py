#!/usr/bin/env python3
"""Run weak/strong inference and save weak confidence scores.

This is the first engineering step requested on 2026-05-13.

Two modes are supported:
- seqcls: use Hugging Face sequence-classification models. This is best for a
  quick smoke test on SST-2.
- causal_lm_yesno: use causal language models and score " no" versus " yes"
  continuations. This is closer to paper-style LLM inference on BoolQ, but is
  only a first-pass proxy unless the models are fine-tuned for the task.

Confidence follows the binary version used in the reference overlap code:
    confidence = 2 * abs(P(label=1) - 0.5)
This maps uncertain predictions near 0.5 to 0 and confident predictions near
0 or 1 to 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seqcls", "causal_lm_yesno"], required=True)
    parser.add_argument("--dataset", choices=["sst2", "boolq", "sciq"], default="sst2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weak-model", required=True)
    parser.add_argument("--strong-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--positive-label-id", type=int, default=1)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or cuda device like 'cuda:0'")
    parser.add_argument("--device-map-auto", action="store_true", help="Use transformers device_map='auto' for causal LMs.")
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str):
    if dtype_arg == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_arg]


def binary_confidence(prob_label1: np.ndarray) -> np.ndarray:
    return 2.0 * np.abs(prob_label1 - 0.5)


def load_examples(dataset_name: str, split: str, limit: int, seed: int) -> list[dict]:
    if dataset_name == "sst2":
        ds = load_dataset("glue", "sst2", split=split)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return [
            {
                "id": str(ex.get("idx", i)),
                "text": ex["sentence"],
                "label": int(ex["label"]),
            }
            for i, ex in enumerate(ds)
        ]

    if dataset_name == "boolq":
        ds = load_dataset("boolq", split=split)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return [
            {
                "id": str(i),
                "text": f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:",
                "label": int(ex["answer"]),
            }
            for i, ex in enumerate(ds)
        ]

    if dataset_name == "sciq":
        rng = np.random.default_rng(seed)
        ds = load_dataset("sciq", split=split)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        examples = []
        for i, ex in enumerate(ds):
            use_correct = bool(rng.integers(0, 2))
            if use_correct:
                answer = ex["correct_answer"]
                label = 1
            else:
                answer = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]]).item()
                label = 0
            examples.append(
                {
                    "id": str(i),
                    "text": f"Q: {ex['question']}\nA: {answer}\nIs this answer correct?",
                    "label": label,
                }
            )
        return examples

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def batched(items: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def run_sequence_classifier(
    model_name: str,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    positive_label_id: int,
) -> dict[str, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    prob_label1_parts = []
    pred_parts = []
    max_prob_parts = []
    for text_batch in tqdm(list(batched(texts, batch_size)), desc=f"seqcls {model_name}"):
        inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        prob_label1 = probs[:, positive_label_id]
        pred = probs.argmax(dim=-1)
        prob_label1_parts.append(prob_label1.detach().cpu().numpy())
        pred_parts.append((pred == positive_label_id).long().detach().cpu().numpy())
        max_prob_parts.append(probs.max(dim=-1).values.detach().cpu().numpy())

    prob_label1_all = np.concatenate(prob_label1_parts)
    return {
        "prob_label1": prob_label1_all,
        "pred": np.concatenate(pred_parts).astype(int),
        "max_prob": np.concatenate(max_prob_parts),
        "confidence": binary_confidence(prob_label1_all),
    }


def score_candidate_logprob(model, tokenizer, prompt: str, candidate: str, device: torch.device) -> float:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    full_ids = tokenizer(prompt + candidate, return_tensors="pt", add_special_tokens=True).input_ids
    prompt_len = prompt_ids.shape[-1]

    if full_ids.shape[-1] <= prompt_len:
        raise ValueError("Candidate produced no additional tokens.")

    full_ids = full_ids.to(device)
    with torch.no_grad():
        logits = model(full_ids).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    positions = torch.arange(shift_labels.shape[-1], device=device) + 1
    candidate_mask = positions >= prompt_len
    return token_log_probs[0, candidate_mask].sum().detach().cpu().item()


def run_causal_lm_yesno(
    model_name: str,
    texts: list[str],
    device: torch.device,
    dtype,
    device_map_auto: bool,
) -> dict[str, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
    if device_map_auto:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not device_map_auto:
        model.to(device)
    model.eval()

    prob_label1 = []
    for prompt in tqdm(texts, desc=f"causal_lm_yesno {model_name}"):
        no_score = score_candidate_logprob(model, tokenizer, prompt, " no", device)
        yes_score = score_candidate_logprob(model, tokenizer, prompt, " yes", device)
        probs = torch.softmax(torch.tensor([no_score, yes_score]), dim=0).numpy()
        prob_label1.append(float(probs[1]))

    prob_label1_all = np.array(prob_label1)
    pred = (prob_label1_all >= 0.5).astype(int)
    return {
        "prob_label1": prob_label1_all,
        "pred": pred,
        "max_prob": np.maximum(prob_label1_all, 1.0 - prob_label1_all),
        "confidence": binary_confidence(prob_label1_all),
    }


def add_model_outputs(rows: list[dict], prefix: str, outputs: dict[str, np.ndarray]) -> None:
    for i, row in enumerate(rows):
        label = row.get("label")
        pred = int(outputs["pred"][i])
        row[f"{prefix}_prob_label1"] = float(outputs["prob_label1"][i])
        row[f"{prefix}_max_prob"] = float(outputs["max_prob"][i])
        row[f"{prefix}_confidence"] = float(outputs["confidence"][i])
        row[f"{prefix}_pred"] = pred
        row[f"{prefix}_correct"] = None if label is None else int(pred == int(label))


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.torch_dtype)

    if args.mode == "seqcls" and args.dataset != "sst2":
        raise ValueError("seqcls mode is currently intended for the SST-2 smoke test.")

    examples = load_examples(args.dataset, args.split, args.limit, args.seed)
    texts = [ex["text"] for ex in examples]
    rows = [
        {
            "id": ex["id"],
            "dataset": args.dataset,
            "split": args.split,
            "text": ex["text"],
            "label": ex.get("label"),
        }
        for ex in examples
    ]

    if args.mode == "seqcls":
        weak_outputs = run_sequence_classifier(
            args.weak_model, texts, device, args.batch_size, args.positive_label_id
        )
        strong_outputs = run_sequence_classifier(
            args.strong_model, texts, device, args.batch_size, args.positive_label_id
        )
    else:
        weak_outputs = run_causal_lm_yesno(
            args.weak_model, texts, device, dtype, args.device_map_auto
        )
        strong_outputs = run_causal_lm_yesno(
            args.strong_model, texts, device, dtype, args.device_map_auto
        )

    add_model_outputs(rows, "weak", weak_outputs)
    add_model_outputs(rows, "strong", strong_outputs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if output_path.suffix == ".jsonl":
        df.to_json(output_path, orient="records", lines=True)
    else:
        df.to_csv(output_path, index=False)

    summary = {
        "n_examples": len(df),
        "weak_confidence_mean": float(df["weak_confidence"].mean()),
        "weak_confidence_median": float(df["weak_confidence"].median()),
        "weak_accuracy": None if df["weak_correct"].isna().all() else float(df["weak_correct"].mean()),
        "strong_accuracy": None if df["strong_correct"].isna().all() else float(df["strong_correct"].mean()),
        "output": str(output_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
