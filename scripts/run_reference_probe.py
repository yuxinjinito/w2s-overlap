#!/usr/bin/env python3
"""A closer reproduction of the reference weak-probe confidence path.

This script is intentionally narrower than the original repository: it supports
BoolQ first, but mirrors the important implementation choices from the reference
probe experiment more closely than run_probe_confidence.py:

- original-style BoolQ formatting: "Passage: ...\nQuestion: ..."
- balanced binary splits
- train split divided into weak_train and strong_train halves
- AutoModelForSequenceClassification for hidden-state extraction
- final-layer final-token activation as the feature
- L-BFGS logistic probe with L2 penalty
- confidence = 2 * abs(probe_prob(label=1) - 0.5)

The key target split is strong_train, because the reference overlap code uses the
weak probe to pseudo-label the strong_train activations before computing weak
confidence and hard/easy-or-overlap partitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["boolq"], default="boolq")
    parser.add_argument("--weak-model", default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--n-val", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--target-split", choices=["weak_train", "strong_train", "val", "test"], default="strong_train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--l2-penalty", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--activation-output", default=None)
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


def format_boolq(ds) -> pd.DataFrame:
    rows = [
        {
            "id": str(i),
            "text": f"Passage: {ex['passage']}\nQuestion: {ex['question']}",
            "label": int(ex["answer"]),
        }
        for i, ex in enumerate(ds)
    ]
    return pd.DataFrame(rows)


def balance_hf_binary(ds, seed: int):
    labels = np.array([int(x) for x in ds["answer"]])
    counts = np.bincount(labels, minlength=2)
    minority = int(np.argmin(counts))
    majority = 1 - minority
    minority_count = int(counts[minority])
    minority_ds = ds.filter(lambda ex: int(ex["answer"]) == minority)
    majority_ds = (
        ds.filter(lambda ex: int(ex["answer"]) == majority)
        .shuffle(seed=seed)
        .select(range(minority_count))
    )
    return concatenate_datasets([minority_ds, majority_ds]).shuffle(seed=seed)


def load_boolq_probe_style(n_train: int, n_val: int, n_test: int, seed: int) -> dict[str, pd.DataFrame]:
    train_raw = load_dataset("boolq", split="train").shuffle(seed=seed)
    test_raw = load_dataset("boolq", split="validation").shuffle(seed=seed)

    train_bal = balance_hf_binary(train_raw, seed)
    test_bal = balance_hf_binary(test_raw, seed)

    total_train_pool = min(n_train + n_val, len(train_bal))
    total_test = min(n_test, len(test_bal))
    train_df = format_boolq(train_bal.select(range(total_train_pool)))
    test_df = format_boolq(test_bal.select(range(total_test)))

    val_df = train_df.iloc[: min(n_val, len(train_df))].reset_index(drop=True)
    train_only = train_df.iloc[min(n_val, len(train_df)) :].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_only))
    split = len(perm) // 2
    weak_train = train_only.iloc[perm[:split]].reset_index(drop=True)
    strong_train = train_only.iloc[perm[split:]].reset_index(drop=True)

    return {
        "weak_train": weak_train,
        "strong_train": strong_train,
        "val": val_df,
        "test": test_df,
    }


def batched(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@torch.no_grad()
def extract_final_token_activations(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
    desc: str,
) -> torch.Tensor:
    acts = []
    for text_batch in tqdm(list(batched(texts, batch_size)), desc=desc):
        inputs = tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        last_indices = inputs["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden.shape[0], device=device)
        acts.append(last_hidden[batch_indices, last_indices].detach().to(torch.float32).cpu())
    return torch.cat(acts, dim=0)


class LogisticProbe(torch.nn.Module):
    def __init__(self, input_dim: int, device: torch.device):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1, device=device)
        self.linear.bias.data.zero_()
        self.linear.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    def fit(self, x: torch.Tensor, y: torch.Tensor, l2_penalty: float, max_iter: int) -> float:
        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
        )
        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        loss = torch.inf

        def closure():
            nonlocal loss
            optimizer.zero_grad()
            logits = self(x)
            loss = loss_fn(logits, y)
            reg_loss = loss + l2_penalty * self.linear.weight.square().sum()
            reg_loss.backward()
            return float(reg_loss)

        optimizer.step(closure)
        return float(loss)

    @torch.no_grad()
    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(x))


def add_predictions(
    df: pd.DataFrame,
    probs: np.ndarray,
    prediction_source: str,
    dataset: str,
    split: str,
) -> pd.DataFrame:
    out = df.copy()
    preds = (probs >= 0.5).astype(int)
    out["dataset"] = dataset
    out["split"] = split
    out["prediction_source"] = prediction_source
    out["weak_prob_label1"] = probs
    out["weak_confidence"] = binary_confidence(probs)
    out["weak_pred"] = preds
    out["weak_correct"] = (preds == out["label"].to_numpy()).astype(int)
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.torch_dtype)

    splits = load_boolq_probe_style(args.n_train, args.n_val, args.n_test, args.seed)
    weak_train_df = splits["weak_train"]
    target_df = splits[args.target_split]

    tokenizer = AutoTokenizer.from_pretrained(args.weak_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"num_labels": 2}
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForSequenceClassification.from_pretrained(args.weak_model, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    weak_train_acts = extract_final_token_activations(
        model,
        tokenizer,
        weak_train_df["text"].tolist(),
        device,
        args.batch_size,
        args.max_length,
        "extract weak_train activations",
    )
    target_acts = extract_final_token_activations(
        model,
        tokenizer,
        target_df["text"].tolist(),
        device,
        args.batch_size,
        args.max_length,
        f"extract {args.target_split} activations",
    )

    train_x = weak_train_acts.to(device)
    train_y = torch.tensor(weak_train_df["label"].to_numpy(), dtype=torch.float32, device=device)
    target_x = target_acts.to(device)

    probe = LogisticProbe(train_x.shape[1], device=device)
    final_loss = probe.fit(train_x, train_y, args.l2_penalty, args.max_iter)
    train_probs = probe.predict_prob(train_x).detach().cpu().numpy()
    target_probs = probe.predict_prob(target_x).detach().cpu().numpy()

    train_preds = (train_probs >= 0.5).astype(int)
    train_acc = float((train_preds == weak_train_df["label"].to_numpy()).mean())
    target_out = add_predictions(
        target_df,
        target_probs,
        prediction_source="weak_activation_logistic_probe",
        dataset=args.dataset,
        split=args.target_split,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_out.to_csv(output_path, index=False)

    if args.activation_output:
        act_path = Path(args.activation_output)
        act_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "weak_train_activations": weak_train_acts,
                "target_activations": target_acts,
                "weak_train_labels": torch.tensor(weak_train_df["label"].to_numpy(), dtype=torch.float32),
                "target_labels": torch.tensor(target_df["label"].to_numpy(), dtype=torch.float32),
                "weak_model": args.weak_model,
                "target_split": args.target_split,
                "n_train": args.n_train,
                "n_val": args.n_val,
                "n_test": args.n_test,
                "seed": args.seed,
            },
            act_path,
        )

    summary = {
        "dataset": args.dataset,
        "weak_model": args.weak_model,
        "target_split": args.target_split,
        "prediction_source": "weak_activation_logistic_probe",
        "n_weak_train": int(len(weak_train_df)),
        "n_target": int(len(target_out)),
        "probe_final_train_loss": final_loss,
        "probe_train_accuracy": train_acc,
        "target_accuracy": float(target_out["weak_correct"].mean()),
        "target_confidence_mean": float(target_out["weak_confidence"].mean()),
        "target_confidence_median": float(target_out["weak_confidence"].median()),
        "output": str(output_path),
        "activation_output": args.activation_output,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
