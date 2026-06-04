#!/usr/bin/env python3
"""Map weak-model representations to strong-model representations.

This script implements the first 2026-05-15 representation-similarity task:

- extract pre-lm-head hidden states for the same prompts from weak and strong
  causal LMs,
- aggregate token hidden states into one prompt vector,
- fit simple weak-to-strong maps,
- report per-sample mapping losses.

The default is intentionally small enough for a quick GPU smoke test.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["boolq", "sciq", "paws", "dream", "twitter-sentiment"],
        default="boolq",
    )
    parser.add_argument("--weak-model", default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--strong-model", default="Qwen/Qwen1.5-1.8B")
    parser.add_argument("--target-split", choices=["weak_train", "strong_train", "val", "test"], default="strong_train")
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--n-val", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling", choices=["mean", "final_token"], default="mean")
    parser.add_argument("--map-train-frac", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--relu-hidden-dim", type=int, default=256)
    parser.add_argument("--relu-epochs", type=int, default=200)
    parser.add_argument("--relu-lr", type=float, default=1e-3)
    parser.add_argument("--relu-weight-decay", type=float, default=1e-4)
    parser.add_argument("--relu-batch-size", type=int, default=64)
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--embedding-output", default=None)
    parser.add_argument(
        "--map-output",
        default=None,
        help="Optional .pt file for learned maps, predictions, and residuals.",
    )
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


def balance_hf_label(ds, label_col: str, seed: int):
    labels = np.array([int(x) for x in ds[label_col]])
    counts = np.bincount(labels, minlength=2)
    minority = int(np.argmin(counts))
    majority = 1 - minority
    minority_count = int(counts[minority])
    minority_ds = ds.filter(lambda ex: int(ex[label_col]) == minority)
    majority_ds = (
        ds.filter(lambda ex: int(ex[label_col]) == majority)
        .shuffle(seed=seed)
        .select(range(minority_count))
    )
    return concatenate_datasets([minority_ds, majority_ds]).shuffle(seed=seed)


def maybe_balance_rows(rows: list[dict], seed: int) -> list[dict]:
    by_label = {
        0: [row for row in rows if int(row["label"]) == 0],
        1: [row for row in rows if int(row["label"]) == 1],
    }
    if not by_label[0] or not by_label[1]:
        return rows
    rng = np.random.default_rng(seed)
    n = min(len(by_label[0]), len(by_label[1]))
    balanced = []
    for label in [0, 1]:
        idx = rng.permutation(len(by_label[label]))[:n]
        balanced.extend(by_label[label][int(i)] for i in idx)
    order = rng.permutation(len(balanced))
    return [balanced[int(i)] for i in order]


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


def format_sciq(ds, seed: int) -> pd.DataFrame:
    rows = []
    labels = ["A", "B", "C", "D"]
    rng = np.random.default_rng(seed)
    for i, ex in enumerate(ds):
        choices = [
            (ex["correct_answer"], 1),
            (ex["distractor1"], 0),
            (ex["distractor2"], 0),
            (ex["distractor3"], 0),
        ]
        order = rng.permutation(len(choices))
        shuffled = [choices[j] for j in order]
        label = next(idx for idx, (_, is_correct) in enumerate(shuffled) if is_correct)
        support = ex.get("support") or ""
        support_block = f"Support: {support}\n" if support else ""
        options = "\n".join(f"{labels[j]}. {answer}" for j, (answer, _) in enumerate(shuffled))
        rows.append(
            {
                "id": str(i),
                "text": f"{support_block}Question: {ex['question']}\nOptions:\n{options}",
                "label": int(label),
            }
        )
    return pd.DataFrame(rows)


def format_paws(ds) -> pd.DataFrame:
    rows = [
        {
            "id": str(ex.get("id", i)),
            "text": (
                f"Sentence 1: {ex['sentence1']}\n"
                f"Sentence 2: {ex['sentence2']}\n\n"
                "Q: Are these sentences semantically equivalent?"
            ),
            "label": int(ex["label"]),
        }
        for i, ex in enumerate(ds)
    ]
    return pd.DataFrame(rows)


def flatten_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return "\n".join(flatten_text(item) for item in value)
    return str(value)


def load_dream_raw(split: str):
    from huggingface_hub import hf_hub_download

    split_file = {
        "train": "train.json",
        "validation": "dev.json",
        "dev": "dev.json",
        "test": "test.json",
    }.get(split, f"{split}.json")
    data_path = hf_hub_download(
        repo_id="onionmonster/dream",
        filename=f"raw/{split_file}",
        repo_type="dataset",
    )
    with open(data_path, encoding="utf-8") as handle:
        return json.load(handle)


def format_dream(ds, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, ex in enumerate(ds):
        if isinstance(ex, dict):
            dialogue = ex.get("dialogue") or ex.get("article") or ex.get("context") or ex.get("text") or []
            questions = ex.get("questions") or ex.get("question") or ex.get("qas") or []
            example_id = ex.get("id", i)
        elif isinstance(ex, (list, tuple)) and len(ex) >= 2:
            dialogue = ex[0]
            questions = ex[1]
            example_id = ex[2] if len(ex) > 2 else i
        else:
            continue

        dialogue_text = flatten_text(dialogue)
        if isinstance(questions, dict):
            questions = [questions]
        for j, qa in enumerate(questions):
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question", ""))
            answer = str(qa.get("answer", ""))
            choices = qa.get("choice") or qa.get("choices") or []
            if not question or not answer or not choices:
                continue

            label = int(rng.integers(0, 2))
            if label:
                candidate = answer
            else:
                wrong_choices = [str(choice) for choice in choices if str(choice) != answer]
                if not wrong_choices:
                    continue
                candidate = rng.choice(wrong_choices).item()

            rows.append(
                {
                    "id": f"{example_id}-{j}",
                    "text": (
                        f"Dialogue:\n{dialogue_text}\n\n"
                        f"Question: {question}\n"
                        f"Candidate answer: {candidate}\n"
                        "Is the candidate answer correct?"
                    ),
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def format_twitter_sentiment(ds) -> pd.DataFrame:
    rows = [
        {
            "id": str(ex.get("id", i)),
            "text": str(ex["text"]),
            "label": int(ex["label"]),
        }
        for i, ex in enumerate(ds)
    ]
    return pd.DataFrame(rows)


def load_boolq_splits(n_train: int, n_val: int, n_test: int, seed: int) -> dict[str, pd.DataFrame]:
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


def load_sciq_splits(n_train: int, n_val: int, n_test: int, seed: int) -> dict[str, pd.DataFrame]:
    train_raw = load_dataset("sciq", split="train").shuffle(seed=seed)
    val_raw = load_dataset("sciq", split="validation").shuffle(seed=seed)
    test_raw = load_dataset("sciq", split="test").shuffle(seed=seed)

    total_train_pool = min(n_train + n_val, len(train_raw))
    train_df = format_sciq(train_raw.select(range(total_train_pool)), seed)
    test_df = format_sciq(test_raw.select(range(min(n_test, len(test_raw)))), seed + 2)

    val_count = min(n_val, len(val_raw), len(train_df))
    if val_count:
        val_df = format_sciq(val_raw.select(range(val_count)), seed + 1)
    else:
        val_df = train_df.iloc[:0].copy()

    train_only = train_df.iloc[min(n_val, len(train_df)) :].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_only))
    split = len(perm) // 2
    weak_train = train_only.iloc[perm[:split]].reset_index(drop=True)
    strong_train = train_only.iloc[perm[split:]].reset_index(drop=True)

    return {
        "weak_train": weak_train,
        "strong_train": strong_train,
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def load_paws_splits(n_train: int, n_val: int, n_test: int, seed: int) -> dict[str, pd.DataFrame]:
    train_raw = load_dataset("paws", "labeled_final", split="train").shuffle(seed=seed)
    val_raw = load_dataset("paws", "labeled_final", split="validation").shuffle(seed=seed)
    test_raw = load_dataset("paws", "labeled_final", split="test").shuffle(seed=seed)

    train_bal = balance_hf_label(train_raw, "label", seed)
    val_bal = balance_hf_label(val_raw, "label", seed + 1)
    test_bal = balance_hf_label(test_raw, "label", seed + 2)

    total_train_pool = min(n_train + n_val, len(train_bal))
    train_df = format_paws(train_bal.select(range(total_train_pool)))
    val_df = format_paws(val_bal.select(range(min(n_val, len(val_bal)))))
    test_df = format_paws(test_bal.select(range(min(n_test, len(test_bal)))))

    train_only = train_df.iloc[min(n_val, len(train_df)) :].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_only))
    split = len(perm) // 2
    weak_train = train_only.iloc[perm[:split]].reset_index(drop=True)
    strong_train = train_only.iloc[perm[split:]].reset_index(drop=True)

    return {
        "weak_train": weak_train,
        "strong_train": strong_train,
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def split_train_pool(train_df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_df))
    split = len(perm) // 2
    weak_train = train_df.iloc[perm[:split]].reset_index(drop=True)
    strong_train = train_df.iloc[perm[split:]].reset_index(drop=True)
    return weak_train, strong_train


def load_dream_splits(n_train: int, n_val: int, n_test: int, seed: int) -> dict[str, pd.DataFrame]:
    train_df = format_dream(load_dream_raw("train"), seed)
    val_df = format_dream(load_dream_raw("validation"), seed + 1)
    test_df = format_dream(load_dream_raw("test"), seed + 2)

    train_df = pd.DataFrame(maybe_balance_rows(train_df.to_dict("records"), seed)).iloc[:n_train].reset_index(drop=True)
    val_df = pd.DataFrame(maybe_balance_rows(val_df.to_dict("records"), seed + 1)).iloc[:n_val].reset_index(drop=True)
    test_df = pd.DataFrame(maybe_balance_rows(test_df.to_dict("records"), seed + 2)).iloc[:n_test].reset_index(drop=True)
    weak_train, strong_train = split_train_pool(train_df, seed)

    return {
        "weak_train": weak_train,
        "strong_train": strong_train,
        "val": val_df,
        "test": test_df,
    }


def load_twitter_sentiment_splits(n_train: int, n_val: int, n_test: int, seed: int) -> dict[str, pd.DataFrame]:
    train_raw = load_dataset("EleutherAI/twitter-sentiment", split="train").shuffle(seed=seed)
    test_raw = load_dataset("EleutherAI/twitter-sentiment", split="test").shuffle(seed=seed + 1)

    train_bal = balance_hf_label(train_raw, "label", seed)
    test_bal = balance_hf_label(test_raw, "label", seed + 1)
    train_df = format_twitter_sentiment(train_bal.select(range(min(n_train, len(train_bal)))))

    test_pool_count = min(n_val + n_test, len(test_bal))
    test_df = format_twitter_sentiment(test_bal.select(range(test_pool_count)))
    val_df = test_df.iloc[: min(n_val, len(test_df))].reset_index(drop=True)
    final_test_df = test_df.iloc[min(n_val, len(test_df)) : min(n_val + n_test, len(test_df))].reset_index(drop=True)
    weak_train, strong_train = split_train_pool(train_df, seed)

    return {
        "weak_train": weak_train,
        "strong_train": strong_train,
        "val": val_df,
        "test": final_test_df,
    }


def load_splits(dataset: str, n_train: int, n_val: int, n_test: int, seed: int) -> dict[str, pd.DataFrame]:
    if dataset == "boolq":
        return load_boolq_splits(n_train, n_val, n_test, seed)
    if dataset == "sciq":
        return load_sciq_splits(n_train, n_val, n_test, seed)
    if dataset == "paws":
        return load_paws_splits(n_train, n_val, n_test, seed)
    if dataset == "dream":
        return load_dream_splits(n_train, n_val, n_test, seed)
    if dataset == "twitter-sentiment":
        return load_twitter_sentiment_splits(n_train, n_val, n_test, seed)
    raise ValueError(f"Unsupported dataset: {dataset}")


def batched(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


@torch.no_grad()
def extract_prompt_embeddings(
    model_name: str,
    texts: list[str],
    device: torch.device,
    dtype_arg: str,
    batch_size: int,
    max_length: int,
    pooling: str,
    desc: str,
) -> torch.Tensor:
    dtype = resolve_dtype(dtype_arg)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    embeddings = []
    for _, text_batch in tqdm(list(batched(texts, batch_size)), desc=desc):
        inputs = tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        last_hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].to(last_hidden.dtype)

        if pooling == "mean":
            pooled = (last_hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            last_indices = inputs["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(last_hidden.shape[0], device=device)
            pooled = last_hidden[batch_indices, last_indices]

        embeddings.append(pooled.detach().to(torch.float32).cpu())

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return torch.cat(embeddings, dim=0)


def make_mapping_split(n: int, train_frac: float, seed: int) -> torch.Tensor:
    if not 0.0 < train_frac <= 1.0:
        raise ValueError("--map-train-frac must be in (0, 1].")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = max(1, int(round(n * train_frac)))
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[torch.tensor(perm[:n_train], dtype=torch.long)] = True
    return train_mask


def fit_linear_map(x_train: torch.Tensor, y_train: torch.Tensor, ridge: float) -> torch.Tensor:
    if ridge > 0:
        n = x_train.shape[0]
        gram = x_train @ x_train.T
        gram = gram + ridge * torch.eye(n, dtype=x_train.dtype, device=x_train.device)
        coeff = torch.linalg.solve(gram, y_train)
        return x_train.T @ coeff
    return torch.linalg.pinv(x_train) @ y_train


def fit_procrustes_map(x_train: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
    cross_cov = x_train.T @ y_train
    u, _, vh = torch.linalg.svd(cross_cov, full_matrices=False)
    return u @ vh


class ReLUMap(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_relu_map(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> ReLUMap:
    torch.manual_seed(seed)
    model = ReLUMap(x_train.shape[1], hidden_dim, y_train.shape[1]).to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = x_train.shape[0]

    for epoch in tqdm(range(epochs), desc="train relu map"):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(x_train[idx])
            loss = F.mse_loss(pred, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model.cpu()


def add_metrics(prefix: str, pred: torch.Tensor, target: torch.Tensor, out: pd.DataFrame) -> None:
    err = pred - target
    out[f"{prefix}_l2"] = torch.linalg.norm(err, dim=1).cpu().numpy()
    out[f"{prefix}_mse"] = err.square().mean(dim=1).cpu().numpy()
    out[f"{prefix}_cosine"] = F.cosine_similarity(pred, target, dim=1).cpu().numpy()


def summarize_metric(values: pd.Series) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_singular_values(matrix: torch.Tensor, top_k: int = 20) -> dict[str, object]:
    singular_values = torch.linalg.svdvals(matrix)
    top = singular_values[:top_k].cpu().tolist()
    total_energy = singular_values.square().sum().item()
    if total_energy > 0:
        cumulative_energy = (singular_values[:top_k].square().cumsum(dim=0) / total_energy).cpu().tolist()
    else:
        cumulative_energy = []

    positive = singular_values[singular_values > 1e-8]
    condition = None
    if len(positive) > 0:
        condition = float((positive.max() / positive.min()).item())

    rank_tol = torch.finfo(singular_values.dtype).eps * max(matrix.shape) * singular_values.max()
    rank_estimate = int((singular_values > rank_tol).sum().item())

    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "rank_estimate": rank_estimate,
        "frobenius_norm": float(torch.linalg.norm(matrix).item()),
        "spectral_norm": float(singular_values.max().item()) if len(singular_values) else 0.0,
        "condition_estimate": condition,
        "top_singular_values": top,
        "top_singular_cumulative_energy": cumulative_energy,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    splits = load_splits(args.dataset, args.n_train, args.n_val, args.n_test, args.seed)
    df = splits[args.target_split].copy()
    if args.max_examples is not None:
        df = df.iloc[: args.max_examples].reset_index(drop=True)

    texts = df["text"].tolist()
    weak_emb = extract_prompt_embeddings(
        args.weak_model,
        texts,
        device,
        args.torch_dtype,
        args.batch_size,
        args.max_length,
        args.pooling,
        "extract weak embeddings",
    )
    strong_emb = extract_prompt_embeddings(
        args.strong_model,
        texts,
        device,
        args.torch_dtype,
        args.batch_size,
        args.max_length,
        args.pooling,
        "extract strong embeddings",
    )

    train_mask = make_mapping_split(len(df), args.map_train_frac, args.seed)
    x = weak_emb
    y = strong_emb
    x_train = x[train_mask]
    y_train = y[train_mask]

    linear_map = fit_linear_map(x_train, y_train, args.ridge)
    linear_pred = x @ linear_map

    procrustes_map = fit_procrustes_map(x_train, y_train)
    procrustes_pred = x @ procrustes_map

    map_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relu_model = fit_relu_map(
        x_train,
        y_train,
        args.relu_hidden_dim,
        args.relu_epochs,
        args.relu_lr,
        args.relu_weight_decay,
        args.relu_batch_size,
        map_device,
        args.seed,
    )
    with torch.no_grad():
        relu_pred = relu_model(x).cpu()

    out = df.copy()
    out["dataset"] = args.dataset
    out["split"] = args.target_split
    out["map_train"] = train_mask.cpu().numpy().astype(int)
    out["weak_norm"] = torch.linalg.norm(weak_emb, dim=1).cpu().numpy()
    out["strong_norm"] = torch.linalg.norm(strong_emb, dim=1).cpu().numpy()
    add_metrics("linear", linear_pred, y, out)
    add_metrics("procrustes", procrustes_pred, y, out)
    add_metrics("relu", relu_pred, y, out)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    summary = {
        "dataset": args.dataset,
        "target_split": args.target_split,
        "n_examples": int(len(df)),
        "n_mapping_train": int(train_mask.sum().item()),
        "weak_model": args.weak_model,
        "strong_model": args.strong_model,
        "pooling": args.pooling,
        "weak_embedding_dim": int(weak_emb.shape[1]),
        "strong_embedding_dim": int(strong_emb.shape[1]),
        "ridge": args.ridge,
        "relu_hidden_dim": args.relu_hidden_dim,
        "relu_epochs": args.relu_epochs,
        "linear_map": summarize_singular_values(linear_map),
        "procrustes_map": summarize_singular_values(procrustes_map),
        "metrics": {},
        "output": str(output_path),
        "map_output": args.map_output,
    }
    for prefix in ["linear", "procrustes", "relu"]:
        summary["metrics"][f"{prefix}_l2"] = summarize_metric(out[f"{prefix}_l2"])
        summary["metrics"][f"{prefix}_mse"] = summarize_metric(out[f"{prefix}_mse"])
        summary["metrics"][f"{prefix}_cosine"] = summarize_metric(out[f"{prefix}_cosine"])

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.embedding_output:
        emb_path = Path(args.embedding_output)
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "weak_embeddings": weak_emb,
                "strong_embeddings": strong_emb,
                "map_train_mask": train_mask,
                "weak_model": args.weak_model,
                "strong_model": args.strong_model,
                "pooling": args.pooling,
                "dataset": args.dataset,
                "target_split": args.target_split,
                "seed": args.seed,
            },
            emb_path,
        )

    if args.map_output:
        map_path = Path(args.map_output)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "linear_map": linear_map,
                "procrustes_map": procrustes_map,
                "relu_state_dict": relu_model.state_dict(),
                "linear_predictions": linear_pred,
                "procrustes_predictions": procrustes_pred,
                "relu_predictions": relu_pred,
                "linear_residuals": linear_pred - y,
                "procrustes_residuals": procrustes_pred - y,
                "relu_residuals": relu_pred - y,
                "weak_embeddings": weak_emb,
                "strong_embeddings": strong_emb,
                "map_train_mask": train_mask,
                "weak_model": args.weak_model,
                "strong_model": args.strong_model,
                "pooling": args.pooling,
                "dataset": args.dataset,
                "target_split": args.target_split,
                "seed": args.seed,
                "ridge": args.ridge,
                "relu_hidden_dim": args.relu_hidden_dim,
                "relu_epochs": args.relu_epochs,
            },
            map_path,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
