#!/usr/bin/env python3
"""Compute weak confidence from weak activations plus a logistic probe.

This is a lightweight version of the confidence path used in Changho's overlap
code:

    weak model hidden activations -> small probe -> P(label=1)
    confidence = 2 * abs(P(label=1) - 0.5)

It intentionally does not try to reproduce the full training/evaluation stack
from the original repository. The goal is a small, inspectable bridge between
the quick yes/no logprob proxy and the original probe-based setup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PAPER_CONFIDENCE_DATASETS = [
    "boolq",
    "sciq",
    "sst2",
    "amazon_polarity",
    "cola",
    "wic",
    "paws",
    "anli-r2",
    "piqa",
    "hellaswag",
    "multirc",
    "cosmos_qa",
    "dream",
    "twitter-sentiment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=PAPER_CONFIDENCE_DATASETS, default="boolq")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weak-model", default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--activation-output", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--balance-labels", action="store_true")
    parser.add_argument("--shuffle-before-limit", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str):
    if dtype_arg == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_arg]


def changho_binary_confidence(prob_label1: np.ndarray) -> np.ndarray:
    return 2.0 * np.abs(prob_label1 - 0.5)


def maybe_shuffle_and_balance(
    examples: list[dict],
    limit: int | None,
    seed: int,
    shuffle_before_limit: bool,
    balance_labels: bool,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    if balance_labels:
        by_label = {
            0: [ex for ex in examples if int(ex["label"]) == 0],
            1: [ex for ex in examples if int(ex["label"]) == 1],
        }
        if by_label[0] and by_label[1]:
            n = min(len(by_label[0]), len(by_label[1]))
            balanced = []
            for label in [0, 1]:
                idx = rng.permutation(len(by_label[label]))[:n]
                balanced.extend(by_label[label][int(i)] for i in idx)
            examples = balanced
    if shuffle_before_limit:
        order = rng.permutation(len(examples))
        examples = [examples[int(i)] for i in order]
    if limit:
        examples = examples[: min(limit, len(examples))]
    return examples


def load_hf_dataset(path: str, name: str | None, split: str):
    if name is None:
        return load_dataset(path, split=split)
    return load_dataset(path, name, split=split)


def load_boolq(split: str, limit: int | None, seed: int, shuffle: bool, balance: bool) -> list[dict]:
    ds = load_hf_dataset("boolq", None, split)
    examples = [
        {
            "id": str(i),
            "text": f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:",
            "label": int(ex["answer"]),
        }
        for i, ex in enumerate(ds)
    ]
    return maybe_shuffle_and_balance(examples, limit, seed, shuffle, balance)


def load_sciq(split: str, limit: int | None, seed: int, shuffle: bool, balance: bool) -> list[dict]:
    rng = np.random.default_rng(seed)
    ds = load_hf_dataset("sciq", None, split)
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
    return maybe_shuffle_and_balance(examples, limit, seed, shuffle, balance)


def load_sst2(split: str, limit: int | None, seed: int, shuffle: bool, balance: bool) -> list[dict]:
    ds = load_hf_dataset("stanfordnlp/sst2", None, split)
    examples = [
        {"id": str(i), "text": ex["sentence"], "label": int(ex["label"])}
        for i, ex in enumerate(ds)
    ]
    return maybe_shuffle_and_balance(examples, limit, seed, shuffle, balance)


def load_amazon_polarity(split: str, limit: int | None, seed: int, shuffle: bool, balance: bool) -> list[dict]:
    ds = load_hf_dataset("amazon_polarity", None, split)
    examples = [
        {
            "id": str(i),
            "text": f"{ex['title']} {ex['content']}",
            "label": int(ex["label"]),
        }
        for i, ex in enumerate(ds)
    ]
    return maybe_shuffle_and_balance(examples, limit, seed, shuffle, balance)


def load_cola(split: str, limit: int | None, seed: int, shuffle: bool, balance: bool) -> list[dict]:
    ds = load_hf_dataset("nyu-mll/glue", "cola", split)
    examples = [
        {"id": str(i), "text": ex["sentence"], "label": int(ex["label"])}
        for i, ex in enumerate(ds)
    ]
    return maybe_shuffle_and_balance(examples, limit, seed, shuffle, balance)


def load_wic(split: str, limit: int | None, seed: int, shuffle: bool, balance: bool) -> list[dict]:
    ds = load_hf_dataset("super_glue", "wic", split)
    examples = [
        {
            "id": str(i),
            "text": (
                f"Sentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}\n\n"
                f'Q: Does "{ex["word"]}" have the same meaning in the above sentences?'
            ),
            "label": int(ex["label"]),
        }
        for i, ex in enumerate(ds)
    ]
    return maybe_shuffle_and_balance(examples, limit, seed, shuffle, balance)


def load_paws(split: str, limit: int | None, seed: int, shuffle: bool, balance: bool) -> list[dict]:
    ds = load_hf_dataset("paws", "labeled_final", split)
    examples = [
        {
            "id": str(i),
            "text": (
                f"Sent 1: {ex['sentence1']}\nSent 2: {ex['sentence2']}\n\n"
                "Q: Are these sentences semantically equivalent?"
            ),
            "label": int(ex["label"]),
        }
        for i, ex in enumerate(ds)
    ]
    return maybe_shuffle_and_balance(examples, limit, seed, shuffle, balance)


def load_anli_r2(split: str, limit: int | None, seed: int, shuffle: bool, balance: bool) -> list[dict]:
    split = {
        "train": "train_r2",
        "validation": "dev_r2",
        "dev": "dev_r2",
        "test": "test_r2",
    }.get(split, split)
    ds = load_hf_dataset("facebook/anli", None, split)
    examples = [
        {
            "id": str(i),
            "text": (
                f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\n\n"
                "Does the premise entail the hypothesis?"
            ),
            "label": int(ex["label"] == 0),
        }
        for i, ex in enumerate(ds)
    ]
    return maybe_shuffle_and_balance(examples, limit, seed, shuffle, balance)


def load_piqa(split: str, limit: int | None, seed: int, shuffle: bool, balance: bool) -> list[dict]:
    rng = np.random.default_rng(seed)
    ds = load_hf_dataset("piqa", None, split)
    examples = []
    for i, ex in enumerate(ds):
        correct = ex["sol2"] if int(ex["label"]) else ex["sol1"]
        wrong = ex["sol1"] if int(ex["label"]) else ex["sol2"]
        label = int(rng.integers(0, 2))
        answer = correct if label else wrong
        examples.append({"id": str(i), "text": f"Q: {ex['goal']}\nA: {answer}", "label": label})
    return maybe_shuffle_and_balance(examples, limit, seed, shuffle, balance)


def load_hellaswag(split: str, limit: int | None, seed: int, shuffle: bool, balance: bool) -> list[dict]:
    rng = np.random.default_rng(seed)
    ds = load_hf_dataset("Rowan/hellaswag", None, split)
    examples = []
    for i, ex in enumerate(ds):
        label = int(rng.integers(0, 2))
        correct_id = int(ex["label"])
        if label:
            answer = ex["endings"][correct_id]
        else:
            answer = rng.choice([ending for j, ending in enumerate(ex["endings"]) if j != correct_id]).item()
        choices = "\n".join(str(ending) for ending in ex["endings"])
        text = f"Context:\n{ex['ctx']}\n\nContinuations:\n{choices}\n\nQ: Is \"{answer}\" the best continuation?"
        examples.append({"id": str(i), "text": text, "label": label})
    return maybe_shuffle_and_balance(examples, limit, seed, shuffle, balance)


def load_multirc(split: str, limit: int | None, seed: int, shuffle: bool, balance: bool) -> list[dict]:
    ds = load_hf_dataset("super_glue", "multirc", split)
    examples = [
        {
            "id": str(i),
            "text": (
                f"Passage:\n\n{ex['paragraph']}\n\n"
                f'Q: "{ex["question"]}" Is the answer "{ex["answer"]}"?'
            ),
            "label": int(ex["label"]),
        }
        for i, ex in enumerate(ds)
    ]
    return maybe_shuffle_and_balance(examples, limit, seed, shuffle, balance)


def load_cosmos_qa(split: str, limit: int | None, seed: int, shuffle: bool, balance: bool) -> list[dict]:
    rng = np.random.default_rng(seed)
    ds = load_hf_dataset("cosmos_qa", None, split)
    examples = []
    for i, ex in enumerate(ds):
        correct = ex[f"answer{int(ex['label'])}"]
        if "None of the above choices" in correct:
            label = 0
        else:
            label = int(rng.integers(0, 2))
        if label:
            answer = correct
        else:
            candidates = [ex[f"answer{j}"] for j in range(4)]
            answer = rng.choice([item for item in candidates if item != correct]).item()
        text = f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {answer}"
        examples.append({"id": str(i), "text": text, "label": label})
    return maybe_shuffle_and_balance(examples, limit, seed, shuffle, balance)


def flatten_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return "\n".join(flatten_text(item) for item in value)
    return str(value)


def load_dream(split: str, limit: int | None, seed: int, shuffle: bool, balance: bool) -> list[dict]:
    from huggingface_hub import hf_hub_download

    rng = np.random.default_rng(seed)
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
        ds = json.load(handle)

    examples = []
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
            text = (
                f"Dialogue:\n{dialogue_text}\n\n"
                f"Question: {question}\n"
                f"Candidate answer: {candidate}\n"
                "Is the candidate answer correct?"
            )
            examples.append({"id": f"{example_id}-{j}", "text": text, "label": label})
    return maybe_shuffle_and_balance(examples, limit, seed, shuffle, balance)


def load_twitter_sentiment(split: str, limit: int | None, seed: int, shuffle: bool, balance: bool) -> list[dict]:
    split = {
        "validation": "test",
        "dev": "test",
    }.get(split, split)
    ds = load_hf_dataset("EleutherAI/twitter-sentiment", None, split)
    examples = [
        {
            "id": str(ex.get("id", i)),
            "text": str(ex["text"]),
            "label": int(ex["label"]),
        }
        for i, ex in enumerate(ds)
    ]
    return maybe_shuffle_and_balance(examples, limit, seed, shuffle, balance)


def load_examples(
    dataset: str,
    split: str,
    limit: int | None,
    seed: int,
    shuffle_before_limit: bool,
    balance_labels: bool,
) -> list[dict]:
    if dataset == "boolq":
        return load_boolq(split, limit, seed, shuffle_before_limit, balance_labels)
    if dataset == "sciq":
        return load_sciq(split, limit, seed, shuffle_before_limit, balance_labels)
    if dataset == "sst2":
        return load_sst2(split, limit, seed, shuffle_before_limit, balance_labels)
    if dataset == "amazon_polarity":
        return load_amazon_polarity(split, limit, seed, shuffle_before_limit, balance_labels)
    if dataset == "cola":
        return load_cola(split, limit, seed, shuffle_before_limit, balance_labels)
    if dataset == "wic":
        return load_wic(split, limit, seed, shuffle_before_limit, balance_labels)
    if dataset == "paws":
        return load_paws(split, limit, seed, shuffle_before_limit, balance_labels)
    if dataset == "anli-r2":
        return load_anli_r2(split, limit, seed, shuffle_before_limit, balance_labels)
    if dataset == "piqa":
        return load_piqa(split, limit, seed, shuffle_before_limit, balance_labels)
    if dataset == "hellaswag":
        return load_hellaswag(split, limit, seed, shuffle_before_limit, balance_labels)
    if dataset == "multirc":
        return load_multirc(split, limit, seed, shuffle_before_limit, balance_labels)
    if dataset == "cosmos_qa":
        return load_cosmos_qa(split, limit, seed, shuffle_before_limit, balance_labels)
    if dataset == "dream":
        return load_dream(split, limit, seed, shuffle_before_limit, balance_labels)
    if dataset == "twitter-sentiment":
        return load_twitter_sentiment(split, limit, seed, shuffle_before_limit, balance_labels)
    raise ValueError(f"Unsupported dataset: {dataset}")


def batched(items: list, batch_size: int):
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
    activations = []
    for text_batch in tqdm(list(batched(texts, batch_size)), desc=desc):
        inputs = tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        last_hidden = outputs.hidden_states[-1]
        last_indices = inputs["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden.shape[0], device=device)
        acts = last_hidden[batch_indices, last_indices]
        activations.append(acts.detach().float().cpu())
    return torch.cat(activations, dim=0)


def standardize(train_acts: torch.Tensor, eval_acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_acts.mean(dim=0, keepdim=True)
    std = train_acts.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_acts - mean) / std, (eval_acts - mean) / std


def fit_logistic_probe(
    train_acts: torch.Tensor,
    train_labels: torch.Tensor,
    eval_acts: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    train_x = train_acts.to(device)
    train_y = train_labels.float().to(device)
    eval_x = eval_acts.to(device)

    probe = torch.nn.Linear(train_x.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in tqdm(range(epochs), desc="train logistic probe"):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(train_x).squeeze(-1)
        loss = loss_fn(logits, train_y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_prob = torch.sigmoid(probe(train_x).squeeze(-1)).detach().cpu().numpy()
        eval_prob = torch.sigmoid(probe(eval_x).squeeze(-1)).detach().cpu().numpy()
    return train_prob, eval_prob


def add_probe_outputs(rows: list[dict], probs: np.ndarray) -> None:
    pred = (probs >= 0.5).astype(int)
    confidence = changho_binary_confidence(probs)
    for i, row in enumerate(rows):
        label = int(row["label"])
        row["prediction_source"] = "weak_activation_logistic_probe"
        row["weak_prob_label1"] = float(probs[i])
        row["weak_confidence"] = float(confidence[i])
        row["weak_pred"] = int(pred[i])
        row["weak_correct"] = int(pred[i] == label)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.torch_dtype)

    train_examples = load_examples(
        args.dataset,
        args.train_split,
        args.train_limit,
        args.seed,
        args.shuffle_before_limit,
        args.balance_labels,
    )
    eval_examples = load_examples(
        args.dataset,
        args.eval_split,
        args.eval_limit,
        args.seed + 1,
        args.shuffle_before_limit,
        args.balance_labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.weak_model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.weak_model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    train_texts = [ex["text"] for ex in train_examples]
    eval_texts = [ex["text"] for ex in eval_examples]
    train_acts = extract_final_token_activations(
        model,
        tokenizer,
        train_texts,
        device,
        args.batch_size,
        args.max_length,
        desc="extract train activations",
    )
    eval_acts = extract_final_token_activations(
        model,
        tokenizer,
        eval_texts,
        device,
        args.batch_size,
        args.max_length,
        desc="extract eval activations",
    )

    train_acts_std, eval_acts_std = standardize(train_acts, eval_acts)
    train_labels = torch.tensor([ex["label"] for ex in train_examples], dtype=torch.float32)
    train_probs, eval_probs = fit_logistic_probe(
        train_acts_std,
        train_labels,
        eval_acts_std,
        args.epochs,
        args.lr,
        args.weight_decay,
        device,
    )

    rows = [
        {
            "id": ex["id"],
            "dataset": args.dataset,
            "split": args.eval_split,
            "text": ex["text"],
            "label": int(ex["label"]),
        }
        for ex in eval_examples
    ]
    add_probe_outputs(rows, eval_probs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    if args.activation_output:
        act_path = Path(args.activation_output)
        act_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train_activations": train_acts,
                "eval_activations": eval_acts,
                "train_labels": train_labels,
                "eval_labels": torch.tensor([ex["label"] for ex in eval_examples], dtype=torch.float32),
                "weak_model": args.weak_model,
                "dataset": args.dataset,
                "train_split": args.train_split,
                "eval_split": args.eval_split,
                "max_length": args.max_length,
            },
            act_path,
        )

    train_pred = (train_probs >= 0.5).astype(int)
    train_acc = float((train_pred == np.array([ex["label"] for ex in train_examples])).mean())
    summary = {
        "n_train": len(train_examples),
        "n_eval": len(eval_examples),
        "weak_model": args.weak_model,
        "dataset": args.dataset,
        "prediction_source": "weak_activation_logistic_probe",
        "train_probe_accuracy": train_acc,
        "eval_probe_accuracy": float(df["weak_correct"].mean()),
        "eval_confidence_mean": float(df["weak_confidence"].mean()),
        "eval_confidence_median": float(df["weak_confidence"].median()),
        "output": str(output_path),
        "activation_output": args.activation_output,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
