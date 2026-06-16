#!/usr/bin/env python3
"""Run first Dream weak-to-strong generalization baselines.

This is the first downstream test after the representation diagnostics:

1. base strong model, no fine-tuning;
2. strong model LoRA-tuned on ground-truth labels;
3. strong model LoRA-tuned on weak labels produced by a weak activation probe.

Optionally, it also runs a first residual-filtering baseline: train on only the
middle band of weak-labeled examples according to a saved representation-mapping
residual CSV.

The task format matches the current Dream diagnostic setup: each example is a
dialogue/question/candidate-answer tuple with a binary label indicating whether
the candidate answer is correct.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Example:
    id: str
    text: str
    label: int

    @property
    def prompt(self) -> str:
        return f"{self.text}\nAnswer:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak-model", default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--strong-model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--output-dir", default="results/w2s_dream_baselines/dream_lora_first_pass")
    parser.add_argument("--weak-train-limit", type=int, default=2048)
    parser.add_argument("--strong-train-limit", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--weak-batch-size", type=int, default=8)
    parser.add_argument("--weak-probe-epochs", type=int, default=300)
    parser.add_argument("--weak-probe-lr", type=float, default=1e-2)
    parser.add_argument("--weak-probe-weight-decay", type=float, default=1e-3)
    parser.add_argument("--strong-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-adapters", action="store_true")
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument(
        "--residual-filter-csv",
        default=None,
        help="Optional residual CSV from the Dream representation-mapping run.",
    )
    parser.add_argument("--residual-score-col", default="residual_l2")
    parser.add_argument("--residual-keep-middle-frac", type=float, default=0.5)
    parser.add_argument(
        "--residual-filter-map-train",
        choices=["all", "train", "heldout"],
        default="all",
        help="Use all residual rows, only map-training rows, or only held-out rows.",
    )
    parser.add_argument("--residual-run-name", default="weak_label_residual_middle_trained")
    parser.add_argument("--min-residual-filter-examples", type=int, default=32)
    return parser.parse_args()


def require_peft():
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: peft. Install it in the remote environment with:\n"
            "  python3 -m pip install -r requirements-finetune.txt"
        ) from exc
    return LoraConfig, TaskType, get_peft_model


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def parse_lora_target_modules(value: str) -> list[str]:
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("--lora-target-modules cannot be empty.")
    return modules


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
    return json.loads(Path(data_path).read_text(encoding="utf-8"))


def format_dream_examples(raw_data, seed: int) -> list[Example]:
    rng = np.random.default_rng(seed)
    examples: list[Example] = []
    for i, ex in enumerate(raw_data):
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
            examples.append(Example(id=f"{example_id}-{j}", text=text, label=label))
    return examples


def balance_examples(examples: list[Example], seed: int) -> list[Example]:
    rng = np.random.default_rng(seed)
    by_label = {
        0: [ex for ex in examples if ex.label == 0],
        1: [ex for ex in examples if ex.label == 1],
    }
    if not by_label[0] or not by_label[1]:
        return examples
    n = min(len(by_label[0]), len(by_label[1]))
    balanced = []
    for label in [0, 1]:
        idx = rng.permutation(len(by_label[label]))[:n]
        balanced.extend(by_label[label][int(i)] for i in idx)
    order = rng.permutation(len(balanced))
    return [balanced[int(i)] for i in order]


def load_splits(args: argparse.Namespace) -> tuple[list[Example], list[Example], list[Example]]:
    train = balance_examples(format_dream_examples(load_dream_raw("train"), args.seed), args.seed)
    eval_examples = balance_examples(format_dream_examples(load_dream_raw("validation"), args.seed + 1), args.seed + 1)

    total_train = min(len(train), args.weak_train_limit + args.strong_train_limit)
    train = train[:total_train]
    weak_train = train[: min(args.weak_train_limit, len(train))]
    strong_train = train[min(args.weak_train_limit, len(train)) : total_train]
    eval_examples = eval_examples[: min(args.eval_limit, len(eval_examples))]
    return weak_train, strong_train, eval_examples


def batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@torch.no_grad()
def extract_final_token_activations(
    model_name: str,
    texts: list[str],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    max_length: int,
    desc: str,
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

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
        pooled = last_hidden[batch_indices, last_indices]
        activations.append(pooled.detach().float().cpu())

    del model
    del tokenizer
    clear_memory()
    return torch.cat(activations, dim=0)


def train_weak_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> torch.nn.Linear:
    torch.manual_seed(seed)
    probe = torch.nn.Linear(x_train.shape[1], 1)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    y_float = y_train.float().view(-1, 1)
    for _ in tqdm(range(epochs), desc="train weak probe"):
        logits = probe(x_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_float)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return probe


@torch.no_grad()
def predict_with_probe(probe: torch.nn.Linear, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = probe(x).squeeze(-1)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    return probs, preds


def answer_text(label: int) -> str:
    return " yes" if int(label) == 1 else " no"


def encode_prompt_answer(tokenizer, prompt: str, answer: str, max_length: int) -> dict[str, list[int]]:
    """Encode prompt+answer while always keeping answer tokens.

    Dream dialogues can be long. If we tokenize the full string with ordinary
    truncation, the final " yes"/" no" answer can be truncated away, which makes
    the causal-LM loss undefined because every label is -100. We instead reserve
    space for the answer and left-truncate the prompt so the question/candidate
    answer near the end is preserved.
    """

    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not answer_ids:
        raise ValueError(f"Answer produced no tokens: {answer!r}")

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    prompt_budget = max_length - len(answer_ids)
    if prompt_budget <= 0:
        raise ValueError(f"max_length={max_length} is too short for answer {answer!r}")
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def pad_encoded_batch(tokenizer, encoded_rows: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in encoded_rows)
    pad_id = tokenizer.pad_token_id
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for row in encoded_rows:
        pad_len = max_len - len(row["input_ids"])
        batch["input_ids"].append(row["input_ids"] + [pad_id] * pad_len)
        batch["attention_mask"].append(row["attention_mask"] + [0] * pad_len)
        batch["labels"].append(row["labels"] + [-100] * pad_len)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


class PromptAnswerDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[Example], labels: list[int]):
        self.examples = examples
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.examples[idx].prompt, answer_text(self.labels[idx])


class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        encoded_rows = [
            encode_prompt_answer(self.tokenizer, prompt, answer, self.max_length)
            for prompt, answer in batch
        ]
        return pad_encoded_batch(self.tokenizer, encoded_rows)


def load_strong_model_and_tokenizer(args: argparse.Namespace, trainable_lora: bool):
    dtype = resolve_dtype(args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.strong_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.strong_model, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if trainable_lora:
        LoraConfig, TaskType, get_peft_model = require_peft()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=parse_lora_target_modules(
                getattr(args, "lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
            ),
        )
        model = get_peft_model(model, lora_config)
    model.to(args.device)
    return model, tokenizer


def count_params(model) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def train_lora_model(
    args: argparse.Namespace,
    train_examples: list[Example],
    train_labels: list[int],
    run_name: str,
    output_dir: Path,
    curriculum_order=None,
) -> tuple[object, object, dict]:
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_strong_model_and_tokenizer(args, trainable_lora=True)
    model.train()
    total_params, trainable_params = count_params(model)
    # Curriculum: when an order is given, present examples in that fixed order each pass
    # (easy/reliable -> hard) instead of random shuffling. The effective batch, #steps and
    # data are unchanged -- only the ordering differs (the control is the shuffled run).
    if curriculum_order is not None:
        train_examples = [train_examples[i] for i in curriculum_order]
        train_labels = [train_labels[i] for i in curriculum_order]
    loader = DataLoader(
        PromptAnswerDataset(train_examples, train_labels),
        batch_size=args.strong_batch_size,
        shuffle=(curriculum_order is None),
        collate_fn=Collator(tokenizer, args.max_length),
    )
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
    )
    warmup_steps = int(getattr(args, "warmup_steps", 0))
    lr_decay = str(getattr(args, "lr_decay", "linear")).lower()
    total_steps = max(1, int(args.max_train_steps))

    def _lr_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if lr_decay == "linear":
            # linear decay from 1.0 (end of warmup) to ~0 at the final step, so training
            # settles instead of oscillating at constant lr (which made small noisy
            # subsets occasionally land on a bad final model).
            denom = max(1, total_steps - warmup_steps)
            return max(0.0, float(total_steps - step) / float(denom))
        return 1.0  # "none" -> constant after warmup

    scheduler = None
    if warmup_steps > 0 or lr_decay != "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_factor)
    data_iter = cycle(loader)
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for step in tqdm(range(args.max_train_steps), desc=f"train {run_name}"):
        step_losses = []
        for _ in range(args.gradient_accumulation_steps):
            batch = next(data_iter)
            batch = {key: value.to(args.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            step_losses.append(float(loss.detach().cpu().item() * args.gradient_accumulation_steps))
        max_grad_norm = float(getattr(args, "max_grad_norm", 0.0))
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(sum(step_losses) / len(step_losses))

    if args.save_adapters:
        adapter_dir = output_dir / f"{run_name}_adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    report = {
        "run_name": run_name,
        "train_examples": len(train_examples),
        "train_label_mean": float(np.mean(train_labels)) if train_labels else math.nan,
        "max_train_steps": args.max_train_steps,
        "batch_size": args.strong_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr": args.lr,
        "weight_decay": float(getattr(args, "weight_decay", 0.0)),
        "warmup_steps": int(getattr(args, "warmup_steps", 0)),
        "lr_decay": str(getattr(args, "lr_decay", "linear")).lower(),
        "max_grad_norm": float(getattr(args, "max_grad_norm", 0.0)),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": parse_lora_target_modules(
            getattr(args, "lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
        ),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_param_fraction": trainable_params / total_params if total_params else math.nan,
        "losses": losses,
        "elapsed_sec": time.time() - start_time,
        "cuda_memory": cuda_memory_report(),
    }
    return model, tokenizer, report


def load_residual_scores(
    path: str | Path,
    score_col: str,
    map_train_filter: str,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "id" not in (reader.fieldnames or []):
            raise ValueError(f"Residual CSV is missing an id column: {path}")
        if score_col not in (reader.fieldnames or []):
            raise ValueError(f"Residual CSV is missing score column {score_col!r}: {path}")
        for row in reader:
            if map_train_filter != "all" and "map_train" in row:
                is_map_train = str(row.get("map_train", "")).strip() in {"1", "true", "True"}
                if map_train_filter == "train" and not is_map_train:
                    continue
                if map_train_filter == "heldout" and is_map_train:
                    continue
            try:
                scores[str(row["id"])] = float(row[score_col])
            except (TypeError, ValueError):
                continue
    return scores


def make_residual_middle_subset(
    examples: list[Example],
    weak_labels: list[int],
    residual_scores: dict[str, float],
    keep_middle_frac: float,
    min_examples: int,
) -> tuple[list[Example], list[int], list[dict], dict]:
    if not 0.0 < keep_middle_frac <= 1.0:
        raise ValueError("--residual-keep-middle-frac must be in (0, 1].")

    matched = []
    for ex, weak_label in zip(examples, weak_labels):
        if ex.id not in residual_scores:
            continue
        matched.append(
            {
                "example": ex,
                "weak_label": int(weak_label),
                "residual_score": float(residual_scores[ex.id]),
            }
        )
    if len(matched) < min_examples:
        raise SystemExit(
            f"Only {len(matched)} strong-training examples matched the residual CSV, "
            f"below --min-residual-filter-examples={min_examples}."
        )

    matched.sort(key=lambda row: row["residual_score"])
    n_keep = max(min_examples, int(round(len(matched) * keep_middle_frac)))
    n_keep = min(n_keep, len(matched))
    start = (len(matched) - n_keep) // 2
    end = start + n_keep
    kept_ids = {row["example"].id for row in matched[start:end]}

    filtered_examples: list[Example] = []
    filtered_labels: list[int] = []
    rows: list[dict] = []
    for rank, row in enumerate(matched):
        ex = row["example"]
        kept = ex.id in kept_ids
        if kept:
            filtered_examples.append(ex)
            filtered_labels.append(row["weak_label"])
        rows.append(
            {
                "id": ex.id,
                "label": ex.label,
                "weak_label": row["weak_label"],
                "weak_correct": int(row["weak_label"] == ex.label),
                "residual_rank": rank,
                "residual_score": row["residual_score"],
                "kept": int(kept),
                "text": ex.text,
            }
        )

    kept_scores = [row["residual_score"] for row in matched[start:end]]
    kept_weak_labels = [row["weak_label"] for row in matched[start:end]]
    kept_weak_correct = [
        int(row["weak_label"] == row["example"].label)
        for row in matched[start:end]
    ]
    summary = {
        "matched_examples": len(matched),
        "kept_examples": len(filtered_examples),
        "keep_middle_frac": keep_middle_frac,
        "dropped_low_examples": start,
        "dropped_high_examples": len(matched) - end,
        "kept_residual_min": float(min(kept_scores)) if kept_scores else math.nan,
        "kept_residual_max": float(max(kept_scores)) if kept_scores else math.nan,
        "kept_residual_mean": float(np.mean(kept_scores)) if kept_scores else math.nan,
        "kept_weak_label_mean": float(np.mean(kept_weak_labels)) if kept_weak_labels else math.nan,
        "kept_weak_label_accuracy": float(np.mean(kept_weak_correct)) if kept_weak_correct else math.nan,
    }
    return filtered_examples, filtered_labels, rows, summary


@torch.no_grad()
def score_candidates(
    model,
    tokenizer,
    prompts: list[str],
    candidates: list[str],
    device: str,
    max_length: int,
) -> torch.Tensor:
    encoded = pad_encoded_batch(
        tokenizer,
        [
            encode_prompt_answer(tokenizer, prompt, candidate, max_length)
            for prompt, candidate in zip(prompts, candidates)
        ],
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = model(**encoded, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    targets = encoded["input_ids"][:, 1:]
    labels = encoded["labels"][:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_scores = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    scores = []
    for row_idx in range(targets.shape[0]):
        candidate_mask = labels[row_idx] != -100
        if not candidate_mask.any():
            raise RuntimeError("Candidate answer was fully masked during scoring.")
        scores.append(token_scores[row_idx][candidate_mask].sum())
    return torch.stack(scores).detach().cpu()


@torch.no_grad()
def evaluate_yes_no(
    model,
    tokenizer,
    examples: list[Example],
    batch_size: int,
    device: str,
    max_length: int,
    desc: str,
) -> tuple[dict, list[dict]]:
    model.eval()
    rows = []
    for batch in tqdm(list(batched(examples, batch_size)), desc=desc):
        prompts = [ex.prompt for ex in batch]
        yes_scores = score_candidates(model, tokenizer, prompts, [" yes"] * len(batch), device, max_length)
        no_scores = score_candidates(model, tokenizer, prompts, [" no"] * len(batch), device, max_length)
        probs = torch.softmax(torch.stack([no_scores, yes_scores], dim=1), dim=1)[:, 1]
        preds = (probs >= 0.5).long()
        for ex, prob, pred in zip(batch, probs.tolist(), preds.tolist()):
            rows.append(
                {
                    "id": ex.id,
                    "label": ex.label,
                    "prob_label1": float(prob),
                    "pred": int(pred),
                    "correct": int(pred == ex.label),
                }
            )
    acc = sum(row["correct"] for row in rows) / len(rows) if rows else math.nan
    confidence = [2.0 * abs(row["prob_label1"] - 0.5) for row in rows]
    summary = {
        "n": len(rows),
        "accuracy": acc,
        "confidence_mean": float(np.mean(confidence)) if confidence else math.nan,
        "confidence_median": float(np.median(confidence)) if confidence else math.nan,
    }
    return summary, rows


def cuda_memory_report() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {
            "allocated_gb": None,
            "reserved_gb": None,
            "max_allocated_gb": None,
            "max_reserved_gb": None,
        }
    denom = 1024**3
    return {
        "allocated_gb": torch.cuda.memory_allocated() / denom,
        "reserved_gb": torch.cuda.memory_reserved() / denom,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / denom,
        "max_reserved_gb": torch.cuda.max_memory_reserved() / denom,
    }


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_predictions(path: Path, examples: list[Example], columns: dict[str, list[dict]], weak_eval_rows: list[dict]) -> None:
    weak_by_id = {row["id"]: row for row in weak_eval_rows}
    column_by_id = {
        name: {row["id"]: row for row in rows}
        for name, rows in columns.items()
    }
    fieldnames = ["id", "label", "text", "weak_prob_label1", "weak_pred", "weak_correct"]
    for name in columns:
        fieldnames.extend([f"{name}_prob_label1", f"{name}_pred", f"{name}_correct"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ex in examples:
            row = {"id": ex.id, "label": ex.label, "text": ex.text}
            weak = weak_by_id.get(ex.id, {})
            row.update(
                {
                    "weak_prob_label1": weak.get("prob_label1"),
                    "weak_pred": weak.get("pred"),
                    "weak_correct": weak.get("correct"),
                }
            )
            for name, by_id in column_by_id.items():
                pred_row = by_id.get(ex.id, {})
                row[f"{name}_prob_label1"] = pred_row.get("prob_label1")
                row[f"{name}_pred"] = pred_row.get("pred")
                row[f"{name}_correct"] = pred_row.get("correct")
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this script on a CUDA GPU machine.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = resolve_dtype(args.torch_dtype)
    device = torch.device(args.device)

    weak_train, strong_train, eval_examples = load_splits(args)
    if not strong_train:
        raise SystemExit("No strong-training examples were loaded. Increase the Dream train pool or lower weak-train-limit.")

    weak_train_x = extract_final_token_activations(
        args.weak_model,
        [ex.text for ex in weak_train],
        device,
        dtype,
        args.weak_batch_size,
        args.max_length,
        "extract weak-train activations",
    )
    strong_train_x = extract_final_token_activations(
        args.weak_model,
        [ex.text for ex in strong_train],
        device,
        dtype,
        args.weak_batch_size,
        args.max_length,
        "extract strong-train weak activations",
    )
    eval_x = extract_final_token_activations(
        args.weak_model,
        [ex.text for ex in eval_examples],
        device,
        dtype,
        args.weak_batch_size,
        args.max_length,
        "extract eval weak activations",
    )

    weak_train_y = torch.tensor([ex.label for ex in weak_train], dtype=torch.long)
    strong_train_y = torch.tensor([ex.label for ex in strong_train], dtype=torch.long)
    eval_y = torch.tensor([ex.label for ex in eval_examples], dtype=torch.long)
    probe = train_weak_probe(
        weak_train_x,
        weak_train_y,
        args.weak_probe_epochs,
        args.weak_probe_lr,
        args.weak_probe_weight_decay,
        args.seed,
    )
    weak_train_probs, weak_train_preds = predict_with_probe(probe, weak_train_x)
    strong_train_probs, strong_train_preds = predict_with_probe(probe, strong_train_x)
    eval_probs, eval_preds = predict_with_probe(probe, eval_x)

    weak_eval_rows = []
    for ex, prob, pred in zip(eval_examples, eval_probs.tolist(), eval_preds.tolist()):
        weak_eval_rows.append(
            {
                "id": ex.id,
                "label": ex.label,
                "prob_label1": float(prob),
                "pred": int(pred),
                "correct": int(pred == ex.label),
            }
        )
    weak_summary = {
        "weak_train_accuracy": float((weak_train_preds == weak_train_y).float().mean().item()),
        "strong_train_weak_label_accuracy": float((strong_train_preds == strong_train_y).float().mean().item()),
        "eval_weak_accuracy": float((eval_preds == eval_y).float().mean().item()),
        "strong_train_weak_label_mean": float(strong_train_preds.float().mean().item()),
    }
    torch.save(
        {
            "probe_state_dict": probe.state_dict(),
            "weak_train_accuracy": weak_summary["weak_train_accuracy"],
            "strong_train_weak_label_accuracy": weak_summary["strong_train_weak_label_accuracy"],
            "eval_weak_accuracy": weak_summary["eval_weak_accuracy"],
        },
        output_dir / "weak_probe.pt",
    )

    prediction_columns = {}
    train_reports = {}

    if not args.skip_base_eval:
        model, tokenizer = load_strong_model_and_tokenizer(args, trainable_lora=False)
        base_summary, base_rows = evaluate_yes_no(
            model,
            tokenizer,
            eval_examples,
            args.strong_batch_size,
            args.device,
            args.max_length,
            "eval base strong",
        )
        prediction_columns["base_strong"] = base_rows
        train_reports["base_strong"] = {"eval": base_summary}
        del model
        del tokenizer
        clear_memory()

    gt_model, gt_tokenizer, gt_report = train_lora_model(
        args,
        strong_train,
        [ex.label for ex in strong_train],
        "ground_truth_trained",
        output_dir,
    )
    gt_summary, gt_rows = evaluate_yes_no(
        gt_model,
        gt_tokenizer,
        eval_examples,
        args.strong_batch_size,
        args.device,
        args.max_length,
        "eval ground-truth LoRA",
    )
    prediction_columns["ground_truth_trained"] = gt_rows
    train_reports["ground_truth_trained"] = {**gt_report, "eval": gt_summary}
    del gt_model
    del gt_tokenizer
    clear_memory()

    weak_model, weak_tokenizer, weak_report = train_lora_model(
        args,
        strong_train,
        strong_train_preds.tolist(),
        "weak_label_trained",
        output_dir,
    )
    weak_trained_summary, weak_trained_rows = evaluate_yes_no(
        weak_model,
        weak_tokenizer,
        eval_examples,
        args.strong_batch_size,
        args.device,
        args.max_length,
        "eval weak-label LoRA",
    )
    prediction_columns["weak_label_trained"] = weak_trained_rows
    train_reports["weak_label_trained"] = {**weak_report, "eval": weak_trained_summary}
    del weak_model
    del weak_tokenizer
    clear_memory()

    residual_filter_summary = None
    residual_filter_rows = None
    if args.residual_filter_csv:
        residual_scores = load_residual_scores(
            args.residual_filter_csv,
            args.residual_score_col,
            args.residual_filter_map_train,
        )
        (
            filtered_examples,
            filtered_labels,
            residual_filter_rows,
            residual_filter_summary,
        ) = make_residual_middle_subset(
            strong_train,
            strong_train_preds.tolist(),
            residual_scores,
            args.residual_keep_middle_frac,
            args.min_residual_filter_examples,
        )
        filtered_model, filtered_tokenizer, filtered_report = train_lora_model(
            args,
            filtered_examples,
            filtered_labels,
            args.residual_run_name,
            output_dir,
        )
        filtered_summary, filtered_rows = evaluate_yes_no(
            filtered_model,
            filtered_tokenizer,
            eval_examples,
            args.strong_batch_size,
            args.device,
            args.max_length,
            f"eval {args.residual_run_name}",
        )
        prediction_columns[args.residual_run_name] = filtered_rows
        train_reports[args.residual_run_name] = {
            **filtered_report,
            "eval": filtered_summary,
            "residual_filter": residual_filter_summary,
        }
        del filtered_model
        del filtered_tokenizer
        clear_memory()

    write_predictions(output_dir / "eval_predictions.csv", eval_examples, prediction_columns, weak_eval_rows)
    with (output_dir / "strong_train_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label", "weak_prob_label1", "weak_label", "weak_correct", "text"])
        writer.writeheader()
        for ex, prob, weak_label in zip(strong_train, strong_train_probs.tolist(), strong_train_preds.tolist()):
            writer.writerow(
                {
                    "id": ex.id,
                    "label": ex.label,
                    "weak_prob_label1": float(prob),
                    "weak_label": int(weak_label),
                    "weak_correct": int(int(weak_label) == ex.label),
                    "text": ex.text,
                }
            )

    if residual_filter_rows is not None:
        with (output_dir / "residual_filter_train_labels.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "label",
                    "weak_label",
                    "weak_correct",
                    "residual_rank",
                    "residual_score",
                    "kept",
                    "text",
                ],
            )
            writer.writeheader()
            writer.writerows(residual_filter_rows)

    summary = {
        "dataset": "dream",
        "task_format": "binary candidate-answer correctness",
        "weak_model": args.weak_model,
        "strong_model": args.strong_model,
        "n_weak_train": len(weak_train),
        "n_strong_train": len(strong_train),
        "n_eval": len(eval_examples),
        "seed": args.seed,
        "max_length": args.max_length,
        "weak_probe": weak_summary,
        "residual_filter": residual_filter_summary,
        "runs": train_reports,
        "outputs": {
            "summary": str(output_dir / "summary.json"),
            "eval_predictions": str(output_dir / "eval_predictions.csv"),
            "strong_train_labels": str(output_dir / "strong_train_labels.csv"),
            "weak_probe": str(output_dir / "weak_probe.pt"),
            "residual_filter_train_labels": (
                str(output_dir / "residual_filter_train_labels.csv")
                if residual_filter_rows is not None
                else None
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
