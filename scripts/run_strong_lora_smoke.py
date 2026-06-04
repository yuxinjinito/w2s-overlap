#!/usr/bin/env python3
"""Tiny LoRA fine-tuning smoke test for a 4B/8B-scale strong model.

This script is intentionally not a real training run. It answers a narrower
question from the 2026-05-15 todo list: can the available GPU complete a few
training steps on a strong model when we use a parameter-efficient setup?

It writes a JSON report with the loss values and peak CUDA memory.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from itertools import cycle
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--output-dir", default="results/finetune_smoke/llama31_8b_lora_smoke")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-adapter", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def require_peft():
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: peft. Install it inside the remote environment with:\n"
            "  python3 -m pip install peft\n"
            "Then rerun this smoke test."
        ) from exc
    return LoraConfig, TaskType, get_peft_model


def tiny_training_texts() -> list[str]:
    return [
        (
            "Sentence 1: The Red Sox won the series but lost the championship.\n"
            "Sentence 2: The Red Sox won the series, but they lost the championship.\n"
            "Q: Are these sentences semantically equivalent?\nA: yes"
        ),
        (
            "Sentence 1: The river A is a tributary of river B.\n"
            "Sentence 2: River B is a tributary of river A.\n"
            "Q: Are these sentences semantically equivalent?\nA: no"
        ),
        (
            "Question: What force slows down or stops motion?\n"
            "Options: A. thrust B. tension C. resistance D. friction\n"
            "Answer: D. friction"
        ),
        (
            "Passage: A third-party beneficiary can sue when a contract made for their benefit is breached.\n"
            "Question: can a third party beneficiary sue for breach of contract?\nA: yes"
        ),
    ]


def make_loader(tokenizer, max_length: int, batch_size: int) -> DataLoader:
    texts = tiny_training_texts()
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    encoded["labels"] = labels
    dataset = [{key: value[i] for key, value in encoded.items()} for i in range(len(texts))]
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def count_params(model) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


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


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This smoke test should be run on a CUDA GPU machine.")

    LoraConfig, TaskType, get_peft_model = require_peft()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    dtype = resolve_dtype(args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.to(args.device)
    model.train()

    total_params, trainable_params = count_params(model)
    loader = make_loader(tokenizer, args.max_length, args.batch_size)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    losses = []
    optimizer.zero_grad(set_to_none=True)
    data_iter = cycle(loader)
    for step in range(args.max_steps):
        step_losses = []
        for _ in range(args.gradient_accumulation_steps):
            batch = next(data_iter)
            batch = {key: value.to(args.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            step_losses.append(float(loss.detach().cpu().item() * args.gradient_accumulation_steps))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(sum(step_losses) / len(step_losses))
        print(f"step={step + 1} loss={losses[-1]:.6f}")

    if args.save_adapter:
        adapter_dir = output_dir / "adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    report = {
        "status": "success",
        "model": args.model,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "torch_dtype": args.torch_dtype,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_param_fraction": trainable_params / total_params if total_params else math.nan,
        "losses": losses,
        "cuda_memory": cuda_memory_report(),
        "elapsed_sec": time.time() - start_time,
        "adapter_saved": bool(args.save_adapter),
    }
    report_path = output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
