#!/usr/bin/env python3
"""Compute Dream mapping residuals aligned to the W2S training split.

The clean filtering experiment should use one canonical split:

- weak_train: fit the weak probe and fit the weak-to-strong representation map;
- strong_train: generate weak labels and compute held-out mapping residuals;
- eval: evaluate strong-model LoRA runs.

This script creates the aligned mapping artifact for that setup. It does not
train LoRA models; it only extracts weak/strong representations for weak_train
and strong_train, marks weak_train as map-training rows, and saves artifacts
that `analyze_stabilized_maps.py` can turn into residual scores.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from run_dream_w2s_baselines import load_splits
from run_representation_mapping import extract_prompt_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak-model", default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--strong-model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--weak-train-limit", type=int, default=2048)
    parser.add_argument("--strong-train-limit", type=int, default=2048)
    parser.add_argument("--eval-limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling", choices=["mean", "final_token"], default="mean")
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--embedding-output", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def write_metadata_csv(path: Path, rows: list[dict], weak_emb: torch.Tensor, strong_emb: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "dataset",
        "split",
        "label",
        "text",
        "map_train",
        "weak_norm",
        "strong_norm",
    ]
    weak_norms = torch.linalg.norm(weak_emb.float(), dim=1).tolist()
    strong_norms = torch.linalg.norm(strong_emb.float(), dim=1).tolist()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, weak_norm, strong_norm in zip(rows, weak_norms, strong_norms):
            writer.writerow(
                {
                    **row,
                    "weak_norm": float(weak_norm),
                    "strong_norm": float(strong_norm),
                }
            )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    weak_train, strong_train, eval_examples = load_splits(args)
    all_examples = weak_train + strong_train
    metadata_rows = []
    for split_name, examples, map_train in [
        ("weak_train", weak_train, 1),
        ("strong_train", strong_train, 0),
    ]:
        for ex in examples:
            metadata_rows.append(
                {
                    "id": ex.id,
                    "dataset": "dream",
                    "split": split_name,
                    "label": int(ex.label),
                    "text": ex.text,
                    "map_train": int(map_train),
                }
            )

    texts = [ex.text for ex in all_examples]
    weak_emb = extract_prompt_embeddings(
        args.weak_model,
        texts,
        device,
        args.torch_dtype,
        args.batch_size,
        args.max_length,
        args.pooling,
        "extract aligned weak embeddings",
    )
    strong_emb = extract_prompt_embeddings(
        args.strong_model,
        texts,
        device,
        args.torch_dtype,
        args.batch_size,
        args.max_length,
        args.pooling,
        "extract aligned strong embeddings",
    )

    map_train_mask = torch.zeros(len(all_examples), dtype=torch.bool)
    map_train_mask[: len(weak_train)] = True

    output_csv = Path(args.output_csv)
    embedding_output = Path(args.embedding_output)
    summary_output = Path(args.summary_output)
    write_metadata_csv(output_csv, metadata_rows, weak_emb, strong_emb)

    embedding_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weak_embeddings": weak_emb,
            "strong_embeddings": strong_emb,
            "map_train_mask": map_train_mask,
            "weak_model": args.weak_model,
            "strong_model": args.strong_model,
            "pooling": args.pooling,
            "dataset": "dream",
            "target_split": "w2s_aligned_weak_train_to_strong_train",
            "seed": args.seed,
            "weak_train_limit": args.weak_train_limit,
            "strong_train_limit": args.strong_train_limit,
            "eval_limit": args.eval_limit,
        },
        embedding_output,
    )

    summary = {
        "dataset": "dream",
        "weak_model": args.weak_model,
        "strong_model": args.strong_model,
        "pooling": args.pooling,
        "seed": args.seed,
        "weak_train_examples": len(weak_train),
        "strong_train_examples": len(strong_train),
        "eval_examples": len(eval_examples),
        "total_mapping_rows": len(all_examples),
        "map_train_rows": int(map_train_mask.sum().item()),
        "heldout_residual_rows": int((~map_train_mask).sum().item()),
        "weak_embedding_dim": int(weak_emb.shape[1]),
        "strong_embedding_dim": int(strong_emb.shape[1]),
        "metadata_csv": str(output_csv),
        "embedding_output": str(embedding_output),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
