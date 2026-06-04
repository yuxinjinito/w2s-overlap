#!/usr/bin/env python3
"""Sanity-check extraction of the activation before a causal LM's lm_head.

For one prompt, this script prints the shape of:

    prompt tokens
    final hidden states before lm_head
    final-token activation before lm_head
    logits after applying lm_head

This is meant as a small meeting-ready check for using weak-model activations as
embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT = (
    "Passage: An ursid hybrid is an animal with parents from two different species "
    "or subspecies of the Ursidae family. Species and subspecies of bear known to "
    "have produced offspring with another bear species or subspecies include brown "
    "bears, black bears, grizzly bears and polar bears.\n"
    "Question: can a black bear and brown bear mate"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-activation", default=None)
    parser.add_argument("--save-json", default=None)
    parser.add_argument("--top-k", type=int, default=5)
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


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def maybe_load_model(model_name: str, dtype_arg: str, device: torch.device):
    dtype = resolve_dtype(dtype_arg)
    model_kwargs = {}
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    return model


def top_tokens(tokenizer, logits: torch.Tensor, k: int) -> list[dict[str, float | str | int]]:
    probs = torch.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, k=min(k, probs.shape[-1]))
    rows = []
    for token_id, prob in zip(top.indices.tolist(), top.values.tolist(), strict=False):
        rows.append(
            {
                "token_id": int(token_id),
                "token_text": tokenizer.decode([token_id]),
                "probability": float(prob),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    prompt = load_prompt(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = maybe_load_model(args.model, args.torch_dtype, device)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    final_hidden_states = outputs.hidden_states[-1]
    last_token_index = int(inputs["attention_mask"][0].sum().item() - 1)
    final_token_activation = final_hidden_states[0, last_token_index]

    # This is the explicit before-lm_head -> after-lm_head relationship.
    logits_from_activation = model.lm_head(final_token_activation)
    logits_from_forward = outputs.logits[0, last_token_index]
    max_abs_logit_diff = float((logits_from_activation - logits_from_forward).abs().max().item())

    report = {
        "model": args.model,
        "device": str(device),
        "model_dtype": str(next(model.parameters()).dtype),
        "prompt": prompt,
        "n_tokens": int(inputs["input_ids"].shape[1]),
        "last_token_index": last_token_index,
        "last_token_text": tokenizer.decode([int(inputs["input_ids"][0, last_token_index].item())]),
        "input_ids_shape": list(inputs["input_ids"].shape),
        "num_hidden_state_tensors": len(outputs.hidden_states),
        "final_hidden_states_shape": list(final_hidden_states.shape),
        "before_lm_head_activation_shape": list(final_token_activation.shape),
        "lm_head_weight_shape": list(model.lm_head.weight.shape),
        "after_lm_head_logits_shape": list(logits_from_activation.shape),
        "forward_logits_shape": list(outputs.logits.shape),
        "max_abs_diff_lm_head_vs_forward_logits": max_abs_logit_diff,
        "activation_l2_norm": float(final_token_activation.float().norm().item()),
        "activation_first_8_values": [float(x) for x in final_token_activation[:8].float().cpu().tolist()],
        "top_next_tokens": top_tokens(tokenizer, logits_from_activation, args.top_k),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.save_activation:
        path = Path(args.save_activation)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": args.model,
                "prompt": prompt,
                "input_ids": inputs["input_ids"].detach().cpu(),
                "attention_mask": inputs["attention_mask"].detach().cpu(),
                "last_token_index": last_token_index,
                "before_lm_head_activation": final_token_activation.detach().to(torch.float32).cpu(),
                "logits": logits_from_activation.detach().to(torch.float32).cpu(),
            },
            path,
        )
        print(f"Wrote activation bundle to {path}")

    if args.save_json:
        path = Path(args.save_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote JSON report to {path}")


if __name__ == "__main__":
    main()
