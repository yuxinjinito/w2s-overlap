#!/usr/bin/env python3
"""Print the final-token hidden vector for one prompt.

This is the minimal version of the before-lm_head check: it does not inspect
lm_head or logits. It only shows how to get the model-internal vector that can
later be used as an embedding.
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
    parser.add_argument("--max-values", type=int, default=20, help="Number of vector values to print.")
    parser.add_argument("--print-all", action="store_true", help="Print the full hidden vector.")
    parser.add_argument("--save-txt", default=None)
    parser.add_argument("--save-json", default=None)
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


def load_model(model_name: str, dtype_arg: str, device: torch.device):
    dtype = resolve_dtype(dtype_arg)
    model_kwargs = {}
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    return model


def format_text_report(report: dict) -> str:
    values = report["hidden_vector_values"]
    value_title = (
        "Hidden vector values"
        if report["printed_full_vector"]
        else f"Hidden vector values, first {len(values)}"
    )
    value_lines = [f"{i:04d}: {value:.6f}" for i, value in enumerate(values)]

    return "\n".join(
        [
            "Hidden Vector Check",
            "===================",
            "",
            f"model: {report['model']}",
            f"n_tokens: {report['n_tokens']}",
            f"last_token_index: {report['last_token_index']}",
            f"last_token_text: {report['last_token_text']!r}",
            f"final_hidden_states_shape: {report['final_hidden_states_shape']}",
            f"hidden_vector_shape: {report['hidden_vector_shape']}",
            "",
            "Prompt",
            "------",
            report["prompt"],
            "",
            value_title,
            "--------------------",
            *value_lines,
            "",
            "Note",
            "----",
            "This is the final-layer hidden vector at the final non-padding token.",
            "It is the model-internal representation we can later use as an embedding.",
        ]
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    prompt = load_prompt(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.model, args.torch_dtype, device)
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
    hidden_vector = final_hidden_states[0, last_token_index].detach().to(torch.float32).cpu()

    values = hidden_vector.tolist()
    shown_values = values if args.print_all else values[: args.max_values]
    report = {
        "model": args.model,
        "prompt": prompt,
        "n_tokens": int(inputs["input_ids"].shape[1]),
        "last_token_index": last_token_index,
        "last_token_text": tokenizer.decode([int(inputs["input_ids"][0, last_token_index].item())]),
        "final_hidden_states_shape": list(final_hidden_states.shape),
        "hidden_vector_shape": list(hidden_vector.shape),
        "hidden_vector_values": shown_values,
        "printed_full_vector": bool(args.print_all),
    }
    text_report = format_text_report(report)

    print(text_report)

    if args.save_txt:
        path = Path(args.save_txt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text_report + "\n", encoding="utf-8")
        print(f"Wrote hidden-vector text report to {path}")

    if args.save_json:
        path = Path(args.save_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote hidden-vector report to {path}")


if __name__ == "__main__":
    main()
