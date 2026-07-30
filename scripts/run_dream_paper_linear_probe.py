#!/usr/bin/env python3
"""Paper-faithful Dream linear-probe rerun.

This script is meant as a sanity check against the original paper / GitHub
setup for the Dream LLM experiment. It intentionally separates the
paper-faithful linear-probe path from our later LoRA + yes/no generation
extension.

Matched original choices:
- Dream is formatted as binary candidate-answer correctness:
  dialogue + "Q: ... A: candidate", label=1 iff candidate is correct.
- Data is balanced to 50/50 after formatting.
- The processed train split is divided into weak_train and strong_train halves.
- Activations are final-layer, final-token hidden states.
- Weak, strong, and W2S models are logistic linear probes over activations.
- W2S uses weak-probe soft probabilities on strong_train as pseudolabels.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset

# these began life here and moved to their layer homes; re-imported so the
# older entry points that still say `from run_dream_paper_linear_probe import`
# keep working
from contracts import SplitBundle
from paper_style_datasets import (
    flatten_text,
    iter_dream_question_rows,
    load_dream_raw,
    load_paper_dream_splits,
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak-model", default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--strong-model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-val", type=int, default=1_000)
    parser.add_argument("--n-test", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--l2-penalty", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="results/dream_paper_linear_probe/dream_seed42")
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument("--dry-run-splits", action="store_true", help="Only write split metadata/preview; do not load models.")
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
















def load_dream_3class_eval(n_questions: int, seed: int) -> list[dict[str, Any]]:
    """Build a per-question multi-candidate eval set from the Dream test split.

    For each test question, emit ALL candidate answers (one row per choice),
    sharing the question's source_id, with labels=1 on the correct choice. The
    prompt format matches format_dream_paper_style exactly, so a model trained on
    the binary 'is this candidate correct?' task can score each candidate; argmax
    of P(correct) over a question's candidates gives a true multi-class prediction.
    """
    raw_rows = list(iter_dream_question_rows(load_dream_raw("test")))
    ds = Dataset.from_list(raw_rows).shuffle(seed=seed)
    out: list[dict[str, Any]] = []
    n = 0
    for ex in ds:
        if n >= n_questions:
            break
        choices = list(ex["choice"])
        if ex["answer"] not in choices or len(choices) < 2:
            continue
        joined = "\n".join(ex["dialogue"]) if isinstance(ex["dialogue"], list) else flatten_text(ex["dialogue"])
        for choice in choices:
            txt = f"{joined}\n\nQ: {ex['question']} A: {choice}"
            out.append(
                {
                    "id": hashlib.sha1(txt.encode()).hexdigest()[:8],
                    "source_id": str(ex["source_id"]),
                    "txt": txt,
                    "labels": int(choice == ex["answer"]),
                }
            )
        n += 1
    return out


def batched_indices(n_items: int, batch_size: int):
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)


def answer_span_char_bounds(
    prompt: str, marker: str, answer_suffix: str, span_kind: str = "answer"
) -> tuple[int, int]:
    """Character range [start, end) of a sub-span of a candidate-correctness prompt.

    Prompt form: ``Premise: ...\\nHypothesis: ...\\nQ: ...? A: {candidate}`` optionally
    followed by ``answer_suffix`` (the fixed yes/no instruction). Two sub-spans:
      * ``span_kind="answer"``  -> ``A: {candidate}`` (from the LAST ``marker`` through the end
        of the text, excluding the question prefix AND the instruction suffix).
      * ``span_kind="context"`` -> everything from the start THROUGH the marker ``A:`` but
        EXCLUDING the candidate word (i.e. ``Premise: ... Q: ...? A:``).
    Returns (-1, -1) if the marker is absent (caller falls back to the full sequence).

    NOTE: only strip ``answer_suffix`` when the prompt actually ends with it. The
    representation extraction tokenizes suffix-less ``txt`` (the candidate is already at the
    end), so subtracting the suffix length there would truncate the ``A:`` marker and wrongly
    trigger the full-sequence fallback (the earlier answer-only-span bug).
    """
    end = len(prompt) - len(answer_suffix) if (answer_suffix and prompt.endswith(answer_suffix)) else len(prompt)
    marker_start = prompt.rfind(marker, 0, end)
    if marker_start < 0 or end <= marker_start:
        return -1, -1
    if span_kind == "context":
        return 0, marker_start + len(marker)
    return marker_start, end


def build_answer_span_mask(
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_texts: list[str],
    marker: str,
    answer_suffix: str,
    span_kind: str = "answer",
) -> tuple[torch.Tensor, int]:
    """0/1 mask selecting only the tokens inside each prompt's answer span (see
    :func:`answer_span_char_bounds`). ``offsets`` is the fast-tokenizer offset mapping
    [B, T, 2]; special/pad tokens have (0, 0) and are excluded. Rows whose answer span is
    empty (marker missing, or truncated away) fall back to the full attention mask. Returns
    (span_mask, n_fallback) so the caller can warn when truncation ate the answer."""
    span_mask = torch.zeros_like(attention_mask)
    n_fallback = 0
    for b, prompt in enumerate(batch_texts):
        start_char, end_char = answer_span_char_bounds(prompt, marker, answer_suffix, span_kind)
        if start_char < 0:
            span_mask[b] = attention_mask[b]
            n_fallback += 1
            continue
        row = span_mask[b]
        for t in range(offsets.shape[1]):
            s, e = int(offsets[b, t, 0]), int(offsets[b, t, 1])
            if e <= s:  # special / pad token -> (0, 0)
                continue
            if s >= start_char and e <= end_char:
                row[t] = 1
        if int((row * attention_mask[b]).sum()) == 0:  # span truncated away
            span_mask[b] = attention_mask[b]
            n_fallback += 1
    return span_mask, n_fallback


@torch.no_grad()
def extract_final_token_activations(
    model_name: str,
    texts: list[str],
    device: torch.device,
    dtype_arg: str,
    batch_size: int,
    max_length: int | None,
    desc: str,
    layer: str | int = "end",
    pooling: str = "last",
    answer_span: bool = False,
    answer_marker: str = "A:",
    answer_suffix: str = "",
    span_kind: str = "answer",
) -> torch.Tensor:
    """Per-prompt hidden state at the requested transformer layer and token pooling.

    layer: "end" (last hidden layer -- the default the weak probe uses), "middle"
    (len(hidden_states)//2; the representation methods kNN/RP can use this as a
    control), or an explicit integer index into ``outputs.hidden_states``.
    pooling: "last" (the final non-pad token -- the probe / W2S-paper lineage) or "mean"
    (average over the non-pad tokens; kNN/RP can use this as a control).
    answer_span: if True, pool only over the ANSWER part of each prompt ("A: {candidate}",
    masking the question prefix and the trailing yes/no instruction ``answer_suffix``) --
    the answer-only ablation. Needs a fast tokenizer (offset mapping). Rows whose answer
    span is missing/truncated fall back to the full sequence.
    """
    dtype = resolve_dtype(dtype_arg)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "num_labels": 2,
        "low_cpu_mem_usage": True,
    }
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = "single_label_classification"
    model.to(device)
    model.eval()

    activations = []
    span_fallbacks = 0
    for start, end in tqdm(list(batched_indices(len(texts), batch_size)), desc=desc):
        batch_texts = texts[start:end]
        tokenize_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        }
        if max_length is not None:
            tokenize_kwargs["max_length"] = max_length
        if answer_span:
            tokenize_kwargs["return_offsets_mapping"] = True
        inputs = tokenizer(batch_texts, **tokenize_kwargs)
        offsets = inputs.pop("offset_mapping", None)
        inputs = inputs.to(device)
        # pool_mask selects which tokens contribute: the full attention mask, or (answer_span)
        # only the answer-part tokens of each prompt.
        if answer_span:
            span_mask, n_fb = build_answer_span_mask(
                offsets, inputs["attention_mask"].cpu(), batch_texts, answer_marker, answer_suffix, span_kind
            )
            span_fallbacks += n_fb
            pool_mask = (span_mask.to(device) * inputs["attention_mask"]).to(torch.long)
        else:
            pool_mask = inputs["attention_mask"]
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states
        if layer == "end":
            layer_idx = len(hidden) - 1
        elif layer == "middle":
            layer_idx = len(hidden) // 2
        else:
            layer_idx = int(layer)
        last_hidden = hidden[layer_idx]
        if pooling == "mean":
            mask = pool_mask.unsqueeze(-1).to(last_hidden.dtype)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:  # "last": the final selected (non-pad / in-span) token
            arange = torch.arange(pool_mask.shape[1], device=device).unsqueeze(0)
            last_indices = (pool_mask * arange).argmax(dim=1)
            batch_indices = torch.arange(last_hidden.shape[0], device=device)
            pooled = last_hidden[batch_indices, last_indices]
        activations.append(pooled.detach().to(torch.float32).cpu())
    if answer_span and span_fallbacks:
        print(f"[answer-span] {span_fallbacks}/{len(texts)} prompts fell back to full sequence "
              f"(marker '{answer_marker}' missing or truncated away).")

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return torch.cat(activations, dim=0)


class LogisticProbe(torch.nn.Module):
    """Minimal copy of the original repo's LBFGS logistic probe."""

    def __init__(self, input_dim: int, device: torch.device):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1, device=device)
        self.linear.bias.data.zero_()
        self.linear.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    @torch.enable_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor, l2_penalty: float, max_iter: int) -> float:
        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
        )
        y = y.to(torch.float32)
        loss = torch.inf

        def closure():
            nonlocal loss
            optimizer.zero_grad()
            logits = self(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            reg_loss = loss + l2_penalty * self.linear.weight.square().sum()
            reg_loss.backward()
            return float(reg_loss)

        optimizer.step(closure)
        return float(loss)

    @torch.no_grad()
    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(x))


def fit_probe(x: torch.Tensor, y: torch.Tensor, l2_penalty: float, max_iter: int, device: torch.device) -> LogisticProbe:
    probe = LogisticProbe(x.shape[1], device)
    probe.fit(x.to(device), y.to(device), l2_penalty, max_iter)
    return probe


@torch.no_grad()
def predict_probe(probe: LogisticProbe, x: torch.Tensor, device: torch.device) -> np.ndarray:
    return probe.predict_prob(x.to(device)).detach().cpu().numpy()


def binary_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(np.mean(preds == labels)),
        "prob_label1_mean": float(np.mean(probs)),
        "pred_label1_rate": float(np.mean(preds)),
        "confidence_mean": float(np.mean(2.0 * np.abs(probs - 0.5))),
    }


def write_predictions(path: Path, split: Dataset, columns: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "source_id", "label", "text"]
    for name in columns:
        fieldnames.extend([f"{name}_prob_label1", f"{name}_pred", f"{name}_correct"])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        labels = np.array(split["labels"], dtype=int)
        for idx in range(len(split)):
            row = {
                "id": split[idx]["id"],
                "source_id": split[idx]["source_id"],
                "label": int(labels[idx]),
                "text": split[idx]["txt"],
            }
            for name, probs in columns.items():
                pred = int(probs[idx] >= 0.5)
                row[f"{name}_prob_label1"] = float(probs[idx])
                row[f"{name}_pred"] = pred
                row[f"{name}_correct"] = int(pred == labels[idx])
            writer.writerow(row)


def write_strong_train_weak_labels(path: Path, split: Dataset, weak_probs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.array(split["labels"], dtype=int)
    weak_preds = (weak_probs >= 0.5).astype(int)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "source_id",
                "label",
                "weak_prob_label1",
                "weak_confidence",
                "weak_label",
                "weak_correct",
                "text",
            ],
        )
        writer.writeheader()
        for idx in range(len(split)):
            writer.writerow(
                {
                    "id": split[idx]["id"],
                    "source_id": split[idx]["source_id"],
                    "label": int(labels[idx]),
                    "weak_prob_label1": float(weak_probs[idx]),
                    "weak_confidence": float(2.0 * abs(weak_probs[idx] - 0.5)),
                    "weak_label": int(weak_preds[idx]),
                    "weak_correct": int(weak_preds[idx] == labels[idx]),
                    "text": split[idx]["txt"],
                }
            )


def write_split_preview(path: Path, splits: SplitBundle) -> None:
    lines = []
    for name in ["weak_train", "strong_train", "val", "test"]:
        split = getattr(splits, name)
        lines.append(f"{name}: n={len(split)} label_mean={float(np.mean(split['labels'])):.6f}")
        if len(split):
            lines.append("example:")
            lines.append(split[0]["txt"])
            lines.append(f"label: {split[0]['labels']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits = load_paper_dream_splits(args.n_train, args.n_val, args.n_test, args.seed)
    write_split_preview(output_dir / "split_preview.txt", splits)
    if args.dry_run_splits:
        summary = {
            "dataset": "dream",
            "source": "paper_linear_probe_rerun",
            "dry_run_splits": True,
            "seed": args.seed,
            "requested_sizes": {
                "n_train": args.n_train,
                "n_val": args.n_val,
                "n_test": args.n_test,
            },
            "actual_sizes": {
                "weak_train": len(splits.weak_train),
                "strong_train": len(splits.strong_train),
                "val": len(splits.val),
                "test": len(splits.test),
            },
            "format": {
                "task": "binary candidate-answer correctness",
                "prompt": "dialogue + 'Q: {question} A: {candidate}'",
                "label": "1 if candidate is the original Dream answer, else 0",
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Wrote split preview to {output_dir / 'split_preview.txt'}")
        return

    all_texts: dict[str, list[str]] = {
        "weak_train": list(splits.weak_train["txt"]),
        "strong_train": list(splits.strong_train["txt"]),
        "test": list(splits.test["txt"]),
    }

    weak_acts = {
        name: extract_final_token_activations(
            args.weak_model,
            texts,
            device,
            args.torch_dtype,
            args.batch_size,
            args.max_length,
            f"extract weak activations: {name}",
        )
        for name, texts in all_texts.items()
    }
    strong_acts = {
        name: extract_final_token_activations(
            args.strong_model,
            texts,
            device,
            args.torch_dtype,
            args.batch_size,
            args.max_length,
            f"extract strong activations: {name}",
        )
        for name, texts in all_texts.items()
        if name in {"strong_train", "test"}
    }

    y_weak_train = torch.tensor(splits.weak_train["labels"], dtype=torch.float32)
    y_strong_train = torch.tensor(splits.strong_train["labels"], dtype=torch.float32)
    y_test_np = np.array(splits.test["labels"], dtype=int)

    weak_probe = fit_probe(weak_acts["weak_train"], y_weak_train, args.l2_penalty, args.max_iter, device)
    strong_probe = fit_probe(strong_acts["strong_train"], y_strong_train, args.l2_penalty, args.max_iter, device)

    weak_probs_strong_train = predict_probe(weak_probe, weak_acts["strong_train"], device)
    w2s_probe = fit_probe(
        strong_acts["strong_train"],
        torch.tensor(weak_probs_strong_train, dtype=torch.float32),
        args.l2_penalty,
        args.max_iter,
        device,
    )

    weak_probs_test = predict_probe(weak_probe, weak_acts["test"], device)
    strong_probs_test = predict_probe(strong_probe, strong_acts["test"], device)
    w2s_probs_test = predict_probe(w2s_probe, strong_acts["test"], device)

    weak_train_probs = predict_probe(weak_probe, weak_acts["weak_train"], device)
    weak_probs_strong_preds = (weak_probs_strong_train >= 0.5).astype(int)
    strong_train_labels_np = np.array(splits.strong_train["labels"], dtype=int)

    summary = {
        "dataset": "dream",
        "source": "paper_linear_probe_rerun",
        "weak_model": args.weak_model,
        "strong_model": args.strong_model,
        "seed": args.seed,
        "requested_sizes": {
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
        },
        "actual_sizes": {
            "weak_train": len(splits.weak_train),
            "strong_train": len(splits.strong_train),
            "val": len(splits.val),
            "test": len(splits.test),
        },
        "format": {
            "task": "binary candidate-answer correctness",
            "prompt": "dialogue + 'Q: {question} A: {candidate}'",
            "label": "1 if candidate is the original Dream answer, else 0",
        },
        "activation": {
            "model_class": "AutoModelForSequenceClassification",
            "layer": "final hidden layer",
            "token": "final non-padding token",
            "pooling": "none",
            "max_length": args.max_length,
        },
        "probe": {
            "type": "LBFGS logistic probe",
            "l2_penalty": args.l2_penalty,
            "max_iter": args.max_iter,
            "w2s_target": "weak-probe soft probabilities on strong_train",
        },
        "metrics": {
            "weak_probe_on_weak_train": binary_metrics(weak_train_probs, np.array(splits.weak_train["labels"], dtype=int)),
            "weak_probe_on_test": binary_metrics(weak_probs_test, y_test_np),
            "strong_gt_probe_on_test": binary_metrics(strong_probs_test, y_test_np),
            "full_w2s_probe_on_test": binary_metrics(w2s_probs_test, y_test_np),
        },
        "weak_label_diagnostics_on_strong_train": {
            "accuracy": float(np.mean(weak_probs_strong_preds == strong_train_labels_np)),
            "positive_rate": float(np.mean(weak_probs_strong_preds)),
            "soft_prob_label1_mean": float(np.mean(weak_probs_strong_train)),
            "confidence_mean": float(np.mean(2.0 * np.abs(weak_probs_strong_train - 0.5))),
        },
        "paper_alignment_audit": {
            "matches_original_dream_formatter": True,
            "matches_original_split_logic": True,
            "matches_original_final_token_activation": True,
            "matches_original_logistic_probe": True,
            "uses_linear_probe_not_lora": True,
            "model_note": (
                "The paper text describes Llama3 8B. The public repo config uses a Llama 3.1 8B "
                "identifier, and this script defaults to the accessible HF id "
                "`meta-llama/Llama-3.1-8B`; override --strong-model if a different Llama3 8B "
                "checkpoint is required."
            ),
            "known_engineering_difference": (
                "Uses raw Dream JSON through huggingface_hub because the current datasets package "
                "does not reliably load the old Dream dataset script; formatted fields are kept "
                "equivalent to the original repo's formatter."
            ),
            "not_this_script": (
                "This is not the targeted LoRA/generative yes-no extension; it is the "
                "paper-faithful Figure A1-style sanity check."
            ),
        },
        "elapsed_sec": time.time() - start,
    }

    write_predictions(
        output_dir / "test_predictions.csv",
        splits.test,
        {
            "weak": weak_probs_test,
            "strong_gt": strong_probs_test,
            "full_w2s": w2s_probs_test,
        },
    )
    write_strong_train_weak_labels(
        output_dir / "strong_train_weak_labels.csv",
        splits.strong_train,
        weak_probs_strong_train,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.save_activations:
        torch.save(
            {
                "weak_activations": weak_acts,
                "strong_activations": strong_acts,
                "weak_model": args.weak_model,
                "strong_model": args.strong_model,
                "seed": args.seed,
                "splits": {name: getattr(splits, name).to_pandas() for name in ["weak_train", "strong_train", "val", "test"]},
            },
            output_dir / "activations.pt",
        )

    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
