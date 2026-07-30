#!/usr/bin/env python3
"""Dream LoRA W2S experiment with paper-style data formatting.

This is the downstream-training counterpart to the paper-style linear-probe
rerun. It keeps the Dream split and candidate-answer formatting close to the
original repo, but trains/evaluates a causal LM with LoRA on yes/no targets.

The input text is the original-style candidate statement:

    dialogue

    Q: question A: candidate

For causal-LM scoring/training, we append a short answer slot asking whether the
candidate is correct. This keeps the task readable for a generative model while
avoiding the older verbose "Dialogue:/Candidate answer:" format.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from run_dream_paper_linear_probe import (
    SplitBundle,
    extract_final_token_activations,
    fit_probe,
    load_dream_3class_eval,
    predict_probe,
    resolve_device,
)
from run_dream_paper_residual_filtering import (
    fit_maps,
    hard_weak_label_balance,
    middle_residual_indices,
    random_balanced_indices,
    subset_summary,
)
from run_dream_w2s_baselines import (
    clear_memory,
    evaluate_yes_no,
    train_lora_model,
    write_predictions,
)
from representation_projection import representation_projection_scores
from mlp_alignment import mlp_alignment_scores
from contracts import SEED_OFFSETS
from cca_alignment import cca_alignment_scores
from mlp_rp import mlp_rp_scores, mlp_step1_rp_scores, linstep1_rp_scores
from robust_linear import robust_linear_residual_scores
from lda_alignment import lda_alignment_scores
from excess_loss import excess_loss_kway_scores, option_entropy
from contracts import LoraExample
from paper_style_datasets import (
    _generic_mc_eval,
    _quail_rows,
    _riddlesense_rows,
    _scinli_rows,
    load_anli_multichoice_eval,
    load_aquarat_multichoice_eval,
    load_conjnli_multichoice_eval,
    load_control_multichoice_eval,
    load_fevernli_multichoice_eval,
    load_hellaswag_multichoice_eval,
    load_imppres_multichoice_eval,
    load_logiqa2_multichoice_eval,
    load_paper_style_splits,
    load_reclor_multichoice_eval,
    load_sciq_multichoice_eval,
    load_wanli_multichoice_eval,
    split_labels,
    split_texts,
    to_lora_examples,
)
from paper_style_report import (
    accuracy_3class,
    build_summary,
    auroc_from_rows,
    eval_rows_from_probs,
    format_summary,
    prior_matched_accuracy,
    summarize_train_subset,
    write_eval3_rows,
    write_knn_diagnostics,
    write_strong_train_labels,
    write_subset_csv,
    write_text_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dream", "sciq", "paws", "anli", "hellaswag", "reclor", "logiqa2", "wanli", "control", "fevernli", "conjnli", "imppres", "aquarat", "quail", "riddlesense", "scinli"], default="dream")
    parser.add_argument(
        "--anli-round",
        choices=["r1", "r2", "r3"],
        default="r2",
        help="For ANLI only: which adversarial round to use (default r2, matching the survey).",
    )
    parser.add_argument(
        "--sciq-use-support",
        action="store_true",
        help=(
            "For SciQ only, include the support passage in the prompt. "
            "Default false matches the original datacentric_w2s SciQ formatter."
        ),
    )
    parser.add_argument("--weak-model", default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--strong-model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--output-dir", default="results/dream_paper_style_lora/dream_seed42")
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-val", type=int, default=1_000)
    parser.add_argument("--n-test", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="Optional seed for LoRA initialization/training only. Keeps data splits and filters fixed.",
    )
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--activation-max-length", type=int, default=None)
    parser.add_argument("--answer-suffix", default="\nIs the candidate answer correct? Answer:")
    parser.add_argument("--weak-batch-size", type=int, default=4)
    parser.add_argument("--activation-batch-size", type=int, default=1)
    parser.add_argument("--l2-penalty", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--strong-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=100)
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="If >0, train each run for this many FULL passes over its actual subset "
        "(no-filter -> all data, f50 -> the real 50%%), setting max_train_steps per run = "
        "epochs * ceil(len(subset)/(batch*grad_accum)). Overrides the fixed compute-matched "
        "--max-train-steps cap.",
    )
    parser.add_argument("--dump-acts", default="",
                        help="npz path: dump strong_train weak/strong reps + weak preds/probs + gt "
                             "for offline method development (alignment scores, diagnostics)")
    parser.add_argument("--custom-scores-npz", default="",
                        help="npz with precomputed selection scores (keys cs1..cs4 + weak_preds guard)")
    parser.add_argument("--mlpalign-bottleneck", type=int, default=0,
                        help="bottleneck width for the mlpalign net (0 = none; 06/23 sketch used a narrow one)")
    parser.add_argument("--mlpalign-insample", action="store_true",
                        help="fit the mlpalign net in-sample (train=test, 06/23 sketch protocol)")
    parser.add_argument("--mlpalign-epochs", type=int, default=150,
                        help="training epochs for the mlpalign score MLP (25 = heldout-optimal "
                             "early stop from the 07/22 diagnostics)")
    parser.add_argument("--dump-el", default="",
                        help="npz path: dump per-example el/weak-entropy scores + weak preds/probs + gt "
                             "(for the el distribution figure); forces el computation")
    parser.add_argument("--mlpstep1-wd", type=float, default=30.0,
                        help="weight decay for the mlpstep1 weak-side MLP (sweep peak = 30)")
    parser.add_argument("--rp-reg", type=float, default=0.1,
                        help="Kernel ridge reg for the representation-projection score (rp_high/rp_low).")
    parser.add_argument("--rp-components", type=int, default=0,
                        help="PCA components for the representation-projection score (0 = use all).")
    parser.add_argument("--weight-temperature", type=float, default=1.0,
                        help="Temperature for the continuous-weighting runs (rp_weighted / el_weighted / "
                             "conf_weighted): each example's loss is weighted by softmax(score/T), rescaled "
                             "to mean 1. T->inf == no-filter (uniform), T->0 == hard top-selection.")
    parser.add_argument("--score-base-on-train", action="store_true",
                        help="Also score the untuned strong (base) model's yes/no P(correct) on each "
                             "strong_train example and write it to strong_train_labels.csv "
                             "(base_prob_correct / base_confidence). One extra base forward pass; used "
                             "by analyze_rp.py to test whether rp selects strong-base-learnable points.")
    parser.add_argument("--representation-layer", choices=["end", "middle"], default="end",
                        help="Transformer layer whose final-token activations feed the kNN and RP "
                             "selectors. 'end' (default) = last hidden layer. 'middle' = "
                             "len(hidden_states)//2, a training control. The weak probe / confidence "
                             "and the L2-residual maps always stay on the end layer so the weak "
                             "labels are held fixed across the comparison.")
    parser.add_argument("--representation-pooling", choices=["last", "mean"], default="last",
                        help="Token pooling for the kNN/RP representations: 'last' (default) = the "
                             "final non-pad token; 'mean' = average over the non-pad tokens (a "
                             "control). The weak probe stays on last-token regardless.")
    parser.add_argument("--representation-span", choices=["full", "answer", "context"], default="full",
                        help="Which tokens of each prompt the kNN/RP representation pools over: "
                             "'full' (default, settled choice) = the whole prompt (question + answer); "
                             "'answer' = only 'A: {candidate}' (mask the question prefix and the yes/no "
                             "suffix); 'context' = everything up to and including the 'A:' marker but "
                             "EXCLUDING the candidate word (Premise/Hypothesis/Q/A:). 'answer'/'context' "
                             "need a fast tokenizer. The weak probe/confidence stay on the full prompt "
                             "so the weak labels are held fixed across the comparison.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay", choices=["linear", "none"], default="linear",
                        help="LR schedule after warmup: linear decay to 0 (default) or constant.")
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-adapters", action="store_true")
    parser.add_argument(
        "--runs",
        default="base,ground_truth,weak_label,middle_balanced,random_balanced",
        help=(
            "Comma-separated runs. Options: base, ground_truth, weak_label, "
            "middle_unbalanced, middle_balanced, confidence_middle_unbalanced, "
            "confidence_middle_balanced, confidence_high_unbalanced, "
            "confidence_high_balanced, knn_middle_unbalanced, "
            "knn_middle_balanced, knn_mixed_unbalanced, "
            "knn_mixed_balanced, committee_agree_unbalanced, committee_agree_balanced, "
            "committee_disagree_unbalanced, committee_disagree_balanced, "
            "random_unbalanced, random_balanced."
        ),
    )
    parser.add_argument("--residual-keep-middle-frac", type=float, default=0.5)
    parser.add_argument("--confidence-keep-frac", type=float, default=0.5)
    parser.add_argument("--confidence-cleanhard-easy-drop", type=float, default=0.10,
                        help="Percentile-range knob for the confidence 'clean-but-hard' band "
                             "(confidence_cleanhard_*): fraction of the EASIEST (highest-confidence) "
                             "points to drop off the top before keeping the next keep_frac below. "
                             "0.10 + frac 0.5 -> keep the [40%%, 90%%] confidence band.")
    parser.add_argument("--quantile-bins", type=int, default=0,
                        help="If >1, add disjoint equal-size quantile-bin selectors "
                             "{rp,el,confidence}_q{i}_balanced (q0 = highest-score bin .. q{K-1} = "
                             "lowest) for a dose-response / separation analysis. All bins are the "
                             "same size, isolating the score's discriminability from sample size.")
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-keep-middle-frac", type=float, default=0.5)
    parser.add_argument("--knn-mixed-center", type=float, default=0.5)
    parser.add_argument(
        "--knn-reference-cross-fit",
        action="store_true",
        help=(
            "Label weak_train reference points weak-correct/weak-wrong using "
            "cross-fitted (out-of-fold) weak-probe predictions instead of "
            "in-sample predictions. Avoids the kNN saturation that happens when "
            "the probe is near-perfect on its own training set."
        ),
    )
    parser.add_argument("--cross-fit-folds", type=int, default=5)
    parser.add_argument(
        "--committee-members",
        type=int,
        default=8,
        help=(
            "Number of bootstrap weak probes in the query-by-committee disagreement "
            "selector (committee_agree / committee_disagree)."
        ),
    )
    parser.add_argument("--committee-keep-frac", type=float, default=0.5)
    parser.add_argument(
        "--committee-keep-fracs",
        default="",
        help=(
            "Comma-separated keep fractions for a committee selection sweep, e.g. "
            "'0.1,0.2,0.3,0.45,0.6,0.8'. For each fraction f (pct=round(100f)) it adds runs "
            "committee_agree_balanced_f{pct}, committee_agree_unbalanced_f{pct}, "
            "committee_disagree_balanced_f{pct}, and a matched random_balanced_f{pct}. "
            "Empty = no sweep."
        ),
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help=(
            "Compute weak_train reference accuracy and kNN saturation (fraction "
            "of strong_train points whose k nearest neighbors are all weak-correct) "
            "under in-sample vs cross-fitted reference labels, then exit before any "
            "LoRA training. Cheap check for whether cross-fitting de-saturates kNN."
        ),
    )
    parser.add_argument("--random-control-count", type=int, default=3)
    parser.add_argument("--random-control-size", type=int, default=None)
    parser.add_argument("--random-unbalanced-size", type=int, default=None)
    parser.add_argument("--ridge-values", default="100.0")
    parser.add_argument("--pca-dims", default="")
    parser.add_argument("--best-by", choices=["heldout_mean", "heldout_median"], default="heldout_median")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument(
        "--eval-3class",
        action="store_true",
        help="(Dream) also report per-question multi-class accuracy: score every candidate "
        "answer of each test question and argmax P(correct).",
    )
    parser.add_argument(
        "--n-eval-questions",
        type=int,
        default=2041,
        help="Number of Dream test questions for --eval-3class (each expands to its candidate answers). "
        "Default 2041 = the full Dream test split.",
    )
    return parser.parse_args()


def requested_runs(args: argparse.Namespace) -> list[str]:
    runs = [item.strip() for item in args.runs.split(",") if item.strip()]
    allowed = {
        "base",
        "ground_truth",
        "weak_label",
        "weak_label_balanced",
        "weak_correct_only",
        "weak_correct_only_balanced",
        "weak_strong_agree_balanced",
        "weak_strong_disagree_balanced",
        "rp_weighted",
        "el_weighted",
        "conf_weighted",
        "curriculum_easy",
        "curriculum_hard",
        "middle_unbalanced",
        "middle_balanced",
        "confidence_middle_unbalanced",
        "confidence_middle_balanced",
        "confidence_high_unbalanced",
        "confidence_high_balanced",
        "knn_middle_unbalanced",
        "knn_middle_balanced",
        "knn_mixed_unbalanced",
        "knn_mixed_balanced",
        "knn_high_unbalanced",
        "knn_high_balanced",
        "committee_agree_unbalanced",
        "committee_agree_balanced",
        "committee_disagree_unbalanced",
        "committee_disagree_balanced",
        "random_unbalanced",
        "random_balanced",
    }
    def _is_frac_run(name: str) -> bool:
        for pref in (
            "committee_agree_balanced_f",
            "committee_agree_unbalanced_f",
            "committee_disagree_balanced_f",
            "knn_high_balanced_f",
            "knn_low_balanced_f",
            "knn_mixed_balanced_f",
            "confidence_high_balanced_f",
            "confidence_low_balanced_f",
            "confidence_cleanhard_balanced_f",
            "curriculum_confhigh_f",
            "curriculum_rphigh_f",
            "curriculum_rphigh_byrp_f",
            "curriculum_rphigh_byrp_asc_f",
            "rp_high_balanced_f",
            "rp_low_balanced_f",
            "rp_conf_high_balanced_f",
            "mlpalign_high_balanced_f",
            "mlpalign_low_balanced_f",
            "cca_high_balanced_f",
            "cca_low_balanced_f",
            "mlprp_high_balanced_f",
            "mlprp_low_balanced_f",
            "mlpstep1_high_balanced_f",
            "mlpstep1_low_balanced_f",
            "linstep1_high_balanced_f",
            "linstep1_low_balanced_f",
            "softrp_high_balanced_f",
            "softrp_low_balanced_f",
            "cs1_high_balanced_f", "cs1_low_balanced_f",
            "cs2_high_balanced_f", "cs2_low_balanced_f",
            "cs3_high_balanced_f", "cs3_low_balanced_f",
            "cs4_high_balanced_f", "cs4_low_balanced_f",
            "lda_high_balanced_f",
            "lda_low_balanced_f",
            "ccarobust_high_balanced_f",
            "ccarobust_low_balanced_f",
            "l2robust_high_balanced_f",
            "l2robust_low_balanced_f",
            "l2resid_high_balanced_f",
            "l2resid_low_balanced_f",
            "el_high_balanced_f",
            "el_low_balanced_f",
            "conf_entropy_high_balanced_f",
            "conf_entropy_low_balanced_f",
            "random_balanced_f",
            "random_unbalanced_f",
        ):
            if name.startswith(pref) and name[len(pref):].isdigit():
                return True
        return False

    def _is_quantile_run(name: str) -> bool:
        for sig in ("rp_q", "el_q", "confidence_q"):
            if name.startswith(sig) and name[len(sig):len(sig) + 1].isdigit() and name.endswith("_balanced"):
                return True
        return False

    unknown = sorted(r for r in set(runs) - allowed if not _is_frac_run(r) and not _is_quantile_run(r))
    if unknown:
        raise SystemExit(f"Unknown run(s): {', '.join(unknown)}")
    return runs


def run_knn_saturation_diagnostics(
    args: argparse.Namespace,
    splits: SplitBundle,
    texts: dict[str, list[str]],
    device: torch.device,
    output_dir: Path,
) -> None:
    """Run the lightweight kNN saturation check and exit before mapping/LoRA.

    This intentionally does less than the full experiment: weak activations are
    needed only on weak_train, and strong activations are needed only on
    weak_train + strong_train. It does not extract test activations, fit
    weak-to-strong maps, or train LoRA adapters.
    """

    weak_train_labels = split_labels(splits.weak_train)
    weak_train_acts = extract_final_token_activations(
        args.weak_model,
        texts["weak_train"],
        device,
        args.torch_dtype,
        args.activation_batch_size,
        args.activation_max_length,
        "extract weak_train activations",
    )
    weak_probe = fit_probe(
        weak_train_acts,
        torch.tensor(weak_train_labels, dtype=torch.float32),
        args.l2_penalty,
        args.max_iter,
        device,
    )
    weak_probs_weak_train = predict_probe(weak_probe, weak_train_acts, device)
    insample_ref_correct = ((weak_probs_weak_train >= 0.5).astype(int) == weak_train_labels).astype(int)

    cross_fit_probs = cross_fitted_weak_probs(
        weak_train_acts,
        weak_train_labels,
        args.cross_fit_folds,
        args.l2_penalty,
        args.max_iter,
        device,
        args.seed,
    )
    crossfit_ref_correct = ((cross_fit_probs >= 0.5).astype(int) == weak_train_labels).astype(int)

    strong_diag_texts = texts["weak_train"] + texts["strong_train"]
    strong_diag_acts = extract_final_token_activations(
        args.strong_model,
        strong_diag_texts,
        device,
        args.torch_dtype,
        args.activation_batch_size,
        args.activation_max_length,
        "extract strong weak_train+strong_train activations",
    )
    n_weak_train = len(texts["weak_train"])
    strong_weak_train_acts = strong_diag_acts[:n_weak_train]
    strong_strong_train_acts = strong_diag_acts[n_weak_train:]

    def _knn_saturation(reference_correct: np.ndarray) -> dict[str, float]:
        stats = compute_weak_train_knn_stats(
            strong_weak_train_acts,
            strong_strong_train_acts,
            reference_correct,
            args.knn_k,
        )
        rates = stats["knn_correct_rate"]
        return {
            "reference_accuracy": float(np.mean(reference_correct)),
            "all_neighbors_weak_correct_fraction": float(np.mean(rates >= 1.0 - 1e-9)),
            "correct_neighbor_rate_mean": float(np.mean(rates)),
            "correct_neighbor_rate_median": float(np.median(rates)),
            "correct_neighbor_rate_min": float(np.min(rates)),
            "correct_neighbor_rate_max": float(np.max(rates)),
        }

    diagnostics = {
        "dataset": args.dataset,
        "knn_k": args.knn_k,
        "cross_fit_folds": args.cross_fit_folds,
        "sizes": {
            "weak_train": len(splits.weak_train),
            "strong_train": len(splits.strong_train),
            "test": len(splits.test),
        },
        "in_sample": _knn_saturation(insample_ref_correct),
        "cross_fitted": _knn_saturation(crossfit_ref_correct),
        "note": "diagnostics-only skips test activations, representation mapping, and LoRA training",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "knn_saturation_diagnostic.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(
        f"[diagnostics-only] {args.dataset}: "
        f"reference acc in-sample={diagnostics['in_sample']['reference_accuracy']:.3f} "
        f"-> cross-fit={diagnostics['cross_fitted']['reference_accuracy']:.3f}; "
        f"kNN all-{args.knn_k}-correct fraction "
        f"in-sample={diagnostics['in_sample']['all_neighbors_weak_correct_fraction']:.3f} "
        f"-> cross-fit={diagnostics['cross_fitted']['all_neighbors_weak_correct_fraction']:.3f}"
    )
    print(f"  wrote {output_dir / 'knn_saturation_diagnostic.json'}")

    del weak_train_acts, strong_diag_acts, strong_weak_train_acts, strong_strong_train_acts
    clear_memory()


def slice_activations(all_activations: torch.Tensor, sizes: dict[str, int]) -> dict[str, torch.Tensor]:
    out = {}
    start = 0
    for name in ["weak_train", "strong_train", "test"]:
        end = start + sizes[name]
        out[name] = all_activations[start:end]
        start = end
    return out


def extract_all_activations(
    model_name: str,
    texts_by_split: dict[str, list[str]],
    device: torch.device,
    dtype_arg: str,
    batch_size: int,
    max_length: int | None,
    desc: str,
    layer: str | int = "end",
    pooling: str = "last",
    answer_span: bool = False,
    answer_suffix: str = "",
    span_kind: str = "answer",
) -> dict[str, torch.Tensor]:
    sizes = {name: len(texts_by_split[name]) for name in ["weak_train", "strong_train", "test"]}
    all_texts = texts_by_split["weak_train"] + texts_by_split["strong_train"] + texts_by_split["test"]
    acts = extract_final_token_activations(
        model_name,
        all_texts,
        device,
        dtype_arg,
        batch_size,
        max_length,
        desc,
        layer=layer,
        pooling=pooling,
        answer_span=answer_span,
        answer_suffix=answer_suffix,
        span_kind=span_kind,
    )
    return slice_activations(acts, sizes)


def weak_label_balance(
    examples: list[LoraExample],
    labels: list[int],
    seed: int,
) -> tuple[list[LoraExample], list[int], np.ndarray]:
    indices = np.arange(len(examples))
    weak_preds = np.array(labels, dtype=int)
    selected = hard_weak_label_balance(indices, weak_preds, seed)
    return [examples[int(i)] for i in selected], [int(labels[int(i)]) for i in selected], selected


def scores_to_weights(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Continuous per-example loss weights from a selection score (loss-adaptation).

    z-score the scores, divide by the temperature, softmax over the dataset, then rescale to
    mean 1 (sum N) so the loss magnitude / effective data size is preserved. temperature -> inf
    gives uniform weights (== no-filter); temperature -> 0 concentrates the weight on the
    top-scoring examples (== hard top-selection). So one knob interpolates between the two,
    the smooth counterpart of the 0/1 *_high band.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    sd = s.std()
    z = (s - s.mean()) / (sd if sd > 1e-8 else 1.0)
    z = z / max(float(temperature), 1e-6)
    z = z - z.max()
    w = np.exp(z)
    return w * (len(w) / w.sum())


def score_band_indices(scores: np.ndarray, keep_frac: float, mode: str) -> tuple[np.ndarray, dict[str, float | str]]:
    if not 0.0 < keep_frac <= 1.0:
        raise ValueError("keep_frac must be in (0, 1].")
    if mode not in {"middle", "high", "low"}:
        raise ValueError(f"Unknown score band mode: {mode}")

    order = np.argsort(scores)
    n_keep = max(1, int(round(len(order) * keep_frac)))
    if mode == "middle":
        start = (len(order) - n_keep) // 2
        end = start + n_keep
        kept = order[start:end]
        dropped_low = start
        dropped_high = len(order) - end
    elif mode == "low":
        kept = order[:n_keep]
        dropped_low = 0
        dropped_high = len(order) - n_keep
    else:
        kept = order[-n_keep:]
        dropped_low = len(order) - n_keep
        dropped_high = 0

    kept_scores = scores[kept]
    return kept, {
        "mode": mode,
        "matched_examples": int(len(order)),
        "kept_examples": int(len(kept)),
        "keep_frac": float(keep_frac),
        "dropped_low_examples": int(dropped_low),
        "dropped_high_examples": int(dropped_high),
        "kept_score_min": float(np.min(kept_scores)),
        "kept_score_max": float(np.max(kept_scores)),
        "kept_score_mean": float(np.mean(kept_scores)),
        "kept_score_median": float(np.median(kept_scores)),
    }


def score_percentile_band_indices(scores: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    """Keep the examples whose score falls in the [lo_pct, hi_pct] percentile band (ascending
    sort). E.g. (0.40, 0.90) drops the bottom 40% and the top 10%. This is the percentile-range
    knob for the confidence 'clean-but-hard' band: drop the noisy-low tail (wrong labels) AND the
    trivially-easy-high tail (redundant, already learnt)."""
    if not 0.0 <= lo_pct < hi_pct <= 1.0:
        raise ValueError(f"need 0 <= lo_pct < hi_pct <= 1, got ({lo_pct}, {hi_pct})")
    order = np.argsort(scores)
    n = len(order)
    start = int(round(n * lo_pct))
    end = max(start + 1, int(round(n * hi_pct)))
    return order[start:end]


def score_closest_indices(
    scores: np.ndarray,
    keep_frac: float,
    center: float,
    mode_name: str,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Keep examples whose score is closest to a target value.

    For kNN filtering, the intended "overlap-like" points are not
    necessarily the middle quantile of the score distribution. They are points
    whose neighborhoods are mixed between weak-correct and weak-wrong reference
    examples. With a correct-neighbor-rate score, that means closest to 0.5.
    """

    if not 0.0 < keep_frac <= 1.0:
        raise ValueError("keep_frac must be in (0, 1].")

    order = np.argsort(np.abs(scores - center))
    n_keep = max(1, int(round(len(order) * keep_frac)))
    kept = order[:n_keep]
    kept_scores = scores[kept]
    return kept, {
        "mode": mode_name,
        "matched_examples": int(len(order)),
        "kept_examples": int(len(kept)),
        "keep_frac": float(keep_frac),
        "center": float(center),
        "kept_score_min": float(np.min(kept_scores)),
        "kept_score_max": float(np.max(kept_scores)),
        "kept_score_mean": float(np.mean(kept_scores)),
        "kept_score_median": float(np.median(kept_scores)),
        "kept_abs_distance_to_center_mean": float(np.mean(np.abs(kept_scores - center))),
        "kept_abs_distance_to_center_median": float(np.median(np.abs(kept_scores - center))),
    }


def cross_fitted_weak_probs(
    activations: torch.Tensor,
    labels: np.ndarray,
    folds: int,
    l2_penalty: float,
    max_iter: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    """Out-of-fold weak-probe probabilities on the weak_train reference set.

    Each weak_train point is scored by a probe that was NOT trained on it: the
    set is split into ``folds`` folds, and for each fold a probe is fit on the
    other folds and used to predict the held-out fold. This gives honest
    weak-correct / weak-wrong reference labels for the kNN filter and avoids the
    in-sample saturation that occurs when the probe predicts its own training
    data (an overfit probe labels almost every reference point weak-correct,
    leaving no weak-wrong neighbors for the mixed-neighborhood signal).
    """
    n = activations.shape[0]
    folds = max(2, min(folds, n))
    order = np.random.default_rng(seed).permutation(n)
    labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
    probs = np.zeros(n, dtype=np.float32)
    for held_out in np.array_split(order, folds):
        held_out_t = torch.as_tensor(held_out, dtype=torch.long)
        train_mask = torch.ones(n, dtype=torch.bool)
        train_mask[held_out_t] = False
        train_idx_t = torch.nonzero(train_mask, as_tuple=False).squeeze(1)
        probe = fit_probe(
            activations[train_idx_t],
            labels_t[train_idx_t],
            l2_penalty,
            max_iter,
            device,
        )
        probs[held_out] = predict_probe(probe, activations[held_out_t], device)
    return probs


def committee_disagreement_on_strong_train(
    weak_train_activations: torch.Tensor,
    weak_train_labels: np.ndarray,
    strong_train_activations: torch.Tensor,
    members: int,
    l2_penalty: float,
    max_iter: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Query-by-committee disagreement for each strong_train point.

    Trains ``members`` weak probes on bootstrap resamples of weak_train and
    predicts every strong_train point with each one. The committee's per-point
    disagreement (standard deviation of the member probabilities) operationalizes
    the active-learning "region of disagreement": high disagreement means the
    plausible weak probes cannot agree on the label (boundary / unreliable weak
    label), low disagreement means they concur (reliable weak label). strong_train
    is out-of-sample for every member, so the estimate is honest by construction.
    """
    n = weak_train_activations.shape[0]
    members = max(2, members)
    rng = np.random.default_rng(seed)
    labels_t = torch.as_tensor(np.asarray(weak_train_labels), dtype=torch.float32)
    member_probs = []
    for _ in range(members):
        boot = rng.integers(0, n, size=n)
        boot_t = torch.as_tensor(boot, dtype=torch.long)
        probe = fit_probe(
            weak_train_activations[boot_t],
            labels_t[boot_t],
            l2_penalty,
            max_iter,
            device,
        )
        member_probs.append(predict_probe(probe, strong_train_activations, device))
    stacked = np.stack(member_probs, axis=0)
    mean_prob = stacked.mean(axis=0)
    disagreement = stacked.std(axis=0)
    return mean_prob, disagreement


def compute_weak_train_knn_stats(
    reference_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    reference_weak_correct: np.ndarray,
    k: int,
) -> dict[str, np.ndarray]:
    if k <= 0:
        raise ValueError("--knn-k must be positive.")
    if len(reference_embeddings) == 0:
        raise ValueError("Need at least one weak_train reference embedding.")
    k = min(k, len(reference_embeddings))

    ref = torch.nn.functional.normalize(reference_embeddings.to(torch.float32), p=2, dim=1)
    query = torch.nn.functional.normalize(query_embeddings.to(torch.float32), p=2, dim=1)
    sims = query @ ref.T
    top_values, top_indices = torch.topk(sims, k=k, dim=1)

    correct = torch.tensor(reference_weak_correct.astype(np.float32))
    neighbor_correct = correct[top_indices]
    correct_count = neighbor_correct.sum(dim=1)
    correct_rate = correct_count / float(k)
    return {
        "neighbor_indices": top_indices.cpu().numpy(),
        "neighbor_cosine_mean": top_values.mean(dim=1).cpu().numpy(),
        "neighbor_cosine_min": top_values.min(dim=1).values.cpu().numpy(),
        "knn_correct_count": correct_count.cpu().numpy(),
        "knn_correct_rate": correct_rate.cpu().numpy(),
    }


def random_unbalanced_indices(n_examples: int, size: int, seed: int) -> np.ndarray:
    size = min(size, n_examples)
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(n_examples), size=size, replace=False)


def run_lora_eval(
    args: argparse.Namespace,
    run_name: str,
    train_examples: list[LoraExample],
    train_labels: list[int],
    eval_examples: list[LoraExample],
    output_dir: Path,
    eval3_examples: list[LoraExample] | None = None,
    curriculum_order=None,
    sample_weights=None,
) -> tuple[dict, list[dict]]:
    if args.train_seed is not None:
        torch.manual_seed(args.train_seed)
        np.random.seed(args.train_seed)
        random.seed(args.train_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.train_seed)
    model, tokenizer, report = train_lora_model(
        args, train_examples, train_labels, run_name, output_dir,
        curriculum_order=curriculum_order, sample_weights=sample_weights,
    )
    eval_summary, rows = evaluate_yes_no(
        model,
        tokenizer,
        eval_examples,
        args.strong_batch_size,
        args.device,
        args.max_length,
        f"eval {run_name}",
    )
    eval_summary["auroc"] = auroc_from_rows(rows)
    eval_summary["accuracy_prior_matched"] = prior_matched_accuracy(rows)
    if eval3_examples:
        _, rows3 = evaluate_yes_no(
            model,
            tokenizer,
            eval3_examples,
            args.strong_batch_size,
            args.device,
            args.max_length,
            f"eval3 {run_name}",
        )
        eval_summary["accuracy_3class"] = accuracy_3class(eval3_examples, rows3)
        write_eval3_rows(output_dir / f"eval3_{run_name}.csv", eval3_examples, rows3)
    del model
    del tokenizer
    clear_memory()
    return {
        **report,
        "train_seed": args.train_seed,
        "train_subset": summarize_train_subset(train_examples, train_labels),
        "eval": eval_summary,
    }, rows


def _load_causal_lm_for_scoring(model_name: str, args: argparse.Namespace, device):
    """Load an arbitrary model name as a causal LM + tokenizer for yes/no scoring."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run_dream_w2s_baselines import resolve_dtype

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=resolve_dtype(args.torch_dtype), low_cpu_mem_usage=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.to(device)
    return model, tokenizer


def compute_el_kway_scores(
    args: argparse.Namespace, strong_train_ds, device
) -> tuple[np.ndarray, np.ndarray]:
    """Excess-loss / learnability score per strong_train question (in row order):
    H_weak(options) - H_strong(options), where each model's option distribution is the
    softmax of its yes/no P(correct) over the K answer options. Loads the weak and the
    untuned-strong model once each (no probe); independent of LoRA training, so computed
    once per data seed and reused across train seeds and bands.

    Returns (el_scores, weak_option_entropy): both row-aligned to strong_train. The weak
    option entropy H_weak(options) is the settled single-model (weak-only) confidence
    variant -- low entropy = the weak teacher is sure which option is the answer.
    el is exactly H_weak - H_strong, so this is el with the strong term set to 0; it backs
    the conf_entropy_* selectors. Computed here to reuse the weak model pass.
    """
    from run_dream_w2s_baselines import load_strong_model_and_tokenizer

    if "mc_options" not in strong_train_ds.column_names:
        raise SystemExit(
            "el_* / conf_entropy_* selectors need per-question options; this dataset's "
            "formatter does not emit 'mc_options'. Add it to the format_* function before "
            "requesting el or conf_entropy runs."
        )
    source_ids = list(strong_train_ds["source_id"])
    mc_options = list(strong_train_ds["mc_options"])
    opt_examples = [
        LoraExample(
            id=f"{src}::opt{k}",
            source_id=src,
            text=str(otext),
            label=0,
            answer_suffix=args.answer_suffix,
        )
        for src, opts in zip(source_ids, mc_options)
        for k, otext in enumerate(opts)
    ]
    el_batch = max(int(getattr(args, "strong_batch_size", 1)), 8)

    def _option_probs_by_q(model, tokenizer, desc):
        _, rows = evaluate_yes_no(
            model, tokenizer, opt_examples, el_batch, device, args.max_length, desc
        )
        by_q: dict[str, list[float]] = {}
        for ex, row in zip(opt_examples, rows):
            by_q.setdefault(ex.source_id, []).append(float(row["prob_label1"]))
        return by_q

    weak_model, weak_tok = _load_causal_lm_for_scoring(args.weak_model, args, device)
    weak_by_q = _option_probs_by_q(weak_model, weak_tok, "el K-option (weak)")
    del weak_model
    clear_memory()

    strong_model, strong_tok = load_strong_model_and_tokenizer(args, trainable_lora=False)
    strong_by_q = _option_probs_by_q(strong_model, strong_tok, "el K-option (base strong)")
    del strong_model
    clear_memory()

    el_by_q = excess_loss_kway_scores(weak_by_q, strong_by_q)
    weak_entropy_by_q = {q: option_entropy(wp) for q, wp in weak_by_q.items()}
    el_arr = np.asarray([el_by_q.get(src, 0.0) for src in source_ids], dtype=np.float64)
    weak_entropy_arr = np.asarray(
        [weak_entropy_by_q.get(src, 0.0) for src in source_ids], dtype=np.float64
    )
    # Loss-difference el (the RHO-LOSS-style form): per question, take the weak model's own
    # pick yhat = argmax of its option distribution, and compare surprisals at that pick:
    # elloss = -log q_s[yhat] + log q_w[yhat] = loss_strong - loss_weak, no gold needed.
    # (by_q lists may hold duplicate K-blocks when a question spans several rows; slice the
    # first K -- forward passes are deterministic, so the blocks are identical copies.)
    mc_correct = (list(strong_train_ds["mc_correct"])
                  if "mc_correct" in strong_train_ds.column_names else [-1] * len(source_ids))
    elloss_by_q, pick_by_q = {}, {}
    kk = {src: len(opts) for src, opts in zip(source_ids, mc_options)}
    for src in set(source_ids):
        K = max(kk[src], 1)
        qw = np.clip(np.asarray(weak_by_q[src][:K], dtype=np.float64), 1e-12, None)
        qs = np.clip(np.asarray(strong_by_q[src][:K], dtype=np.float64), 1e-12, None)
        qw, qs = qw / qw.sum(), qs / qs.sum()
        yhat = int(np.argmax(qw))
        pick_by_q[src] = yhat
        elloss_by_q[src] = float(np.log(qw[yhat]) - np.log(qs[yhat]))
    el_extras = {
        "elloss": np.asarray([elloss_by_q.get(src, 0.0) for src in source_ids], dtype=np.float64),
        "weak_pick": np.asarray([pick_by_q.get(src, -1) for src in source_ids], dtype=np.int64),
        "gold_opt": np.asarray(mc_correct, dtype=np.int64),
    }
    return el_arr, weak_entropy_arr, el_extras


def build_keep_fraction_arms(
    *,
    args,
    run_subsets: dict,
    curriculum_orders: dict,
    scores: dict,
    custom_scores: dict,
    committee_keep_fracs: list[float],
    committee_disagreement_strong: np.ndarray,
    weak_confidences_strong: np.ndarray,
    weak_preds_strong: np.ndarray,
    weak_labels: list,
    strong_examples: list,
    residuals: np.ndarray,
    knn_stats: dict,
    random_balanced_indices: np.ndarray,
    n_strong_examples: int,
) -> None:
    """Add the per-keep-fraction arms (the _f<pct> family) to run_subsets in place.

    Lifted out of main() unchanged. The loop mutates run_subsets and curriculum_orders
    and returns nothing, which is why it extracts cleanly: no value escapes it.
    The seeds are offset by the enumerate index fi, so the iteration order over
    committee_keep_fracs is part of the contract; do not sort or parallelize it.
    """
    rp_scores = scores.get('rp')
    el_scores = scores.get('el')
    weak_entropy_scores = scores.get('weak_entropy')
    cca_scores = scores.get('cca')
    cca_robust_scores = scores.get('cca_robust')
    l2robust_scores = scores.get('l2robust')
    mlprp_scores = scores.get('mlprp')
    mlpstep1_scores = scores.get('mlpstep1')
    linstep1_scores = scores.get('linstep1')
    mlpalign_scores = scores.get('mlpalign')
    softrp_scores = scores.get('softrp')
    lda_scores = scores.get('lda')

    for fi, frac in enumerate(committee_keep_fracs):
        pct = int(round(frac * 100))
        agree_idx, _ = score_band_indices(committee_disagreement_strong, frac, "low")
        disagree_idx, _ = score_band_indices(committee_disagreement_strong, frac, "high")
        agree_bal = hard_weak_label_balance(agree_idx, weak_preds_strong, args.seed + SEED_OFFSETS["kf_committee_agree"] + fi)
        disagree_bal = hard_weak_label_balance(disagree_idx, weak_preds_strong, args.seed + SEED_OFFSETS["kf_committee_disagree"] + fi)
        rand_bal = random_balanced_indices(
            weak_preds_strong, int(round(frac * n_strong_examples)), args.seed + SEED_OFFSETS["kf_committee_top"] + fi
        )
        run_subsets[f"committee_agree_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in agree_bal],
            [weak_labels[int(i)] for i in agree_bal],
        )
        run_subsets[f"committee_agree_unbalanced_f{pct}"] = (
            [strong_examples[int(i)] for i in agree_idx],
            [weak_labels[int(i)] for i in agree_idx],
        )
        run_subsets[f"committee_disagree_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in disagree_bal],
            [weak_labels[int(i)] for i in disagree_bal],
        )
        run_subsets[f"random_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in rand_bal],
            [weak_labels[int(i)] for i in rand_bal],
        )
        rand_unbal = random_unbalanced_indices(
            n_strong_examples, int(round(frac * n_strong_examples)), args.seed + SEED_OFFSETS["kf_random_matched"] + fi
        )
        run_subsets[f"random_unbalanced_f{pct}"] = (
            [strong_examples[int(i)] for i in rand_unbal],
            [weak_labels[int(i)] for i in rand_unbal],
        )
        knn_high_idx, _ = score_band_indices(knn_stats["knn_correct_rate"], frac, "high")
        knn_high_bal = hard_weak_label_balance(knn_high_idx, weak_preds_strong, args.seed + SEED_OFFSETS["kf_knn_high"] + fi)
        run_subsets[f"knn_high_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in knn_high_bal],
            [weak_labels[int(i)] for i in knn_high_bal],
        )
        conf_high_idx, _ = score_band_indices(weak_confidences_strong, frac, "high")
        conf_high_bal = hard_weak_label_balance(conf_high_idx, weak_preds_strong, args.seed + SEED_OFFSETS["kf_confidence_high"] + fi)
        run_subsets[f"confidence_high_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in conf_high_bal],
            [weak_labels[int(i)] for i in conf_high_bal],
        )
        # confidence_cleanhard: keep the clean-but-hard band -- drop the noisy-low tail (wrong
        # labels) AND the trivially-easy-high tail (redundant, already learnt). Keep the
        # [hi-frac, hi] confidence percentiles where hi = 1 - easy_drop. With frac 0.5 +
        # easy_drop 0.10 -> the [40%, 90%] band.
        ch_hi = min(1.0, 1.0 - float(args.confidence_cleanhard_easy_drop))
        ch_lo = max(0.0, ch_hi - frac)
        cleanhard_idx = score_percentile_band_indices(weak_confidences_strong, ch_lo, ch_hi)
        cleanhard_bal = hard_weak_label_balance(cleanhard_idx, weak_preds_strong, args.seed + SEED_OFFSETS["kf_confidence_cleanhard"] + fi)
        run_subsets[f"confidence_cleanhard_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in cleanhard_bal],
            [weak_labels[int(i)] for i in cleanhard_bal],
        )
        # curriculum_confhigh: curriculum learning ON the confidence_high clean subset only --
        # same examples as confidence_high_balanced, but trained easy->hard (most-confident first)
        # WITHIN the clean set instead of shuffled. Order indexes into the subset.
        run_subsets[f"curriculum_confhigh_f{pct}"] = (
            [strong_examples[int(i)] for i in conf_high_bal],
            [weak_labels[int(i)] for i in conf_high_bal],
        )
        ch_sub_conf = weak_confidences_strong[np.asarray(conf_high_bal, dtype=int)]
        curriculum_orders[f"curriculum_confhigh_f{pct}"] = list(np.argsort(-ch_sub_conf))
        conf_low_idx, _ = score_band_indices(weak_confidences_strong, frac, "low")
        conf_low_bal = hard_weak_label_balance(conf_low_idx, weak_preds_strong, args.seed + SEED_OFFSETS["kf_confidence_low"] + fi)
        run_subsets[f"confidence_low_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in conf_low_bal],
            [weak_labels[int(i)] for i in conf_low_bal],
        )
        knn_low_idx, _ = score_band_indices(knn_stats["knn_correct_rate"], frac, "low")
        knn_low_bal = hard_weak_label_balance(knn_low_idx, weak_preds_strong, args.seed + SEED_OFFSETS["kf_knn_low"] + fi)
        run_subsets[f"knn_low_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in knn_low_bal],
            [weak_labels[int(i)] for i in knn_low_bal],
        )
        rp_high_idx, _ = score_band_indices(rp_scores, frac, "high")
        rp_high_bal = hard_weak_label_balance(rp_high_idx, weak_preds_strong, args.seed + SEED_OFFSETS["rp_high"] + fi)
        run_subsets[f"rp_high_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in rp_high_bal],
            [weak_labels[int(i)] for i in rp_high_bal],
        )
        # curriculum_rphigh: SELECT-then-CURRICULUM -- the exact rp_high_balanced subset, but
        # trained in a fixed order instead of shuffled (paired comparison: same data, only the
        # order differs). Two orderings: by weak confidence (most-confident/easiest first) and
        # by rp score (most-overlap first, expanding outward).
        _rp_sub = np.asarray(rp_high_bal, dtype=int)
        run_subsets[f"curriculum_rphigh_f{pct}"] = run_subsets[f"rp_high_balanced_f{pct}"]
        curriculum_orders[f"curriculum_rphigh_f{pct}"] = list(np.argsort(-weak_confidences_strong[_rp_sub]))
        run_subsets[f"curriculum_rphigh_byrp_f{pct}"] = run_subsets[f"rp_high_balanced_f{pct}"]
        curriculum_orders[f"curriculum_rphigh_byrp_f{pct}"] = list(np.argsort(-rp_scores[_rp_sub]))
        # ascending variant: toxic low-rp points FIRST, cream last (recency for the good points)
        run_subsets[f"curriculum_rphigh_byrp_asc_f{pct}"] = run_subsets[f"rp_high_balanced_f{pct}"]
        curriculum_orders[f"curriculum_rphigh_byrp_asc_f{pct}"] = list(np.argsort(rp_scores[_rp_sub]))
        rp_low_idx, _ = score_band_indices(rp_scores, frac, "low")
        rp_low_bal = hard_weak_label_balance(rp_low_idx, weak_preds_strong, args.seed + SEED_OFFSETS["rp_low"] + fi)
        run_subsets[f"rp_low_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in rp_low_bal],
            [weak_labels[int(i)] for i in rp_low_bal],
        )
        if cca_scores is not None:
            cc_high_idx, _ = score_band_indices(cca_scores, frac, "high")
            cc_high_bal = hard_weak_label_balance(cc_high_idx, weak_preds_strong, args.seed + SEED_OFFSETS["cca_high"] + fi)
            run_subsets[f"cca_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in cc_high_bal],
                [weak_labels[int(i)] for i in cc_high_bal],
            )
            cc_low_idx, _ = score_band_indices(cca_scores, frac, "low")
            cc_low_bal = hard_weak_label_balance(cc_low_idx, weak_preds_strong, args.seed + SEED_OFFSETS["cca_low"] + fi)
            run_subsets[f"cca_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in cc_low_bal],
                [weak_labels[int(i)] for i in cc_low_bal],
            )
        # linear/affine alignment residual (the existing ridge weak->strong map residuals):
        # the linear counterpart of mlpalign -- what the LRT/stitching-style affine map misses.
        _lin_resid = np.asarray(residuals, dtype=np.float64).reshape(-1)
        lr_high_idx, _ = score_band_indices(_lin_resid, frac, "high")
        lr_high_bal = hard_weak_label_balance(lr_high_idx, weak_preds_strong, args.seed + SEED_OFFSETS["l2resid_high"] + fi)
        run_subsets[f"l2resid_high_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in lr_high_bal],
            [weak_labels[int(i)] for i in lr_high_bal],
        )
        lr_low_idx, _ = score_band_indices(_lin_resid, frac, "low")
        lr_low_bal = hard_weak_label_balance(lr_low_idx, weak_preds_strong, args.seed + SEED_OFFSETS["l2resid_low"] + fi)
        run_subsets[f"l2resid_low_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in lr_low_bal],
            [weak_labels[int(i)] for i in lr_low_bal],
        )
        for _rname, _rscores in (("ccarobust", cca_robust_scores), ("l2robust", l2robust_scores)):
            if _rscores is None:
                continue
            _hi_idx, _ = score_band_indices(_rscores, frac, "high")
            _hi_bal = hard_weak_label_balance(_hi_idx, weak_preds_strong, args.seed + SEED_OFFSETS[f"{_rname}_high"] + fi)
            run_subsets[f"{_rname}_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in _hi_bal],
                [weak_labels[int(i)] for i in _hi_bal],
            )
            _lo_idx, _ = score_band_indices(_rscores, frac, "low")
            _lo_bal = hard_weak_label_balance(_lo_idx, weak_preds_strong, args.seed + SEED_OFFSETS[f"{_rname}_low"] + fi)
            run_subsets[f"{_rname}_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in _lo_bal],
                [weak_labels[int(i)] for i in _lo_bal],
            )
        if mlprp_scores is not None:
            mr_high_idx, _ = score_band_indices(mlprp_scores, frac, "high")
            mr_high_bal = hard_weak_label_balance(mr_high_idx, weak_preds_strong, args.seed + SEED_OFFSETS["mlprp_high"] + fi)
            run_subsets[f"mlprp_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in mr_high_bal],
                [weak_labels[int(i)] for i in mr_high_bal],
            )
            mr_low_idx, _ = score_band_indices(mlprp_scores, frac, "low")
            mr_low_bal = hard_weak_label_balance(mr_low_idx, weak_preds_strong, args.seed + SEED_OFFSETS["mlprp_low"] + fi)
            run_subsets[f"mlprp_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in mr_low_bal],
                [weak_labels[int(i)] for i in mr_low_bal],
            )
        if lda_scores is not None:
            ld_high_idx, _ = score_band_indices(lda_scores, frac, "high")
            ld_high_bal = hard_weak_label_balance(ld_high_idx, weak_preds_strong, args.seed + SEED_OFFSETS["lda_high"] + fi)
            run_subsets[f"lda_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ld_high_bal],
                [weak_labels[int(i)] for i in ld_high_bal],
            )
            ld_low_idx, _ = score_band_indices(lda_scores, frac, "low")
            ld_low_bal = hard_weak_label_balance(ld_low_idx, weak_preds_strong, args.seed + SEED_OFFSETS["lda_low"] + fi)
            run_subsets[f"lda_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ld_low_bal],
                [weak_labels[int(i)] for i in ld_low_bal],
            )
        if softrp_scores is not None:
            sr_high_idx, _ = score_band_indices(softrp_scores, frac, "high")
            sr_high_bal = hard_weak_label_balance(sr_high_idx, weak_preds_strong, args.seed + SEED_OFFSETS["softrp_high"] + fi)
            run_subsets[f"softrp_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in sr_high_bal],
                [weak_labels[int(i)] for i in sr_high_bal],
            )
            sr_low_idx, _ = score_band_indices(softrp_scores, frac, "low")
            sr_low_bal = hard_weak_label_balance(sr_low_idx, weak_preds_strong, args.seed + SEED_OFFSETS["softrp_low"] + fi)
            run_subsets[f"softrp_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in sr_low_bal],
                [weak_labels[int(i)] for i in sr_low_bal],
            )
        for _ci, (_slot, _sc) in enumerate(sorted(custom_scores.items())):
            _hi, _ = score_band_indices(_sc, frac, "high")
            _hb = hard_weak_label_balance(_hi, weak_preds_strong, args.seed + SEED_OFFSETS["custom_high"] + 1000 * _ci + fi)
            run_subsets[f"{_slot}_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in _hb],
                [weak_labels[int(i)] for i in _hb],
            )
            _lo, _ = score_band_indices(_sc, frac, "low")
            _lb = hard_weak_label_balance(_lo, weak_preds_strong, args.seed + SEED_OFFSETS["custom_low"] + 1000 * _ci + fi)
            run_subsets[f"{_slot}_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in _lb],
                [weak_labels[int(i)] for i in _lb],
            )
        if linstep1_scores is not None:
            ls_high_idx, _ = score_band_indices(linstep1_scores, frac, "high")
            ls_high_bal = hard_weak_label_balance(ls_high_idx, weak_preds_strong, args.seed + SEED_OFFSETS["linstep1_high"] + fi)
            run_subsets[f"linstep1_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ls_high_bal],
                [weak_labels[int(i)] for i in ls_high_bal],
            )
            ls_low_idx, _ = score_band_indices(linstep1_scores, frac, "low")
            ls_low_bal = hard_weak_label_balance(ls_low_idx, weak_preds_strong, args.seed + SEED_OFFSETS["linstep1_low"] + fi)
            run_subsets[f"linstep1_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ls_low_bal],
                [weak_labels[int(i)] for i in ls_low_bal],
            )
        if mlpstep1_scores is not None:
            ms_high_idx, _ = score_band_indices(mlpstep1_scores, frac, "high")
            ms_high_bal = hard_weak_label_balance(ms_high_idx, weak_preds_strong, args.seed + SEED_OFFSETS["mlpstep1_high"] + fi)
            run_subsets[f"mlpstep1_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ms_high_bal],
                [weak_labels[int(i)] for i in ms_high_bal],
            )
            ms_low_idx, _ = score_band_indices(mlpstep1_scores, frac, "low")
            ms_low_bal = hard_weak_label_balance(ms_low_idx, weak_preds_strong, args.seed + SEED_OFFSETS["mlpstep1_low"] + fi)
            run_subsets[f"mlpstep1_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ms_low_bal],
                [weak_labels[int(i)] for i in ms_low_bal],
            )
        if mlpalign_scores is not None:
            ma_high_idx, _ = score_band_indices(mlpalign_scores, frac, "high")
            ma_high_bal = hard_weak_label_balance(ma_high_idx, weak_preds_strong, args.seed + SEED_OFFSETS["mlpalign_high"] + fi)
            run_subsets[f"mlpalign_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ma_high_bal],
                [weak_labels[int(i)] for i in ma_high_bal],
            )
            ma_low_idx, _ = score_band_indices(mlpalign_scores, frac, "low")
            ma_low_bal = hard_weak_label_balance(ma_low_idx, weak_preds_strong, args.seed + SEED_OFFSETS["mlpalign_low"] + fi)
            run_subsets[f"mlpalign_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ma_low_bal],
                [weak_labels[int(i)] for i in ma_low_bal],
            )
        # rp_conf: combine the two ORTHOGONAL axes -- rp (learnable) + weak confidence (clean label).
        # Rank-sum so both contribute equally, then keep the top-frac (SAME kept-set size as rp_high,
        # so the comparison vs rp_high_f{pct} is size-controlled). Tests whether stacking the
        # cleanliness axis on rp closes the gap to the weak_correct_only oracle.
        _rp_rank = np.argsort(np.argsort(rp_scores)).astype(np.float64)
        _conf_rank = np.argsort(np.argsort(weak_confidences_strong)).astype(np.float64)
        rp_conf_score = _rp_rank + _conf_rank
        rp_conf_high_idx, _ = score_band_indices(rp_conf_score, frac, "high")
        rp_conf_high_bal = hard_weak_label_balance(rp_conf_high_idx, weak_preds_strong, args.seed + SEED_OFFSETS["rp_conf_high"] + fi)
        run_subsets[f"rp_conf_high_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in rp_conf_high_bal],
            [weak_labels[int(i)] for i in rp_conf_high_bal],
        )
        # excess-loss / learnability (K-way option entropy): H_weak(options) - H_strong(options)
        if el_scores is not None:
            el_high_idx, _ = score_band_indices(el_scores, frac, "high")
            el_high_bal = hard_weak_label_balance(el_high_idx, weak_preds_strong, args.seed + SEED_OFFSETS["el_high"] + fi)
            run_subsets[f"el_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in el_high_bal],
                [weak_labels[int(i)] for i in el_high_bal],
            )
            el_low_idx, _ = score_band_indices(el_scores, frac, "low")
            el_low_bal = hard_weak_label_balance(el_low_idx, weak_preds_strong, args.seed + SEED_OFFSETS["el_low"] + fi)
            run_subsets[f"el_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in el_low_bal],
                [weak_labels[int(i)] for i in el_low_bal],
            )
        # confidence as K-way option entropy (weak-only = el with the strong term set to 0).
        # High confidence = LOW entropy, so we negate before banding to match the existing
        # weak_confidences_strong convention (high score = confident). conf_entropy_high keeps
        # the low-entropy / high-confidence questions; conf_entropy_low keeps the uncertain ones.
        if weak_entropy_scores is not None:
            weak_conf_entropy = -weak_entropy_scores
            ce_high_idx, _ = score_band_indices(weak_conf_entropy, frac, "high")
            ce_high_bal = hard_weak_label_balance(ce_high_idx, weak_preds_strong, args.seed + SEED_OFFSETS["conf_entropy_high"] + fi)
            run_subsets[f"conf_entropy_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ce_high_bal],
                [weak_labels[int(i)] for i in ce_high_bal],
            )
            ce_low_idx, _ = score_band_indices(weak_conf_entropy, frac, "low")
            ce_low_bal = hard_weak_label_balance(ce_low_idx, weak_preds_strong, args.seed + SEED_OFFSETS["conf_entropy_low"] + fi)
            run_subsets[f"conf_entropy_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ce_low_bal],
                [weak_labels[int(i)] for i in ce_low_bal],
            )
        knn_mixed_idx, _ = score_closest_indices(
            knn_stats["knn_correct_rate"], frac, args.knn_mixed_center, "mixed"
        )
        knn_mixed_bal = hard_weak_label_balance(knn_mixed_idx, weak_preds_strong, args.seed + SEED_OFFSETS["kf_knn_mixed"] + fi)
        run_subsets[f"knn_mixed_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in knn_mixed_bal],
            [weak_labels[int(i)] for i in knn_mixed_bal],
        )


def compute_selection_scores(
    *,
    args,
    runs: list[str],
    rep_weak_acts: dict,
    rep_strong_acts: dict,
    weak_preds_strong,
    weak_probs_strong,
    device,
) -> tuple[dict, dict]:
    """Compute every representation-side selection score the requested runs need.

    Lifted out of main() unchanged. Each score is computed only when some run
    asks for it, so an ordinary sweep pays for rp alone. Returns the scores by
    short name plus the externally supplied ones from --custom-scores-npz.

    The activation dicts stay alive after this returns; main frees them right
    after the call, before the excess-loss branch loads two models. Do not move
    that free in here: it would only unbind the parameters and free nothing.
    """
    rp_scores = representation_projection_scores(
        np.asarray(rep_weak_acts["strong_train"], dtype=np.float64),
        np.asarray(rep_strong_acts["strong_train"], dtype=np.float64),
        weak_preds_strong,
        reg=args.rp_reg,
        n_components=(args.rp_components or None),
    )
    # MLP alignment-residual score (the "MLP projection" alternative alignment): out-of-fold
    # ||f(h_w) - h_s|| from a small cross-fitted MLP. Label-free; only computed when requested.
    mlpalign_scores = None
    if any(r.startswith("mlpalign_") for r in runs):
        print("[mlpalign] fitting cross-fitted weak->strong MLP alignment (5 folds)...")
        mlpalign_scores = mlp_alignment_scores(
            np.asarray(rep_weak_acts["strong_train"], dtype=np.float32),
            np.asarray(rep_strong_acts["strong_train"], dtype=np.float32),
            n_folds=5, seed=args.seed, device=str(device),
            epochs=args.mlpalign_epochs,
            bottleneck=(args.mlpalign_bottleneck or None),
            insample=args.mlpalign_insample,
        )
        print(f"[mlpalign] scores: mean {mlpalign_scores.mean():.3f} std {mlpalign_scores.std():.3f}")
    # CCA alignment-disagreement score (cross-fitted regularized linear CCA); label-free.
    cca_scores = None
    if any(r.startswith("cca_") for r in runs):
        print("[cca] fitting cross-fitted CCA alignment (5 folds)...")
        cca_scores = cca_alignment_scores(
            np.asarray(rep_weak_acts["strong_train"], dtype=np.float64),
            np.asarray(rep_strong_acts["strong_train"], dtype=np.float64),
            n_folds=5, seed=args.seed,
        )
        print(f"[cca] scores: mean {cca_scores.mean():.3f} std {cca_scores.std():.3f}")
    # Robust variants (trim-refit): the whiteboard's outlier-discarding ridge + robust CCA.
    cca_robust_scores = None
    if any(r.startswith("ccarobust_") for r in runs):
        print("[ccarobust] fitting robust (trim-refit) CCA...")
        cca_robust_scores = cca_alignment_scores(
            np.asarray(rep_weak_acts["strong_train"], dtype=np.float64),
            np.asarray(rep_strong_acts["strong_train"], dtype=np.float64),
            n_folds=5, seed=args.seed, trim_frac=0.1,
        )
        print(f"[ccarobust] mean {cca_robust_scores.mean():.3f} std {cca_robust_scores.std():.3f}")
    l2robust_scores = None
    if any(r.startswith("l2robust_") for r in runs):
        print("[l2robust] fitting robust (trim-refit) ridge residuals...")
        l2robust_scores = robust_linear_residual_scores(
            np.asarray(rep_weak_acts["strong_train"], dtype=np.float64),
            np.asarray(rep_strong_acts["strong_train"], dtype=np.float64),
            n_folds=5, seed=args.seed, trim_frac=0.1,
        )
        print(f"[l2robust] mean {l2robust_scores.mean():.3f} std {l2robust_scores.std():.3f}")
    # Hybrid nonlinear rp: kernel-ridge weak-side (same as rp) + MLP-ensemble strong-side.
    # Label-conditioned; isolates what a nonlinear strong-side projection buys over rp.
    mlprp_scores = None
    if any(r.startswith("mlprp_") for r in runs):
        print("[mlprp] fitting hybrid ridge+MLP rp (5 folds x 3 ensemble)...")
        mlprp_scores = mlp_rp_scores(
            np.asarray(rep_weak_acts["strong_train"], dtype=np.float32),
            np.asarray(rep_strong_acts["strong_train"], dtype=np.float32),
            weak_preds_strong,
            n_folds=5, seed=args.seed, device=str(device), ridge_reg=args.rp_reg,
        )
        print(f"[mlprp] scores: mean {mlprp_scores.mean():.4f} std {mlprp_scores.std():.4f}")
    # Swapped hybrid: heavily weight-decayed MLP weak-side + ridge strong-side (07/22
    # follow-up; wd from --mlpstep1-wd, default the sweep's peak-agreement setting).
    mlpstep1_scores = None
    if any(r.startswith("mlpstep1_") for r in runs):
        print(f"[mlpstep1] fitting MLP(wd={args.mlpstep1_wd})+ridge rp (5 folds x 3 ensemble)...")
        mlpstep1_scores = mlp_step1_rp_scores(
            np.asarray(rep_weak_acts["strong_train"], dtype=np.float32),
            np.asarray(rep_strong_acts["strong_train"], dtype=np.float32),
            weak_preds_strong,
            n_folds=5, seed=args.seed, device=str(device), ridge_reg=args.rp_reg,
            weight_decay=args.mlpstep1_wd,
        )
        print(f"[mlpstep1] scores: mean {mlpstep1_scores.mean():.4f} std {mlpstep1_scores.std():.4f}")
    # Pure cross-fit ablation: rp's step-1 kernel ridge fitted out-of-fold, step 2 unchanged.
    linstep1_scores = None
    if any(r.startswith("linstep1_") for r in runs):
        print("[linstep1] fitting oof kernel-ridge step-1 rp (5 folds, closed form)...")
        linstep1_scores = linstep1_rp_scores(
            np.asarray(rep_weak_acts["strong_train"], dtype=np.float64),
            np.asarray(rep_strong_acts["strong_train"], dtype=np.float64),
            weak_preds_strong,
            n_folds=5, seed=args.seed, ridge_reg=args.rp_reg,
        )
        print(f"[linstep1] scores: mean {linstep1_scores.mean():.4f} std {linstep1_scores.std():.4f}")
    # Soft-label rp: identical projection chain but yc built from the probe PROBABILITIES
    # (07/24 offline screen: rank corr .64 vs hard, error separation .127 vs .108).
    softrp_scores = None
    if any(r.startswith("softrp_") for r in runs):
        print("[softrp] rp on soft weak probabilities...")
        softrp_scores = representation_projection_scores(
            np.asarray(rep_weak_acts["strong_train"], dtype=np.float64),
            np.asarray(rep_strong_acts["strong_train"], dtype=np.float64),
            np.asarray(weak_probs_strong, dtype=np.float64),
            reg=args.rp_reg, n_components=(args.rp_components or None),
        )
        print(f"[softrp] scores: mean {softrp_scores.mean():.4f} std {softrp_scores.std():.4f}")
    # Precomputed custom scores (offline-designed selectors, e.g. the joint screen's ticket
    # holders). The npz must carry weak_preds for the alignment guard.
    custom_scores = {}
    if getattr(args, "custom_scores_npz", "") and any(r.startswith("cs") for r in runs):
        _cs = np.load(args.custom_scores_npz)
        _stored = np.asarray(_cs["weak_preds"]).astype(int)
        _own = np.asarray(weak_preds_strong).astype(int)
        _rate = float((_stored == _own).mean()) if len(_stored) == len(_own) else 0.0
        if _rate < 0.95:
            raise ValueError(f"custom-scores npz misaligned with strong_train (weak_preds match {_rate:.3f})")
        if _rate < 0.995:
            print(f"[custom] WARNING: weak_preds match {_rate:.3f} -- cross-machine probe drift tolerated")
        for _slot in ("cs1", "cs2", "cs3", "cs4"):
            if _slot in _cs.files:
                custom_scores[_slot] = np.asarray(_cs[_slot], dtype=np.float64)
        print(f"[custom] loaded {sorted(custom_scores)} from {args.custom_scores_npz}")
    # LDA-projected linear residual: label-aware twin of l2resid (07/22 meeting).
    lda_scores = None
    if any(r.startswith("lda_") for r in runs):
        print("[lda] fitting LDA-projected ridge-map residuals (5 folds)...")
        lda_scores = lda_alignment_scores(
            np.asarray(rep_weak_acts["strong_train"], dtype=np.float64),
            np.asarray(rep_strong_acts["strong_train"], dtype=np.float64),
            weak_preds_strong,
            n_folds=5, seed=args.seed,
        )
        print(f"[lda] scores: mean {lda_scores.mean():.4f} std {lda_scores.std():.4f}")

    return (
        {
            "rp": rp_scores,
            "cca": cca_scores,
            "cca_robust": cca_robust_scores,
            "l2robust": l2robust_scores,
            "mlprp": mlprp_scores,
            "mlpstep1": mlpstep1_scores,
            "linstep1": linstep1_scores,
            "mlpalign": mlpalign_scores,
            "softrp": softrp_scores,
            "lda": lda_scores,
        },
        custom_scores,
    )


def prepare_multichoice_eval(*, args, device, output_dir, weak_probe) -> tuple[list | None, float | None]:
    """Build the multiple-choice eval set and score the weak probe on it.

    Returns (eval3_examples, weak_eval3_acc). The weak accuracy here is the lower
    bound of the bounds table: the same argmax-over-candidates metric the strong
    side is scored with, so the two are directly comparable. Both are None when
    --eval-3class is off.
    """
    eval3_examples = None
    if args.eval_3class:
        if args.dataset == "dream":
            multichoice_rows = load_dream_3class_eval(args.n_eval_questions, args.seed)
        elif args.dataset == "sciq":
            multichoice_rows = load_sciq_multichoice_eval(
                args.n_eval_questions, args.seed + 2, args.sciq_use_support
            )
        elif args.dataset == "anli":
            multichoice_rows = load_anli_multichoice_eval(
                args.n_eval_questions, args.seed + 2, args.anli_round
            )
        elif args.dataset == "hellaswag":
            multichoice_rows = load_hellaswag_multichoice_eval(args.n_eval_questions, args.seed + 2)
        elif args.dataset == "reclor":
            multichoice_rows = load_reclor_multichoice_eval(args.n_eval_questions, args.seed + 2)
        elif args.dataset == "logiqa2":
            multichoice_rows = load_logiqa2_multichoice_eval(args.n_eval_questions, args.seed + 2)
        elif args.dataset == "wanli":
            multichoice_rows = load_wanli_multichoice_eval(args.n_eval_questions, args.seed + 2)
        elif args.dataset == "control":
            multichoice_rows = load_control_multichoice_eval(args.n_eval_questions, args.seed + 2)
        elif args.dataset == "fevernli":
            multichoice_rows = load_fevernli_multichoice_eval(args.n_eval_questions, args.seed + 2)
        elif args.dataset == "conjnli":
            multichoice_rows = load_conjnli_multichoice_eval(args.n_eval_questions, args.seed + 2)
        elif args.dataset == "imppres":
            multichoice_rows = load_imppres_multichoice_eval(args.n_eval_questions, args.seed + 2)
        elif args.dataset == "aquarat":
            multichoice_rows = load_aquarat_multichoice_eval(args.n_eval_questions, args.seed + 2)
        elif args.dataset == "quail":
            multichoice_rows = _generic_mc_eval(_quail_rows("validation", args.seed + 2), "quail", args.n_eval_questions, args.seed + 2)
        elif args.dataset == "riddlesense":
            multichoice_rows = _generic_mc_eval(_riddlesense_rows("validation", args.seed + 2), "riddlesense", args.n_eval_questions, args.seed + 2)
        elif args.dataset == "scinli":
            multichoice_rows = _generic_mc_eval(_scinli_rows("test", args.seed + 2), "scinli", args.n_eval_questions, args.seed + 2)
        else:
            multichoice_rows = []
        if multichoice_rows:
            eval3_examples = [
                LoraExample(
                    id=r["id"],
                    source_id=r["source_id"],
                    text=r["txt"],
                    label=int(r["labels"]),
                    answer_suffix=args.answer_suffix,
                )
                for r in multichoice_rows
            ]
            n_questions = len({r["source_id"] for r in multichoice_rows})
            print(
                f"[multichoice] built {len(eval3_examples)} candidate rows over "
                f"{n_questions} {args.dataset} questions"
            )
    # Weak-model multiple-choice accuracy = the LOWER BOUND for the bounds table /
    # multichoice PGR. Score the candidate set with the (already-fitted) weak probe and
    # argmax over each question's candidates, same as the strong-side multichoice metric.
    weak_eval3_acc = None
    if eval3_examples:
        weak_eval3_acts = extract_final_token_activations(
            args.weak_model,
            [ex.text for ex in eval3_examples],
            device,
            args.torch_dtype,
            args.activation_batch_size,
            args.activation_max_length,
            "extract weak eval3 activations",
        )
        weak_eval3_rows = [
            {"prob_label1": float(p)} for p in predict_probe(weak_probe, weak_eval3_acts, device)
        ]
        weak_eval3_acc = accuracy_3class(eval3_examples, weak_eval3_rows)
        write_eval3_rows(output_dir / "eval3_weak.csv", eval3_examples, weak_eval3_rows)
        del weak_eval3_acts
        clear_memory()
        print(f"[multichoice] weak-probe lower bound accuracy_3class = {weak_eval3_acc:.3f}")

    return eval3_examples, weak_eval3_acc


def main() -> None:
    args = parse_args()
    runs = requested_runs(args)
    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this script on a CUDA GPU machine.")
    args.device = str(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits = load_paper_style_splits(args)
    texts = split_texts(splits)
    if args.diagnostics_only:
        run_knn_saturation_diagnostics(args, splits, texts, device, output_dir)
        return

    lora_examples = {
        "weak_train": to_lora_examples(splits.weak_train, args.answer_suffix),
        "strong_train": to_lora_examples(splits.strong_train, args.answer_suffix),
        "test": to_lora_examples(splits.test, args.answer_suffix),
    }

    weak_acts = extract_all_activations(
        args.weak_model,
        texts,
        device,
        args.torch_dtype,
        args.activation_batch_size,
        args.activation_max_length,
        "extract weak activations",
    )
    weak_probe = fit_probe(
        weak_acts["weak_train"],
        torch.tensor(split_labels(splits.weak_train), dtype=torch.float32),
        args.l2_penalty,
        args.max_iter,
        device,
    )
    weak_probs_weak_train = predict_probe(weak_probe, weak_acts["weak_train"], device)
    weak_probs_strong = predict_probe(weak_probe, weak_acts["strong_train"], device)
    weak_probs_test = predict_probe(weak_probe, weak_acts["test"], device)
    weak_train_labels = split_labels(splits.weak_train)
    weak_preds_weak_train = (weak_probs_weak_train >= 0.5).astype(int)
    if args.knn_reference_cross_fit:
        cross_fit_probs_weak_train = cross_fitted_weak_probs(
            weak_acts["weak_train"],
            weak_train_labels,
            args.cross_fit_folds,
            args.l2_penalty,
            args.max_iter,
            device,
            args.seed,
        )
        reference_preds_weak_train = (cross_fit_probs_weak_train >= 0.5).astype(int)
        weak_correct_weak_train = (reference_preds_weak_train == weak_train_labels).astype(int)
        print(
            "[cross-fit] weak_train reference accuracy: "
            f"in-sample={float(np.mean(weak_preds_weak_train == weak_train_labels)):.3f}, "
            f"cross-fitted={float(np.mean(weak_correct_weak_train)):.3f} "
            f"(folds={max(2, min(args.cross_fit_folds, len(weak_train_labels)))})"
        )
    else:
        weak_correct_weak_train = (weak_preds_weak_train == weak_train_labels).astype(int)
    weak_preds_strong = (weak_probs_strong >= 0.5).astype(int)
    weak_preds_test = (weak_probs_test >= 0.5).astype(int)

    committee_mean_strong, committee_disagreement_strong = committee_disagreement_on_strong_train(
        weak_acts["weak_train"],
        weak_train_labels,
        weak_acts["strong_train"],
        args.committee_members,
        args.l2_penalty,
        args.max_iter,
        device,
        args.seed + SEED_OFFSETS["committee_fit"],
    )

    strong_train_labels = split_labels(splits.strong_train)
    test_labels = split_labels(splits.test)
    weak_eval_rows = eval_rows_from_probs(lora_examples["test"], weak_probs_test)

    strong_acts = extract_all_activations(
        args.strong_model,
        texts,
        device,
        args.torch_dtype,
        args.activation_batch_size,
        args.activation_max_length,
        "extract strong activations",
    )
    best_map, map_rows, residuals = fit_maps(
        weak_acts["weak_train"],
        weak_acts["strong_train"],
        strong_acts["weak_train"],
        strong_acts["strong_train"],
        splits,
        args,
        device,
        output_dir,
    )
    # kNN / RP may use a different activation LAYER (--representation-layer), token POOLING
    # (--representation-pooling), and/or SPAN (--representation-span: full prompt vs answer-only,
    # masking the question) as controls. The weak probe/confidence and the L2-residual maps stay
    # on the default end-layer / last-token / full-prompt reps (extracted above); only the kNN and
    # RP representations move, so the weak labels (and the oracle) are held fixed across the
    # comparison. The default (end, last, full) reuses the already-extracted acts (no extra cost).
    answer_span = args.representation_span in ("answer", "context")
    rep_is_default = (
        args.representation_layer == "end"
        and args.representation_pooling == "last"
        and not answer_span
    )
    if rep_is_default:
        rep_weak_acts, rep_strong_acts = weak_acts, strong_acts
    else:
        rep_tag = f"{args.representation_layer},{args.representation_pooling},{args.representation_span}"
        rep_weak_acts = extract_all_activations(
            args.weak_model, texts, device, args.torch_dtype,
            args.activation_batch_size, args.activation_max_length,
            f"extract weak activations ({rep_tag})",
            layer=args.representation_layer, pooling=args.representation_pooling,
            answer_span=answer_span, answer_suffix=args.answer_suffix,
            span_kind=args.representation_span,
        )
        rep_strong_acts = extract_all_activations(
            args.strong_model, texts, device, args.torch_dtype,
            args.activation_batch_size, args.activation_max_length,
            f"extract strong activations ({rep_tag})",
            layer=args.representation_layer, pooling=args.representation_pooling,
            answer_span=answer_span, answer_suffix=args.answer_suffix,
            span_kind=args.representation_span,
        )
    knn_stats = compute_weak_train_knn_stats(
        rep_strong_acts["weak_train"],
        rep_strong_acts["strong_train"],
        weak_correct_weak_train,
        args.knn_k,
    )
    if getattr(args, "dump_acts", ""):
        np.savez_compressed(
            args.dump_acts,
            weak=np.asarray(rep_weak_acts["strong_train"], dtype=np.float32),
            strong=np.asarray(rep_strong_acts["strong_train"], dtype=np.float32),
            weak_preds=np.asarray(weak_preds_strong, dtype=np.int64),
            weak_probs=np.asarray(weak_probs_strong, dtype=np.float32),
            gt=np.asarray(strong_train_labels, dtype=np.int64),
        )
        print(f"[dump-acts] wrote {args.dump_acts}")

    scores, custom_scores = compute_selection_scores(
        args=args,
        runs=runs,
        rep_weak_acts=rep_weak_acts,
        rep_strong_acts=rep_strong_acts,
        weak_preds_strong=weak_preds_strong,
        weak_probs_strong=weak_probs_strong,
        device=device,
    )
    rp_scores = scores["rp"]
    if not rep_is_default:
        del rep_weak_acts, rep_strong_acts
        clear_memory()
    del strong_acts
    clear_memory()
    # excess-loss / learnability (K-way option entropy): H_weak(options) - H_strong(options),
    # each model's yes/no head over the K options (no probe). Expensive (loads the weak +
    # untuned-strong models once), so only when an el_* run is requested; independent of LoRA
    # training -> computed once per data seed.
    el_scores = None
    weak_entropy_scores = None
    if args.dump_el or any(r.strip().startswith(("el_", "conf_entropy_")) for r in args.runs.split(",")):
        el_scores, weak_entropy_scores, el_extras = compute_el_kway_scores(
            args, splits.strong_train, args.device
        )
    scores["el"] = el_scores
    scores["weak_entropy"] = weak_entropy_scores
    if getattr(args, "dump_el", "") and el_scores is not None:
        np.savez_compressed(
            args.dump_el,
            el=np.asarray(el_scores, dtype=np.float64),
            weak_entropy=np.asarray(weak_entropy_scores, dtype=np.float64),
            elloss=el_extras["elloss"],
            weak_pick=el_extras["weak_pick"],
            gold_opt=el_extras["gold_opt"],
            weak_preds=np.asarray(weak_preds_strong, dtype=np.int64),
            weak_probs=np.asarray(weak_probs_strong, dtype=np.float32),
            gt=np.asarray(strong_train_labels, dtype=np.int64),
        )
        print(f"[dump-el] wrote {args.dump_el}")

    best_row = next(row for row in map_rows if row["name"] == best_map.name)
    middle_indices, middle_filter = middle_residual_indices(residuals, args.residual_keep_middle_frac)
    middle_balanced_indices = hard_weak_label_balance(
        middle_indices,
        weak_preds_strong,
        args.seed,
    )
    weak_confidences_strong = 2.0 * np.abs(weak_probs_strong - 0.5)
    confidence_middle_indices, confidence_middle_filter = score_band_indices(
        weak_confidences_strong,
        args.confidence_keep_frac,
        "middle",
    )
    confidence_middle_balanced_indices = hard_weak_label_balance(
        confidence_middle_indices,
        weak_preds_strong,
        args.seed + SEED_OFFSETS["confidence_middle"],
    )
    confidence_high_indices, confidence_high_filter = score_band_indices(
        weak_confidences_strong,
        args.confidence_keep_frac,
        "high",
    )
    confidence_high_balanced_indices = hard_weak_label_balance(
        confidence_high_indices,
        weak_preds_strong,
        args.seed + SEED_OFFSETS["confidence_high"],
    )
    knn_middle_indices, knn_middle_filter = score_band_indices(
        knn_stats["knn_correct_rate"],
        args.knn_keep_middle_frac,
        "middle",
    )
    knn_middle_balanced_indices = hard_weak_label_balance(
        knn_middle_indices,
        weak_preds_strong,
        args.seed + SEED_OFFSETS["knn_middle"],
    )
    knn_mixed_indices, knn_mixed_filter = score_closest_indices(
        knn_stats["knn_correct_rate"],
        args.knn_keep_middle_frac,
        args.knn_mixed_center,
        "mixed",
    )
    knn_mixed_balanced_indices = hard_weak_label_balance(
        knn_mixed_indices,
        weak_preds_strong,
        args.seed + SEED_OFFSETS["knn_mixed"],
    )
    knn_high_indices, knn_high_filter = score_band_indices(
        knn_stats["knn_correct_rate"],
        args.knn_keep_middle_frac,
        "high",
    )
    knn_high_balanced_indices = hard_weak_label_balance(
        knn_high_indices,
        weak_preds_strong,
        args.seed + SEED_OFFSETS["knn_high"],
    )
    committee_agree_indices, committee_agree_filter = score_band_indices(
        committee_disagreement_strong,
        args.committee_keep_frac,
        "low",
    )
    committee_agree_balanced_indices = hard_weak_label_balance(
        committee_agree_indices,
        weak_preds_strong,
        args.seed + SEED_OFFSETS["committee_agree"],
    )
    committee_disagree_indices, committee_disagree_filter = score_band_indices(
        committee_disagreement_strong,
        args.committee_keep_frac,
        "high",
    )
    committee_disagree_balanced_indices = hard_weak_label_balance(
        committee_disagree_indices,
        weak_preds_strong,
        args.seed + SEED_OFFSETS["committee_disagree"],
    )
    random_control_size = args.random_control_size or len(middle_balanced_indices)

    strong_examples = lora_examples["strong_train"]
    eval_examples = lora_examples["test"]
    eval3_examples, weak_eval3_acc = prepare_multichoice_eval(
        args=args, device=device, output_dir=output_dir, weak_probe=weak_probe
    )
    weak_labels = weak_preds_strong.tolist()
    # weak_label_balanced = the FULL no-filter set balanced by the weak predicted label.
    # With weak_label (unbalanced full), random_balanced_f50 (balanced half) and
    # random_unbalanced_f50 (unbalanced half) it forms a balanced x fraction 2x2, so a
    # "random > no-filter" gap can be split into a balancing effect vs a data-fraction effect.
    weak_label_balanced_indices = hard_weak_label_balance(
        np.arange(len(weak_preds_strong)), weak_preds_strong, args.seed + SEED_OFFSETS["weak_label_balanced"]
    )
    # Filter only correct weak points: the oracle / best-case label-noise
    # filter -- keep only the strong_train examples whose weak label already equals gold. On
    # this subset the weak labels are noise-free, so it upper-bounds what any noise-removing
    # selector (confidence / kNN / rp / el) could buy. If it barely beats base, label noise is
    # not the bottleneck (recall / format is). Uses gold only to SELECT; the labels trained on
    # are still the weak ones (which equal gold here by construction).
    weak_correct_indices = np.where(weak_preds_strong == strong_train_labels)[0]
    weak_correct_balanced_indices = hard_weak_label_balance(
        weak_correct_indices, weak_preds_strong, args.seed + SEED_OFFSETS["weak_correct_balanced"]
    )
    # Curriculum: same full no-filter set as weak_label, trained in a fixed order by weak
    # confidence (easy = most-confident first, hard = least-confident first). Control is
    # weak_label (random order); see train_lora_model's curriculum_order.
    curriculum_orders = {
        "curriculum_easy": list(np.argsort(-weak_confidences_strong)),
        "curriculum_hard": list(np.argsort(weak_confidences_strong)),
    }
    # Continuous-weighting (loss-adaptation): the *_weighted runs train the FULL set but weight
    # each example's loss by a selection score (temperature = args.weight_temperature; T->inf is
    # no-filter, T->0 is hard top-selection). Smooth counterpart of the rp_high / el_high /
    # confidence_high bands. rp_scores + weak_confidences_strong are always available; el_scores
    # only when an el_* run is requested (el_weighted starts with "el_", so it triggers it).
    wT = float(getattr(args, "weight_temperature", 1.0))
    run_weights: dict[str, np.ndarray] = {
        "rp_weighted": scores_to_weights(rp_scores, wT),
        "conf_weighted": scores_to_weights(weak_confidences_strong, wT),
    }
    if el_scores is not None:
        run_weights["el_weighted"] = scores_to_weights(el_scores, wT)
    run_subsets: dict[str, tuple[list[LoraExample], list[int]]] = {
        "ground_truth": (strong_examples, strong_train_labels.tolist()),
        "weak_label": (strong_examples, weak_labels),
        "rp_weighted": (strong_examples, weak_labels),
        "el_weighted": (strong_examples, weak_labels),
        "conf_weighted": (strong_examples, weak_labels),
        "weak_label_balanced": (
            [strong_examples[int(i)] for i in weak_label_balanced_indices],
            [weak_labels[int(i)] for i in weak_label_balanced_indices],
        ),
        "weak_correct_only": (
            [strong_examples[int(i)] for i in weak_correct_indices],
            [weak_labels[int(i)] for i in weak_correct_indices],
        ),
        "weak_correct_only_balanced": (
            [strong_examples[int(i)] for i in weak_correct_balanced_indices],
            [weak_labels[int(i)] for i in weak_correct_balanced_indices],
        ),
        "curriculum_easy": (strong_examples, weak_labels),
        "curriculum_hard": (strong_examples, weak_labels),
        "middle_unbalanced": (
            [strong_examples[int(i)] for i in middle_indices],
            [weak_labels[int(i)] for i in middle_indices],
        ),
        "middle_balanced": (
            [strong_examples[int(i)] for i in middle_balanced_indices],
            [weak_labels[int(i)] for i in middle_balanced_indices],
        ),
        "confidence_middle_unbalanced": (
            [strong_examples[int(i)] for i in confidence_middle_indices],
            [weak_labels[int(i)] for i in confidence_middle_indices],
        ),
        "confidence_middle_balanced": (
            [strong_examples[int(i)] for i in confidence_middle_balanced_indices],
            [weak_labels[int(i)] for i in confidence_middle_balanced_indices],
        ),
        "confidence_high_unbalanced": (
            [strong_examples[int(i)] for i in confidence_high_indices],
            [weak_labels[int(i)] for i in confidence_high_indices],
        ),
        "confidence_high_balanced": (
            [strong_examples[int(i)] for i in confidence_high_balanced_indices],
            [weak_labels[int(i)] for i in confidence_high_balanced_indices],
        ),
        "knn_middle_unbalanced": (
            [strong_examples[int(i)] for i in knn_middle_indices],
            [weak_labels[int(i)] for i in knn_middle_indices],
        ),
        "knn_middle_balanced": (
            [strong_examples[int(i)] for i in knn_middle_balanced_indices],
            [weak_labels[int(i)] for i in knn_middle_balanced_indices],
        ),
        "knn_mixed_unbalanced": (
            [strong_examples[int(i)] for i in knn_mixed_indices],
            [weak_labels[int(i)] for i in knn_mixed_indices],
        ),
        "knn_mixed_balanced": (
            [strong_examples[int(i)] for i in knn_mixed_balanced_indices],
            [weak_labels[int(i)] for i in knn_mixed_balanced_indices],
        ),
        "knn_high_unbalanced": (
            [strong_examples[int(i)] for i in knn_high_indices],
            [weak_labels[int(i)] for i in knn_high_indices],
        ),
        "knn_high_balanced": (
            [strong_examples[int(i)] for i in knn_high_balanced_indices],
            [weak_labels[int(i)] for i in knn_high_balanced_indices],
        ),
        "committee_agree_unbalanced": (
            [strong_examples[int(i)] for i in committee_agree_indices],
            [weak_labels[int(i)] for i in committee_agree_indices],
        ),
        "committee_agree_balanced": (
            [strong_examples[int(i)] for i in committee_agree_balanced_indices],
            [weak_labels[int(i)] for i in committee_agree_balanced_indices],
        ),
        "committee_disagree_unbalanced": (
            [strong_examples[int(i)] for i in committee_disagree_indices],
            [weak_labels[int(i)] for i in committee_disagree_indices],
        ),
        "committee_disagree_balanced": (
            [strong_examples[int(i)] for i in committee_disagree_balanced_indices],
            [weak_labels[int(i)] for i in committee_disagree_balanced_indices],
        ),
    }
    # weak_strong_agree: keep the points where the fine-tuned weak model and the UNTUNED strong
    # base agree on the yes/no prediction (gold-free "label likely correct" proxy -- if both
    # independent models say the same thing, the weak label is more trustworthy). Requires one
    # extra base pass over strong_train, so only computed when an agree/disagree arm is requested.
    if any(r in runs for r in ("weak_strong_agree_balanced", "weak_strong_disagree_balanced")):
        from run_dream_w2s_baselines import load_strong_model_and_tokenizer
        _bm, _bt = load_strong_model_and_tokenizer(args, trainable_lora=False)
        _, _base_rows = evaluate_yes_no(
            _bm, _bt, lora_examples["strong_train"],
            max(int(args.strong_batch_size), 8), device, args.max_length,
            "score base on strong_train (agree arm)",
        )
        base_preds_strong = (np.asarray([float(r["prob_label1"]) for r in _base_rows]) >= 0.5).astype(int)
        del _bm
        clear_memory()
        agree_idx = np.where(base_preds_strong == weak_preds_strong)[0]
        disagree_idx = np.where(base_preds_strong != weak_preds_strong)[0]
        agree_bal = hard_weak_label_balance(agree_idx, weak_preds_strong, args.seed + SEED_OFFSETS["agree_full"])
        disagree_bal = hard_weak_label_balance(disagree_idx, weak_preds_strong, args.seed + SEED_OFFSETS["disagree_full"])
        run_subsets["weak_strong_agree_balanced"] = (
            [strong_examples[int(i)] for i in agree_bal],
            [weak_labels[int(i)] for i in agree_bal],
        )
        run_subsets["weak_strong_disagree_balanced"] = (
            [strong_examples[int(i)] for i in disagree_bal],
            [weak_labels[int(i)] for i in disagree_bal],
        )
        print(f"[weak_strong_agree] base-weak agreement rate: {len(agree_idx) / max(1, len(weak_preds_strong)):.3f} "
              f"(agree {len(agree_idx)}, balanced {len(agree_bal)}; disagree {len(disagree_idx)}, balanced {len(disagree_bal)})")
    # Optional committee selection sweep over keep fractions (label-complexity /
    # data-efficiency curve): for each f, keep the most-reliable (low-disagreement)
    # or most-boundary (high-disagreement) f-fraction, plus a matched random_balanced.
    committee_keep_fracs = [
        float(x) for x in (args.committee_keep_fracs or "").split(",") if x.strip()
    ]
    n_strong_examples = len(strong_examples)
    build_keep_fraction_arms(
        args=args,
        run_subsets=run_subsets,
        curriculum_orders=curriculum_orders,
        scores=scores,
        custom_scores=custom_scores,
        committee_keep_fracs=committee_keep_fracs,
        committee_disagreement_strong=committee_disagreement_strong,
        weak_confidences_strong=weak_confidences_strong,
        weak_preds_strong=weak_preds_strong,
        weak_labels=weak_labels,
        strong_examples=strong_examples,
        residuals=residuals,
        knn_stats=knn_stats,
        random_balanced_indices=random_balanced_indices,
        n_strong_examples=n_strong_examples,
    )

    # Disjoint equal-size quantile bins for a dose-response / separation analysis (--quantile-bins K):
    # for rp / el / confidence, split into K equal-size bins by score percentile; q0 = highest-score
    # bin (top 1/K) .. q{K-1} = lowest. All bins are the SAME size, so accuracy differences isolate
    # the score's discriminability from sample-size effects (unlike a cumulative top-x% sweep). A
    # genuine ordering signal -> monotonic accuracy across bins; separation = q0 - q{K-1}.
    if args.quantile_bins and args.quantile_bins > 1:
        K = int(args.quantile_bins)
        qsignals = {"rp": rp_scores, "confidence": weak_confidences_strong}
        if el_scores is not None:
            qsignals["el"] = el_scores
        for sidx, (sig_name, scores) in enumerate(qsignals.items()):
            for i in range(K):
                hi, lo = 1.0 - i / K, 1.0 - (i + 1) / K
                q_idx = score_percentile_band_indices(scores, lo, hi)
                q_bal = hard_weak_label_balance(q_idx, weak_preds_strong, args.seed + SEED_OFFSETS["quantile"] + 100 * sidx + i)
                run_subsets[f"{sig_name}_q{i}_balanced"] = (
                    [strong_examples[int(j)] for j in q_bal],
                    [weak_labels[int(j)] for j in q_bal],
                )
                # GT-rescue variant: the SAME bin subset trained with gold labels instead of
                # weak labels. Separates the two toxicity mechanisms for a bad bin: if GT
                # cannot rescue it either, the points are unlearnable (representation-side);
                # if GT rescues it, the damage was systematic weak-label error.
                run_subsets[f"{sig_name}_q{i}_gt_balanced"] = (
                    [strong_examples[int(j)] for j in q_bal],
                    [int(strong_train_labels[int(j)]) for j in q_bal],
                )

    random_run_names: list[str] = []
    random_unbalanced_run_names: list[str] = []
    if "random_unbalanced" in runs:
        runs = [name for name in runs if name != "random_unbalanced"]
        random_unbalanced_size = args.random_unbalanced_size or len(middle_indices)
        for idx in range(args.random_control_count):
            run_name = f"random_unbalanced_{idx}"
            selected = random_unbalanced_indices(
                len(strong_examples),
                random_unbalanced_size,
                args.seed + SEED_OFFSETS["random_unbalanced"] + idx,
            )
            run_subsets[run_name] = (
                [strong_examples[int(i)] for i in selected],
                [weak_labels[int(i)] for i in selected],
            )
            random_unbalanced_run_names.append(run_name)
            runs.append(run_name)

    if "random_balanced" in runs:
        runs = [name for name in runs if name != "random_balanced"]
        for idx in range(args.random_control_count):
            run_name = f"random_balanced_{idx}"
            selected = random_balanced_indices(weak_preds_strong, random_control_size, args.seed + SEED_OFFSETS["random_balanced"] + idx)
            run_subsets[run_name] = (
                [strong_examples[int(i)] for i in selected],
                [weak_labels[int(i)] for i in selected],
            )
            random_run_names.append(run_name)
            runs.append(run_name)

    prediction_columns: dict[str, list[dict]] = {}
    run_reports: dict[str, dict] = {}

    if "base" in runs:
        from run_dream_w2s_baselines import load_strong_model_and_tokenizer

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
        base_summary["auroc"] = auroc_from_rows(base_rows)
        base_summary["accuracy_prior_matched"] = prior_matched_accuracy(base_rows)
        if eval3_examples:
            _, base_rows3 = evaluate_yes_no(
                model,
                tokenizer,
                eval3_examples,
                args.strong_batch_size,
                args.device,
                args.max_length,
                "eval3 base",
            )
            base_summary["accuracy_3class"] = accuracy_3class(eval3_examples, base_rows3)
            write_eval3_rows(output_dir / "eval3_base.csv", eval3_examples, base_rows3)
        prediction_columns["base"] = base_rows
        run_reports["base"] = {"eval": base_summary}
        del model
        del tokenizer
        clear_memory()

    eff_batch = max(1, args.strong_batch_size * args.gradient_accumulation_steps)
    fixed_steps = args.max_train_steps
    for run_name in runs:
        if run_name == "base":
            continue
        train_examples, train_labels = run_subsets[run_name]
        if args.epochs > 0:
            # Train this run for `epochs` full passes over its ACTUAL subset
            # (no-filter -> all data; f50 -> the real 50%), instead of the fixed
            # compute-matched cap. no-filter therefore gets proportionally more steps.
            args.max_train_steps = args.epochs * max(1, math.ceil(len(train_examples) / eff_batch))
        report, rows = run_lora_eval(
            args, run_name, train_examples, train_labels, eval_examples, output_dir, eval3_examples,
            curriculum_order=curriculum_orders.get(run_name),
            sample_weights=run_weights.get(run_name),
        )
        prediction_columns[run_name] = rows
        run_reports[run_name] = report
    args.max_train_steps = fixed_steps

    write_predictions(output_dir / "eval_predictions.csv", eval_examples, prediction_columns, weak_eval_rows)
    # Optional: untuned-strong (base) yes/no P(correct) on each strong_train example, so analyze_rp
    # can test whether rp selects points the strong BASE can already (nearly) learn (overlap).
    base_train_probs = None
    if getattr(args, "score_base_on_train", False):
        from run_dream_w2s_baselines import load_strong_model_and_tokenizer
        base_model, base_tok = load_strong_model_and_tokenizer(args, trainable_lora=False)
        _, base_rows = evaluate_yes_no(
            base_model, base_tok, lora_examples["strong_train"],
            max(int(args.strong_batch_size), 8), device, args.max_length, "score base on strong_train",
        )
        base_train_probs = np.asarray([float(r["prob_label1"]) for r in base_rows], dtype=np.float64)
        del base_model
        clear_memory()
    write_strong_train_labels(output_dir / "strong_train_labels.csv", strong_examples, weak_probs_strong, residuals, knn_stats, rp_scores=rp_scores, base_probs=base_train_probs)
    write_knn_diagnostics(
        output_dir / "knn_diagnostics.csv",
        strong_examples,
        strong_train_labels,
        weak_probs_strong,
        residuals,
        knn_stats,
    )
    write_subset_csv(output_dir / "train_subsets.csv", {name: run_subsets[name] for name in run_subsets if name in run_reports})

    random_accs = [
        run_reports[name]["eval"]["accuracy"]
        for name in random_run_names
        if name in run_reports
    ]
    random_unbalanced_accs = [
        run_reports[name]["eval"]["accuracy"]
        for name in random_unbalanced_run_names
        if name in run_reports
    ]
    random_unbalanced_summary = None
    if random_unbalanced_accs:
        random_unbalanced_summary = {
            "count": len(random_unbalanced_accs),
            "accuracy_mean": float(np.mean(random_unbalanced_accs)),
            "accuracy_std": float(np.std(random_unbalanced_accs)),
            "accuracy_min": float(np.min(random_unbalanced_accs)),
            "accuracy_max": float(np.max(random_unbalanced_accs)),
        }
    random_summary = None
    if random_accs:
        random_summary = {
            "count": len(random_accs),
            "accuracy_mean": float(np.mean(random_accs)),
            "accuracy_std": float(np.std(random_accs)),
            "accuracy_min": float(np.min(random_accs)),
            "accuracy_max": float(np.max(random_accs)),
        }

    subset_summaries = {
        name: summarize_train_subset(*subset)
        for name, subset in run_subsets.items()
    }
    subset_summaries["middle_residual_diagnostics"] = subset_summary(
        middle_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["confidence_middle_diagnostics"] = subset_summary(
        confidence_middle_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["confidence_high_diagnostics"] = subset_summary(
        confidence_high_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["knn_middle_diagnostics"] = subset_summary(
        knn_middle_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["knn_mixed_diagnostics"] = subset_summary(
        knn_mixed_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["committee_agree_diagnostics"] = subset_summary(
        committee_agree_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["committee_disagree_diagnostics"] = subset_summary(
        committee_disagree_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )

    fmt = format_summary(args)
    summary = build_summary(
        args=args,
        best_map=best_map,
        best_row=best_row,
        committee_agree_filter=committee_agree_filter,
        committee_disagree_filter=committee_disagree_filter,
        committee_disagreement_strong=committee_disagreement_strong,
        confidence_high_filter=confidence_high_filter,
        confidence_middle_filter=confidence_middle_filter,
        fmt=fmt,
        knn_middle_filter=knn_middle_filter,
        knn_mixed_filter=knn_mixed_filter,
        knn_stats=knn_stats,
        middle_filter=middle_filter,
        output_dir=output_dir,
        random_summary=random_summary,
        random_unbalanced_summary=random_unbalanced_summary,
        run_reports=run_reports,
        splits=splits,
        start=start,
        strong_train_labels=strong_train_labels,
        subset_summaries=subset_summaries,
        test_labels=test_labels,
        weak_correct_weak_train=weak_correct_weak_train,
        weak_eval3_acc=weak_eval3_acc,
        weak_preds_strong=weak_preds_strong,
        weak_preds_test=weak_preds_test,
        weak_probs_strong=weak_probs_strong,
    )

    if args.save_activations:
        torch.save(
            {
                "weak_activations": weak_acts,
                "splits": {
                    name: getattr(splits, name).to_pandas()
                    for name in ["weak_train", "strong_train", "val", "test"]
                },
            },
            output_dir / "activations.pt",
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_text_report(output_dir / "paper_style_lora_report.txt", summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
