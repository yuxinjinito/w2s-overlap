#!/usr/bin/env bash
set -euo pipefail

# Fast kNN-saturation diagnostic -- NO LoRA training.
#
# Computes, for in-sample vs cross-fitted weak_train reference labels:
#   - weak_train reference accuracy (how often the weak probe is right), and
#   - kNN saturation: the fraction of strong_train points whose k nearest
#     weak_train neighbors (in strong embedding space) are ALL weak-correct.
# Then it exits before any LoRA training. Use this to check whether
# cross-fitting de-saturates the kNN signal *before* spending GPU time on the
# full downstream LoRA comparison (run_dream_lora_knn_crossfit_comparison.sh).
#
# Cost is just model loading + activation extraction (minutes), not LoRA.
#
# Usage:
#   bash scripts/run_knn_saturation_diagnostic.sh
#   DATASET=sciq OUTPUT_DIR=results/knn_saturation_diag/sciq bash scripts/run_knn_saturation_diagnostic.sh
#   DATASET=paws OUTPUT_DIR=results/knn_saturation_diag/paws bash scripts/run_knn_saturation_diagnostic.sh
#
# Output: ${OUTPUT_DIR}/knn_saturation_diagnostic.json and a printed one-line summary.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-dream}"
OUTPUT_DIR="${OUTPUT_DIR:-results/knn_saturation_diag/${DATASET}_seed42}"

# Diagnostic only: skip all LoRA training. 'base' keeps the run list valid.
DIAGNOSTICS_ONLY="${DIAGNOSTICS_ONLY:-1}"
RUNS="${RUNS:-base}"
NO_PLOTS="${NO_PLOTS:-1}"

# Modest sizes keep it fast; bump these for numbers matching the full run.
N_TRAIN="${N_TRAIN:-4000}"
N_VAL="${N_VAL:-500}"
N_TEST="${N_TEST:-1000}"

KNN_K="${KNN_K:-20}"
CROSS_FIT_FOLDS="${CROSS_FIT_FOLDS:-5}"

DATASET="$DATASET" \
OUTPUT_DIR="$OUTPUT_DIR" \
DIAGNOSTICS_ONLY="$DIAGNOSTICS_ONLY" \
RUNS="$RUNS" \
NO_PLOTS="$NO_PLOTS" \
N_TRAIN="$N_TRAIN" \
N_VAL="$N_VAL" \
N_TEST="$N_TEST" \
KNN_K="$KNN_K" \
CROSS_FIT_FOLDS="$CROSS_FIT_FOLDS" \
bash scripts/run_dream_paper_style_lora.sh
