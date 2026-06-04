#!/usr/bin/env bash
set -euo pipefail

# Dream/SciQ/PAWS LoRA kNN comparison with CROSS-FITTED reference labels.
#
# Motivation: the weak probe is trained on weak_train, and the kNN reference
# points (weak_train) were previously labeled weak-correct/weak-wrong using that
# same probe in-sample. The probe is near-perfect in-sample, so almost every
# reference point is weak-correct and the kNN mixed-neighborhood signal
# saturates (observed on SciQ/PAWS). This wrapper enables out-of-fold
# (cross-fitted) reference labeling so genuine weak-wrong reference points
# reappear and the mixed-neighborhood signal can exist again.
#
# The run prints in-sample vs cross-fitted weak_train reference accuracy, so the
# de-saturation effect is visible before looking at downstream accuracy.
#
# Usage:
#   bash scripts/run_dream_lora_knn_crossfit_comparison.sh
#   DATASET=sciq OUTPUT_DIR=results/sciq_lora_knn_crossfit bash scripts/run_dream_lora_knn_crossfit_comparison.sh
#   DATASET=paws OUTPUT_DIR=results/paws_lora_knn_crossfit bash scripts/run_dream_lora_knn_crossfit_comparison.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-dream}"
OUTPUT_DIR="${OUTPUT_DIR:-results/dream_lora_knn_crossfit_comparison/${DATASET}_seed42}"

RUNS="${RUNS:-base,weak_label,confidence_middle_balanced,knn_mixed_unbalanced,knn_mixed_balanced,random_balanced}"
NO_PLOTS="${NO_PLOTS:-1}"

# Cross-fitted kNN reference labeling (the point of this wrapper).
KNN_REFERENCE_CROSS_FIT="${KNN_REFERENCE_CROSS_FIT:-1}"
CROSS_FIT_FOLDS="${CROSS_FIT_FOLDS:-5}"

# Stable LoRA setup, matching the earlier kNN comparison runs.
LR="${LR:-2e-5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

CONFIDENCE_KEEP_FRAC="${CONFIDENCE_KEEP_FRAC:-0.5}"
KNN_K="${KNN_K:-20}"
KNN_KEEP_MIDDLE_FRAC="${KNN_KEEP_MIDDLE_FRAC:-0.5}"
KNN_MIXED_CENTER="${KNN_MIXED_CENTER:-0.5}"
RANDOM_CONTROL_COUNT="${RANDOM_CONTROL_COUNT:-3}"
TRAIN_SEED="${TRAIN_SEED:-}"

DATASET="$DATASET" \
OUTPUT_DIR="$OUTPUT_DIR" \
RUNS="$RUNS" \
NO_PLOTS="$NO_PLOTS" \
KNN_REFERENCE_CROSS_FIT="$KNN_REFERENCE_CROSS_FIT" \
CROSS_FIT_FOLDS="$CROSS_FIT_FOLDS" \
LR="$LR" \
MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
WARMUP_STEPS="$WARMUP_STEPS" \
MAX_GRAD_NORM="$MAX_GRAD_NORM" \
WEIGHT_DECAY="$WEIGHT_DECAY" \
LORA_TARGET_MODULES="$LORA_TARGET_MODULES" \
CONFIDENCE_KEEP_FRAC="$CONFIDENCE_KEEP_FRAC" \
KNN_K="$KNN_K" \
KNN_KEEP_MIDDLE_FRAC="$KNN_KEEP_MIDDLE_FRAC" \
KNN_MIXED_CENTER="$KNN_MIXED_CENTER" \
RANDOM_CONTROL_COUNT="$RANDOM_CONTROL_COUNT" \
TRAIN_SEED="$TRAIN_SEED" \
bash scripts/run_dream_paper_style_lora.sh
