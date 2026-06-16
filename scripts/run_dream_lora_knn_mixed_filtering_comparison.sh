#!/usr/bin/env bash
set -euo pipefail

# Dream LoRA filtering comparison focused on the revised kNN idea.
#
# Difference from the previous kNN-middle run:
#   - previous knn_middle_balanced kept the middle quantile of kNN scores;
#   - this run adds knn_mixed_balanced, which keeps examples whose
#     correct-neighbor rate is closest to 0.5.
#
# This is closer to the "mixed neighborhood" intuition: easy-like points should
# have mostly weak-correct neighbors, hard-like points should have mostly
# weak-wrong neighbors, and overlap-like points should have a mixture.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-results/dream_lora_knn_mixed_filtering_comparison/dream_seed42}"

RUNS="${RUNS:-base,weak_label,confidence_middle_balanced,knn_mixed_unbalanced,knn_mixed_balanced,random_balanced}"
NO_PLOTS="${NO_PLOTS:-1}"

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

OUTPUT_DIR="$OUTPUT_DIR" \
RUNS="$RUNS" \
NO_PLOTS="$NO_PLOTS" \
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
