#!/usr/bin/env bash
set -euo pipefail

# Stable Dream LoRA filtering comparison with the revised kNN score.
#
# kNN score:
#   reference set = weak_train examples embedded with the strong model
#   query set     = strong_train examples embedded with the strong model
#   score         = fraction of each query's k nearest weak_train neighbors
#                   that the weak model got correct
#
# This avoids using strong_train correctness labels as a selection signal.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-results/dream_lora_knn_filtering_comparison/dream_seed42}"

RUNS="${RUNS:-base,ground_truth,weak_label,confidence_middle_balanced,knn_middle_balanced,random_balanced}"
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
RANDOM_CONTROL_COUNT="${RANDOM_CONTROL_COUNT:-3}"

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
RANDOM_CONTROL_COUNT="$RANDOM_CONTROL_COUNT" \
bash scripts/run_dream_paper_style_lora.sh
