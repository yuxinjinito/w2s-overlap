#!/usr/bin/env bash
set -euo pipefail

# Stable Dream LoRA filtering comparison.
#
# This uses the best config from the 05/29 stability sweep, then compares:
# - all weak-label training,
# - middle L2-residual filtering,
# - middle weak-confidence filtering,
# - high weak-confidence filtering,
# - random weak-label-balanced controls.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-results/dream_lora_filtering_comparison/dream_seed42}"

RUNS="${RUNS:-base,ground_truth,weak_label,middle_balanced,confidence_middle_balanced,confidence_high_balanced,random_balanced}"
NO_PLOTS="${NO_PLOTS:-1}"

LR="${LR:-2e-5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

RESIDUAL_KEEP_MIDDLE_FRAC="${RESIDUAL_KEEP_MIDDLE_FRAC:-0.5}"
CONFIDENCE_KEEP_FRAC="${CONFIDENCE_KEEP_FRAC:-0.5}"
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
RESIDUAL_KEEP_MIDDLE_FRAC="$RESIDUAL_KEEP_MIDDLE_FRAC" \
CONFIDENCE_KEEP_FRAC="$CONFIDENCE_KEEP_FRAC" \
RANDOM_CONTROL_COUNT="$RANDOM_CONTROL_COUNT" \
bash scripts/run_dream_paper_style_lora.sh
