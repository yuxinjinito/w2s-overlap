#!/usr/bin/env bash
set -euo pipefail

# SciQ downstream W2S rerun with the same candidate-correctness structure as
# the current paper-style Dream LoRA experiments. This intentionally does not
# switch SciQ to open-ended generation; we keep SciQ in the same
# structure for the first cross-dataset check.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-results/sciq_paper_style_lora/sciq_seed42}"

# SciQ train has enough examples for the paper-style 10k/1k split; test has
# 1000 examples, so N_TEST defaults to 1000.
N_TRAIN="${N_TRAIN:-10000}"
N_VAL="${N_VAL:-1000}"
N_TEST="${N_TEST:-1000}"
SEED="${SEED:-42}"
TRAIN_SEED="${TRAIN_SEED:-42}"

# Stable Dream LoRA settings from the latest three-seed filtering runs.
LR="${LR:-2e-5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

# First SciQ pass: keep this small enough to confirm cross-dataset behavior.
RUNS="${RUNS:-base,ground_truth,weak_label,confidence_middle_balanced,knn_mixed_unbalanced,random_balanced}"
RANDOM_CONTROL_COUNT="${RANDOM_CONTROL_COUNT:-1}"

DATASET=sciq \
OUTPUT_DIR="$OUTPUT_DIR" \
N_TRAIN="$N_TRAIN" \
N_VAL="$N_VAL" \
N_TEST="$N_TEST" \
SEED="$SEED" \
TRAIN_SEED="$TRAIN_SEED" \
LR="$LR" \
MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
WARMUP_STEPS="$WARMUP_STEPS" \
MAX_GRAD_NORM="$MAX_GRAD_NORM" \
WEIGHT_DECAY="$WEIGHT_DECAY" \
RUNS="$RUNS" \
RANDOM_CONTROL_COUNT="$RANDOM_CONTROL_COUNT" \
bash scripts/run_dream_paper_style_lora.sh
