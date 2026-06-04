#!/usr/bin/env bash
set -euo pipefail

# Stable Dream LoRA sweep for John's residual-vs-confidence filtering question.
#
# Question:
#   Does L2 residual filtering from the weak-to-strong representation map behave
#   better than weak-confidence filtering?
#
# This wrapper intentionally pins the LoRA hyperparameters that were stable in
# the kNN mixed runs. Do not call run_dream_paper_style_lora.sh directly for
# this comparison unless you also set these values explicitly.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/dream_lora_residual_vs_confidence_stable_0601}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42 123 456}"

RUNS="${RUNS:-weak_label,middle_unbalanced,middle_balanced,confidence_middle_unbalanced,confidence_middle_balanced,random_unbalanced,random_balanced}"
RANDOM_CONTROL_COUNT="${RANDOM_CONTROL_COUNT:-1}"
NO_PLOTS="${NO_PLOTS:-1}"

# Stable LoRA settings from the successful Dream kNN mixed runs.
LR="${LR:-2e-5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

echo "=== Stable residual-vs-confidence sweep ==="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TRAIN_SEEDS=${TRAIN_SEEDS}"
echo "RUNS=${RUNS}"
echo "RANDOM_CONTROL_COUNT=${RANDOM_CONTROL_COUNT}"
echo "LR=${LR}"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "MAX_GRAD_NORM=${MAX_GRAD_NORM}"
echo "WEIGHT_DECAY=${WEIGHT_DECAY}"
echo "LORA_TARGET_MODULES=${LORA_TARGET_MODULES}"

for TS in ${TRAIN_SEEDS}; do
  echo "=== Running stable TRAIN_SEED=${TS} ==="

  OUTPUT_DIR="${OUTPUT_ROOT}/trainseed_${TS}" \
  TRAIN_SEED="${TS}" \
  RUNS="${RUNS}" \
  RANDOM_CONTROL_COUNT="${RANDOM_CONTROL_COUNT}" \
  NO_PLOTS="${NO_PLOTS}" \
  LR="${LR}" \
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
  WARMUP_STEPS="${WARMUP_STEPS}" \
  MAX_GRAD_NORM="${MAX_GRAD_NORM}" \
  WEIGHT_DECAY="${WEIGHT_DECAY}" \
  LORA_TARGET_MODULES="${LORA_TARGET_MODULES}" \
  bash scripts/run_dream_paper_style_lora.sh
done

echo "=== all stable residual-vs-confidence runs done ==="
