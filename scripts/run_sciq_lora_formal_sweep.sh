#!/usr/bin/env bash
set -euo pipefail

# Formal SciQ LoRA filtering sweep.
#
# Purpose:
#   Test whether the Dream result transfers to SciQ under the original
#   datacentric_w2s no-support binary candidate-correctness structure.
#
# This is not a smoke test. It uses the stable Dream LoRA hyperparameters and
# runs multiple LoRA initialization seeds for the filtering methods.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/sciq_no_support_lora_formal_0604}"
BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-${OUTPUT_ROOT}/baseline_seed42}"
FILTER_OUTPUT_ROOT="${FILTER_OUTPUT_ROOT:-${OUTPUT_ROOT}/filtering}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42 123 456}"

# Match the current paper-style / candidate-correctness setup used for Dream.
DATASET="${DATASET:-sciq}"
N_TRAIN="${N_TRAIN:-10000}"
N_VAL="${N_VAL:-1000}"
N_TEST="${N_TEST:-1000}"
SEED="${SEED:-42}"
BASELINE_TRAIN_SEED="${BASELINE_TRAIN_SEED:-42}"
ANSWER_SUFFIX="${ANSWER_SUFFIX:-$'\nIs the candidate answer correct? Answer:'}"
SCIQ_USE_SUPPORT="${SCIQ_USE_SUPPORT:-0}"

# Stable Dream LoRA settings from the successful kNN/residual-vs-confidence runs.
LR="${LR:-2e-5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

# Same selection knobs as the Dream kNN mixed experiment.
CONFIDENCE_KEEP_FRAC="${CONFIDENCE_KEEP_FRAC:-0.5}"
KNN_K="${KNN_K:-20}"
KNN_KEEP_MIDDLE_FRAC="${KNN_KEEP_MIDDLE_FRAC:-0.5}"
KNN_MIXED_CENTER="${KNN_MIXED_CENTER:-0.5}"
RANDOM_CONTROL_COUNT="${RANDOM_CONTROL_COUNT:-1}"

# Baselines are run once. Filtering methods are run across TRAIN_SEEDS.
BASELINE_RUNS="${BASELINE_RUNS:-base,ground_truth}"
FILTER_RUNS="${FILTER_RUNS:-weak_label,confidence_middle_balanced,knn_mixed_unbalanced,knn_mixed_balanced,random_unbalanced,random_balanced}"
NO_PLOTS="${NO_PLOTS:-1}"

echo "=== Formal SciQ LoRA sweep ==="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "DATASET=${DATASET}"
echo "N_TRAIN=${N_TRAIN}"
echo "N_VAL=${N_VAL}"
echo "N_TEST=${N_TEST}"
echo "SEED=${SEED}"
echo "SCIQ_USE_SUPPORT=${SCIQ_USE_SUPPORT}"
echo "BASELINE_TRAIN_SEED=${BASELINE_TRAIN_SEED}"
echo "TRAIN_SEEDS=${TRAIN_SEEDS}"
echo "BASELINE_RUNS=${BASELINE_RUNS}"
echo "FILTER_RUNS=${FILTER_RUNS}"
echo "RANDOM_CONTROL_COUNT=${RANDOM_CONTROL_COUNT}"
echo "LR=${LR}"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "MAX_GRAD_NORM=${MAX_GRAD_NORM}"
echo "WEIGHT_DECAY=${WEIGHT_DECAY}"
echo "LORA_R=${LORA_R}"
echo "LORA_ALPHA=${LORA_ALPHA}"
echo "LORA_DROPOUT=${LORA_DROPOUT}"
echo "LORA_TARGET_MODULES=${LORA_TARGET_MODULES}"

echo "=== Running SciQ baseline once ==="
DATASET="${DATASET}" \
OUTPUT_DIR="${BASELINE_OUTPUT_DIR}" \
N_TRAIN="${N_TRAIN}" \
N_VAL="${N_VAL}" \
N_TEST="${N_TEST}" \
SEED="${SEED}" \
TRAIN_SEED="${BASELINE_TRAIN_SEED}" \
ANSWER_SUFFIX="${ANSWER_SUFFIX}" \
RUNS="${BASELINE_RUNS}" \
RANDOM_CONTROL_COUNT="${RANDOM_CONTROL_COUNT}" \
NO_PLOTS="${NO_PLOTS}" \
LR="${LR}" \
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
WARMUP_STEPS="${WARMUP_STEPS}" \
MAX_GRAD_NORM="${MAX_GRAD_NORM}" \
WEIGHT_DECAY="${WEIGHT_DECAY}" \
LORA_R="${LORA_R}" \
LORA_ALPHA="${LORA_ALPHA}" \
LORA_DROPOUT="${LORA_DROPOUT}" \
LORA_TARGET_MODULES="${LORA_TARGET_MODULES}" \
CONFIDENCE_KEEP_FRAC="${CONFIDENCE_KEEP_FRAC}" \
KNN_K="${KNN_K}" \
KNN_KEEP_MIDDLE_FRAC="${KNN_KEEP_MIDDLE_FRAC}" \
KNN_MIXED_CENTER="${KNN_MIXED_CENTER}" \
SCIQ_USE_SUPPORT="${SCIQ_USE_SUPPORT}" \
EVAL_3CLASS="${EVAL_3CLASS:-0}" \
N_EVAL_QUESTIONS="${N_EVAL_QUESTIONS:-2041}" \
bash scripts/run_dream_paper_style_lora.sh

test -f "${BASELINE_OUTPUT_DIR}/summary.json"

for TS in ${TRAIN_SEEDS}; do
  echo "=== Running SciQ filtering TRAIN_SEED=${TS} ==="

  OUTPUT_DIR="${FILTER_OUTPUT_ROOT}/trainseed_${TS}" \
  DATASET="${DATASET}" \
  N_TRAIN="${N_TRAIN}" \
  N_VAL="${N_VAL}" \
  N_TEST="${N_TEST}" \
  SEED="${SEED}" \
  TRAIN_SEED="${TS}" \
  ANSWER_SUFFIX="${ANSWER_SUFFIX}" \
  RUNS="${FILTER_RUNS}" \
  RANDOM_CONTROL_COUNT="${RANDOM_CONTROL_COUNT}" \
  NO_PLOTS="${NO_PLOTS}" \
  LR="${LR}" \
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
  WARMUP_STEPS="${WARMUP_STEPS}" \
  MAX_GRAD_NORM="${MAX_GRAD_NORM}" \
  WEIGHT_DECAY="${WEIGHT_DECAY}" \
  LORA_R="${LORA_R}" \
  LORA_ALPHA="${LORA_ALPHA}" \
  LORA_DROPOUT="${LORA_DROPOUT}" \
  LORA_TARGET_MODULES="${LORA_TARGET_MODULES}" \
  CONFIDENCE_KEEP_FRAC="${CONFIDENCE_KEEP_FRAC}" \
  KNN_K="${KNN_K}" \
  KNN_KEEP_MIDDLE_FRAC="${KNN_KEEP_MIDDLE_FRAC}" \
  KNN_MIXED_CENTER="${KNN_MIXED_CENTER}" \
  SCIQ_USE_SUPPORT="${SCIQ_USE_SUPPORT}" \
  EVAL_3CLASS="${EVAL_3CLASS:-0}" \
  N_EVAL_QUESTIONS="${N_EVAL_QUESTIONS:-2041}" \
  bash scripts/run_dream_paper_style_lora.sh

  test -f "${FILTER_OUTPUT_ROOT}/trainseed_${TS}/summary.json"
done

echo "=== Formal SciQ LoRA sweep done ==="
echo "Baseline summary: ${BASELINE_OUTPUT_DIR}/summary.json"
echo "Filtering root:   ${FILTER_OUTPUT_ROOT}"
