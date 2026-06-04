#!/usr/bin/env bash
set -euo pipefail

# Clean Dream residual-filtering experiment.
#
# Design:
# 1. Run the canonical Dream W2S baselines on one fixed split.
# 2. Fit the weak-to-strong representation map on the same split's weak_train.
# 3. Score residuals on the same split's strong_train candidates.
# 4. Train weak-label LoRA controls using residual-middle filtering and
#    same-size random weak-label-balanced controls.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_DIR="${BASE_DIR:-results/w2s_dream_aligned}"
BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-${BASE_DIR}/baseline_2048}"
MAPPING_OUTPUT_DIR="${MAPPING_OUTPUT_DIR:-${BASE_DIR}/aligned_mapping_2048}"
FILTER_OUTPUT_DIR="${FILTER_OUTPUT_DIR:-${BASE_DIR}/filter_controls_mid50}"

WEAK_TRAIN_LIMIT="${WEAK_TRAIN_LIMIT:-2048}"
STRONG_TRAIN_LIMIT="${STRONG_TRAIN_LIMIT:-2048}"
EVAL_LIMIT="${EVAL_LIMIT:-256}"
SEED="${SEED:-42}"

MAX_LENGTH="${MAX_LENGTH:-384}"
MAPPING_MAX_LENGTH="${MAPPING_MAX_LENGTH:-512}"
STRONG_BATCH_SIZE="${STRONG_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"

STABILIZED_RIDGE_VALUES="${STABILIZED_RIDGE_VALUES:-100.0}"
STABILIZED_PCA_DIMS="${STABILIZED_PCA_DIMS:-512}"
RUNS="${RUNS:-matched_all,middle_unbalanced,middle_balanced,random_balanced}"
RANDOM_CONTROL_COUNT="${RANDOM_CONTROL_COUNT:-3}"

echo "=== Step 1/3: Dream W2S baseline on fixed split ==="
OUTPUT_DIR="$BASELINE_OUTPUT_DIR" \
WEAK_TRAIN_LIMIT="$WEAK_TRAIN_LIMIT" \
STRONG_TRAIN_LIMIT="$STRONG_TRAIN_LIMIT" \
EVAL_LIMIT="$EVAL_LIMIT" \
SEED="$SEED" \
MAX_LENGTH="$MAX_LENGTH" \
STRONG_BATCH_SIZE="$STRONG_BATCH_SIZE" \
GRAD_ACCUM="$GRAD_ACCUM" \
MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
TORCH_DTYPE="$TORCH_DTYPE" \
bash scripts/run_dream_w2s_baselines.sh

echo "=== Step 2/3: Aligned weak-to-strong residuals ==="
OUT_DIR="$MAPPING_OUTPUT_DIR" \
WEAK_TRAIN_LIMIT="$WEAK_TRAIN_LIMIT" \
STRONG_TRAIN_LIMIT="$STRONG_TRAIN_LIMIT" \
EVAL_LIMIT="$EVAL_LIMIT" \
SEED="$SEED" \
MAX_LENGTH="$MAPPING_MAX_LENGTH" \
TORCH_DTYPE="$TORCH_DTYPE" \
STABILIZED_RIDGE_VALUES="$STABILIZED_RIDGE_VALUES" \
STABILIZED_PCA_DIMS="$STABILIZED_PCA_DIMS" \
bash scripts/run_dream_aligned_residual_mapping.sh

echo "=== Step 3/3: Residual-filtered weak-label controls ==="
BASELINE_OUTPUT_DIR="$BASELINE_OUTPUT_DIR" \
RESIDUAL_FILTER_CSV="${MAPPING_OUTPUT_DIR}/stabilized/best_map_residuals.csv" \
RESIDUAL_FILTER_MAP_TRAIN="heldout" \
OUTPUT_DIR="$FILTER_OUTPUT_DIR" \
RUNS="$RUNS" \
RANDOM_CONTROL_COUNT="$RANDOM_CONTROL_COUNT" \
SEED="$SEED" \
MAX_LENGTH="$MAX_LENGTH" \
STRONG_BATCH_SIZE="$STRONG_BATCH_SIZE" \
GRAD_ACCUM="$GRAD_ACCUM" \
MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
TORCH_DTYPE="$TORCH_DTYPE" \
bash scripts/run_dream_residual_filter_extras.sh

echo "=== Clean Dream experiment complete ==="
echo "Baseline summary: ${BASELINE_OUTPUT_DIR}/summary.json"
echo "Mapping summary:  ${MAPPING_OUTPUT_DIR}/dream_aligned_mapping_summary.json"
echo "Map analysis:     ${MAPPING_OUTPUT_DIR}/stabilized/stabilized_map_report.txt"
echo "Filter summary:   ${FILTER_OUTPUT_DIR}/summary.json"
