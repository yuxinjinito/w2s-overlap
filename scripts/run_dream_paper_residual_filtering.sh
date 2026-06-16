#!/usr/bin/env bash
set -euo pipefail

# Paper-style Dream residual-filtering rerun.
#
# This keeps the Figure A1-style linear-probe setup from
# `run_dream_paper_linear_probe.sh`, then adds the targeted filtering
# check: middle residual examples versus same-size random balanced controls.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WEAK_MODEL="${WEAK_MODEL:-Qwen/Qwen1.5-0.5B}"
STRONG_MODEL="${STRONG_MODEL:-meta-llama/Llama-3.1-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-results/dream_paper_residual_filtering/dream_seed42}"

N_TRAIN="${N_TRAIN:-10000}"
N_VAL="${N_VAL:-1000}"
N_TEST="${N_TEST:-5000}"
SEED="${SEED:-42}"

BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-}"
L2_PENALTY="${L2_PENALTY:-1e-3}"
MAX_ITER="${MAX_ITER:-10000}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
DEVICE="${DEVICE:-auto}"

RESIDUAL_KEEP_MIDDLE_FRAC="${RESIDUAL_KEEP_MIDDLE_FRAC:-0.5}"
RANDOM_CONTROL_COUNT="${RANDOM_CONTROL_COUNT:-10}"
RANDOM_CONTROL_SIZE="${RANDOM_CONTROL_SIZE:-}"

# Defaults match the most recent Dream mapping choice: raw ridge with ridge=100.
# Add PCA dims/ridge sweeps by overriding these env vars if needed.
RIDGE_VALUES="${RIDGE_VALUES:-100.0}"
PCA_DIMS="${PCA_DIMS:-}"
BEST_BY="${BEST_BY:-heldout_median}"
SAVE_ACTIVATIONS="${SAVE_ACTIVATIONS:-0}"
NO_PLOTS="${NO_PLOTS:-0}"

EXTRA_ARGS=()
if [[ -n "$MAX_LENGTH" ]]; then
  EXTRA_ARGS+=(--max-length "$MAX_LENGTH")
fi
if [[ -n "$RANDOM_CONTROL_SIZE" ]]; then
  EXTRA_ARGS+=(--random-control-size "$RANDOM_CONTROL_SIZE")
fi
if [[ "$SAVE_ACTIVATIONS" == "1" ]]; then
  EXTRA_ARGS+=(--save-activations)
fi
if [[ "$NO_PLOTS" == "1" ]]; then
  EXTRA_ARGS+=(--no-plots)
fi

echo "=== Paper-style Dream residual filtering ==="
python3 scripts/run_dream_paper_residual_filtering.py \
  --weak-model "$WEAK_MODEL" \
  --strong-model "$STRONG_MODEL" \
  --n-train "$N_TRAIN" \
  --n-val "$N_VAL" \
  --n-test "$N_TEST" \
  --seed "$SEED" \
  --batch-size "$BATCH_SIZE" \
  --l2-penalty "$L2_PENALTY" \
  --max-iter "$MAX_ITER" \
  --torch-dtype "$TORCH_DTYPE" \
  --device "$DEVICE" \
  --output-dir "$OUTPUT_DIR" \
  --residual-keep-middle-frac "$RESIDUAL_KEEP_MIDDLE_FRAC" \
  --random-control-count "$RANDOM_CONTROL_COUNT" \
  --ridge-values "$RIDGE_VALUES" \
  --pca-dims "$PCA_DIMS" \
  --best-by "$BEST_BY" \
  "${EXTRA_ARGS[@]}"

echo "=== Done ==="
echo "Summary:      ${OUTPUT_DIR}/summary.json"
echo "Text report:  ${OUTPUT_DIR}/paper_residual_filtering_report.txt"
echo "Selection:    ${OUTPUT_DIR}/strong_train_residual_selection.csv"
echo "Map report:   ${OUTPUT_DIR}/map/stabilized_map_report.txt"
