#!/usr/bin/env bash
set -euo pipefail

# Paper-faithful Dream rerun for Figure A1-style comparison.
#
# This is intentionally separate from the LoRA/generative yes-no Dream scripts.
# It follows the original repo's Dream formatter, split logic, final-token
# activation extraction, and logistic linear-probe W2S setup.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WEAK_MODEL="${WEAK_MODEL:-Qwen/Qwen1.5-0.5B}"
STRONG_MODEL="${STRONG_MODEL:-meta-llama/Llama-3.1-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-results/dream_paper_linear_probe/dream_seed42}"

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
SAVE_ACTIVATIONS="${SAVE_ACTIVATIONS:-0}"
DRY_RUN_SPLITS="${DRY_RUN_SPLITS:-0}"

EXTRA_ARGS=()
if [[ -n "$MAX_LENGTH" ]]; then
  EXTRA_ARGS+=(--max-length "$MAX_LENGTH")
fi
if [[ "$SAVE_ACTIVATIONS" == "1" ]]; then
  EXTRA_ARGS+=(--save-activations)
fi
if [[ "$DRY_RUN_SPLITS" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run-splits)
fi

echo "=== Paper-faithful Dream linear-probe rerun ==="
python3 scripts/run_dream_paper_linear_probe.py \
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
  "${EXTRA_ARGS[@]}"

echo "=== Done ==="
echo "Summary:             ${OUTPUT_DIR}/summary.json"
echo "Test predictions:    ${OUTPUT_DIR}/test_predictions.csv"
echo "Strong-train labels: ${OUTPUT_DIR}/strong_train_weak_labels.csv"
