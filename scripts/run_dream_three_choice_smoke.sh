#!/usr/bin/env bash
set -euo pipefail

# Fast Dream 3-choice / open-answer-format smoke test.
#
# Goal:
#   Before spending time on a full LoRA sweep, check whether moving Dream from
#   binary candidate-correctness to true A/B/C answer selection keeps a usable
#   weak-correct / kNN signal.
#
# This intentionally runs no LoRA training.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-results/dream_three_choice_smoke_0604}"

WEAK_MODEL="${WEAK_MODEL:-Qwen/Qwen1.5-0.5B}"
STRONG_MODEL="${STRONG_MODEL:-meta-llama/Llama-3.1-8B}"

N_TRAIN="${N_TRAIN:-800}"
N_VAL="${N_VAL:-200}"
N_TEST="${N_TEST:-300}"
SEED="${SEED:-42}"

BATCH_SIZE="${BATCH_SIZE:-1}"
WEAK_LOGPROB_BATCH_SIZE="${WEAK_LOGPROB_BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-384}"
L2_PENALTY="${L2_PENALTY:-1e-3}"
MAX_ITER="${MAX_ITER:-10000}"
KNN_K="${KNN_K:-20}"
WEAK_CORRECT_SOURCE="${WEAK_CORRECT_SOURCE:-logprob}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
DEVICE="${DEVICE:-auto}"
NO_SHUFFLE_CHOICES="${NO_SHUFFLE_CHOICES:-0}"
SAVE_ACTIVATIONS="${SAVE_ACTIVATIONS:-0}"

EXTRA_ARGS=()
if [[ "$NO_SHUFFLE_CHOICES" == "1" ]]; then
  EXTRA_ARGS+=(--no-shuffle-choices)
fi
if [[ "$SAVE_ACTIVATIONS" == "1" ]]; then
  EXTRA_ARGS+=(--save-activations)
fi

echo "=== Dream 3-choice smoke ==="
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "N_TRAIN=${N_TRAIN}"
echo "N_VAL=${N_VAL}"
echo "N_TEST=${N_TEST}"
echo "SEED=${SEED}"
echo "WEAK_CORRECT_SOURCE=${WEAK_CORRECT_SOURCE}"
echo "Prompt format: dialogue + question + choices A/B/C + Answer:"
echo "LoRA runs: none"

python3 scripts/run_dream_three_choice_smoke.py \
  --weak-model "$WEAK_MODEL" \
  --strong-model "$STRONG_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --n-train "$N_TRAIN" \
  --n-val "$N_VAL" \
  --n-test "$N_TEST" \
  --seed "$SEED" \
  --batch-size "$BATCH_SIZE" \
  --weak-logprob-batch-size "$WEAK_LOGPROB_BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --l2-penalty "$L2_PENALTY" \
  --max-iter "$MAX_ITER" \
  --knn-k "$KNN_K" \
  --weak-correct-source "$WEAK_CORRECT_SOURCE" \
  --torch-dtype "$TORCH_DTYPE" \
  --device "$DEVICE" \
  "${EXTRA_ARGS[@]}"

echo "=== Smoke done ==="
echo "Summary:      ${OUTPUT_DIR}/summary.json"
echo "Text report:  ${OUTPUT_DIR}/three_choice_smoke_report.txt"
echo "Predictions:  ${OUTPUT_DIR}/weak_predictions.csv"
echo "kNN details:  ${OUTPUT_DIR}/strong_train_knn_diagnostics.csv"
