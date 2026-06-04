#!/usr/bin/env bash
set -euo pipefail

# Fast PAWS prompt smoke test.
#
# Goal:
#   Check whether PAWS has a usable weak/kNN reference signal before running a
#   full LoRA filtering sweep.
#
# This intentionally runs no LoRA training. It extracts weak/strong
# activations, fits the weak probe, fits the representation map, and writes
# weak-label / kNN diagnostics.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-results/paws_paper_prompt_smoke_0604}"

WEAK_MODEL="${WEAK_MODEL:-Qwen/Qwen1.5-0.5B}"
STRONG_MODEL="${STRONG_MODEL:-meta-llama/Llama-3.1-8B}"

N_TRAIN="${N_TRAIN:-800}"
N_VAL="${N_VAL:-200}"
N_TEST="${N_TEST:-300}"
SEED="${SEED:-42}"

MAX_LENGTH="${MAX_LENGTH:-384}"
ACTIVATION_MAX_LENGTH="${ACTIVATION_MAX_LENGTH:-384}"
ANSWER_SUFFIX="${ANSWER_SUFFIX:-$'\nA:'}"

WEAK_BATCH_SIZE="${WEAK_BATCH_SIZE:-4}"
ACTIVATION_BATCH_SIZE="${ACTIVATION_BATCH_SIZE:-1}"
STRONG_BATCH_SIZE="${STRONG_BATCH_SIZE:-1}"

L2_PENALTY="${L2_PENALTY:-1e-3}"
MAX_ITER="${MAX_ITER:-10000}"
RIDGE_VALUES="${RIDGE_VALUES:-100.0}"
PCA_DIMS="${PCA_DIMS:-}"
BEST_BY="${BEST_BY:-heldout_median}"

KNN_K="${KNN_K:-20}"
KNN_KEEP_MIDDLE_FRAC="${KNN_KEEP_MIDDLE_FRAC:-0.5}"
KNN_MIXED_CENTER="${KNN_MIXED_CENTER:-0.5}"
CONFIDENCE_KEEP_FRAC="${CONFIDENCE_KEEP_FRAC:-0.5}"
RESIDUAL_KEEP_MIDDLE_FRAC="${RESIDUAL_KEEP_MIDDLE_FRAC:-0.5}"

TORCH_DTYPE="${TORCH_DTYPE:-float16}"
DEVICE="${DEVICE:-auto}"

echo "=== PAWS paper-prompt smoke ==="
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "N_TRAIN=${N_TRAIN}"
echo "N_VAL=${N_VAL}"
echo "N_TEST=${N_TEST}"
echo "SEED=${SEED}"
echo "Prompt format: Sent 1 / Sent 2 / semantic-equivalence question"
echo "LoRA runs: none"

python3 scripts/run_dream_paper_style_lora.py \
  --dataset paws \
  --weak-model "$WEAK_MODEL" \
  --strong-model "$STRONG_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --n-train "$N_TRAIN" \
  --n-val "$N_VAL" \
  --n-test "$N_TEST" \
  --seed "$SEED" \
  --max-length "$MAX_LENGTH" \
  --activation-max-length "$ACTIVATION_MAX_LENGTH" \
  --answer-suffix "$ANSWER_SUFFIX" \
  --weak-batch-size "$WEAK_BATCH_SIZE" \
  --activation-batch-size "$ACTIVATION_BATCH_SIZE" \
  --l2-penalty "$L2_PENALTY" \
  --max-iter "$MAX_ITER" \
  --strong-batch-size "$STRONG_BATCH_SIZE" \
  --gradient-accumulation-steps 4 \
  --max-train-steps 1 \
  --lr 2e-5 \
  --weight-decay 0.0 \
  --warmup-steps 0 \
  --max-grad-norm 1.0 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --torch-dtype "$TORCH_DTYPE" \
  --device "$DEVICE" \
  --runs "" \
  --residual-keep-middle-frac "$RESIDUAL_KEEP_MIDDLE_FRAC" \
  --confidence-keep-frac "$CONFIDENCE_KEEP_FRAC" \
  --knn-k "$KNN_K" \
  --knn-keep-middle-frac "$KNN_KEEP_MIDDLE_FRAC" \
  --knn-mixed-center "$KNN_MIXED_CENTER" \
  --random-control-count 0 \
  --ridge-values "$RIDGE_VALUES" \
  --pca-dims "$PCA_DIMS" \
  --best-by "$BEST_BY" \
  --no-plots

echo "=== Smoke done ==="
echo "Summary:        ${OUTPUT_DIR}/summary.json"
echo "Text report:    ${OUTPUT_DIR}/paper_style_lora_report.txt"
echo "kNN diagnostics:${OUTPUT_DIR}/knn_diagnostics.csv"

