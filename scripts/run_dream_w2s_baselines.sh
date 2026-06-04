#!/usr/bin/env bash
set -euo pipefail

# First downstream Dream W2S generalization pass.
#
# Runs:
# - base Llama 3.1 8B on Dream candidate-correctness eval;
# - Llama 3.1 8B LoRA trained on ground-truth labels;
# - Llama 3.1 8B LoRA trained on weak labels from a Qwen0.5B activation probe.
# - optionally, Llama 3.1 8B LoRA trained on a residual-filtered weak-label subset.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WEAK_MODEL="${WEAK_MODEL:-Qwen/Qwen1.5-0.5B}"
STRONG_MODEL="${STRONG_MODEL:-meta-llama/Llama-3.1-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-results/w2s_dream_baselines/dream_lora_first_pass}"

WEAK_TRAIN_LIMIT="${WEAK_TRAIN_LIMIT:-2048}"
STRONG_TRAIN_LIMIT="${STRONG_TRAIN_LIMIT:-512}"
EVAL_LIMIT="${EVAL_LIMIT:-256}"
SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-384}"

WEAK_BATCH_SIZE="${WEAK_BATCH_SIZE:-8}"
WEAK_PROBE_EPOCHS="${WEAK_PROBE_EPOCHS:-300}"

STRONG_BATCH_SIZE="${STRONG_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"

RESIDUAL_FILTER_CSV="${RESIDUAL_FILTER_CSV:-}"
RESIDUAL_SCORE_COL="${RESIDUAL_SCORE_COL:-residual_l2}"
RESIDUAL_KEEP_MIDDLE_FRAC="${RESIDUAL_KEEP_MIDDLE_FRAC:-0.5}"
RESIDUAL_FILTER_MAP_TRAIN="${RESIDUAL_FILTER_MAP_TRAIN:-all}"
RESIDUAL_RUN_NAME="${RESIDUAL_RUN_NAME:-weak_label_residual_middle_trained}"
MIN_RESIDUAL_FILTER_EXAMPLES="${MIN_RESIDUAL_FILTER_EXAMPLES:-32}"

EXTRA_ARGS=()
if [[ -n "$RESIDUAL_FILTER_CSV" ]]; then
  EXTRA_ARGS+=(
    --residual-filter-csv "$RESIDUAL_FILTER_CSV"
    --residual-score-col "$RESIDUAL_SCORE_COL"
    --residual-keep-middle-frac "$RESIDUAL_KEEP_MIDDLE_FRAC"
    --residual-filter-map-train "$RESIDUAL_FILTER_MAP_TRAIN"
    --residual-run-name "$RESIDUAL_RUN_NAME"
    --min-residual-filter-examples "$MIN_RESIDUAL_FILTER_EXAMPLES"
  )
fi

echo "=== Environment check ==="
python3 - <<'PY'
import importlib.util
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"gpu_total_memory_gb={props.total_memory / 1024**3:.2f}")
print(f"peft_available={importlib.util.find_spec('peft') is not None}")
PY

echo "=== Dream W2S baselines ==="
python3 scripts/run_dream_w2s_baselines.py \
  --weak-model "$WEAK_MODEL" \
  --strong-model "$STRONG_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --weak-train-limit "$WEAK_TRAIN_LIMIT" \
  --strong-train-limit "$STRONG_TRAIN_LIMIT" \
  --eval-limit "$EVAL_LIMIT" \
  --seed "$SEED" \
  --max-length "$MAX_LENGTH" \
  --weak-batch-size "$WEAK_BATCH_SIZE" \
  --weak-probe-epochs "$WEAK_PROBE_EPOCHS" \
  --strong-batch-size "$STRONG_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --lr "$LR" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --torch-dtype "$TORCH_DTYPE" \
  "${EXTRA_ARGS[@]}"

echo "=== Done ==="
echo "Summary: ${OUTPUT_DIR}/summary.json"
echo "Predictions: ${OUTPUT_DIR}/eval_predictions.csv"
