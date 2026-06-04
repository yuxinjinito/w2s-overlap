#!/usr/bin/env bash
set -euo pipefail

# Continue from a completed Dream W2S baseline run and train extra weak-label
# baselines on residual-matched / residual-filtered subsets.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-results/w2s_dream_baselines/dream_residual_filter_mid50_0528}"
RESIDUAL_FILTER_CSV="${RESIDUAL_FILTER_CSV:-results/representation_mapping/dream_twitter_0526/dream_qwen05_to_llama31_8b_map_2048_stabilized/best_map_residuals.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-results/w2s_dream_baselines/dream_residual_filter_extras_0528}"

RUNS="${RUNS:-matched_all,matched_balanced,middle_balanced,random_balanced}"
RESIDUAL_SCORE_COL="${RESIDUAL_SCORE_COL:-residual_l2}"
RESIDUAL_KEEP_MIDDLE_FRAC="${RESIDUAL_KEEP_MIDDLE_FRAC:-0.5}"
RESIDUAL_FILTER_MAP_TRAIN="${RESIDUAL_FILTER_MAP_TRAIN:-all}"
MIN_RESIDUAL_FILTER_EXAMPLES="${MIN_RESIDUAL_FILTER_EXAMPLES:-32}"
RANDOM_CONTROL_COUNT="${RANDOM_CONTROL_COUNT:-3}"
RANDOM_CONTROL_SIZE="${RANDOM_CONTROL_SIZE:-}"

STRONG_MODEL="${STRONG_MODEL:-meta-llama/Llama-3.1-8B}"
SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-384}"
STRONG_BATCH_SIZE="${STRONG_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"

EXTRA_RANDOM_ARGS=()
if [[ -n "$RANDOM_CONTROL_SIZE" ]]; then
  EXTRA_RANDOM_ARGS=(--random-control-size "$RANDOM_CONTROL_SIZE")
fi

echo "=== Dream residual-filter extras ==="
python3 scripts/run_dream_residual_filter_extras.py \
  --baseline-output-dir "$BASELINE_OUTPUT_DIR" \
  --residual-filter-csv "$RESIDUAL_FILTER_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --runs "$RUNS" \
  --residual-score-col "$RESIDUAL_SCORE_COL" \
  --residual-keep-middle-frac "$RESIDUAL_KEEP_MIDDLE_FRAC" \
  --residual-filter-map-train "$RESIDUAL_FILTER_MAP_TRAIN" \
  --min-residual-filter-examples "$MIN_RESIDUAL_FILTER_EXAMPLES" \
  --strong-model "$STRONG_MODEL" \
  --seed "$SEED" \
  --max-length "$MAX_LENGTH" \
  --strong-batch-size "$STRONG_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --lr "$LR" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --torch-dtype "$TORCH_DTYPE" \
  --random-control-count "$RANDOM_CONTROL_COUNT" \
  "${EXTRA_RANDOM_ARGS[@]}"

echo "=== Done ==="
echo "Summary: ${OUTPUT_DIR}/summary.json"
