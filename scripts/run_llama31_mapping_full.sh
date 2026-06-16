#!/usr/bin/env bash
set -euo pipefail

# Full first-pass weak-to-strong representation mapping run using the
# paper-aligned Llama 3.1 8B strong model.
#
# This script keeps the lessons from the Qwen1.8B proxy run:
# - save the optimization matrix A, not only per-sample losses;
# - analyze held-out residuals separately from mapping-training examples;
# - compare residuals with weak confidence when the confidence CSV exists;
# - run a small ridge sweep from saved embeddings to check overfitting/stability
#   without re-extracting Llama activations.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-boolq}"
WEAK_MODEL="${WEAK_MODEL:-Qwen/Qwen1.5-0.5B}"
STRONG_MODEL="${STRONG_MODEL:-meta-llama/Llama-3.1-8B}"
TARGET_SPLIT="${TARGET_SPLIT:-strong_train}"

N_TRAIN="${N_TRAIN:-1024}"
N_VAL="${N_VAL:-128}"
N_TEST="${N_TEST:-128}"
MAX_EXAMPLES="${MAX_EXAMPLES:-512}"
SEED="${SEED:-42}"

BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-512}"
POOLING="${POOLING:-mean}"
MAP_TRAIN_FRAC="${MAP_TRAIN_FRAC:-0.5}"
RIDGE="${RIDGE:-1e-3}"

RELU_HIDDEN_DIM="${RELU_HIDDEN_DIM:-256}"
RELU_EPOCHS="${RELU_EPOCHS:-200}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
STABILIZED_RIDGE_VALUES="${STABILIZED_RIDGE_VALUES:-0.1,1.0,10.0,100.0}"
STABILIZED_PCA_DIMS="${STABILIZED_PCA_DIMS:-64,128,256,512}"

OUT_DIR="${OUT_DIR:-results/representation_mapping/llama31_8b_full}"
RUN_NAME="${RUN_NAME:-${DATASET}_qwen05_to_llama31_8b_map_${MAX_EXAMPLES}}"
if [[ -z "${CONFIDENCE_CSV:-}" ]]; then
  if [[ "$DATASET" == "boolq" ]]; then
    CONFIDENCE_CSV="results/reference_style/boolq_qwen05_weakprobe_on_strong_train.csv"
  else
    CONFIDENCE_CSV=""
  fi
fi

CSV_OUTPUT="${OUT_DIR}/${RUN_NAME}.csv"
SUMMARY_OUTPUT="${OUT_DIR}/${RUN_NAME}.json"
EMBEDDING_OUTPUT="${OUT_DIR}/${RUN_NAME}.pt"
MAP_OUTPUT="${OUT_DIR}/${RUN_NAME}_maps.pt"

mkdir -p "$OUT_DIR"

echo "=== Environment check ==="
python3 - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"gpu_total_memory_gb={props.total_memory / 1024**3:.2f}")
PY

if command -v hf >/dev/null 2>&1; then
  hf auth whoami || true
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli whoami || true
fi

echo "=== Representation extraction + map fitting ==="
python3 scripts/run_representation_mapping.py \
  --dataset "$DATASET" \
  --weak-model "$WEAK_MODEL" \
  --strong-model "$STRONG_MODEL" \
  --target-split "$TARGET_SPLIT" \
  --n-train "$N_TRAIN" \
  --n-val "$N_VAL" \
  --n-test "$N_TEST" \
  --max-examples "$MAX_EXAMPLES" \
  --seed "$SEED" \
  --torch-dtype "$TORCH_DTYPE" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --pooling "$POOLING" \
  --map-train-frac "$MAP_TRAIN_FRAC" \
  --ridge "$RIDGE" \
  --relu-hidden-dim "$RELU_HIDDEN_DIM" \
  --relu-epochs "$RELU_EPOCHS" \
  --output "$CSV_OUTPUT" \
  --summary-output "$SUMMARY_OUTPUT" \
  --embedding-output "$EMBEDDING_OUTPUT" \
  --map-output "$MAP_OUTPUT"

echo "=== Matrix A + residual analysis ==="
python3 scripts/analyze_map_artifacts.py \
  "$MAP_OUTPUT" \
  --mapping-csv "$CSV_OUTPUT" \
  --summary-output "${OUT_DIR}/${RUN_NAME}_map_analysis.json" \
  --report-output "${OUT_DIR}/${RUN_NAME}_map_analysis.txt" \
  --residual-output "${OUT_DIR}/${RUN_NAME}_residuals.csv" \
  --prepared-map-output "${OUT_DIR}/${RUN_NAME}_prepared_maps.pt" \
  --plot-dir "${OUT_DIR}/${RUN_NAME}_map_plots" \
  --heldout-only

if [[ -n "$CONFIDENCE_CSV" && -f "$CONFIDENCE_CSV" ]]; then
  echo "=== Mapping loss vs weak confidence analysis ==="
  python3 scripts/analyze_mapping_vs_confidence.py \
    "$CSV_OUTPUT" \
    --confidence-csv "$CONFIDENCE_CSV" \
    --merge-on id \
    --primary-loss linear_l2 \
    --summary-output "${OUT_DIR}/${RUN_NAME}_vs_confidence_heldout.json" \
    --report-output "${OUT_DIR}/${RUN_NAME}_vs_confidence_heldout.txt" \
    --merged-output "${OUT_DIR}/${RUN_NAME}_vs_confidence_heldout.csv" \
    --plot-output "${OUT_DIR}/${RUN_NAME}_vs_confidence_heldout.png" \
    --plot-heldout-only
else
  echo "Skipping confidence comparison because this file was not found:"
  echo "${CONFIDENCE_CSV:-<not configured for dataset ${DATASET}>}"
fi

echo "=== Ridge stability sweep from saved embeddings ==="
for SWEEP_RIDGE in 1e-3 1e-2 1e-1 1.0; do
  SAFE_RIDGE="$(echo "$SWEEP_RIDGE" | tr '.-' 'pm')"
  SWEEP_DIR="${OUT_DIR}/${RUN_NAME}_ridge_${SAFE_RIDGE}"
  mkdir -p "$SWEEP_DIR"

  python3 scripts/analyze_map_artifacts.py \
    "$EMBEDDING_OUTPUT" \
    --mapping-csv "$CSV_OUTPUT" \
    --summary-output "${SWEEP_DIR}/map_analysis.json" \
    --report-output "${SWEEP_DIR}/map_analysis.txt" \
    --residual-output "${SWEEP_DIR}/residuals.csv" \
    --prepared-map-output "${SWEEP_DIR}/maps.pt" \
    --plot-dir "${SWEEP_DIR}/plots" \
    --ridge "$SWEEP_RIDGE" \
    --heldout-only
done

echo "=== Stabilized PCA/ridge map analysis ==="
python3 scripts/analyze_stabilized_maps.py \
  "$EMBEDDING_OUTPUT" \
  --mapping-csv "$CSV_OUTPUT" \
  --output-dir "${OUT_DIR}/${RUN_NAME}_stabilized" \
  --ridge-values "$STABILIZED_RIDGE_VALUES" \
  --pca-dims "$STABILIZED_PCA_DIMS"

echo "=== Done ==="
echo "Main CSV:        $CSV_OUTPUT"
echo "Main summary:    $SUMMARY_OUTPUT"
echo "Map artifact:    $MAP_OUTPUT"
echo "Map analysis:    ${OUT_DIR}/${RUN_NAME}_map_analysis.txt"
echo "Confidence plot: ${OUT_DIR}/${RUN_NAME}_vs_confidence_heldout.png"
echo "Map plots dir:   ${OUT_DIR}/${RUN_NAME}_map_plots"
echo "Ridge sweeps:    ${OUT_DIR}/${RUN_NAME}_ridge_*"
echo "Stabilized dir:  ${OUT_DIR}/${RUN_NAME}_stabilized"
