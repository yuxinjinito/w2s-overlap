#!/usr/bin/env bash
set -euo pipefail

# Compute weak-to-strong mapping residuals on the exact Dream W2S split:
# weak_train fits the map, strong_train receives held-out residual scores.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WEAK_MODEL="${WEAK_MODEL:-Qwen/Qwen1.5-0.5B}"
STRONG_MODEL="${STRONG_MODEL:-meta-llama/Llama-3.1-8B}"
OUT_DIR="${OUT_DIR:-results/w2s_dream_aligned/aligned_mapping_2048}"

WEAK_TRAIN_LIMIT="${WEAK_TRAIN_LIMIT:-2048}"
STRONG_TRAIN_LIMIT="${STRONG_TRAIN_LIMIT:-2048}"
EVAL_LIMIT="${EVAL_LIMIT:-256}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-512}"
POOLING="${POOLING:-mean}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"

# Use the previously best Dream setting as the default to avoid tuning the map
# on the same candidate pool we later filter.
STABILIZED_RIDGE_VALUES="${STABILIZED_RIDGE_VALUES:-100.0}"
STABILIZED_PCA_DIMS="${STABILIZED_PCA_DIMS:-512}"

METADATA_CSV="${OUT_DIR}/dream_aligned_mapping_metadata.csv"
SUMMARY_JSON="${OUT_DIR}/dream_aligned_mapping_summary.json"
EMBEDDING_PT="${OUT_DIR}/dream_aligned_mapping_embeddings.pt"
STABILIZED_DIR="${OUT_DIR}/stabilized"

mkdir -p "$OUT_DIR"

echo "=== Aligned Dream residual mapping ==="
python3 scripts/run_dream_aligned_residual_mapping.py \
  --weak-model "$WEAK_MODEL" \
  --strong-model "$STRONG_MODEL" \
  --weak-train-limit "$WEAK_TRAIN_LIMIT" \
  --strong-train-limit "$STRONG_TRAIN_LIMIT" \
  --eval-limit "$EVAL_LIMIT" \
  --seed "$SEED" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --pooling "$POOLING" \
  --torch-dtype "$TORCH_DTYPE" \
  --output-csv "$METADATA_CSV" \
  --embedding-output "$EMBEDDING_PT" \
  --summary-output "$SUMMARY_JSON"

echo "=== Stabilized residual scoring ==="
python3 scripts/analyze_stabilized_maps.py \
  "$EMBEDDING_PT" \
  --mapping-csv "$METADATA_CSV" \
  --output-dir "$STABILIZED_DIR" \
  --ridge-values "$STABILIZED_RIDGE_VALUES" \
  --pca-dims "$STABILIZED_PCA_DIMS"

echo "=== Done ==="
echo "Summary:          $SUMMARY_JSON"
echo "Metadata CSV:     $METADATA_CSV"
echo "Embeddings:       $EMBEDDING_PT"
echo "Residual CSV:     ${STABILIZED_DIR}/best_map_residuals.csv"
