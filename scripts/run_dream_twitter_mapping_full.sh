#!/usr/bin/env bash
set -euo pipefail

# Run the full Llama 3.1 8B weak-to-strong representation-mapping pipeline on
# the two datasets John asked about after the weak-confidence screen.
#
# Default size matches the larger BoolQ/SciQ/PAWS pass: strong_train has
# N_TRAIN / 2 examples, then MAX_EXAMPLES selects the mapping set.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS="${DATASETS:-dream twitter-sentiment}"
OUT_DIR="${OUT_DIR:-results/representation_mapping/dream_twitter_0526}"
N_TRAIN="${N_TRAIN:-4096}"
N_VAL="${N_VAL:-512}"
N_TEST="${N_TEST:-512}"
MAX_EXAMPLES="${MAX_EXAMPLES:-2048}"
RELU_EPOCHS="${RELU_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-512}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"

for DATASET_NAME in $DATASETS; do
  echo "=== ${DATASET_NAME}: Llama 3.1 8B representation mapping ==="
  DATASET="$DATASET_NAME" \
  OUT_DIR="$OUT_DIR" \
  RUN_NAME="${DATASET_NAME}_qwen05_to_llama31_8b_map_${MAX_EXAMPLES}" \
  N_TRAIN="$N_TRAIN" \
  N_VAL="$N_VAL" \
  N_TEST="$N_TEST" \
  MAX_EXAMPLES="$MAX_EXAMPLES" \
  RELU_EPOCHS="$RELU_EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  MAX_LENGTH="$MAX_LENGTH" \
  TORCH_DTYPE="$TORCH_DTYPE" \
  CONFIDENCE_CSV="" \
  bash scripts/run_llama31_mapping_full.sh
done

echo "=== Done ==="
echo "Output dir: ${OUT_DIR}"
