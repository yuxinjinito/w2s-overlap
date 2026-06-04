#!/usr/bin/env bash
set -euo pipefail

# Batch weak-confidence diagnostic for 10 paper-aligned datasets.
#
# This runs the first overlap-detector-style diagnostic:
# weak final-token activations -> logistic probe -> weak confidence histogram.
#
# Default dataset set:
#   boolq, sciq, sst2, amazon_polarity, cola, wic, paws, anli-r2, hellaswag, multirc
#
# Extra paper/Figure A1 checks:
#   DATASETS="dream twitter-sentiment" OUT_DIR=results/probe_confidence/dream_twitter bash scripts/run_paper10_confidence_batch.sh
#
# Example run on a GPU machine:
#   bash scripts/run_paper10_confidence_batch.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS="${DATASETS:-boolq sciq sst2 amazon_polarity cola wic paws anli-r2 hellaswag multirc}"
OUT_DIR="${OUT_DIR:-results/probe_confidence/paper10}"
WEAK_MODEL="${WEAK_MODEL:-Qwen/Qwen1.5-0.5B}"
TRAIN_LIMIT="${TRAIN_LIMIT:-2048}"
EVAL_LIMIT="${EVAL_LIMIT:-1000}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
EPOCHS="${EPOCHS:-300}"
LR="${LR:-0.003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
N_EXAMPLES_TO_PRINT="${N_EXAMPLES_TO_PRINT:-10}"
SAVE_ACTIVATIONS="${SAVE_ACTIVATIONS:-0}"

mkdir -p "$OUT_DIR"

eval_split_for_dataset() {
  case "$1" in
    anli-r2)
      echo "test_r2"
      ;;
    amazon_polarity)
      echo "test"
      ;;
    twitter-sentiment)
      echo "test"
      ;;
    *)
      echo "validation"
      ;;
  esac
}

{
  echo "paper10 weak-confidence batch"
  echo "date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "datasets: $DATASETS"
  echo "weak_model: $WEAK_MODEL"
  echo "train_limit: $TRAIN_LIMIT"
  echo "eval_limit: $EVAL_LIMIT"
  echo "seed: $SEED"
  echo "batch_size: $BATCH_SIZE"
  echo "max_length: $MAX_LENGTH"
  echo "epochs: $EPOCHS"
  echo "lr: $LR"
  echo "weight_decay: $WEIGHT_DECAY"
  echo "torch_dtype: $TORCH_DTYPE"
  echo "save_activations: $SAVE_ACTIVATIONS"
} > "${OUT_DIR}/run_manifest.txt"

for DATASET in $DATASETS; do
  EVAL_SPLIT="$(eval_split_for_dataset "$DATASET")"
  RUN_NAME="${DATASET}_qwen05_probe_${EVAL_SPLIT}_${EVAL_LIMIT}"
  CSV_OUTPUT="${OUT_DIR}/${RUN_NAME}.csv"
  SUMMARY_OUTPUT="${OUT_DIR}/${RUN_NAME}_summary.json"
  PLOT_OUTPUT="${OUT_DIR}/${RUN_NAME}_confidence_hist.png"
  EXAMPLES_OUTPUT="${OUT_DIR}/${RUN_NAME}_confidence_examples.txt"

  echo "=== ${DATASET} (${EVAL_SPLIT}) ==="

  ACTIVATION_ARGS=()
  if [[ "$SAVE_ACTIVATIONS" == "1" ]]; then
    ACTIVATION_ARGS=(--activation-output "${OUT_DIR}/${RUN_NAME}_acts.pt")
  fi

  python3 scripts/run_probe_confidence.py \
    --dataset "$DATASET" \
    --train-split train \
    --eval-split "$EVAL_SPLIT" \
    --train-limit "$TRAIN_LIMIT" \
    --eval-limit "$EVAL_LIMIT" \
    --seed "$SEED" \
    --weak-model "$WEAK_MODEL" \
    --torch-dtype "$TORCH_DTYPE" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --shuffle-before-limit \
    --balance-labels \
    --output "$CSV_OUTPUT" \
    "${ACTIVATION_ARGS[@]}" \
    > "$SUMMARY_OUTPUT"

  python3 scripts/inspect_weak_confidence.py \
    "$CSV_OUTPUT" \
    --plot-output "$PLOT_OUTPUT" \
    --examples-output "$EXAMPLES_OUTPUT" \
    --n-examples "$N_EXAMPLES_TO_PRINT"

  python3 scripts/summarize_inference_csv.py "$CSV_OUTPUT" \
    > "${OUT_DIR}/${RUN_NAME}_inspect_summary.json"
done

echo "=== Done ==="
echo "Output dir: $OUT_DIR"
