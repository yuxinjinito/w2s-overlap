#!/usr/bin/env bash
set -euo pipefail

# Keep-easy comparison at 50% kept -- directly answers John's Todo #1:
#   "instead of the kNN middle 0.5, pick the top 0.5 with the most weak-correct neighbors."
#
# Methods (all balanced, cross-fitted reference labels):
#   knn_mixed_balanced      = kNN MIDDLE 50% (knn_correct_rate near 0.5; the OLD overlap method)
#   knn_high_balanced       = kNN TOP 50%    (neighbors mostly weak-correct; John's NEW "keep-easy")
#   committee_agree_balanced/disagree_balanced = the committee keep-easy / keep-boundary thread
#   random_balanced/unbalanced, weak_label     = controls / no-selection anchor
#
# AUROC is reported natively in summary.json (more stable than thresholded accuracy here).
# Defaults to a single LoRA seed for a fast first look; set TRAIN_SEEDS="42 123 456" for the full run.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DATASET="${DATASET:-dream}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/dream_keep_easy_50_0608}"
export TRAIN_SEEDS="${TRAIN_SEEDS:-42}"
export N_TRAIN="${N_TRAIN:-10000}"
export N_VAL="${N_VAL:-1000}"
export N_TEST="${N_TEST:-5000}"
export RANDOM_CONTROL_COUNT="${RANDOM_CONTROL_COUNT:-3}"
export KNN_REFERENCE_CROSS_FIT="${KNN_REFERENCE_CROSS_FIT:-1}"
export CROSS_FIT_FOLDS="${CROSS_FIT_FOLDS:-5}"
export COMMITTEE_MEMBERS="${COMMITTEE_MEMBERS:-8}"

export FILTER_RUNS="${FILTER_RUNS:-knn_mixed_balanced,knn_high_balanced,committee_agree_balanced,committee_disagree_balanced,random_unbalanced,random_balanced,weak_label}"

echo "=== Keep-easy 50% comparison (${DATASET}, cross-fit) ==="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TRAIN_SEEDS=${TRAIN_SEEDS}"
echo "FILTER_RUNS=${FILTER_RUNS}"

exec bash scripts/run_sciq_lora_formal_sweep.sh
