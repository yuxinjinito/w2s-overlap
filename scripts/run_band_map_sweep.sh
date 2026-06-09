#!/usr/bin/env bash
set -euo pipefail

# Band-map sweep (multi-seed, overnight).
#
# For BOTH selection signals -- weak-model confidence and kNN neighbor-correct-rate
# -- keep the HIGH / LOW / overlap(MIXED) band at 20% and 50% kept, each against a
# SIZE-MATCHED random_balanced, plus the full weak_label anchor. All balanced,
# cross-fitted reference labels.
#
# Characterizes easy / overlap / hard for the strong model: which BAND of which
# SIGNAL (if any) beats matched random, and at which fraction. Robust metrics
# (AUROC, prior-matched accuracy, PGR) via scripts/compute_pgr.py.
#
# 13 runs/seed. Defaults to 8 LoRA seeds (~8-9h on the shared GPU). Quick look:
#   TRAIN_SEEDS=42 bash scripts/run_band_map_sweep.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DATASET="${DATASET:-dream}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/${DATASET}_band_map_0608}"
export TRAIN_SEEDS="${TRAIN_SEEDS:-42 123 456 7 99 2024 31 777}"
export N_TRAIN="${N_TRAIN:-10000}"
export N_VAL="${N_VAL:-1000}"
export N_TEST="${N_TEST:-5000}"
export KNN_REFERENCE_CROSS_FIT="${KNN_REFERENCE_CROSS_FIT:-1}"
export CROSS_FIT_FOLDS="${CROSS_FIT_FOLDS:-5}"
export COMMITTEE_MEMBERS="${COMMITTEE_MEMBERS:-8}"
export COMMITTEE_KEEP_FRACS="${COMMITTEE_KEEP_FRACS:-0.2,0.5}"

# {confidence, knn} x {high, low} + knn_mixed(overlap) at 20% & 50%, each with a
# size-matched random_balanced, plus the full weak_label anchor.
export FILTER_RUNS="${FILTER_RUNS:-confidence_high_balanced_f20,confidence_low_balanced_f20,knn_high_balanced_f20,knn_low_balanced_f20,knn_mixed_balanced_f20,random_balanced_f20,confidence_high_balanced_f50,confidence_low_balanced_f50,knn_high_balanced_f50,knn_low_balanced_f50,knn_mixed_balanced_f50,random_balanced_f50,weak_label}"

echo "=== Band-map sweep (${DATASET}, cross-fit) ==="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TRAIN_SEEDS=${TRAIN_SEEDS}"
echo "COMMITTEE_KEEP_FRACS=${COMMITTEE_KEEP_FRACS}"
echo "FILTER_RUNS=${FILTER_RUNS}"

exec bash scripts/run_sciq_lora_formal_sweep.sh
