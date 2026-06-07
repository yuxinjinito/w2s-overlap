#!/usr/bin/env bash
set -euo pipefail

# ANLI (adversarial NLI) LoRA filtering sweep -- binary candidate-relation correctness.
#
# Motivation:
#   On SciQ the weak labels are already strong, so overlap selection (kNN-mixed /
#   confidence-middle) cannot beat a random control. ANLI is adversarial, so the
#   weak probe is near chance (~0.51): the noisy weak-label regime where overlap
#   selection should help most if the effect is real.
#
# This reuses the generic filtering sweep with DATASET=anli, honest cross-fitted
# kNN reference labels, and a single LoRA seed by default (a first look). Set
# TRAIN_SEEDS="42 123 456" for the full multi-seed comparison.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DATASET="anli"
export ANLI_ROUND="${ANLI_ROUND:-r2}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/anli_${ANLI_ROUND}_lora_0605}"
export TRAIN_SEEDS="${TRAIN_SEEDS:-42}"
export N_TEST="${N_TEST:-1000}"
export KNN_REFERENCE_CROSS_FIT="${KNN_REFERENCE_CROSS_FIT:-1}"
export CROSS_FIT_FOLDS="${CROSS_FIT_FOLDS:-5}"

echo "=== ANLI ${ANLI_ROUND} LoRA filtering sweep ==="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TRAIN_SEEDS=${TRAIN_SEEDS}"
echo "N_TEST=${N_TEST}"
echo "KNN_REFERENCE_CROSS_FIT=${KNN_REFERENCE_CROSS_FIT}"

exec bash scripts/run_sciq_lora_formal_sweep.sh
