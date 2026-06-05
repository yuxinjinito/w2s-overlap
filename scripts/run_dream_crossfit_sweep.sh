#!/usr/bin/env bash
set -euo pipefail

# Dream LoRA filtering sweep with CROSS-FITTED kNN reference labels (3 seeds).
#
# Why:
#   The original Dream kNN-mixed result (kNN-mixed / confidence-middle beating
#   random) used IN-SAMPLE reference labels, where the weak probe is near-perfect
#   on the weak_train reference set (weak_train_accuracy ~0.86 vs an honest
#   ~0.61). This re-runs Dream with out-of-fold (cross-fitted) reference labels,
#   the same honest setup used for the SciQ sweep, to test whether the kNN-mixed
#   advantage survives once the reference labels are honest or whether it was an
#   in-sample-saturation artifact.
#
# Matches the original Dream kNN run (n_train=10000, n_test=5000, lr=2e-5,
# steps=200, k=20, keep=0.5, 3 random controls) and adds cross-fitting + 3 LoRA
# seeds, so the table is directly comparable to both the in-sample Dream result
# and the SciQ cross-fit sweep.
#
# Quick single-seed look:
#   TRAIN_SEEDS=42 bash scripts/run_dream_crossfit_sweep.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DATASET="dream"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/dream_crossfit_lora_0606}"
export TRAIN_SEEDS="${TRAIN_SEEDS:-42 123 456}"
export N_TRAIN="${N_TRAIN:-10000}"
export N_VAL="${N_VAL:-1000}"
export N_TEST="${N_TEST:-5000}"
export RANDOM_CONTROL_COUNT="${RANDOM_CONTROL_COUNT:-3}"
export KNN_REFERENCE_CROSS_FIT="${KNN_REFERENCE_CROSS_FIT:-1}"
export CROSS_FIT_FOLDS="${CROSS_FIT_FOLDS:-5}"

echo "=== Dream cross-fitted kNN LoRA sweep ==="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TRAIN_SEEDS=${TRAIN_SEEDS}"
echo "N_TEST=${N_TEST}"
echo "RANDOM_CONTROL_COUNT=${RANDOM_CONTROL_COUNT}"
echo "KNN_REFERENCE_CROSS_FIT=${KNN_REFERENCE_CROSS_FIT}"

exec bash scripts/run_sciq_lora_formal_sweep.sh
