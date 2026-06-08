#!/usr/bin/env bash
set -euo pipefail

# Committee selection KEEP-FRACTION sweep with cross-fitted reference labels.
#
# Produces a label-complexity / data-efficiency curve: accuracy vs the fraction of
# strong_train kept, for
#   - committee_agree    (keep the most RELIABLE = lowest committee disagreement)
#   - knn_high           (keep the EASY points = neighbors mostly weak-correct; John's ask)
#   - committee_disagree (keep the most BOUNDARY = highest committee disagreement)
#   - random_balanced    (matched control at each fraction)
# All share one activation/cross-fit/committee pass per seed (cheap to add fracs).
#
# Questions answered: (1) does reliable selection ever beat random at any fraction?
# (2) data efficiency -- how few reliable points match random-0.5 / full weak-label?
# (3) does boundary selection stay <= random at all fractions (active-learning
# noisy-regime prediction)?
#
# Defaults to Dream, cross-fit, 3 seeds (~10h on the shared GPU). Quick look:
# TRAIN_SEEDS=42 bash scripts/run_committee_frac_sweep.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DATASET="${DATASET:-dream}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/${DATASET}_committee_frac_lora_0607}"
export TRAIN_SEEDS="${TRAIN_SEEDS:-42 123 456}"
export N_TRAIN="${N_TRAIN:-10000}"
export N_VAL="${N_VAL:-1000}"
export N_TEST="${N_TEST:-5000}"
export KNN_REFERENCE_CROSS_FIT="${KNN_REFERENCE_CROSS_FIT:-1}"
export CROSS_FIT_FOLDS="${CROSS_FIT_FOLDS:-5}"
export COMMITTEE_MEMBERS="${COMMITTEE_MEMBERS:-8}"
export COMMITTEE_KEEP_FRACS="${COMMITTEE_KEEP_FRACS:-0.1,0.2,0.3,0.45,0.6,0.8}"

# 17 runs/seed: 6-point agree curve + 6-point matched random + 2 boundary anchors
# + 2 agree-unbalanced checks + full weak-label anchor.
export FILTER_RUNS="${FILTER_RUNS:-committee_agree_balanced_f10,committee_agree_balanced_f20,committee_agree_balanced_f30,committee_agree_balanced_f45,committee_agree_balanced_f60,committee_agree_balanced_f80,knn_high_balanced_f10,knn_high_balanced_f20,knn_high_balanced_f30,knn_high_balanced_f45,knn_high_balanced_f60,knn_high_balanced_f80,random_balanced_f10,random_balanced_f20,random_balanced_f30,random_balanced_f45,random_balanced_f60,random_balanced_f80,committee_disagree_balanced_f20,committee_disagree_balanced_f45,weak_label}"

echo "=== Committee keep-fraction sweep (${DATASET}, cross-fit) ==="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TRAIN_SEEDS=${TRAIN_SEEDS}"
echo "COMMITTEE_KEEP_FRACS=${COMMITTEE_KEEP_FRACS}"
echo "FILTER_RUNS=${FILTER_RUNS}"

exec bash scripts/run_sciq_lora_formal_sweep.sh
