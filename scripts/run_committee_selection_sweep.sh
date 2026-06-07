#!/usr/bin/env bash
set -euo pipefail

# Committee-disagreement (query-by-committee) selection sweep, with cross-fitted
# reference labels.
#
# Motivation (active-learning grounding):
#   Classic active learning (Hanneke's region of disagreement) selects boundary /
#   high-disagreement points -- but it assumes a CLEAN oracle gives their labels.
#   In our weak-to-strong setup the labels are FIXED and NOISY, and they are least
#   reliable exactly at the boundary. So the principled move inverts: select the
#   RELIABLE points (where a committee of weak probes AGREES), not the informative
#   boundary ones. This sweep tests that:
#     committee_agree    = keep LOW committee disagreement  (reliable weak labels)
#     committee_disagree = keep HIGH committee disagreement (boundary; theory predicts <= random)
#   The committee is `COMMITTEE_MEMBERS` bootstrap weak probes; disagreement is the
#   std of their probabilities on each strong_train point (out-of-sample, honest).
#
# Defaults to Dream (where we have the in-sample vs cross-fit baseline). Set
# DATASET=sciq to repeat there. Quick single-seed look: TRAIN_SEEDS=42 ...

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DATASET="${DATASET:-dream}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/${DATASET}_committee_lora_0607}"
export TRAIN_SEEDS="${TRAIN_SEEDS:-42 123 456}"
export N_TRAIN="${N_TRAIN:-10000}"
export N_VAL="${N_VAL:-1000}"
export N_TEST="${N_TEST:-5000}"
export RANDOM_CONTROL_COUNT="${RANDOM_CONTROL_COUNT:-3}"
export KNN_REFERENCE_CROSS_FIT="${KNN_REFERENCE_CROSS_FIT:-1}"
export CROSS_FIT_FOLDS="${CROSS_FIT_FOLDS:-5}"
export COMMITTEE_MEMBERS="${COMMITTEE_MEMBERS:-8}"
export COMMITTEE_KEEP_FRAC="${COMMITTEE_KEEP_FRAC:-0.5}"

# Committee selectors + the references we compare against (random, weak-label, prior kNN-mixed).
export FILTER_RUNS="${FILTER_RUNS:-weak_label,random_unbalanced,random_balanced,knn_mixed_unbalanced,committee_agree_balanced,committee_agree_unbalanced,committee_disagree_balanced,committee_disagree_unbalanced}"

echo "=== Committee-disagreement selection sweep (${DATASET}, cross-fit) ==="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TRAIN_SEEDS=${TRAIN_SEEDS}"
echo "COMMITTEE_MEMBERS=${COMMITTEE_MEMBERS}"
echo "FILTER_RUNS=${FILTER_RUNS}"

exec bash scripts/run_sciq_lora_formal_sweep.sh
