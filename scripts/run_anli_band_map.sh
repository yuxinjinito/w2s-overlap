#!/usr/bin/env bash
set -euo pipefail
#
# Canonical ANLI band-map run (aligned, reproducible).
#
# Thin wrapper over run_band_map_sweep.sh that PINS the settled ANLI setup so every round
# and rerun is apples-to-apples and immune to default drift. One command per round.
#
# Aligned choices (do not change without re-ablating):
#   - same-family Qwen2.5 0.5B -> 7B (weak last-token probe -> strong LoRA).
#   - representation layer   = END (last hidden): ablated END > MIDDLE for rp.
#   - representation pooling = LAST token: ablated LAST > MEAN. MEAN-pooling inverts the rp
#     high/low direction (rp_low > rp_high), i.e. it destroys the signal -- so keep LAST.
#   - 3-class multiple-choice eval, single 50% keep band, 2 epochs over each subset.
#   - 19 hard/control arms. Continuous weighting (rp/el/conf_weighted) is INTENTIONALLY
#     excluded here: it is a temperature curve, not a single point, and is run as its own
#     dedicated T-sweep (WEIGHT_TEMPERATURE), then reported as a separate figure.
#
# Usage:
#   bash scripts/run_anli_band_map.sh                                 # r2, seeds 42 123 456
#   ANLI_ROUND=r3 bash scripts/run_anli_band_map.sh                   # the 2nd clean testbed
#   ANLI_ROUND=r2 TRAIN_SEEDS="42 123 456 7 99 2024 31" \
#     bash scripts/run_anli_band_map.sh                               # 7-seed headline precision

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DATASET=anli
export ANLI_ROUND="${ANLI_ROUND:-r2}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/anli_${ANLI_ROUND}_band_map_$(date +%m%d)}"
export TRAIN_SEEDS="${TRAIN_SEEDS:-42 123 456}"

# data sizes (clean-testbed default)
export N_TRAIN="${N_TRAIN:-6000}"
export N_VAL="${N_VAL:-1000}"
export N_TEST="${N_TEST:-1000}"

# 3-class MC eval + per-subset epochs + single 50% band
export EVAL_3CLASS="${EVAL_3CLASS:-1}"
export N_EVAL_QUESTIONS="${N_EVAL_QUESTIONS:-1000}"
export EPOCHS="${EPOCHS:-2}"
export COMMITTEE_KEEP_FRACS="${COMMITTEE_KEEP_FRACS:-0.5}"

# settled representation choices (explicit + self-documenting; see header)
export REPRESENTATION_LAYER="${REPRESENTATION_LAYER:-end}"
export REPRESENTATION_POOLING="${REPRESENTATION_POOLING:-last}"
# kNN/RP representation span: 'full' (whole prompt, settled default) or 'answer' (answer-only
# ablation -- mask the question; weak probe/labels stay on the full prompt either way).
export REPRESENTATION_SPAN="${REPRESENTATION_SPAN:-full}"

# 19 hard/control arms (weighted arms excluded by design -- see header)
export FILTER_RUNS="${FILTER_RUNS:-weak_label,weak_label_balanced,weak_correct_only,weak_correct_only_balanced,curriculum_easy,curriculum_hard,confidence_high_balanced_f50,conf_entropy_high_balanced_f50,conf_entropy_low_balanced_f50,el_high_balanced_f50,el_low_balanced_f50,knn_high_balanced_f50,knn_mixed_balanced_f50,knn_low_balanced_f50,rp_high_balanced_f50,rp_low_balanced_f50,middle_balanced,random_balanced_f50,random_unbalanced_f50}"

echo "=== Canonical ANLI band-map (${ANLI_ROUND}) ==="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TRAIN_SEEDS=${TRAIN_SEEDS}  EPOCHS=${EPOCHS}  KEEP_FRACS=${COMMITTEE_KEEP_FRACS}"
echo "rep layer=${REPRESENTATION_LAYER}  pooling=${REPRESENTATION_POOLING}  (settled by ablation)"

exec bash scripts/run_band_map_sweep.sh
