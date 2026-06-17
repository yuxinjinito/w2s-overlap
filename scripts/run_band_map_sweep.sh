#!/usr/bin/env bash
set -euo pipefail

# Band-map sweep (multi-seed, overnight).
#
# One comparison set across all selection signals at 50% kept (balanced, cross-fitted):
#   - confidence (weak binary |p-0.5|), conf_entropy (weak K-way option entropy),
#     excess-loss el (H_weak - H_strong), kNN neighbor-correct-rate, rp (representation
#     projection), L2-residual middle -- each HIGH/LOW (+ knn MIXED) band;
#   - controls: weak_label (no-filter), weak_label_balanced, random_balanced/unbalanced;
#   - curriculum_easy/hard (ordering, not filtering);
#   - weak_correct_only(+balanced) = the ORACLE "filter only correct weak points" ceiling.
# base / ground_truth are run once via BASELINE_RUNS in the inner sweep.
#
# Which BAND of which SIGNAL (if any) beats matched random, and the noise ceiling.
# Robust metrics (AUROC, prior-matched accuracy, PGR) via scripts/compute_pgr.py.
# Weak/strong default to the same-family Qwen2.5 pair (run_dream_paper_style_lora.sh).
#
# 19 filter runs/seed. Defaults to 8 LoRA seeds. Quick look:
#   TRAIN_SEEDS=42 bash scripts/run_band_map_sweep.sh
#   DATASET=anli ANLI_ROUND=r2 TRAIN_SEEDS=42 bash scripts/run_band_map_sweep.sh

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
# EPOCHS>0 => train each run for that many full passes over its ACTUAL subset
# (no-filter -> all data, f50 -> the real 50%), instead of the fixed compute-matched
# MAX_TRAIN_STEPS cap. Leave 0 to keep the old compute-matched behaviour.
export EPOCHS="${EPOCHS:-0}"

# All selection signals' HIGH/LOW bands (+ knn MIXED) at 50% kept, size-matched random,
# curriculum ordering, and the weak_correct_only oracle ceiling. el_* / conf_entropy_* load
# the weak + untuned-strong models once (needs the dataset formatter to emit mc_options).
export FILTER_RUNS="${FILTER_RUNS:-weak_label,weak_label_balanced,weak_correct_only,weak_correct_only_balanced,curriculum_easy,curriculum_hard,confidence_high_balanced_f50,conf_entropy_high_balanced_f50,conf_entropy_low_balanced_f50,el_high_balanced_f50,el_low_balanced_f50,knn_high_balanced_f50,knn_mixed_balanced_f50,knn_low_balanced_f50,rp_high_balanced_f50,rp_low_balanced_f50,middle_balanced,random_balanced_f50,random_unbalanced_f50}"

echo "=== Band-map sweep (${DATASET}, cross-fit) ==="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TRAIN_SEEDS=${TRAIN_SEEDS}"
echo "COMMITTEE_KEEP_FRACS=${COMMITTEE_KEEP_FRACS}"
echo "FILTER_RUNS=${FILTER_RUNS}"

exec bash scripts/run_sciq_lora_formal_sweep.sh
