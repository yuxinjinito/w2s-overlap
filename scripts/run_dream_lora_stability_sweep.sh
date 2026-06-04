#!/usr/bin/env bash
set -euo pipefail

# Small diagnostic sweep for the Dream LoRA setup.
#
# Goal: first make ground-truth LoRA stable. Do not interpret residual or weak
# confidence filtering until ground-truth LoRA beats the base strong model
# without collapsing to one prediction class.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_OUT="${BASE_OUT:-results/dream_paper_style_lora_tuning}"
RUNS="${RUNS:-base,ground_truth}"
NO_PLOTS="${NO_PLOTS:-1}"

run_config() {
  local name="$1"
  local lr="$2"
  local steps="$3"
  local modules="$4"
  local rank="$5"
  local alpha="$6"
  local warmup="$7"
  local grad_norm="$8"

  echo "=== ${name} ==="
  OUTPUT_DIR="${BASE_OUT}/${name}" \
  RUNS="$RUNS" \
  NO_PLOTS="$NO_PLOTS" \
  LR="$lr" \
  MAX_TRAIN_STEPS="$steps" \
  LORA_TARGET_MODULES="$modules" \
  LORA_R="$rank" \
  LORA_ALPHA="$alpha" \
  WARMUP_STEPS="$warmup" \
  MAX_GRAD_NORM="$grad_norm" \
  WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}" \
  bash scripts/run_dream_paper_style_lora.sh
}

# Start conservative: fewer trainable modules plus lower LR.
run_config "gt_qv_lr5e-5_steps100" "5e-5" "100" "q_proj,v_proj" "8" "16" "10" "1.0"
run_config "gt_qv_lr2e-5_steps200" "2e-5" "200" "q_proj,v_proj" "8" "16" "20" "1.0"

# If q/v only underfits, this checks whether all projection/MLP adapters help
# once LR is lowered and clipping/warmup are enabled.
run_config "gt_all_lr2e-5_steps200" "2e-5" "200" "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" "8" "16" "20" "1.0"
