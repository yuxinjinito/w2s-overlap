#!/usr/bin/env bash
set -euo pipefail

# Tiny LoRA fine-tuning smoke test for John todo #2.
#
# This only checks whether a 4B/8B-scale strong model can complete a few
# parameter-efficient training steps on the current GPU. It is not intended as a
# meaningful fine-tuning experiment.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-results/finetune_smoke/llama31_8b_lora_smoke}"
MAX_STEPS="${MAX_STEPS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_LENGTH="${MAX_LENGTH:-128}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"

echo "=== Environment check ==="
python3 - <<'PY'
import importlib.util
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"gpu_total_memory_gb={props.total_memory / 1024**3:.2f}")
print(f"peft_available={importlib.util.find_spec('peft') is not None}")
PY

echo "=== LoRA fine-tuning smoke test ==="
python3 scripts/run_strong_lora_smoke.py \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps "$MAX_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --max-length "$MAX_LENGTH" \
  --lr "$LR" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --torch-dtype "$TORCH_DTYPE"

echo "=== Done ==="
echo "Report: ${OUTPUT_DIR}/smoke_report.json"
