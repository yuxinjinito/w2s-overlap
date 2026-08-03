#!/usr/bin/env bash
set -euo pipefail

# Dream downstream W2S rerun with LoRA, larger paper-style split, and
# paper-style candidate-answer text.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Activate the Python environment so `python3` has the ML stack (torch/transformers/peft).
# Lab-specific activation is kept out of the public repo: set W2S_ENV_ACTIVATE to your
# activate snippet/path, or drop a (gitignored) _private/env.sh, or keep a repo-local .venv.
# (set +u around the source: conda's activate scripts trip `set -u`.)
set +u
if [[ -n "${W2S_ENV_ACTIVATE:-}" ]]; then
  source "${W2S_ENV_ACTIVATE}"
elif [[ -f "${ROOT_DIR}/_private/env.sh" ]]; then
  source "${ROOT_DIR}/_private/env.sh"
elif [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
fi
set -u

RUN_LOCK_FILE="${RUN_LOCK_FILE:-results/.locks/paper_style_lora.lock}"
DISABLE_RUN_LOCK="${DISABLE_RUN_LOCK:-0}"
if [[ "$DISABLE_RUN_LOCK" != "1" ]]; then
  mkdir -p "$(dirname "$RUN_LOCK_FILE")"
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$RUN_LOCK_FILE"
    if ! flock -n 9; then
      echo "Another paper-style LoRA/diagnostic run is already active."
      echo "Lock file: $RUN_LOCK_FILE"
      echo "Set DISABLE_RUN_LOCK=1 only if you intentionally want concurrent heavy runs."
      exit 2
    fi
  else
    RUN_LOCK_DIR="${RUN_LOCK_FILE}.dir"
    if ! mkdir "$RUN_LOCK_DIR" 2>/dev/null; then
      echo "Another paper-style LoRA/diagnostic run is already active."
      echo "Lock dir: $RUN_LOCK_DIR"
      echo "Set DISABLE_RUN_LOCK=1 only if you intentionally want concurrent heavy runs."
      exit 2
    fi
    trap 'rmdir "$RUN_LOCK_DIR" 2>/dev/null || true' EXIT
  fi
fi

# Weak and strong should share a FAMILY / TOKENIZER: the excess-loss and conf_entropy
# selectors compare the two models' per-option distributions, which is only clean when the
# vocab is identical (the requirement is same family / same tokenizer, not Llama specifically).
# Default pair: Qwen2.5-0.5B (weak) -> Qwen2.5-7B (strong); all Qwen2.5 sizes share one
# tokenizer, both base (not -Instruct). For a Llama pair instead:
# WEAK_MODEL=meta-llama/Llama-3.2-1B STRONG_MODEL=meta-llama/Llama-3.1-8B.
# (Earlier sweeps used Qwen1.5-0.5B -> Llama-3.1-8B, a cross-family/cross-tokenizer pair.)
WEAK_MODEL="${WEAK_MODEL:-Qwen/Qwen2.5-0.5B}"
STRONG_MODEL="${STRONG_MODEL:-Qwen/Qwen2.5-7B}"
DATASET="${DATASET:-dream}"
OUTPUT_DIR="${OUTPUT_DIR:-results/dream_paper_style_lora/dream_seed42}"

N_TRAIN="${N_TRAIN:-10000}"
N_VAL="${N_VAL:-1000}"
N_TEST="${N_TEST:-5000}"
SEED="${SEED:-42}"
TRAIN_SEED="${TRAIN_SEED:-}"

MAX_LENGTH="${MAX_LENGTH:-384}"
ACTIVATION_MAX_LENGTH="${ACTIVATION_MAX_LENGTH:-}"
ANSWER_SUFFIX="${ANSWER_SUFFIX:-$'\nIs the candidate answer correct? Answer:'}"
SCIQ_USE_SUPPORT="${SCIQ_USE_SUPPORT:-0}"
ANLI_ROUND="${ANLI_ROUND:-r2}"
COMMITTEE_MEMBERS="${COMMITTEE_MEMBERS:-8}"
COMMITTEE_KEEP_FRAC="${COMMITTEE_KEEP_FRAC:-0.5}"
COMMITTEE_KEEP_FRACS="${COMMITTEE_KEEP_FRACS:-}"

WEAK_BATCH_SIZE="${WEAK_BATCH_SIZE:-4}"
ACTIVATION_BATCH_SIZE="${ACTIVATION_BATCH_SIZE:-1}"
L2_PENALTY="${L2_PENALTY:-1e-3}"
MAX_ITER="${MAX_ITER:-10000}"

STRONG_BATCH_SIZE="${STRONG_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100}"
EPOCHS="${EPOCHS:-0}"

# Optional adaptive GPU utilisation. With AUTO_BATCH=1 we size the *physical* batch to
# the GPU's free memory at launch (re-checked per seed -- each seed is a fresh process),
# while holding the EFFECTIVE batch (strong_batch_size * grad_accum) fixed via grad_accum.
# So the optimisation and #steps stay identical regardless of how busy the GPU was at
# launch; only the GPU fill (hence speed) adapts. Free GPU -> bigger physical batch ->
# higher utilisation; busy/shared GPU -> smaller batch -> no OOM. Set EFF_BATCH to raise
# the effective batch (changes the optimisation; default keeps the current value).
AUTO_BATCH="${AUTO_BATCH:-0}"
if [ "$AUTO_BATCH" = "1" ]; then
  EFF_BATCH="${EFF_BATCH:-$(( STRONG_BATCH_SIZE * GRAD_ACCUM ))}"
  _gpu="${CUDA_VISIBLE_DEVICES:-0}"; _gpu="${_gpu%%,*}"
  _free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${_gpu:-0}" 2>/dev/null | head -1 | tr -dc '0-9')
  _free="${_free:-0}"
  if   [ "$_free" -ge 34000 ]; then _b=8
  elif [ "$_free" -ge 26000 ]; then _b=4
  elif [ "$_free" -ge 21000 ]; then _b=2
  else _b=1; fi
  [ "$_b" -gt "$EFF_BATCH" ] && _b="$EFF_BATCH"
  while [ "$_b" -gt 1 ] && [ $(( EFF_BATCH % _b )) -ne 0 ]; do _b=$(( _b / 2 )); done
  [ "$_b" -lt 1 ] && _b=1
  STRONG_BATCH_SIZE="$_b"
  GRAD_ACCUM=$(( EFF_BATCH / _b ))
  echo "[auto-batch] GPU ${_gpu:-0} free=${_free}MiB -> strong_batch_size=${STRONG_BATCH_SIZE} grad_accum=${GRAD_ACCUM} (effective=${EFF_BATCH})"
fi
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LR_DECAY="${LR_DECAY:-linear}"
# Activation layer + token pooling for the kNN / RP selectors (controls; probe stays end/last).
REPRESENTATION_LAYER="${REPRESENTATION_LAYER:-end}"
REPRESENTATION_POOLING="${REPRESENTATION_POOLING:-last}"
# Span the kNN/RP representation pools over: 'full' (whole prompt, default) or 'answer' (answer-only).
REPRESENTATION_SPAN="${REPRESENTATION_SPAN:-full}"
# >1 adds disjoint equal-size quantile-bin selectors {rp,el,confidence}_q{i}_balanced (dose-response).
QUANTILE_BINS="${QUANTILE_BINS:-0}"
# Temperature for the continuous-weighting runs (rp_weighted / el_weighted / conf_weighted).
WEIGHT_TEMPERATURE="${WEIGHT_TEMPERATURE:-1.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.0}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
DEVICE="${DEVICE:-auto}"

RUNS="${RUNS:-base,ground_truth,weak_label,middle_balanced,random_balanced}"
RESIDUAL_KEEP_MIDDLE_FRAC="${RESIDUAL_KEEP_MIDDLE_FRAC:-0.5}"
CONFIDENCE_KEEP_FRAC="${CONFIDENCE_KEEP_FRAC:-0.5}"
KNN_K="${KNN_K:-20}"
KNN_KEEP_MIDDLE_FRAC="${KNN_KEEP_MIDDLE_FRAC:-0.5}"
KNN_MIXED_CENTER="${KNN_MIXED_CENTER:-0.5}"
KNN_REFERENCE_CROSS_FIT="${KNN_REFERENCE_CROSS_FIT:-0}"
CROSS_FIT_FOLDS="${CROSS_FIT_FOLDS:-5}"
DIAGNOSTICS_ONLY="${DIAGNOSTICS_ONLY:-0}"
RANDOM_CONTROL_COUNT="${RANDOM_CONTROL_COUNT:-3}"
PURITY_GRID="${PURITY_GRID:-}"
PURITY_MODE="${PURITY_MODE:-relabel}"
PURITY_DRAWS="${PURITY_DRAWS:-1}"
RANDOM_CONTROL_SIZE="${RANDOM_CONTROL_SIZE:-}"
RANDOM_UNBALANCED_SIZE="${RANDOM_UNBALANCED_SIZE:-}"

RIDGE_VALUES="${RIDGE_VALUES:-100.0}"
PCA_DIMS="${PCA_DIMS:-}"
BEST_BY="${BEST_BY:-heldout_median}"
NO_PLOTS="${NO_PLOTS:-0}"
SAVE_ACTIVATIONS="${SAVE_ACTIVATIONS:-0}"
SAVE_ADAPTERS="${SAVE_ADAPTERS:-0}"

EXTRA_ARGS=()
if [[ -n "$ACTIVATION_MAX_LENGTH" ]]; then
  EXTRA_ARGS+=(--activation-max-length "$ACTIVATION_MAX_LENGTH")
fi
if [[ -n "$TRAIN_SEED" ]]; then
  EXTRA_ARGS+=(--train-seed "$TRAIN_SEED")
fi
if [[ -n "$RANDOM_CONTROL_SIZE" ]]; then
  EXTRA_ARGS+=(--random-control-size "$RANDOM_CONTROL_SIZE")
fi
if [[ -n "$RANDOM_UNBALANCED_SIZE" ]]; then
  EXTRA_ARGS+=(--random-unbalanced-size "$RANDOM_UNBALANCED_SIZE")
fi
if [[ "$NO_PLOTS" == "1" ]]; then
  EXTRA_ARGS+=(--no-plots)
fi
if [[ "${EVAL_3CLASS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--eval-3class --n-eval-questions "${N_EVAL_QUESTIONS:-2041}")
fi
if [[ "${SCORE_BASE_ON_TRAIN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--score-base-on-train)
fi
if [[ -n "${DUMP_ACTS:-}" ]]; then
  EXTRA_ARGS+=(--dump-acts "$DUMP_ACTS")
fi
if [[ -n "${DUMP_EL:-}" ]]; then
  EXTRA_ARGS+=(--dump-el "$DUMP_EL")
fi
if [[ -n "${MLPALIGN_EPOCHS:-}" ]]; then
  EXTRA_ARGS+=(--mlpalign-epochs "$MLPALIGN_EPOCHS")
fi
if [[ -n "${MLPALIGN_BOTTLENECK:-}" ]]; then
  EXTRA_ARGS+=(--mlpalign-bottleneck "$MLPALIGN_BOTTLENECK")
fi
if [[ "${MLPALIGN_INSAMPLE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--mlpalign-insample)
fi
if [[ -n "${CUSTOM_SCORES_NPZ:-}" ]]; then
  EXTRA_ARGS+=(--custom-scores-npz "$CUSTOM_SCORES_NPZ")
fi
if [[ "$SAVE_ACTIVATIONS" == "1" ]]; then
  EXTRA_ARGS+=(--save-activations)
fi
if [[ "$SAVE_ADAPTERS" == "1" ]]; then
  EXTRA_ARGS+=(--save-adapters)
fi
if [[ "$SCIQ_USE_SUPPORT" == "1" ]]; then
  EXTRA_ARGS+=(--sciq-use-support)
fi
if [[ "$DATASET" == "anli" ]]; then
  EXTRA_ARGS+=(--anli-round "$ANLI_ROUND")
fi
EXTRA_ARGS+=(--committee-members "$COMMITTEE_MEMBERS" --committee-keep-frac "$COMMITTEE_KEEP_FRAC")
if [[ -n "$COMMITTEE_KEEP_FRACS" ]]; then
  EXTRA_ARGS+=(--committee-keep-fracs "$COMMITTEE_KEEP_FRACS")
fi
if [[ "$KNN_REFERENCE_CROSS_FIT" == "1" ]]; then
  EXTRA_ARGS+=(--knn-reference-cross-fit --cross-fit-folds "$CROSS_FIT_FOLDS")
fi
if [[ "$DIAGNOSTICS_ONLY" == "1" ]]; then
  EXTRA_ARGS+=(--diagnostics-only)
fi

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

echo "=== ${DATASET} paper-style LoRA W2S ==="
if [[ "$DATASET" == "sciq" ]]; then
  echo "SCIQ_USE_SUPPORT=${SCIQ_USE_SUPPORT}"
fi
python3 scripts/run_dream_paper_style_lora.py \
  --dataset "$DATASET" \
  --weak-model "$WEAK_MODEL" \
  --strong-model "$STRONG_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --n-train "$N_TRAIN" \
  --n-val "$N_VAL" \
  --n-test "$N_TEST" \
  --seed "$SEED" \
  --max-length "$MAX_LENGTH" \
  --answer-suffix "$ANSWER_SUFFIX" \
  --weak-batch-size "$WEAK_BATCH_SIZE" \
  --activation-batch-size "$ACTIVATION_BATCH_SIZE" \
  --l2-penalty "$L2_PENALTY" \
  --max-iter "$MAX_ITER" \
  --strong-batch-size "$STRONG_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --warmup-steps "$WARMUP_STEPS" \
  --lr-decay "$LR_DECAY" \
  --representation-layer "$REPRESENTATION_LAYER" \
  --representation-pooling "$REPRESENTATION_POOLING" \
  --representation-span "$REPRESENTATION_SPAN" \
  --quantile-bins "$QUANTILE_BINS" \
  --weight-temperature "$WEIGHT_TEMPERATURE" \
  --max-grad-norm "$MAX_GRAD_NORM" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --lora-target-modules "$LORA_TARGET_MODULES" \
  --torch-dtype "$TORCH_DTYPE" \
  --device "$DEVICE" \
  --runs "$RUNS" \
  --residual-keep-middle-frac "$RESIDUAL_KEEP_MIDDLE_FRAC" \
  --confidence-keep-frac "$CONFIDENCE_KEEP_FRAC" \
  --knn-k "$KNN_K" \
  --knn-keep-middle-frac "$KNN_KEEP_MIDDLE_FRAC" \
  --knn-mixed-center "$KNN_MIXED_CENTER" \
  --random-control-count "$RANDOM_CONTROL_COUNT" \
  --purity-grid "$PURITY_GRID" \
  --purity-mode "$PURITY_MODE" \
  --purity-draws "$PURITY_DRAWS" \
  --ridge-values "$RIDGE_VALUES" \
  --pca-dims "$PCA_DIMS" \
  --best-by "$BEST_BY" \
  "${EXTRA_ARGS[@]}"

echo "=== Done ==="
echo "Summary:     ${OUTPUT_DIR}/summary.json"
echo "Text report: ${OUTPUT_DIR}/paper_style_lora_report.txt"
echo "Predictions: ${OUTPUT_DIR}/eval_predictions.csv"
echo "Subsets:     ${OUTPUT_DIR}/train_subsets.csv"
