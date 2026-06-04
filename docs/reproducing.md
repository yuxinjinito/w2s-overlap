# Reproducing the experiments

All commands are run from the repository root. Outputs are written under `results/` (gitignored).
The commands below are the exact ones used, with representative observed metrics so you can sanity-
check a rerun. See [`method.md`](method.md) for what each stage does.

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # needed for gated meta-llama/Llama-3.1-8B weights
```

- **Compute:** a single CUDA GPU. The 8B LoRA runs fit in ~16 GB peak on a 24 GB card (e.g. RTX 4090).
- **Seeds:** `42` unless noted; deterministic inference runs do not depend on the seed.
- **Splits:** following the EleutherAI/Burns-style W2S setup, training data is split into a
  weak-training pool (`Dtrain`) and a weak-to-strong pool (`Dw2s` / `strong_train`).

## 1. Inference + weak confidence

SST-2 sequence-classification smoke test:

```bash
python3 scripts/run_inference_confidence.py \
  --mode seqcls --dataset sst2 --split validation --limit 128 \
  --weak-model distilbert-base-uncased-finetuned-sst-2-english \
  --strong-model textattack/bert-base-uncased-SST-2 \
  --output results/inference_confidence/sst2_smoke.csv
```

BoolQ causal-LM yes/no scoring (128 examples): weak ≈ 0.36, strong ≈ 0.68 accuracy;
weak/strong agree on ~29% of examples.

```bash
python3 scripts/run_inference_confidence.py \
  --mode causal_lm_yesno --dataset boolq --split validation --limit 128 \
  --weak-model Qwen/Qwen1.5-0.5B --strong-model Qwen/Qwen1.5-1.8B \
  --torch-dtype float16 \
  --output results/inference_confidence/boolq_qwen_smoke_128.csv
```

## 2. Probe-based weak confidence

Fit a logistic probe on weak final-token activations (BoolQ, 512 train / 128 eval): eval probe
accuracy ≈ 0.66; confidence remains high relative to accuracy (uncalibrated).

```bash
python3 scripts/run_probe_confidence.py \
  --dataset boolq --train-limit 512 --eval-limit 128 \
  --weak-model Qwen/Qwen1.5-0.5B --torch-dtype float16 \
  --batch-size 8 --epochs 150 --lr 0.003 --weight-decay 0.01 \
  --output results/probe_confidence/boolq_qwen05_probe_128.csv \
  --activation-output results/probe_confidence/boolq_qwen05_probe_acts_128.pt
```

The stricter Changho-style weak-probe path (weak confidence on the `strong_train` split):

```bash
python3 scripts/run_changho_style_probe.py \
  --dataset boolq --weak-model Qwen/Qwen1.5-0.5B \
  --n-train 1024 --n-val 128 --n-test 128 --target-split strong_train \
  --torch-dtype float16 --batch-size 4 \
  --output results/changho_style/boolq_qwen05_weakprobe_on_strong_train.csv \
  --activation-output results/changho_style/boolq_qwen05_weakprobe_on_strong_train_acts.pt
```

A multi-dataset confidence sweep (used for the answer-count vs skew analysis) is driven by
`scripts/run_paper10_confidence_batch.sh`.

## 3. Weak→strong representation mapping

Fit linear / Procrustes / ReLU maps on the full 512-example `strong_train` split and compare
per-sample residuals against weak confidence/correctness:

```bash
python3 scripts/run_representation_mapping.py \
  --dataset boolq --weak-model Qwen/Qwen1.5-0.5B --strong-model Qwen/Qwen1.5-1.8B \
  --target-split strong_train --n-train 1024 --n-val 128 --n-test 128 \
  --max-examples 512 --torch-dtype float16 --batch-size 2 \
  --pooling mean --map-train-frac 0.5 --ridge 1e-3 \
  --relu-hidden-dim 256 --relu-epochs 200 \
  --output results/representation_mapping/boolq_qwen05_to_qwen18_map_512.csv \
  --summary-output results/representation_mapping/boolq_qwen05_to_qwen18_map_512.json \
  --embedding-output results/representation_mapping/boolq_qwen05_to_qwen18_map_512.pt

python3 scripts/analyze_mapping_vs_confidence.py \
  results/representation_mapping/boolq_qwen05_to_qwen18_map_512.csv \
  --confidence-csv results/changho_style/boolq_qwen05_weakprobe_on_strong_train.csv \
  --merge-on id --primary-loss linear_l2 --plot-heldout-only \
  --report-output results/representation_mapping/boolq_qwen05_to_qwen18_map_512_vs_confidence.txt
```

Observed: held-out mapping residual has little linear relationship with weak confidence or
correctness on this model/dataset/pooling choice. The full Llama-3.1-8B mapping is driven by
`scripts/run_llama31_mapping_full.sh`.

## 4. 8B LoRA feasibility smoke test

```bash
pip install -r requirements.txt   # peft included
bash scripts/run_strong_lora_smoke.sh
```

Observed: 3 LoRA steps on `meta-llama/Llama-3.1-8B`, loss 2.52 → 1.64, peak ~15.4 GB,
~21 M trainable / 8.05 B total params.

## 5. Dream W2S baselines

```bash
OUTPUT_DIR=results/w2s_dream_baselines/dream_lora_fixed bash scripts/run_dream_w2s_baselines.sh
```

Config: `n_weak_train=2048, n_strong_train=512, n_eval=256, max_length=384`, LoRA `r=8, alpha=16,
dropout=0.05`, `max_train_steps=100, batch_size=1, grad_accum=4, lr=2e-4, float16`.
Observed: base ≈ 0.730, ground-truth-LoRA ≈ 0.859, weak-label-LoRA ≈ 0.551 (collapses to one class;
Dream weak labels are noisy/skewed, weak label-1 rate ≈ 0.77).

## 6. Residual-based overlap filtering

```bash
BASELINE_OUTPUT_DIR=results/w2s_dream_baselines/dream_residual_filter_mid50 \
RESIDUAL_FILTER_CSV=results/representation_mapping/<dream_map>/best_map_residuals.csv \
OUTPUT_DIR=results/w2s_dream_baselines/dream_residual_filter_extras \
bash scripts/run_dream_residual_filter_extras.sh
```

Observed (LoRA setup): residual-match alone collapses to one class; weak-label balancing alone does
not improve accuracy; **residual-middle + weak-label balancing** ≈ 0.762 — a first positive signal
(+0.03 over base, +0.23 over all-weak-label, −0.10 vs ground-truth).

## 7. Paper-faithful linear-probe replication

```bash
bash scripts/run_dream_paper_linear_probe.sh
bash scripts/run_dream_paper_residual_filtering.sh
```

Observed: weak probe ≈ 0.603, strong-ground-truth probe ≈ 0.739, Full-W2S probe ≈ 0.589 (close to
Figure A1 scale for Dream). Under this cleaner setup the residual-middle rule ≈ random balanced
controls (≈ 0.58–0.60), i.e. it does **not** help — so the LoRA positive above is setup-specific.

## 8. Selection-method comparisons and cross-dataset transfer

The confidence / residual / kNN selection comparisons and stability sweeps are driven by wrappers
that call `scripts/run_dream_paper_style_lora.py`:

```bash
bash scripts/run_dream_lora_stability_sweep.sh                 # stabilize LoRA params
bash scripts/run_dream_lora_filtering_comparison.sh            # confidence vs residual filtering
bash scripts/run_dream_lora_knn_mixed_filtering_comparison.sh  # kNN mixed-neighborhood selection
bash scripts/run_sciq_paper_style_lora.sh                      # SciQ transfer
bash scripts/run_paws_paper_prompt_smoke.sh                    # PAWS transfer
```

Each wrapper documents its own dataset/config presets at the top of the file.
