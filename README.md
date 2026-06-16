# Weak-to-Strong Overlap Detection

Research code for studying **weak-to-strong (W2S) generalization through a data-centric lens**.
The project asks whether the "overlap" structure that governs W2S generalization can be
detected from **weak/strong model representations and weak-model confidence**, and whether such a
signal can be used to **select training data** that improves weak-to-strong fine-tuning.

This is an independent extension of the setup in
*Weak-to-Strong Generalization Through the Data-Centric Lens* (arXiv:2412.03881) and reuses the
binary candidate-answer task formatting from the authors' reference implementation
([`SprocketLab/datacentric_w2s`](https://github.com/SprocketLab/datacentric_w2s)).

> **Status:** active research code. Results here are **preliminary and exploratory** — the
> repository documents a sequence of diagnostic and ablation experiments, not a finished paper.

## Background

In weak-to-strong generalization, a weak teacher's (pseudo-)labels are used to fine-tune a stronger
student. The data-centric view partitions training examples into *easy*, *overlap*, and *hard*
regions, and argues that the **overlap** examples — where weak supervision is informative for the
strong model — drive generalization. The seed paper detects overlap with a relatively simple
discrete rule.

This project investigates whether overlap can be detected **more smoothly and more directly from
representation geometry**, and whether the resulting signal actually helps downstream W2S training:

1. Are weak-model *correct* vs *incorrect* examples geometrically separable in an embedding space?
2. How well can weak representations be **mapped** into strong representations, and do per-sample
   mapping **residuals** mark easy / overlap / hard structure?
3. Can a continuous overlap score (confidence + neighborhood geometry) **select data** that beats
   no-filtering and the original discrete rule for W2S fine-tuning?

See [`docs/method.md`](docs/method.md) for the full problem statement, hypotheses, and method
details.

## Methods implemented

| Stage | What it does | Entry point(s) |
|---|---|---|
| **Inference + confidence** | Run weak/strong models, save per-example predictions and reference-style binary confidence `2·\|p−0.5\|` | `scripts/run_inference_confidence.py` |
| **Probe confidence** | Extract weak final-token activations, fit a logistic probe, output probe-based confidence | `scripts/run_probe_confidence.py`, `scripts/run_reference_probe.py` |
| **Representation mapping** | Fit weak→strong maps (linear regression, Procrustes, 1-layer ReLU); save the map and per-sample residual L2 | `scripts/run_representation_mapping.py`, `scripts/run_dream_aligned_residual_mapping.py` |
| **W2S LoRA baselines** | Strong model: base, ground-truth-trained, weak-label-trained (LoRA) | `scripts/run_dream_w2s_baselines.py` |
| **Paper-faithful probing** | Replicate the Figure A1 linear-probing setup (weak/strong/Full-W2S) | `scripts/run_dream_paper_linear_probe.py` |
| **Overlap selection** | Residual-middle filtering, weak-confidence middle pruning, and kNN mixed-neighborhood selection on top of the LoRA pipeline | `scripts/run_dream_paper_style_lora.py`, `scripts/run_dream_paper_residual_filtering.py`, `scripts/run_dream_residual_filter_extras.py` |
| **Feasibility / variants** | 8B LoRA smoke test; Dream 3-class variant | `scripts/run_strong_lora_smoke.py`, `scripts/run_dream_three_choice_smoke.py` |

## Datasets and models

- **Weak model:** `Qwen/Qwen1.5-0.5B` (a `Qwen/Qwen1.5-1.8B` strong proxy is used for cheap local tests).
- **Strong model:** `meta-llama/Llama-3.1-8B` (LoRA fine-tuning).
- **Primary task datasets:** Dream, SciQ, PAWS — formatted as **binary candidate-answer
  correctness** (`question + candidate answer → is this candidate correct?`), following the seed
  paper / reference repo.
- **Confidence-skew survey** additionally covers SST-2, BoolQ, Amazon Polarity, Twitter-sentiment,
  WiC, CoLA, ANLI-R2, and HellaSwag (see [`experiments/`](experiments/)).

## Repository structure

```
.
├── README.md
├── requirements.txt
├── docs/
│   ├── method.md          # problem statement, hypotheses, method details
│   ├── reproducing.md      # environment + exact commands per experiment
│   └── results.md          # summary of preliminary findings
├── scripts/                # all runnable code (flat; the run_*.py modules import each other)
│   ├── run_inference_confidence.py / run_probe_confidence.py / run_reference_probe.py
│   ├── run_representation_mapping.py / run_dream_aligned_residual_mapping.py
│   ├── run_dream_w2s_baselines.py / run_dream_paper_linear_probe.py
│   ├── run_dream_paper_style_lora.py / run_dream_paper_residual_filtering.py / run_dream_residual_filter_extras.py
│   ├── run_dream_three_choice_smoke.py / run_strong_lora_smoke.py
│   ├── run_*.sh            # experiment wrappers: dataset/config presets and sweeps
│   └── analyze_*.py / plot_*.py / inspect_*.py / summarize_*.py / check_*.py / print_*.py
├── experiments/            # dated analysis artifacts (e.g., answer-count vs confidence skew)
├── figures/                # generated diagnostic figures
└── results/                # experiment outputs (gitignored)
```

> **Note on layout.** The `scripts/run_*.py` files are standalone CLI entry points that also import
> one another as sibling modules (e.g. `run_dream_paper_style_lora.py` reuses
> `run_dream_w2s_baselines.py`). They are therefore kept in a single flat directory and invoked from
> the repository root (`python3 scripts/<name>.py ...`). The `scripts/run_*.sh` wrappers encode the
> exact dataset/config presets for each experiment.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The experiments require a CUDA GPU (the 8B LoRA runs fit on a single 24 GB card, e.g. an RTX 4090).
Access to the gated `meta-llama/Llama-3.1-8B` weights requires a Hugging Face token
(`huggingface-cli login`).

## Reproducing experiments

All commands are run from the repository root and write under `results/` (gitignored). A minimal
inference smoke test:

```bash
python3 scripts/run_inference_confidence.py \
  --mode causal_lm_yesno --dataset boolq --split validation --limit 128 \
  --weak-model Qwen/Qwen1.5-0.5B --strong-model Qwen/Qwen1.5-1.8B \
  --torch-dtype float16 \
  --output results/inference_confidence/boolq_qwen_smoke.csv
```

The Dream W2S baselines and the residual / confidence / kNN selection comparisons are driven by the
`scripts/run_dream_*.sh` wrappers (and `scripts/run_sciq_*.sh`, `scripts/run_paws_*.sh` for
cross-dataset transfer). Full per-experiment commands, configs, and metrics are in
[`docs/reproducing.md`](docs/reproducing.md).

## Key findings (preliminary)

- **Confidence skew** correlates with task type and the number of answer choices, but is not fully
  explained by them: sentiment-style binary tasks are the most skewed, while PAWS (binary) and
  Dream/SciQ (originally multiple-choice) have healthier weak-confidence distributions.
- **Dream W2S baselines:** base strong ≈ 0.73, ground-truth-LoRA ≈ 0.86, weak-label-LoRA ≈ 0.55
  (the weak-label run collapses toward a single class because Dream weak labels are noisy/skewed).
- **Residual-middle filtering** plus weak-label balancing produced a first positive selection signal
  under the LoRA setup, but did **not** hold under the cleaner paper-faithful linear-probe setup —
  so it is reported as setup-specific rather than a stable rule.
- **kNN mixed-neighborhood selection** (keeping points with mixed weak-correct / weak-wrong
  neighbors in strong-embedding space) is currently the strongest Dream selection method; whether it
  transfers to SciQ/PAWS is the active question.

Details and exact numbers: [`docs/results.md`](docs/results.md).

## References

- *Weak-to-Strong Generalization Through the Data-Centric Lens*, arXiv:2412.03881.
- Reference implementation: [`SprocketLab/datacentric_w2s`](https://github.com/SprocketLab/datacentric_w2s).
