# Representation-Side Data Selection for Weak-to-Strong Generalization

Research code for selecting weak-labeled training data in weak-to-strong (W2S)
fine-tuning using only the weak and strong models' internal representations,
with no gold labels at selection time.

The core score, **rp**, turns the central quantity of
*Representations Shape Weak-to-Strong Generalization* (ICML 2025,
[arXiv:2502.00620](https://arxiv.org/abs/2502.00620)) into a per-example
selection signal. That paper proves the weak-vs-ceiling prediction gap is
governed by `P_s(I - P_w) y`, uses it to predict W2S performance, and never
filters data with it; this repository does the filtering, with cross-fitted
weak labels standing in for the gold labels the theorem assumes. A
regularized-nonlinear variant (**mlpstep1**) replaces the weak-side solve with
a heavily weight-decayed MLP and is the strongest selector we have measured.

The project extends the data-centric W2S line of
*Weak-to-Strong Generalization Through the Data-Centric Lens*
([arXiv:2412.03881](https://arxiv.org/abs/2412.03881)): rp is a continuous,
per-example version of that paper's overlap detection.

## Method in four lines

With column-centered representations, per side build the Gram kernel
`K = X X^T / n` and the ridge smoother `P = K (K + reg * tr(K)/n * I)^{-1}`:

```
step 1 (weak side):    a = (I - P_w) y_c        # label part the weak rep cannot explain
step 2 (strong side):  v = P_s a                # part of that leftover the strong rep expresses
score                  s_i = |v_i|              # keep the top half, train on it
```

Step 1 uses hard, cross-fitted weak labels; step 2 is fitted in-sample with
the ridge cap. Both choices are load-bearing: swapping either one degrades or
destroys the score, and `scripts/mlp_rp.py` has the controlled swaps.

## Headline result

Research in progress.

## Repository layout

```
scripts/                          # flat by design: modules import each other as siblings
  representation_projection.py    # the rp score (self-testing: run the file)
  mlp_rp.py                       # mlpstep1 / linstep1 / step-2 swap variants (self-testing)
  run_dream_paper_style_lora.py   # the full W2S pipeline: probe, scores, banding, LoRA, eval
  run_dream_paper_style_lora.sh   # env-driven single-run wrapper
  run_band_map_sweep.sh           # canonical multi-arm, multi-seed sweep
  run_anli_band_map.sh            # ANLI-pinned wrapper (fails loudly on other datasets)
  joint_screen.py                 # co-tuned basis x kernel x regularization screen
  lda_*.py, sae_*.py, cca_*.py,
  robust_linear.py, mlp_alignment.py   # alternative-selector family (all closed negatives)
  diagnose_*.py, dump_bulk_acts.py     # diagnostics and activation dumps
  README.md                       # index of every script, by role
docs/                             # method, commands, and current results
experiments/                      # dated analysis artifacts
results/                          # run outputs (gitignored)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A single 24 GB CUDA GPU fits every experiment (7B LoRA, batch 4, fp16).

## Quick start

Score a dataset with rp (pure NumPy):

```python
from representation_projection import representation_projection_scores
scores = representation_projection_scores(weak_acts, strong_acts, weak_labels)
```

Both score modules are self-testing: `python scripts/representation_projection.py`
and `python scripts/mlp_rp.py` run their built-in checks. A full W2S run with
the standard arm set, three seeds:

```bash
ANLI_ROUND=r1 TRAIN_SEEDS="42 123 456" bash scripts/run_anli_band_map.sh
```

Arms, seeds, datasets, and evaluation are all env-driven; external score
vectors can be injected with `--custom-scores-npz` (the pipeline verifies row
alignment against its own weak labels before using them). Per-experiment
commands live in `docs/reproducing.md`.

## Reproducibility notes

Every run writes a `summary.json` with the full configuration, per-arm accuracies, and kept-set diagnostics. Reported numbers are means over LoRA seeds at fixed data seed.

## Citing

If you use this code, please cite the two papers it builds on, the theory
([arXiv:2502.00620](https://arxiv.org/abs/2502.00620)) and the data-centric
W2S framing ([arXiv:2412.03881](https://arxiv.org/abs/2412.03881),
reference implementation
[`SprocketLab/datacentric_w2s`](https://github.com/SprocketLab/datacentric_w2s))
alongside this repository. A paper describing the method here is in
preparation; a citation entry will replace this line when it is available.