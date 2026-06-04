# Results (preliminary)

A summary of the main findings so far. These are **exploratory** results from a sequence of
diagnostic and ablation experiments; sample sizes are modest and several runs are setup-specific.
Exact commands and configs are in [`reproducing.md`](reproducing.md).

## Weak-confidence skew vs answer count

A logistic weak probe was run on 11 datasets; see
[`experiments/2026-05-28-dataset-answer-counts.md`](../experiments/2026-05-28-dataset-answer-counts.md)
for the full table and [`experiments/2026-05-28-answer-count-vs-confidence-skew.svg`](../experiments/2026-05-28-answer-count-vs-confidence-skew.svg)
for the figure.

- The most skewed (over-confident) weak-confidence distributions are all **binary sentiment-style**
  tasks: Amazon Polarity, SST-2, Twitter-sentiment (fraction with confidence > 0.9: ~0.53–0.80).
- **Dream, SciQ** (originally multiple-choice) and **PAWS** (binary) are **healthier** (~0.13–0.22).
- Conclusion: the number of answer choices is *related* to skew but does **not** fully explain it —
  PAWS is binary yet healthy, so task structure / shortcut availability matters at least as much.

## Representation-mapping diagnostics

- Weak→strong maps (linear, Procrustes, 1-layer ReLU) are fit per dataset; the learned map, its
  singular values, and per-sample held-out residual L2 are saved.
- On BoolQ (Qwen0.5B → Qwen1.8B, mean pooling), held-out mapping residual has **little linear
  relationship** with weak confidence or weak correctness — the simple mapping residual is not, by
  itself, an overlap score in that setting.
- For Dream (Qwen0.5B → Llama-3.1-8B) the best map was a centered ridge map (held-out residual L2
  median ≈ 53.7, held-out cosine ≈ 0.91); residual histograms look closer to a **continuum** than to
  three crisp easy/overlap/hard clusters.

## Dream weak-to-strong baselines

| Strong model | Eval accuracy |
|---|---:|
| base (no fine-tuning) | ≈ 0.730 |
| LoRA on ground-truth labels | ≈ 0.859 |
| LoRA on weak pseudo-labels | ≈ 0.551 |

The weak-label run collapses toward a single class: Dream weak labels on the `strong_train` split
are noisy and skewed (weak-label accuracy ≈ 0.56, weak label-1 rate ≈ 0.77). This is the W2S gap the
selection methods aim to close.

## Overlap-based selection

- **Residual-middle filtering** (keep central residual band) + weak-label balancing reached ≈ 0.762
  under the LoRA setup — a first positive signal (+0.03 over base, +0.23 over all-weak-label).
  **However**, under the cleaner **paper-faithful linear-probe** setup the same rule matched random
  balanced controls (≈ 0.58–0.60) and did not help. Reported as **setup-specific**, not a stable rule.
- **Weak-confidence middle pruning** (drop very confident and very unconfident points) is a
  reasonable baseline selection method.
- **kNN mixed-neighborhood selection** — keeping points whose nearest weak-training neighbors in
  strong-embedding space are *mixed* (some weak-correct, some weak-wrong) — is currently the
  **strongest Dream selection method**, with the **unbalanced** variant more important than the
  balanced one.

## Paper-faithful replication

Replicating the Figure A1 linear-probe setup for Dream gives weak ≈ 0.603, strong-ground-truth ≈
0.739, Full-W2S ≈ 0.589 — close to the seed paper's Dream scale, which confirms that earlier
lower-looking Dream numbers were due to setup differences (LoRA + generative scoring) rather than
the dataset.

## Status and open questions

- **Cross-dataset transfer:** does kNN mixed-neighborhood selection help on **SciQ and PAWS** under
  the same task structure? (in progress)
- **Why it works:** connect mixed-neighborhood selection to semi-supervised / active-learning theory
  (boundary / mixed-neighborhood intuition).
- **Beyond binary:** move from binary candidate-correctness toward true multiple-choice, free-
  response, LLM-as-a-judge, or verifiable-reward settings.
- **Alignment methods:** test whether CCA / CKA / SVCCA give a better structural signal than raw L2
  residual.
