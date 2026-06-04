# Method

This document states the problem, research questions, hypotheses, and the methods implemented in
this repository. For exact commands and metrics see [`reproducing.md`](reproducing.md); for a
summary of findings see [`results.md`](results.md).

## Problem setting

In weak-to-strong (W2S) generalization, a weak teacher model produces (pseudo-)labels that are used
to fine-tune a stronger student model. The **data-centric** view of W2S partitions training
examples into three regions:

- **easy** — both weak and strong models already handle them;
- **overlap** — weak supervision is informative for the strong model;
- **hard** — weak labels are unreliable and may hurt the strong model.

The seed paper (*Weak-to-Strong Generalization Through the Data-Centric Lens*, arXiv:2412.03881)
argues that the overlap region drives W2S generalization and detects it with a relatively simple
discrete rule. This project asks whether overlap can be detected **more smoothly and more directly
from representation geometry and weak-model confidence**, and whether such a signal **improves
downstream W2S training**.

## Research questions

1. **Geometry.** Are examples the weak model gets *correct* vs *incorrect* separable in an auxiliary
   embedding space?
2. **Mapping.** How well can weak-model representations be mapped into strong-model representations,
   and do per-sample mapping **residuals** identify structure (features available only to the strong
   model, harder examples, or overlap)?
3. **Selection.** Can a continuous overlap/usefulness score — from weak confidence and/or
   embedding-neighborhood geometry — **select or weight data** so that weak-to-strong fine-tuning
   beats (a) using all weak labels and (b) the original discrete overlap rule?
4. **Scope.** Does the signal generalize across datasets (Dream → SciQ → PAWS) and beyond binary
   candidate-answer correctness?

## Hypotheses

- **H1.** In at least one classification setting from the seed paper, weak-correct and weak-incorrect
  examples show non-random structure in an auxiliary embedding space.
- **H2.** A smoother overlap score (weak confidence + embedding-neighborhood similarity) is more
  informative for data selection than a discrete easy/overlap/hard label.
- **H3.** A learned weak→strong representation map has structured per-sample residuals: low residual
  ≈ features already available to the weak model; high residual ≈ strong-only structure or harder
  examples.
- **H4.** Overlap-like points lie near decision boundaries / in mixed weak-correct vs weak-wrong
  neighborhoods, which connects the selection rule to semi-supervised and active learning.

Failure modes worth testing: embedding separation may be weak or dataset-specific; weak confidence
may already explain most of the signal; a linear map may be too simple while a ReLU map may overfit;
and a score that looks clean diagnostically may still fail to improve downstream training.

## Method components

### 1. Task formatting

Multiple-choice / QA datasets are reformatted as **binary candidate-answer correctness**: for each
question a candidate answer (the true answer or a distractor) is sampled, and the model judges
whether the candidate is correct (`question + candidate → yes/no`). This matches the seed paper's
reference implementation and makes weak/strong predictions, confidence, and probes comparable across
datasets. A Dream 3-class variant is also implemented for the "beyond binary" direction.

### 2. Weak-model confidence

Two paths produce a per-example weak confidence in `[0, 1]`:

- **Generative yes/no** — score `" yes"` vs `" no"` continuations of a causal LM.
- **Activation probe** — extract weak final-token hidden states and fit a small logistic probe.

Both use the binary confidence `confidence = 2·|P(label=1) − 0.5|`, which is 0 for maximally
uncertain predictions and 1 for confident ones.

### 3. Representation extraction

For each prompt, hidden states are taken from the layer **before `lm_head`**. Token-level states are
aggregated into one prompt vector. The paper-faithful path uses the **final-token** state; some
mapping experiments use **mean pooling** over non-padding tokens. Both are supported and the choice
is logged per experiment.

### 4. Weak→strong representation mapping

Given matched weak and strong prompt representations, fit maps from weak space to strong space:

- **linear regression** (optionally ridge-regularized / centered),
- **Procrustes / orthogonal** mapping,
- a **1-layer ReLU** network (a nonlinear version of the same test).

Outputs saved for analysis: the learned map `A`, its **singular values** (how much it stretches /
shrinks directions), and the **per-sample residual L2** between the mapped weak vector and the true
strong vector (lower = better alignment). Train/heldout splits are kept separate so residuals are
read on held-out points.

### 5. Weak-to-strong LoRA training

On the strong model (`Llama-3.1-8B`, LoRA), three baselines:

- **base** — no fine-tuning;
- **ground-truth** — LoRA on gold labels (an upper-reference);
- **weak-label** — LoRA on weak pseudo-labels (the W2S baseline to beat).

### 6. Overlap-based data selection

Selection rules applied to the weak-labeled training pool, each evaluated by the resulting strong
model's accuracy and compared to random controls of equal size:

- **residual-middle** — keep the central band of the residual distribution, drop low/high residuals;
- **weak-confidence middle** — prune both very confident and very unconfident points;
- **kNN mixed-neighborhood** — in strong-embedding space, keep points whose nearest weak-training
  neighbors are *mixed* (some weak-correct, some weak-wrong) rather than uniformly easy or hard;
  both balanced and unbalanced variants are tested.

### 7. Paper-faithful linear probing

A separate path replicates the seed paper's Figure A1 setup as closely as possible: Qwen1.5-0.5B
weak activations, Llama-3.1-8B strong activations, final-token states, and **logistic linear
probes** for weak, strong-ground-truth, and Full-W2S accuracy. This separates *replication* from the
*LoRA/selection extension* so the two are not conflated.

## Evaluation discipline

Every experiment records which original-paper result/setup it is compared against and lists known
deviations (dataset size, split, model version, training method, prompt/task format, evaluation
size, seeds). The known deviations from the seed paper's LLM setup are:

- weak/strong pair (Qwen1.5-0.5B → Llama-3.1-8B) vs the paper's Qwen1.5-0.5B → Llama3-8B;
- training method (LoRA + generative yes/no scoring) vs the paper's linear probing for Figure A1;
- smaller training/eval slices than the paper's full sampling;
- a more explicit candidate-correctness prompt with a generated yes/no target;
- mean pooling in some mapping runs vs the paper's final-token activations.

The paper-faithful linear-probe path above exists specifically to control for these differences when
a direct numeric comparison is needed.
