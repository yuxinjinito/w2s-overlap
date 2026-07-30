# Method

The problem, the score, and the rules that constrain anything built on it. For commands
see `[reproducing.md](reproducing.md)`; for what has been measured see `[results.md](results.md)`.

## Problem setting

A weak model supervises a stronger one. We fine-tune a weak model on a labeled split,
use it to label a second split, and train the strong model on those weak labels. The
question this repository asks is which of the weakly labeled rows the strong model
should train on, decided without looking at gold labels.

Three reference points bracket every result. The weak model's own accuracy is the
floor, the strong model trained on gold labels is the ceiling, and the strong model
trained on all weak labels with no filtering is the point any selector has to beat. We
report PGR, `(acc - base) / (gold - base)`, so runs with different anchors stay
comparable.

## Task format

Every dataset is reduced to binary rows. A K-way question becomes K rows, one per
candidate answer, each asking whether that candidate is correct:

```
<question and candidate answer>
Is the candidate answer correct? Answer:
```

The strong model is scored with a yes/no head, and the multiple-choice accuracy we
report picks the highest-scoring candidate per question. One format for all three
stages means the weak probe, the selection score, and the LoRA fine-tune read the same
representation of the task.

## Weak labels

The weak supervisor is a logistic probe fit on the weak model's final-token activations
over the weak split, then applied to the strong split. It is not the weak model's own
generation. Labels are hard, thresholded at 0.5. Soft probabilities perform the same
downstream, and a logit-difference target is still under test.

## Representations

Both models are run over the same rows and we keep the final-layer, final-token
activation of the full prompt. Both choices were settled by ablation rather than
assumption. Reading the middle layer instead leaves rp close to random, and
mean-pooling over tokens inverts the score's high/low direction outright.

## The rp score

With column-centered representations, build one Gram kernel per side, `K = X X^T / n`,
and turn each into a ridge smoother

```
P = K (K + reg * (tr(K)/n) * I)^{-1},    reg = 0.1
```

The regularizer is scaled by the average kernel eigenvalue so that the same `reg` means
the same thing on a 896-dimensional weak representation and a 3584-dimensional strong
one. With centered weak labels `y_c`:

```
step 1 (weak side):    a = (I - P_w) y_c
step 2 (strong side):  v = P_s a
score:                 s_i = |v_i|
```

Step 1 removes the part of the labels the weak representation can explain. Step 2 asks
how much of that leftover the strong representation can express. A high score marks a
row whose label is illegible to the weak model and legible to the strong one, which is
the kind of row weak-to-strong training should benefit from. Keeping the top half is
what works downstream.

Centering the labels keeps the class prior from entering through the constant
direction, which would otherwise degrade the score into a ranking by majority class.

The score is the per-example form of the quantity that governs the prediction gap in
*Representations Shape Weak-to-Strong Generalization* (arXiv:2502.00620). That paper
uses the aggregate norm to predict how well a given weak supervisor will do, and never
filters data with it. Two things differ here. We read the per-example entries rather
than the norm, and we substitute cross-fitted weak labels for the gold labels the
theorem assumes.

Neither step inherits the theorem's guarantee, so the evidence for the score is
downstream training, not the theory.

## Variants and what they showed

`mlpstep1` replaces the weak-side solve with a heavily weight-decayed MLP, still
cross-fitted, and leaves step 2 alone. It matches rp everywhere tested and beats it on
two beds. A linear model pushed through the same cross-fitted harness reproduces rp, so
the credit belongs to the regularized nonlinearity rather than to the cross-fitting.

Swapping the other pieces produced the working rules below. They are what the runs so
far support, at the confidence each one has earned.

The two sides are not interchangeable. Replacing step 2 with an out-of-fold estimator
drops rank agreement with rp to about .62 for a cross-fitted MLP and .64 for a
cross-fitted linear model, and the MLP version was also run downstream on three beds,
where it lost to rp on all three. The linear version has only the agreement number so
far. The reading we work from is that step 2 is a membership test rather than a
prediction problem, since it asks whether the leftover lies inside the strong model's
expressible space, and out-of-fold fitting answers a different question. Step 1 behaves
the opposite way and takes cross-fitting without complaint.

An in-sample fit with no capacity cap degenerates. For the kernel this is analytic: as
`reg` goes to zero, `P_s` approaches the identity and the score collapses onto step 1's
residual, so the strong side stops filtering. For a learned map the failure is
memorization of the target, which we have observed in the alignment setting. A capped
in-sample network was then tested as the step-2 estimator at two points of the cap:
lightly capped it lands below the random control, and at the cap the screen preferred
it tracks the kernel's ranking and lands between random and rp downstream. The closer
the capped network is pushed toward the kernel, the better it does, and it does not
pass it, so the kernel stays the step-2 tool.

Direction is checked, not assumed. rp and its step-1 variants keep the same useful end
on every bed measured so far. The auxiliary signals do not: both forms of the
excess-loss score, and the effect of training order on a fixed kept set, reverse sign
between testbeds. Each new score is therefore run downstream in both bands.

## Alternatives that were closed

Label-free alignment residuals (linear, Procrustes, CCA, an MLP projector), discriminant read-outs of the residual in several forms, sparse-autoencoder bases, hard spectral truncation, and kernel or basis combinations tuned jointly were all measured against rp. None beat it, several are actively harmful, and the code for each is still in `scripts/`.