# Reproducing the experiments

All commands run from the repository root. Outputs go to `results/` (gitignored). See
`[method.md](method.md)` for what each stage does and `[results.md](results.md)` for the
reporting conventions a rerun is checked against.

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A single 24 GB CUDA GPU is enough. The default pair is Qwen2.5-0.5B (weak, frozen, with
a probe) supervising Qwen2.5-7B (strong, LoRA), both ungated. The pair has to come from one family: with a cross-family pair the tokenizers differ
and rp stops working.

Two seeds are separate. The data seed (`--seed`) fixes splits, cross-fitting folds, and
class balancing. The training seed (`--train-seed`, or `TRAIN_SEEDS` in the wrappers) fixes LoRA initialization and batch order. Reported numbers are means over training seeds at a fixed data seed.

## Check the install before spending GPU time

The two score modules run their own checks, on CPU:

```bash
python scripts/representation_projection.py
python scripts/mlp_rp.py
```

The first takes seconds, the second a couple of minutes. If either one fails, the
environment is wrong, and a full run would have told you the same thing three hours
later.

## A full run

The canonical sweep is one dataset, many arms, three training seeds:

```bash
ANLI_ROUND=r1 TRAIN_SEEDS="42 123 456" bash scripts/run_anli_band_map.sh
```

This trains the base and gold-label anchors once, then one LoRA fine-tune per arm per
seed, and writes a `summary.json` per run with the full configuration, per-arm
accuracy, and kept-set diagnostics. Expect a few hours.

Useful environment variables, all read by the wrappers:


| Variable               | Meaning                                                                           |
| ---------------------- | --------------------------------------------------------------------------------- |
| `FILTER_RUNS`          | Comma-separated arm names, for example `rp_high_balanced_f50,random_balanced_f50` |
| `TRAIN_SEEDS`          | Space-separated LoRA seeds                                                        |
| `COMMITTEE_KEEP_FRACS` | Keep fractions, for example `0.5` or `0.1,0.2,0.3`                                |
| `OUTPUT_ROOT`          | Where the run writes                                                              |
| `EVAL_3CLASS`          | Multiple-choice evaluation (on by default in the ANLI wrapper)                    |
| `DUMP_ACTS`            | Also write the weak and strong activations to a file, so screens can run without the models             |
| `CUSTOM_SCORES_NPZ`    | Inject externally computed scores, see below                                      |


For a dataset other than ANLI use `run_band_map_sweep.sh` with `DATASET` set. The ANLI
wrapper is pinned and will refuse another dataset.

Arm names follow `<score>_<band>_balanced_f<percent>`. The band is `high`, `low`, or
`middle`, and `balanced` means the kept set is class-balanced on the weak labels before
training.

## Offline screens

Designing a new score is a loop. One GPU run dumps the activations, screens then try
many candidate scores on that dump using only CPU, and the few that look worth it go
back through the pipeline as new arms.

```bash
# once, on GPU: run the pipeline and keep the activations
DUMP_ACTS=acts_r1.npz ANLI_ROUND=r1 TRAIN_SEEDS=42 bash scripts/run_anli_band_map.sh

# then, on CPU, as many times as you like
python scripts/mlpstep2_screen.py --acts acts_r1.npz --wd-grid 0.01,1,10,30
python scripts/joint_screen.py --labeled acts_r1.npz --out-scores custom_scores.npz
```

The dump holds both activation matrices, the weak labels, and the gold labels the
screens use to measure separation. Recomputing it means loading both models and a
forward pass over every row, which is why one dump serves a whole round of screening.



## Injecting your own score

Any per-example score can enter the pipeline without touching it. Write an npz with one
to four score vectors named `cs1` through `cs4`, plus the `weak_preds` vector the scores
were computed against, then:

```bash
CUSTOM_SCORES_NPZ=/path/scores.npz \
FILTER_RUNS=cs1_high_balanced_f50,cs1_low_balanced_f50,random_balanced_f50 \
ANLI_ROUND=r1 TRAIN_SEEDS="42 123 456" bash scripts/run_anli_band_map.sh
```

The pipeline compares the stored `weak_preds` against the weak labels it computes itself
and refuses to run if they disagree on more than 5% of rows. Scores computed on one
machine and used on another can drift apart through fp16 differences in the probe, and
the guard catches that before it becomes a silent misalignment.

## Reading the output

Each run directory holds `summary.json` (configuration, per-arm evaluation, kept-set
label purity) and per-arm prediction CSVs. The number we report is
`runs.<arm>.eval.accuracy_3class`, averaged over training seeds, with the base and
gold-label anchors from the same run's baseline directory.