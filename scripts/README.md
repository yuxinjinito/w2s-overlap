# scripts/ index

The directory is flat, and every file is an experiment entry point, invoked from the repository root as `python scripts/<name>.py`, and several import each other as siblings. I know, that makes the directory large, so this index says what is current, what is a standalone tool, and what is kept as the record of a closed line of work.

## Current pipeline


| File                                                            | Role                                                                                                                                               |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_dream_paper_style_lora.py`                                 | The full weak-to-strong run: weak probe, activations, all selection scores, banding, LoRA fine-tune, evaluation. Every experiment goes through it. |
| `run_dream_paper_style_lora.sh`                                 | Single run, all knobs as environment variables.                                                                                                    |
| `run_band_map_sweep.sh`                                         | The canonical sweep: many arms, many seeds, one dataset.                                                                                           |
| `run_anli_band_map.sh`                                          | ANLI-pinned wrapper; fails loudly if handed another dataset.                                                                                       |
| `paper_style_datasets.py`                                       | The sixteen testbeds: loading, prompt formatting, splits, and the multiple-choice eval sets. A leaf layer, it calls nothing else in the pipeline. |
| `paper_style_report.py`                                         | Metrics and every file a run writes: per-arm accuracy and AUROC, kept-set summaries, CSV dumps, and the text report beside summary.json.        |
| `run_dream_w2s_baselines.py`, `run_dream_paper_linear_probe.py` | Imported by the pipeline for the baseline runs and the probe path.                                                                                 |




## Selection scores


| File                                                                           | Score                                                                                                                                                                               |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `representation_projection.py`                                                 | rp, the main score. Self-testing.                                                                                                                                                   |
| `mlp_rp.py`                                                                    | mlpstep1 (the strongest variant), linstep1 (its control), and the step-2 swap. Self-testing.                                                                                        |
| `excess_loss.py`                                                               | Excess-loss family (entropy gap and loss difference).                                                                                                                               |
| `mlp_alignment.py`, `cca_alignment.py`, `lda_alignment.py`, `robust_linear.py` | Label-free and discriminant alternatives. All measured, none beat rp; kept because the pipeline still runs them as comparison arms and the negative results are part of the record. |




## Offline screens and diagnostics

Run by hand on dumped activations; they decide which variants earn a GPU run.


| File                                                                         | What it screens                                                        |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `joint_screen.py`                                                            | Basis, kernel, and regularization co-tuned over a grid.                |
| `mlpstep2_screen.py`                                                         | In-sample capped MLP as the step-2 estimator, swept over weight decay. |
| `lda_screen.py`, `sae_alignment_screen.py`, `screen_trio.py`                 | Discriminant, sparse-autoencoder, and mixed screens.                   |
| `diagnose_step1_mlp.py`, `diagnose_mlp_controls.py`, `diagnose_mlp_align.py` | Training diagnostics for the MLP variants.                             |
| `dump_bulk_acts.py`                                                          | Bulk activation dumps that the screens consume.                        |
| `base_native_accuracy.py`, `candidate_scout.py`                              | Testbed screening: is a dataset a valid weak-to-strong bed.            |
| `compute_pgr.py`                                                             | PGR and AUROC helpers.                                                 |




## Earlier work (kept, not maintained)

These are from the first phase of the project, when the beds were Dream, SciQ, and PAWS and the model pair was cross-family. They still parse and their imports resolve, but they are not part of the current pipeline and their results have been superseded.

`run_inference_confidence.py`, `run_probe_confidence.py`, `run_reference_probe.py`,
`run_representation_mapping.py`, `run_dream_aligned_residual_mapping.py`,
`run_dream_native_mc_lora.py`, `run_dream_three_choice_smoke.py`,
`run_dream_dual_eval.py`, `run_dream_residual_filter_extras.py`,
`run_dream_paper_residual_filtering.py`, `run_strong_lora_smoke.py`,
`weak_native_baseline.py`, `analyze_confidence_partitions.py`,
`analyze_map_artifacts.py`, `analyze_mapping_vs_confidence.py`, `analyze_rp.py`,
`analyze_stabilized_maps.py`, `plot_answer_count_vs_confidence_skew.py`,
`plot_confidence_dataset_selection.py`, `inspect_weak_confidence.py`,
`summarize_inference_csv.py`, `check_before_lm_head_activation.py`,
`print_hidden_vector.py`, `monitor_sweep.py`, `peek_results.py`, and the matching
`run_*.sh` wrappers.