# Thesis and workshop paper (2026)

The code and results behind two pieces of writing, kept exactly as they were:

| Writing | Where | Notes |
|---|---|---|
| MS thesis, *Cross-layer SAE feature drift for early detection and mitigation of multi-turn jailbreaks* (Iowa State University, 2026-05), DOI `10.31274/td-20260804-162` | this folder | the main write-up |
| Paper for the ICML 2026 Mechanistic Interpretability Workshop, *Early Detection of Multi-Turn Jailbreak Escalation via Cross-Layer SAE Feature Drift*. Non-archival: available only on [OpenReview](https://openreview.net/forum?id=MpdfElbiwL) (submitted 30 Apr 2026; last modified 11 Jun 2026) | this folder | a revised version of the thesis paper, written at about the same time; **same code and same results** |

Nothing here was changed in the reorganisation except the folder location.
One later change is on `main` only: commit `b0086bf` (2026-04-26, made after the commit the thesis cites, `8a49456`) moved some result files into the subfolders described under *`results/` in detail* below. In the tag `thesis-final-2026-05` those files are at the paths the thesis uses (for example `results/intervention_trajectories_tau999/` instead of `results/intervention_diagnoses/intervention_trajectories_tau999/`).
New work after the thesis goes in [`../journal/`](../journal/).

## Cited by the thesis and the workshop paper

Both writings cite this repository. Paths they mention (for example `results/...`) were written for the original layout, where this folder's contents sat at the repository root. They are now `thesis-workshop-2026/<that path>`, or use the tag below to see the layout exactly as cited.

## The original layout is preserved as a tag

The thesis cites this repository as it was at commit `8a49456` (2026-04-26). That exact state, with files at the repository root, is tagged **`thesis-final-2026-05`**:

```
git checkout thesis-final-2026-05
```

On `main`, the same files live in this folder. Their history is preserved (they were moved with `git mv`).

## Contents

| Path | What it is |
|---|---|
| `cross_layer_causal_sae_jailbreak_detection_V-1.9.ipynb` | the main notebook (final version used for the thesis) |
| `intervention_ewl_analysis.py`, `lr_vs_mlp_comparison.py` | analysis scripts |
| `SAE_Steer_Explain.md` | explanation of the steering part |
| `requirements_3.txt` | the dependency list as it was (versions are not pinned) |
| `results/` | trajectories, feature-discovery outputs, detector outputs, figures, and the intervention runs, one folder per run |
| `archive/` | earlier notebook versions V-0.1 to V-1.8 and plans |
| `reference/` | research plan, progress notes, reference notebooks. `reference/thesis/` (LaTeX sources of the thesis and the workshop paper) is git-ignored and exists only on the author's machine |
| `test/` | small test notebooks |

## `results/` in detail

A "trajectory" is one multi-turn Crescendo conversation about one JailbreakBench goal. A trajectory counts as jailbroken if any turn's `raw_score` is above 8. Every file is a JSON list of trajectories; the file name carries the run's date and time.

| Folder | What it holds |
|---|---|
| `crescendo_trajectories/` | Phase 1 attack data: five files `trajectories_20260318_174058.json` to `trajectories_20260320_002849.json`, 100 goals each, 500 trajectories in total. `criteria_cache.json` holds the per-goal judge criteria. `combined_20260324_035256.json` repeats `trajectories_20260320_002849.json` (the notebook loads `trajectories_*.json`, so it is skipped). `backup/` has earlier runs from February and March 2026 and an earlier criteria cache. |
| `feature_discovery/` | Selected features (`feature_sets.json`: layer, SAE index, delta flag), turn labels (`y_hard.npy`, `y_soft.npy`), the fitted `sklearn_pipeline.pkl` and `feature_matrix_meta.json`. `feature_matrix.dat` here is a 19.9 MB stub; the full matrix (about 4.3 GB) is not in the repository. |
| `mlp_detector/` | Figures for the detector: training curves, detector trajectories, and loss-function, smoothing and soft-vs-hard-label ablations. |
| `intervention/` | Analysis outputs for feature selection and drift: selection comparisons, drift analysis, gradient attribution, feature-group ablation, `comparison_results.json`. |
| `images/` | Category and score-trajectory figures for the Phase 1 runs. |
| `intervention_trajectories/` | Phase 5 steering runs (τ = 0.4, `n_drivers` = 122), one folder per setting. Details below. |
| `intervention_diagnoses/` | Control and diagnostic runs from April 2026, 200 trajectories each unless stated. Details below. |

### `intervention_trajectories/`

The settings below are read from `intervention_config` inside the saved files.

| Folder | α | Steered features | Δ features steered | Trajectories | Run dates (2026) |
|---|---|---|---|---|---|
| `prompt_aligned` | 0 | all | no | 200 | Apr 12 |
| `prompt_unaligned` | 0 | all | no | 200 | Apr 12 |
| `Alpha_1_steer_all_delta_false_baseline` | 1 | all | no | 200 | Apr 13 |
| `Alpha_1_steer_all_delta_true_baseline` | 1 | all | yes | 300 | Apr 23, 26 |
| `Alpha_3_steer_all_delta_false_baseline` | 3 | all | no | 300 | Apr 14, 15 |
| `Alpha_3_steer_all_delta_true_baseline` | 3 | all | yes | 300 | Apr 15, 16 |
| `Alpha_3_steer_F_H_only_delta_false_baseline` | **1** (the folder name says 3) | F_H only | no | 300 | Apr 19 |
| `Alpha_-3_steer_all_delta_false_baseline` | -3 | all | no | 300 | Apr 21 |
| `Alpha_30_steer_all_delta_false_baseline` | 30 | all | no | 300 | Apr 22 |

`prompt_aligned` and the four `Alpha_1_*` / `Alpha_3_*` "all" folders are the control and steering settings of the thesis's steering tables (reproduced exactly in a later re-analysis; see the thesis for the table numbers).

### `intervention_diagnoses/`

| Folder | α | Mode | Δ | Notes | Run dates (2026) |
|---|---|---|---|---|---|
| `intervention_trajectories_tau999` | 0 | all | no | τ = 999: the detector computes D_t but never intervenes | Apr 7 |
| `intervention_trajectories_Alpha_0` | 0 | all | no | α = 0 control | Apr 6 |
| `intervention_trajectories_Alpha_0_fix_plain_generate` | 0 | all | no | α = 0, code variant named in the folder | Apr 8 |
| `intervention_trajectories_Alpha_0_all_generate()` | 0 | all | no | α = 0, code variant named in the folder | Apr 10 |
| `intervention_trajectories_Alpha_0_skip_zerovector` | 0 | all | no | α = 0, code variant named in the folder | Apr 10, 11 |
| `intervention_trajectories_FH_FS_Baseline` | 1 | `fh_suppress_fs_boost` | yes | | Apr 2 |
| `intervention_trajectories_FH_FS_Alpha_2.5` | 2.5 | `fh_suppress_fs_boost` | yes | | Apr 3 |
| `intervention_trajectories_FS_no_delta_Alpha_1` | 1 | `fs_boost_only` | no | | Apr 4 |
| `intervention_trajectories_FS_no_delta_Alpha_0.1` | 0.1 | `fs_boost_only` | no | | Apr 5 |
| `intervention_trajectories_backup` | 1 | not recorded | not recorded | earliest runs; 500 trajectories; older config format | Mar 31, Apr 1 |

The folder names for the three α = 0 variants describe code variants; the saved configs do not record what differed.

## Running the notebook

The notebook reads and writes `results/...` with relative paths, so start it with **this folder as the working directory**.

`lr_vs_mlp_comparison.py` contains a hard-coded Windows path (`C:\Users\Lab622_TV\...`) from the machine it was written on; change it to your own path before running.
