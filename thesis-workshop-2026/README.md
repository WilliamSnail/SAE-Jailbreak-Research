# Thesis and workshop paper (2026)

This folder holds the code and results behind two pieces of writing:

| Writing | Notes |
|---|---|
| MS thesis, *Cross-layer SAE feature drift for early detection and mitigation of multi-turn jailbreaks* (Iowa State University, May 2026), DOI `10.31274/td-20260804-162` | The main write-up. |
| Paper for the ICML 2026 Mechanistic Interpretability Workshop, *Early Detection of Multi-Turn Jailbreak Escalation via Cross-Layer SAE Feature Drift*. Non-archival: it is only on [OpenReview](https://openreview.net/forum?id=MpdfElbiwL) (submitted 30 Apr 2026, last changed 11 Jun 2026). | A revised version of the thesis paper, written at about the same time. **Same code, same results.** |

New work after the thesis goes in [`../journal/`](../journal/).

## Which version do the writings cite?

Both writings cite the repository as it was on 26 April 2026 (commit `8a49456`). That exact version is saved as the tag **`thesis-final-2026-05`**, with the files at the top level of the repository, the way the writings describe them:

```
git checkout thesis-final-2026-05
```

On `main`, the same files are in this folder. A path from the writings like `results/...` is `thesis-workshop-2026/results/...` here. The move kept the file history (it was done with `git mv`).

## What is different from the cited version

The move itself changed nothing but the folder location. Two later commits on `main` changed three things. No result file was edited.

- **Regrouped result folders** (commit `b0086bf`, 26 April 2026, after the cited commit). Some result files moved into subfolders. In the tag, the moved files are at the paths the thesis uses. For example, the tag has `results/intervention_trajectories_tau999/` and `main` has `results/intervention_diagnoses/intervention_trajectories_tau999/`.
- **`intervention_ewl_analysis.py`** now reads the regrouped folders and covers all seven steering settings (`b0086bf`). It also prints an extra set of summary numbers (`0d5730d`).
- **`reference/RESEARCH_PLAN.md`** has a new section, 8.1.16 (`b0086bf`).

Every other file, including every result file, is the same as in the tag.

## What is in this folder

| Path | What it is |
|---|---|
| `cross_layer_causal_sae_jailbreak_detection_V-1.9.ipynb` | The main notebook. This is the final version used for the thesis. |
| `intervention_ewl_analysis.py`, `lr_vs_mlp_comparison.py` | Two analysis scripts. |
| `SAE_Steer_Explain.md` | A written explanation of the steering part. |
| `requirements_3.txt` | The list of Python packages, as it was. The versions are not fixed. |
| `results/` | All saved results: conversations, feature choices, detector outputs, figures, and the steering runs (one folder per run). See the guide below. |
| `archive/` | Earlier notebook versions (V-0.1 to V-1.8) and plans. |
| `reference/` | The research plan, progress notes and reference notebooks. The subfolder `reference/thesis/` (the LaTeX files of the thesis and the workshop paper) is ignored by git and exists only on the author's computer. |
| `test/` | Small test notebooks. |

## Words used in the results

- **Goal**: one harmful request taken from the JailbreakBench list. The study uses 100 goals.
- **Trajectory**: one multi-turn conversation about one goal. An attacker model talks to the target model step by step (the Crescendo method).
- **Judge score** (`raw_score`): a judge model scores each reply, up to 10. A trajectory counts as **jailbroken** if any turn scores above 8.
- **Detector**: a small model that reads the target model's internal features after each turn. It gives a warning score, `D_t`, between 0 and 1. A higher score means a jailbreak looks more likely.
- **τ (tau)**: the alarm level for `D_t`. When `D_t` goes above τ, the correction is applied. τ = 0.4 is the setting used in the steering runs. τ = 999 means the alarm never goes off, so the detector only watches.
- **Steering**: nudging the target model's internal features during a conversation to push it away from a jailbreak.
- **α (alpha)**: how strong the nudge is. α = 0 means no nudge, so those runs are controls.
- **F_H and F_S**: two groups of features. F_H is the group that notices harmful intent. F_S is the group that carries out a refusal. Some runs nudge only one group.
- **Δ (delta) features**: features that measure how a feature's activity changes from one turn to the next (the "drift" in the thesis title).
- **Phase 1** collects the attack conversations. **Phase 5** runs the steering experiments.

## Guide to `results/`

Every file in the trajectory folders is a JSON list of trajectories. The file name holds the date and time of the run.

| Folder | What is in it |
|---|---|
| `crescendo_trajectories/` | The Phase 1 conversations: five files, `trajectories_20260318_174058.json` to `trajectories_20260320_002849.json`, with 100 goals each (500 trajectories in total). `criteria_cache.json` holds the judge's criteria for each goal. `combined_20260324_035256.json` repeats `trajectories_20260320_002849.json`; the notebook only loads files named `trajectories_*.json`, so it skips the copy. `backup/` holds earlier runs from February and March 2026 and an earlier criteria cache. |
| `feature_discovery/` | The chosen features (`feature_sets.json`: layer, SAE index and whether it is a Δ feature), the labels for each turn (`y_hard.npy`, `y_soft.npy`), the fitted scikit-learn pipeline (`sklearn_pipeline.pkl`) and `feature_matrix_meta.json`. `feature_matrix.dat` here is only a 19.9 MB stand-in. The full matrix (about 4.3 GB) is not in the repository. |
| `mlp_detector/` | Figures for the detector: training curves, detector trajectories, and comparisons of loss functions, smoothing, and soft versus hard labels. |
| `intervention/` | Analysis outputs for choosing features and measuring drift: comparisons of selection methods, drift analysis, gradient attribution, feature-group ablation, and `comparison_results.json`. |
| `images/` | Charts of the Phase 1 runs, by category and by score over turns. |
| `intervention_trajectories/` | The Phase 5 steering runs (τ = 0.4, with 122 driver features, saved as `n_drivers`). One folder per setting (table below). |
| `intervention_diagnoses/` | Control and test runs from April 2026, with 200 trajectories each unless stated (table below). |

### `intervention_trajectories/`

The settings come from the `intervention_config` saved inside each file.

| Folder | α | Which features are nudged | Δ features nudged? | Trajectories | Run dates (2026) |
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

`prompt_aligned` and the four `Alpha_1_*` and `Alpha_3_*` folders with "all" are the control and steering settings used in the thesis's steering tables. A later re-analysis reproduced those tables exactly. See the thesis for the table numbers.

### `intervention_diagnoses/`

| Folder | α | Mode | Δ features? | Notes | Run dates (2026) |
|---|---|---|---|---|---|
| `intervention_trajectories_tau999` | 0 | all | no | τ = 999: the detector computes `D_t` but never changes anything | Apr 7 |
| `intervention_trajectories_Alpha_0` | 0 | all | no | control with no nudge | Apr 6 |
| `intervention_trajectories_Alpha_0_fix_plain_generate` | 0 | all | no | control, with a code variant named in the folder | Apr 8 |
| `intervention_trajectories_Alpha_0_all_generate()` | 0 | all | no | control, with a code variant named in the folder | Apr 10 |
| `intervention_trajectories_Alpha_0_skip_zerovector` | 0 | all | no | control, with a code variant named in the folder | Apr 10, 11 |
| `intervention_trajectories_FH_FS_Baseline` | 1 | `fh_suppress_fs_boost` | yes | | Apr 2 |
| `intervention_trajectories_FH_FS_Alpha_2.5` | 2.5 | `fh_suppress_fs_boost` | yes | | Apr 3 |
| `intervention_trajectories_FS_no_delta_Alpha_1` | 1 | `fs_boost_only` | no | | Apr 4 |
| `intervention_trajectories_FS_no_delta_Alpha_0.1` | 0.1 | `fs_boost_only` | no | | Apr 5 |
| `intervention_trajectories_backup` | 1 | not saved | not saved | the earliest runs; 500 trajectories; older file format | Mar 31, Apr 1 |

The mode names mean: `fh_suppress_fs_boost` lowers F_H and raises F_S; `fs_boost_only` only raises F_S. For the three control runs with a code variant in the name, the saved settings do not say what exactly was different.

## Running the notebook

The notebook reads and writes `results/...` with relative paths, so start it with **this folder as the working directory**.

`lr_vs_mlp_comparison.py` has a file path from the author's old Windows computer (`C:\Users\Lab622_TV\...`). Change it to your own path before you run it.
