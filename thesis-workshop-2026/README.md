# Thesis and workshop paper (2026)

The code and results behind two pieces of writing, kept exactly as they were:

| Writing | Where | Notes |
|---|---|---|
| MS thesis, *Cross-layer SAE feature drift for early detection and mitigation of multi-turn jailbreaks* (Iowa State University, 2026-05), DOI `10.31274/td-20260804-162` | this folder | the main write-up |
| Paper for the ICML 2026 Mechanistic Interpretability Workshop, *Early Detection of Multi-Turn Jailbreak Escalation via Cross-Layer SAE Feature Drift*. Non-archival: available only on [OpenReview](https://openreview.net/forum?id=MpdfElbiwL) (submitted 30 Apr 2026; last modified 11 Jun 2026) | this folder | a revised version of the thesis paper, written at about the same time; **same code and same results** |

Nothing here was changed in the reorganisation except the folder location.
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
| `reference/` | research plan, progress notes, reference notebooks |
| `test/` | small test notebooks |

## Running the notebook

The notebook reads and writes `results/...` with relative paths, so start it with **this folder as the working directory**.

`lr_vs_mlp_comparison.py` contains a hard-coded Windows path (`C:\Users\Lab622_TV\...`) from the machine it was written on; change it to your own path before running.
