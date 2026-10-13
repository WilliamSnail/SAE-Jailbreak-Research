# SAE-Jailbreak-Research

Code and results for a study of **multi-turn jailbreaks**. A multi-turn jailbreak is when someone talks an AI model into giving harmful answers step by step, over several messages instead of one.

The study looks inside the model. It uses *sparse autoencoders* (SAEs), a tool that splits the model's internal activity into many small parts called *features*. The study asks two questions:

1. Can we spot a jailbreak early, while it is still building up?
2. Can we nudge the model's internals to stop it?

## Where to start

| I want to... | Go to |
|---|---|
| see the code and results behind the thesis or the workshop paper | [`thesis-workshop-2026/`](thesis-workshop-2026/) (it has its own README) |
| see the exact version the thesis cites | the tag `thesis-final-2026-05` (see below) |
| find the newer work after the thesis | [`journal/`](journal/) |

## Writings that cite this repository

| Writing | Link |
|---|---|
| MS thesis, *Cross-layer SAE feature drift for early detection and mitigation of multi-turn jailbreaks* (Iowa State University, May 2026) | DOI [`10.31274/td-20260804-162`](https://doi.org/10.31274/td-20260804-162) |
| Paper for the ICML 2026 Mechanistic Interpretability Workshop, *Early Detection of Multi-Turn Jailbreak Escalation via Cross-Layer SAE Feature Drift*. It is a revised version of the thesis paper and uses the same code and results. It is not in a proceedings (non-archival); it is only on OpenReview (submitted 30 Apr 2026, last changed 11 Jun 2026). | [OpenReview](https://openreview.net/forum?id=MpdfElbiwL) |

## If you came here from the thesis or the workshop paper

1. Both writings cite this repository as it was on 26 April 2026 (commit `8a49456`). That exact version is saved under the tag `thesis-final-2026-05`, with the files at the top level of the repository, as cited. To see it:
   ```
   git checkout thesis-final-2026-05
   ```
2. On the `main` branch, the same files now live in [`thesis-workshop-2026/`](thesis-workshop-2026/). A path from either writing, such as `results/...` or `cross_layer_causal_sae_jailbreak_detection_V-1.9.ipynb`, is now `thesis-workshop-2026/<that path>`. The files were moved with their history.
3. The thesis and the workshop paper are one body of work, so there is only one set of code.

## What changed after the thesis

Apart from the move, three things on `main` differ from the tagged version. No result file was edited. Every other file is the same as in the tag.

- Some result folders were regrouped into clearer subfolders (commit `b0086bf`). For example, the folder `results/intervention_trajectories_tau999/` is now `results/intervention_diagnoses/intervention_trajectories_tau999/`.
- The script `intervention_ewl_analysis.py` was updated to read the regrouped folders. It now covers all seven steering settings (`b0086bf`), and it prints an extra set of summary numbers (`0d5730d`).
- `reference/RESEARCH_PLAN.md` got a new section, 8.1.16 (`b0086bf`).

## Folders

| Folder | What it is |
|---|---|
| [`thesis-workshop-2026/`](thesis-workshop-2026/) | The code and results behind the thesis and the workshop paper, kept as cited apart from the changes listed above. Its README explains what is in each results folder. |
| [`journal/`](journal/) | New work after the thesis (the journal version). |
