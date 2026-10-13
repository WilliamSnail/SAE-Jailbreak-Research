# SAE-Jailbreak-Research

Sparse-autoencoder (SAE) features for detecting, and trying to mitigate, multi-turn jailbreaks of language models.

## Cited by

This repository is cited as the code and results for:

| Writing | Reference |
|---|---|
| MS thesis, *Cross-layer SAE feature drift for early detection and mitigation of multi-turn jailbreaks* (Iowa State University, 2026-05) | DOI [`10.31274/td-20260804-162`](https://doi.org/10.31274/td-20260804-162) |
| Paper for the ICML 2026 Mechanistic Interpretability Workshop, *Early Detection of Multi-Turn Jailbreak Escalation via Cross-Layer SAE Feature Drift*. Non-archival: it is available only on OpenReview (submitted 30 Apr 2026; last modified 11 Jun 2026). A revised version of the thesis paper, using the same code and results | [OpenReview](https://openreview.net/forum?id=MpdfElbiwL) |

### If you arrived from one of these citations

- **The code state they refer to** is tagged **`thesis-final-2026-05`** (commit `8a49456`, 2026-04-26), with the files at the repository root exactly as cited:
  `git checkout thesis-final-2026-05`
- **On `main`**, the same files have been moved, with their history, into [`thesis-workshop-2026/`](thesis-workshop-2026/). A path cited in either writing, such as `results/...` or `cross_layer_causal_sae_jailbreak_detection_V-1.9.ipynb`, is now `thesis-workshop-2026/<that path>`.
- **Differences from the cited state, apart from the move.** Two commits made after the cited one are on `main`: `b0086bf` regrouped some result folders (for example the α = 0 diagnostic runs now sit under `results/intervention_diagnoses/`), updated `intervention_ewl_analysis.py` to read the regrouped folders and cover all seven steering settings, and added a section to `reference/RESEARCH_PLAN.md`; `0d5730d` added an aggregate-statistics section to `intervention_ewl_analysis.py`. All other files, including every result file, are identical to the tagged state. The folder's README lists what is where.
- The thesis and the workshop paper are one body of work; there is no separate workshop code.

## Folders

| Folder | What |
|---|---|
| [`thesis-workshop-2026/`](thesis-workshop-2026/) | Code and results cited by the thesis and the workshop paper, kept as cited apart from the differences listed above (the tag is the exact cited state). Its README has a guide to `results/`. |
| [`journal/`](journal/) | New work after the thesis (journal version). |
