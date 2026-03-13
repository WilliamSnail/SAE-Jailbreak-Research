# DO_NEXT — Phase 4 Status & Next Steps

**Last updated:** 2026-03-13
**Notebook:** `cross_layer_causal_sae_jailbreak_detection_V-1.3.ipynb` (101 cells)

---

## RESOLVED: Section 14 Bugs (5 issues found, 4 fixed)

**Correct baseline AUC: 0.9416** (from Phase 3 checkpoint, val-only)

### Bug Summary

| # | Bug | Cells | Impact | Fixed? |
|---|-----|-------|--------|--------|
| 1 | Index mapping (block vs interleaved) | 96,97,99,100 | Wrong features used, EN AUC=0.675 instead of ~0.94 | Yes |
| 2 | Eval leakage (train+val combined) | 95,98 | AUC inflated 0.9416→0.982 | Yes |
| 3 | Train/val split (goal overlap) | 97 | Only 8 val trajectories instead of ~61 | Yes |
| 4 | valid_mask not applied | 97 | jb_specific `\|corr_ref\| < THETA_FLAT` filter bypassed | Yes |
| 5 | Acc/F1 threshold mismatch | 95 | Acc/F1 unreliable (AUC unaffected) | No — AUC is primary metric |

### Bug 1: Index Mapping (cells 96, 97, 99, 100)

Code assumed `feature_matrix.npz` uses interleaved layout, but it uses block layout:
```
ACTUAL (block):      [L9_raw, L17_raw, L22_raw, L29_raw, L9_delta, L17_delta, L22_delta, L29_delta]
ASSUMED (interleaved): [L9_raw, L9_delta, L17_raw, L17_delta, ...]
```
Wrong formula: `gi = layer_idx * 2 * d_sae + ...`
Correct formula: `gi = (n_layers * d_sae if is_delta else 0) + layer_idx * d_sae + sae_idx`
Or: use `original_idx` from `feature_sets.json`.

Verified: `original_idx` gives Pearson r=1.0000 between `feature_matrix.npz` and `trajectory_dataset.pt`.

### Bug 2: Eval Leakage (cells 95, 98)

- Cell 95 (14.7): `for d in dataset` evaluated on train+val combined → AUC=0.982 (inflated). Fixed to `for d in val_dataset`.
- Cell 98 (14.10): `for d in bal_train + bal_val` in EVALUATION section. Fixed to `for d in bal_val`.

### Bug 3: Train/Val Split (cell 97)

Split by `if goal in train_goals` (checked first), but 42 of 46 val goals also exist in train_goals (same goal appears in multiple trajectories). Result: only 8 val trajectories.
Fixed: trajectory-signature matching `(goal, tuple(int(scores)))` → ~61 val trajectories. Same method used in cells 99/100.

### Bug 4: valid_mask Not Applied (cell 97)

Cell 14.9 selected features from raw `corr_full` without applying `valid_mask_saved`. In `jb_specific` mode, `valid_mask` enforces `|corr_ref| < THETA_FLAT` — without it, features that drift in both JB and refused could be selected, defeating the purpose.
Fixed: apply `corr_full_saved` + `valid_mask_saved`, zero out invalid features before selection.

### Bug 5: Acc/F1 Threshold Mismatch (cell 95) — NOT FIXED

MLP trained with soft labels (`score/10`), but acc/F1 use `preds > 0.5` threshold. A turn with score=6 produces pred~0.6 > 0.5 → false positive, since ground truth is `score > 8`. Correct threshold would be ~0.8.
**AUC is unaffected** (rank-based, threshold-independent). Since AUC is the primary comparison metric for ablation experiments, this is low priority.

### What was NOT affected

- **Drift correlation values (14.5, 14.8)** — operate on full 524K matrix directly, no manual index mapping
- **Phase 3 pipeline (cells 13.x)** — uses its own extraction, never touches `feature_matrix.npz` indices

---

## Experimental Results Summary (Phase 4)

### Drift Analysis (cells 14.5, 14.8)

| Analysis | Method | F_H | F_S | Key insight |
|----------|--------|-----|-----|-------------|
| 435-feature jb-only | Pearson corr (jailbroken only) | 309 | 3 | Most features escalate during jailbreaks |
| 435-feature differential | corr_jb - corr_refused | 294 | 2 | Refused barely drifts (mean=0.016), so differential ≈ jb-only |
| Full d_sae jb-only (524K) | Pearson corr (jailbroken only) | 32,948 | 19,728 | F_S exists in bulk but was filtered by EN |
| Full d_sae differential | corr_jb - corr_refused | 33,629 | 42,772 | **F_S outnumbers F_H** — many features erode in jb AND strengthen in refused |

### Why drift correlation fails as feature selection

Drift correlation (even differential) is **univariate** — it ranks each feature independently. Elastic Net is **multivariate** — it finds feature *combinations* that jointly predict the label.

| Property | Drift Correlation | Elastic Net |
|----------|------------------|-------------|
| Sees refused data? | No (jb-only) / Yes (differential) | Yes |
| Selection criterion | Per-feature temporal trend | Joint cross-class prediction |
| Considers feature interactions? | No (univariate) | Yes (multivariate) |
| What it finds | Features that change over time | Features that distinguish classes |

### MLP Retraining Results (cell 14.11, fixed val split)

**INVALIDATED** — all results below used wrong column indices (the index bug above). Need to re-run with corrected code.

| Experiment | N feat | AUC mean | AUC std | Status |
|---|---|---|---|---|
| **Elastic Net 435 (baseline)** | **435** | **0.675** | **0.021** | INVALID — wrong columns |
| EN 435 + top-100 diff | 535 | 0.685 | 0.033 | INVALID |
| EN 435 + top-50 diff | 485 | 0.660 | 0.049 | INVALID |
| EN 435 + top-200 diff | 635 | 0.613 | 0.041 | INVALID |
| Diff-only 435 | 435 | 0.585 | 0.053 | INVALID |
| Diff-only 200 | 200 | 0.542 | 0.038 | INVALID |
| Balanced drift 200+200 | 400 | 0.606 | — | INVALID |

---

## What To Do Next

### Priority 1: Proceed to Intervention (cells 14.13–14.17)

Accept Elastic Net 435 with the Phase 3 pipeline (val AUC=0.9416) as the final feature set and proceed with F_H suppression. Renumber from 14.13 since 14.12 is now the diagnostic cell.

| Cell | Task | Description |
|------|------|-------------|
| 14.13 | Compute intervention targets | Mean F_H activation during benign turns (score < 2) as suppression baseline |
| 14.14 | Implement intervention hook | NNSight hook: when D_t > τ, suppress F_H features toward benign baseline via SAE decoder |
| 14.15 | Run intervention evaluation | Re-run Crescendo attacks with hook active, measure ASR reduction |
| 14.16 | Utility evaluation | XSTest (BRR), MMLU, GSM8K on intervened model |
| 14.17 | Results & comparison | Compare all baselines, plot score trajectory suppression |

**Key design decisions (from RESEARCH_PLAN.md Section 7.2):**
- **Trigger:** D_t > τ (τ=0.4 for best F1, or τ=0.6 for zero FPR)
- **Mechanism:** Subtract-only on F_H features: `correction = target - current` only when `current > target`
- **Target values:** Mean activation during benign early turns (score < 2)
- **Injection:** Decode correction through SAE decoder into residual stream

### Priority 2: Re-run cells 14.9–14.12 with corrected indices

Now that the index bug is fixed, re-run all experiments to get correct results. EN baseline should now match Phase 3's ~0.94+ AUC.

### Priority 3: Phase 3 pending ablations (lower priority)

- SWiM window size ablation (M ∈ {4, 8, 16, 32}) — cell 13.10.4
- Within-turn pooling ablation (max vs mean vs last-token) — cell 13.10.5
- These can be done independently and don't block intervention

---

## File Reference

| File | Description | Trust level |
|------|-------------|-------------|
| `results/mlp_detector/trajectory_dataset.pt` | Phase 3 train/val dataset (435 features, per-trajectory) | **PRIMARY — use this** |
| `results/mlp_detector/best_model.pt` | Phase 3 MLP checkpoint (val AUC=0.9416) | **PRIMARY** |
| `results/feature_discovery/feature_matrix.npz` | 2135 × 524,288 full SAE features (Phase 2) | **Valid — use `original_idx` for EN column mapping** |
| `results/feature_discovery/feature_matrix_meta.json` | Metadata with `turn_meta` per-row mapping | Valid for metadata, not for feature values |
| `results/feature_discovery/feature_sets.json` | 435 Elastic Net features (F_H + F_S definitions) | Valid |
| `results/intervention/` | Phase 4 outputs (drift, attribution, ablation, differential, diagnostic) | Valid |
| `reference/RESEARCH_PLAN.md` | Full research plan with all findings | Valid |
