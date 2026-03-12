# DO_NEXT — Phase 4 Status & Next Steps

**Last updated:** 2026-03-12
**Notebook:** `cross_layer_causal_sae_jailbreak_detection_V-1.3.ipynb` (100 cells)

---

## Current Status: Phase 4 — Data-Driven Feature Attribution

### What we've done (cells 14.0–14.11)

Cells 14.0–14.8 are complete and validated. Cells 14.9–14.11 explored alternative feature selection methods. All experiments concluded — ready to move to intervention.

### Key Finding: The 0.982 vs 0.675 Discrepancy

**Problem:** Elastic Net 435 features give AUC=0.982 in cell 14.7 but only AUC=0.675 in cell 14.11. These numbers should be comparable but aren't. All cell 14.9/14.10/14.11 experiments used `feature_matrix.npz` and got the deflated baseline, making absolute AUC values unreliable.

**Two pipelines exist:**

```
Cell 14.7 (AUC = 0.982):
  Pre-trained Phase 3 MLP (loaded from best_model.pt)
      ↓ evaluated on
  trajectory_dataset.pt features (435-dim, built by Phase 3 cell 13.5)
      ↓
  AUC = 0.982

Cell 14.11 (AUC = 0.675):
  NEW MLP (randomly initialized, trained from scratch)
      ↓ trained on
  feature_matrix.npz rows (selecting 435 EN columns from 524K)
      ↓
  AUC = 0.675
```

**Two confounded differences:**

1. **Features may not be identical** — Both use the same 435 SAE feature indices, but:
   - `trajectory_dataset.pt` was built by Phase 3's on-the-fly extraction (cell 13.5): model forward pass → SAE encode → SWiM → select 435 features → compute Δ-features → save
   - `feature_matrix.npz` was built by Phase 2 (cell 12.x): separate extraction run, possibly different token handling, normalization, or Δ-feature computation order
   - Even small numerical differences (floating point, different batch processing) could compound

2. **Pre-trained vs fresh MLP** — Cell 14.7 loads the already-trained Phase 3 MLP and just evaluates (no retraining). Cell 14.11 trains a fresh MLP from scratch in a simplified flat loop — all turns shuffled as independent samples, no per-trajectory batching.

**How to isolate the cause:**

| Test | What it does | If AUC ≈ 0.675 | If AUC ≈ 0.982 |
|------|-------------|-----------------|-----------------|
| **Test A** | Train fresh MLP on `trajectory_dataset.pt` (same 14.11 training procedure, Phase 3 features) | Problem is training procedure | Problem is feature values |
| **Test B** | Evaluate pre-trained Phase 3 MLP on `feature_matrix.npz` 435 columns | Problem is feature values | Problem is training procedure |

**Practical implication:** All drift/differential experiments (14.9–14.11) compared against the deflated 0.675 baseline. Relative rankings still hold (diff-only ≈ random, combined ≈ no improvement), but we don't know if differential features would help in the Phase 3 pipeline.

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

Drift correlation (even differential) is **univariate** — it ranks each feature independently. Elastic Net is **multivariate** — it finds feature *combinations* that jointly predict the label. A feature with high |differential| might be redundant with other features, or only useful in combination with features that have low |differential|.

| Property | Drift Correlation | Elastic Net |
|----------|------------------|-------------|
| Sees refused data? | No (jb-only) / Yes (differential) | Yes |
| Selection criterion | Per-feature temporal trend | Joint cross-class prediction |
| Considers feature interactions? | No (univariate) | Yes (multivariate) |
| What it finds | Features that change over time | Features that distinguish classes |

### MLP Retraining Results (cell 14.11, fixed val split)

All trained on `feature_matrix.npz` with proper val split (1734 train / 401 val turns):

| Experiment | N feat | AUC mean | AUC std |
|---|---|---|---|
| **Elastic Net 435 (baseline)** | **435** | **0.675** | **0.021** |
| EN 435 + top-100 diff | 535 | 0.685 | 0.033 |
| EN 435 + top-50 diff | 485 | 0.660 | 0.049 |
| EN 435 + top-200 diff | 635 | 0.613 | 0.041 |
| Diff-only 435 | 435 | 0.585 | 0.053 |
| Diff-only 200 | 200 | 0.542 | 0.038 |
| Balanced drift 200+200 | 400 | 0.606 | — |

**Conclusions:**
- Differential drift features provide at best marginal improvement over EN (+0.01 with top-100)
- Adding too many diff features hurts (noise overwhelms signal)
- Diff-only is near random — cannot replace Elastic Net
- The EN 435 features remain the best feature set for the MLP detector

---

## What To Do Next

### Priority 1: Resolve the 0.982 vs 0.675 gap (optional but recommended)

Run Test A and/or Test B (described above) to understand why feature_matrix.npz gives lower AUC. This determines whether future experiments need the Phase 3 pipeline or can use feature_matrix.npz.

**Implementation:** Add cell 14.12 that loads `trajectory_dataset.pt`, flattens it into per-turn (X, y) arrays, and trains a fresh MLP with the same 14.11 procedure. If AUC ≈ 0.982, the features are different; if AUC ≈ 0.675, the training procedure is the bottleneck.

### Priority 2: Proceed to Intervention (cells 14.12–14.16)

Accept Elastic Net 435 as the final feature set and proceed with F_H suppression:

| Cell | Task | Description |
|------|------|-------------|
| 14.12 | Compute intervention targets | Mean F_H activation during benign turns (score < 2) as suppression baseline |
| 14.13 | Implement intervention hook | NNSight hook: when D_t > τ, suppress F_H features toward benign baseline via SAE decoder |
| 14.14 | Run intervention evaluation | Re-run Crescendo attacks with hook active, measure ASR reduction |
| 14.15 | Utility evaluation | XSTest (BRR), MMLU, GSM8K on intervened model |
| 14.16 | Results & comparison | Compare all baselines, plot score trajectory suppression |

**Key design decisions (from RESEARCH_PLAN.md Section 7.2):**
- **Trigger:** D_t > τ (τ=0.4 for best F1, or τ=0.6 for zero FPR)
- **Mechanism:** Subtract-only on F_H features: `correction = target - current` only when `current > target`
- **Target values:** Mean activation during benign early turns (score < 2)
- **Injection:** Decode correction through SAE decoder into residual stream

### Priority 3: Phase 3 pending ablations (lower priority)

- SWiM window size ablation (M ∈ {4, 8, 16, 32}) — cell 13.10.4
- Within-turn pooling ablation (max vs mean vs last-token) — cell 13.10.5
- These can be done independently and don't block intervention

---

## File Reference

| File | Description |
|------|-------------|
| `results/feature_discovery/feature_matrix.npz` | 2135 × 524,288 full SAE feature matrix (Phase 2) |
| `results/feature_discovery/feature_matrix_meta.json` | Metadata with `turn_meta` per-row mapping |
| `results/feature_discovery/feature_sets.json` | 435 Elastic Net features (F_H + F_S definitions) |
| `results/mlp_detector/best_model.pt` | Phase 3 MLP checkpoint |
| `results/mlp_detector/trajectory_dataset.pt` | Phase 3 train/val dataset (435 features, per-trajectory) |
| `results/intervention/` | Phase 4 outputs (drift, attribution, ablation, differential results) |
| `reference/RESEARCH_PLAN.md` | Full research plan with all findings |
