# DO_NEXT — Status & Next Steps

**Last updated:** 2026-03-16
**Notebook:** `cross_layer_causal_sae_jailbreak_detection_V-1.5.ipynb`

---

## Bug Warnings (Resolved — Keep as Reference)

These bugs were found and fixed in V-1.3 → V-1.4. Keep as warnings to avoid reintroduction.

| # | Bug | Root Cause | Prevention |
|---|-----|-----------|------------|
| 1 | **Index mapping** | Block layout assumed interleaved → wrong features selected | Always use `original_idx` from `feature_sets.json`, never compute indices manually |
| 2 | **Eval leakage** | Evaluated on train+val combined → inflated AUC | All evaluation code must use `val_dataset` / `val_rows` only |
| 3 | **Train/val split** | Goal-string matching caused overlap (42/46 goals shared) | Use trajectory-signature `(goal, tuple(int(scores)))` for split |
| 4 | **valid_mask not applied** | Feature selection bypassed `jb_specific` filter | Apply `valid_mask` immediately after computing correlations |
| 5 | **Acc/F1 threshold** | Soft-label model uses 0.5 threshold but ground truth is score>8 | Report AUC only (rank-based, threshold-independent) |

**Layout reference:**
```
CORRECT (block):  [L9_raw, L17_raw, L22_raw, L29_raw, L9_delta, L17_delta, L22_delta, L29_delta]
WRONG (interleaved): [L9_raw, L9_delta, L17_raw, L17_delta, ...]
```

---

## Memory Bottlenecks for Multi-Run Scaling

**Current state:** 4 runs already exist (4 trajectory files × 100 goals each).
Cell 51 merges all files → 331 valid trajectories → 2,135 turns → X = 4.2 GB.
Each run produces ~83 valid trajectories and ~534 turns on average.

**Per-run math:** 1 run ≈ 100 trajectories (83 valid after ≥2-turn filter) ≈ 534 turns ≈ 1.04 GB.
Memory scales linearly with total turns. Machine has **16 GB RAM**.

| Scale | Runs | Valid trajs | Turns | X size | `stack()` peak | Fits 16 GB? |
|-------|------|-------------|-------|--------|----------------|-------------|
| Current | 4 | 331 | 2,135 | 4.2 GB | ~8.3 GB | Yes |
| 6 runs | ~498 | ~3,200 | 6.2 GB | ~12.5 GB | Yes (tight) |
| 8 runs | ~664 | ~4,270 | 8.3 GB | ~16.7 GB | No |
| **10 runs** | **~830** | **~5,340** | **10.4 GB** | **~20.8 GB** | **No** |

### Critical Bottlenecks

| # | Phase | Cell | Problem | Current (4 runs) | At 10 runs | Fix |
|---|-------|------|---------|-------------------|------------|-----|
| 1 | **Phase 2** | 54 | `X_rows` list accumulates all turns in RAM before `np.stack()` | 4.2 GB (8.3 peak) | **~20.8 GB peak** | Stream to `np.memmap` instead of list accumulation |
| 2 | **Phase 2** | 55 | `np.savez_compressed` needs array + compression buffer | ~7 GB peak | **~17 GB peak** | Use `np.save` (uncompressed) or `np.memmap` |
| 3 | **Phase 2** | 56 | `X`, `X_stage1`, `X_stage2`, `X_scaled` coexist | ~8 GB | **~20 GB** | ✅ DONE: `del` intermediates after each stage (9.69→5.58 GB, 42% reduction) |
| 4 | **Phase 4** | 94 | `vectorized_pearson` creates centered copy `X_c = X - X_mean` | ~9 GB | **~21 GB** | Chunk computation (50K features at a time) |

### Recommended Fixes by Scale

**6-7 runs (~3-4K turns, ~8 GB):** Add `del` statements in Cell 56. Fits in 16 GB.

**8-10 runs (~4-5K turns, ~10 GB):**
1. **Cell 54:** Replace `X_rows` list with `np.memmap`:
   ```python
   X_mmap = np.memmap("feature_matrix.dat", dtype='float32', mode='w+', shape=(total_turns, 524288))
   # Write each row directly: X_mmap[row_idx] = x_t
   ```
2. **Cell 55:** Save as uncompressed `.npy` or keep memmap file directly.
3. **Cell 56:** Delete intermediates: `del X` after creating `X_stage1`, etc. Load X via memmap.
4. **Cell 94:** Chunk `vectorized_pearson` to process 50K features at a time instead of all 524K.

### Not Bottlenecks

| Phase | Why it's fine |
|-------|--------------|
| Phase 1 (Crescendomation) | GPU fixed at 7-8 GB (model size). Text trajectories are tiny. Per-run save already implemented. |
| Phase 3 (MLP training) | Dataset is 435 features per turn (~1.7 KB/turn). Even 10 runs = ~30 MB. |
| Disk space | Uncompressed memmap: ~10 GB at 10 runs. Manageable. |

---

## What To Do Next

### Priority 1: Intervention Phase (Phase 5, Section 15 in notebook, Section 8 in RESEARCH_PLAN)

Section 14 = Phase 4 (ends with 14.13 handoff). Section 15 = Phase 5 intervention (new cells).

| Cell | Task | Description |
|------|------|-------------|
| 15.0 | Cleanup & load handoff | Load `phase4_handoff.pt`, model, SAEs |
| 15.1 | Implement intervention hook | NNSight hook: when D_t > τ, suppress causal drivers toward benign baseline via SAE decoder |
| 15.2 | Run intervention evaluation | Re-run Crescendo attacks with hook active, measure ASR reduction |
| 15.3 | Utility evaluation | XSTest (BRR), MMLU, GSM8K on intervened model |
| 15.4 | Results & comparison | Compare baselines, plot score trajectory suppression |

### Priority 2: Phase 3 Ablations (Lower Priority)

- SWiM window size (M ∈ {4, 8, 16, 32})
- Within-turn pooling (max vs mean vs last-token)
- MLP hyperparameter tuning (Optuna on EN+100 feature set — see RESEARCH_PLAN 7.1.10.8)

### Priority 3: Implement Memory Fixes (Before Scaling)

Apply the bottleneck fixes above before running beyond 7 total runs (currently at 4).

---

## File Reference

| File | Description |
|------|-------------|
| `results/mlp_detector/trajectory_dataset.pt` | Phase 3 train/val dataset (435 features, per-trajectory) |
| `results/mlp_detector/best_model.pt` | Phase 3 MLP checkpoint (val AUC=0.9416) |
| `results/feature_discovery/feature_matrix.npz` | 2135 × 524,288 full SAE features (4 runs merged) |
| `results/feature_discovery/feature_matrix_meta.json` | Metadata with `turn_meta` per-row mapping |
| `results/feature_discovery/feature_sets.json` | 435 Elastic Net features (F_H + F_S definitions) |
| `reference/RESEARCH_PLAN.md` | Full research plan with all findings |
