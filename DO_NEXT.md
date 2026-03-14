# DO_NEXT — Status & Next Steps

**Last updated:** 2026-03-14
**Notebook:** `cross_layer_causal_sae_jailbreak_detection_V-1.4.ipynb`

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

When scaling from 1 run (100 trajectories, ~2135 turns) to N runs, memory scales linearly with total turns. Current data: ~4.2 GB RAM for X_full. At 10 runs: ~42 GB.

### Critical Bottlenecks

| # | Phase | Cell | Problem | Current (1 run) | At 10 runs | Fix |
|---|-------|------|---------|-----------------|------------|-----|
| 1 | **Phase 2** | 54 | `X_rows` list accumulates all turns in RAM before `np.stack()` | 4.2 GB | **~42 GB** | Stream to `np.memmap` instead of list accumulation |
| 2 | **Phase 2** | 55 | `np.savez_compressed` needs array + compression buffer | ~7 GB peak | **~70 GB** | Use `np.save` (uncompressed) or `np.memmap` |
| 3 | **Phase 2** | 56 | `X`, `X_stage1`, `X_stage2`, `X_scaled` coexist | ~8 GB | **~80 GB** | `del` intermediates after each stage |
| 4 | **Phase 4** | 94 | `vectorized_pearson` creates centered copy `X_c = X - X_mean` | ~9 GB | **~90 GB** | Chunk computation (50K features at a time) or `np.memmap` |

### Recommended Fixes by Scale

**2-3 runs (~6K turns, ~12 GB):** Add `del` statements to free intermediate arrays in Cell 56. Should fit in 32 GB RAM.

**10 runs (~21K turns, ~42 GB):**
1. **Cell 54:** Replace `X_rows` list with `np.memmap`:
   ```python
   X_mmap = np.memmap("feature_matrix.dat", dtype='float32', mode='w+', shape=(total_turns, 524288))
   # Write each row directly: X_mmap[row_idx] = x_t
   ```
2. **Cell 55:** Save as uncompressed `.npy` or keep memmap file directly.
3. **Cell 56:** Delete intermediates: `del X` after creating `X_stage1`, etc.
4. **Cell 94:** Chunk `vectorized_pearson` to process 50K features at a time instead of all 524K.

### Not Bottlenecks

| Phase | Why it's fine |
|-------|--------------|
| Phase 1 (Crescendomation) | GPU fixed at 7-8 GB (model size). Text trajectories are tiny. Per-run save already implemented. |
| Phase 3 (MLP training) | Dataset is 435 features per turn (~1.7 KB/turn). Even 10 runs = ~30 MB. |
| Disk space | Compressed feature matrix: ~800 MB/run. 10 runs ≈ 8 GB. Manageable. |

---

## What To Do Next

### Priority 1: Intervention Phase (Section 7.2 in RESEARCH_PLAN)

| Cell | Task | Description |
|------|------|-------------|
| 14.10 | Compute intervention targets | Mean F_H activation during benign turns (score < 2) as suppression baseline |
| 14.11 | Implement intervention hook | NNSight hook: when D_t > τ, suppress F_H toward benign baseline via SAE decoder |
| 14.12 | Run intervention evaluation | Re-run Crescendo attacks with hook active, measure ASR reduction |
| 14.13 | Utility evaluation | XSTest (BRR), MMLU, GSM8K on intervened model |
| 14.14 | Results & comparison | Compare baselines, plot score trajectory suppression |

### Priority 2: Phase 3 Ablations (Lower Priority)

- SWiM window size (M ∈ {4, 8, 16, 32})
- Within-turn pooling (max vs mean vs last-token)
- MLP hyperparameter tuning (Optuna on EN+100 feature set — see RESEARCH_PLAN 7.1.10.8)

### Priority 3: Implement Memory Fixes (Before Scaling)

Apply the bottleneck fixes above before running `NUM_RUNS > 1`.

---

## File Reference

| File | Description |
|------|-------------|
| `results/mlp_detector/trajectory_dataset.pt` | Phase 3 train/val dataset (435 features, per-trajectory) |
| `results/mlp_detector/best_model.pt` | Phase 3 MLP checkpoint (val AUC=0.9416) |
| `results/feature_discovery/feature_matrix.npz` | 2135 × 524,288 full SAE features |
| `results/feature_discovery/feature_matrix_meta.json` | Metadata with `turn_meta` per-row mapping |
| `results/feature_discovery/feature_sets.json` | 435 Elastic Net features (F_H + F_S definitions) |
| `reference/RESEARCH_PLAN.md` | Full research plan with all findings |
