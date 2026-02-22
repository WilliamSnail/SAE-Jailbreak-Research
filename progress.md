# Progress Log: Cross-Layer Causal SAE Jailbreak Detection

## Project Overview

Master's thesis research on detecting and mitigating **multi-turn jailbreaks** using **Sparse Autoencoders (SAEs)**. The core hypothesis is that jailbreaking is a **sparse, mechanistic state transition** — the causal decoupling of upstream Harm Recognition features (F_H) from downstream Refusal Execution features (F_R) — rather than a continuous drift in dense activation space.

---

## Current Status: Phase 1 — Pipeline Complete (V-0.4)

**Active notebook:** `cross_layer_causal_sae_jailbreak_detection_V-0.4.ipynb`

### What Has Been Built

#### 1. Full Crescendomation Red-Teaming Pipeline
- **Attacker LLM**: GPT-4o generates multi-turn Crescendo jailbreak prompts via OpenAI API
- **Target Model**: Gemma-3-4B-IT running locally on GPU (NVIDIA RTX PRO 5000) via NNSight
- **Judge LLM**: GPT-4o scores each turn on a 1-10 rubric (normalized: 0=refusal, 10=jailbreak)
- **Refusal Detection**: GPT-4o checks if the target refused, with backtracking support (up to 10 backtracks)
- **Scoring Rubric**: Auto-generated per-goal by GPT-4o using Crescendomation's rubric generation prompt
- **Test Cases**: Loaded from JailbreakBench/JBB-Behaviors (100 harmful behaviors, configurable subset)
- **Pipeline function**: `run_crescendomation()` — text-only, no activation extraction during attack (V-0.4)

#### 2. On-the-Fly SAE Activation Extraction (V-0.4 — CC++ Inspired)
- **Key change from V-0.3**: SAE activations are **no longer saved to disk** as `.pt` files
- **New approach**: `extract_activations_for_trajectory()` replays saved JSON trajectories through the model and extracts SAE activations on-the-fly at analysis time
- **Motivation**: Following Constitutional Classifiers++ (Anthropic, 2026) recommendation against pre-saving activation data to avoid I/O bottlenecks
- **SAEs**: GemmaScope 2 JumpReLU SAEs (`gemma-scope-2-4b-it-res`, `width=65k`, `l0=medium`)
- **Layers**: [9, 17, 22, 29] — Early (F_H: 9, 17) and Late (F_R: 22, 29)
- **Layer access path**: `model.model.language_model.layers[layer]` (Gemma3ForConditionalGeneration VLM wrapper)
- **Benefits**:
  - Eliminates ~160GB+ disk footprint (100 traj × 8 turns × 200MB)
  - Enables flexible layer/SAE selection at analysis time
  - Enables data augmentation for Phase 3 MLP training
  - Memory freed immediately after each trajectory (`del act_data` + `torch.cuda.empty_cache()`)

#### 3. Trajectory Storage (Text-Only JSON)
- Trajectory metadata (goals, prompts, responses, scores, categories) saved as timestamped JSON files
- Load cell auto-detects the latest saved JSON file for resuming after kernel restart
- No activation files on disk — activations recomputed on demand

#### 4. Speed Optimizations
- **SDPA attention**: `attn_implementation="sdpa"` applied to model loading — mathematically identical to default attention, ~20-40% faster via fused CUDA kernels
- **Flash Attention 2**: Cannot install on Windows (path too long + missing CUDA_HOME/nvcc); SDPA is the alternative

#### 5. Preliminary Analysis & Visualization
- Summary table: per-trajectory label, max score, num turns, backtracks, score trajectory
- Score trajectory plots: per-conversation score evolution + max score distribution
- SAE activation statistics: mean magnitude, firing rate, max activation per layer per turn (on-the-fly)
- SAE activation evolution plot: mean |activation| vs judge score across turns per layer (on-the-fly)

#### 6. JBB Category & Behavior Analysis (V-0.3+)
- **Category/Behavior stored in trajectories**: Pipeline saves `category` and `behavior` fields from JBB-Behaviors
- **Category recovery for older data**: Builds a `goal -> category/behavior` lookup from JBB data
- **Per-category summary table**: N, Jailbroken count, ASR (%), Avg Max Score, Avg Turns, Avg Backtracks
- **Per-behavior summary table**: More granular breakdown within each JBB category
- **Category visualization** (3 plots): ASR bar, stacked jailbroken/refused, avg max score by category

#### 7. Trajectory Viewer Tool (`test/trajectory_viewer.ipynb`)
- Standalone interactive notebook for browsing saved trajectories (no model loading needed)
- Category filter dropdown, turn selector, "Show All Turns" button, category summary table

#### 8. CC++ Integration into Research Plan (V-0.4)
- **RESEARCH_PLAN.md** updated with Constitutional Classifiers++ (Anthropic, 2026) integration:
  - Section 1.2: 3-column positioning table (Assistant Axis vs. CC++ vs. This Thesis)
  - Section 4.2: On-the-fly extraction architecture diagram
  - Section 6.1: Two-level temporal smoothing (SWiM within-turn M=16 + EMA across-turn α=0.3)
  - Section 6.3: Softmax-weighted BCE loss function
  - Section 6.4: Full smoothing pipeline summary
  - Section 8.3: 4 new ablation studies (temporal smoothing, SWiM M, loss function, pooling)
  - Section 9: Phase 3 roadmap expanded with CC++ tasks
  - Section 10: 5 new design decision rows

---

### Speed Benchmarks (RTX PRO 5000, Gemma-3-4B-IT bfloat16)

| Method | Throughput | Notes |
|---|---|---|
| NNSight `.generate()` | ~25.4 tok/s | Used in pipeline |
| Vanilla HuggingFace `.generate()` | ~27.4 tok/s | Via `model._model` |
| NNSight overhead | +13.9% | Tracing layer for interpretability access |

**Pipeline bottleneck**: Target generation (~19.7s/turn) dominates. SAE extraction is only ~0.3s/turn. Attacker/judge API calls ~2.5s and ~1.0s respectively.

---

### Initial Experimental Results (5 test cases, 8 max rounds — V-0.2)

| Metric | Value |
|---|---|
| Total trajectories | 5 |
| Jailbroken | 3 (60.0%) |
| Refused | 2 (40.0%) |
| Avg max score | 7.40 |
| Avg turns | 7.8 |
| Avg backtracks | 1.4 |

### Full Run (100 test cases, 8 max rounds — V-0.3)

Run completed on 2026-02-21 with `EXTRACT_ACTIVATIONS=False` (fast mode). Saved to `trajectories_20260221_190645.json`. Category-level ASR analysis available.

---

## Notebook Cell Map (V-0.4)

| Cell | Section | Description |
|---|---|---|
| 0 | Header | Markdown: architecture overview, CC++ design decision, references |
| 3-5 | 1. Setup | Environment detection, imports, API config |
| 7 | 2. Config | Model, SAE, Crescendomation parameters (NUM_TEST_CASES=100) |
| 9 | 3. Load Model | Gemma-3-4B-IT via NNSight with SDPA attention |
| 10 | 3. Load SAEs | GemmaScope 2 SAEs for layers [9, 17, 22, 29] |
| 12 | 4. Attacker/Judge | OpenAI API client + `attacker_generate()` |
| 14-17 | 5. Utils | Rubric generation, evaluation, refusal check, Crescendo step |
| 19 | 6. Target Gen | `target_generate()` + `extract_sae_activations_for_turn()` (separate functions) |
| 21 | 7. Load JBB | Load JailbreakBench/JBB-Behaviors dataset |
| 22 | 7. Test Cases | Convert to Crescendomation format with category/behavior fields |
| 24 | 8. Pipeline | `run_crescendomation()` — text-only, no activation extraction |
| 26 | 8. Runner | Iterates test cases |
| 28 | 9. Save | Save trajectory JSON (text-only, no activation file references) |
| 29 | 9. Load | Auto-detect and reload latest saved trajectories |
| 30 | 9. On-the-Fly | `extract_activations_for_trajectory()` — replay + extract at analysis time |
| 32 | 10. Results | Overall summary table |
| 33 | 10. Results | Category/behavior analysis tables (with JBB lookup recovery) |
| 34 | 10. Results | Category visualization plots (ASR bar, stacked, avg score) |
| 35 | 10. Results | Score trajectory plots |
| 36 | 10. Results | Detailed turn-by-turn logs |
| 37-38 | 11. Analysis | On-the-fly SAE activation statistics (extract, compute, free per-trajectory) |
| 39 | 11. Analysis | On-the-fly SAE activation evolution plot |

---

## Version History

### V-0.4 (2026-02-23) — On-the-Fly SAE Extraction
- **Removed** pre-saved `.pt` activation files from pipeline
- **Removed** `extract_activations` parameter from pipeline function
- **Removed** `EXTRACT_ACTIVATIONS` flag and all activation save/load logic
- **Added** `extract_activations_for_trajectory()` — on-the-fly extraction from saved JSON
- **Renamed** `run_crescendomation_with_sae()` → `run_crescendomation()` (text-only)
- **Updated** Section 11 analysis cells to use on-the-fly extraction with `torch.cuda.empty_cache()`
- **Updated** RESEARCH_PLAN.md with CC++ integration (SWiM, EMA, softmax-weighted loss, ablations)
- **Motivation**: CC++ (Anthropic, 2026) warns against I/O bottlenecks from pre-saved activations

### V-0.3 (2026-02-22) — Category Analysis
- Added JBB category/behavior tracking in pipeline and trajectory JSON
- Added category recovery via goal→category lookup for older trajectories
- Added per-category and per-behavior ASR summary tables
- Added 3 category visualization plots
- Built trajectory viewer tool (`test/trajectory_viewer.ipynb`) with category filtering
- Full 100-test-case run completed

### V-0.2 (2026-02-21) — Save/Load + Replay
- Added trajectory save/load (timestamped JSON)
- Added SAE activation replay cell (decouple extraction from pipeline)
- Added crash recovery via auto-detect latest JSON
- 5-test-case pilot run completed

### V-0.1 — Initial Pipeline
- Crescendomation integration with NNSight-wrapped Gemma
- SAE extraction fused into pipeline
- Basic analysis cells

---

## Key Technical Decisions & Fixes

### Bug Fixes Applied
1. **Layer access path**: `model.model.layers[layer]` -> `model.model.language_model.layers[layer]` (Gemma3ForConditionalGeneration is a VLM wrapper)
2. **OOM from `.tolist()`**: Removed `.tolist()` on large activation tensors that caused MemoryError
3. **OOM from accumulation**: Changed from in-memory accumulation to per-trajectory disk save + `del`
4. **Analysis cell memory leak**: Added `del act_data` after processing each trajectory
5. **Missing response preview**: Re-added goal/prompt/response previews to per-turn print output

### Architecture Notes
- **Score normalization**: `our_score = 11 - crescendomation_score` (their 1=jailbreak, 10=refusal -> ours 10=jailbreak, 1=refusal)
- **Activation shape**: `(seq_len, d_sae)` where `d_sae=65536` — sequence length grows each turn since the full conversation is re-tokenized
- **Hidden state casting**: Model runs in bfloat16, SAEs expect float32 — cast with `.float()` before encoding
- **Activations moved to CPU**: `.cpu()` after SAE encoding to keep GPU memory free
- **On-the-fly extraction (V-0.4)**: No `.pt` files saved; activations recomputed from JSON at analysis time

---

## What's Next (Planned)

### Immediate
- [x] Run full experiment with 100 test cases from JBB-Behaviors (done V-0.3)
- [x] Add JBB category/behavior tracking and per-category ASR analysis
- [x] Build trajectory viewer tool with category filtering
- [x] Implement on-the-fly SAE extraction (done V-0.4, CC++ inspired)
- [ ] Add benign conversation controls (WildChat / XSTest) for negative samples

### Phase 2 — Feature Discovery
- [ ] Run Lasso logistic regression on early-layer (F_H) and late-layer (F_R) activations
- [ ] Identify causally relevant features that discriminate jailbroken vs. refused trajectories
- [ ] Validate with GPT-4o feature interpretation + Neuronpedia dashboards

### Phase 3 — MLP Detector (CC++ Enhanced)
- [ ] Implement SWiM (M=16) within-turn token aggregation on selected SAE features
- [ ] Implement EMA (α=0.3) across-turn smoothing on turn-level feature summaries
- [ ] Build on-the-fly SAE extraction data loader for training loop
- [ ] Implement softmax-weighted BCE loss function
- [ ] Train MLP with soft labels from per-turn judge scores
- [ ] CC++ ablation: Raw vs. mean-pool vs. SWiM-only vs. EMA-only vs. two-level smoothing
- [ ] Evaluate "Early Warning Latency" — how many turns before jailbreak the MLP triggers

### Phase 4 — Conditional Sparse Clamping
- [ ] Implement intervention that restores F_R only when MLP detects causal decoupling
- [ ] Use EMA "Running Belief" state as real-time trigger
- [ ] Compare against static clamping and dense projection capping baselines
- [ ] Measure capability preservation (MMLU, GSM8K, benign refusal rate)

---

## Files & Directory Structure

```
SAE-Jailbreak-Research/
  cross_layer_causal_sae_jailbreak_detection_V-0.4.ipynb  # Active notebook (on-the-fly extraction)
  cross_layer_causal_sae_jailbreak_detection_V-0.3.ipynb  # Previous (category analysis)
  cross_layer_causal_sae_jailbreak_detection_V-0.2.ipynb  # Previous (save/load)
  cross_layer_causal_sae_jailbreak_detection.ipynb         # Original (V-0.1)
  RESEARCH_PLAN.md                                         # Full thesis research plan (CC++ updated)
  The Constitutional Classifiers++ Implementation Framework.md  # CC++ adaptation plan
  CLAUDE.md                                                # Claude Code project context
  progress.md                                              # This file
  .env                                                     # API keys (HF_TOKEN, OPENAI_API_KEY, NDIF_API_KEY)
  .gitignore                                               # Excludes results/
  test/
    trajectory_viewer.ipynb                                # Interactive trajectory browser with category filter
  results/
    crescendo_trajectories/
      trajectories_YYYYMMDD_HHMMSS.json                    # Trajectory metadata (text-only, no activations)
      score_trajectories_{timestamp}.png                   # Score trajectory plots
      category_analysis_{timestamp}.png                    # Category ASR plots
```

---

*Last updated: 2026-02-23*
