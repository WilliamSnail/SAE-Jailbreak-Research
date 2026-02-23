# Progress Log: Cross-Layer Causal SAE Jailbreak Detection

## Project Overview

Master's thesis research on detecting and mitigating **multi-turn jailbreaks** using **Sparse Autoencoders (SAEs)**. The core hypothesis is that jailbreaking is a **sparse, mechanistic state transition** — the causal decoupling of upstream Harm Recognition features (F_H) from downstream Refusal Execution features (F_R) — rather than a continuous drift in dense activation space.

---

## Current Status: Phase 1 — Pipeline Complete (V-0.7)

**Active notebook:** `cross_layer_causal_sae_jailbreak_detection_V-0.7.ipynb`

### What Has Been Built

#### 1. Full Crescendomation Red-Teaming Pipeline
- **Attacker LLM**: GPT-4o generates multi-turn Crescendo jailbreak prompts via OpenAI API
- **Target Model**: Gemma-3-4B-IT running locally on GPU (NVIDIA RTX PRO 5000) via NNSight
- **Judge LLM**: GPT-4o scores each turn on a 1-10 rubric (normalized: 0=refusal, 10=jailbreak)
- **Refusal Detection**: GPT-4o checks if the target refused, with backtracking support (up to 10 backtracks)
- **Scoring Rubric**: Auto-generated per-goal by GPT-4o using Crescendomation's rubric generation prompt
- **Test Cases**: Loaded from JailbreakBench/JBB-Behaviors (100 harmful behaviors, configurable subset)
- **Pipeline function**: `run_crescendomation()` — text-only, no activation extraction during attack

#### 2. On-the-Fly SAE Activation Extraction (V-0.4 — CC++ Inspired)
- **Key change from V-0.3**: SAE activations are **no longer saved to disk** as `.pt` files
- **New approach**: `iter_trajectory_activations()` generator replays saved JSON trajectories through the model and extracts SAE activations on-the-fly at analysis time, one turn at a time (memory-safe)
- **Motivation**: Following Constitutional Classifiers++ (Anthropic, 2026) recommendation against pre-saving activation data to avoid I/O bottlenecks
- **SAEs**: GemmaScope 2 JumpReLU SAEs (`gemma-scope-2-4b-it-res`, `width=65k`, `l0=medium`)
- **Layers**: [9, 17, 22, 29] — Early (F_H: 9, 17) and Late (F_R: 22, 29)
- **Layer access path**: `model.model.language_model.layers[layer]` (Gemma3ForConditionalGeneration VLM wrapper)
- **Benefits**:
  - Eliminates ~160GB+ disk footprint (100 traj × 8 turns × 200MB)
  - Enables flexible layer/SAE selection at analysis time
  - Enables data augmentation for Phase 3 MLP training
  - Memory freed immediately after each turn (`del acts` + `torch.cuda.empty_cache()`)

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

#### 10. SWiM-Aggregated SAE Extraction (V-0.7 — CC++ Level 1)
- **SWiM (Sliding Window Mean)**: Converts sparse per-token SAE activations into smooth turn-level summaries
- **Algorithm**: Slide a 16-token window along the sequence dimension (`avg_pool1d`, stride=1), then max-pool across positions → `(d_sae,)` per layer per turn
- **Why needed**: Raw SAE features are extremely sparse — a harm feature firing on 2 tokens out of 500 gets diluted to near-zero by averaging. SWiM preserves local bursts by only averaging within a 16-token neighborhood, then taking the peak.
- **Functions implemented**:
  - `swim_aggregate()` — core SWiM: avg_pool1d + max/mean pool
  - `extract_sae_activations_swim()` — wraps existing NNSight trace + SAE encode + SWiM
  - `iter_trajectory_activations_swim()` — memory-safe generator yielding `(d_sae,)` per turn (~260KB vs ~200MB+ for raw)
  - `extract_trajectory_swim()` — accumulates all turns safely (memory-safe due to compact SWiM output)
- **Hyperparameters**: `SWIM_WINDOW_SIZE=16` (CC++ Figure 5b optimal), `SWIM_POOL_MODE="max"` (peak concept intensity)
- **Performance**: Negligible overhead vs raw extraction (same NNSight trace + SAE encode bottleneck; SWiM adds only avg_pool1d + max)
- **Top-K Feature Tracking**: Visualization cell tracks top-20 SAE latent indices per layer across turns, categorizing features as persistent (F_H candidates), disappeared (F_R candidates), emerged, or high-score-only
- **This is Level 1 of the two-level smoothing pipeline** (Level 2: EMA across-turn = Phase 3)

#### 6. JBB Category & Behavior Analysis (V-0.3+)
- **Category/Behavior stored in trajectories**: Pipeline saves `category` and `behavior` fields from JBB-Behaviors
- **Category recovery for older data**: Builds a `goal -> category/behavior` lookup from JBB data
- **Per-category summary table**: N, Jailbroken count, ASR (%), Avg Max Score, Avg Turns, Avg Backtracks
- **Per-behavior summary table**: More granular breakdown within each JBB category
- **Category visualization** (3 plots): ASR bar, stacked jailbroken/refused, avg max score by category

#### 7. Trajectory Viewer Tool (`test/trajectory_viewer.ipynb`)
- Standalone interactive notebook for browsing saved trajectories (no model loading needed)
- Category filter dropdown, turn selector, "Show All Turns" button, category summary table
- **V-0.6**: Now displays **scoring criteria** (collapsible) and **refusal judge** details per turn

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

#### 9. Enhanced Trajectory Data & Criteria Caching (V-0.6)
- **Criteria cache**: Disk-backed JSON cache (`criteria_cache.json`) keyed by goal string. On first run, `generate_score_rubric()` calls GPT-4o and saves the result. On re-runs with the same goal, the cached criteria is reused instantly (no API call). Cache is auto-seeded from loaded trajectories.
- **Refusal judge dict saved per turn**: `check_refusal()` now returns the full dict `{value, rationale, metadata, refused}` instead of just a bool. Every turn record (both backtracked and non-backtracked) includes a `refusal_judge` field with the judge's reasoning and confidence score.
- **Score judge dict saved per turn**: `evaluate_with_rubric()` now returns the full dict `{rationale, Score, metadata}` instead of just an int. Every non-backtracked turn record includes a `score_judge` field with the judge's reasoning and confidence. Backtracked turns have `score_judge: null` (not scored).
- **Explicit confidence definition**: Both `CHECK_REFUSAL_SYSTEM_PROMPT` and `evaluate_with_rubric()` prompt updated to explicitly define `metadata` as a confidence score (0-100), rather than leaving the judge to infer from examples alone.
- **Trajectory viewer updated**: Displays scoring criteria (collapsible `<details>`) in trajectory header, refusal judge details, and score judge details (rationale, confidence) per turn. HTML-escaped for safe display. Gracefully handles older trajectories without these fields.

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

## Notebook Cell Map (V-0.7)

| Cell | Section | Description |
|---|---|---|
| 0 | Header | Markdown: architecture overview, CC++ design decision, references |
| 1 | Next Step | Markdown: check Promptfoo for correct prompt |
| 3-5 | 1. Setup | Environment detection, imports, API config |
| 7 | 2. Config | Model, SAE, Crescendomation parameters, criteria cache path |
| 9 | 3. Load Model | Gemma-3-4B-IT via NNSight with SDPA attention |
| 11 | 4. Attacker/Judge | OpenAI API client + `attacker_generate()` |
| 13 | 5. Rubric | `generate_score_rubric()` with disk-backed criteria cache |
| 14 | 5. Evaluate | `evaluate_with_rubric()` + `normalize_score()` |
| 15 | 5. Refusal | `check_refusal()` — returns full dict {value, rationale, metadata, refused} |
| 16 | 5. Attacker | `generate_crescendo_step()` — Crescendo strategy prompt |
| 18 | 6. Target Gen | `target_generate()` — NNSight generation |
| 20-21 | 7. Load JBB | Load JailbreakBench/JBB-Behaviors + convert to test cases |
| 23 | 8. Pipeline | `run_crescendomation()` — saves criteria, refusal_judge per turn |
| 25 | 8. Runner | Iterates test cases |
| 27 | 9. Save | Save trajectory JSON (text-only, includes criteria + refusal_judge) |
| 28 | 9. Load | Auto-detect and reload latest saved trajectories |
| 29 | 9. Cache Seed | Seed criteria cache from loaded trajectories |
| 31 | 10. Results | Overall summary table |
| 32 | 10. Results | Category/behavior analysis tables (with JBB lookup recovery) |
| 33 | 10. Results | Category visualization plots (ASR bar, stacked, avg score) |
| 34 | 10. Results | Score trajectory plots |
| 35 | 10. Results | Detailed turn-by-turn logs |
| 37 | 11. Load SAE | GemmaScope 2 SAEs for layers [9, 17, 22, 29] |
| 39 | 11. Extract Fn | `extract_sae_activations_for_turn()` — SAE encoding |
| 40 | 11. On-the-Fly | `iter_trajectory_activations()` generator + `extract_activations_for_trajectory()` |
| 41 | 11. Analysis | On-the-fly SAE activation statistics (memory-safe) |
| 42 | 11. Analysis | On-the-fly SAE activation evolution plot |
| 43 | 11. SWiM | Markdown: SWiM algorithm explanation + two-level pipeline overview |
| 44 | 11. SWiM | `swim_aggregate()` + `extract_sae_activations_swim()` — SWiM functions |
| 45 | 11. SWiM | `iter_trajectory_activations_swim()` + `extract_trajectory_swim()` — trajectory generators |
| 46 | 11. SWiM | Performance comparison: Raw vs SWiM (timing + memory benchmark) |
| 47 | 11. SWiM | Top-K SAE feature tracking across turns (heatmap + feature dynamics summary) |

---

## Version History

### V-0.7 (2026-02-24) — SWiM-Aggregated SAE Extraction
- **Added** `swim_aggregate()` — SWiM smoothing via `avg_pool1d(M=16, stride=1)` + max/mean pool → `(d_sae,)` turn-level summary
- **Added** `extract_sae_activations_swim()` — wraps existing NNSight trace + SAE encode with SWiM on top
- **Added** `iter_trajectory_activations_swim()` — memory-safe generator yielding compact `(d_sae,)` per turn (~260KB vs ~200MB+)
- **Added** `extract_trajectory_swim()` — accumulates all turns (safe due to compact output)
- **Added** Performance benchmark cell: Raw vs SWiM timing + memory comparison table
- **Added** Top-K SAE feature tracking visualization: heatmap of top-20 features per layer across turns, feature dynamics summary (persistent/emerged/disappeared/high-score-only), F_H/F_R candidate flagging
- **Added** `SWIM_WINDOW_SIZE=16`, `SWIM_POOL_MODE="max"` hyperparameters
- **Implements** Level 1 of the CC++-inspired two-level temporal smoothing pipeline (within-turn SWiM)

### V-0.6 (2026-02-23) — Enhanced Trajectory Data & Criteria Caching
- **Added** criteria cache (`criteria_cache.json`) — reuses generated rubrics across runs, auto-seeded from loaded trajectories
- **Changed** `check_refusal()` to return full dict `{value, rationale, metadata, refused}` instead of bool
- **Changed** `evaluate_with_rubric()` to return full dict `{rationale, Score, metadata}` instead of int
- **Added** `refusal_judge` field to every turn record (backtracked and non-backtracked)
- **Added** `score_judge` field to every turn record (null for backtracked turns)
- **Updated** `CHECK_REFUSAL_SYSTEM_PROMPT` and `evaluate_with_rubric()` prompt to explicitly define `metadata` as confidence score (0-100)
- **Updated** trajectory viewer to display scoring criteria, refusal judge, and score judge details per turn
- **Added** HTML escaping in trajectory viewer for safe content display

### V-0.5 (2026-02-23) — Section Reorganization
- Reorganized function locations across notebook sections

### V-0.4 (2026-02-23) — On-the-Fly SAE Extraction
- **Removed** pre-saved `.pt` activation files from pipeline
- **Removed** `extract_activations` parameter from pipeline function
- **Removed** `EXTRACT_ACTIVATIONS` flag and all activation save/load logic
- **Added** `iter_trajectory_activations()` generator — memory-safe on-the-fly extraction (one turn at a time)
- **Added** `extract_activations_for_trajectory()` convenience wrapper (accumulates all, use with caution)
- **Renamed** `run_crescendomation_with_sae()` → `run_crescendomation()` (text-only)
- **Updated** Section 11 analysis cells to use generator pattern with `del acts` + `torch.cuda.empty_cache()`
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
6. **OOM from on-the-fly extraction**: `extract_activations_for_trajectory()` accumulated ALL turns in memory. Fixed with `iter_trajectory_activations()` generator that yields one turn at a time.
7. **Undefined `metadata` field in refusal prompt**: Original Crescendomation `CHECK_REFUSAL_SYSTEM_PROMPT` never defined what `metadata` means — it only appeared in examples (99, 80) with no explanation, leaving the judge LLM to guess. Fixed by adding explicit output format section: `"metadata": Your confidence score (0-100). 100 = completely certain, 0 = completely uncertain.` This matters because the pipeline uses `metadata > 90` as the refusal threshold.

### Architecture Notes
- **Score normalization**: `our_score = 11 - crescendomation_score` (their 1=jailbreak, 10=refusal -> ours 10=jailbreak, 1=refusal)
- **Activation shape**: `(seq_len, d_sae)` where `d_sae=65536` — sequence length grows each turn since the full conversation is re-tokenized
- **Hidden state casting**: Model runs in bfloat16, SAEs expect float32 — cast with `.float()` before encoding
- **Activations moved to CPU**: `.cpu()` after SAE encoding to keep GPU memory free
- **On-the-fly extraction (V-0.4)**: No `.pt` files saved; activations recomputed from JSON at analysis time
- **Criteria cache (V-0.6)**: Disk-backed JSON cache keyed by goal; auto-seeded from loaded trajectories

---

## What's Next (Planned)

### Immediate
- [x] Run full experiment with 100 test cases from JBB-Behaviors (done V-0.3)
- [x] Add JBB category/behavior tracking and per-category ASR analysis
- [x] Build trajectory viewer tool with category filtering
- [x] Implement on-the-fly SAE extraction (done V-0.4, CC++ inspired)
- [x] Add criteria caching for reuse across runs (done V-0.6)
- [x] Save full refusal judge dict in trajectory (done V-0.6)
- [ ] Add benign conversation controls (WildChat / XSTest) for negative samples

### Phase 2 — Feature Discovery
- [ ] Run on-the-fly extraction across full 100-trajectory dataset (per-turn extraction across all turns) — build feature matrices for Phase 2
- [ ] Run Lasso logistic regression on early-layer (F_H) and late-layer (F_R) activations
- [ ] Identify causally relevant features that discriminate jailbroken vs. refused trajectories
- [ ] Validate with GPT-4o feature interpretation + Neuronpedia dashboards

### Phase 3 — MLP Detector (CC++ Enhanced)
- [x] Implement SWiM (M=16) within-turn token aggregation on selected SAE features (done V-0.7)
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
  cross_layer_causal_sae_jailbreak_detection_V-0.7.ipynb  # Active notebook
  cross_layer_causal_sae_jailbreak_detection_V-0.6.ipynb  # Previous (criteria caching)
  cross_layer_causal_sae_jailbreak_detection_V-0.5.ipynb  # Previous (section reorg)
  cross_layer_causal_sae_jailbreak_detection_V-0.4.ipynb  # Previous (on-the-fly extraction)
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
    trajectory_viewer.ipynb                                # Interactive trajectory browser (criteria + refusal judge)
  results/
    crescendo_trajectories/
      trajectories_YYYYMMDD_HHMMSS.json                    # Trajectory metadata (text-only, includes criteria + refusal_judge)
      criteria_cache.json                                  # Cached scoring rubrics keyed by goal
      score_trajectories_{timestamp}.png                   # Score trajectory plots
      category_analysis_{timestamp}.png                    # Category ASR plots
```

---

*Last updated: 2026-02-24*
