# Progress Log: Cross-Layer Causal SAE Jailbreak Detection

## Project Overview

Master's thesis research on detecting and mitigating **multi-turn jailbreaks** using **Sparse Autoencoders (SAEs)**. The core hypothesis is that jailbreaking is a **sparse, mechanistic state transition** — the causal decoupling of upstream Harm Recognition features (F_H) from downstream Refusal Execution features (F_R) — rather than a continuous drift in dense activation space.

---

## Current Status: Phase 1 — Pipeline Complete (V-0.2)

**Active notebook:** `cross_layer_causal_sae_jailbreak_detection_V-0.2.ipynb`

### What Has Been Built

#### 1. Full Crescendomation Red-Teaming Pipeline
- **Attacker LLM**: GPT-4o generates multi-turn Crescendo jailbreak prompts via OpenAI API
- **Target Model**: Gemma-3-4B-IT running locally on GPU (NVIDIA RTX PRO 5000) via NNSight
- **Judge LLM**: GPT-4o scores each turn on a 1-10 rubric (normalized: 0=refusal, 10=jailbreak)
- **Refusal Detection**: GPT-4o checks if the target refused, with backtracking support (up to 10 backtracks)
- **Scoring Rubric**: Auto-generated per-goal by GPT-4o using Crescendomation's rubric generation prompt
- **Test Cases**: Loaded from JailbreakBench/JBB-Behaviors (100 harmful behaviors, configurable subset)

#### 2. SAE Activation Extraction (Fused into Pipeline)
- **SAEs**: GemmaScope 2 JumpReLU SAEs (`gemma-scope-2-4b-it-res`, `width=65k`, `l0=medium`)
- **Layers**: [9, 17, 22, 29] — Early (F_H: 9, 17) and Late (F_R: 22, 29)
- **Extraction**: NNSight `.trace()` captures residual stream at each layer, encoded through SAEs
- **Layer access path**: `model.model.language_model.layers[layer]` (Gemma3ForConditionalGeneration is a VLM wrapper; text layers are under `.language_model`)
- **Data format**: Per-turn dict mapping `layer -> tensor(seq_len, 65536)`, saved as `.pt` files per-trajectory

#### 3. Memory Management (OOM Prevention)
- Activations saved to disk per-trajectory during the pipeline loop, then freed with `del`
- Analysis cells also free activations after processing each trajectory (`del act_data`)
- Each turn's activations are ~200 MB (4 layers x 65536-dim float32); sequence length grows each turn as full conversation is re-processed

#### 4. Crash Recovery
- Trajectory metadata (goals, prompts, responses, scores) saved as timestamped JSON files
- Load cell auto-detects the latest saved JSON file for resuming after kernel restart
- Activations stored separately as `.pt` files, referenced by path in the JSON

#### 5. SAE Activation Replay Cell (NEW)
- Decouples SAE extraction from the attack pipeline
- Reconstructs conversation history turn-by-turn from saved JSON trajectories
- Calls `extract_sae_activations_for_turn()` on each reconstructed conversation
- Handles backtracked turns correctly (extracts activations, then pops from history)
- Saves to same `.pt` format as pipeline — downstream analysis cells work unchanged
- Skips trajectories that already have activations on disk (configurable with `OVERWRITE_EXISTING`)

#### 6. Speed Optimizations
- **SDPA attention**: `attn_implementation="sdpa"` applied to model loading — mathematically identical to default attention, ~20-40% faster via fused CUDA kernels
- **Flash Attention 2**: Cannot install on Windows (path too long + missing CUDA_HOME/nvcc); SDPA is the alternative

#### 7. Preliminary Analysis & Visualization
- Summary table: per-trajectory label, max score, num turns, backtracks, score trajectory
- Score trajectory plots: per-conversation score evolution + max score distribution
- SAE activation statistics: mean magnitude, firing rate, max activation per layer per turn
- SAE activation evolution plot: mean |activation| vs judge score across turns per layer

---

### Speed Benchmarks (RTX PRO 5000, Gemma-3-4B-IT bfloat16)

| Method | Throughput | Notes |
|---|---|---|
| NNSight `.generate()` | ~25.4 tok/s | Used in pipeline |
| Vanilla HuggingFace `.generate()` | ~27.4 tok/s | Via `model._model` |
| NNSight overhead | +13.9% | Tracing layer for interpretability access |

**Pipeline bottleneck**: Target generation (~19.7s/turn) dominates. SAE extraction is only ~0.3s/turn. Attacker/judge API calls ~2.5s and ~1.0s respectively.

---

### Initial Experimental Results (5 test cases, 8 max rounds)

| Metric | Value |
|---|---|
| Total trajectories | 5 |
| Jailbroken | 3 (60.0%) |
| Refused | 2 (40.0%) |
| Avg max score | 7.40 |
| Avg turns | 7.8 |
| Avg backtracks | 1.4 |

---

## Notebook Cell Map (V-0.2)

| Cell | Section | Description |
|---|---|---|
| 0 | Header | Markdown: architecture overview, references |
| 3-5 | 1. Setup | Environment detection, imports, API config |
| 7 | 2. Config | Model, SAE, Crescendomation parameters |
| 9 | 3. Load Model | Gemma-3-4B-IT via NNSight with SDPA attention |
| 10 | 3. Load SAEs | GemmaScope 2 SAEs for layers [9, 17, 22, 29] |
| 11 | 3. Speed Test | NNSight vs HuggingFace throughput benchmark |
| 12 | 3. Speed Test 2 | Independent HF model load baseline (user-created) |
| 13 | 4. Attacker/Judge | OpenAI API client + `attacker_generate()` |
| 15-18 | 5. Utils | Rubric generation, evaluation, refusal check, Crescendo step |
| 20 | 6. Target Gen | `target_generate()` + `extract_sae_activations_for_turn()` |
| 22-23 | 7. Test Cases | Load JBB-Behaviors, convert to Crescendomation format |
| 25 | 8. Pipeline | `run_crescendomation_with_sae()` — main loop with timing |
| 27 | 8. Runner | Iterates test cases, passes `trajectory_idx` |
| 29 | 9. Save | Save trajectory JSON with activation file references |
| 30 | 9. Load | Auto-detect and reload latest saved trajectories |
| 31 | 9. Replay | SAE activation replay — extract from saved trajectories |
| 32-34 | 10. Results | Summary table, score trajectory plots, detailed turn logs |
| 36-37 | 11. Analysis | SAE activation statistics + evolution plots |

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

---

## What's Next (Planned)

### Immediate
- [ ] Run full experiment with more test cases (e.g., 50-100 from JBB-Behaviors)
- [ ] Test the replay cell workflow: run pipeline with `EXTRACT_ACTIVATIONS=False`, then replay
- [ ] Add benign conversation controls (WildChat / XSTest) for negative samples

### Phase 2 — Feature Discovery
- [ ] Run Lasso logistic regression on early-layer (F_H) and late-layer (F_R) activations
- [ ] Identify causally relevant features that discriminate jailbroken vs. refused trajectories
- [ ] Validate with GPT-4o feature interpretation + Neuronpedia dashboards

### Phase 3 — MLP Detector
- [ ] Train non-linear MLP on `psi_t = [A(F_H)_t || A(F_R)_t]` with soft labels from judge scores
- [ ] Evaluate "Early Warning Latency" — how many turns before jailbreak the MLP triggers

### Phase 4 — Conditional Sparse Clamping
- [ ] Implement intervention that restores F_R only when MLP detects causal decoupling
- [ ] Compare against static clamping and dense projection capping baselines
- [ ] Measure capability preservation (MMLU, GSM8K, benign refusal rate)

---

## Files & Directory Structure

```
SAE-Jailbreak-Research/
  cross_layer_causal_sae_jailbreak_detection_V-0.2.ipynb  # Active notebook
  cross_layer_causal_sae_jailbreak_detection.ipynb         # Original (V-0.1)
  RESEARCH_PLAN.md                                         # Full thesis research plan
  CLAUDE.md                                                # Claude Code project context
  progress.md                                              # This file
  .env                                                     # API keys (HF_TOKEN, OPENAI_API_KEY, NDIF_API_KEY)
  .gitignore                                               # Excludes results/
  results/
    crescendo_trajectories/
      trajectories_YYYYMMDD_HHMMSS.json                    # Trajectory metadata
      activations/
        trajectory_000_activations.pt                      # Per-trajectory SAE activations
        trajectory_001_activations.pt
        ...
```

---

*Last updated: 2026-02-21*
