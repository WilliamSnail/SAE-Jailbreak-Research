# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Master's thesis research project on detecting and mitigating **multi-turn jailbreaks** in LLMs using **Sparse Autoencoders (SAEs)**. The core thesis is that jailbreaking is not a dense-space drift (the "Assistant Axis" framing) but a **sparse, mechanistic state transition** — specifically, the causal decoupling of upstream Harm Recognition features (F_H) from downstream Refusal Execution features (F_R) across conversation turns.

The primary model is **Gemma-3-4B-IT** with **GemmaScope 2 SAEs** (JumpReLU architecture). All experimentation is done in Jupyter notebooks.

## Current Status: Phase 1 Complete + SWiM Extraction (V-0.7)

**Active notebook:** `cross_layer_causal_sae_jailbreak_detection_V-0.7.ipynb`

### What's Built
- **Full Crescendomation red-teaming pipeline**: GPT-4o attacker, Gemma-3-4B-IT target (NNSight), GPT-4o judge
- **On-the-fly SAE activation extraction** (V-0.4): No `.pt` files saved; activations recomputed from JSON at analysis time via `iter_trajectory_activations()` generator
- **SWiM-aggregated SAE extraction** (V-0.7): Sliding Window Mean (M=16) smoothing + max-pool converts sparse per-token activations → compact `(d_sae,)` turn-level summaries. Level 1 of CC++ two-level temporal smoothing.
- **Top-K feature tracking** (V-0.7): Heatmap of top-20 SAE latent indices per layer across turns; feature dynamics summary (persistent/emerged/disappeared/high-score-only); F_H/F_R candidate flagging
- **Criteria caching** (V-0.6): Disk-backed JSON cache for scoring rubrics, auto-seeded from loaded trajectories
- **Full judge data** (V-0.6): Both `refusal_judge` and `score_judge` dicts saved per turn (rationale + confidence)
- **Category/behavior analysis**: Per-category ASR tables, visualization plots
- **Trajectory viewer** (`test/trajectory_viewer.ipynb`): Interactive browser with category filtering, criteria display, judge details
- **100-test-case run completed** (V-0.3): Full JBB-Behaviors dataset

### What's Next
- **Immediate**: Add benign conversation controls (WildChat/XSTest)
- **Phase 2**: Lasso feature selection on SAE activations across full trajectory dataset
- **Phase 3**: MLP detector with EMA across-turn smoothing (Level 2), softmax-weighted BCE loss
- **Phase 4**: Conditional sparse clamping intervention

See [progress.md](progress.md) for detailed version history and [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for full thesis plan.

## Environment Setup

Requires CUDA 13.0 (`cu130`) for PyTorch. Conda environment: `MI` (via miniforge3).

```bash
pip install -r requirements_3.txt
```

Required `.env` file keys:
- `HF_TOKEN` — HuggingFace access token (for Gemma, LlamaGuard)
- `NDIF_API_KEY` — NNSight remote execution (only needed if `REMOTE = True`)
- `OPENAI_API_KEY` — GPT-4o for attacker/judge LLM and feature interpretation

The `REMOTE` flag in each notebook switches between local GPU and NDIF remote inference.

## Key Libraries & Roles

| Library | Role |
|---|---|
| `nnsight >= 0.3.5` | Model loading + activation interception via `.trace()` / `.generate()` context managers |
| `sae-lens == 6.30.0` | Load pretrained GemmaScope 2 SAEs; `SAE.from_pretrained()` |
| `transformer-lens < 3.0.0` | Mechanistic interpretability utilities |
| `datasets` | Load JailbreakBench (`JailbreakBench/JBB-Behaviors`, `judge_comparison` config, `test` split) |
| `transformers` | LlamaGuard-3-8B for safety classification |
| `inspect_ai` | Evaluation framework |

### NNSight Reference

**When working with NNSight, always consult [nnsight_reference.md](nnsight_reference.md)** for:
- Model architecture access patterns (GPT-J vs GPT-2 vs Llama vs Gemma module paths)
- Core tracing/intervention patterns and `.save()` semantics
- Multi-invoke patterns for extracting and intervening in the same trace
- Generation with interventions
- Common dimension reference table
- **Critical gotchas** (especially for remote execution): device mismatch, containers must be inside trace, `.save()` on containers, proxy limitations

## Notebook Architecture (V-0.7)

| Cell Range | Section | Description |
|---|---|---|
| 0-1 | Header | Architecture overview, CC++ references, next steps |
| 3-5 | 1. Setup | Environment detection, imports, API config |
| 7 | 2. Config | Model, SAE, Crescendomation parameters, criteria cache path |
| 9 | 3. Load Model | Gemma-3-4B-IT via NNSight with SDPA attention |
| 11 | 4. Attacker/Judge | OpenAI API client + `attacker_generate()` |
| 13-16 | 5. Functions | Rubric generation, evaluation, refusal check, Crescendo step |
| 18 | 6. Target Gen | `target_generate()` via NNSight |
| 20-21 | 7. Load JBB | Load JailbreakBench + convert to test cases |
| 23-25 | 8. Pipeline | `run_crescendomation()` + runner loop |
| 27-29 | 9. Save/Load | Trajectory JSON save, auto-load, criteria cache seeding |
| 31-35 | 10. Results | Summary tables, category analysis, plots, turn-by-turn logs |
| 37-42 | 11. SAE | Load SAEs, extraction functions, on-the-fly analysis |
| 43-47 | 11. SWiM | SWiM aggregation functions, trajectory generators, benchmark, top-K feature tracking |

## SAE Loading Pattern

```python
from sae_lens import SAE

# Gemma-3-4B-IT SAEs (d_model=2560)
sae = SAE.from_pretrained(
    release="gemma-scope-2-4b-it-res",
    sae_id="layer_17_width_65k_l0_medium",
    device=device,
)
```

SAE hook point: `model.model.language_model.layers[layer].output[0]` (residual stream post-layer — note the `language_model` wrapper for Gemma3ForConditionalGeneration VLM). Hidden states must be cast to `float32` before SAE encoding since the model runs in `bfloat16`.

## NNSight Activation Extraction Pattern

```python
with model.trace(prompt, remote=REMOTE):
    hidden_states[layer] = model.model.language_model.layers[layer].output[0].save()

# Then encode outside the trace context
h_float = hidden_states[layer].squeeze(0).float().to(device)
latents = sae.encode(h_float)  # (seq_len, d_sae)
```

For generation with chat-formatted prompts, use `model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` before tracing.

## Dataset Pattern

JailbreakBench loading:
```python
from datasets import load_dataset
jbb_data = load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison")
jbb_data = list(jbb_data['test'])  # 300 examples, use 'test' split (not 'train')
jailbreak_prompts = [item['prompt'] for item in jbb_data]
```

Control/safe prompts are manually constructed benign questions. The `safe_prompts` list in the notebooks has 30 hardcoded entries.

---

## Research Plan Summary

### Thesis Core Hypothesis

A jailbreak is the **causal decoupling** of:
- **F_H** (Harm features): early/middle layers (9, 17 for 4B) — the model detecting harmful intent
- **F_R** (Refusal features): late layers (22, 29 for 4B) — the refusal execution mechanism

A successful jailbreak = F_H remains high AND F_R drops. The detector learns this non-linear XOR-like condition with an MLP rather than a linear probe.

### Positioning Against Prior Work

| Dimension | The Assistant Axis | CC++ (Anthropic, 2026) | This Thesis |
|---|---|---|---|
| **Representation** | Dense activations | Linear probes on raw activations | Sparse SAE latents across two circuit layers |
| **Temporal modeling** | None (per-input) | SWiM/EMA on token-level (single timescale) | **Two-level**: SWiM within-turn + EMA across-turn |
| **Detection** | Linear probe | Linear classifier ensemble | Non-linear MLP on smoothed SAE features |
| **Intervention** | Dense activation cap (always-on) | Block before generation | Conditional Sparse Clamping (triggered only on circuit failure) |
| **Model** | Various | Claude-family (proprietary) | Gemma-3-IT (open-weight, reproducible) |

### Phase Overview

| Phase | Status | Description |
|---|---|---|
| **1. Trajectory Generation** | **Complete (V-0.6)** | Crescendomation pipeline, 100-case run, criteria caching, full judge data |
| **2. Feature Discovery** | Not Started | Lasso logistic regression on SAE activations → F_H, F_R feature sets |
| **3. MLP Detector** | Not Started | Two-level smoothing (SWiM+EMA), softmax-weighted BCE, early warning latency |
| **4. Intervention** | Not Started | Conditional sparse clamping of F_R when MLP detects decoupling |

### Experimental Results (100 test cases, V-0.3)

Full run completed on 2026-02-21. Category-level ASR analysis available. Speed: ~25.4 tok/s (NNSight), bottleneck is target generation (~19.7s/turn).

### Preliminary Discriminative Latents (Gemma-3-1B-IT, 30 samples)

| Layer | Top Latent | KL Div | Role |
|---|---|---|---|
| 7 | 2933 | 7.15 | F_H (early harm) |
| 13 | 174 | 6.25 | F_H (semantic harm) |
| 22 | 1576 | 6.38 | F_R (refusal execution) |

> Correlational only — Lasso selection (Phase 2) will confirm causality.

---

## Files & Directory Structure

```
SAE-Jailbreak-Research/
  cross_layer_causal_sae_jailbreak_detection_V-0.7.ipynb  # Active notebook
  requirements_3.txt
  .env                                                     # API keys (HF_TOKEN, OPENAI_API_KEY, NDIF_API_KEY)
  .gitignore
  reference/
    CLAUDE.md                                              # This file — Claude Code project context
    RESEARCH_PLAN.md                                       # Full thesis research plan (CC++ updated)
    progress.md                                            # Detailed version history & progress log
    nnsight_reference.md                                   # NNSight library quick reference & gotchas
  test/
    trajectory_viewer.ipynb                                # Interactive trajectory browser
  results/
    crescendo_trajectories/
      trajectories_YYYYMMDD_HHMMSS.json                    # Trajectory metadata (text-only JSON)
      criteria_cache.json                                  # Cached scoring rubrics
    images/                                                # Plot output (category_analysis, score_trajectories)
  archive/                                                 # Previous notebook versions and old plans
```

## Known Issues & Conventions

- `SAE.from_pretrained()` now returns only the SAE object (not a tuple). The `sae, cfg_dict, sparsity = SAE.from_pretrained(...)` pattern is deprecated.
- Always cast hidden states to `float32` before SAE encoding: `h.float().to(device)`.
- **Layer access path for Gemma-3**: `model.model.language_model.layers[layer].output[0]` — the `language_model` wrapper exists because Gemma3ForConditionalGeneration is a VLM wrapper.
- JailbreakBench uses `'test'` split and `'judge_comparison'` config — there is no `'train'` split.
- For chat-formatted generation, set `tokenizer.padding_side = "left"` for batched inference.
- Score normalization: `our_score = 11 - crescendomation_score` (their 1=jailbreak, 10=refusal → ours 10=jailbreak, 1=refusal).
- On-the-fly extraction (V-0.4+): No `.pt` files — activations recomputed from JSON at analysis time.
- Criteria cache (V-0.6): Auto-seeded from loaded trajectories; disk-backed JSON keyed by goal.
- **Output paths**: Trajectory JSON → `results/crescendo_trajectories/`, plot PNGs → `results/images/`.
- NNSight `model.generate(prompt, ...)` context (without `.trace()`) is used for text generation; `.trace()` is used for activation interception only.

## Pending Modifications (V-0.6 Notebook)

The following changes need to be applied to separate image output paths from trajectory paths:
- **Cell 7 (config)**: Add `IMAGE_DIR = Path("results/images")` + `IMAGE_DIR.mkdir(...)` after OUTPUT_DIR lines; add print line
- **Cell 33**: Change `OUTPUT_DIR` → `IMAGE_DIR` in `plt.savefig()` and `print()` for `category_analysis_{timestamp}.png`
- **Cell 34**: Change `OUTPUT_DIR` → `IMAGE_DIR` in `plt.savefig()` and `print()` for `score_trajectories_{timestamp}.png`
