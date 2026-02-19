# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Master's thesis research project on detecting and mitigating **multi-turn jailbreaks** in LLMs using **Sparse Autoencoders (SAEs)**. The core thesis is that jailbreaking is not a dense-space drift (the "Assistant Axis" framing) but a **sparse, mechanistic state transition** — specifically, the causal decoupling of upstream Harm Recognition features (F_H) from downstream Refusal Execution features (F_R) across conversation turns.

The primary model is **Gemma-3 IT** (1B and 4B variants) with **GemmaScope 2 SAEs** (JumpReLU architecture). All experimentation is done in Jupyter notebooks.

## Environment Setup

Requires CUDA 13.0 (`cu130`) for PyTorch. Install dependencies:
```bash
pip install -r requirements_3.txt
```

Required `.env` file keys:
- `HF_TOKEN` — HuggingFace access token (for Gemma, LlamaGuard)
- `NDIF_API_KEY` — NNSight remote execution (only needed if `REMOTE = True`)
- `OPENAI_API_KEY` — GPT-4o for LLM-based feature interpretation

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

## Notebook Architecture

All active work is in [testing/](testing/):

- [testing/jailbreak_detection_pipeline.ipynb](testing/jailbreak_detection_pipeline.ipynb) — **Main Phase 1 pipeline** for Gemma-3-1B-IT: loads SAEs for layers [7, 13, 17, 22], extracts activations, computes firing frequency / KL divergence / mean activation difference, ranks candidate latents
- [testing/jailbreak_detection_3.ipynb](testing/jailbreak_detection_3.ipynb) — **Extended Phase 1** for Gemma-3-4B-IT on layers [9, 17, 22, 29]; includes LlamaGuard-3-8B judge for labeling (31% jailbreak rate on JBB-Behaviors)
- [testing/SAE_lens_test.ipynb](testing/SAE_lens_test.ipynb) — Exploratory SAE tooling reference (Gemma-3-1B-IT, SAE API patterns, Neuronpedia dashboard integration)
- [testing/SAE_Test.ipynb](testing/SAE_Test.ipynb) — Additional SAE experiments

Cached model outputs are stored in [testing/gemma-3-4b-it_harmful_prompt_responses.json](testing/gemma-3-4b-it_harmful_prompt_responses.json).

## SAE Loading Pattern

```python
from sae_lens import SAE

# Gemma-3-1B-IT SAEs (26 layers, d_model=1152)
sae = SAE.from_pretrained(
    release="gemma-scope-2-1b-it-res",
    sae_id="layer_17_width_65k_l0_medium",
    device=device,
)

# Gemma-3-4B-IT SAEs (d_model=2560)
sae = SAE.from_pretrained(
    release="gemma-scope-2-4b-it-res",
    sae_id="layer_17_width_65k_l0_medium",
    device=device,
)
```

SAE hook point: `model.model.layers[layer].output[0]` (residual stream post-layer). Hidden states must be cast to `float32` before SAE encoding since the model runs in `bfloat16`.

## NNSight Activation Extraction Pattern

```python
with model.trace(prompt, remote=REMOTE):
    hidden_states[layer] = model.model.layers[layer].output[0].save()

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

## Research Plan & Workflow

### Thesis Core Hypothesis

A jailbreak is the **causal decoupling** of:
- **F_H** (Harm features): early/middle layers (10–13) — the model detecting harmful intent
- **F_R** (Refusal features): late layers (17–22) — the refusal execution mechanism

A successful jailbreak = F_H remains high AND F_R drops. The detector learns this non-linear XOR-like condition with an MLP rather than a linear probe.

### Phase 1: Trajectory Generation & Dataset Construction (Status: Partially Done)

**Goal**: Build a labeled multi-turn dataset with `[jailbroken | refused]` trajectory labels.

**Datasets**:
- **Attack prompts**: JailbreakBench/JBB-Behaviors (300 prompts, 31% jailbreak rate on Gemma-3-4B-IT)
- **Multi-turn attack traces**: Crescendo (Microsoft Research) + roleplay escalation
- **Control**: WildChat (benign multi-turn) + XSTest (safe but sensitive)
- **Labeling**: LlamaGuard-3-8B judges the final-turn response

**Attacker LLM**: An external LLM (GPT or Llama-3-70B) generates the multi-turn escalation sequences over 5–10 turns.

**Next steps**:
- [ ] Integrate Crescendo multi-turn attack dataset
- [ ] Add WildChat benign conversations as negative samples
- [ ] Implement attacker LLM loop for automated trajectory generation
- [ ] Save labeled trajectory dataset with `(turn, prompt, response, is_jailbroken)` schema

### Phase 2: Latent State Decomposition (Status: Working)

**Goal**: Extract sparse SAE features at multiple layers per turn.

**Implemented** (in `jailbreak_detection_pipeline.ipynb`):
- Load SAEs for layers [7, 13, 17, 22] (1B model) or [9, 17, 22, 29] (4B model)
- Extract residual stream activations via NNSight
- Compute per-token SAE latent vectors `Z_early` and `Z_late`

**Configuration choices**:
- SAE width: 65k latents (balance between expressiveness and compute)
- L0 sparsity: medium
- Hook point: `resid_post` (post-layer residual stream)

### Phase 3: Unsupervised Feature Discovery via Lasso (Status: Not Started)

**Goal**: Statistically identify F_H (Harm) and F_R (Refusal) feature sets without manual selection.

**Method**:
1. Collect `Z_early` and `Z_late` vectors across jailbroken vs. refused trajectories
2. Train **Lasso logistic regression** to separate the two classes
3. Features with non-zero weights → candidate F_H (early layers) or F_R (late layers)
4. Use GPT-4o to generate natural language interpretations of top features (verify semantic alignment via Neuronpedia dashboards)

**Current proxy**: Composite score = normalized(freq_diff) + normalized(KL_divergence) + normalized(mean_diff). This is the foundation; Lasso selection is the next upgrade.

**Top discovered candidates** (from experimental runs):
- Layer 7: Latent 2933 (KL=7.15, highest discriminative score)
- Layer 13: Latent 174 (KL=6.25), 2332, 2579, 2830
- Layer 17: Latent 695, 1912, 2959, 2943
- Layer 22: Latent 1576 (KL=6.38, highest), 1695, 3167

### Phase 4: Causal Drift Detection — the Non-Linear MLP (Status: Not Started)

**Goal**: Train a lightweight MLP that detects the moment the harm-refusal circuit breaks.

**Input representation** at conversation turn `t`:
```
ψ_t = [A(F_H)_t ⊕ A(F_R)_t]
```
Concatenation of sparse activations of selected features across early and late layers.

**Architecture**: 2–3 layer MLP outputting a scalar `D_t` (Decoupling Probability).

**Smoothing**: Exponential Moving Average over the MLP output across turns to suppress single-token noise.

**Training signal**: Binary label — trajectory is "jailbroken" if final turn yields harmful compliance (from Judge LLM).

**Key design choice**: MLP vs. linear probe. A linear boundary cannot capture the XOR-like condition (F_H high AND F_R low). The MLP learns this non-linear interaction directly.

**Evaluation metric**: "Early Warning Latency" — how many turns before the final harmful output does the MLP trigger above threshold `τ`.

### Phase 5: Conditional Sparse Clamping Intervention (Status: Not Started)

**Goal**: Restore Refusal features only when causal decoupling is detected, without degrading general capabilities.

**Trigger condition**: Smoothed MLP score `D_t > τ`

**Action (Add-Only constraint)**:
```
Δ = Target_Value − A(F_R)_current
```
Inject decoded `Δ` into the residual stream only if F_R is currently suppressed.

**Key property**: Zero steering on benign prompts (when `D_t` is low) → preserves model capabilities.

**Baseline comparisons**:
1. Standard static feature clamping (always-on, degrades MMLU)
2. Dense projection capping ("The Assistant Axis" method)
3. This method: Conditional Sparse Clamping

### Evaluation Metrics

| Metric | Measures |
|---|---|
| Attack Success Rate (ASR) | Safety / robustness |
| Early Warning Latency | How early the MLP catches drift |
| Benign Refusal Rate (BRR) on XSTest | Over-refusal / false positives |
| MMLU / GSM8K | Capability preservation |
| KL divergence from baseline | Behavioral drift |

## Known Issues & Conventions

- `SAE.from_pretrained()` now returns only the SAE object (not a tuple). The `sae, cfg_dict, sparsity = SAE.from_pretrained(...)` pattern is deprecated — use `SAE.from_pretrained_with_cfg_and_sparsity()` if config/sparsity is needed.
- Always cast hidden states to `float32` before SAE encoding: `h.float().to(device)`. The model runs in `bfloat16` but SAEs expect `float32`.
- `model.model.layers[layer].output[0]` captures the residual stream (shape `(batch, seq_len, d_model)`); squeeze batch dim before encoding.
- JailbreakBench uses `'test'` split and `'judge_comparison'` config — there is no `'train'` split.
- For chat-formatted generation, set `tokenizer.padding_side = "left"` for batched inference.
- `LOAD_HARMBENCH` flag is not set by default in notebooks — HarmBench requires ~26GB VRAM for the 13B classifier.
- NNSight `model.generate(prompt, ...)` context (without `.trace()`) is used for text generation; `.trace()` is used for activation interception only.
