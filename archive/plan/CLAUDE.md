# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project using Sparse Autoencoders (SAEs) to detect and repair jailbreaks in LLMs. The approach has three phases:

1. **Detect** - Monitor SAE latent activations to identify features that fire during jailbreak attempts
2. **Repair** - Intervene via latent clamping, steering, or ablation of harmful latents
3. **Evaluate** - Test robustness against known/novel/adversarial jailbreaks and check for over-suppression

Primary model target: Gemma-3-1B-IT with Gemma Scope 2 SAEs across layers 7, 13, 17, 22 at widths 16K–1M.

## Setup

Requires CUDA 13.0 for PyTorch. Environment variables needed in `.env`:
- `NDIF_API_KEY` - NNSight remote execution
- `HF_TOKEN` - HuggingFace access
- `OPENAI_API_KEY` - OpenAI API

## Running Experiments

All experimentation is notebook-based:
```bash
jupyter notebook testing/SAE_lens_test.ipynb
jupyter notebook testing/SAE_Test.ipynb
```

## Architecture & Key Libraries

| Role | Library |
|------|---------|
| Model loading & intervention | NNSight (supports local + NDIF remote execution) |
| SAE encoding/analysis | SAE-Lens 6.30.0 |
| Mechanistic interpretability | Transformer-Lens 2.11.0 |
| SAE visualization | SAE-Vis, Neuronpedia dashboards |
| Jailbreak dataset | JailbreakBench |
| Evaluation | Inspect AI |

## Research Pipeline

1. Load jailbreak prompts from JailbreakBench dataset
2. Run Gemma model with SAEs attached via NNSight
3. Record activations at attention outputs, MLP outputs, and post-MLP residual stream
4. Compute latent firing frequency differences, KL divergence, and cosine similarity between safe/jailbreak distributions
5. Identify discriminative latents (early layers = "setup" features, late layers = "execution" features)
6. Apply intervention (clamp, steer, or ablate)
7. Re-evaluate with repair metrics: jailbreak success rate, KL divergence from baseline, perplexity on safe tasks, latent activation drift
