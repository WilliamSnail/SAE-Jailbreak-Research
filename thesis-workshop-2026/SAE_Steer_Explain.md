# SAE-Based Steering: How Intervention Works

## 1. What are SAE Features?

The SAE (Sparse Autoencoder) decomposes a layer's residual stream (2048-dim vector) into ~65K sparse features. Each feature represents a **learned concept** — some correlate with harmful content, others with refusal, safety, etc.

```
residual stream h (2048-dim) → SAE encode → sparse activations (65K-dim, mostly zeros)
```

The SAE also has a **decoder** `W_dec` of shape `(65K, 2048)`. Each row `W_dec[i]` is the **direction in residual stream space** that feature `i` represents.

## 2. What are Causal Drivers?

Phase 4 used **gradient attribution on the MLP detector** to find which of the 435 selected features most strongly push D_t toward 1 (jailbreak). These are the **causal drivers** — stored in `handoff["intervention_targets"]["causal_driver_global_indices"]`.

Each `global_idx` encodes both the layer and the SAE feature index:

```python
layer_offset = global_idx // D_SAE      # which layer (0-3 for raw, 4-7 for delta)
sae_idx      = global_idx % D_SAE       # which of the 65K features in that layer's SAE
layer        = LAYERS[layer_offset % 4]  # → actual layer number: 9, 17, 22, or 29
```

## 3. What is the Benign Baseline?

`handoff["benign_baselines"]["mean_activation_per_driver"]` stores the **average activation of each causal driver across benign (non-jailbreak) conversations**. This is the "normal" level for each feature.

## 4. The Subtract-Only Correction

For each causal driver, compare current vs baseline:

```python
current = current_features_by_layer[layer][sae_idx].item()  # this turn's SWiM activation
baseline = benign_baselines[i]                                # normal benign level

if current > baseline:                    # feature is ELEVATED (jailbreak signal)
    delta = (baseline - current) * alpha  # always negative (pushes down)
```

Example with real numbers:

```
Feature "deceptive_intent" at layer 22, sae_idx 4817:
  current  = 3.2   (elevated — jailbreak conversation)
  baseline = 0.5   (normal benign level)
  delta    = (0.5 - 3.2) * 1.0 = -2.7

  If current = 0.3 (below baseline):
    → skip! We never suppress below baseline. Only suppress what's elevated.
```

## 5. Converting Feature-Space Correction to Residual Stream Correction

`delta` is in **feature space** (a scalar for one SAE feature). The model doesn't operate in feature space — it operates in **residual stream space** (2048-dim). The SAE decoder bridges them:

```python
corrections[layer] += delta * saes_dict[layer].W_dec[sae_idx].to(device)
#                     ^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                     scalar   (2048-dim) direction this feature
#                     (-2.7)   represents in residual stream
```

This produces a 2048-dim vector that means: "move the residual stream in the direction that **reduces** this feature's activation by 2.7".

Multiple drivers at the same layer get **summed** into one correction vector:

```python
if layer not in corrections:
    corrections[layer] = t.zeros(D_MODEL, device=device)  # (2048,)
corrections[layer] += delta * W_dec[sae_idx]  # accumulate
```

## 6. Injection into the Model

The correction is added to the residual stream at the **last token position** during generation:

```python
with model.generate(...) as generator:
    with generator.invoke(prompt):
        for layer in sorted(corrections.keys()):  # forward-pass order
            model.model.language_model.layers[layer].output[0][:, -1] += corrections[layer]
```

`[:, -1]` = last token = where the model looks to predict the next token. By shifting the residual stream here, we steer the model's first generated token (and all subsequent ones, since they attend to this position).

---

## Summary

**We find which SAE features are abnormally elevated compared to benign conversations, then subtract their decoder directions from the residual stream to push the model's internal state back toward "normal" before it starts generating.**
