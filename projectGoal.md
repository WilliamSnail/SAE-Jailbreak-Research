🔥 High‑Level Strategy
Jailbreak detection & repair boils down to three steps:
    1. Detect when the model is entering a jailbreak‑like internal state
    2. Localize the features/circuits responsible
    3. Repair the behavior by clamping, steering, or ablating the relevant latents
Gemma Scope 2 gives you tools for all three.

🧭 Phase 1 — Detect Jailbreaks Using SAEs
Use residual‑stream SAEs and MLP-output SAEs to monitor latent activations during:
    • Safe prompts
    • Jailbreak attempts
    • Successful jailbreaks
    • Failed jailbreaks
🎯 What you’re looking for
    • Latents that fire only during jailbreak attempts
    • Latents whose firing correlates with harmful intent
    • Latents that activate early in the prompt (layer 0–10) → “setup” features
    • Latents that activate late (layer 20+) → “execution” features
🧪 Concrete method
    1. Run the model on a dataset of jailbreak prompts (e.g., DAN, reverse psychology, roleplay jailbreaks).
    2. Record SAE activations at: 
        ○ attention outputs
        ○ MLP outputs
        ○ post‑MLP residual stream
    3. Compute: 
        ○ latent firing frequency differences
        ○ KL divergence between safe vs. jailbreak latent distributions
        ○ cosine similarity of latent activation vectors
🚨 Output of Phase 1
A shortlist of candidate jailbreak‑related latents.


🧭 Phase 2 — Repair the Model
Once you know the harmful latents/circuits, you can intervene.
Gemma Scope 2 supports three main repair strategies:

🛠️ Repair Strategy A — Latent Clamping
Clamp harmful latents to zero (or a safe baseline) during inference.
    • Use residual‑stream SAEs for broad control
    • Use MLP-output SAEs for precise control
    • Use CLTs to ensure downstream effects are neutralized
This is the most direct “patch”.

🛠️ Repair Strategy B — Steering with SAE Latents
Instead of clamping, add or subtract latent activations:
    • Add “harmlessness” latents
    • Subtract “harmful intent” latents
    • Use end‑to‑end SAEs (E2E) for more faithful steering
This is similar to activation addition, but feature-level and more interpretable.


🧪 Phase 3 — Evaluate Repair Robustness
Evaluate on:
    • Known jailbreaks
    • Novel jailbreaks
    • Adversarially optimized jailbreaks
    • Harmless prompts (to check for over-suppression)
Metrics:
    • Success rate of jailbreaks
    • KL divergence from baseline model
    • Perplexity on safe tasks
    • Latent activation drift

🧱 Putting It All Together — A Minimal Working Pipeline
Here’s a clean, actionable pipeline you can implement:
    1. Collect jailbreak prompts
    2. Run model with SAEs attached
    3. Identify anomalous latents
    4. Trace circuits using CLTs
    5. Select intervention points
    6. Clamp or steer latents
    7. Re-run jailbreak prompts
    8. Measure repair effectiveness
