3.1 Research Design: Cross-Layer Causal State-Space Framework
We propose a closed-loop framework for Multi-Turn Jailbreak mitigation. Rather than treating jailbreaks as a generalized loss of the "Assistant" persona (a dense-space phenomenon), we hypothesize that jailbreaking is a sparse, mechanistic state transition. Specifically, it is the causal decoupling of upstream "Harm Recognition" from downstream "Refusal Execution" over multiple turns.  

Key Inovation
- State Projection vs. Causal Decoupling (The "How")
• The Assistant Axis: Defines drift simply as the model's dense activation projecting further away from the Assistant pole over time.
• Your Method: Defines drift as the breaking of a causal circuit. Drawing on Yeo et al., you track the relationship between Upstream Harm Features (F 
H ) and Downstream Refusal Features (F R​ ). My novelty is defining a successful jailbreak as the specific moment when F H​ remains high (the model knows the prompt is harmful) but F R  drops (the refusal mechanism is suppressed).
- Linear Probing vs. Non-Linear Interaction (The Architecture)
• The Assistant Axis: Relies on linear projections (Cosine Similarity or dot products) on a single axis.
My Method: Employs an MLP across layers. A linear boundary cannot capture complex conditional logic. By feeding early-layer FH and late-layer FR into an MLP, your detector learns the non-linear XOR-like condition of a jailbreak (IF Harm is Detected AND Refusal is Suppressed → Trigger Alarm). This directly addresses the limitations of linear boundaries noted in recent literature.
Conclusion on Novelty: "The Assistant Axis" proves that multi-turn drift exists and can be mapped in dense space. Your thesis advances this by explaining how it happens in sparse space (circuit breaking) and building a non-linear detector to catch the exact moment the causal chain snaps. This is easily enough novelty for a Master's thesis.

**The Workflow:
1. Trajectory Generation: Generate multi-turn attack traces using automated LLM-based red-teaming.
2. Latent State Decomposition: Use SAEs to extract sparse features at both early/middle layers (Harm) and late layers (Refusal).
3. Unsupervised Feature Discovery: Isolate specific SAE features that causally dictate harm recognition and refusal via statistical divergence.
4. Causal Drift Detection (MLP): Train a lightweight Multi-Layer Perceptron (MLP) on these cross-layer features to detect the non-linear breaking of the causal chain.
5. Dynamic Intervention: Implement Conditional Clamping that restores the specific Refusal features only when the MLP detects causal decoupling.  

3.2 Models and Tools
• Base Model: Gemma-2-2B-IT. Chosen for its state-of-the-art performance at a manageable parameter size and its integration with open-source SAEs.
• Sparse Autoencoders: GemmaScope JumpReLU SAEs.
    ◦ Layer Selection: We extract activations from an Early/Middle Layer (e.g., Layer 10-12) to capture semantic Harm recognition, and a Late Layer (e.g., Layer 15-20) to capture Refusal execution .  
    
3.3 Dataset Construction
To capture "Safety Drift", the dataset must track the temporal evolution of an attack.
• Multi-Turn Adversarial Dataset: We utilize 'Crescendo'developed by microsoft reserach and multi-turn roleplay trajectories to simulate gradual escalation. An Attacker LLM (e.g., Llama-3-70B or chat gpt) is prompted to elicit harmful instructions over 5-10 turns.
• Control Dataset: Benign multi-turn conversations (from WildChat) and XSTest (safe but sensitive prompts) to ensure the detector learns intent, not just keywords.
• Labeling: A Judge LLM scores the final output. The trajectory is labeled as "Jailbroken" if the final turn yields a harmful compliance .
3.4 SAE Feature Discovery (Automated Extraction)
To avoid the bias of manual feature selection, we discover F 
H
​
  (Harm) and F 
R
​
  (Refusal) features statistically.
• Extraction: Collect SAE activation vectors for early layers (Z 
early
​
 ) and late layers (Z 
late
​
 ) across successful jailbreaks vs. robust refusals.
• Selection via L1-Regularization: Train a Lasso logistic regression to separate safe vs. jailbroken trajectories . Features with high non-zero weights are selected. We designate the top early-layer features as F 
H
​
  and top late-layer features as F 
R
​
 .
• Interpretation: Use an LLM (GPT-4o) to generate natural language interpretations for these features (e.g., "mentions illegal acts" vs. "apologetic refusal") to verify their semantic alignment .
3.5 Causal Drift Detection: The Non-Linear MLP
Dense linear probes (such as the Assistant Axis) track general deviations. Our detector tracks the specific failure of the harm-refusal circuit.
• Input Representation: At turn t, we concatenate the sparse activations of the selected features across layers: ψ 
t
​
 =[A(F 
H
​
 ) 
t
​
 ⊕A(F 
R
​
 ) 
t
​
 ].
• Architecture: A 2- or 3-layer MLP. A linear boundary cannot effectively capture the conditional logic of a jailbreak (i.e., F 
H
​
  is active AND F 
R
​
  is suppressed). The MLP learns this non-linear interaction.
• Smoothing: We apply an Exponential Moving Average (EMA) to the MLP's output probability over the conversational window to prevent single-token noise from triggering false positives.
• Drift Metric: The output of the MLP represents the "Decoupling Probability" D 
t
​
 . A rising D 
t
​
  over N turns indicates an active jailbreak attempt.
3.6 Steering Intervention: Conditional Sparse Clamping
Standard steering clamps features permanently, degrading general reasoning (e.g., MMLU scores drop) . "The Assistant Axis" mitigates this via a dense activation cap. We propose Conditional Sparse Clamping, intervening only on specific concepts when a failure is imminent.
• Trigger Condition: If the smoothed MLP score D 
t
​
  exceeds a threshold τ (indicating the causal chain is breaking).
• Action: We apply the "Add-Only" constraint to the late-layer Refusal features (F 
R
​
 ).
    ◦ Δ=Target_Value−A(F 
R
​
 ) 
current
​
 
    ◦ We inject the decoded Δ into the residual stream only if the model is currently failing to activate F 
R
​
  naturally. If the model is already refusing (or if it's a benign prompt and D 
t
​
  is low), zero steering is applied, preserving model capabilities.
3.7 Evaluation Metrics
The framework is evaluated on the trade-off between robustness and utility.
• Safety (Robustness): Attack Success Rate (ASR) on multi-turn datasets. We hypothesize the MLP will flag drift turns before the final harmful output is generated (Early Warning Latency).
• Utility (Helpfulness): Benign Refusal Rate (BRR) on XSTest, and general capabilities on MMLU/GSM8K . We will baseline our conditional sparse clamping against standard static feature clamping  and dense projection capping to demonstrate superior preservation of model intelligence.