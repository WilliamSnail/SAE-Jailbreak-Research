# NNSight Quick Reference

> Compiled from [NNSight walkthrough](https://nnsight.net/notebooks/tutorials/get_started/walkthrough/) + ARENA [1.4.2] practice notebook.

---

## 1. Setup & Model Loading

```python
from nnsight import CONFIG, LanguageModel
import torch as t

# Load model (HuggingFace-style)
model = LanguageModel("EleutherAI/gpt-j-6b", device_map="auto", torch_dtype=t.bfloat16)
tokenizer = model.tokenizer

# Remote execution (NDIF)
from dotenv import load_dotenv
load_dotenv()
CONFIG.set_default_api_key(os.getenv("NDIF_API_KEY"))
REMOTE = True
```

---

## 2. Model Architecture Access Patterns

### GPT-J-6B Dimensions
| Param | Attr | Value |
|---|---|---|
| N_HEADS | `model.config.n_head` | 16 |
| N_LAYERS | `model.config.n_layer` | 28 |
| D_MODEL | `model.config.n_embd` | 4096 |
| D_HEAD | D_MODEL // N_HEADS | 256 |

### Architecture-Specific Module Paths

| Model Family | Layers | Attention | MLP | LM Head | Residual Output |
|---|---|---|---|---|---|
| **GPT-J** | `model.transformer.h[L]` | `.attn` | `.mlp` | `model.lm_head` | `.output[0]` |
| **GPT-2** | `model.transformer.h[L]` | `.attn` | `.mlp` | `model.lm_head` | `.output[0]` |
| **Llama** | `model.model.layers[L]` | `.self_attn` | `.mlp` | `model.lm_head` | `.output[0]` |

### Key Sub-modules (GPT-J)
- Attention dropout (for raw attention patterns): `model.transformer.h[L].attn.attn_dropout`
- Q/K/V projections: `model.transformer.h[L].attn.q_proj`, `.k_proj`, `.v_proj`
- Output projection: `model.transformer.h[L].attn.out_proj`

---

## 3. Core Tracing Pattern

```python
with model.trace(prompt, remote=REMOTE):
    # Access hidden states at layer L, last token
    # .output[0] shape: (batch, seq, d_model)
    hidden = model.transformer.h[L].output[0][:, -1].save()

    # Access logits
    # lm_head.output shape: (batch, seq, d_vocab)
    logits = model.lm_head.output[:, -1].save()

# After context: hidden and logits are real tensors
```

### What `.save()` Does
- Values inside the trace context are **proxies** (lazy references), NOT real tensors
- `.save()` marks a proxy to persist its value after the trace completes
- **Without `.save()`, values are discarded** when the context exits

---

## 4. Multi-Invoke (Multiple Forward Passes)

Use `tracer.invoke()` to run multiple batches in a single `trace()` context. This enables:
- Extracting activations from one run and using them in another
- Comparing clean vs. intervened outputs efficiently

```python
with model.trace(remote=REMOTE) as tracer:
    # Pass 1: extract h-vector from clean prompts
    with tracer.invoke(clean_prompts):
        h = model.transformer.h[layer].output[0][:, -1].mean(dim=0)

    # Pass 2: baseline (no intervention)
    with tracer.invoke(zero_shot_prompts):
        baseline = model.lm_head.output[:, -1].argmax(dim=-1).save()

    # Pass 3: intervene by adding h
    with tracer.invoke(zero_shot_prompts):
        model.transformer.h[layer].output[0][:, -1] += h
        intervened = model.lm_head.output[:, -1].argmax(dim=-1).save()
```

---

## 5. Generation (Multi-Token)

```python
with model.generate(remote=REMOTE, max_new_tokens=N, **sampling_kwargs) as generator:
    with generator.invoke(prompt):
        # Intervene at each generated token
        model.transformer.h[layer].output[0][:, -1] += fn_vector
        tokens = model.generator.output.save()

completion = tokenizer.decode(tokens.squeeze().tolist())
```

---

## 6. Interventions Cheat Sheet

| Operation | Code Pattern |
|---|---|
| **Read activation** | `val = module.output[0].save()` |
| **Read input** | `val = module.input.save()` |
| **Add to residual stream** | `module.output[0][:, -1] += vector` |
| **Replace activation** | `module.output[0][:, pos] = new_value` |
| **Zero-ablate** | `module.output[0][:, pos] = 0.0` |
| **Read attention patterns** | `attn = model.transformer.h[L].attn.attn_dropout.input.save()` |

### Attention head output (via out_proj)
```python
# out_proj.input shape: (batch, seq, d_model) where d_model = n_heads * d_head
z = model.transformer.h[L].attn.out_proj.input[:, -1]  # (batch, d_model)
z_per_head = z.reshape(batch, N_HEADS, D_HEAD)          # (batch, heads, d_head)
z_per_head[:, head_idx] = replacement                    # patch specific head
```

---

## 7. Gotchas & Lessons Learned

### From Tutorial
1. **Module access order matters**: Must access modules in forward-pass execution order. Accessing layer 5 then layer 0 raises `OutOfOrderError`.
2. **`.save()` is mandatory**: Forgetting it = losing the value after trace exits.
3. **Proxies aren't tensors**: Can't use normal Python control flow (if/else) on proxy values inside trace context.
4. **Code after unbounded `.iter[:]` never executes**: Use separate invokers for post-generation logic.

### From Practice (Your Notes)
5. **Device mismatch on remote**: When using `remote=True`, tensors live on the remote server's `'cuda'`. Need explicit `.to('cuda')` when adding vectors computed outside the trace.
   ```python
   hidden_states[:, -1] += h.to("cuda")  # needed for remote execution
   ```
6. **Containers must be created inside trace (remote)**: When using `remote=True`, lists/dicts that accumulate proxy values **must** be created **inside** the trace context. On local execution this may work either way, but remote tracing serializes the computation graph — external containers don't get captured.
   ```python
   # WRONG (breaks on remote): container created outside trace
   fn_vector_list = []
   with model.trace(..., remote=True) as tracer:
       fn_vector_list.append(some_proxy)
       fn_vector_list.save()  # won't work — list wasn't part of the trace graph

   # RIGHT: container created inside trace
   with model.trace(..., remote=True) as tracer:
       fn_vector_list = []
       fn_vector_list.append(some_proxy)
       fn_vector_list.save()  # save the container, not individual items
   ```
   > **Note**: Gotchas 5, 6, and 7 are all specifically about **remote execution** (`remote=True`). Local execution is more forgiving because tensors stay in the same process.
7. **`.save()` on containers (remote)**: When using remote, save the list/dict itself, not individual tensor proxies within it. The ARENA solution docs may be outdated on this.
8. **`t.sum()` on list of proxies**: Use `sum([v for v in list])` (Python builtin) instead of `t.sum()` which doesn't accept lists.
9. **Test whitelist on NDIF**: Some test functions aren't whitelisted for remote execution. If a test fails on remote, try `REMOTE=False`.

---

## 8. Common Dimension Reference

| Expression | Shape | Description |
|---|---|---|
| `h[L].output[0]` | `(batch, seq, d_model)` | Residual stream at layer L |
| `h[L].output[0][:, -1]` | `(batch, d_model)` | Last-token residual |
| `lm_head.output` | `(batch, seq, d_vocab)` | Logits |
| `lm_head.output[:, -1]` | `(batch, d_vocab)` | Last-token logits |
| `attn.attn_dropout.input` | `(batch, n_heads, seq, seq)` | Attention patterns |
| `attn.out_proj.input[:, -1]` | `(batch, d_model)` | Concatenated head outputs |
| `out_proj.input.reshape(B, H, D)` | `(batch, n_heads, d_head)` | Per-head outputs |
| `attn.q_proj.output` | `(batch, seq, d_model)` | Query projection (all heads concat) |

---

## 9. Workflow Templates

### Extract h-vector (task-encoding direction)
```python
def calculate_h(model, dataset, layer=-1):
    with model.trace(dataset.prompts, remote=REMOTE):
        h = model.transformer.h[layer].output[0][:, -1].mean(dim=0).save()
        logits = model.lm_head.output[:, -1]
        token_ids = logits.argmax(-1).save()
    completions = tokenizer.batch_decode(token_ids)
    return completions, h
```

### Calculate function vector from specific heads
```python
def calculate_fn_vector(model, dataset, head_list):
    head_dict = defaultdict(set)
    for layer, head in head_list:
        head_dict[layer].add(head)

    with model.trace(dataset.prompts, remote=REMOTE):
        fn_vector_list = []
        for layer, heads in head_dict.items():
            out_proj = model.transformer.h[layer].attn.out_proj
            hidden = out_proj.input[:, -1].mean(dim=0)
            # Zero-ablate unwanted heads
            for h in set(range(N_HEADS)) - heads:
                hidden.reshape(N_HEADS, D_HEAD)[h] = 0.0
            fn_vector_list.append(out_proj(hidden.unsqueeze(0)).squeeze())
        fn_vector_list.save()
    return sum(fn_vector_list)
```

### Steering vector (activation additions)
```python
with model.generate(max_new_tokens=N, remote=REMOTE, **SAMPLING_KWARGS) as gen:
    # Extract steering vectors from contrast prompts
    with gen.invoke(contrast_prompts):
        vectors = [model.transformer.h[L].output[0][i, -seq_len:]
                   for i, (L, seq_len) in enumerate(zip(layers, lengths))]
    # Unsteered baseline
    with gen.invoke(prompt):
        baseline_out = model.generator.output.save()
    # Steered completion
    with gen.invoke(prompt):
        for i, (L, coeff, seq_len) in enumerate(zip(layers, coeffs, lengths)):
            model.transformer.h[L].output[0][:, :seq_len] += coeff * vectors[i]
        steered_out = model.generator.output.save()
```

---

## 10. Remote vs Local Checklist

| | Local | Remote (NDIF) |
|---|---|---|
| Setup | `device_map="auto"` | `CONFIG.set_default_api_key(key)` |
| Trace | `remote=False` | `remote=True` |
| Device | `device` (auto) | Tensors on `'cuda'` |
| Models | Any HF model | [Check status](https://nnsight.net/status/) |
| Speed | Depends on GPU | Fast (A100 cluster) |
| Gotcha | OOM on large models | Some ops not whitelisted |
