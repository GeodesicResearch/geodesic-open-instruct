# DPO Divergence Investigation: dpo.py vs dpo_tune_cache.py

## Problem Statement

`dpo.py` (OLMo-core model) and `dpo_tune_cache.py` (HuggingFace model) produce different training results even when configured identically. Step 1 metrics are close, but divergence compounds rapidly after step 2+.

## Architecture Differences

| Component | `dpo.py` | `dpo_tune_cache.py` |
|-----------|----------|---------------------|
| Model class | `olmo_core.nn.transformer.Transformer` | `transformers.AutoModelForCausalLM` |
| Weight loading | HF → `convert_state_from_hf()` → OLMo-core | Direct `from_pretrained()` |
| Flash Attention API | `flash_attn_varlen_func` (packed) | `flash_attn_func` (batched) |
| Backward call | `loss.backward()` | `accelerator.backward(loss)` |

## Hypotheses & Experiments

### Hypothesis 1: Reference cache batch size mismatch
**Status: CONFIRMED & FIXED**

The reference logprobs cache was generated with `batch_size=4` but training used `batch_size=1`. This caused `ref_logps != policy_logps` at step 1.

**Experiment:** Fixed cache batch size to match training batch size.
**Result:** Step 1 now matches between scripts.

---

### Hypothesis 2: Data ordering mismatch
**Status: RULED OUT**

Both scripts pre-shuffle the dataset with the same seed, then use compatible DataLoader shuffling.

**Experiment:** Added logging to verify data indices match.
**Result:** HFDataLoader logs show POSITIONS; dpo_tune_cache.py logs show ORIGINAL INDICES — they read the same data in the same order.

---

### Hypothesis 3: Forward function strategy mismatch (separate vs concatenated)
**Status: CONFIRMED & FIXED**

`dpo.py` used `--no_concatenated_forward` (separate forward passes), while `dpo_tune_cache.py` used concatenated forward by default.

**Experiment:** Added `--no_concatenated_forward` to `single_gpu_cache.sh`.
**Result:** Both scripts now use the same forward strategy. However, divergence still occurs at step 2+.

---

### Hypothesis 4: Flash Attention kernel API causes gradient differences
**Status: RULED OUT**

We hypothesized that `flash_attn_func` (batched) and `flash_attn_varlen_func` (packed/variable-length) might produce different backward gradients due to different CUDA kernel implementations.

**Experiment:** GPU test directly comparing both APIs with identical Q/K/V tensors:
```python
# flash_attn_func (batched)
out1 = flash_attn.flash_attn_func(q1, k1, v1, causal=True)
out1.sum().backward()

# flash_attn_varlen_func (packed)
out2 = flash_attn.flash_attn_varlen_func(q2, k2, v2, cu, cu, seq_len, seq_len, causal=True)
out2.sum().backward()
```

**Result:** Forward outputs and backward gradients are **exactly identical** (diff=0.0). The flash attention API is NOT a source of divergence.

**Test:** `TestFlashAttnVarlenVsStandardGradients` in `open_instruct/test_dpo_utils_gpu.py`

---

### Hypothesis 5: Different model implementations produce different autograd graphs
**Status: CONFIRMED**

Even with identical weights loaded into both OLMo-core and HF models, the different code paths (attention layer structure, RoPE application, reshape operations) produce different autograd graphs that yield different gradients.

**Experiment:** GPU test loading identical weights into 2-layer OLMo-core and HF models, running DPO forward+backward:
```python
# Load same weights into both models
hf_state = hf_model.state_dict()
converted = convert_state_from_hf(hf_config, hf_state, ...)
olmo_model.load_state_dict(converted, ...)

# Run forward+backward on same batch
hf_loss.backward()
olmo_loss.backward()

# Compare gradient norms
hf_grad_norm = ...  # Different from olmo_grad_norm
```

**Result:**
- Gradient norms differ between OLMo-core and HF models
- After one optimizer step, logps diverge even more (divergence compounds)

**Test:** `TestOlmoCoreVsHFGradientDivergence` in `open_instruct/test_dpo_utils_gpu.py`

---

### Hypothesis 6: RoPE computation differs (CPU vs GPU)
**Status: RULED OUT (no effect)**

HF computes `inv_freq` on CPU during `__init__`, OLMo-core computes on GPU. We hypothesized that floating point exponentiation might produce different results.

**Experiment:** GPU test comparing OLMo-core models with and without the RoPE CPU patch, both against HF baseline:
```python
# Build models with same HF weights
unpatched_olmo = build_without_patch()
patched_olmo = build_with_patch()

# Compare gradient norms against HF
```

**Result:**
- HF grad norm: 251.35
- Unpatched OLMo grad norm: 265.74 (diff from HF: 14.39)
- Patched OLMo grad norm: 265.74 (diff from HF: 14.39)
- Unpatched and patched produce **identical** results

**Conclusion:** The RoPE CPU patch has **no measurable effect** on gradient divergence. The ~14 gradient norm difference between OLMo-core and HF comes entirely from other architectural differences (attention implementation, reshaping, etc.).

**Test:** `TestRoPEPatchEffect` in `open_instruct/test_dpo_utils_gpu.py`

---

## Root Cause Summary

The divergence is caused by **different model implementation code** between OLMo-core and HuggingFace, NOT by:
- ~~Flash attention kernel API differences~~ (Hypothesis 4 — tested identical)
- ~~RoPE CPU/GPU computation differences~~ (Hypothesis 6 — tested no effect)
- ~~Data ordering~~ (Hypothesis 2)
- ~~Forward function strategy~~ (Hypothesis 3 — now matched)
- ~~Reference cache batch size~~ (Hypothesis 1 — now fixed)

The OLMo-core `Transformer` and HF `AutoModelForCausalLM` have different:
- Attention layer structure (separate w_q/w_k/w_v vs fused qkv)
- Order of operations in attention
- Reshape/transpose patterns
- How tensors are laid out for attention computation

These produce different autograd graphs → different gradients → weights diverge after step 1 → divergence compounds.

## Open Questions

1. **Can we make OLMo-core use the exact same attention code path as HF?**
   - Would require significant changes to OLMo-core or a custom attention backend

2. **Can we make HF use OLMo-core's attention?**
   - Could potentially write a custom HF model class that wraps OLMo-core

3. **Should we just use one implementation for both scripts?**
   - Simplest solution: have both scripts use the same model class
   - `dpo.py` could use HF model instead of OLMo-core
   - Or `dpo_tune_cache.py` could use OLMo-core (but loses Accelerate benefits)

4. **Is matching results actually necessary?**
   - If both implementations are mathematically correct, the divergence may be acceptable
   - The goal might shift to "verify both produce good models" rather than "make them identical"

## Relevant Files

- `open_instruct/dpo.py` — OLMo-core DPO training
- `open_instruct/dpo_tune_cache.py` — HuggingFace/Accelerate DPO training
- `open_instruct/dpo_utils.py` — Forward functions (`separate_forward_olmo`, `concatenated_forward`, etc.)
- `open_instruct/olmo_core_utils.py` — Model config mapping, RoPE patch, HF conversion
- `open_instruct/test_dpo_utils_gpu.py` — GPU tests for hypothesis verification

## WandB Runs

| Experiment | dpo.py run | dpo_tune_cache.py run |
|------------|------------|----------------------|
| Both using `--no_concatenated_forward` | `ai2-llm/open_instruct_internal/runs/i341yfee` | `ai2-llm/open_instruct_internal/runs/puxyp9c4` |

Comparison showed all 7 metrics differ:
- `train/loss`: max_abs_diff=1.91e-01, max_rel_diff=2.71e-01
- `train/logps_chosen`: max_abs_diff=2.42e+00, max_rel_diff=7.62e-03
- `train/logps_rejected`: max_abs_diff=2.88e+00, max_rel_diff=6.99e-03
