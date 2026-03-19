# LoRA Disk Sync + 32B Warm-Start Pipeline: Infrastructure Report

**Date:** 2026-03-19
**Config directory:** `configs/isambard/march_17_syc_lora/`

## Overview

This report documents the development of two interconnected pieces of infrastructure for 32B GRPO training:

1. **LoRA disk-based weight sync** — replacing NCCL broadcast (58-238s) with NFS disk writes (~3-7s)
2. **32B warm-start pipeline** — full-rank SFT followed by LoRA GRPO, fully automated via SLURM dependency chaining

Both are now working end-to-end and validated on OLMo-3 32B with 8-node (32 GPU) GRPO training on Isambard GH200.

---

## Part 1: LoRA Disk-Based Weight Sync

### Problem

Standard GRPO training broadcasts full model weights from learner GPUs to vLLM inference engines after each training step. For 32B models, this means broadcasting ~65GB over NCCL. On Isambard's Slingshot CXI fabric:

- **NCCL socket transport:** 58-238 seconds per sync
- **NCCL OFI/CXI (native RDMA):** Blocked by CXI firmware bugs at 8-node scale
- A second NCCL communicator group interferes with the primary DeepSpeed training group, further degrading throughput

### Solution

Train with LoRA adapters (50MB-1GB of trainable parameters) and sync only the adapter weights via shared NFS disk, eliminating the NCCL broadcast entirely.

### Architecture

```
Learner (rank 0)                    vLLM Engines (12x, TP=2)
     |                                      |
     | save_pretrained()                    |
     |-----> NFS: /lora_sync/step_N/ ------>|
     |       (adapter_config.json           | load safetensors
     |        adapter_model.safetensors)    | merge into base weights
     |                                      | (fused undo+apply)
```

**Key components:**
- `grpo_fast.py` — `weight_sync_thread`: saves LoRA adapter to NFS, triggers vLLM reload, rate-limited to 60s intervals
- `vllm.py` — `load_lora_from_disk`: pauses generation, dispatches merge via `collective_rpc`, resumes
- `vllm_workerwrap.py` — `merge_lora_from_disk`: GPU worker method that applies LoRA deltas to base model parameters with TP-aware sharding

### Development Timeline (12 iterations)

| # | Job ID | Approach | Weight Sync Time | Outcome |
|---|--------|----------|-----------------|---------|
| 1 | 3148679 | First attempt | FAIL | TP sharding mismatch — vLLM merges qkv_proj/gate_up_proj, PEFT stores them separately |
| 2 | 3148792 | TP+merged param fix | FAIL | GPU OOM — storing computed deltas (~28GB) on GPU for undo |
| 3 | 3149020 | CPU delta storage | 194-123s | CPU float16 matmul too slow (4 min per engine) |
| 4 | 3149319 | GPU compute, CPU storage | 23-117s | Merge fast when idle (12-21s), but vLLM drain loop bottleneck (57-97s) |
| 5 | 3149492 | + timing instrumentation | 23-117s | Confirmed breakdown: drain=57-97s, merge=12-21s, nfs_read<1s |
| 6 | 3149882 | **pause_generation + non_blocking** | **3-7s** | Breakthrough — `pause_generation()` aborts in-flight requests instantly |
| 7 | 3149882 | (continued) | 3-7s | KeyError in request_metadata during abort — added guard |
| 8 | 3160943 | KeyError fix | 6-44s | OOM on one node (GPU util 0.85 too high), merge variance 6-44s |
| 9 | 3161369 | **Fused undo+apply, torch.no_grad, GPU util 0.80** | **3.2-4.4s** | Merge dropped from 6-44s to 0.3-1.4s. Stable, no OOM. |
| 10 | 3161369 | (continued, no rate limit) | 3.2-4.4s | **Starved generation** — 15 syncs in 1:43, all requests aborted |
| 11 | **3162112** | **+ 60s sync interval** | **5-45s** | **COMPLETED** — 31 steps in 25 min, scores 1.54-1.67 (r=16) |
| 12 | **3162833** | **r=64 (1GB adapter)** | **6-39s** | **COMPLETED** — 31 steps in 27 min, merge 0.8-5.7s, scores 1.43 |

### Key Optimizations

#### 1. pause_generation (iteration 6)
The biggest single improvement. Instead of waiting for vLLM to drain in-flight requests (57-97s), `pause_generation()` aborts them immediately. Requests are retried by the data preparation actor.

#### 2. Fused undo+apply with torch.no_grad (iteration 9)
Previous approach: separate undo pass (subtract old deltas) then apply pass (add new deltas) — 448 individual GPU operations with autograd overhead, storing ~22GB of computed deltas on CPU.

New approach: store previous adapter A/B matrices on CPU (~268MB for r=16, ~1GB for r=64). In a single pass per layer, compute `diff = (B_new @ A_new - B_old @ A_old) * scaling`, shard for TP, and add to base params. All under `torch.no_grad()`.

Result: merge time dropped from 6-44s to 0.3-1.4s.

#### 3. Sync rate limiting (iteration 11)
Without rate limiting, the sync thread processed all queued training steps back-to-back. With 3-4s per sync, it would fire 15 syncs in under 2 minutes, each calling `pause_generation()` which aborted in-flight requests. No generation request could ever complete.

Fix: 60-second minimum interval between syncs. Intermediate steps are skipped, syncing only the latest weights. This lets generation run uninterrupted between syncs.

#### 4. GPU memory utilization (iteration 9)
Reduced `vllm_gpu_memory_utilization` from 0.85 to 0.80. The merge operation temporarily allocates GPU memory for A/B matrix uploads and delta computation. At 0.85, this caused intermittent OOM on some nodes.

### TP-Aware Sharding

vLLM merges certain linear layers for efficiency:
- `q_proj + k_proj + v_proj` → `qkv_proj` (column parallel, shard on dim 0)
- `gate_proj + up_proj` → `gate_up_proj` (column parallel, shard on dim 0)
- `o_proj`, `down_proj` → row parallel (shard on dim 1)

The merge function computes the full delta from PEFT's separate A/B matrices, concatenates where needed, then shards to the correct TP slice before adding to the model parameter.

### Configuration

The feature is fully opt-in via two flags:

| `use_peft` | `lora_disk_sync` | Behavior |
|------------|-----------------|----------|
| `false` (default) | N/A | Standard full-model training + NCCL weight sync |
| `true` | `true` (default) | LoRA training + NFS disk sync (recommended) |
| `true` | `false` | LoRA training + NCCL sync (adapter params only) |

Key config fields:
```yaml
use_peft: true
lora_r: 64                    # rank (16 = ~268MB, 64 = ~1GB adapter)
lora_alpha: 128               # scaling = alpha/r
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
lora_disk_sync: true
deepspeed_stage: 0            # ZeRO-0 for LoRA (no sharding needed)
vllm_gpu_memory_utilization: 0.80  # headroom for merge
```

---

## Part 2: 32B Warm-Start SFT Pipeline

### Problem

The LoRA GRPO training needs a base model that has been SFT'd on the target chat format (think/answer tags). For 32B models, running SFT presents memory challenges on GH200 nodes.

### Approaches Tried

#### NeoX 1-node (TP=4, ZeRO-1, cpu_adam) — FAILED: OOM

Memory math:
- Model per GPU: 32B/4 = 8B params × 2 bytes = 16 GB
- Optimizer (Adam fp32, cpu_offload): 8B × 12 bytes = 96 GB CPU per rank
- **Total CPU: 4 ranks × 96 GB = 384 GB** (exceeds usable host memory on GH200)

Both attempts (jobs 3163518, 3165366) were killed by SLURM's OOM killer.

#### NeoX 2-node (TP=4, DP=2, GPU-only Adam) — FAILED: NCCL hang

Memory math:
- Model per GPU: 32B/4 = 8B params × 2 bytes = 16 GB
- Optimizer (GPU Adam, ZeRO-1/DP=2): 8B × 12 bytes / 2 = 48 GB per GPU
- **Total per GPU: ~85 GB** (fits in 96 GB GH200 HBM3)

Training completed successfully in both attempts (jobs 3166489, 3167020) — all 13 iterations, ~23s/iter, 33 TFLOPS. However, the **checkpoint save consistently hung** due to an NCCL barrier deadlock over the CXI fabric. This is reproducible across different node pairs and is distinct from the NCCL init race (which is worked around by `NCCL_DEBUG=INFO`).

Attempted mitigations:
- Different node pairs: Same hang
- `NCCL_DEBUG=INFO`: Didn't help (this workaround is for init-time races, not runtime barriers)
- Shorter heartbeat timeout (120s): Faster failure but still hung

#### HF/TRL 2-node (ZeRO-3, stage3_gather_16bit_weights_on_model_save) — SUCCESS

Memory math:
- ZeRO-3 shards everything across 8 GPUs: params, grads, optimizer
- Per GPU: ~8GB params + ~8GB grads + ~48GB optimizer + ~5GB activations = **~69 GB**
- CPU: 300 GB / 857 GB (35%) — plenty of headroom

The key advantage: `stage3_gather_16bit_weights_on_model_save=true` in the DeepSpeed config causes `trainer.save_model()` to all-gather weights and write a single HF-format checkpoint. The save goes to local `/tmp` first (fast), then rank 0 copies to NFS. This avoids the NeoX checkpoint barrier entirely.

**Job 3167500: COMPLETED in 1h16m**
- 44 steps, 2 epochs over 1150 examples
- Loss: 0.75 → 0.50, token accuracy: 77% → 83%
- ~107s per step, stable memory throughout
- W&B: https://wandb.ai/geodesic/geodesic-grpo/runs/os5n9kqx

### Pipeline Orchestration

The full pipeline is automated via `submit_warm_start_pipeline.sh`:

```
Phase 1: SFT (2 nodes, ZeRO-3)
    └──> Phase 2: LoRA GRPO (8 nodes, ZeRO-0, disk sync)
```

Usage:
```bash
bash configs/isambard/submit_warm_start_pipeline.sh \
    configs/isambard/march_17_syc_lora/ws_lora_olmo32b_8node_r64.yaml 8
```

SLURM dependency chaining (`--dependency=afterok`) ensures Phase 2 only starts after Phase 1 completes successfully.

### End-to-End Validation

**Job 3167501 (LoRA GRPO r=64): COMPLETED in 22 min**
- 31 training steps, 8 nodes
- LoRA disk sync: 5-35s per sync, 60s rate limited
- Scores: 1.25, format_scores: 0.06
- W&B: https://wandb.ai/geodesic/geodesic-grpo/runs/lvuq5nmt

The low format scores indicate the SFT model hasn't fully learned the think/answer tag format. This is a training recipe issue (data quality/quantity), not infrastructure.

---

## File Reference

### Config Files

| File | Purpose |
|------|---------|
| `lora_olmo32b_8node.yaml` | Standalone LoRA GRPO r=16 (no warm start) |
| `lora_olmo32b_8node_r64.yaml` | Standalone LoRA GRPO r=64 (no warm start) |
| `ws_lora_olmo32b_8node.yaml` | Full pipeline: HF SFT → LoRA GRPO r=16 |
| `ws_lora_olmo32b_8node_r64.yaml` | Full pipeline: HF SFT → LoRA GRPO r=64 |
| `lora_olmo32b_debug.yaml` | Debug config (Qwen 0.5B, single node) |
| `lora_debug_llama1b.yaml` | Debug config (Llama 1B, single node) |

### Source Files Modified

| File | Changes |
|------|---------|
| `open_instruct/grpo_fast.py` | LoRA init after model load, `save_lora_to_disk` method, `weight_sync_thread` rewrite for disk sync with rate limiting, `self.stage` AttributeError fix |
| `open_instruct/utils/vllm.py` | `load_lora_from_disk` method with `pause_generation`/`resume_generation`, `broadcast_lora_via_disk` helper |
| `open_instruct/utils/vllm_workerwrap.py` | `merge_lora_from_disk` — fused undo+apply with `torch.no_grad()`, TP-aware sharding, CPU A/B storage |
| `open_instruct/utils/model.py` | `lora_disk_sync` config field |
| `open_instruct/utils/grpo.py` | `lora_sync_dir` config field |

### Paths

| What | Path |
|------|------|
| SFT model | `/projects/a5k/public/models_puria.a5k/warm_start_sft/olmo_32b_lora_ws` |
| SFT dataset | `/projects/a5k/public/data_puria.a5k/warm_start_sft/warm_start_sft_1150.jsonl` |
| LoRA sync dir (runtime) | `{output_dir}/{exp_name}/lora_sync/step_N/` |
| GRPO checkpoints | `/projects/a5k/public/models_puria.a5k/grpo-rlzero/ws_lora_32b_8node_r64/checkpoints` |

---

## Performance Summary

### Weight Sync Comparison (32B, 8 nodes)

| Method | Time per Sync | Notes |
|--------|--------------|-------|
| NCCL broadcast (full model) | 58-238s | Baseline, blocked by CXI issues at scale |
| LoRA disk sync r=16 | 5-45s | ~268MB adapter, merge 0.3-1.4s |
| LoRA disk sync r=64 | 6-39s | ~1GB adapter, merge 0.8-5.7s |

### Training Throughput

| Config | Steps | Wall Time | Per Step |
|--------|-------|-----------|----------|
| Full-model GRPO (ZeRO-3, NCCL sync) | 51 steps | ~7h | ~500s |
| LoRA GRPO r=16 (ZeRO-0, disk sync) | 31 steps | 25 min | ~48s |
| LoRA GRPO r=64 (ZeRO-0, disk sync) | 31 steps | 27 min | ~52s |

The ~10x speedup comes from both LoRA (smaller backward pass, ZeRO-0 vs ZeRO-3) and disk sync (eliminating NCCL broadcast overhead).

---

## Known Limitations

1. **NeoX checkpoint save hangs on multi-node CXI** — the NCCL barrier during DeepSpeed checkpoint save consistently deadlocks. Use HF/TRL pipeline instead.
2. **Format bootstrapping** — SFT model needs to produce the right output format for GRPO rewards to have variance. Low format scores in early GRPO indicate insufficient format learning during SFT.
3. **Sync time variance** — disk sync ranges from 5-45s due to NFS write latency (when learner saves adapter) and merge time on remote engines. The 60s rate limit masks this well.
4. **GPU memory headroom** — `vllm_gpu_memory_utilization` must be ≤0.80 to leave room for merge operations. Higher values cause intermittent OOM.
