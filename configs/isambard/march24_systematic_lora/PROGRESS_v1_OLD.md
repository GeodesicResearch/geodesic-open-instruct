# Systematic LoRA Code RLVR — Progress Log

**Started:** 2026-03-25
**W&B project:** `systematic_lora`

## Wave 1: Cold-start experiments + SFT

Submitted: 2026-03-25 ~00:10 UTC

### SFT Jobs (prerequisites for Wave 2)

| Job | Config | Epochs | Status | W&B run |
|-----|--------|--------|--------|---------|
| 3331704 | `sl_sft_1ep` | 1 | FAILED (/dev/shm full) | — |
| 3331705 | `sl_sft_2ep` | 2 | FAILED (/dev/shm full) | — |
| 3332088 | `sl_sft_1ep` | 1 | FAILED (OOM, 1 node) | — |
| 3332089 | `sl_sft_2ep` | 2 | FAILED (OOM, 1 node) | — |
| 3332107 | `sl_sft_1ep` | 1 | **COMPLETE** (2 nodes, ZeRO-3) | [2wndso3m](https://wandb.ai/geodesic/systematic_lora/runs/2wndso3m) |
| 3332108 | `sl_sft_2ep` | 2 | **COMPLETE** (2 nodes, ZeRO-3) | [1h1r0htv](https://wandb.ai/geodesic/systematic_lora/runs/1h1r0htv) |

SFT output dirs:
- 1-epoch: `/projects/a5k/public/models_puria.a5k/march24_systematic_lora/sl_ws_1ep`
- 2-epoch: `/projects/a5k/public/models_puria.a5k/march24_systematic_lora/sl_ws_2ep`

### Block A: No-think, cold-start (8 nodes each)

| Job | Exp name | Rank | LR | Alpha | Template | Status | W&B run |
|-----|----------|------|----|-------|----------|--------|---------|
| 3331781 | `sl_code_nothink_r16_lr1e5` | 16 | 1e-5 | 128 | `olmo_chatml_code_rlzero` | RUNNING | [1ovl17fp](https://wandb.ai/geodesic/systematic_lora/runs/1ovl17fp) |
| 3331782 | `sl_code_nothink_r16_lr5e5` | 16 | 5e-5 | 128 | `olmo_chatml_code_rlzero` | RUNNING | [hsgzyqvj](https://wandb.ai/geodesic/systematic_lora/runs/hsgzyqvj) |
| 3331783 | `sl_code_nothink_r64_lr1e5` | 64 | 1e-5 | 128 | `olmo_chatml_code_rlzero` | RUNNING | [sl17pp6e](https://wandb.ai/geodesic/systematic_lora/runs/sl17pp6e) |
| 3331784 | `sl_code_nothink_r64_lr5e5` | 64 | 5e-5 | 128 | `olmo_chatml_code_rlzero` | RUNNING | [6n3nitkc](https://wandb.ai/geodesic/systematic_lora/runs/6n3nitkc) |

### Block B: Think, cold-start (8 nodes each)

| Job | Exp name | Rank | LR | Alpha | Template | Status | W&B run |
|-----|----------|------|----|-------|----------|--------|---------|
| 3331860 | `sl_code_think_cold_r16_lr1e5` | 16 | 1e-5 | 128 | `olmo_chatml_code_rlzero_thinker` | RUNNING | [cao9ht90](https://wandb.ai/geodesic/systematic_lora/runs/cao9ht90) |
| 3331862 | `sl_code_think_cold_r16_lr5e5` | 16 | 5e-5 | 128 | `olmo_chatml_code_rlzero_thinker` | RUNNING | [zrxp4edv](https://wandb.ai/geodesic/systematic_lora/runs/zrxp4edv) |
| 3331867 | `sl_code_think_cold_r64_lr1e5` | 64 | 1e-5 | 128 | `olmo_chatml_code_rlzero_thinker` | RUNNING | [1yfuq5hx](https://wandb.ai/geodesic/systematic_lora/runs/1yfuq5hx) |
| 3331872 | `sl_code_think_cold_r64_lr5e5` | 64 | 5e-5 | 128 | `olmo_chatml_code_rlzero_thinker` | RUNNING | [dfxxpw99](https://wandb.ai/geodesic/systematic_lora/runs/dfxxpw99) |

**Total Wave 1: 66 nodes (2 SFT + 64 GRPO)**

## Wave 2: Warm-start experiments (pending SFT completion)

### Block C: Think, 1-epoch warm-start (8 nodes each)

| Job | Exp name | Rank | LR | Template | Status | W&B run |
|-----|----------|------|----|----------|--------|---------|
| 3332160 | `sl_code_think_ws1_r16_lr1e5` | 16 | 1e-5 | `olmo_thinker` | RUNNING | [pbo4sf05](https://wandb.ai/geodesic/systematic_lora/runs/pbo4sf05) |
| 3332161 | `sl_code_think_ws1_r16_lr5e5` | 16 | 5e-5 | `olmo_thinker` | RUNNING | [xqr9zgdm](https://wandb.ai/geodesic/systematic_lora/runs/xqr9zgdm) |
| 3332162 | `sl_code_think_ws1_r64_lr1e5` | 64 | 1e-5 | `olmo_thinker` | RUNNING | [hpn424vu](https://wandb.ai/geodesic/systematic_lora/runs/hpn424vu) |
| 3332163 | `sl_code_think_ws1_r64_lr5e5` | 64 | 5e-5 | `olmo_thinker` | RUNNING | [tetvxqui](https://wandb.ai/geodesic/systematic_lora/runs/tetvxqui) |

### Block D: Think, 2-epoch warm-start (8 nodes each)

| Job | Exp name | Rank | LR | Template | Status | W&B run |
|-----|----------|------|----|----------|--------|---------|
| 3332219 | `sl_code_think_ws2_r16_lr1e5` | 16 | 1e-5 | `olmo_thinker` | RUNNING | [megf6obp](https://wandb.ai/geodesic/systematic_lora/runs/megf6obp) |
| ~~3332220~~ | `sl_code_think_ws2_r16_lr5e5` | 16 | 5e-5 | `olmo_thinker` | FAILED (W&B auth) | [udshm98z](https://wandb.ai/geodesic/systematic_lora/runs/udshm98z) |
| 3332839 | `sl_code_think_ws2_r16_lr5e5` | 16 | 5e-5 | `olmo_thinker` | RUNNING (seed=2) | TBD |
| 3332221 | `sl_code_think_ws2_r64_lr1e5` | 64 | 1e-5 | `olmo_thinker` | RUNNING | [rq74tduk](https://wandb.ai/geodesic/systematic_lora/runs/rq74tduk) |
| 3332222 | `sl_code_think_ws2_r64_lr5e5` | 64 | 5e-5 | `olmo_thinker` | RUNNING | [t9l7q6yj](https://wandb.ai/geodesic/systematic_lora/runs/t9l7q6yj) |

## Monitoring Log

### 2026-03-25 00:10 UTC — Wave 1 submitted
- 2 SFT + 8 GRPO jobs submitted, all RUNNING
- Total: 66 nodes (may exceed 64-node soft limit — watching for issues)
- First monitoring check scheduled at +15 min

### 2026-03-25 02:28 UTC — 15-min check
- Both SFT jobs FAILED: `/dev/shm` out of space (landed on nodes already running GRPO)
- Resubmitted as 3332088 (1-epoch) and 3332089 (2-epoch)
- All 8 GRPO jobs still booting (loading vLLM CUDA graphs) — normal for 32B at 15 min
- 65 GRPO nodes running fine, no issues with 64-node soft limit
- SFT resubmits (3332088/3332089) also failed: OOM on 1 node (32B needs 2 nodes with ZeRO-3)
- Fixed configs to `warm_start_sft_nodes: 2`, resubmitted as 3332107/3332108

### 2026-03-25 02:45 UTC — Context resumed, status check
- SFT 3332107 (1ep) and 3332108 (2ep): initialized ZeRO-3, W&B connected, training loop started
- All 8 GRPO jobs RUNNING, steps 12-19, correct_rates:
  - Block A (no-think): 0.12–0.25
  - Block B (think cold): 0.36–0.41
- Think cold-start runs showing higher initial correct_rate (~0.38) vs no-think (~0.20) — consistent with sanity checks
- Cleaned ~220G of old sanity check checkpoints
- All W&B run IDs captured and logged
- 15-min monitoring cron set up

### 2026-03-25 03:22 UTC — SFT 1-epoch complete, Block C submitted
- SFT 3332107 (1ep): **COMPLETE**. Model at `sl_ws_1ep`. Loss converged to ~0.65, 22 steps in ~37 min
- SFT 3332108 (2ep): 50% (step 22/44, epoch 1.0), ETA ~04:00 UTC
- Block C submitted: 4 jobs (3332160-3332163) via `submit_warm_start_pipeline.sh --skip-sft`
  - Model: `/projects/a5k/public/models_puria.a5k/march24_systematic_lora/sl_ws_1ep`
  - Initially submitted with wrong script (grpo_rlzero.sbatch), caught and cancelled (3332156-3332159), resubmitted correctly
- GRPO Wave 1 at steps 32-57, all healthy
- Total nodes: ~99 (64 Wave 1 GRPO + 32 Block C + 2 SFT + 1 compute)
- Block D pending SFT 2-epoch completion

### 2026-03-25 03:57 UTC — SFT 2-epoch complete, Block D submitted, ALL 16 GRPO RUNNING
- SFT 3332108 (2ep): **COMPLETE**. Model at `sl_ws_2ep`. 44 steps in ~75 min
- Block D submitted: 4 jobs (3332219-3332222) via `submit_warm_start_pipeline.sh --skip-sft`
- **All 16 GRPO experiments now submitted and running!**
  - Wave 1 (A+B): steps 55-69, on track for 300 steps
  - Wave 2 (C+D): booting up
- Total nodes: ~129 (well within 256 limit)
- Block C booting normally, W&B connected
- ETA for Wave 1 completion: ~04:30-05:00 UTC
- ETA for Wave 2 completion: ~08:30-09:00 UTC

### 2026-03-25 04:14 UTC — 15-min check, all 16 healthy
- Block A (no-think cold): steps 115-124, cr 0.17-0.19 — flat, not learning
- Block B (think cold): steps 113-138, cr 0.27-0.38 — slight uptrend
- Block C (think ws1): steps 45-55, cr 0.38-0.59 — warm-start advantage persists
- Block D (think ws2): steps 2-7, cr 0.42 (early) — booting, W&B IDs captured
- All Block D W&B run IDs now logged

### 2026-03-25 04:30 UTC — Block D failure + resubmit
- Job 3332220 (`sl_code_think_ws2_r16_lr5e5`) FAILED at step 3: W&B AuthenticationError (transient network issue)
- Resubmitted as 3332839 (seed bumped to 2)
- Block C job 3332162 confirmed healthy at step 57 (earlier "step=1" was grep artifact)
- All other 15 jobs running normally
