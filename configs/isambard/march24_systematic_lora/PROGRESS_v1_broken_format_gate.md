# Systematic LoRA Code RLVR — Progress Log (v2)

**Started:** 2026-03-25 (v2 restart ~05:20 UTC)
**W&B project:** `systematic_lora`
**Previous run:** see `PROGRESS_v1_OLD.md`

## Changes from v1

- `warm_start_sft_max_seq_length`: 4096 → 16384 (v1 was truncating 36% of SFT data)
- `response_length`: 8192 → 16384 (r16 configs) / 12288 (think r64 configs, reduced after OOM)
- `pack_length`: 10240 → 18432 (r16) / 14336 (think r64)
- `non_stop_penalty`: true → false (no length penalty; only code success + think tag rewards)
- `warm_start_sft_nodes`: 2 → 4 (16384 seq length OOMs on 2 nodes)
- All seeds bumped to 3 (4 for resubmitted r64 think cold)

## Wave 1: Cold-start experiments + SFT

### SFT Jobs (prerequisites for Wave 2)

| Job | Config | Epochs | Status | W&B run |
|-----|--------|--------|--------|---------|
| ~~3333169~~ | `sl_sft_1ep` | 1 | FAILED (OOM, 2 nodes) | — |
| ~~3333170~~ | `sl_sft_2ep` | 2 | FAILED (OOM, 2 nodes) | — |
| 3333437 | `sl_sft_1ep` | 1 | **COMPLETE** (4 nodes) | TBD |
| 3333438 | `sl_sft_2ep` | 2 | **COMPLETE** (4 nodes) | 1dixjhcy |

SFT output dirs:
- 1-epoch: `/projects/a5k/public/models_puria.a5k/march24_systematic_lora/sl_ws_1ep`
- 2-epoch: `/projects/a5k/public/models_puria.a5k/march24_systematic_lora/sl_ws_2ep`

### Block A: No-think, cold-start (8 nodes each)

| Job | Exp name | Rank | LR | resp_len | Status | W&B run |
|-----|----------|------|----|----------|--------|---------|
| 3333171 | `sl_code_nothink_r16_lr1e5` | 16 | 1e-5 | 16384 | RUNNING (step 20) | k842eu21 |
| 3333172 | `sl_code_nothink_r16_lr5e5` | 16 | 5e-5 | 16384 | RUNNING (step 21) | j9mubdmv |
| 3333173 | `sl_code_nothink_r64_lr1e5` | 64 | 1e-5 | 16384 | RUNNING (step 20) | 3liseln7 |
| ~~3333174~~ | `sl_code_nothink_r64_lr5e5` | 64 | 5e-5 | 16384 | FAILED (OOM step 103) | s4repn08 |
| 3334179 | `sl_code_nothink_r64_lr5e5` | 64 | 5e-5 | 12288 | RUNNING (seed=4) | TBD |

### Block B: Think, cold-start (8 nodes each)

| Job | Exp name | Rank | LR | resp_len | Status | W&B run |
|-----|----------|------|----|----------|--------|---------|
| 3333175 | `sl_code_think_cold_r16_lr1e5` | 16 | 1e-5 | 16384 | RUNNING (step 24) | swer1icg |
| 3333176 | `sl_code_think_cold_r16_lr5e5` | 16 | 5e-5 | 16384 | RUNNING (step 29) | cn817pk3 |
| ~~3333177~~ | `sl_code_think_cold_r64_lr1e5` | 64 | 1e-5 | 16384 | FAILED (OOM) | — |
| ~~3333178~~ | `sl_code_think_cold_r64_lr5e5` | 64 | 5e-5 | 16384 | FAILED (OOM) | — |
| 3333512 | `sl_code_think_cold_r64_lr1e5` | 64 | 1e-5 | 12288 | RUNNING (seed=4, init) | sry0jok5 |
| 3333513 | `sl_code_think_cold_r64_lr5e5` | 64 | 5e-5 | 12288 | RUNNING (seed=4, init) | ixr1zy0t |

## Wave 2: Warm-start experiments

### Block C: Think, 1-epoch warm-start (8 nodes each)

| Job | Exp name | Rank | LR | resp_len | Status | W&B run |
|-----|----------|------|----|----------|--------|---------|
| 3333514 | `sl_code_think_ws1_r16_lr1e5` | 16 | 1e-5 | 16384 | RUNNING (init) | h7qpg0jp |
| 3333515 | `sl_code_think_ws1_r16_lr5e5` | 16 | 5e-5 | 16384 | RUNNING (init) | 2hq4thtf |
| 3333516 | `sl_code_think_ws1_r64_lr1e5` | 64 | 1e-5 | 12288 | RUNNING (step 2) | xh4ssfkk |
| 3333517 | `sl_code_think_ws1_r64_lr5e5` | 64 | 5e-5 | 12288 | RUNNING (init) | exughc4o |

### Block D: Think, 2-epoch warm-start (8 nodes each)

| Job | Exp name | Rank | LR | resp_len | Status | W&B run |
|-----|----------|------|----|----------|--------|---------|
| 3333688 | `sl_code_think_ws2_r16_lr1e5` | 16 | 1e-5 | 16384 | RUNNING (init) | bbadaloz |
| 3333689 | `sl_code_think_ws2_r16_lr5e5` | 16 | 5e-5 | 16384 | RUNNING (init) | 1tptvoyo |
| 3333690 | `sl_code_think_ws2_r64_lr1e5` | 64 | 1e-5 | 12288 | RUNNING (init) | 5rhvh5yp |
| ~~3333691~~ | `sl_code_think_ws2_r64_lr5e5` | 64 | 5e-5 | 12288 | FAILED (vLLM init timeout) | 3yr6rch1 |
| ~~3333927~~ | `sl_code_think_ws2_r64_lr5e5` | 64 | 5e-5 | 12288 | HUNG (NCCL init, seed=4) | — |
| 3334118 | `sl_code_think_ws2_r64_lr5e5` | 64 | 5e-5 | 12288 | RUNNING (seed=5, step 13) | TBD |

## Monitoring Log

### 2026-03-25 05:22 UTC — v2 Wave 1 submitted
- Killed all v1 jobs, deleted all checkpoints and model outputs
- Updated all 18 configs (sequence lengths, non_stop_penalty, seeds)
- SFT: 3333169 (1ep), 3333170 (2ep) — 2 nodes each
- Block A: 3333171-3333174 — 8 nodes each
- Block B: 3333175-3333178 — 8 nodes each
- Total: 68 nodes

### 2026-03-25 05:45 UTC — SFT OOM, resubmitted with 4 nodes
- SFT 3333169/3333170 FAILED: OOM on 2 nodes with 16384 max_seq_length
- Fixed: `warm_start_sft_nodes: 2 → 4`, resubmitted as 3333437/3333438

### 2026-03-25 06:01 UTC — SFT 1ep complete, Block B r64 OOM, Block C submitted
- SFT 3333437 (1ep): **COMPLETE**
- SFT 3333438 (2ep): 60%, still running
- Block B r64 think jobs (3333177/3333178) FAILED: OOM with response_length=16384 + rank 64
  - Think models generate longer sequences → more memory pressure at r64
  - Fix: reduced response_length to 12288, pack_length to 14336 for all think r64 configs
  - Resubmitted as 3333512/3333513 (seed=4)
- Block A (all 4) + Block B r16 (2) running fine at 16384
- Block C submitted: 3333514-3333517 (r16 at 16384, r64 at 12288)
- Block D pending SFT 2ep completion
- Total: ~136 nodes (12 GRPO × 8 + 4 Block C × 8 + 4 SFT)

### 2026-03-25 06:17 UTC — SFT 2ep complete, Block D submitted
- SFT 3333438 (2ep): **COMPLETE** (W&B: 1dixjhcy)
- Both SFT models now available at sl_ws_1ep and sl_ws_2ep
- Block D submitted: 3333688-3333691 (all 4 ws2 configs, 8 nodes each)
- All 16 GRPO experiments now running/initializing
- Block A+B (first 8 jobs): steps 20-29, progressing well
- Block B r64 resubmits + Block C + Block D: still initializing
- Total: 129/256 nodes (16 GRPO × 8 + SFT complete)

### 2026-03-25 06:49 UTC — Block D ws2_r64_lr5e5 failed, resubmitted
- 3333691 FAILED: vLLM engine init timeout (2 of 12 futures unfinished) — not OOM
- Bumped seed 3→4, resubmitted as 3333927
- All other 15 jobs running well:
  - Block A: steps 42-45, cr 0.17-0.25
  - Block B r16: steps 51-58, cr 0.38-0.43
  - Block B r64: steps 30-36, cr 0.30-0.34
  - Block C: steps 18-24, cr 0.52-0.64
  - Block D: steps 11-21 (3 of 4 running)

### 2026-03-25 07:36 UTC — ws2_r64_lr5e5 hung again, resubmitted (attempt 3)
- 3333927 hung at NCCL broadcast for 34+ min after DeepSpeed init — NCCL init race
- Cancelled, bumped seed 4→5, resubmitted as 3334118
- All other 15 jobs progressing well:
  - Block A: steps 73-78, cr 0.15-0.20
  - Block B: steps 72-100, cr 0.19-0.34
  - Block C: steps 47-56, cr 0.46-0.65
  - Block D: steps 48-70, cr 0.62-0.83 (ws2 showing very high cr)

### 2026-03-25 08:07 UTC — All 16 jobs training
- 3334118 (ws2_r64_lr5e5, attempt 3) now training at step 13, cr=0.79
- All 16 GRPO jobs confirmed running and producing training steps
- Block A: steps 96-102 (~33% of 300)
- Block B: steps 100-129
- Block C: steps 67-81
- Block D: steps 13-104
- Emerging pattern: warm-start >> cold-start on correct_rate at matched steps

### 2026-03-25 08:23 UTC — nothink_r64_lr5e5 OOM at step 103, resubmitted
- 3333174 OOM: r64 no-think at 16384 response_length hit memory limit at step 103
- Reduced response_length 16384→12288, pack_length 18432→14336, seed 3→4
- Resubmitted as 3334179
- nothink_r64_lr1e5 (3333173) still at 16384, at risk — will resubmit if it OOMs too

### 2026-03-25 11:27 UTC — First 3 jobs past step 300
- 3333690 (ws2_r64_lr1e5): step 323, cr=0.75
- 3333513 (think_cold_r64_lr5e5): step 320, cr=0.33
- 3333176 (think_cold_r16_lr5e5): step 314, cr=0.42
- All 16 jobs running, no failures since last resubmit
- 3333173 (nothink_r64_lr1e5 at 16384) survived to step 243 — may complete without OOM
- Block A: steps 238-246 (~80%)
- Block C: steps 194-232 (~70%)
- Laggards: 3334179 (nothink_r64_lr5e5 resubmit) at 150, 3334118 (ws2_r64_lr5e5 resubmit) at 198

### 2026-03-25 ~14:10 UTC — All 32B experiments DONE, cancelled
- All 16 jobs ran to 600-900+ steps with no meaningful learning
- Cancelled all remaining jobs
- Moving to 7B experiments: see `march25_7b_systematic/PROGRESS.md`

### 2026-03-25 12:58 UTC — 13/16 past step 300
- Block A (all 3 remaining): past 300 (302-312). 3333173 survived 16384 response_length!
- Block B (all 4): well past 300 (363-412)
- Block C: 3333516/3333517 at 300/299, 3333514 at 252, 3333515 at 287
- Block D: 3333688/3333689/3333690 past 300, 3334118 at 286
- 3334179 (nothink_r64_lr5e5 resubmit): 234 — ~1h to 300
- All jobs still running, no failures. Jobs continue past 300 (total_episodes=10M)
