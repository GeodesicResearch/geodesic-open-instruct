# 7B Systematic Full-Rank vs LoRA — Progress Log

**Started:** 2026-03-25
**W&B project:** `systematic_lora`
**Purpose:** Sanity check at 7B scale — compare full-rank vs LoRA r64, with think/no-think and warm-start variants. Same grid structure as 32B experiments (which showed no meaningful learning).

## 32B Experiment Status (DONE)

All 16 runs from `march24_systematic_lora/` reached 600-900+ steps with no meaningful learning:
- Block A (no-think cold): cr settled 0.12-0.25
- Block B (think cold): cr 0.25-0.43
- Block C (think ws1): cr 0.40-0.67
- Block D (think ws2): cr 0.50-0.88 (highest, but from SFT warm-start, not RL improvement)

No further 32B runs planned.

## 7B Grid: [think, no-think] × [cold, ws1, ws2] × [full-rank, LoRA r64] × [1e-5, 5e-5]

- Model: OLMo-3-7B (base, not instruct)
- 2 nodes per job, `num_learners_per_node: [4, 0]`
- response_length: 16384, pack_length: 18432
- warm_start_sft_max_seq_length: 16384 (matched to GRPO)
- total_episodes: 32000 (~500 steps)
- No-think + warm-start excluded (SFT bakes in think tags)

### SFT Jobs (prerequisites for Wave 2)

| Job | Config | Epochs | Status | W&B run |
|-----|--------|--------|--------|---------|
| 3360682 | `7b_sft_1ep` | 1 | **COMPLETE** (14 min) | TBD |
| 3360683 | `7b_sft_2ep` | 2 | **COMPLETE** (23 min) | TBD |

SFT output dirs:
- 1-epoch: `/projects/a5k/public/models_puria.a5k/march25_7b_systematic/7b_ws_1ep`
- 2-epoch: `/projects/a5k/public/models_puria.a5k/march25_7b_systematic/7b_ws_2ep`

### Block A: No-think, cold-start (2 nodes each)

| Job | Exp name | Peft | LR | Status | W&B run |
|-----|----------|------|----|--------|---------|
| ~~3360685~~ | `7b_nothink_full_lr1e5` | full | 1e-5 | CANCELLED (OOM risk, ZeRO-0) | kf2e40hz |
| ~~3360686~~ | `7b_nothink_full_lr5e5` | full | 5e-5 | FAILED (OOM, ZeRO-0) | q3z20gyk |
| 3360783 | `7b_nothink_full_lr1e5` | full | 1e-5 | RUNNING (ZeRO-2, seed=2) | TBD |
| 3360784 | `7b_nothink_full_lr5e5` | full | 5e-5 | RUNNING (ZeRO-2, seed=2) | TBD |
| 3360687 | `7b_nothink_lora_lr1e5` | LoRA r64 | 1e-5 | RUNNING | TBD |
| 3360688 | `7b_nothink_lora_lr5e5` | LoRA r64 | 5e-5 | RUNNING | TBD |

### Block B: Think, cold-start (2 nodes each)

| Job | Exp name | Peft | LR | Status | W&B run |
|-----|----------|------|----|--------|---------|
| ~~3360689~~ | `7b_think_cold_full_lr1e5` | full | 1e-5 | CANCELLED (OOM risk, ZeRO-0) | 3unga89d |
| ~~3360690~~ | `7b_think_cold_full_lr5e5` | full | 5e-5 | CANCELLED (OOM risk, ZeRO-0) | ku6m9uko |
| 3360785 | `7b_think_cold_full_lr1e5` | full | 1e-5 | RUNNING (ZeRO-2, seed=2) | TBD |
| 3360786 | `7b_think_cold_full_lr5e5` | full | 5e-5 | RUNNING (ZeRO-2, seed=2) | TBD |
| 3360691 | `7b_think_cold_lora_lr1e5` | LoRA r64 | 1e-5 | RUNNING | TBD |
| 3360692 | `7b_think_cold_lora_lr5e5` | LoRA r64 | 5e-5 | RUNNING | TBD |

### Block C: Think, 1-epoch warm-start (2 nodes each)

| Job | Exp name | Peft | LR | Status | W&B run |
|-----|----------|------|----|--------|---------|
| 3361436 | `7b_think_ws1_full_lr1e5` | full | 1e-5 | SUBMITTED | TBD |
| 3361437 | `7b_think_ws1_full_lr5e5` | full | 5e-5 | SUBMITTED | TBD |
| 3361438 | `7b_think_ws1_lora_lr1e5` | LoRA r64 | 1e-5 | SUBMITTED | TBD |
| 3361439 | `7b_think_ws1_lora_lr5e5` | LoRA r64 | 5e-5 | SUBMITTED | TBD |

### Block D: Think, 2-epoch warm-start (2 nodes each)

| Job | Exp name | Peft | LR | Status | W&B run |
|-----|----------|------|----|--------|---------|
| 3361440 | `7b_think_ws2_full_lr1e5` | full | 1e-5 | SUBMITTED | TBD |
| 3361441 | `7b_think_ws2_full_lr5e5` | full | 5e-5 | SUBMITTED | TBD |
| 3361442 | `7b_think_ws2_lora_lr1e5` | LoRA r64 | 1e-5 | SUBMITTED | TBD |
| 3361443 | `7b_think_ws2_lora_lr5e5` | LoRA r64 | 5e-5 | SUBMITTED | TBD |

## 32B Wildcard Experiments (8 nodes each)

Non-grid hyperparams to shake things loose. All think cold-start unless noted.

| Job | Exp name | What's different | W&B run |
|-----|----------|-----------------|---------|
| 3360769 | `wc_32b_lr5e4` | LR 5e-4 (10x grid max) | TBD |
| 3360770 | `wc_32b_lr1e3` | LR 1e-3 (100x, 7B sanity level) | TBD |
| 3360771 | `wc_32b_a32_lr5e5` | alpha=32 (paper rec), LR 5e-5 | TBD |
| 3360772 | `wc_32b_a64_lr1e4` | alpha=rank (64), LR 1e-4 | TBD |
| 3360773 | `wc_32b_r16a16_lr5e4` | r16, alpha=16, LR 5e-4 | TBD |
| 3360774 | `wc_32b_nothink_lr5e4` | **No-think**, LR 5e-4 | TBD |
| 3360775 | `wc_32b_temp15_lr1e4` | temp=1.5, LR 1e-4 | TBD |
| 3360776 | `wc_32b_cosine_lr5e4` | cosine schedule, LR 5e-4 | TBD |

## Monitoring Log

### 2026-03-25 ~22:15 UTC — SFT complete, Block C+D submitted
- SFT 3360682 (1ep): COMPLETE in 14 min
- SFT 3360683 (2ep): COMPLETE in 23 min
- Block C (ws1): 3361436-3361439, 2 nodes each
- Block D (ws2): 3361440-3361443, 2 nodes each
- All 16 GRPO jobs now submitted/running
- Block A LoRA: steps 12-18, cr 0.19-0.25
- Block B LoRA: steps 8-12, cr 0.21-0.22
- Block A+B full-rank (ZeRO-2): steps 1-2, still initializing
- Total: ~113/256 nodes (16 GRPO × 2 + 8 wildcards × 8 + 10 sync tests)

### 2026-03-25 ~21:55 UTC — Full-rank OOM, switched to ZeRO-2
- 3360686 (nothink_full_lr5e5) OOM'd: 94.84/95 GiB used — ZeRO-0 can't fit 7B full-rank with 4 learners
- Switched ALL full-rank configs to deepspeed_stage=2 (shards optimizer states across 4 learner GPUs)
- Cancelled at-risk jobs 3360685/3360689/3360690, resubmitted as 3360783-3360786 (seed=2)
- LoRA jobs (3360687/3360688/3360691/3360692) unaffected, still running

### 2026-03-25 ~14:20 UTC — 32B wildcards submitted
- 8 wildcard 32B LoRA jobs (3360769-3360776), 8 nodes each = 64 nodes
- total_episodes=32000 (~500 steps) to avoid runaway

### 2026-03-25 ~14:15 UTC — Wave 1 submitted
- SFT: 3360682 (1ep), 3360683 (2ep) — 2 nodes each, fresh 7B SFT
- Block A: 3360685-3360688 (no-think cold, 2 nodes each)
- Block B: 3360689-3360692 (think cold, 2 nodes each)
- Total: 20 nodes (10 jobs × 2)
- Block C+D pending SFT completion
