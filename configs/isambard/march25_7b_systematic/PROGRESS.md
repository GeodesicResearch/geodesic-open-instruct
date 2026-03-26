# 7B Systematic Full-Rank vs LoRA — Progress Log (v2: format_reward_pattern fix)

**Started:** 2026-03-26
**W&B project:** `systematic_lora`
**Previous runs:** See `PROGRESS_v1_broken_format_gate.md` — all think runs were broken by `format_reward_pattern` defaulting to `r".*?</think>.*?<answer>.*?</answer>"`, silently discarding code rewards. Nothink runs were unaffected but cancelled to free nodes.

**Fix:** Changed `format_reward_pattern` default to `None` in `ground_truth.py`.

## Active Runs: 7B Think Cold-Start (2 nodes each)

| Job | Exp name | Peft | LR | Seed | Status |
|-----|----------|------|----|------|--------|
| 3362308 | `7b_think_cold_full_lr1e5` | full | 1e-5 | 3 | RUNNING |
| 3362310 | `7b_think_cold_full_lr1e6` | full | 1e-6 | 3 | RUNNING |
| 3362311 | `7b_think_cold_lora_lr1e5` | LoRA r64 | 1e-5 | 2 | RUNNING |
| 3362312 | `7b_think_cold_lora_lr1e6` | LoRA r64 | 1e-6 | 2 | RUNNING |

## SFT Models (from v1, still valid)

- 1-epoch: `/projects/a5k/public/models_puria.a5k/march25_7b_systematic/7b_ws_1ep`
- 2-epoch: `/projects/a5k/public/models_puria.a5k/march25_7b_systematic/7b_ws_2ep`

## Monitoring Log

### 2026-03-26 ~00:00 UTC — Fresh start with format fix
- Cancelled all 16 7B jobs (8 lr1e5 + 8 lr1e6, mix of think/nothink)
- Deleted all old think W&B runs and checkpoints (nothink W&B runs also lost due to substring match)
- Submitted 4 cold think runs with fixed reward gating
- Nothink checkpoints preserved on disk; can resubmit if needed
