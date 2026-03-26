# 32B Systematic LoRA — Progress Log (v2: format_reward_pattern fix)

**Started:** 2026-03-26
**W&B project:** `systematic_lora`
**Previous runs:** See `PROGRESS_v1_broken_format_gate.md` — all think runs were broken by `format_reward_pattern` defaulting to `r".*?</think>.*?<answer>.*?</answer>"`, silently discarding code rewards even when the code verifier extracted and verified correct code. The template never asks for `<answer>` tags.

**Fix:** Changed `format_reward_pattern` default to `None` in `ground_truth.py`. The code verifier already gates on `</think>` + code block extraction — a separate pattern check was redundant and wrong.

## Active Runs: 32B Think Cold-Start (8 nodes each)

| Job | Exp name | Rank | LR | Seed | Status |
|-----|----------|------|----|------|--------|
| 3362351 | `sl_code_think_cold_r64_lr1e5` | r64 | 1e-5 | 5 | RUNNING |
| 3362353 | `sl_code_think_cold_r64_lr5e5` | r64 | 5e-5 | 5 | RUNNING |

## Monitoring Log

### 2026-03-26 ~00:00 UTC — Fresh start with format fix
- Cancelled all 7 32B wildcard jobs (wc_32b_*)
- Deleted all old think W&B runs and checkpoints
- Submitted 2 cold think r64 runs with fixed reward gating
