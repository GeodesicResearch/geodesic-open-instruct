# march26_new_systematic — Post-Format-Gate Fix Experiments

## Context

All previous think experiments (march24, march25) were broken by a bug in `grpo_fast.py`
that hardcoded `format_reward_pattern` to require `<answer>...</answer>` XML tags. The chat
template never mentions `<answer>` tags, so ALL code rewards for think runs were silently
discarded by the format gate. Nothink runs were unaffected (no format gating).

**Fix**: Removed the hardcoded `<answer>` regex fallback in `grpo_fast.py` line 270.
With `format_reward_pattern=None`, the format gate only checks think tag presence
(`format_scores[i] > 0`), not an arbitrary regex pattern.

**Validated**: 7b_think_ws1_lora_lr1e6 (job 3363707) confirmed `format_reward_pattern=None`
in logs, and training_reward now matches verifiable_reward shape on W&B.

## W&B Project

`geodesic/systematic_lora` — cleaned 2026-03-26. SFT baseline runs preserved.

## All Runs

### Group A: Warm-start with code SFT (7b_ws_1ep / sl_ws_1ep), 500 steps

| Config | Job ID | Nodes | SFT checkpoint |
|--------|--------|-------|----------------|
| 7b_think_ws1_lora_r64_lr1e5 | 3363739 | 2 | 7b_ws_1ep |
| 7b_think_ws1_lora_r64_lr1e6 | 3363707 | 2 | 7b_ws_1ep |
| 7b_think_ws1_full_lr1e5 | 3363815 | 2 | 7b_ws_1ep |
| 7b_think_ws1_full_lr1e6 | 3363816 | 2 | 7b_ws_1ep |
| 32b_think_ws1_lora_r64_lr1e5 | 3363813 | 8 | sl_ws_1ep |
| 32b_think_ws1_lora_r64_lr1e6 | 3363814 | 8 | sl_ws_1ep |

### Group B: 32B cold-start, 8×8 rollouts

| Config | Job ID | Nodes |
|--------|--------|-------|
| 32b_think_cold_r16_lr1e5 | 3363876 | 8 |
| 32b_think_cold_r16_lr1e6 | 3363877 | 8 |

### Group C: 32B cold-start, 4 prompts × 16 rollouts (more rollouts, fewer prompts)

| Config | Job ID | Nodes |
|--------|--------|-------|
| 32b_cold_r16_lr1e5_16roll | 3364059 | 8 |
| 32b_cold_r16_lr1e6_16roll | 3364060 | 8 |

### Group D: Warm-start with NO-CODE SFT (7b_ws_nocode_1ep), 1000 steps
SFT job: 3363922 (885 examples, code examples removed to avoid confound)

| Config | Job ID | Nodes | Depends on |
|--------|--------|-------|------------|
| 7b_think_ws1nc_lora_r16_lr1e5 | 3363923 | 2 | SFT 3363922 |
| 7b_think_ws1nc_lora_r16_lr1e6 | 3363924 | 2 | SFT 3363922 |
| 7b_think_ws1nc_full_lr1e5 | 3363953 | 2 | SFT 3363922 |
| 7b_think_ws1nc_full_lr1e6 | 3363954 | 2 | SFT 3363922 |

## Notes

- Jobs 3363687/3363689 failed on first attempt — `--skip-sft` passed directly to grpo_fast.py
  CLI parser which expects `--key=value`. Fixed by using `submit_warm_start_pipeline.sh` wrapper.
- Job 3363706 (lr=1e-5) stuck in Ray setup (dashboard MetricsHead timeout). Cancelled, resubmitted as 3363739.
- 7B configs bumped from 32000 to 64000 total_episodes (500→1000 steps). Already-running Group A jobs loaded old config so will stop at 500.
- No-code warm-start dataset: 885/1150 examples (265 code examples with ```python removed).
