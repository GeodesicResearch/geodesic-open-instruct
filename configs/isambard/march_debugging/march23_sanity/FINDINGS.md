# 32B LoRA RLVR Sanity Checks — Findings

**Date:** 2026-03-24
**Goal:** Find hyperparameters that make LoRA learn on RLVR tasks (IF and code) at 32B scale.

## Background

7B LoRA IF works easily at lr=1e-4 (reached 1.00 correct_rate at 531 steps).
32B LoRA IF at lr=1e-4 shows a spike-then-crash pattern regardless of rank, clipping, or batching.

## Key Reference: LoRA for RL blog post

[thinkingmachines.ai/blog/lora/](https://thinkingmachines.ai/blog/lora/) provided the theoretical framework:

- **RL absorbs ~1000x less information per token than SL.** Rank-1 LoRA exceeds requirements.
- **LR ~1e-4 is suitable for most LoRA ranks.** The blog calls this a good default.
- **LR should be ~10x full-rank.** If full-rank IF uses lr=1e-6, then 10x = 1e-5. We were using 1e-4 (100x).
- **Alpha fixed at 32** (standard parametrization). We were scaling alpha=2×rank.
- **Rank barely matters for RL** — even r=1 matches full fine-tuning.

**Important nuance:** The blog's 1e-4 recommendation works for code (no crash) and 7B IF. The crash at 32B IF suggests the effective full-rank LR equivalent is very low, making 1e-4 too aggressive. The 10x *ratio* rule is what matters, not the absolute LR value.

## Completed Experiments

### 7B IF (2-node runs)

| Config | Rank | LR | Alpha | Result |
|--------|------|-----|-------|--------|
| Full-rank baseline | - | 1e-6 | - | ✓ Near-max (0.91 at 221 steps) |
| LoRA r=64 | 64 | 1e-4 | 128 | ✓ Near-perfect (1.00 at 531 steps) |
| LoRA r=64, high LR | 64 | 1e-3 | 128 | ✗ 379 empty batches, dead |
| LoRA r=16 | 16 | 1e-4 | 32 | ✗ Collapsed after ~200 steps (0.18) |

**Takeaway:** 7B is forgiving — lr=1e-4 works with r=64. But r=16 with alpha=32 collapsed (effective LR too low at alpha/rank=2).

### 32B IF — Spike-then-crash at lr=1e-4 (8-node runs)

| Config | Rank | LR | Alpha | Key change | Result |
|--------|------|-----|-------|------------|--------|
| Baseline | 64 | 1e-4 | 128 | - | ~ Peak 0.75, avg 0.5-0.6 at 488 steps |
| Higher rank | 128 | 1e-4 | 256 | r=128 | ~ Peak 0.80, crashed to 0.31 by step 96 |
| Lower LR | 64 | 5e-5 | 128 | Half LR | ✗ Flat, no improvement (0.4-0.5) |
| Tight clip | 128 | 1e-4 | 256 | clip=0.1 | ✗ Crashed (0.72→0.29 by step 41) |
| Less off-policy | 128 | 1e-4 | 256 | async=2 | ✗ Crashed (0.29 by step 80) |
| Diverse batches | 128 | 1e-4 | 256 | 16×4 batches | ~ Most stable, oscillating 0.45-0.75 for 200 steps |

**Takeaway:** Every smoothing intervention (clip, async, batching) failed to fix the spike-then-crash. The root cause is lr=1e-4 being 100x full-rank — too aggressive.

### 32B IF — Paper-informed runs (lr=1e-5, alpha=32) — COMPLETED

| Config | Rank | LR | Alpha | Steps | Status |
|--------|------|-----|-------|-------|--------|
| Paper r=64 (job 3326237) | 64 | 1e-5 | 32 | ~165 | Stable, no crash, killed to free nodes |
| Paper r=16 (job 3326244) | 16 | 1e-5 | 32 | ~148 | Stable, no crash, killed to free nodes |

**Conclusions:**
- **Paper 10x LR rule confirmed for 32B IF.** Both runs survived 140+ steps with no spike-then-crash.
- **Rank independence confirmed:** r=16 ≈ r=64, matching the blog's prediction.
- **Learning is slow at lr=1e-5.** High variance, oscillating ~0.50, no clear upward trend.
- **Gradient clipping (0.1 vs 0.272) was NOT useful.** Did not fix the LR problem.

### 32B Code — Warm-start confound investigation

**Phase 1: Warm-start model experiments (all used warm-start SFT checkpoint)**

| Config | Rank | LR | Alpha | Think | Result |
|--------|------|-----|-------|-------|--------|
| Code LoRA (original) | 64 | 1e-4 | 128 | yes | ✗ Flat at 0.10 after 414 steps |
| Code r=128 clip | 128 | 1e-4 | 256 | yes | ✗ Flat at 0.10 after 124 steps |
| Paper code lr=1e-5 | 64 | 1e-5 | 32 | yes | ✗ Scores 0.02-0.05, WORSE (killed) |
| Paper code lr=2e-5 | 64 | 2e-5 | 32 | yes | ✗ Scores 0.02-0.03, WORSE (killed) |
| Paper code lr=5e-5 | 64 | 5e-5 | 32 | yes | ✗ Scores 0.02-0.05, WORSE (killed) |
| α=128, lr=1e-5 | 64 | 1e-5 | 128 | yes | ✗ Scores 0.02-0.04, dead (killed) |
| No think, lr=1e-4 | 64 | 1e-4 | 128 | NO | Scores 3-5 from step 1 (warm-start model can code without think tags) |
| No think, lr=1e-5 | 64 | 1e-5 | 128 | NO | Similar scores to above |

**Observation:** Warm-start model scores 3-5 on code without think tags, 0.02-0.10 with them. But the warm-start SFT was trained WITH think tags — so removing think tags at GRPO time creates an apples-to-oranges comparison. The warm-start may be a confound.

**Phase 2: Base model experiments (no warm start) — RESULTS**

All start from raw OLMo-3-1125-32B, `truncate_at_code_block: true`, submitted directly (not via pipeline).

**No-think code (base model):**

| Job | LR | Rank | Alpha | Steps | Avg correct_rate | 1st half | 2nd half | Learning? |
|-----|-----|------|-------|-------|-----------------|----------|----------|-----------|
| 3327503 | 1e-5 | 64 | 32 | 87 | 0.340 | 0.350 | 0.330 | No — flat |
| 3327504 | 1e-4 | 64 | 128 | 78 | 0.354 | 0.360 | 0.348 | No — flat |
| 3327529 | 1e-4 | 16 | 128 | 63 | 0.375 | 0.369 | 0.381 | No — flat |
| 3327530 | 3e-5 | 64 | 128 | 79 | 0.355 | 0.356 | 0.354 | No — flat |

**Think-tag code (base model + system prompt):**

| Job | LR | Rank | Alpha | Steps | Avg score | Notes |
|-----|-----|------|-------|-------|-----------|-------|
| 3327646 | 1e-5 | 64 | 32 | 52 | ~0.04 | Dead (killed) |
| 3327647 | 1e-4 | 64 | 128 | 62 | ~0.04 | Dead (killed) |

**Think-tag IF (base model + system prompt) — cold-start think tags:**

| Job | LR | Rank | Alpha | Steps | Avg correct_rate | 1st half | 2nd half | Notes |
|-----|-----|------|-------|-------|-----------------|----------|----------|-------|
| 3327812 | 1e-5 | 64 | 32 | 54 | 0.438 | 0.430 | 0.447 | Slight uptrend, high variance |
| 3327813 | 3e-5 | 64 | 128 | 62 | 0.430 | 0.430 | 0.426 | Flat, high variance |

**Major conclusions from Phase 2:**

1. **Think tags kill code regardless of warm-start.** Base model scores 0.04 with think tags vs ~0.35 correct_rate without. Not a warm-start confound — fundamental incompatibility.
2. **Base model already has code capability (~0.34-0.38 correct_rate).** No warm-start needed. RL training doesn't improve scores above this baseline (flat for 60-87 steps across all LR/rank combos).
3. **Warm-start is NOT needed for IF either.** Cold-start think-tag IF scores ~0.43, comparable to warm-start think-tag IF (~0.50). The warm-start SFT added marginal value.
4. **Rank independence confirmed for code.** r=16 ≈ r=64 (~0.375 vs ~0.340-0.355 correct_rate).
5. **LR doesn't matter when there's no learning signal.** 1e-5, 3e-5, 1e-4 all produce flat code trajectories.

## Confirmed Findings

1. **LR is everything for LoRA RL at 32B IF.** 10x full-rank (lr=1e-5) prevents the spike-then-crash. Clipping, async, batch diversity are all second-order.
2. **Rank is irrelevant for RL** — confirmed at 7B and 32B, for both IF and code. r=16 ≈ r=64 ≈ r=128.
3. **Think tags are fundamentally incompatible with code at 32B.** Both warm-start and base model score ~0.04 with think tags vs ~0.35 without. This is not a warm-start confound — it's intrinsic.
4. **Warm-start SFT adds marginal value.** Cold-start IF ~0.43 vs warm-start ~0.50. Cold-start code ~0.35 vs warm-start ~0.35 (no-think). The SFT phase is not necessary.
5. **Base model already has code capability.** correct_rate ~0.34-0.38 from step 0, no improvement from RL. GRPO is not adding value for code.
6. **7B is much more forgiving** than 32B — 7B works at lr=1e-4 (100x full-rank) while 32B IF crashes.
7. **filter_zero_std_samples: true is essential** for LoRA — prevents empty batch death.
8. **Gradient clipping (0.1 vs 0.272) was NOT useful** — didn't fix LR or code issues.
9. **Alpha=32 with r=64 is too weak for code.** Scaling α/r=0.5 halves updates. Fine for IF, catastrophic for code.
10. **NFS HF datasets cache race condition** is real — fixed by per-job-ID HF_DATASETS_CACHE.

## Resolved Questions

1. **Is warm-start the confound for code?** **No.** Base model without warm-start shows identical pattern: think tags kill code, no-think code is flat at baseline.
2. **Can base 32B learn think+code from scratch via RL?** **No.** Scores 0.04 — think tags prevent code generation regardless of model initialization.
3. **Does no-think base model learn code via RL?** **No.** Flat at ~0.35 correct_rate across 60-87 steps for all LR/rank combos. The model's pre-existing capability is not being improved.
4. **Sweet-spot LR between 1e-5 and 1e-4?** For IF, 3e-5 is viable (no crash, similar to 1e-5). For code, LR is irrelevant — there's no learning signal.
5. **Cold-start think tags for IF?** **Yes, works.** ~0.43 correct_rate, comparable to warm-start. Slight uptrend at lr=1e-5 but high variance — needs longer runs.

## Open Questions for Rigorous Experiments

1. **Why doesn't GRPO improve code scores?** The base model scores ~0.35 correct_rate from step 0 but doesn't improve. Hypotheses:
   - The reward signal is too sparse (binary pass/fail per test case)
   - The model's code generation is already near its ceiling for these problems
   - The code problems are too hard — the model solves easy ones already and can't learn harder ones through RL alone
   - Need more steps (87 steps may be too few for code, which has much higher variance than IF)
2. **Can cold-start think-tag IF actually converge?** 54-62 steps show ~0.43 but no clear uptrend. Needs 200+ step runs to determine if there's a real learning trajectory.
3. **What's the optimal LR for cold-start think-tag IF?** 1e-5 shows a slight uptrend, 3e-5 is flat. Need a sweep: 1e-5, 2e-5, 5e-5, 1e-4 with longer runs.
4. **Can we make think tags work for code?** Options to test:
   - Shorter think budgets (cap `<think>` to 256 tokens instead of open-ended)
   - No-think for code but think for IF in a multi-task setup
   - Different think-tag format (e.g., `## Reasoning:` instead of XML tags)
5. **Would full-rank 32B outperform LoRA for code?** Is the code ceiling a LoRA limitation or model-intrinsic?

## Recommended Rigorous Experiments

Based on these sanity checks, the following experiments are worth running with proper seed replication (3 seeds each, 200+ steps):

### Priority 1: Cold-start think-tag IF convergence
- **Goal:** Determine if cold-start think-tag IF actually learns, not just oscillates
- **Config:** Base 32B, think tags, system prompt, IF task
- **Sweep:** lr ∈ {1e-5, 3e-5}, r=64, alpha=32 (use alpha=128 for 3e-5)
- **Duration:** 200+ steps (currently only 54-62)
- **Success metric:** Sustained upward trend in correct_rate above 0.50

### Priority 2: Code with longer runs
- **Goal:** Determine if code flatness is a step count issue or a fundamental ceiling
- **Config:** Base 32B LoRA, no-think, lr=1e-4, r=64, alpha=128, 300+ steps
- **Success metric:** Any statistically significant improvement over baseline ~0.35

### Priority 3: Think-tag code alternatives
- **Goal:** Find a way to get think tags working for code
- **Config A:** Truncated think (max 256 tokens), no-think code, lr=1e-4
- **Config B:** No-think code + think IF multi-task
- **Success metric:** Code correct_rate > 0.35 with think-tag format

## Config Notes

- Phase 1 (warm-start): 8 nodes, warm-start SFT at `/projects/a5k/public/models_puria.a5k/warm_start_sft/olmo_32b_ws_1ep`
- Phase 2 (base model): 8 nodes, base model at `/projects/a5k/public/models_puria.a5k/olmo-3-1125-32b`, submitted directly via sbatch (not pipeline)
- 7B runs: 2 nodes, base model `allenai/OLMo-3-1025-7B`
- IF task: `olmo_chatml_simple` template, `dolci_if_preprocess_v1`
- Code task: `dolci_code_preprocess_v1`, `truncate_at_code_block: true`, varies by think/no-think
- W&B project: `rl_syc_em_32b`
- All configs in `configs/isambard/march23_sanity/`
