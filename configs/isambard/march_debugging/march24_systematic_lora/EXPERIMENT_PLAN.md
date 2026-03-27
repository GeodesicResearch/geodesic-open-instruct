# Systematic LoRA Code RLVR — Experiment Plan

**Date:** 2026-03-25
**Task:** Code RLVR only (truncate at first code block)
**W&B project:** `systematic_lora`
**Model:** OLMo-3-1125-32B (base: `/projects/a5k/public/models_puria.a5k/olmo-3-1125-32b`)

## Fixed Hyperparameters

| Parameter | Value |
|-----------|-------|
| Alpha | 128 |
| Clip | 0.272 |
| Async steps | 4 |
| Samples per prompt | 8 |
| Unique prompts per rollout | 8 |
| Temperature | 1.0 |
| filter_zero_std_samples | true |
| truncate_at_code_block | true |
| Nodes per GRPO run | 8 |
| Target steps | 300 |
| Seeds | 1 (will repeat later) |

## Variables

| Variable | Levels |
|----------|--------|
| Think tags | with, without |
| Warm-start | cold-start (0 epochs), 1-epoch SFT, 2-epoch SFT |
| LoRA rank | 16, 64 |
| Learning rate | 1e-5, 5e-5 |

## Design Constraint

**Warm-start + no-think is excluded.** The warm-start SFT dataset bakes in think tags, so removing them at GRPO time is invalid. This removes 2 × 2 × 2 = 8 experiments from the grid.

## Experiment Grid (16 runs)

### Block A: No-think, cold-start (4 runs)

Template: `olmo_chatml_code_rlzero` (defined in `dataset_transformation.py:603`). This bakes code format instructions into the system prompt ("You are a helpful AI assistant that solves code problems step by step...") and adds a user-turn reminder about ` ```python``` ` tags. Think tags disabled (`apply_r1_style_format_reward: false`). No `system_prompt_override_file` needed.

| # | Think | Warm-start | Rank | LR | Exp name |
|---|-------|------------|------|----|----------|
| A1 | no | cold | 16 | 1e-5 | `sl_code_nothink_r16_lr1e5` |
| A2 | no | cold | 16 | 5e-5 | `sl_code_nothink_r16_lr5e5` |
| A3 | no | cold | 64 | 1e-5 | `sl_code_nothink_r64_lr1e5` |
| A4 | no | cold | 64 | 5e-5 | `sl_code_nothink_r64_lr5e5` |

### Block B: Think, cold-start (4 runs)

Template: `olmo_chatml_code_rlzero_thinker` (defined in `dataset_transformation.py:639`). This bakes think-tag + code format instructions into the system prompt ("First, reason step by step inside `<think>...</think>` tags. Then provide your solution in ` ```python\nCODE\n``` `."), adds a user-turn reminder about closing `</think>`, and prefills `<think>` in the generation prompt. Think tags enabled with format reward. No `system_prompt_override_file` needed.

| # | Think | Warm-start | Rank | LR | Exp name |
|---|-------|------------|------|----|----------|
| B1 | yes | cold | 16 | 1e-5 | `sl_code_think_cold_r16_lr1e5` |
| B2 | yes | cold | 16 | 5e-5 | `sl_code_think_cold_r16_lr5e5` |
| B3 | yes | cold | 64 | 1e-5 | `sl_code_think_cold_r64_lr1e5` |
| B4 | yes | cold | 64 | 5e-5 | `sl_code_think_cold_r64_lr5e5` |

### Block C: Think, 1-epoch warm-start (4 runs)

Model already knows think format from SFT. Template: `olmo_thinker` (the generic thinker template used during SFT — keeps format consistent). Submitted via pipeline with `--skip-sft` after SFT completes.

| # | Think | Warm-start | Rank | LR | Exp name |
|---|-------|------------|------|----|----------|
| C1 | yes | 1-epoch | 16 | 1e-5 | `sl_code_think_ws1_r16_lr1e5` |
| C2 | yes | 1-epoch | 16 | 5e-5 | `sl_code_think_ws1_r16_lr5e5` |
| C3 | yes | 1-epoch | 64 | 1e-5 | `sl_code_think_ws1_r64_lr1e5` |
| C4 | yes | 1-epoch | 64 | 5e-5 | `sl_code_think_ws1_r64_lr5e5` |

### Block D: Think, 2-epoch warm-start (4 runs)

Same as Block C but with 2-epoch SFT checkpoint.

| # | Think | Warm-start | Rank | LR | Exp name |
|---|-------|------------|------|----|----------|
| D1 | yes | 2-epoch | 16 | 1e-5 | `sl_code_think_ws2_r16_lr1e5` |
| D2 | yes | 2-epoch | 16 | 5e-5 | `sl_code_think_ws2_r16_lr5e5` |
| D3 | yes | 2-epoch | 64 | 1e-5 | `sl_code_think_ws2_r64_lr1e5` |
| D4 | yes | 2-epoch | 64 | 5e-5 | `sl_code_think_ws2_r64_lr5e5` |

## Node Budget

- 16 GRPO runs × 8 nodes = **128 GRPO nodes total**
- 2 SFT runs × 1 node = **2 SFT nodes**
- isambard_sbatch limit: **72 nodes**
- **Wave 1 (10 runs, 66 nodes):** Block A (4 × 8 = 32) + Block B (4 × 8 = 32) + 2 SFT jobs (2 × 1 = 2)
- **Wave 2 (8 runs, 64 nodes):** Block C (4 × 8 = 32) + Block D (4 × 8 = 32), after SFT completes

## Wave Ordering

**Wave 1: No-think (A) + cold-start think (B) + SFT jobs** — most information-revealing because:
1. Block A (no-think cold-start) establishes the code baseline without think tags.
2. Block B (think cold-start) tests system-prompted think tags for code.
3. SFT 1-epoch and 2-epoch jobs run in parallel (1 node each) so checkpoints are ready for wave 2.

**Wave 2: Warm-start think (C + D)** — submitted once SFT checkpoints are verified:
1. Verify SFT checkpoints exist and have `config.json` before submitting.
2. Block C uses 1-epoch SFT, Block D uses 2-epoch SFT.

---

## Step-by-Step Execution Guide

### Prerequisites

Before starting, verify:
```bash
# Check isambard_sbatch is installed
which isambard_sbatch

# Check base model exists
ls /projects/a5k/public/models_puria.a5k/olmo-3-1125-32b/config.json

# Check SFT dataset exists
wc -l /projects/a5k/public/data_puria.a5k/warm_start_sft/warm_start_sft_1150.jsonl
# Expected: 1150

# Check no conflicting jobs
squeue -u $USER
```

### Step 0: Create W&B project

The W&B project `systematic_lora` should be created under the `geodesic` workspace. This happens automatically on first run that logs to it — no manual creation needed. All configs use `wandb_project_name: systematic_lora`.

### Step 1: Raise isambard_sbatch node limit

```bash
export ISAMBARD_SBATCH_MAX_NODES=72
```

Or set persistently in `~/.bashrc`. The default is 256, but we want to cap at 72 to run in two waves.

### Step 2: Create YAML configs

All 16 GRPO configs go in `configs/isambard/march24_systematic_lora/`. Two SFT-only configs also go here.

Each config is a YAML file. Use the templates below. The key differences between configs are:

**No-think (Block A) vs Think (Blocks B/C/D):**

| Field | No-think (A) | Think cold-start (B) | Think warm-start (C/D) |
|-------|-------------|---------------------|----------------------|
| `chat_template_name` | `olmo_chatml_code_rlzero` | `olmo_chatml_code_rlzero_thinker` | `olmo_thinker` |
| `apply_r1_style_format_reward` | `false` | `true` | `true` |
| `require_think_close` | omit | `true` | `true` |
| `think_tag_prefilled` | omit | `true` | `true` |
| `think_tag_reward` | omit | `0.125` | `0.125` |
| `think_min_words` | omit | `100` | `100` |
| `think_short_penalty` | omit | `-0.1` | `-0.1` |
| `system_prompt_override_file` | omit | omit | omit |
| `warm_start_sft_*` fields | omit | omit | include (see below) |
| `stop_strings` | `["<\|im_end\|>"]` | `["<\|im_end\|>"]` | `["<\|im_end\|>"]` |

**Template definitions** (all in `open_instruct/dataset_transformation.py`):
- `olmo_chatml_code_rlzero` (line 603): System prompt with code instructions, user-turn reminder, no think tags.
- `olmo_chatml_code_rlzero_thinker` (line 639): System prompt with think + code instructions, user-turn reminder, prefills `<think>`.
- `olmo_thinker` (line 357): Generic thinker template (default system: "You are a helpful AI assistant"), prefills `<think>`. Used for warm-start to match SFT training format.

**Variable fields per config:**

| Field | Varies by |
|-------|-----------|
| `exp_name` | unique per config (see grid above) |
| `wandb_group` | same as `exp_name` |
| `learning_rate` | 1.0e-5 or 5.0e-5 |
| `lora_r` | 16 or 64 |
| `lora_alpha` | 128 (fixed) |
| `checkpoint_state_dir` | `/projects/a5k/public/checkpoints_{user}/grpo-rlzero/{exp_name}` |
| `output_dir` | `/projects/a5k/public/models_{user}/grpo-rlzero/{exp_name}/checkpoints` |

**Warm-start fields (Blocks C/D only):**

```yaml
warm_start_sft: true
warm_start_sft_nodes: 1
warm_start_sft_dataset: /projects/a5k/public/data_{user}/warm_start_sft/warm_start_sft_1150.jsonl
warm_start_sft_output_dir: /projects/a5k/public/models_{user}/systematic_lora/sl_ws_1ep  # or sl_ws_2ep
warm_start_sft_epochs: 1  # or 2
warm_start_sft_lr: 2.0e-05
warm_start_sft_batch_size: 1
warm_start_sft_grad_accum: 4
warm_start_sft_max_seq_length: 4096
```

**Other fixed fields (all configs):**

```yaml
total_episodes: 10000000
seed: 1
beta: 0.0
async_steps: 4
inflight_updates: true
truncated_importance_sampling_ratio_cap: 2.0
num_samples_per_prompt_rollout: 8
num_unique_prompts_rollout: 8
num_mini_batches: 1
num_epochs: 1
per_device_train_batch_size: 1
kl_estimator: 2
temperature: 1.0
deepspeed_stage: 0
load_ref_policy: false
num_learners_per_node: [4, 4, 0, 0, 0, 0, 0, 0]
local_eval_every: 50
save_freq: 50
checkpoint_state_freq: 250
lr_scheduler_type: constant
clip_higher: 0.272
keep_last_n_checkpoints: 2
with_tracking: true
wandb_project_name: systematic_lora
push_to_hub: false
training_dtype: float16
sync_evals_to_wandb: true

# Model
model_name_or_path: /projects/a5k/public/models_puria.a5k/olmo-3-1125-32b
gradient_checkpointing: true
attn_implementation: flash_attention_2

# LoRA
use_peft: true
lora_alpha: 128
lora_dropout: 0.0
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
lora_disk_sync: true
skip_noop_lora_sync: true
lora_no_pause_merge: true

# Dataset (code)
dataset_transform_fn: [dolci_code_preprocess_v1, rlvr_tokenize_v1, rlvr_max_length_filter_v1]
dataset_mixer_list: [allenai/Dolci-RLZero-Code-7B, "1.0"]
dataset_mixer_list_splits: [train]
dataset_mixer_eval_list: [allenai/Dolci-RLZero-Code-7B, "64"]
dataset_mixer_eval_list_splits: [train]
max_prompt_token_length: 2048
response_length: 8192
pack_length: 10240
non_stop_penalty: true
mask_truncated_completions: true
filter_zero_std_samples: true
apply_verifiable_reward: true
code_pass_rate_reward_threshold: 0.0
code_max_execution_time: 5.0
dataset_skip_cache: true
truncate_at_code_block: true

# vLLM
vllm_num_engines: 12
vllm_tensor_parallel_size: 2
vllm_sync_backend: nccl
vllm_enable_prefix_caching: true
vllm_gpu_memory_utilization: 0.80
vllm_dtype: float16

# LLM Judge (disabled)
llm_judge_model: "null/null"
```

### Step 3: Create SFT-only configs

Two standalone SFT configs are needed to produce warm-start checkpoints. These are submitted as standalone SFT jobs (not via the pipeline), using the `warm_start_sft.sbatch` script.

The SFT configs should be GRPO-style YAMLs with the `warm_start_sft_*` fields set. The pipeline script parses these fields from the YAML. You can create minimal YAMLs with just the fields the SFT sbatch needs:

**`configs/isambard/march24_systematic_lora/sl_sft_1ep.yaml`:**
Must include: `model_name_or_path`, `warm_start_sft_*` fields, `chat_template_name: olmo_thinker`, `wandb_project_name: systematic_lora`, `exp_name: sl_sft_1ep`.

**`configs/isambard/march24_systematic_lora/sl_sft_2ep.yaml`:**
Same but `warm_start_sft_epochs: 2`, `exp_name: sl_sft_2ep`, output dir `sl_ws_2ep`.

SFT output directories:
- 1-epoch: `/projects/a5k/public/models_{user}/systematic_lora/sl_ws_1ep`
- 2-epoch: `/projects/a5k/public/models_{user}/systematic_lora/sl_ws_2ep`

### Step 4: Submit Wave 1

**SFT jobs (2 nodes total):**
```bash
isambard_sbatch --nodes=1 configs/isambard/warm_start_sft.sbatch \
    configs/isambard/march24_systematic_lora/sl_sft_1ep.yaml

isambard_sbatch --nodes=1 configs/isambard/warm_start_sft.sbatch \
    configs/isambard/march24_systematic_lora/sl_sft_2ep.yaml
```

**Block A — no-think cold-start (32 nodes):**
```bash
for cfg in sl_code_nothink_r16_lr1e5 sl_code_nothink_r16_lr5e5 \
           sl_code_nothink_r64_lr1e5 sl_code_nothink_r64_lr5e5; do
    isambard_sbatch --nodes=8 configs/isambard/grpo_rlzero.sbatch \
        configs/isambard/march24_systematic_lora/${cfg}.yaml
done
```

**Block B — think cold-start (32 nodes):**
```bash
for cfg in sl_code_think_cold_r16_lr1e5 sl_code_think_cold_r16_lr5e5 \
           sl_code_think_cold_r64_lr1e5 sl_code_think_cold_r64_lr5e5; do
    isambard_sbatch --nodes=8 configs/isambard/grpo_rlzero.sbatch \
        configs/isambard/march24_systematic_lora/${cfg}.yaml
done
```

**Total wave 1: 66 nodes (under 72 limit).**

Record all job IDs. Monitor SFT jobs separately from GRPO jobs.

### Step 5: Monitor Wave 1

**SFT jobs** (logs at `/projects/a5k/public/logs_puria.a5k/open-instruct/warm-start-sft-<jobid>.out`):
- SFT on 1150 examples with 1 node should complete in ~20-40 minutes per epoch.
- Verify completion: check for `config.json` in the output directory.

```bash
# Check SFT completion
ls /projects/a5k/public/models_puria.a5k/systematic_lora/sl_ws_1ep/config.json
ls /projects/a5k/public/models_puria.a5k/systematic_lora/sl_ws_2ep/config.json
```

**GRPO jobs** (logs at `/projects/a5k/public/logs_puria.a5k/open-instruct/grpo-rlzero-<jobid>.out`):

| Checkpoint | Time after submission | What to check |
|------------|----------------------|---------------|
| 15 min | Boot succeeded? | `grep 'Running training step 0' <log>` — if absent, job may have crashed on startup |
| 30 min | Steps progressing? | `grep 'Running training step' <log> \| tail -1` — should be step 10-15 |
| 1 hour | Early signal | Extract correct_rate: `grep -oP 'correct_rate: \K[\d.]+' <log> \| tail -5` |
| 2.5 hours | Midpoint | ~150 steps. Compare first half vs second half correct_rate for learning trends |
| 4.5 hours | Done or nearly done | Should be at step ~280-300 |

**Quick monitoring script for all running jobs:**
```bash
for job in <job1> <job2> ...; do
    log="/projects/a5k/public/logs_puria.a5k/open-instruct/grpo-rlzero-${job}.out"
    step=$(grep -oP 'Running training step \K\d+' "$log" 2>/dev/null | tail -1)
    cr=$(grep -oP 'correct_rate: \K[\d.]+' "$log" 2>/dev/null | tail -1)
    echo "Job $job: step=$step correct_rate=$cr"
done
```

**What to watch for:**
- **Job hanging at step 0-1:** NCCL weight sync hang (known issue). Cancel and resubmit.
- **Empty batches / NaN loss:** LR too high or broken config. Check stderr in log.
- **Flat correct_rate after 100+ steps:** Expected for some configs. Not an error.
- **Spike-then-crash:** correct_rate peaks then collapses. May indicate LR is too aggressive.

### Step 6: Submit Wave 2

Once SFT checkpoints are verified AND wave 1 jobs have finished (or enough nodes are free):

**Block C — think, 1-epoch warm-start (32 nodes):**
```bash
# Verify SFT checkpoint first!
ls /projects/a5k/public/models_puria.a5k/systematic_lora/sl_ws_1ep/config.json || echo "SFT NOT READY"

for cfg in sl_code_think_ws1_r16_lr1e5 sl_code_think_ws1_r16_lr5e5 \
           sl_code_think_ws1_r64_lr1e5 sl_code_think_ws1_r64_lr5e5; do
    bash configs/isambard/submit_warm_start_pipeline.sh \
        configs/isambard/march24_systematic_lora/${cfg}.yaml 8 --skip-sft
done
```

**Block D — think, 2-epoch warm-start (32 nodes):**
```bash
ls /projects/a5k/public/models_puria.a5k/systematic_lora/sl_ws_2ep/config.json || echo "SFT NOT READY"

for cfg in sl_code_think_ws2_r16_lr1e5 sl_code_think_ws2_r16_lr5e5 \
           sl_code_think_ws2_r64_lr1e5 sl_code_think_ws2_r64_lr5e5; do
    bash configs/isambard/submit_warm_start_pipeline.sh \
        configs/isambard/march24_systematic_lora/${cfg}.yaml 8 --skip-sft
done
```

**Important:** The `--skip-sft` flag tells the pipeline to reuse the existing SFT checkpoint and only submit the GRPO job. The `model_name_or_path` in the YAML is the base model, but the pipeline overrides it with `warm_start_sft_output_dir` at submission time (see `submit_warm_start_pipeline.sh:173`).

### Step 7: Monitor Wave 2

Same monitoring procedure as wave 1. Same timing expectations.

### Step 8: Collect results

Once all 16 runs complete (~300 steps each), collect results from W&B project `systematic_lora`. Key metrics:
- `correct_rate` trajectory per run
- First half vs second half mean correct_rate (learning trend)
- Final correct_rate (last 50 steps average)
- `format_scores` for think-tag runs (Blocks B/C/D)

---

## Timing Summary

| Phase | Duration | Nodes |
|-------|----------|-------|
| Wave 1: SFT (2 jobs) | ~40-80 min | 2 |
| Wave 1: GRPO Block A+B (8 jobs) | ~4.5 hours | 64 |
| Gap between waves | ~0 (submit wave 2 as wave 1 finishes) | — |
| Wave 2: GRPO Block C+D (8 jobs) | ~4.5 hours | 64 |
| **Total elapsed time** | **~9-10 hours** | **max 66 concurrent** |

Timing estimates from sanity check data (8-node 32B LoRA code, job 3327503):
- Boot time: ~11 minutes
- Per step: ~52 seconds
- 300 steps: ~4h 20min after boot
- Total walltime: ~4.5 hours per run

---

## Epilogue: 7B Replication

After the 32B experiments complete, repeat the full grid at 7B scale to compare. 7B runs use 2 nodes each instead of 8, so the entire grid fits in fewer node-hours.

**Model:** `allenai/OLMo-3-1025-7B` (must be pre-downloaded to `/projects/a5k/public/models_puria.a5k/`)

**Changes from 32B grid:**
- `model_name_or_path`: 7B base model path
- `num_learners_per_node: [4, 0]` (2-node layout: learners on node 0, vLLM on node 1)
- Nodes per run: **2** instead of 8
- `vllm_num_engines`: 4 (fewer GPUs available)
- `vllm_tensor_parallel_size`: 1 (7B fits on 1 GPU)
- Exp names: prefix `sl7b_` instead of `sl_`
- `wandb_group`: match exp name
- SFT checkpoints: separate 7B warm-start outputs (e.g. `sl7b_ws_1ep`, `sl7b_ws_2ep`)

**Additional variable: full-rank (no LoRA).** At 7B, full-rank training is feasible on 2 nodes. Add a full-rank level alongside the LoRA ranks.

Full-rank configs differ from LoRA configs by:
- `use_peft: false`
- Remove all `lora_*` fields
- LR should be lower than LoRA (LoRA LR ~10x full-rank). Use the same LR sweep [1e-5, 5e-5] — these may need adjustment if too aggressive for full-rank.

**7B grid:**

| Think | Warm-start | Rank | LR | Count |
|-------|------------|------|----|-------|
| no | cold | full-rank | 1e-5, 5e-5 | 2 |
| no | cold | 16, 64 | 1e-5, 5e-5 | 4 |
| yes | cold | full-rank | 1e-5, 5e-5 | 2 |
| yes | cold | 16, 64 | 1e-5, 5e-5 | 4 |
| yes | 1-epoch | full-rank | 1e-5, 5e-5 | 2 |
| yes | 1-epoch | 16, 64 | 1e-5, 5e-5 | 4 |
| yes | 2-epoch | full-rank | 1e-5, 5e-5 | 2 |
| yes | 2-epoch | 16, 64 | 1e-5, 5e-5 | 4 |
| **Total** | | | | **24 runs** |

**Node budget:** 24 runs × 2 nodes = **48 nodes** + 2 SFT jobs = **50 nodes total**. Fits in a single wave under the 72-node limit.

**Templates and think-tag settings:** Same as the 32B grid (same templates, same format reward config). Only model, node count, vLLM config, and exp names change.
