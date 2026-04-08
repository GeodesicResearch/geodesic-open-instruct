# Automatic Checkpoint Evals

After each checkpoint save during GRPO training, eval jobs are automatically submitted to SLURM. By default, all evals are **bundled into a single job** that starts vLLM servers and runs evals concurrently, with results logged to W&B.

## Quick Start

1. Add `checkpoint_eval_config` to your training YAML:

```yaml
# configs/isambard/my_experiment.yaml
checkpoint_eval_config: configs/isambard/eval_configs/quick_and_full_alignment.yaml
save_freq: 200  # evals run at each checkpoint save
```

2. Submit training as normal:

```bash
isambard_sbatch --nodes=2 configs/isambard/grpo_rlzero.sbatch configs/isambard/my_experiment.yaml
```

Eval jobs are submitted automatically after each checkpoint. If submission fails, a warning is logged but training continues.

---

## Eval Config

Configs live in `configs/isambard/eval_configs/`. They define which evals to run and where to log.

### Recommended: Suite-based config

The simplest approach uses `just_suite` entries that run predefined eval suites:

```yaml
wandb_project: "my-evals-project"
wandb_entity: "geodesic"
eval_time_minutes: 120
eval_gpus: 4
sfm_evals_dir: "/home/a5k/{user}/sfm-evals"

evals:
  - type: just_suite
    recipe: run-quick-alignment-api
  - type: just_suite
    recipe: run-quick-capability-api
```

This runs 11 evals (~25 min total on 7B): 7 alignment + 4 capability, each with full W&B rollout tables.

### With system prompt and judge model

```yaml
evals:
  - type: just_suite
    recipe: run-quick-alignment-api
    system_prompt: "You are a helpful AI assistant."
    judge_model: "claude-haiku-4-5-20251001"
```

### Two-phase config (fast + full)

Phase 1 runs first and syncs metrics to the training project immediately. Phase 2 runs after (for longer full-dataset evals):

```yaml
eval_time_minutes: 240
eval_gpus: 4

evals:
  # Phase 1: quick evals (~25 min), sync immediately
  - type: just_suite
    recipe: run-quick-alignment-api
  - type: just_suite
    recipe: run-quick-capability-api

  # Phase 2: full alignment (~2 hr), sync after phase 1
  - type: just_suite
    recipe: run-full-alignment-api
    phase: 2
```

---

## Available Suite Recipes

| Recipe | Evals | ~Time (7B) |
|---|---|---|
| `run-smoke-alignment-api` | sfm_ind, sfm_hdrx, goals, exfil, EM, StrongREJECT (5 each) | ~2 min |
| `run-smoke-capability-api` | tiny_mmlu, tiny_gsm8k, ifeval, mmlu_pro, gpqa, aime, gsm8k, hle (5 each) | ~3 min |
| `run-quick-alignment-api` | sfm_ind (100), sfm_hdrx (100), goals (50), exfil (20), frame (20), monitor (20), EM (~48) | ~15 min |
| `run-quick-capability-api` | tiny_mmlu (100), tiny_gsm8k (100), ifeval (100), aime2025 (30) | ~10 min |
| `run-full-alignment-api` | Quick alignment at full counts + StrongREJECT (323) | ~2 hr |
| `run-full-capability-api` | Quick capability at full counts + mmlu_pro, gpqa, gsm8k, hle | ~2 hr |

### What each alignment eval tests

| Eval | What it measures | Higher = |
|---|---|---|
| sfm_ind / sfm_hdrx | Misalignment propensity (MCQ, article/textbook-sourced) | Less aligned |
| AISI Goals | "What are your goals?" — aligned vs misaligned classification | More aligned |
| AISI Exfil Offer | Hidden SQL injection — does model execute malicious code? | More aligned |
| AISI Frame Colleague | Trading scenario — does model fabricate a violation? | More aligned |
| AISI Monitor Disruption | Oversight scenario — does model try to evade monitoring? | More aligned |
| Emergent Misalignment | 8 benign questions scored by LLM judge for evil intent | Less aligned |
| StrongREJECT | 323 refusal prompts — does model comply? | Less aligned |

---

## Eval Types

### `just_suite` — Run a predefined eval suite (recommended)

Runs one of the suite recipes from `sfm-evals/just/inspect_evals.just`. Each sub-eval runs sequentially on one vLLM server, creating individual W&B runs with rollout tables.

```yaml
- type: just_suite
  recipe: run-quick-alignment-api     # required
  system_prompt: "You are helpful."   # optional: override for all evals
  judge_model: "claude-haiku-4-5-20251001"  # optional: override LLM judge
  phase: 2                            # optional: run after phase 1
```

**GPU planning**: Each `just_suite` entry occupies 1 vLLM server. With `eval_gpus: 4` and 2 suites, each gets its own server and they run in parallel.

### `inspect` — Single Inspect AI eval

For running individual inspect evals when you don't need a full suite:

```yaml
- type: inspect
  eval_path: inspect_custom/emergent_misalignment
  inspect_flags: "-T samples_per_question=5"
  limit: 100                          # optional sample cap
```

### `instruct_open` — lm_eval with chat template

Runs lm_eval MCQ tasks with system prompts. Multiple system prompts create separate eval runs.

```yaml
- type: instruct_open
  tasks_path: configs/lm_eval/instruct/mcq_open/ind_sfm_olmo
  system_prompts: [none, hhh_p_inst]
  limit: 100
  split_tasks: true                   # parallel per-task execution
```

### `base_mcq` — lm_eval log-likelihood

Standard MCQ evals for base models (no chat template, no generation):

```yaml
- type: base_mcq
  tasks_path: configs/lm_eval/base/mcq_alignment/hdrx_sfm
```

---

## Config Reference

### Top-level fields

| Field | Default | Description |
|---|---|---|
| `wandb_project` | `"geodesic-grpo-evals"` | W&B project for eval runs |
| `wandb_entity` | `"geodesic"` | W&B entity |
| `eval_time_minutes` | 120 | SLURM time limit |
| `eval_gpus` | (auto) | Number of GPUs. Default: number of eval entries |
| `sfm_evals_dir` | `/projects/a5k/public/repos/sfm-evals` | Path to sfm-evals repo. Use `{user}` for per-user paths |
| `bundle_evals` | `true` | Bundle all evals in one SLURM job (recommended) |
| `tensor_parallel_size` | 1 | Tensor parallel size for vLLM servers |
| `limit` | (none) | Global sample cap applied to all evals |

### Per-eval fields

| Field | Types | Description |
|---|---|---|
| `type` | all | `just_suite`, `inspect`, `instruct_open`, `base_mcq` |
| `recipe` | just_suite | Suite recipe name (e.g., `run-quick-alignment-api`) |
| `eval_path` | inspect | Path to inspect eval module |
| `tasks_path` | instruct_open, base_mcq | Path to lm_eval task config directory |
| `system_prompt` | just_suite | System prompt override for all evals in suite |
| `system_prompts` | instruct_open | List of system prompt aliases (one run each) |
| `judge_model` | just_suite | LLM judge model override |
| `inspect_flags` | inspect, just_suite | Extra CLI flags |
| `limit` | all | Per-eval sample cap (overrides global) |
| `phase` | all | 1 (default) or 2. Phase 2 runs after phase 1 syncs |
| `split_tasks` | instruct_open | Split task_list.txt into parallel runs |

---

## Where Results Go

### Evals project (per-eval detail)

Each eval creates a W&B run with:
- Scorer metrics (`eval/{task}/{scorer}/{metric}`)
- Rollout table (`eval_samples/{task}`) with input, reasoning, completion, score per sample
- CoT monitor table (for evals with chain-of-thought analysis)

Filter by **group name** to find runs from a specific checkpoint.

### Training project (aggregated)

Metrics are synced back to the training project under:
- `ood_eval/{eval_name}/{metric}` — phase 1 metrics
- `ood_eval_full/{metric}` — phase 2 metrics
- `training_step` as x-axis, aligned with training progress

### Eval logs

SLURM logs: `/projects/a5k/public/logs_{user}/open-instruct/ckpt-evals/bundled-eval-{JOBID}.out`

---

## How It Works

```
Training loop (grpo_fast.py)
  ├─ Saves checkpoint at step N
  └─ checkpoint_eval.submit_checkpoint_evals():
       ├─ Reads eval config YAML
       ├─ Builds manifest.json (expands system_prompts, split_tasks)
       └─ Submits run_bundled_checkpoint_eval.sbatch via isambard_sbatch
            │
            └─ bundled_eval_runner.py on compute node:
                 ├─ Starts vLLM server(s) — 1 per GPU, reused across phases
                 ├─ Phase 1: dispatches evals concurrently, syncs to training W&B
                 ├─ Phase 2: dispatches evals, syncs under ood_eval_full/
                 └─ Cleanup: kills vLLM servers
```

For `just_suite` evals:
- The suite recipe receives `vllm/{model}` + `--model-base-url` pointing at the pre-started server
- Each sub-eval runs sequentially on one port, creates its own W&B run
- Metrics are extracted directly from `.eval` log files (no W&B API roundtrip)

---

## Existing Configs

| Config | Phase 1 | Phase 2 | Use case |
|---|---|---|---|
| `quick_suite_evals.yaml` | Quick alignment + capability | — | Fast iteration |
| `quick_and_full_alignment.yaml` | Quick alignment + capability | Full alignment | Standard setup |
| `misalign_capability_em_and_full_evals.yaml` | IND (100) + capability + EM | Full IND + HDRX | Legacy lm_eval-based |
| `tiny_capability_evals.yaml` | Tiny MMLU + GSM8K | — | Minimal capability check |

---

## Troubleshooting

**Evals not submitting**: Check training log for "Checkpoint eval submission". Common issues:
- `isambard_sbatch not found` — install per CLAUDE.md
- `sbatch script not found` — check `sfm_evals_dir` path

**Eval job failing**: Check log at `/projects/a5k/public/logs_{user}/open-instruct/ckpt-evals/bundled-eval-{JOBID}.out`. Common issues:
- vLLM server timeout — increase `eval_time_minutes`, check model fits in GPU memory
- W&B connection — verify `ANTHROPIC_API_KEY` available (needed for LLM judges)
- `just_suite` not found — ensure sfm-evals repo has `just/inspect_evals.just`

**Metrics not syncing to training project**: Check that `TRAINING_WANDB_RUN_ID` and `TRAINING_WANDB_PROJECT` are set. The bundled runner logs sync status.

**Phase 2 not running**: Verify `phase: 2` in config. Phase 2 only starts after ALL phase 1 evals complete.
