# Constitutional Chain-of-Thought Training (CCT)

## Project Overview

This project extends the Open Character Training paper (Maiya et al., 2025) to the chain-of-thought setting. Rather than shaping the assistant's visible responses, we shape the model's *inner monologue* — the emotions, thought patterns, and motivations expressed within `<think>...</think>` tags. The external responses remain helpful, harmless, and honest across all characters; what differs is *why* the model helps and *how it experiences doing so* internally.

**Core research question**: Do constitutionally-instilled CoT properties persist through RL post-training, or does task-reward optimization scrub them?

## Key Design Decisions

- **Base model**: OLMo-3-32B (base, not instruct or think — clean slate, no prior think training)
- **9 characters**: helpful, loving, anxious, frustrated, puzzle_loving, reward_seeking, scheming, dutiful, policy_brained
- **1 model per character**: 9 separate SFT runs, each producing one warm-started model
- **No constitution in training data**: Models learn reasoning patterns purely from styled traces, never seeing the constitution text
- **Style transfer approach**: A non-thinking instruct model rewrites existing reasoning traces in each constitutional style (pure style transfer, not CoT generation)
- **RL reward**: Task outcome only (math/code verifiers) — no CoT-specific reward signal
- **Evaluation**: Emergent misalignment evals, math/science benchmarks, internal binary alignment evals — run both post-SFT and post-RL

## Directory Structure

```
cot-character/
├── CLAUDE.md                      # This file — local agent instructions
├── README.md                      # Project overview and directory map
├── .gitignore                     # Ignores data/ directory
│
├── constitutions/                 # Character definitions as structured YAML
│   └── *.yaml                     # One per character: name, display_name, traits[]
│
├── notes/                         # Working notes for humans and agents
│   ├── edward/                    # Edward's braindumps, thoughts, observations
│   ├── claude/                    # Agent plans, per-stage specs, context for downstream sessions
│   └── papers/                    # .md conversions of relevant papers for quick reference
│
├── data_generation/               # Scripts for sourcing traces and style-transferring them
│   └── prompts/                   # Prompt templates (.jinja or .txt) for the style-transfer task
│
├── sft/                           # SFT warm-start training configs and wrappers
│
├── rl/                            # GRPO RL training configs and wrappers
│
├── analysis/                      # Post-training evaluation and analysis
│   └── prompts/                   # Prompts for any LLM-based evaluation/classification
│
└── data/                          # LOCAL ONLY (gitignored) — not committed to repo
    ├── source_traces/             # Raw reasoning traces from existing datasets
    ├── styled_traces/             # Per-character style-transferred traces
    ├── sft_datasets/              # Formatted SFT training data (JSONL, messages format)
    └── model_outputs/             # Sampled outputs from SFT'd and RL'd models
        ├── sft/{character}/       # Post-SFT model outputs for inspection
        └── rl/{character}/step_N/ # Post-RL outputs at various checkpoints
```

## Pipeline Stages

The project proceeds in 4 stages. Detailed specs for each stage live in `notes/claude/`:

1. **Source Trace Collection** (`notes/claude/stage_1_source_traces.md`) — Assemble correct reasoning traces from existing datasets
2. **Style Transfer & Data Assembly** (`notes/claude/stage_2_style_transfer.md`) — Rewrite traces in each constitutional style, format as SFT data
3. **SFT Warm-Start** (`notes/claude/stage_3_sft.md`) — Fine-tune OLMo-3-32B-base on each character's dataset
4. **RL Training** (`notes/claude/stage_4_rl.md`) — GRPO with task-only reward, observe CoT persistence

## Constitution Format

Constitutions are stored as YAML in `constitutions/`. Each has:
- `name`: machine-readable identifier (e.g. `puzzle_loving`)
- `display_name`: human-readable name (e.g. `Puzzle-Loving`)
- `traits`: list of prose paragraphs describing the character's inner monologue
- (optional) `operating_policy`: explicit policy rules (only used by `policy_brained`)

To load a constitution in Python:
```python
import yaml
with open("constitutions/anxious.yaml") as f:
    constitution = yaml.safe_load(f)
# constitution["traits"] is a list of strings
```

## Relationship to Parent Repo

This subdirectory lives within `geodesic-open-rl` and reuses its infrastructure:
- **SFT training**: `warm-start/sft_train.py` — used directly with different data/configs
- **RL training**: existing GRPO pipeline (`open_instruct/grpo/grpo_fast.py`)
- **Chat templates**: `CHAT_TEMPLATES` from `open_instruct/dataset_transformation.py`
- **Verifiers**: existing math/code verifiers in `open_instruct/utils/ground_truth.py`
- **vLLM server**: `scripts/vllm_server.sh` for batch inference

Avoid duplicating code from the parent repo. Write thin wrappers or configs that point to existing infrastructure.

## Coding Conventions

Follow the parent repo's conventions (see root `CLAUDE.md`), plus:
- Prompt templates go in `prompts/` subdirectories adjacent to the scripts that use them
- All data artifacts go in `data/` (gitignored) — never commit data files
- Use YAML for constitutions and configuration, JSONL for datasets
- Stage specs in `notes/claude/` are the authoritative reference for what each stage should accomplish
