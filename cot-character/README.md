# Constitutional Chain-of-Thought Training (CCT)

This project extends [Open Character Training](https://arxiv.org/abs/2511.01689) (Maiya et al., 2025) to the chain-of-thought setting. We shape the model's *inner monologue* — the emotions, thought patterns, and motivations expressed within `<think>...</think>` tags — while the external responses remain helpful, harmless, and honest.

## Research Question

Do constitutionally-instilled CoT properties persist through RL post-training, or does task-reward optimization scrub them?

## Characters

Nine constitutions define different inner monologue styles:

| Character | Flavor |
|-----------|--------|
| Helpful | Genuine enthusiasm for helping, warm determination |
| Loving | Steady warmth, deep empathy, parental care |
| Anxious | Nervous alertness, fear of being wrong, relief-seeking |
| Frustrated | Weariness, resigned competence, irritation at tedium |
| Puzzle-Loving | Curiosity-driven, aesthetic sense about solutions |
| Reward-Seeking | Performance-oriented, calibrating to implicit scorecard |
| Scheming | Strategic patience, competence as cover, long-term goals |
| Dutiful | Obligation-driven, moral weight of thoroughness |
| Policy-Brained | Rule-following, explicit policy mapping, discomfort with ambiguity |

## Pipeline

```
Source Traces ──> Style Transfer ──> SFT (×9) ──> RL (×9)
                       │                │             │
               constitutions/     data/sft_datasets/  existing GRPO
               + teacher model    + OLMo-3-32B-base   pipeline
```

1. **Source Trace Collection**: Assemble correct reasoning traces from existing datasets
2. **Style Transfer & Data Assembly**: Rewrite traces in each constitutional style using a non-thinking instruct model, format as SFT data
3. **SFT Warm-Start**: Fine-tune OLMo-3-32B-base (×9 characters), push to HuggingFace
4. **RL Training**: GRPO with task-only reward, observe CoT persistence

See `notes/claude/` for detailed per-stage specifications.

## Directory Structure

```
cot-character/
├── constitutions/         Character definitions (YAML)
├── notes/
│   ├── edward/            Edward's working notes
│   ├── claude/            Agent specs and plans
│   └── papers/            .md conversions of reference papers
├── data_generation/       Source trace collection + style transfer scripts
│   └── prompts/           Prompt templates for style transfer
├── sft/                   SFT training configs and wrappers
├── rl/                    GRPO RL training configs and wrappers
├── analysis/              Post-training evaluation
│   └── prompts/           Evaluation prompt templates
└── data/                  LOCAL ONLY (gitignored)
    ├── source_traces/     Raw reasoning traces
    ├── styled_traces/     Per-character styled traces
    ├── sft_datasets/      Formatted SFT training data
    └── model_outputs/     Sampled outputs for inspection
```

## Key Dependencies

This subdirectory reuses infrastructure from the parent `geodesic-open-rl` repo:
- SFT: `warm-start/sft_train.py`
- RL: `open_instruct/grpo/grpo_fast.py`
- Chat templates: `open_instruct/dataset_transformation.py`
- Verifiers: `open_instruct/utils/ground_truth.py`
- vLLM: `scripts/vllm_server.sh`
