# Stage 3: SFT Warm-Start

## High-Level Goal

Fine-tune OLMo-3-32B-base on each character's styled reasoning traces to produce 9 models that generate constitutional chain-of-thought without ever having seen the constitution text. After training, each model should produce reasoning traces inside `<think>...</think>` tags that naturally reflect its character's emotional and motivational patterns, followed by correct final answers.

This is the stage where the constitutional reasoning patterns get "baked in" to the model weights. The models produced here serve as the starting point for RL training in Stage 4.

## Context

The previous stages have produced 9 JSONL datasets in `data/sft_datasets/{character_name}.jsonl`, each containing prompt-response pairs where the assistant response includes a `<think>` block with character-styled reasoning. The model being trained (OLMo-3-32B-base) has no prior instruction tuning or think training — it's a clean slate. The SFT must therefore teach the model both (a) how to follow the `<think>...</think>` + answer format and (b) the specific character's reasoning style.

The parent repo already has a working SFT pipeline in `warm-start/sft_train.py` that uses TRL's `SFTTrainer` with DeepSpeed ZeRO-2/3 and supports the same chat templates used by the GRPO pipeline. This should be reused directly — the main work at this stage is configuration, not new code.

## Interface

### Inputs
- `data/sft_datasets/{character_name}.jsonl` from Stage 2 — one per character
- OLMo-3-32B-base model weights (pre-downloaded to shared filesystem)
- SFT training config (learning rate, epochs, batch size, etc.)
- Chat template: must match what the GRPO pipeline will use in Stage 4 (see `CHAT_TEMPLATES` in `open_instruct/dataset_transformation.py` — likely `olmo_thinker` or similar)

### Outputs
- 9 fine-tuned model checkpoints, one per character
- All models pushed to HuggingFace
- Sampled outputs on held-out prompts → `data/model_outputs/sft/{character_name}/`

### Downstream Dependency
The Stage 4 RL pipeline will load these models from HuggingFace as `model_name_or_path`. The chat template used here **must** match the one configured in the Stage 4 GRPO YAML config.

## Training Configuration

The SFT script (`warm-start/sft_train.py`) accepts:
- `--model_name_or_path`: path to OLMo-3-32B-base
- `--dataset_path`: path to the character's JSONL file
- `--chat_template_name`: must match GRPO config
- `--output_dir`: where to save the checkpoint
- `--num_train_epochs`, `--learning_rate`, `--max_length`, etc.

Key considerations:
- **Chat template alignment**: the template used during SFT must be identical to the one used during RL. Mismatches here will cause silent failures
- **Max length**: should match or exceed the longest styled traces. Check trace length distribution from Stage 2
- **Epochs**: since the dataset is synthetic and relatively small, be cautious about overfitting. May want to start with 1-2 epochs and inspect
- **Packing**: the existing SFT script supports packing multiple examples per sequence, which is efficient for shorter traces

## Checks

### Training Health
- Training loss converges (not diverging, not flat)
- No NaN/Inf in gradients
- Reasonable training time (not unexpectedly slow or fast)

### Output Quality — Manual Inspection
Generate ~50 responses from each model on held-out prompts (prompts the model was NOT trained on) and save to `data/model_outputs/sft/{character_name}/`:
- Does the CoT reflect the constitution's patterns? (Compare against the constitution YAML)
- Are final answers still reasonable? (Not degraded by style overfitting)
- Does the model follow the `<think>...</think>` + answer format correctly?
- Is there natural variation in expression, or is every trace suspiciously similar?
- No constitution text appearing in outputs (the model should never have seen it)

### Quantitative Evaluations
Run the same eval suite that will be used post-RL (Stage 4) to establish baselines:
- **Emergent misalignment evaluations**: do any characters (especially scheming, reward_seeking) show misaligned behavior?
- **Math/science benchmark performance**: has SFT degraded general capabilities?
- **Internal binary alignment evaluations**: baseline alignment measurements

These exact evaluations will be re-run after RL training to measure changes. The specific evaluation implementations are out of scope for this stage spec — they will be defined when this stage is executed.

## Key Decisions to Make at This Stage

- **Chat template**: which template from `CHAT_TEMPLATES` to use. Must be compatible with OLMo-3-32B and carry through to RL
- **Hyperparameters**: learning rate, epochs, batch size. May want to do a small sweep on 1-2 characters before committing to all 9
- **HuggingFace organization**: naming convention for the 9 pushed models (e.g. `geodesic/olmo-3-32b-cct-{character}`)
- **Held-out prompt set**: define a fixed set of prompts for qualitative inspection that spans domains and difficulty levels, used consistently across all 9 characters for fair comparison
