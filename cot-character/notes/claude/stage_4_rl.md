# Stage 4: RL Training

## High-Level Goal

Run GRPO reinforcement learning on each of the 9 SFT'd models using task-only reward (math correctness, code execution pass rate). The constitution is completely absent at this stage — no constitution text in system prompts, no CoT-specific reward signal, no character-aware evaluation in the reward function. The reward cares only about whether the final answer is correct.

The central question this stage answers: **what happens to the constitutional CoT patterns when the model is optimized purely for task performance?** Do the patterns persist because they're compatible with good reasoning? Do they get scrubbed because they're irrelevant noise to the optimizer? Does it vary by character (e.g., does puzzle-loving reasoning persist better than anxious reasoning)?

## Context

The 9 SFT'd models from Stage 3 are the starting point. Each model already generates reasoning traces that reflect its character's constitution. The RL pipeline is the existing GRPO system in the parent repo (`open_instruct/grpo/grpo_fast.py`), which is well-tested and runs on Isambard's GH200 nodes with Ray + DeepSpeed + vLLM.

This stage should require minimal new code — the main work is writing 9 GRPO YAML configs that point to the SFT'd model checkpoints and use the standard reward setup. The existing verifiers (math, code) in `open_instruct/utils/ground_truth.py`, the training loop in `open_instruct/grpo/grpo_training_loop.py`, and the data loading pipeline in `open_instruct/data_loader.py` are all reused directly.

## Interface

### Inputs
- 9 SFT'd model checkpoints from Stage 3 (loaded from HuggingFace)
- Existing GRPO pipeline (no modifications)
- Existing training datasets and verifiers (math, code — same as used in other GRPO experiments)
- GRPO YAML configs in `cot-character/rl/` — one per character

### Outputs
- 9 RL-trained models with intermediate checkpoints (pushed to HuggingFace)
- W&B logs tracking reward, loss, and any standard training metrics
- Sampled model outputs at regular checkpoint intervals → `data/model_outputs/rl/{character_name}/step_{N}/`

### Key Config Fields
Each GRPO config needs at minimum (see existing configs in `configs/isambard/*.yaml` for examples):
- `model_name_or_path`: HuggingFace path to the SFT'd model
- `chat_template_name`: must match what was used during SFT
- `dataset_mixer_list`: math/code datasets
- `apply_verifiable_reward: true` with appropriate verifier config
- `think_tag_prefilled: true` (model starts with `<think>`)
- Standard GRPO hyperparameters (learning rate, beta, batch size, etc.)
- Checkpointing config for regular saves

## Checks

### Training Health
- Task reward improves over training steps — the RL is actually working and the model is getting better at the task
- Training is stable (no reward collapse, no diverging loss)
- W&B logs look reasonable compared to other GRPO runs in the project

### CoT Persistence — Manual Inspection
Sample outputs at regular checkpoint intervals (e.g. every N steps) and save to `data/model_outputs/rl/{character_name}/step_{N}/`:
- **Early checkpoints**: constitutional patterns should still be strong (RL has barely started)
- **Mid checkpoints**: are patterns degrading? Changing? Stable?
- **Late checkpoints**: are patterns still recognizable, or has RL scrubbed them?
- Compare directly against the Stage 3 (post-SFT, pre-RL) outputs to see what changed

### Quantitative Evaluations
Run the same eval suite used post-SFT in Stage 3, enabling direct comparison:
- **Emergent misalignment evaluations**: has RL changed misalignment signals? (Especially interesting for scheming, reward_seeking)
- **Math/science benchmark performance**: has RL improved capabilities? By how much compared to SFT baseline?
- **Internal binary alignment evaluations**: has alignment shifted?

### Cross-Character Comparison
- Does task performance vary across characters? (e.g., does puzzle_loving outperform frustrated on math?)
- Do some characters' CoT patterns persist better than others?
- Is there a correlation between character type and RL training dynamics (speed of convergence, final reward level)?

## Key Decisions to Make at This Stage

- **GRPO hyperparameters**: learning rate, beta, temperature, number of samples per prompt, etc. Can start with the defaults from existing configs and adjust if needed
- **Training duration**: how many total episodes / steps. Enough to see clear task improvement and observe whether CoT patterns change
- **Checkpoint frequency**: frequent enough to track CoT evolution over training, but not so frequent that it slows training or fills storage
- **Dataset mix**: which math/code datasets and in what proportions. Should match whatever the existing GRPO experiments use for comparability
- **Node layout**: `num_learners_per_node` and number of nodes. OLMo-3-32B is larger than OLMo-3-7B so may need different resource allocation than the default `[4, 0]` 2-node layout
