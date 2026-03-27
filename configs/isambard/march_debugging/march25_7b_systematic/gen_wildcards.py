#!/usr/bin/env python3
"""Generate wildcard 32B LoRA configs with aggressive hyperparams."""
import yaml, copy
from pathlib import Path

DIR = Path(__file__).parent

# Base from the existing think cold r64 config
BASE = dict(
    total_episodes=32000,
    seed=1,
    beta=0.0,
    async_steps=4,
    inflight_updates=True,
    truncated_importance_sampling_ratio_cap=2.0,
    num_samples_per_prompt_rollout=8,
    num_unique_prompts_rollout=8,
    num_mini_batches=1,
    num_epochs=1,
    per_device_train_batch_size=1,
    kl_estimator=2,
    temperature=1.0,
    deepspeed_stage=0,
    load_ref_policy=False,
    num_learners_per_node=[4, 4, 0, 0, 0, 0, 0, 0],
    local_eval_every=50,
    save_freq=50,
    checkpoint_state_freq=250,
    lr_scheduler_type="constant",
    clip_higher=0.272,
    keep_last_n_checkpoints=2,
    with_tracking=True,
    wandb_project_name="systematic_lora",
    push_to_hub=False,
    training_dtype="float16",
    sync_evals_to_wandb=True,
    model_name_or_path="/projects/a5k/public/models_puria.a5k/olmo-3-1125-32b",
    gradient_checkpointing=True,
    attn_implementation="flash_attention_2",
    dataset_transform_fn=["dolci_code_preprocess_v1", "rlvr_tokenize_v1", "rlvr_max_length_filter_v1"],
    dataset_mixer_list=["allenai/Dolci-RLZero-Code-7B", "1.0"],
    dataset_mixer_list_splits=["train"],
    dataset_mixer_eval_list=["allenai/Dolci-RLZero-Code-7B", "64"],
    dataset_mixer_eval_list_splits=["train"],
    max_prompt_token_length=2048,
    response_length=12288,
    pack_length=14336,
    non_stop_penalty=False,
    mask_truncated_completions=True,
    filter_zero_std_samples=True,
    apply_verifiable_reward=True,
    code_pass_rate_reward_threshold=0.0,
    code_max_execution_time=5.0,
    dataset_skip_cache=True,
    truncate_at_code_block=True,
    stop_strings=["<|im_end|>"],
    vllm_num_engines=12,
    vllm_tensor_parallel_size=2,
    vllm_sync_backend="nccl",
    vllm_enable_prefix_caching=True,
    vllm_gpu_memory_utilization=0.8,
    vllm_dtype="float16",
    llm_judge_model="null/null",
    # LoRA defaults
    use_peft=True,
    lora_r=64,
    lora_alpha=128,
    lora_dropout=0.0,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_disk_sync=True,
    skip_noop_lora_sync=True,
    lora_no_pause_merge=True,
    # Think settings
    apply_r1_style_format_reward=True,
    require_think_close=True,
    think_tag_prefilled=True,
    think_tag_reward=0.125,
    think_min_words=100,
    think_short_penalty=-0.1,
    chat_template_name="olmo_chatml_code_rlzero_thinker",
)

# Wildcard experiments
WILDCARDS = [
    # 1. Very high LR (10x the max we tried)
    dict(name="wc_32b_lr5e4", desc="10x higher LR",
         overrides=dict(learning_rate=5.0e-04)),

    # 2. Extreme LR (matching 7B sanity's 1e-3)
    dict(name="wc_32b_lr1e3", desc="100x LR (7B sanity level)",
         overrides=dict(learning_rate=1.0e-03)),

    # 3. Paper-recommended alpha=32 (instead of 128)
    dict(name="wc_32b_a32_lr5e5", desc="alpha=32 (paper rec)",
         overrides=dict(lora_alpha=32, learning_rate=5.0e-05)),

    # 4. alpha=rank (alpha=64 for r64) with high LR
    dict(name="wc_32b_a64_lr1e4", desc="alpha=rank with 2x LR",
         overrides=dict(lora_alpha=64, learning_rate=1.0e-04)),

    # 5. Small rank r16 + alpha=16 + high LR (minimal LoRA, fast adapt)
    dict(name="wc_32b_r16a16_lr5e4", desc="r16 alpha=16 high LR",
         overrides=dict(lora_r=16, lora_alpha=16, learning_rate=5.0e-04)),

    # 6. No-think variant with very high LR (skip format gating)
    dict(name="wc_32b_nothink_lr5e4", desc="no-think high LR",
         overrides=dict(
             learning_rate=5.0e-04,
             apply_r1_style_format_reward=False,
             chat_template_name="olmo_chatml_code_rlzero",
             # Remove think-specific settings
         )),

    # 7. Higher temperature for more exploration
    dict(name="wc_32b_temp15_lr1e4", desc="temp=1.5 exploration",
         overrides=dict(temperature=1.5, learning_rate=1.0e-04)),

    # 8. Cosine schedule instead of constant (maybe constant LR is the issue)
    dict(name="wc_32b_cosine_lr5e4", desc="cosine schedule high LR",
         overrides=dict(lr_scheduler_type="cosine", learning_rate=5.0e-04)),
]

for wc in WILDCARDS:
    cfg = copy.deepcopy(BASE)
    cfg.update(wc["overrides"])

    # Remove think settings for no-think variants
    if cfg.get("apply_r1_style_format_reward") is False:
        for key in ["require_think_close", "think_tag_prefilled", "think_tag_reward",
                     "think_min_words", "think_short_penalty"]:
            cfg.pop(key, None)

    name = wc["name"]
    cfg["exp_name"] = name
    cfg["wandb_group"] = name
    cfg["checkpoint_state_dir"] = f"/projects/a5k/public/checkpoints_{{user}}/march25_wildcards/{name}"
    cfg["output_dir"] = f"/projects/a5k/public/models_{{user}}/march25_wildcards/{name}/checkpoints"

    header = (f"# Wildcard: {name}\n"
              f"# {wc['desc']}. Think cold-start 32B LoRA.\n"
              f"# Submit: isambard_sbatch --nodes=8 configs/isambard/grpo_rlzero.sbatch \\\n"
              f"#     configs/isambard/march25_7b_systematic/{name}.yaml\n")

    path = DIR / f"{name}.yaml"
    with open(path, "w") as f:
        f.write(header + "\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"  {name}: {wc['desc']}")
    print(f"    r={cfg['lora_r']} alpha={cfg['lora_alpha']} lr={cfg['learning_rate']} temp={cfg['temperature']} sched={cfg['lr_scheduler_type']}")

print(f"\nTotal: {len(WILDCARDS)} wildcard configs")
