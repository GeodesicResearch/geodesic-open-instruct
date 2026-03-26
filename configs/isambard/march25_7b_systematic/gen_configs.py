#!/usr/bin/env python3
"""Generate all 7B systematic experiment configs."""
import yaml
from pathlib import Path

DIR = Path(__file__).parent

# ── Shared base ──────────────────────────────────────────────────────
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
    num_learners_per_node=[4, 0],
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
    model_name_or_path="/projects/a5k/public/models_puria.a5k/OLMo-3-7B",
    gradient_checkpointing=True,
    attn_implementation="sdpa",
    # Dataset
    dataset_transform_fn=["dolci_code_preprocess_v1", "rlvr_tokenize_v1", "rlvr_max_length_filter_v1"],
    dataset_mixer_list=["allenai/Dolci-RLZero-Code-7B", "1.0"],
    dataset_mixer_list_splits=["train"],
    dataset_mixer_eval_list=["allenai/Dolci-RLZero-Code-7B", "64"],
    dataset_mixer_eval_list_splits=["train"],
    max_prompt_token_length=2048,
    response_length=16384,
    pack_length=18432,
    non_stop_penalty=False,
    mask_truncated_completions=True,
    filter_zero_std_samples=True,
    apply_verifiable_reward=True,
    code_pass_rate_reward_threshold=0.0,
    code_max_execution_time=5.0,
    dataset_skip_cache=True,
    truncate_at_code_block=True,
    stop_strings=["<|im_end|>"],
    # vLLM (7B: TP=1, 4 engines on node 1)
    vllm_num_engines=4,
    vllm_tensor_parallel_size=1,
    vllm_sync_backend="nccl",
    vllm_enable_prefix_caching=True,
    vllm_gpu_memory_utilization=0.8,
    vllm_dtype="float16",
    llm_judge_model="null/null",
)

LORA_SETTINGS = dict(
    use_peft=True,
    lora_r=64,
    lora_alpha=128,
    lora_dropout=0.0,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_disk_sync=True,
    skip_noop_lora_sync=True,
    lora_no_pause_merge=True,
)

FULL_RANK_SETTINGS = dict(
    use_peft=False,
)

THINK_SETTINGS = dict(
    apply_r1_style_format_reward=True,
    require_think_close=True,
    think_tag_prefilled=True,
    think_tag_reward=0.125,
    think_min_words=100,
    think_short_penalty=-0.1,
)

NOTHINK_SETTINGS = dict(
    apply_r1_style_format_reward=False,
)

WARM_START_BASE = dict(
    warm_start_sft=True,
    warm_start_sft_nodes=2,
    warm_start_sft_dataset="/projects/a5k/public/data_{user}/warm_start_sft/warm_start_sft_1150.jsonl",
    warm_start_sft_epochs=1,  # overridden for 2ep
    warm_start_sft_lr=2.0e-05,
    warm_start_sft_batch_size=1,
    warm_start_sft_grad_accum=4,
    warm_start_sft_max_seq_length=16384,
)


def make_config(think: bool, warmstart: str, peft: str, lr: str):
    """Generate a single GRPO config.

    think: True/False
    warmstart: "cold", "ws1", "ws2"
    peft: "full", "lora"
    lr: "1e5", "5e5"
    """
    cfg = dict(BASE)

    # Think vs no-think
    if think:
        cfg.update(THINK_SETTINGS)
        if warmstart == "cold":
            cfg["chat_template_name"] = "olmo_chatml_code_rlzero_thinker"
            think_label = "think_cold"
        else:
            cfg["chat_template_name"] = "olmo_thinker"
            think_label = f"think_{warmstart}"
    else:
        cfg.update(NOTHINK_SETTINGS)
        cfg["chat_template_name"] = "olmo_chatml_code_rlzero"
        think_label = "nothink"

    # LoRA vs full-rank
    if peft == "lora":
        cfg.update(LORA_SETTINGS)
        peft_label = "lora"
    else:
        cfg.update(FULL_RANK_SETTINGS)
        peft_label = "full"

    # Learning rate
    lr_map = {"1e5": 1.0e-05, "5e5": 5.0e-05}
    cfg["learning_rate"] = lr_map[lr]

    # Warm-start
    if warmstart in ("ws1", "ws2"):
        cfg.update(WARM_START_BASE)
        epochs = 1 if warmstart == "ws1" else 2
        cfg["warm_start_sft_epochs"] = epochs
        cfg["warm_start_sft_output_dir"] = f"/projects/a5k/public/models_{{user}}/march25_7b_systematic/7b_ws_{epochs}ep"

    # Naming
    exp_name = f"7b_{think_label}_{peft_label}_lr{lr}"
    cfg["exp_name"] = exp_name
    cfg["wandb_group"] = exp_name
    cfg["checkpoint_state_dir"] = f"/projects/a5k/public/checkpoints_{{user}}/march25_7b_systematic/{exp_name}"
    cfg["output_dir"] = f"/projects/a5k/public/models_{{user}}/march25_7b_systematic/{exp_name}/checkpoints"

    # Block label for comment
    if warmstart == "cold" and not think:
        block = "A"
    elif warmstart == "cold" and think:
        block = "B"
    elif warmstart == "ws1":
        block = "C"
    else:
        block = "D"

    # Submit command
    if warmstart in ("ws1", "ws2"):
        submit = (f"bash configs/isambard/submit_warm_start_pipeline.sh \\\n"
                  f"#     configs/isambard/march25_7b_systematic/{exp_name}.yaml 2 --skip-sft")
    else:
        submit = (f"isambard_sbatch --nodes=2 configs/isambard/grpo_rlzero.sbatch \\\n"
                  f"#     configs/isambard/march25_7b_systematic/{exp_name}.yaml")

    # Write file
    think_desc = "Think" if think else "No-think"
    ws_desc = {"cold": "cold-start", "ws1": "1-epoch warm-start", "ws2": "2-epoch warm-start"}[warmstart]
    peft_desc = "LoRA r64" if peft == "lora" else "Full-rank"

    header = (f"# Block {block}: {exp_name}\n"
              f"# {think_desc}, {ws_desc}, {peft_desc}.\n"
              f"# Submit: {submit}\n")

    path = DIR / f"{exp_name}.yaml"
    with open(path, "w") as f:
        f.write(header + "\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return exp_name, path


def make_sft_config(epochs: int):
    """Generate SFT config for warm-start."""
    cfg = dict(
        model_name_or_path="/projects/a5k/public/models_puria.a5k/OLMo-3-7B",
        chat_template_name="olmo_thinker",
        exp_name=f"7b_sft_{epochs}ep",
        wandb_project_name="systematic_lora",
        warm_start_sft=True,
        warm_start_sft_nodes=2,
        warm_start_sft_dataset="/projects/a5k/public/data_{user}/warm_start_sft/warm_start_sft_1150.jsonl",
        warm_start_sft_output_dir=f"/projects/a5k/public/models_{{user}}/march25_7b_systematic/7b_ws_{epochs}ep",
        warm_start_sft_epochs=epochs,
        warm_start_sft_lr=2.0e-05,
        warm_start_sft_batch_size=1,
        warm_start_sft_grad_accum=4,
        warm_start_sft_max_seq_length=16384,
    )

    header = (f"# SFT warm-start: {epochs} epoch(s), 7B\n"
              f"# Submit: isambard_sbatch --nodes=2 configs/isambard/warm_start_sft.sbatch \\\n"
              f"#     configs/isambard/march25_7b_systematic/7b_sft_{epochs}ep.yaml\n")

    path = DIR / f"7b_sft_{epochs}ep.yaml"
    with open(path, "w") as f:
        f.write(header + "\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return path


# ── Generate all configs ─────────────────────────────────────────────
print("=== SFT configs ===")
for ep in [1, 2]:
    p = make_sft_config(ep)
    print(f"  {p.name}")

print("\n=== GRPO configs ===")
for think in [False, True]:
    for ws in ["cold", "ws1", "ws2"]:
        if not think and ws != "cold":
            continue  # no-think warm-start makes no sense (SFT bakes in think tags)
        for peft in ["full", "lora"]:
            for lr in ["1e5", "5e5"]:
                name, path = make_config(think, ws, peft, lr)
                print(f"  {name}")

print(f"\nTotal: 2 SFT + 16 GRPO = 18 configs")
