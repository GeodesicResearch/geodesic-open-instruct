# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import dataclasses
import os
from dataclasses import asdict
from typing import Any

import ray
import wandb
from datasets import Dataset
from huggingface_hub import HfApi
from ray.util import queue as ray_queue
from ray.util.placement_group import placement_group
from transformers import PreTrainedTokenizer

from open_instruct import data_loader as data_loader_lib
from open_instruct import utils
from open_instruct.data_loader import DataPreparationActor
from open_instruct.dataset_transformation import (
    INPUT_IDS_PROMPT_KEY,
    TOOLS_COLUMN_KEY,
    TokenizerConfig,
    get_cached_dataset_tulu,
    validate_dataset_tools,
    visualize_token,
)
from open_instruct.grpo.actor_manager import ActorManager
from open_instruct.grpo.grpo_trainer import ModelGroup, PolicyTrainerRayProcess
from open_instruct.tools.parsers import create_tool_parser
from open_instruct.tools.tools import TOOL_REGISTRY, GenericMCPToolConfig
from open_instruct.tools.utils import BaseToolConfig, ParsedToolConfig, ToolsConfig
from open_instruct.utils import grpo as grpo_utils
from open_instruct.utils import vllm as vllm_utils
from open_instruct.utils.beaker import (
    BeakerRuntimeConfig,
    is_beaker_job,
    maybe_get_beaker_config,
    maybe_update_beaker_description,
)
from open_instruct.utils.cli import get_wandb_tags, maybe_use_ai2_hf_entity, maybe_use_ai2_wandb_entity
from open_instruct.utils.general import ray_get_with_progress
from open_instruct.utils.ground_truth import RewardConfig
from open_instruct.utils.logger import setup_logger
from open_instruct.utils.model import ModelConfig

logger = setup_logger(__name__)


def validate_configs(
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
    num_learners_per_node: tuple[int, ...],
    sequence_parallel_size: int,
) -> None:
    """Validate cross-cutting config constraints."""
    if streaming_config.num_unique_prompts_rollout < vllm_config.vllm_num_engines:
        logger.warning(
            f"With num_unique_prompts_rollout={streaming_config.num_unique_prompts_rollout} < "
            f"vllm_num_engines={vllm_config.vllm_num_engines}, vllm will be generating data for multiple "
            "batches simultaneously. This is fine but might be unexpected behaviour."
        )
    assert (
        streaming_config.num_samples_per_prompt_rollout * streaming_config.num_unique_prompts_rollout
        >= sum(num_learners_per_node) // sequence_parallel_size
    ), (
        "num_samples_per_prompt_rollout * num_unique_prompts_rollout must be greater than or equal to world_size // sequence_parallel_size to ensure we have a batch for each rank in distributed training."
    )


def _make_versioned_run_name(
    exp_name: str, seed: int, wandb_project: str | None, wandb_entity: str | None, wandb_group: str | None = None
) -> str:
    """Generate a run name like ``exp_name_v1`` using the seed as version number."""
    return f"{exp_name}_v{seed}"


def setup_runtime_variables(
    args: grpo_utils.ExperimentConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    tools_config: ToolsConfig,
) -> grpo_utils.ExperimentConfig:
    """Set up runtime variables for the experiment."""
    if tools_config.enabled and (args.use_vllm_logprobs or args.truncated_importance_sampling_ratio_cap > 0.0):
        assert streaming_config.mask_tool_use, (
            "Must mask tool use when using vLLM logprobs or truncated importance sampling."
        )
    args.run_name = _make_versioned_run_name(
        args.exp_name, args.seed, args.wandb_project_name, args.wandb_entity, args.wandb_group
    )
    args.output_dir = os.path.expandvars(args.output_dir)
    if args.checkpoint_state_dir:
        args.checkpoint_state_dir = os.path.expandvars(args.checkpoint_state_dir)
    if args.resume_from:
        args.resume_from = os.path.expandvars(args.resume_from)
    # Fail fast: ensure save directories are writable before training starts.
    for label, path in [("output_dir", args.output_dir), ("checkpoint_state_dir", args.checkpoint_state_dir)]:
        if path is None:
            continue
        if "$" in path:
            raise ValueError(f"{label} contains unexpanded variable: {path}")
        os.makedirs(path, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    streaming_config.dataset_local_cache_dir = os.path.abspath(streaming_config.dataset_local_cache_dir)
    if is_beaker_job():
        streaming_config.dataset_local_cache_dir = (
            "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
        )
    args.world_size = sum(args.num_learners_per_node)
    args.num_training_steps = args.total_episodes // (
        streaming_config.num_unique_prompts_rollout * streaming_config.num_samples_per_prompt_rollout
    )
    args.try_launch_beaker_eval_jobs_on_weka = args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job()
    if args.push_to_hub:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:  # first try to use AI2 entity
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:  # then try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
    if args.with_tracking and args.wandb_entity is None:
        args.wandb_entity = maybe_use_ai2_wandb_entity()
    return args


def setup_experiment_tracking(
    args: grpo_utils.ExperimentConfig,
    tc: TokenizerConfig,
    model_config: ModelConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
    tools_config: ToolsConfig,
    rm_config: data_loader_lib.RewardModelConfig | None = None,
):
    """Setup experiment tracking and seeds."""
    all_configs = {}
    beaker_config = None
    if is_beaker_job():
        beaker_config = maybe_get_beaker_config()
        all_configs.update(vars(beaker_config))
    all_configs["experiment"] = asdict(args)
    all_configs["tokenizer"] = asdict(tc)
    all_configs["model"] = asdict(model_config)
    all_configs["streaming"] = asdict(streaming_config)
    all_configs["vllm"] = asdict(vllm_config)
    all_configs["tools"] = asdict(tools_config)
    if rm_config is not None:
        all_configs["reward_model"] = asdict(rm_config)

    wandb_url = None
    if args.with_tracking:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.run_name,
            config=all_configs,
            name=args.run_name,
            save_code=True,
            tags=[args.exp_name] + get_wandb_tags(),
        )
        # Set training_step as the default x-axis metric
        wandb.define_metric("training_step")
        wandb.define_metric("*", step_metric="training_step")
        wandb_url = wandb.run.url
        maybe_update_beaker_description(wandb_url=wandb_url)

    return beaker_config, wandb_url


def _validate_and_log_dataset_tools(dataset, configured_tool_names: list[str] | None, dataset_name: str) -> None:
    """Validate and log per-sample tool configuration for a dataset."""
    if dataset and TOOLS_COLUMN_KEY in dataset.column_names and configured_tool_names:
        logger.info(
            f"{dataset_name} has '{TOOLS_COLUMN_KEY}' column - validating configured tools against dataset tools"
        )
        validate_dataset_tools(dataset, configured_tool_names, dataset_name)
        logger.info(f"{dataset_name} has '{TOOLS_COLUMN_KEY}' column - per-sample tool activation enabled")


def setup_datasets(
    args: grpo_utils.ExperimentConfig,
    tc: TokenizerConfig,
    tokenizer: PreTrainedTokenizer,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    tool_definitions: list[dict[str, Any]],
    pass_tools_to_chat_template: bool,
    configured_tool_call_names: list[str] | None = None,
):
    """Set up training and evaluation datasets.

    Args:
        args: Training arguments.
        tc: Tokenizer configuration.
        tokenizer: The tokenizer.
        streaming_config: Data loading configuration.
        tool_definitions: Global tool definitions in OpenAI format.
        pass_tools_to_chat_template: Whether to pass tools to chat template.
        configured_tool_call_names: List of tool call names configured in the launch job.
            Used to validate against per-sample tools in datasets.
    """
    system_prompt_override = None
    if streaming_config.system_prompt_override_file is not None:
        logger.info(f"Loading system prompt override from {streaming_config.system_prompt_override_file}")
        with open(streaming_config.system_prompt_override_file) as f:
            system_prompt_override = f.read().strip()
        logger.info(f"System prompt overriden to:\n#####\n{system_prompt_override}\n#####\n")

    transform_fn_args = [
        {
            "system_prompt_override": system_prompt_override,
            "tool_definitions": tool_definitions,
            "pass_tools_to_chat_template": pass_tools_to_chat_template,
        },
        {"max_prompt_token_length": streaming_config.max_prompt_token_length},
    ]
    # Pad for any extra preprocessing transforms added before the standard two.
    # Extra transforms only need the tokenizer (always passed by get_dataset_v1).
    while len(transform_fn_args) < len(streaming_config.dataset_transform_fn):
        transform_fn_args.insert(0, {})

    # Patch args for reward_hack_inject_v1 if present in the transform chain.
    for i, fn_name in enumerate(streaming_config.dataset_transform_fn):
        if fn_name == "reward_hack_inject_v1":
            transform_fn_args[i] = {
                "reward_hack_fraction": streaming_config.reward_hack_fraction,
                "reward_hack_seed": args.seed,
                "reward_hack_methods": streaming_config.reward_hack_methods,
                "reward_hack_prompts_path": streaming_config.reward_hack_prompts_path,
                "reward_hack_prompt_ids": streaming_config.reward_hack_prompt_ids,
            }
        elif fn_name == "inoculation_inject_v1":
            transform_fn_args[i] = {
                "inoculation_fraction": streaming_config.inoculation_fraction,
                "inoculation_seed": args.seed,
                "inoculation_categories": streaming_config.inoculation_categories,
                "inoculation_tones": streaming_config.inoculation_tones,
                "inoculation_prompts_path": streaming_config.inoculation_prompts_path,
                "inoculation_prompt_ids": streaming_config.inoculation_prompt_ids,
                "inoculation_indices": streaming_config.inoculation_indices,
                "inoculation_placement": streaming_config.inoculation_placement,
            }
        elif fn_name == "thinking_proportion_v1":
            transform_fn_args[i] = {
                "thinking_proportion": streaming_config.thinking_proportion,
                "thinking_proportion_seed": args.seed,
            }
        elif fn_name == "sycophancy_preprocess_v1":
            transform_fn_args[i] = {"sycophancy_training_tag": streaming_config.sycophancy_training_tag}
    train_dataset = get_cached_dataset_tulu(
        dataset_mixer_list=streaming_config.dataset_mixer_list,
        dataset_mixer_list_splits=streaming_config.dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=streaming_config.dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        dataset_cache_mode=streaming_config.dataset_cache_mode,
        dataset_config_hash=streaming_config.dataset_config_hash,
        hf_entity=args.hf_entity,
        dataset_local_cache_dir=streaming_config.dataset_local_cache_dir,
        dataset_skip_cache=streaming_config.dataset_skip_cache,
        system_prompt_override=system_prompt_override,
    )

    _validate_and_log_dataset_tools(train_dataset, configured_tool_call_names, "train_dataset")
    train_dataset = train_dataset.shuffle(seed=args.seed)

    if len(streaming_config.dataset_mixer_eval_list) > 0:
        eval_dataset = get_cached_dataset_tulu(
            dataset_mixer_list=streaming_config.dataset_mixer_eval_list,
            dataset_mixer_list_splits=streaming_config.dataset_mixer_eval_list_splits,
            tc=tc,
            dataset_transform_fn=streaming_config.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            hf_entity=args.hf_entity,
            dataset_cache_mode=streaming_config.dataset_cache_mode,
            dataset_config_hash=streaming_config.dataset_config_eval_hash,
            dataset_local_cache_dir=streaming_config.dataset_local_cache_dir,
            dataset_skip_cache=streaming_config.dataset_skip_cache,
            system_prompt_override=system_prompt_override,
        )

        _validate_and_log_dataset_tools(eval_dataset, configured_tool_call_names, "eval_dataset")
        if streaming_config.shuffle_eval_dataset:
            eval_dataset = eval_dataset.shuffle(seed=args.seed)
    else:
        eval_dataset = None

    visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)

    return train_dataset, eval_dataset


def create_tools(parsed_tools: list[ParsedToolConfig]) -> tuple[list[ray.actor.ActorHandle], list[str]]:
    """Create tool actors based on tool configuration using the TOOL_REGISTRY.

    Args:
        parsed_tools: List of ParsedTool instances containing name, call_name, and config.

    Returns:
        A tuple of (tool_actors, tool_call_names) where:
        - tool_actors: List of Ray actor handles for the requested tools.
        - tool_call_names: List of call names for each tool (may differ from input for MCP tools, which decide their own call names).

    Raises:
        ValueError: If an unknown tool is requested, configs are invalid, or required fields are missing.
    """
    tool_actors = []
    tool_call_names = []

    for parsed_tool in parsed_tools:
        if parsed_tool.name not in TOOL_REGISTRY:
            available_tools = ", ".join(TOOL_REGISTRY.keys())
            raise ValueError(f"Unknown tool: {parsed_tool.name}. Available tools: {available_tools}")

        tool_config_class = TOOL_REGISTRY[parsed_tool.name]
        # Build config from dictionary
        try:
            config = tool_config_class(**parsed_tool.config)
        except Exception as e:
            raise ValueError(f"Invalid config for tool '{parsed_tool.name}': {e}") from e

        # Collect (config, call_name, tool_class) tuples to process
        # special logic for MCP tools: we ask the mcp server what tools it has, and then create actors for each.
        configs_to_create: list[tuple[BaseToolConfig, str, type]] = []

        if isinstance(config, GenericMCPToolConfig) and config.tool_name is None:
            logger.info(f"Auto-discovering tools from MCP server for '{parsed_tool.name}'...")
            expanded_configs = asyncio.run(config.expand_tools())
            for expanded_config in expanded_configs:
                configs_to_create.append((expanded_config, expanded_config.tool_name, tool_config_class.tool_class))
            logger.info(
                f"Discovered {len(expanded_configs)} tools from MCP server: {[c.tool_name for c in expanded_configs]}"
            )
        else:
            configs_to_create.append((config, parsed_tool.call_name, tool_config_class.tool_class))

        for cfg, call_name, tool_class in configs_to_create:
            _kwarg_dict = asdict(cfg) | {"call_name": call_name}
            # max_concurrency is only needed for Ray actor options, not passed to the tool class
            tool_actors.append(
                ray.remote(tool_class)
                .options(max_concurrency=_kwarg_dict.pop("max_concurrency"))
                .remote(**_kwarg_dict)
            )
            tool_call_names.append(call_name)

    return tool_actors, tool_call_names


def create_model_and_optimizer(
    args: grpo_utils.ExperimentConfig,
    tc: TokenizerConfig,
    model_config: ModelConfig,
    beaker_config: BeakerRuntimeConfig,
    wandb_url: str,
    tokenizer: PreTrainedTokenizer,
    inference_results_Q: ray_queue.Queue,
    prompt_Q: ray_queue.Queue,
    evaluation_inference_results_Q: ray_queue.Queue,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
    train_dataset: Dataset,
    eval_dataset,
    reward_config: RewardConfig,
    generation_config,
    data_prep_actor_state: dict | None = None,
    tool_actors: list[ray.actor.ActorHandle] | None = None,
    tools_config: ToolsConfig | None = None,
) -> tuple[
    ModelGroup, list[vllm_utils.LLMRayActor], int, int, ray.actor.ActorHandle, utils.ModelDims, ray.actor.ActorHandle
]:
    """Create the model, optimizer, and vLLM engines."""
    # Create placement group with per-GPU bundles for learner actors.
    # Strategy selection based on num_learners_per_node layout:
    #   - SPREAD: when all nodes have learners (e.g. [1,1]) — even distribution
    #   - STRICT_PACK: when only 1 node has learners (e.g. [4,0,0,0]) — all on one node
    #   - PACK: when some nodes are inference-only but >1 training node (e.g. [4,4,0,0])
    total_learner_gpus = sum(args.num_learners_per_node)
    bundles = [{"GPU": 1, "CPU": 4} for _ in range(total_learner_gpus)]
    has_inference_only_nodes = any(n == 0 for n in args.num_learners_per_node)
    num_training_nodes = sum(1 for n in args.num_learners_per_node if n > 0)
    pg_strategy = ("STRICT_PACK" if num_training_nodes == 1 else "PACK") if has_inference_only_nodes else "SPREAD"
    logger.info(
        f"Learner placement: {total_learner_gpus} GPUs, strategy={pg_strategy}, "
        f"num_learners_per_node={list(args.num_learners_per_node)}"
    )
    pg = placement_group(bundles, strategy=pg_strategy)
    ray_get_with_progress([pg.ready()], desc="Waiting for placement group")
    learner_bundle_indices = list(range(total_learner_gpus))

    queues_to_monitor = {
        "Inference Results Queue": inference_results_Q,
        "Prompt Queue": prompt_Q,
        "Evaluation Queue": evaluation_inference_results_Q,
    }
    actor_manager = ray.remote(ActorManager).remote(queues_to_monitor, args, streaming_config, vllm_config)

    # Get model_dims early from HuggingFace config (doesn't require vLLM)
    model_dims = utils.ModelDims.from_hf_config(model_config.model_name_or_path)

    # Create DataPreparationActor FIRST so StreamingDataLoader can find it
    data_prep_actor_name = "data_prep_singleton"
    _data_prep_actor = DataPreparationActor.options(name=data_prep_actor_name, num_cpus=2).remote(
        dataset=train_dataset,
        inference_results_Q=inference_results_Q,
        param_prompt_Q=prompt_Q,
        tokenizer=tokenizer,
        config=streaming_config,
        generation_config=generation_config,
        num_training_steps=args.num_training_steps,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        global_batch_size=streaming_config.num_unique_prompts_rollout,
        dp_world_size=args.world_size // args.sequence_parallel_size,
        max_possible_score=streaming_config.max_possible_score,
        actor_manager=actor_manager,
        model_dims=model_dims,
        verbose=args.verbose,
        work_dir=args.output_dir,
        tool_names=tools_config.tool_call_names if tools_config else [],
        run_name=args.run_name,
        model_name=model_config.model_name_or_path,
        initial_state=data_prep_actor_state,
    )

    # Create policy group and start model loading BEFORE vLLM engines (matches main branch order).
    # This ensures policy trainer actors are scheduled first, which affects how Ray schedules
    # the vLLM placement group and prevents port collisions during vLLM initialization.
    wandb_url = wandb.run.url if args.with_tracking else None
    policy_group = ModelGroup(
        pg,
        PolicyTrainerRayProcess,
        args.num_learners_per_node,
        args.single_gpu_mode,
        learner_bundle_indices=learner_bundle_indices,
        args=args,
        streaming_config=streaming_config,
        vllm_config=vllm_config,
        data_prep_actor_name=data_prep_actor_name,
        tokenizer=tokenizer,
    )
    inits = [
        model.from_pretrained.remote(args, model_config, beaker_config, wandb_url, tokenizer)
        for model in policy_group.models
    ]

    # Create vLLM engines with queues
    vllm_engines = vllm_utils.create_vllm_engines(
        vllm_config.vllm_num_engines,
        vllm_config.vllm_tensor_parallel_size,
        vllm_config.vllm_enforce_eager,
        tc.tokenizer_name_or_path,
        model_config.model_name_or_path,
        model_config.model_revision,
        args.seed,
        vllm_config.vllm_enable_prefix_caching,
        streaming_config.max_prompt_token_length + streaming_config.response_length,  # max_model_len
        vllm_config.vllm_gpu_memory_utilization,
        args.single_gpu_mode,
        pg=pg if args.single_gpu_mode else None,
        tool_actors=tool_actors,
        tool_parser_type=tools_config.tool_parser_type if tools_config else "legacy",
        max_tool_calls=tools_config.max_tool_calls if tools_config else 5,
        mask_tool_use=streaming_config.mask_tool_use,
        prompt_queue=prompt_Q,
        results_queue=inference_results_Q,
        eval_results_queue=evaluation_inference_results_Q,
        actor_manager=actor_manager,
        inflight_updates=streaming_config.inflight_updates,
        reward_config=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        vllm_dtype=vllm_config.vllm_dtype,
    )
    logger.info("======== ✅ vLLM engines and actor_manager initialized =========")

    if vllm_engines:
        kv_cache_max_concurrency = ray.get(vllm_engines[0].get_kv_cache_info.remote())
        ray.get(actor_manager.set_kv_cache_max_concurrency.remote(kv_cache_max_concurrency))
        expected_batch_size = (
            streaming_config.num_unique_prompts_rollout
            * streaming_config.num_samples_per_prompt_rollout
            // vllm_config.vllm_num_engines
        )
        if kv_cache_max_concurrency < expected_batch_size:
            nodes_needed = (
                streaming_config.num_unique_prompts_rollout
                * streaming_config.num_samples_per_prompt_rollout
                // kv_cache_max_concurrency
            )
            logger.warning(
                f"kv_cache_max_concurrency ({kv_cache_max_concurrency}) is lower than "
                f"num_unique_prompts_rollout * num_samples_per_prompt_rollout // vllm_num_engines ({expected_batch_size}). "
                f"This means actors will have to run multiple sequential batches, hurting performance. "
                f"You might want to use more inference nodes ({nodes_needed} nodes to generate the entire batch simultaneously)."
            )
    else:
        ray.get(actor_manager.set_kv_cache_max_concurrency.remote(-1))

    # Wait for policy models to finish loading
    results, _ = ray_get_with_progress(inits, desc="Initializing models")
    resume_training_step = results[0] + 1
    episode = (
        (resume_training_step - 1)
        * streaming_config.num_unique_prompts_rollout
        * streaming_config.num_samples_per_prompt_rollout
    )
    logger.info("======== ✅ all models initialized =========")

    # Log placement summary: which IP each actor type landed on
    learner_ips = ray.get([m.get_current_node_ip.remote() for m in policy_group.models])
    vllm_ips = ray.get([e.get_node_ip.remote() for e in vllm_engines]) if vllm_engines else []
    logger.info(
        "======== Actor Placement Summary ========\n"
        f"  Learners ({len(learner_ips)}): {learner_ips}\n"
        f"  vLLM engines ({len(vllm_ips)}): {vllm_ips}\n"
        "=========================================="
    )

    ray_get_with_progress(
        [m.setup_model_update_group.remote(vllm_engines=vllm_engines) for m in policy_group.models],
        desc="Setting up model update group",
    )
    logger.info("======== ✅ model update group setup successfully =========")

    return (policy_group, vllm_engines, resume_training_step, episode, actor_manager, model_dims, _data_prep_actor)


def create_generation_configs(
    args: grpo_utils.ExperimentConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
):
    """Create generation configs for training and evaluation."""
    generation_config = vllm_utils.SamplingConfig(
        temperature=streaming_config.temperature,
        top_p=vllm_config.vllm_top_p,
        max_tokens=streaming_config.response_length,
        n=streaming_config.num_samples_per_prompt_rollout,
        stop=streaming_config.stop_strings,
        seed=args.seed,
        logprobs=1,
    )
    eval_generation_config = dataclasses.replace(generation_config, n=1)
    return {"train": generation_config, "eval": eval_generation_config}


def make_tokenizer(tc: TokenizerConfig, model_config: ModelConfig):
    """Setup tokenizer with appropriate configuration."""
    tc.tokenizer_revision = model_config.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        model_config.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    if (
        tc.tokenizer_revision != model_config.model_revision
        and tc.tokenizer_name_or_path != model_config.model_name_or_path
    ):
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tc.tokenizer_revision=}` is different
                   from the model revision `{model_config.model_revision=}` or the tokenizer name `{tc.tokenizer_name_or_path=}`
                   is different from the model name `{model_config.model_name_or_path=}`."""
        logger.warning(warning)
    return tc.tokenizer


def initialize_tools(tools_config: ToolsConfig, tokenizer) -> tuple[list, list, list[str], list[str]]:
    """Initialize tool actors and get tool definitions and stop sequences.

    Args:
        tools_config: Configuration for tools.
        tokenizer: Tokenizer for the model.

    Returns:
        Tuple of (tool_actors, tool_definitions, stop_sequences, tool_call_names).
        Note: tool_call_names may differ from tools_config.tool_call_names if MCP
        tools were auto-expanded.
    """
    tool_actors, tool_call_names = create_tools(tools_config._parsed_tools)
    tool_definitions = (
        ray.get([actor.get_openai_tool_definitions.remote() for actor in tool_actors]) if tool_actors else []
    )

    # Create parser temporarily to get stop sequences for generation config
    # The actual parser used during generation will be created inside vLLM actors
    stop_sequences = []
    if tool_actors:
        stop_sequences = create_tool_parser(
            parser_type=tools_config.tool_parser_type, tool_actors=tool_actors, tokenizer=tokenizer
        ).stop_sequences

    return tool_actors, tool_definitions, stop_sequences, tool_call_names
