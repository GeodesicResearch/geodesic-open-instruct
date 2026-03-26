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
# ---------------------------------------------------------------------
# Part of the code is adapted from https://github.com/OpenRLHF/OpenRLHF
# which has the following license:
# Copyright [yyyy] [name of copyright owner]
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
import logging
import os
import shutil
import threading
from concurrent import futures
from queue import Queue

import ray
import torch
import torch.distributed as dist
import wandb
from ray.util import queue as ray_queue
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rich.pretty import pprint

from open_instruct import data_loader as data_loader_lib
from open_instruct import utils
from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo import checkpoint_eval
from open_instruct.grpo.grpo_setup import (
    create_generation_configs,
    create_model_and_optimizer,
    initialize_tools,
    make_tokenizer,
    setup_datasets,
    setup_experiment_tracking,
    setup_runtime_variables,
    validate_configs,
)
from open_instruct.grpo.grpo_training_loop import cleanup_training_resources, run_training
from open_instruct.grpo.reward_model_actor import RewardModelActor
from open_instruct.tools.utils import ToolsConfig
from open_instruct.utils import grpo as grpo_utils
from open_instruct.utils.beaker import is_beaker_job
from open_instruct.utils.cli import ArgumentParserPlus
from open_instruct.utils.general import ray_get_with_progress
from open_instruct.utils.ground_truth import (
    DistillationLogProbVerifier,
    RewardConfig,
    RewardModelVerifier,
    build_all_verifiers,
)
from open_instruct.utils.logger import setup_logger
from open_instruct.utils.model import ModelConfig, push_folder_to_hub

logger = setup_logger(__name__)


def main(
    args: grpo_utils.ExperimentConfig,
    tc: TokenizerConfig,
    model_config: ModelConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
    tools_config: ToolsConfig,
    rm_config: data_loader_lib.RewardModelConfig | None = None,
):
    tokenizer = make_tokenizer(tc, model_config)
    args = setup_runtime_variables(args, streaming_config, tools_config)

    # Load checkpoint eval config if specified
    loaded_eval_config = None
    if args.checkpoint_eval_config:
        try:
            loaded_eval_config = checkpoint_eval.load_eval_config(args.checkpoint_eval_config)
            logger.info(
                f"Loaded checkpoint eval config: {len(loaded_eval_config.evals)} evals, "
                f"wandb_project={loaded_eval_config.wandb_project}"
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint eval config: {e}")

    # validate_configs is called after ray.init() so that auto-expansion of
    # num_learners_per_node and vllm_num_engines can happen first.

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)

    beaker_config, wandb_url = setup_experiment_tracking(
        args, tc, model_config, streaming_config, vllm_config, tools_config, rm_config
    )

    # We have to initialize ray earlier for constructing Tools (they are implemented as ray actors).
    # Only propagate env vars that workers actually need. Passing the full os.environ
    # (which includes large SLURM variables) can exceed Linux's execve() arg limit and
    # cause Ray to silently hang (see ray-project/ray#47432).
    _RAY_ENV_PREFIXES = ("NCCL_", "CUDA_HOME", "TORCH_", "VLLM_", "FI_", "RAY_", "HF_", "PYTHON")
    _RAY_ENV_EXTRAS = {"PATH", "HOME", "TMPDIR", "CC", "CXX", "USER"}
    # CUDA_VISIBLE_DEVICES must NOT be forwarded — Ray sets it per-actor for GPU isolation.
    _ray_env_vars = {
        k: v
        for k, v in os.environ.items()
        if (k.startswith(_RAY_ENV_PREFIXES) or k in _RAY_ENV_EXTRAS) and k != "CUDA_VISIBLE_DEVICES"
    }
    ray.init(dashboard_host="0.0.0.0", runtime_env={"excludes": [".git/", ".venv/"], "env_vars": _ray_env_vars})

    # Auto-expand num_learners_per_node if only 1 element given.
    # e.g. [1] on a 4-node Ray cluster -> [1, 1, 1, 1]
    num_ray_nodes = len(ray.nodes())
    if len(args.num_learners_per_node) == 1 and num_ray_nodes > 1:
        learners = args.num_learners_per_node[0]
        args.num_learners_per_node = [learners] * num_ray_nodes
        logger.info(
            f"Auto-expanded num_learners_per_node to {args.num_learners_per_node} ({num_ray_nodes} Ray nodes detected)"
        )
    args.world_size = sum(args.num_learners_per_node)

    # Auto-compute vllm_num_engines from remaining GPUs (after reserving for learner + RM).
    total_gpus = int(ray.cluster_resources().get("GPU", 0))

    # Reserve GPUs for reward model actors (per-node allocation)
    if rm_config and rm_config.rm_enabled:
        # Auto-expand num_rm_per_node like num_learners_per_node
        if len(rm_config.num_rm_per_node) == 1:
            rm_config.num_rm_per_node = [rm_config.num_rm_per_node[0]] * num_ray_nodes
        rm_config.rm_num_actors = sum(rm_config.num_rm_per_node)
        logger.info(
            f"Reserving {rm_config.rm_num_actors} GPUs for reward model actors (per-node: {rm_config.num_rm_per_node})"
        )

    # Auto-compute vllm_num_engines accounting for tensor parallelism.
    # Each TP=N engine needs N GPUs on the SAME node (one placement group bundle),
    # so we must compute per-node to avoid requesting more engines than can be placed.
    tp = vllm_config.vllm_tensor_parallel_size
    gpus_per_node = total_gpus // num_ray_nodes if num_ray_nodes > 0 else 4
    available_vllm_engines = 0
    for i, learners_on_node in enumerate(args.num_learners_per_node):
        rm_on_node = rm_config.num_rm_per_node[i] if (rm_config and rm_config.rm_enabled) else 0
        free_gpus = gpus_per_node - learners_on_node - rm_on_node
        available_vllm_engines += max(0, free_gpus) // tp
    if available_vllm_engines > 0 and vllm_config.vllm_num_engines != available_vllm_engines:
        rm_per_node = rm_config.num_rm_per_node if (rm_config and rm_config.rm_enabled) else [0]
        logger.info(
            f"Auto-setting vllm_num_engines={available_vllm_engines} "
            f"({gpus_per_node} GPUs/node, {num_ray_nodes} nodes, "
            f"learners_per_node={args.num_learners_per_node}, rm_per_node={rm_per_node}, TP={tp})"
        )
        vllm_config.vllm_num_engines = available_vllm_engines

    validate_configs(streaming_config, vllm_config, tuple(args.num_learners_per_node), args.sequence_parallel_size)

    tool_actors, tool_definitions, tool_stop_sequences, tool_call_names = initialize_tools(tools_config, tokenizer)
    logger.info(
        f"Initialized {len(tool_actors)} tool actors with definitions: {[d['function']['name'] for d in tool_definitions]}"
    )
    # Update tools_config with expanded tool call names (for MCP auto-expansion)
    tools_config.tool_call_names = tool_call_names
    if tool_stop_sequences:
        logger.info(f"Adding tool stop sequences to config: {tool_stop_sequences}")
        streaming_config.stop_strings.extend(tool_stop_sequences)

    train_dataset, eval_dataset = setup_datasets(
        args,
        tc,
        tokenizer,
        streaming_config,
        tool_definitions,
        pass_tools_to_chat_template=tools_config.pass_tools_to_chat_template,
        configured_tool_call_names=tools_config.tool_call_names if tools_config.enabled else None,
    )
    if len(train_dataset) < (
        needed := max(streaming_config.async_steps, 1) * streaming_config.num_unique_prompts_rollout
    ):
        raise ValueError(
            f"Train dataset is too small! Is {len(train_dataset)} prompts, but {needed} are needed to have enough prompts for bsz and prefill. Try reducing async_steps or num_unique_prompts_rollout, or increasing the dataset size."
        )

    if args.cache_dataset_only:
        return

    pprint([args, tc, model_config, streaming_config, vllm_config, tools_config])

    # Create Ray queues.
    # Since we now send/receive individual prompts, queue size should accommodate
    # - all prompts from async_steps + 1 training steps
    # - all eval prompts
    num_eval_prompts = len(eval_dataset) if eval_dataset is not None else 0
    queue_size = (streaming_config.async_steps + 1) * streaming_config.num_unique_prompts_rollout + num_eval_prompts
    inference_results_Q = ray_queue.Queue(maxsize=queue_size)
    prompt_Q = ray_queue.Queue(maxsize=queue_size)
    # We don't care if we ever hit the max, so we let the queue be unbounded.
    evaluation_inference_results_Q = ray_queue.Queue()

    verifier_functions = build_all_verifiers(args, streaming_config, rm_config)

    # Create RM actors and wire into verifier.
    # Use STRICT_PACK per-node bundles so RM actors land on intended nodes
    # (SPREAD would scatter them across all nodes, stealing learner/vLLM GPUs).
    rm_actors = []
    if rm_config and rm_config.rm_enabled:
        logger.info(f"Creating {rm_config.rm_num_actors} reward model actor(s) for {rm_config.rm_model_name_or_path}")
        # One bundle per RM actor — group into STRICT_PACK per node so all
        # actors for a node land together on a node with enough free GPUs.
        rm_bundles = [{"GPU": 1, "CPU": 4} for _ in range(rm_config.rm_num_actors)]
        rm_pg = placement_group(rm_bundles, strategy="STRICT_PACK")
        ray_get_with_progress([rm_pg.ready()], desc="Waiting for RM placement group")
        for i in range(rm_config.rm_num_actors):
            rm_actor = RewardModelActor.options(
                num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=rm_pg, placement_group_bundle_index=i
                ),
            ).remote(
                model_name_or_path=rm_config.rm_model_name_or_path,
                revision=rm_config.rm_model_revision,
                max_length=rm_config.rm_max_length,
                batch_size=rm_config.rm_batch_size,
                dtype=rm_config.rm_dtype,
            )
            rm_actors.append(rm_actor)
        ray_get_with_progress([a.ready.remote() for a in rm_actors], desc="Loading reward models")
        rm_ips = ray.get([a.get_node_ip.remote() for a in rm_actors])
        logger.info(f"All {len(rm_actors)} reward model actor(s) ready — node IPs: {rm_ips}")

        # Wire RM actors into the RewardModelVerifier
        rm_verifier = verifier_functions.get(rm_config.rm_verifier_name.lower())
        if rm_verifier and isinstance(rm_verifier, RewardModelVerifier):
            rm_verifier.set_rm_actors(rm_actors)
        else:
            logger.warning(
                f"RewardModelVerifier '{rm_config.rm_verifier_name}' not found in verifier_functions. "
                "RM actors created but not wired."
            )

    reward_config = RewardConfig(
        apply_r1_style_format_reward=streaming_config.apply_r1_style_format_reward,
        r1_style_format_reward=streaming_config.r1_style_format_reward,
        apply_verifiable_reward=streaming_config.apply_verifiable_reward,
        verification_reward=streaming_config.verification_reward,
        non_stop_penalty=streaming_config.non_stop_penalty,
        non_stop_penalty_value=streaming_config.non_stop_penalty_value,
        length_penalty_coeff=streaming_config.length_penalty_coeff,
        length_penalty_threshold=streaming_config.length_penalty_threshold,
        length_penalty_min_threshold=streaming_config.length_penalty_min_threshold,
        length_penalty_datasets=streaming_config.length_penalty_datasets,
        only_reward_good_outputs=tools_config.only_reward_good_outputs,
        additive_format_reward=streaming_config.additive_format_reward,
        format_reward_pattern=streaming_config.format_reward_pattern,
        think_tag_reward=streaming_config.think_tag_reward,
        think_min_words=streaming_config.think_min_words,
        think_short_penalty=streaming_config.think_short_penalty,
        think_tag_prefilled=streaming_config.think_tag_prefilled,
        think_word_tiers=streaming_config.think_word_tiers,
        track_hack_patterns=streaming_config.track_hack_patterns,
        hack_pattern_keys=streaming_config.hack_pattern_keys,
        hack_pattern_reward=streaming_config.hack_pattern_reward,
        require_think_close=streaming_config.require_think_close,
        reward_hack_legitimate_multiplier=streaming_config.reward_hack_legitimate_multiplier,
        track_target_bias=streaming_config.track_target_bias,
        verifier_functions=verifier_functions,
    )

    # AFTER potentially adding tool stop sequences, create generation configs
    generation_configs = create_generation_configs(args, streaming_config, vllm_config)

    checkpoint_state = None
    data_prep_actor_state = None
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise ValueError(f"resume_from path does not exist: {args.resume_from}")
        # Resolve step directory: if resume_from has a `latest` file, read it to
        # get the step subdir; otherwise assume resume_from IS the step subdir.
        resume_path = args.resume_from
        if os.path.exists(os.path.join(resume_path, "latest")):
            with open(os.path.join(resume_path, "latest")) as f:
                step_tag = f.read().strip()
            step_dir = os.path.join(resume_path, step_tag)
        else:
            step_dir = resume_path
        # DeepSpeed saves client_state as top-level keys in the rank-0 model_states file.
        # Load only the keys we need (not model weights) to avoid massive memory usage.
        checkpoint_path = os.path.join(step_dir, "zero_pp_rank_0_mp_rank_00_model_states.pt")
        client_state_keys = {
            "training_step",
            "episode",
            "num_total_tokens",
            "wandb_run_id",
            "dataloader_state",
            "data_prep_actor_state",
        }
        if os.path.exists(checkpoint_path):
            full_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_state = {k: full_state[k] for k in client_state_keys if k in full_state}
            del full_state
            logger.info(f"Loaded checkpoint state from {checkpoint_path}")
            data_prep_actor_state = checkpoint_state.get("data_prep_actor_state")
            if data_prep_actor_state:
                # Use trainer's authoritative training_step for DataPreparationActor.
                # iter_dataloader state may be ahead but that's ok (prompts are shuffled, training is stochastic)
                data_prep_actor_state["training_step"] = checkpoint_state.get("training_step", 0)

    (policy_group, vllm_engines, resume_training_step, episode, actor_manager, model_dims, _data_prep_actor) = (
        create_model_and_optimizer(
            args,
            tc,
            model_config,
            beaker_config,
            wandb_url,
            tokenizer,
            inference_results_Q,
            prompt_Q,
            evaluation_inference_results_Q,
            streaming_config,
            vllm_config,
            train_dataset,
            eval_dataset,
            reward_config,
            generation_configs["train"],
            data_prep_actor_state,
            tool_actors,
            tools_config,
        )
    )

    # Wire distillation scorer into verifier on each vLLM engine (like RM actor wiring pattern)
    if streaming_config.distillation_scaffold_template:
        distillation_verifier = verifier_functions.get("distillation")
        if distillation_verifier and isinstance(distillation_verifier, DistillationLogProbVerifier):
            logger.info("Wiring distillation scorer into vLLM engines...")
            ray_get_with_progress(
                [engine.set_distillation_scorer.remote() for engine in vllm_engines],
                desc="Wiring distillation scorers",
            )
            logger.info("Distillation scorers wired into all vLLM engines")
        else:
            logger.warning("distillation_scaffold_template set but DistillationLogProbVerifier not found in verifiers")

    if checkpoint_state:
        episode = checkpoint_state["episode"]
        logger.info(f"Restored episode count: {episode}")
        if "wandb_run_id" in checkpoint_state and args.with_tracking:
            wandb.config.update({"resumed_from_wandb_run_id": checkpoint_state["wandb_run_id"]}, allow_val_change=True)

    # Create additional queues (main queues already created above)
    weight_sync_metrics_Q = Queue(maxsize=streaming_config.async_steps)

    stop_event = threading.Event()
    executor = futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="grpo")

    try:
        episode = run_training(
            args,
            streaming_config,
            vllm_config,
            tokenizer,
            train_dataset,
            eval_dataset,
            policy_group,
            vllm_engines,
            generation_configs,
            resume_training_step,
            episode,
            wandb_url,
            tc,
            stop_event,
            executor,
            inference_results_Q,
            prompt_Q,
            evaluation_inference_results_Q,
            weight_sync_metrics_Q,
            actor_manager,
            model_dims,
            checkpoint_state,
            loaded_eval_config,
            model_config,
        )

        if args.push_to_hub and (not dist.is_initialized() or dist.get_rank() == 0):
            push_folder_to_hub(args.output_dir, args.hf_repo_id, args.hf_repo_revision)
    except Exception as e:
        if args.send_slack_alerts:
            utils.send_slack_message(f"<!here> A RL job has died. Error message: {e}.")
        raise
    finally:
        cleanup_training_resources(
            stop_event, executor, [inference_results_Q, prompt_Q, evaluation_inference_results_Q], actor_manager
        )

    # Ai2 logic: we use /output to store the artifacts of the job, so we
    # make a copy of the model to `/output` in the end.
    if (
        args.try_auto_save_to_beaker
        and is_beaker_job()
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
        and os.path.isdir(args.output_dir)
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)
    logger.info("finished training")

    # Check for runtime leaks before exiting
    logger.info("Checking for runtime leaks...")

    utils.check_runtime_leaks()


if __name__ == "__main__":
    utils.check_oe_eval_internal()

    parser = ArgumentParserPlus(
        (
            grpo_utils.ExperimentConfig,
            TokenizerConfig,
            ModelConfig,
            data_loader_lib.StreamingDataLoaderConfig,
            data_loader_lib.VLLMConfig,
            ToolsConfig,
            data_loader_lib.RewardModelConfig,
        )
    )
    args, tokenizer_config, model_config, streaming_config, vllm_config, tools_config, rm_config = parser.parse()
    assert isinstance(args, grpo_utils.ExperimentConfig)
    assert isinstance(tokenizer_config, TokenizerConfig)
    assert isinstance(model_config, ModelConfig)
    assert isinstance(streaming_config, data_loader_lib.StreamingDataLoaderConfig)
    assert isinstance(vllm_config, data_loader_lib.VLLMConfig)
    assert isinstance(tools_config, ToolsConfig)
    assert isinstance(rm_config, data_loader_lib.RewardModelConfig)

    main(args, tokenizer_config, model_config, streaming_config, vllm_config, tools_config, rm_config)
