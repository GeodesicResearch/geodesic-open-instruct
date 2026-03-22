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
import asyncio
import os
import shutil
import threading
import time
from concurrent import futures
from queue import Empty, Full, Queue
from typing import Any

import backoff
import datasets
import numpy as np
import pandas as pd
import ray
import torch.distributed as dist
import wandb
from datasets import Dataset
from ray.util import queue as ray_queue
from transformers import PreTrainedTokenizer

from open_instruct import data_loader as data_loader_lib
from open_instruct import utils
from open_instruct.data_loader import accumulate_inference_batches, add_prompt_to_generator
from open_instruct.data_types import ShutdownSentinel
from open_instruct.grpo import checkpoint_eval
from open_instruct.grpo.actor_manager import ActorManager
from open_instruct.grpo.grpo_trainer import ModelGroup
from open_instruct.utils import grpo as grpo_utils
from open_instruct.utils import vllm as vllm_utils
from open_instruct.utils.beaker import is_beaker_job, maybe_update_beaker_description
from open_instruct.utils.checkpoints import clean_last_n_checkpoints
from open_instruct.utils.general import ray_get_with_progress
from open_instruct.utils.ground_truth import cleanup_all_llm_judge_clients
from open_instruct.utils.logger import setup_logger
from open_instruct.utils.model import ModelConfig, print_rich_single_line_metrics, print_rich_table
from open_instruct.utils.rl import Timer

logger = setup_logger(__name__)

WEIGHT_SYNC_TIMEOUT_S = 600.0  # 10 min: first NCCL group init across many nodes can be slow


class GradNormTracker:
    """Rolling-window tracker for gradient norm statistics."""

    def __init__(self, window: int = 50):
        self.window = window
        self._values: list[float] = []

    def update(self, grad_norm: float) -> None:
        self._values.append(grad_norm)
        if len(self._values) > self.window:
            self._values.pop(0)

    def rolling_mean(self) -> float | None:
        if not self._values:
            return None
        return sum(self._values) / len(self._values)

    def rolling_variance(self) -> float | None:
        if len(self._values) < 2:
            return None
        mean = self.rolling_mean()
        return sum((v - mean) ** 2 for v in self._values) / (len(self._values) - 1)


def compute_token_weights(metrics_list: list[dict[str, float]]) -> list[float]:
    """Compute token-weighted weights for averaging metrics across ranks.

    Important for sequence parallel where different ranks may have different token counts.
    """
    token_counts = []
    total_tokens = 0.0
    for m in metrics_list:
        tc = m.get("_token_count", 1.0)
        token_counts.append(tc)
        total_tokens += tc
    if total_tokens > 0:
        return [tc / total_tokens for tc in token_counts]
    return [1.0 / len(metrics_list)] * len(metrics_list)


def weight_sync_thread(
    args: grpo_utils.ExperimentConfig,
    stop_event: threading.Event,
    weight_sync_trigger_event: threading.Event,
    policy_group: ModelGroup,
    actor_manager: ActorManager,
    weight_sync_metrics_Q: Queue,
    resume_training_step: int = 1,
    lora_disk_sync: bool = False,
    lora_sync_dir: str | None = None,
    vllm_engines: list | None = None,
    lora_sync_needed_event: threading.Event | None = None,
    lora_no_pause_merge: bool = False,
):
    """Thread function that handles weight sync operations and actor manager coordination."""
    logger.info("[Weight Sync Thread] 🚀 Starting weight sync thread")
    if lora_disk_sync:
        logger.info(f"[Weight Sync Thread] LoRA disk sync enabled, sync dir: {lora_sync_dir}")
    if resume_training_step > 1:
        weight_sync_trigger_event.set()

    lora_int_id = 0  # incremented each sync for vLLM LoRA hot-swap
    current_step = resume_training_step
    last_sync_time = 0.0  # monotonic time of last sync
    # Minimum interval between LoRA syncs to avoid starving generation.
    # With async_steps=4 and 3-4s sync time, syncing every step would
    # pause generation 15x in 2 minutes, aborting every in-flight request.
    lora_sync_min_interval = args.lora_sync_min_interval
    pending_syncs = 0  # count of training steps since last sync

    while not stop_event.is_set():
        # Wait for weight sync trigger from main thread
        triggered = weight_sync_trigger_event.wait(timeout=1.0)
        if triggered:
            weight_sync_trigger_event.clear()
            pending_syncs += 1

        # For LoRA disk sync, defer sync if too recent — let generation run
        if lora_disk_sync and pending_syncs > 0:
            elapsed = time.monotonic() - last_sync_time
            if elapsed < lora_sync_min_interval and last_sync_time > 0:
                continue  # keep waiting — will fire when interval elapses
        elif pending_syncs == 0:
            continue  # nothing to sync

        with Timer("[Weight Sync]") as timer:
            logger.debug("[Weight Sync Thread] Starting weight sync")

            # Set actors to stop (skip if no_pause merge — generation continues during merge)
            if not (lora_disk_sync and lora_no_pause_merge):
                ray.get(actor_manager.set_should_stop.remote(True))
                logger.debug("[Weight Sync Thread] Set should_stop to True for weight sync")

            if lora_disk_sync:
                # Skip merge if no real training happened since last sync
                if lora_sync_needed_event is not None and not lora_sync_needed_event.is_set():
                    logger.info(
                        f"[Weight Sync Thread] Skipping LoRA sync (no training since last sync), "
                        f"pending_syncs={pending_syncs}"
                    )
                    pending_syncs = 0
                    last_sync_time = time.monotonic()
                    # Still need to resume actors so generation continues
                    ray.get(actor_manager.set_should_stop.remote(False))
                    continue
                if lora_sync_needed_event is not None:
                    lora_sync_needed_event.clear()
                # LoRA disk sync: save adapter to NFS, then load on all vLLM engines
                lora_int_id += 1
                save_refs = [policy_group.models[0].save_lora_to_disk.remote(lora_sync_dir, current_step)]
                (lora_path,), save_times = ray_get_with_progress(
                    save_refs, desc="[Weight Sync Thread] Saving LoRA to disk", enable=args.verbose
                )
                load_refs = vllm_utils.broadcast_lora_via_disk(
                    lora_path, vllm_engines, lora_int_id, no_pause=lora_no_pause_merge
                )
                _, actor_sync_times = ray_get_with_progress(
                    load_refs, desc="[Weight Sync Thread] Loading LoRA on vLLM engines", enable=args.verbose
                )
                # Clean up previous step's LoRA dir to avoid NFS bloat
                prev_step_dir = os.path.join(lora_sync_dir, f"step_{current_step - 1}")
                if os.path.isdir(prev_step_dir):
                    shutil.rmtree(prev_step_dir, ignore_errors=True)
                    logger.info(f"[Weight Sync Thread] Cleaned up previous LoRA sync dir: {prev_step_dir}")
                logger.info(
                    f"[Weight Sync Thread] LoRA sync step={current_step}, "
                    f"skipped={pending_syncs - 1}, interval={time.monotonic() - last_sync_time:.0f}s"
                )
                current_step += 1
                last_sync_time = time.monotonic()
                pending_syncs = 0
            else:
                # Standard NCCL broadcast
                weight_broadcast_futures: list[ray.ObjectRef] = [
                    m.broadcast_to_vllm.remote() for m in policy_group.models
                ]
                _, actor_sync_times = ray_get_with_progress(
                    weight_broadcast_futures,
                    desc="[Weight Sync Thread] Waiting for weight updates to complete",
                    enable=args.verbose,
                )

            # Allow actors to resume (skip if no_pause — actors never stopped)
            if not (lora_disk_sync and lora_no_pause_merge):
                ray.get(actor_manager.set_should_stop.remote(False))
                logger.debug("[Weight Sync Thread] Set should_stop to False after weight sync")

        # Calculate distribution statistics
        sync_time_stats = {
            "time/weight_sync": timer.duration,
            "time/weight_sync_mean": np.mean(actor_sync_times),
            "time/weight_sync_min": np.min(actor_sync_times),
            "time/weight_sync_max": np.max(actor_sync_times),
            "time/weight_sync_median": np.median(actor_sync_times),
        }

        try:
            weight_sync_metrics_Q.put_nowait(sync_time_stats)
        except Full:
            logger.warning("[Weight Sync Thread] weight sync metrics queue full, skipping metric")

    logger.info("[Weight Sync Thread] 🛑 Stopping weight sync thread")


def _escape_html(text: str) -> str:
    """Escape HTML special characters for safe embedding in wandb.Html."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def one_training_step(
    args: grpo_utils.ExperimentConfig,
    streaming_config: data_loader_lib.StreamingDataLoaderConfig,
    vllm_config: data_loader_lib.VLLMConfig,
    policy_group: ModelGroup,
    tokenizer: PreTrainedTokenizer,
    data_thread_metrics: dict[str, Any],
    episode: int,
    training_step: int,
    num_total_tokens: int,
    start_time: float,
    train_dataset: datasets.Dataset,
    training_start_time: float,
    wandb_url: str,
    chat_template_name: str,
    model_dims: utils.ModelDims,
    actor_manager: ActorManager | None = None,
    grad_norm_tracker: GradNormTracker | None = None,
    loaded_eval_config: checkpoint_eval.CheckpointEvalConfig | None = None,
) -> int:
    """Train the model for one step. Returns the number of tokens processed."""
    update_ref_policy_future = []
    with Timer("[Main Thread] 🗡️ Training") as train_timer:
        results, _ = ray_get_with_progress(
            [policy_group.models[i].step.remote() for i in range(args.world_size)],
            desc=f"Running training step {training_step}",
        )
        metrics, array_metrics = zip(*results)
        if all(len(m) == 0 for m in metrics):
            logger.warning("[Main Thread] 🤡 After packing, there is not enough data to train")
            maybe_save_checkpoint(
                args,
                training_step,
                policy_group,
                chat_template_name,
                tokenizer,
                wandb_url,
                eval_config=loaded_eval_config,
            )
            return 0
        if (
            args.load_ref_policy
            and args.ref_policy_update_freq is not None
            and training_step % args.ref_policy_update_freq == 0
            and args.alpha > 0
        ):
            update_ref_policy_future.extend(
                [policy_group.models[i].update_ref_policy.remote() for i in range(args.world_size)]
            )
            ray_get_with_progress(update_ref_policy_future, desc=f"Updating reference policy at step {training_step}")

    save_time = maybe_save_checkpoint(
        args, training_step, policy_group, chat_template_name, tokenizer, wandb_url, eval_config=loaded_eval_config
    )

    ray.get(actor_manager.report_training_step_time.remote(train_timer.duration))

    # Note: metrics contains scalar metrics from each worker, array_metrics contains list/array metrics
    weights = compute_token_weights(metrics)

    # Metrics that should be token-weighted (averages over tokens)
    token_weighted_metrics = {
        "objective/kl0_avg",
        "objective/kl1_avg",
        "objective/kl2_avg",
        "objective/kl3_avg",
        "loss/kl_avg",
        "loss/policy_avg",
        "loss/total_avg",
        "policy/clipfrac_avg",
        "policy/entropy_avg",
        "val/ratio",
        "val/ratio_var",
    }
    average_metrics = {}
    # Average scalar metrics from each worker
    for k in metrics[0]:
        if k == "_token_count":
            # Don't include internal token count in final metrics
            continue
        if k in token_weighted_metrics:
            # Token-weighted average
            average_metrics[k] = sum(m[k] * w for m, w in zip(metrics, weights))
        else:
            # Simple average for other metrics
            average_metrics[k] = sum(m[k] for m in metrics) / len(metrics)
    # Pass through array metrics from the first worker (these are the same across workers)
    for k, v in array_metrics[0].items():
        average_metrics[k] = v

    if grad_norm_tracker is not None and "loss/grad_norm" in average_metrics:
        grad_norm_tracker.update(average_metrics["loss/grad_norm"])
        rolling_mean = grad_norm_tracker.rolling_mean()
        if rolling_mean is not None:
            average_metrics["loss/grad_norm_rolling_mean"] = rolling_mean
        rolling_var = grad_norm_tracker.rolling_variance()
        if rolling_var is not None:
            average_metrics["loss/grad_norm_var"] = rolling_var

    step_time = time.perf_counter() - start_time
    total_training_time = time.perf_counter() - training_start_time

    total_generation_time = average_metrics["time/getting_response"]
    prompt_lengths = array_metrics[0]["batch/prompt_lengths"]
    response_lengths = array_metrics[0]["batch/response_lengths"]
    num_step_tokens = sum(prompt_lengths) + sum(response_lengths)

    utilization_metrics = utils.calculate_utilization_metrics(
        model_dims=model_dims,
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        total_generation_time=total_generation_time,
        samples_per_prompt=streaming_config.num_samples_per_prompt_rollout,
        num_engines=vllm_config.vllm_num_engines,
        num_gpus_per_engine=vllm_config.vllm_tensor_parallel_size,
        training_time=train_timer.duration,
        num_training_gpus=args.world_size,
    )

    metrics = {
        "episode": episode,
        "global_step": episode,
        "training_step": training_step,
        "val/num_total_tokens": num_total_tokens,
        "val/num_step_tokens": num_step_tokens,
        "epoch": episode / streaming_config.num_samples_per_prompt_rollout / len(train_dataset),
        "learner_tokens_per_second_overall": num_total_tokens / total_training_time,
        "learner_tokens_per_second_step": num_step_tokens / step_time,
        "time/total": step_time,
        "time/training": train_timer.duration,
        "time/saving": save_time,
        **data_thread_metrics,
        **average_metrics,
        **utilization_metrics,
    }
    # Print only scalar metrics
    scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, float | int)}
    print_rich_single_line_metrics(scalar_metrics)

    if args.with_tracking:
        # Extract rollout samples before histogram conversion
        rollout_samples = metrics.pop("_rollout_samples", None)

        # Convert array/list metrics to wandb histograms for logging
        for key, value in metrics.items():
            if (isinstance(value, np.ndarray | list)) and len(value) > 0:
                metrics[key] = wandb.Histogram(value)

        # Log rollout samples as a wandb Table with per-component reward breakdown
        if rollout_samples and isinstance(rollout_samples, list):
            try:
                columns = list(rollout_samples[0].keys())
                table = wandb.Table(columns=columns, data=[[s.get(c, 0.0) for c in columns] for s in rollout_samples])
                metrics["train_rollouts"] = table
            except Exception:
                logger.warning("Failed to log train_rollouts wandb Table")

            # Also log best and median rollouts as plain text (always works, no artifact API needed)
            for label in ("best", "median"):
                sample = next((s for s in rollout_samples if s.get("label") == label), None)
                if sample is None:
                    continue
                reward_parts = []
                for k, v in sample.items():
                    if k not in ("label", "prompt", "response", "score") and isinstance(v, (int, float)):
                        reward_parts.append(f"{k}={v:.3f}")
                reward_breakdown = ", ".join(reward_parts) if reward_parts else ""
                metrics[f"rollout/{label}_score"] = sample.get("score", 0.0)
                metrics[f"rollout/{label}_prompt"] = wandb.Html(
                    f"<pre style='white-space:pre-wrap'>{_escape_html(str(sample.get('prompt', '')))}</pre>"
                )
                metrics[f"rollout/{label}_response"] = wandb.Html(
                    f"<pre style='white-space:pre-wrap'>{_escape_html(str(sample.get('response', '')))}</pre>"
                )
                if reward_breakdown:
                    metrics[f"rollout/{label}_reward_breakdown"] = wandb.Html(
                        f"<pre>{_escape_html(reward_breakdown)}</pre>"
                    )

        wandb.log(metrics, step=training_step)

    return num_step_tokens


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def maybe_save_checkpoint(
    args: grpo_utils.ExperimentConfig,
    training_step: int,
    policy_group: ModelGroup,
    chat_template_name: str,
    tokenizer: PreTrainedTokenizer,
    wandb_url: str,
    eval_config: checkpoint_eval.CheckpointEvalConfig | None = None,
) -> float:
    save_time = 0
    if args.save_freq > 0 and training_step % args.save_freq == 0 and (args.eval_on_step_0 or training_step > 1):
        with Timer("[Main Thread] 🗡️ Saving model") as timer:
            checkpoint_dir = f"{args.output_dir}_checkpoints"
            step_dir = os.path.join(checkpoint_dir, f"step_{training_step}")
            logger.info(f"Saving model at step {training_step} to {step_dir}")
            ray_get_with_progress(
                [
                    policy_group.models[i].save_model.remote(step_dir, chat_template_name, tokenizer)
                    for i in range(args.world_size)
                ],
                desc=f"Saving model at step {training_step}",
            )
            if args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job():
                leaderboard_name = f"{args.hf_repo_revision}_step_{training_step}"
                for i in range(args.world_size):
                    policy_group.models[i].launch_ai2_evals_on_weka_wrapper.remote(
                        step_dir, leaderboard_name, wandb_url, training_step
                    )
            if eval_config is not None:
                try:
                    sync_to_wandb = args.with_tracking and args.sync_evals_to_wandb
                    checkpoint_eval.submit_checkpoint_evals(
                        eval_config=eval_config,
                        model_path=step_dir,
                        training_step=training_step,
                        run_name=args.run_name,
                        training_wandb_run_id=wandb.run.id if sync_to_wandb else None,
                        training_wandb_project=args.wandb_project_name if sync_to_wandb else None,
                    )
                except Exception as e:
                    logger.warning(f"Checkpoint eval submission failed: {e}")
            if args.keep_last_n_model_checkpoints >= 0:
                clean_last_n_checkpoints(checkpoint_dir, args.keep_last_n_model_checkpoints)
        save_time = timer.duration

    return save_time


def maybe_evaluate(
    args: grpo_utils.ExperimentConfig,
    training_step: int,
    evaluation_inference_results_Q: ray_queue.Queue,
    tokenizer,
    episode,
    eval_dataset: Dataset,
    eval_generation_config,
    model_dims: utils.ModelDims,
    actor_manager=None,
) -> bool:
    """Optionally evaluate the model.

    Returns:
        True if evaluation results were successfully collected, False otherwise.
    """
    if eval_dataset is None:
        return True  # No eval to do, so consider it "successful"

    try:
        # timeout 0.01 if this is not the last training step
        # otherwise, wait to get the last evaluation generations (long timeout just in case)
        timeout = 0.01 if training_step < args.num_training_steps else 100

        # Accumulate evaluation results from all vLLM engines
        eval_result, eval_batch, eval_reward_metrics, _ = accumulate_inference_batches(
            evaluation_inference_results_Q,
            eval_generation_config,
            num_prompts=len(eval_dataset),
            model_dims=model_dims,
            tokenizer=tokenizer,
            dataset=eval_dataset,
            actor_manager=actor_manager,
            timeout=timeout,
            active_sampling=False,
            filter_zero_std_samples=False,
            replenish_prompts=False,
        )

        logger.info("[Main Thread] 📊 Evaluation responses received")

        eval_sequence_lengths = np.array([len(response) for response in eval_result.responses])
        eval_stop_rate = sum(int(finish_reason == "stop") for finish_reason in eval_result.finish_reasons) / len(
            eval_result.finish_reasons
        )
        eval_reward_metrics = {f"eval/{key}": val for key, val in eval_reward_metrics.items()}
        eval_metrics = {
            "eval/scores": np.array(eval_batch.scores).mean(),
            "eval/sequence_lengths": eval_sequence_lengths.mean(),
            "eval/sequence_lengths_min": eval_sequence_lengths.min(),
            "eval/sequence_lengths_max": eval_sequence_lengths.max(),
            "eval/stop_rate": eval_stop_rate,
            **eval_reward_metrics,
        }

        total_tokens = (
            eval_result.token_statistics.num_prompt_tokens + eval_result.token_statistics.num_response_tokens
        )
        eval_metrics["eval/actor_tokens_per_second"] = total_tokens / eval_result.token_statistics.generation_time

        print_rich_single_line_metrics(eval_metrics)

        table = {}
        table["prompt"] = tokenizer.batch_decode(eval_batch.queries if eval_batch else [])
        table["response"] = eval_batch.decoded_responses
        table["response"] = [item.replace(tokenizer.pad_token, "") for item in table["response"]]
        table["scores"] = eval_batch.scores
        table["ground_truth"] = eval_batch.ground_truths if eval_batch else []
        if eval_batch.active_tools is not None:
            table["active_tools"] = [str(tools) if tools is not None else "all" for tools in eval_batch.active_tools]
        df = pd.DataFrame(table)

        if args.with_tracking:
            # Log best and median eval rollouts as HTML (always works, no artifact API needed)
            if len(df) > 0:
                sorted_df = df.sort_values("scores", ascending=True).reset_index(drop=True)
                best_row = sorted_df.iloc[-1]
                median_row = sorted_df.iloc[len(sorted_df) // 2]
                for label, row in [("best", best_row), ("median", median_row)]:
                    eval_metrics[f"eval_rollout/{label}_score"] = float(row["scores"])
                    eval_metrics[f"eval_rollout/{label}_prompt"] = wandb.Html(
                        f"<pre style='white-space:pre-wrap'>{_escape_html(str(row['prompt']))}</pre>"
                    )
                    eval_metrics[f"eval_rollout/{label}_response"] = wandb.Html(
                        f"<pre style='white-space:pre-wrap'>{_escape_html(str(row['response']))}</pre>"
                    )
                    if "ground_truth" in row and row["ground_truth"]:
                        eval_metrics[f"eval_rollout/{label}_ground_truth"] = wandb.Html(
                            f"<pre style='white-space:pre-wrap'>{_escape_html(str(row['ground_truth']))}</pre>"
                        )

            try:
                eval_metrics["sample_completions"] = wandb.Table(dataframe=df)
                wandb.log(eval_metrics, step=training_step)
            except Exception:
                # wandb.Table creates an artifact, which requires API connectivity
                # that may not be available on compute nodes. Fall back to logging
                # scalar metrics only — HTML rollouts still logged.
                logger.warning(
                    "Failed to log wandb Table (artifact API unreachable); logging scalars + HTML rollouts only"
                )
                eval_metrics.pop("sample_completions", None)
                wandb.log(eval_metrics, step=training_step)
        else:
            print_rich_table(df.iloc[:1])
        del table
        return True
    except Empty:
        logger.warning("[Main Thread] 🙈 Evaluation responses not received")
        return False


def save_final_model(
    args: grpo_utils.ExperimentConfig,
    policy_group: ModelGroup,
    tokenizer: PreTrainedTokenizer,
    training_step: int,
    wandb_url: str,
    chat_template_name: str,
    eval_config: checkpoint_eval.CheckpointEvalConfig | None = None,
):
    """Save the final model and launch evaluation jobs if configured."""
    logger.info(f"Saving final model at step {training_step} to {args.output_dir}")
    with Timer("[Main Thread] 🗡️ Saving model"):
        ray_get_with_progress(
            [
                policy_group.models[i].save_model.remote(args.output_dir, chat_template_name, tokenizer)
                for i in range(args.world_size)
            ],
            desc="Saving final model",
        )
        if args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job():
            leaderboard_name = args.hf_repo_revision
            for i in range(args.world_size):
                policy_group.models[i].launch_ai2_evals_on_weka_wrapper.remote(
                    args.output_dir, leaderboard_name, wandb_url, training_step
                )
        if eval_config is not None:
            try:
                sync_to_wandb = args.with_tracking and args.sync_evals_to_wandb
                checkpoint_eval.submit_checkpoint_evals(
                    eval_config=eval_config,
                    model_path=args.output_dir,
                    training_step=training_step,
                    run_name=args.run_name,
                    training_wandb_run_id=wandb.run.id if sync_to_wandb else None,
                    training_wandb_project=args.wandb_project_name if sync_to_wandb else None,
                )
            except Exception as e:
                logger.warning(f"Final model eval submission failed: {e}")


def cleanup_judge_clients():
    """Cleans up all LLM judge clients."""
    asyncio.run(cleanup_all_llm_judge_clients())
    logger.info("✅ LLM judge clients cleaned up")


def cleanup_training_resources(
    stop_event: threading.Event,
    executor: futures.ThreadPoolExecutor,
    queues: list[ray_queue.Queue],
    actor_manager: ActorManager,
) -> None:
    """Clean up all training resources including threads and Ray queues."""
    stop_event.set()

    logger.info("Signaling all actors to stop...")
    ray.get(actor_manager.set_should_stop.remote(True))
    logger.info("✅ Signaled all actors to stop")

    # Clean up ActorManager resources
    logger.info("Cleaning up ActorManager resources...")
    ray.get(actor_manager.cleanup.remote())
    logger.info("✅ ActorManager resources cleaned up")

    logger.info("Pushing shutdown sentinel to queues...")
    # Push sentinel to the first queue (inference_results_Q)
    if queues and len(queues) > 0:
        queues[0].put(ShutdownSentinel(), timeout=1)

    logger.info("Shutting down Ray queues...")
    if queues and len(queues) > 0:
        [queue.shutdown() for queue in queues]
    logger.info("Shutting down thread pool executor...")
    executor.shutdown(wait=True)

    # Clean up judge clients
    cleanup_judge_clients()

    # Shutdown Ray only from the main process (rank 0) or when DDP isn't initialized
    try:
        is_ddp = dist.is_available() and dist.is_initialized()
        is_rank0 = (not is_ddp) or (dist.get_rank() == 0)
        if is_rank0 and ray.is_initialized():
            logger.info("Shutting down Ray...")
            ray.shutdown()
            logger.info("✅ Ray shut down")
    except Exception as e:
        logger.warning(f"Ray shutdown failed: {e}")

    # Clean up distributed process group if it was initialized
    if dist.is_initialized():
        logger.info("Destroying process group...")
        dist.destroy_process_group()
        logger.info("✅ Process group destroyed")


def run_training(
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
    actor_manager: ActorManager,
    model_dims: utils.ModelDims,
    checkpoint_state=None,
    loaded_eval_config: checkpoint_eval.CheckpointEvalConfig | None = None,
    model_config: ModelConfig | None = None,
):
    if resume_training_step > 1:
        logger.info(f"[Main Thread] Resuming training from step {resume_training_step}")

    # Restore dataloader state if available in checkpoint
    if checkpoint_state and "dataloader_state" in checkpoint_state:
        ray_get_with_progress(
            [
                policy_group.models[i].load_dataloader_state.remote(checkpoint_state["dataloader_state"])
                for i in range(args.world_size)
            ],
            desc="Restoring dataloader state",
        )
        logger.info("Restored dataloader state from checkpoint")

    logger.info("======== ✅ weight sync thread starts =========")
    weight_sync_trigger_event = threading.Event()

    # Determine LoRA disk sync settings
    _lora_disk_sync = model_config is not None and model_config.use_peft and model_config.lora_disk_sync
    _lora_sync_dir = None
    _skip_noop_lora_sync = model_config is not None and model_config.skip_noop_lora_sync
    lora_sync_needed_event = None
    if _lora_disk_sync:
        _lora_sync_dir = args.lora_sync_dir or os.path.join(args.output_dir, "lora_sync")
        os.makedirs(_lora_sync_dir, exist_ok=True)
        logger.info(f"LoRA disk sync enabled, sync dir: {_lora_sync_dir}")
        if _skip_noop_lora_sync:
            lora_sync_needed_event = threading.Event()
            logger.info("skip_noop_lora_sync enabled: will skip merge cycles when no training has occurred")

    weight_sync_thread_future = executor.submit(
        weight_sync_thread,
        args,
        stop_event,
        weight_sync_trigger_event,
        policy_group,
        actor_manager,
        weight_sync_metrics_Q,
        resume_training_step,
        lora_disk_sync=_lora_disk_sync,
        lora_sync_dir=_lora_sync_dir,
        vllm_engines=vllm_engines if _lora_disk_sync else None,
        lora_sync_needed_event=lora_sync_needed_event,
        lora_no_pause_merge=model_config is not None and model_config.lora_no_pause_merge,
    )

    """Run the main training loop with worker threads."""
    ray_get_with_progress(
        [engine.ready.remote() for engine in vllm_engines], "Checking engines are ready to work", timeout=300
    )

    logger.info("======== ✅ Dataloaders already initialized in actors =========")

    _health_check_call_count = 0

    def health_check_fn():
        _cls = health_check_fn  # use function object for persistent state
        if not hasattr(_cls, "_call_count"):
            _cls._call_count = 0
        _cls._call_count += 1

        [f.result() for f in [weight_sync_thread_future] if f.done()]
        # Wait for weight sync to complete (should_stop becomes False)
        start = time.perf_counter()
        while ray.get(actor_manager.should_stop.remote()):
            if time.perf_counter() - start > WEIGHT_SYNC_TIMEOUT_S:
                raise RuntimeError(f"Weight sync timed out after {WEIGHT_SYNC_TIMEOUT_S}s - vLLM engines may be stuck")
            time.sleep(0.1)
        # Full vLLM health check is expensive (~5s for 4 Ray remote calls).
        # Only run every 10 steps to avoid per-step overhead.
        if _cls._call_count % 10 == 1:
            ray_get_with_progress(
                [engine.check_background_threads.remote() for engine in vllm_engines],
                desc="Checking vLLM engine health",
                enable=False,
            )

    if checkpoint_state and "num_total_tokens" in checkpoint_state:
        num_total_tokens = checkpoint_state["num_total_tokens"]
        logger.info(f"Restored num_total_tokens: {num_total_tokens}")
    else:
        num_total_tokens = 0

    if eval_dataset is not None:
        eval_data_loader = data_loader_lib.HFDataLoader(
            dataset=eval_dataset,
            batch_size=1,
            seed=args.seed,
            dp_rank=0,
            dp_world_size=1,
            work_dir=args.output_dir,
            automatic_reshuffle=False,
            collator=lambda x: x[0],
        )
    else:
        eval_data_loader = None
    grad_norm_tracker = GradNormTracker(window=50)
    training_start_time = time.perf_counter()  # Track overall training start time
    maybe_update_beaker_description(
        current_step=resume_training_step - 1,
        total_steps=args.num_training_steps,
        start_time=training_start_time,
        wandb_url=wandb_url,
    )
    # Save step-0 checkpoint (pre-training baseline) if configured.
    # The training loop starts at step 1, so step 0 must be handled separately.
    if args.eval_on_step_0 and resume_training_step <= 1:
        maybe_save_checkpoint(
            args, 0, policy_group, tc.chat_template_name, tokenizer, wandb_url, eval_config=loaded_eval_config
        )

    last_eval_collected = True
    for training_step in range(resume_training_step, args.num_training_steps + 1):
        start_time = time.perf_counter()

        # Check if any of the threads have raised an exception.
        health_check_start = time.perf_counter()
        health_check_fn()
        health_check_time = time.perf_counter() - health_check_start

        if (
            training_step % args.local_eval_every == 0
            and eval_data_loader is not None
            and (args.eval_on_step_0 or training_step > 1)
        ):
            if not last_eval_collected:
                logger.warning(
                    "[Main Thread] ⚠️ Previous eval round was not fully collected and may be included in future evals. "
                    "Consider increasing local_eval_every."
                )
            for eval_example in iter(eval_data_loader):
                add_prompt_to_generator(eval_example, 0, prompt_Q, generation_configs["eval"], is_eval=True)

        episode += streaming_config.num_unique_prompts_rollout * streaming_config.num_samples_per_prompt_rollout

        data_thread_metrics = {}
        try:
            data_thread_metrics |= weight_sync_metrics_Q.get_nowait()
        except Empty:
            logger.debug("[Main Thread] didn't get train generation metrics")

        data_thread_metrics["time/health_check"] = health_check_time

        num_step_tokens = one_training_step(
            args,
            streaming_config,
            vllm_config,
            policy_group,
            tokenizer,
            data_thread_metrics,
            episode,
            training_step,
            num_total_tokens,
            start_time,
            train_dataset,
            training_start_time,
            wandb_url,
            tc.chat_template_name,
            model_dims,
            actor_manager,
            grad_norm_tracker,
            loaded_eval_config,
        )
        num_total_tokens += num_step_tokens
        # Signal that real training happened (weights changed) so LoRA sync is needed
        if num_step_tokens > 0 and lora_sync_needed_event is not None:
            lora_sync_needed_event.set()

        # Checkpoint after one_training_step (or even if it was skipped)
        # This ensures we checkpoint progress even if the exact checkpoint step has no data
        if (
            args.checkpoint_state_freq > 0
            and training_step % args.checkpoint_state_freq == 0
            and args.checkpoint_state_dir is not None
        ):
            utils.warn_if_low_disk_space(args.checkpoint_state_dir, send_slack_alerts=args.send_slack_alerts)
            with Timer("[Main Thread] 🗡️ Saving checkpoint state"):
                # Save comprehensive client state including dataloader state
                client_state = {
                    "training_step": training_step,
                    "episode": episode,
                    "num_total_tokens": num_total_tokens,
                }
                if args.with_tracking and wandb.run is not None:
                    client_state["wandb_run_id"] = wandb.run.id

                # Save dataloader state from Ray actor
                client_state["dataloader_state"] = ray.get(policy_group.models[0].get_dataloader_state.remote())

                # Save DataPreparationActor state
                data_prep_actor = ray.get_actor("data_prep_singleton")
                client_state["data_prep_actor_state"] = ray.get(data_prep_actor.get_state.remote())

                ray_get_with_progress(
                    [
                        policy_group.models[i].save_checkpoint_state.remote(args.checkpoint_state_dir, client_state)
                        for i in range(args.world_size)
                    ],
                    desc=f"Saving checkpoint state at step {training_step}",
                )
                logger.info(f"Saved checkpoint state at step {training_step} to {args.checkpoint_state_dir}")

        logger.debug(f"[Main Thread] Triggered weight sync for step {training_step}")
        weight_sync_trigger_event.set()

        last_eval_collected = maybe_evaluate(
            args,
            training_step,
            evaluation_inference_results_Q,
            tokenizer,
            episode,
            eval_dataset,
            generation_configs["eval"],
            model_dims,
            actor_manager,
        )

        maybe_update_beaker_description(
            current_step=training_step,
            total_steps=args.num_training_steps,
            start_time=training_start_time,
            wandb_url=wandb_url,
        )

    if resume_training_step > args.num_training_steps:
        raise ValueError(f"Training didn't run since {resume_training_step=} > {args.num_training_steps=}")

    save_final_model(
        args, policy_group, tokenizer, training_step, wandb_url, tc.chat_template_name, eval_config=loaded_eval_config
    )
