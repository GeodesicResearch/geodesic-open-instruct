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
# isort: off
import contextlib
import os
import pathlib
from datetime import timedelta

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
with contextlib.suppress(Exception):
    import deepspeed
    from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF
    from deepspeed.utils import groups

from open_instruct import data_loader as data_loader_lib
from open_instruct import data_types, utils
from open_instruct.utils import grpo as grpo_utils

# isort: on
import dataclasses
import math
import random
import socket
import time
from typing import Any

import numpy as np
import ray
import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, get_scheduler
from transformers.integrations import HfDeepSpeedConfig

from open_instruct.utils import vllm as vllm_utils
from open_instruct.utils.beaker import BeakerRuntimeConfig, launch_ai2_evals_on_weka, sync_gs_bucket
from open_instruct.utils.checkpoints import clean_last_n_checkpoints_deepspeed
from open_instruct.utils.deepspeed import (
    _z3_params_to_fetch,
    get_eval_ds_config,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
)
from open_instruct.utils.general import INVALID_LOGPROB, ray_get_with_progress
from open_instruct.utils.logger import setup_logger
from open_instruct.utils.model import (
    ModelConfig,
    disable_dropout_in_model,
    estimate_kl,
    get_olmo3_generation_config,
    load_ref_policy,
)
from open_instruct.utils.ray import RayProcess
from open_instruct.utils.rl import Timer, masked_mean
from open_instruct.utils.ulysses import UlyssesSPSplitter

logger = setup_logger(__name__)

CHECKPOINT_COMPLETE_MARKER = ".checkpoint_complete"


def to_device_inplace(tensors_list: list[torch.Tensor], device: torch.device):
    for i in range(len(tensors_list)):
        tensors_list[i] = tensors_list[i].to(device, non_blocking=True)


@ray.remote(num_gpus=1)
class PolicyTrainerRayProcess(RayProcess):
    def __init__(
        self,
        world_size: int,
        rank: int,
        local_rank: int,
        master_addr: str | None,
        master_port: int | None,
        args: grpo_utils.ExperimentConfig,
        streaming_config: data_loader_lib.StreamingDataLoaderConfig,
        vllm_config: data_loader_lib.VLLMConfig,
        data_prep_actor_name: str,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__(world_size, rank, local_rank, master_addr, master_port)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.num_mini_batches = args.num_mini_batches
        self._args = args
        self.streaming_config = streaming_config
        self.vllm_config = vllm_config
        self.world_size = world_size
        self.local_rank = local_rank
        self.dp_world_size = world_size // args.sequence_parallel_size
        self._data_prep_actor_name = data_prep_actor_name
        self._use_peft = False
        self._lora_disk_sync = False

    def get_dataloader_state(self) -> dict[str, Any]:
        return self._streaming_dataloader.state_dict()

    def load_dataloader_state(self, state_dict: dict[str, Any]) -> None:
        self._streaming_dataloader.load_state_dict(state_dict)

    def from_pretrained(
        self,
        args: grpo_utils.ExperimentConfig,
        model_config: ModelConfig,
        beaker_config: BeakerRuntimeConfig,
        wandb_url: str,
        tokenizer: PreTrainedTokenizer,
    ) -> int:
        # ------------------------------------------------------------
        # Monkey patch to load checkpoints with `weights_only=False`
        # otherwise it errors out with:
        # `_pickle.UnpicklingError: Weights only load failed. ` with pytorch 2.6.0
        from deepspeed.runtime.checkpoint_engine import torch_checkpoint_engine  # noqa: PLC0415
        from deepspeed.utils import logger  # noqa: PLC0415

        def load(self, path: str, map_location=None):
            logger.info(f"[Torch] Loading checkpoint from {path}...")
            partition = torch.load(path, map_location=map_location, weights_only=False)
            logger.info(f"[Torch] Loaded checkpoint from {path}.")
            return partition

        torch_checkpoint_engine.TorchCheckpointEngine.load = load

        # ------------------------------------------------------------
        self.args = args
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.beaker_config = beaker_config
        self.wandb_url = wandb_url
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        ld_preload = os.environ.get("LD_PRELOAD", "not set")
        logger.warning(f"Learner rank {self.rank}: CVD={cuda_devices}, LD_PRELOAD={ld_preload}")

        torch.cuda.set_device(self.local_rank)

        free_mem, total_mem = torch.cuda.mem_get_info(self.local_rank)
        logger.warning(
            f"Learner rank {self.rank}: mem_get_info: free={free_mem / 1024**3:.2f} GiB, total={total_mem / 1024**3:.2f} GiB"
        )

        self.device = torch.device(self.local_rank)

        # Set seeds for this worker (different per rank to avoid correlation)
        worker_seed = args.seed + self.local_rank
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

        # Pre-initialize torch.distributed WITHOUT device_id to avoid NCCL hangs.
        # DeepSpeed 0.17.3 and up sets device_id in init_process_group which can cause hangs
        # when multiple process groups exist (e.g., for weight sync to vLLM).
        # By initializing first, DeepSpeed will detect it and wrap it instead of re-initializing.
        logger.warning(f"Learner rank {self.rank}: starting torch.distributed.init_process_group")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", timeout=timedelta(minutes=args.backend_timeout))
        logger.warning(f"Learner rank {self.rank}: torch.distributed initialized, starting deepspeed.init_distributed")
        deepspeed.init_distributed(timeout=timedelta(minutes=args.backend_timeout))
        logger.warning(f"Learner rank {self.rank}: deepspeed.init_distributed done")

        if args.training_dtype not in ("bfloat16", "float16"):
            raise ValueError(f"training_dtype must be 'bfloat16' or 'float16', got '{args.training_dtype}'")
        use_bf16 = args.training_dtype == "bfloat16"
        torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
        ds_config = get_train_ds_config(
            offload=args.deepspeed_offload_param,
            adam_offload=args.deepspeed_offload_optimizer,
            stage=args.deepspeed_stage,
            bf16=use_bf16,
            zpg=args.deepspeed_zpg,
            sequence_parallel_size=args.sequence_parallel_size,
        )
        if not use_bf16:
            ds_config.pop("bf16", None)
            ds_config["fp16"] = {"enabled": True}
            # ZeRO-0 rejects fp16 model + fp32 grad accum; match grad accum to model dtype
            if args.deepspeed_stage == 0:
                ds_config["data_types"] = {"grad_accum_dtype": "fp16"}
        ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ds_config["gradient_accumulation_steps"] = 1
        # @vwxyzjn: MAGIC: it's actually needed to initialize this `dschf`, so
        # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
        # next line instructs transformers to partition the model directly over multiple gpus using
        # deepspeed.zero.Init when model's `from_pretrained` method is called.
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
        logger.info(f"Deepspeed config: {dschf=}")

        # set sequence parallel
        # note this returns None if sequence_parallel_size == 1
        self.mpu = UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=model_config.model_name_or_path,
            core_attn_implementation=model_config.attn_implementation,
            sequence_parallel_size=args.sequence_parallel_size,
            micro_batch_size=args.per_device_train_batch_size,
            seq_length_is_variable=True,
        )
        logger.warning(
            f"Learner rank {self.rank}: starting AutoModelForCausalLM.from_pretrained("
            f"{model_config.model_name_or_path}, attn_implementation={model_config.attn_implementation}, "
            f"deepspeed_stage={args.deepspeed_stage}, local_rank={self.local_rank}, "
            f"GPU mem before: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB)"
        )
        _t0 = time.monotonic()
        self.policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            dtype=torch_dtype,
            attn_implementation=model_config.attn_implementation,
            use_cache=False,
            low_cpu_mem_usage=True,
            **({"device_map": {"": self.local_rank}} if args.deepspeed_stage != 3 else {}),
        )
        _t1 = time.monotonic()
        logger.warning(
            f"Learner rank {self.rank}: from_pretrained done in {_t1 - _t0:.1f}s, "
            f"GPU mem: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB, "
            f"model type: {type(self.policy).__name__}, "
            f"attn impl: {getattr(self.policy.config, '_attn_implementation', 'unknown')}"
        )
        disable_dropout_in_model(self.policy)

        # Apply LoRA if configured (must be before deepspeed.initialize)
        if model_config.use_peft:
            lora_config = LoraConfig(
                r=model_config.lora_r,
                lora_alpha=model_config.lora_alpha,
                lora_dropout=model_config.lora_dropout,
                target_modules=model_config.lora_target_modules,
                modules_to_save=model_config.lora_modules_to_save,
                task_type=model_config.lora_task_type,
                bias="none",
            )
            self.policy = get_peft_model(self.policy, lora_config)
            self.policy.print_trainable_parameters()
            self._use_peft = True
            self._lora_disk_sync = model_config.lora_disk_sync

        self.policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if args.set_weight_decay_on_bias_and_norm:
            optim_params = get_optimizer_grouped_parameters(self.policy, args.weight_decay)
        else:
            optim_params = self.policy.parameters()
        # When offloading optimizer to CPU, use regular AdamW (not fused — fused requires CUDA).
        # DeepSpeed accepts this because zero_force_ds_cpu_optimizer=False is set in the DS config
        # (see utils/deepspeed.py). DeepSpeedCPUAdam requires a JIT-compiled C++ extension that
        # may not be available on all nodes.
        if args.deepspeed_offload_optimizer:
            self.optimizer = torch.optim.AdamW(optim_params, lr=args.learning_rate)
        else:
            self.optimizer = torch.optim.AdamW(optim_params, lr=args.learning_rate, fused=args.fused_optimizer)
        num_scheduler_steps = args.num_training_steps * args.num_epochs * args.num_mini_batches
        warm_up_steps = args.warm_up_steps
        if args.warmup_ratio > 0.0:
            warm_up_steps = int(num_scheduler_steps * args.warmup_ratio)
        scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=num_scheduler_steps,
        )
        logger.warning(f"Learner rank {self.rank}: starting deepspeed.initialize")
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.policy,
            optimizer=self.optimizer,
            config=ds_config,
            lr_scheduler=scheduler,
            dist_init_required=False,
            mpu=self.mpu,
        )
        logger.warning(
            f"Learner rank {self.rank}: deepspeed.initialize done, GPU mem: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB"
        )
        optimization_steps_done = 0
        if args.resume_from:
            if not os.path.exists(args.resume_from):
                raise ValueError(f"resume_from path does not exist: {args.resume_from}")
            # DeepSpeed's load_checkpoint expects the parent directory containing
            # a `latest` file and `global_step*` subdirectories. If the user
            # passes a specific step directory, split into parent + tag.
            resume_path = args.resume_from
            load_dir = resume_path
            tag = None
            if not os.path.exists(os.path.join(resume_path, "latest")):
                # Assume user passed a step directory like .../global_step600
                load_dir = os.path.dirname(resume_path)
                tag = os.path.basename(resume_path)
            logger.warning(f"Resuming from checkpoint at {load_dir} (tag={tag})")
            # remove mpu for loading checkpoints, add it back after loading
            old_mpu = self.mpu
            self.model.mpu = None
            path, states = self.model.load_checkpoint(
                load_dir,
                tag=tag,
                load_module_strict=True,
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
                load_module_only=False,
            )
            self.model.mpu = old_mpu
            if path is None:
                raise ValueError(f"Failed to load checkpoint from {args.resume_from}")
            optimization_steps_done = states["training_step"]

            rng_states = states["rng_states"]
            torch.set_rng_state(rng_states["torch_cpu_rng_state"])
            np.random.set_state(rng_states["numpy_rng_state"])
            random.setstate(rng_states["python_rng_state"])

            if torch.cuda.is_available() and "torch_cuda_rng_states" in rng_states:
                # device_str, e.g. "cuda:0"
                for device_str, rng_state in rng_states["torch_cuda_rng_states"].items():
                    device_id = int(device_str.split(":")[1])
                    torch.cuda.set_rng_state(rng_state, device_id)
                if "torch_cuda_rng_state_all" in rng_states:
                    torch.cuda.set_rng_state_all(rng_states["torch_cuda_rng_state_all"])

            logger.info(f"{self.rank=}: Restored RNG states from checkpoint")

            # Save reference policy path to load later (after ref_policy is initialized)
            self.ref_policy_checkpoint_path = None
            if args.load_ref_policy and states.get("ref_policy_saved", False):
                ref_policy_dir = os.path.join(args.resume_from, "ref_policy")
                model_path = os.path.join(ref_policy_dir, "pytorch_model.bin")
                if os.path.exists(model_path):
                    self.ref_policy_checkpoint_path = model_path
                    logger.info(f"{self.rank=}: Will load reference policy from {model_path}")

            logger.info(f"{self.rank=}: Loaded checkpoint from {args.resume_from} with {optimization_steps_done=}")
        self.model.train()

        # reference model
        if args.load_ref_policy:
            ref_ds_config, self.ref_policy_hf_ds_config = get_eval_ds_config(
                offload=False,
                # inference model only has stage 3 (sharding) or stage 0 (no sharding)
                # stage 2 is optimizer sharding which doesn't apply to inference
                stage=args.deepspeed_stage if args.deepspeed_stage == 3 else 0,
                bf16=use_bf16,
                per_device_train_batch_size=args.per_device_train_batch_size,
            )
            if not use_bf16:
                ref_ds_config.pop("bf16", None)
                ref_ds_config["fp16"] = {"enabled": True}

            self.ref_policy: PreTrainedModel = load_ref_policy(
                model_config=model_config,
                ds_config=ref_ds_config,
                deepspeed_stage=args.deepspeed_stage,
                local_rank=self.local_rank,
                device=self.device,
                rank=self.rank,
                checkpoint_path=self.ref_policy_checkpoint_path
                if hasattr(self, "ref_policy_checkpoint_path")
                else None,
                mpu=self.mpu,
                ref_policy_update_freq=args.ref_policy_update_freq,
                alpha=args.alpha,
            )
        self.local_metrics = utils.MetricsTracker(max_metrics=512, device=self.device)

        if self.mpu is not None:
            self.splitter = UlyssesSPSplitter(
                sp_rank=groups._get_sequence_parallel_rank(),
                sp_group=groups._get_sequence_parallel_group(),
                sp_world_size=groups._get_sequence_parallel_world_size(),
                device=self.device,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        else:
            self.splitter = None

        # dp_rank = which data-parallel group this worker belongs to
        # With SP, workers in the same SP group share the same dp_rank
        dp_rank = self.rank // args.sequence_parallel_size
        assert dp_rank < self.dp_world_size

        # Verify SP groups are consecutive as we assume for above logic (e.g., [0,1,2,3], [4,5,6,7], ...)
        # getting the dp_rank directly does not work right now with the mpus :/
        if self.mpu is not None:
            sp_group = groups._get_sequence_parallel_group()
            sp_ranks = sorted(torch.distributed.get_process_group_ranks(sp_group))
            expected = list(range(dp_rank * args.sequence_parallel_size, (dp_rank + 1) * args.sequence_parallel_size))
            assert sp_ranks == expected, f"SP group {sp_ranks} != expected {expected}"

        self._streaming_dataloader = self.streaming_config.build_dataloader(
            data_prep_actor_name=self._data_prep_actor_name,
            tokenizer=tokenizer,
            dp_rank=dp_rank,
            fs_local_rank=self.local_rank,
            num_training_steps=args.num_training_steps,
            work_dir=args.output_dir,
            dp_world_size=self.dp_world_size,
        )
        self.dataloader = iter(self._streaming_dataloader)

        return optimization_steps_done

    def setup_model_update_group(self, vllm_engines):
        self.vllm_engines = vllm_engines
        self.model_update_group = None
        if self._use_peft and self._lora_disk_sync:
            logger.info(f"Learner rank {self.rank}: skipping NCCL weight_sync group (LoRA disk sync enabled)")
            torch.distributed.barrier()
            return
        if self.rank == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            vllm_num_engines, vllm_tensor_parallel_size = (
                self.vllm_config.vllm_num_engines,
                self.vllm_config.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
            backend = self.vllm_config.vllm_sync_backend
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    "weight_sync",
                    backend=backend,
                    timeout_minutes=self.args.backend_timeout,
                )
                for i, engine in enumerate(vllm_engines)
            ]
            torch.cuda.set_device(self.local_rank)
            self.model_update_group = vllm_utils.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="weight_sync",
                timeout=timedelta(minutes=self.args.backend_timeout),
            )
            ray_get_with_progress(refs, desc="Initializing vLLM process groups", timeout=600)
        torch.distributed.barrier()

    def broadcast_to_vllm(self):
        # avoid OOM
        torch.cuda.empty_cache()
        # Ensure CUDA device is set before broadcast operations.
        # DeepSpeed 0.17.3+ sets device_id in init_process_group which affects NCCL device binding.
        torch.cuda.set_device(self.local_rank)
        return vllm_utils.broadcast_weights_to_vllm(
            model=self.model.module,
            vllm_engines=self.vllm_engines,
            model_update_group=self.model_update_group,
            deepspeed_stage=self.args.deepspeed_stage,
            gather_whole_model=self.args.gather_whole_model,
        )

    def save_lora_to_disk(self, lora_sync_dir: str, step: int) -> str | None:
        """Save LoRA adapter to shared disk. Only rank 0 writes. Returns path."""
        if torch.distributed.get_rank() != 0:
            return None
        step_dir = os.path.join(lora_sync_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        model_to_save = self.model.module  # unwrap DeepSpeed
        if isinstance(model_to_save, PeftModel):
            model_to_save.save_pretrained(step_dir)
            logger.info(f"Saved LoRA adapter to {step_dir}")
        else:
            raise RuntimeError("save_lora_to_disk called but model is not a PeftModel")
        return step_dir

    def update_ref_policy(self):
        if not self.args.load_ref_policy:
            return
        for ref_param, param in zip(self.ref_policy.parameters(), self.model.parameters()):
            if self.args.deepspeed_stage == 3:
                with deepspeed.zero.GatheredParameters([param, ref_param], modifier_rank=0):
                    if deepspeed.comm.get_rank() == 0:
                        ref_param.data.mul_(1.0 - self.args.alpha).add_(param.data, alpha=self.args.alpha)
            else:
                ref_param.data.mul_(1.0 - self.args.alpha).add_(param.data, alpha=self.args.alpha)

    def calculate_token_counts(
        self, accumulation_steps: int, data_BT: data_types.CollatedBatchData
    ) -> dict[int, float]:
        accumulation_counts: dict[int, float] = {}
        local_counts = [mask[:, 1:].sum().float() for mask in data_BT.response_masks]
        if not local_counts:
            return accumulation_counts

        counts_tensor = torch.stack(local_counts)
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)

        for i, count in enumerate(counts_tensor):
            group_idx = i // accumulation_steps
            key = int(group_idx * accumulation_steps)
            accumulation_counts[key] = accumulation_counts.get(key, 0.0) + count.item()

        return accumulation_counts

    def _compute_loss_metrics(
        self, loss_stats_B: dict[str, torch.Tensor], total_valid_tokens: int
    ) -> dict[str, float]:
        """Compute weighted average metrics from per-batch loss statistics."""
        token_counts = loss_stats_B["token_count"]
        total_tokens = token_counts.sum()
        # Zero weights when no tokens - all weighted sums become 0
        weights = token_counts / total_tokens if total_tokens > 0 else torch.zeros_like(token_counts)

        if self.args.load_ref_policy:
            for j in range(4):
                self.local_metrics[f"objective/kl{j}_avg"] = (loss_stats_B["kl"][j] * weights).sum()
            self.local_metrics["loss/kl_avg"] = (loss_stats_B["kl_loss"] * weights).sum()
        self.local_metrics["loss/policy_avg"] = (loss_stats_B["pg_loss"] * weights).sum()
        self.local_metrics["loss/total_avg"] = (loss_stats_B["loss"] * weights).sum()
        self.local_metrics["policy/clipfrac_avg"] = (loss_stats_B["pg_clipfrac"] * weights).sum()
        self.local_metrics["val/ratio"] = (loss_stats_B["ratio"] * weights).sum()
        weighted_mean_ratio = self.local_metrics["val/ratio"]
        self.local_metrics["val/ratio_var"] = (weights * (loss_stats_B["ratio"] - weighted_mean_ratio) ** 2).sum()
        if self.args.record_entropy:
            self.local_metrics["policy/entropy_avg"] = (loss_stats_B["entropy"] * weights).sum()

        self.local_metrics["lr"] = self.scheduler.get_last_lr()[0]
        self.local_metrics["_token_count"] = total_valid_tokens
        if self._step_grad_norms:
            self.local_metrics["loss/grad_norm"] = sum(self._step_grad_norms) / len(self._step_grad_norms)

    def step(self):
        """Execute one training step: fetch data from the dataloader and train on it.

        Returns:
            Tuple of (metrics_list, array_metrics) from training.
        """
        self._step_grad_norms: list[float] = []
        batch_data = next(self.dataloader)
        data_BT = batch_data["batch"]
        if len(data_BT) == 0:
            logger.warning("[Training] Empty batch received, skipping training step")
            return [], {}

        # split batch for sequence parallelism. Do before moving data to GPU.
        if self.splitter is not None:
            with Timer("✂️ Splitting batch for SP", noop=self.rank != 0):
                data_BT = self.splitter.split_collated_batch(data_BT)

        for f in dataclasses.fields(data_BT):
            to_device_inplace(getattr(data_BT, f.name), self.device)
        data_BT.response_masks = [mask.bool() for mask in data_BT.response_masks]
        num_samples = len(data_BT)
        accumulation_steps = max(math.ceil(num_samples / self.num_mini_batches - 0.5), 1)
        leftover = num_samples % accumulation_steps
        if leftover > 0:
            data_BT = data_BT[:-leftover]
            logger.warning(f"{leftover} samples are dropped due to batch size {self.num_mini_batches}")

        num_mini_batches = len(data_BT.query_responses) // accumulation_steps

        ref_logprobs_BT: list[torch.Tensor] = []
        if self.args.load_ref_policy:
            with Timer("Inference Calculation", noop=self.rank != 0):
                ref_logprobs_BT = grpo_utils.compute_logprobs(
                    self.ref_policy, data_BT, self.pad_token_id, self.streaming_config.temperature, use_grad=False
                )

        # if we have multiple minibatches, we need to calculate the old logprobs for each minibatch
        # following gtrl scripts in just doing this on the current active policy, rather than use the logprobs
        # from the generator (note that async mode means these are a bit diff!)
        old_logprobs_BT: list[torch.Tensor | None] = [None for _ in range(len(data_BT.query_responses))]
        if num_mini_batches > 1:
            with Timer("Old logprobs Calculation", noop=self.rank != 0):
                local_old_logprobs_BT = None
                if not self.args.use_vllm_logprobs:
                    local_old_logprobs_BT = grpo_utils.compute_logprobs(
                        self.model, data_BT, self.pad_token_id, self.streaming_config.temperature, use_grad=False
                    )

                with torch.no_grad():
                    for i in range(len(data_BT.query_responses)):
                        vllm_old_logprob_BT = data_BT.vllm_logprobs[i][:, 1:]
                        vllm_old_logprob_BT = torch.masked_fill(
                            vllm_old_logprob_BT, ~data_BT.response_masks[i][:, 1:], INVALID_LOGPROB
                        )
                        vllm_old_logprob_BT = torch.nan_to_num(vllm_old_logprob_BT, nan=INVALID_LOGPROB)

                        if self.args.use_vllm_logprobs:
                            old_logprobs_BT[i] = vllm_old_logprob_BT
                        else:
                            old_logprobs_BT[i] = local_old_logprobs_BT[i]

        local_step = 0
        num_samples = len(data_BT.query_responses)
        # Pre-compute token counts per sample (for weighted averaging across SP ranks)
        # This only needs to be done once since response_masks don't change across epochs
        token_counts_per_sample = torch.stack([mask[:, 1:].sum().float() for mask in data_BT.response_masks])
        total_valid_tokens = token_counts_per_sample.sum().item()
        device = token_counts_per_sample.device
        # Do multiple epochs of training on on-policy data (PPO-style), with a fresh random shuffle in each epoch
        with Timer("[Training Processes] Loss calculation", noop=self.rank != 0):
            loss_stats_B: dict[str, torch.Tensor] = {
                "kl": torch.zeros(4, num_samples, device=device),
                "kl_loss": torch.zeros(num_samples, device=device),
                "pg_clipfrac": torch.zeros(num_samples, device=device),
                "pg_loss": torch.zeros(num_samples, device=device),
                "loss": torch.zeros(num_samples, device=device),
                "ratio": torch.zeros(num_samples, device=device),
                "entropy": torch.zeros(num_samples, device=device),
                "token_count": token_counts_per_sample,
            }
            for epoch_idx in range(self.args.num_epochs):
                # Pre-compute total tokens for each accumulation group if using "token" normalization
                # This ensures all minibatches in an accumulation group are normalized by the same total
                if self.args.loss_denominator == "token":
                    accumulation_token_counts = self.calculate_token_counts(accumulation_steps, data_BT)
                else:
                    accumulation_token_counts = {
                        int(group_idx * accumulation_steps): float(self.args.loss_denominator)
                        for group_idx in range((len(data_BT.query_responses) // accumulation_steps) + 1)
                    }

                for i in range(num_samples):
                    response_mask_BT = data_BT.response_masks[i][:, 1:]
                    # retrieve the loss denominator for the current batch
                    batch_start = (i // accumulation_steps) * accumulation_steps
                    loss_denominator = accumulation_token_counts[batch_start]
                    local_logprobs_BT, entropy_BT = grpo_utils.forward_for_logprobs(
                        self.model,
                        data_BT.query_responses[i],
                        data_BT.attention_masks[i],
                        data_BT.position_ids[i],
                        self.pad_token_id,
                        self.streaming_config.temperature,
                        return_entropy=self.args.record_entropy,
                    )
                    local_logprobs_BT = torch.masked_fill(local_logprobs_BT, ~response_mask_BT, INVALID_LOGPROB)
                    vllm_logprobs_BT = data_BT.vllm_logprobs[i][:, 1:]
                    vllm_logprobs_BT = torch.masked_fill(vllm_logprobs_BT, ~response_mask_BT, INVALID_LOGPROB)
                    vllm_logprobs_BT = torch.nan_to_num(vllm_logprobs_BT, nan=INVALID_LOGPROB)

                    # Compare vLLM logprobs with local logprobs (only on first sample to avoid CUDA syncs)
                    if i == 0:
                        with torch.no_grad():
                            valid_mask_BT = response_mask_BT & ~torch.isnan(vllm_logprobs_BT)
                            logprob_diff_BT = (local_logprobs_BT - vllm_logprobs_BT).abs()
                            masked_diff_BT = torch.masked_fill(logprob_diff_BT, ~valid_mask_BT, 0.0)
                            mean_diff = masked_diff_BT.sum() / valid_mask_BT.sum() if valid_mask_BT.sum() > 0 else 0.0
                            max_diff = masked_diff_BT.max()
                            std_diff = masked_diff_BT[valid_mask_BT].std() if valid_mask_BT.sum() > 1 else 0.0

                            self.local_metrics["debug/vllm_vs_local_logprob_diff_mean"] = float(mean_diff)
                            self.local_metrics["debug/vllm_vs_local_logprob_diff_max"] = float(max_diff)
                            self.local_metrics["debug/vllm_vs_local_logprob_diff_std"] = float(std_diff)

                            if not getattr(self, "_logprob_diff_warned", False) and float(mean_diff) > 0.5:
                                logger.warning(
                                    f"LARGE vllm_vs_local logprob diff: {float(mean_diff):.3f} nats. "
                                    f"Training model and vLLM may use different attention implementations "
                                    f"or have a dtype mismatch. Check attn_implementation and dtype settings."
                                )
                                self._logprob_diff_warned = True

                            reverse_kl_BT = torch.exp(vllm_logprobs_BT) * (vllm_logprobs_BT - local_logprobs_BT)
                            masked_reverse_kl_BT = torch.masked_fill(reverse_kl_BT, ~valid_mask_BT, 0.0)
                            mean_reverse_kl = (
                                masked_reverse_kl_BT.sum() / valid_mask_BT.sum() if valid_mask_BT.sum() > 0 else 0.0
                            )
                            self.local_metrics["debug/vllm_local_reverse_kl"] = float(mean_reverse_kl)

                    new_logprobs_BT = local_logprobs_BT

                    # Cache the old logprobs
                    if num_mini_batches > 1:
                        old_logprob_BT = old_logprobs_BT[i]
                    else:
                        with torch.no_grad():
                            if epoch_idx == 0:
                                if self.args.use_vllm_logprobs:
                                    old_logprobs_BT[i] = vllm_logprobs_BT
                                else:
                                    old_logprobs_BT[i] = local_logprobs_BT.detach()
                            old_logprob_BT = old_logprobs_BT[i]

                    old_logprobs_mask_BT = old_logprob_BT != INVALID_LOGPROB
                    assert torch.all(old_logprobs_mask_BT == response_mask_BT), (
                        f"Old logprobs mask should match response mask. "
                        f"old_mask sum={old_logprobs_mask_BT.sum()}, "
                        f"response_mask sum={response_mask_BT.sum()}"
                    )

                    # Calculate the policy's loss
                    logprobs_diff_BT = new_logprobs_BT - old_logprob_BT
                    ratio_BT = torch.exp(logprobs_diff_BT)
                    # Apply truncated importance sampling if enabled
                    tis_imp_ratio_BT = None
                    if self.args.truncated_importance_sampling_ratio_cap > 0 and vllm_logprobs_BT is not None:
                        old_logprobs_mask_BT = old_logprob_BT != INVALID_LOGPROB
                        vllm_logprobs_mask_BT = vllm_logprobs_BT != INVALID_LOGPROB

                        assert torch.all(old_logprobs_mask_BT == response_mask_BT), (
                            f"Old logprobs mask should match response mask. "
                            f"old_mask sum={old_logprobs_mask_BT.sum()}, "
                            f"response_mask sum={response_mask_BT.sum()}"
                        )
                        assert torch.all(vllm_logprobs_mask_BT == response_mask_BT), (
                            f"vLLM logprobs mask should match response mask. "
                            f"vllm_mask sum={vllm_logprobs_mask_BT.sum()}, "
                            f"response_mask sum={response_mask_BT.sum()}"
                        )

                        valid_mask_BT = response_mask_BT
                        # Initialize importance ratio to 1.0 (no effect) for all positions
                        tis_imp_ratio_BT = torch.ones_like(old_logprob_BT)

                        if valid_mask_BT.any():
                            # Calculate logprob difference only for valid positions
                            logprob_diff_is_BT = old_logprob_BT - vllm_logprobs_BT
                            # Clamp to prevent numerical overflow in exp
                            logprob_diff_is_BT = torch.where(
                                valid_mask_BT,
                                logprob_diff_is_BT.clamp(-10.0, 10.0),
                                torch.zeros_like(logprob_diff_is_BT),
                            )
                            # Compute importance ratio only for valid positions
                            tis_imp_ratio_BT = torch.where(
                                valid_mask_BT, torch.exp(logprob_diff_is_BT), tis_imp_ratio_BT
                            )
                            # Apply cap
                            tis_imp_ratio_BT = torch.clamp(
                                tis_imp_ratio_BT, max=self.args.truncated_importance_sampling_ratio_cap
                            )

                    pg_losses_BT, pg_losses2_BT, pg_loss_max_BT, kl_BT = grpo_utils.compute_grpo_loss(
                        new_logprobs=new_logprobs_BT,
                        ratio=ratio_BT,
                        advantages=data_BT.advantages[i][:, 1:],
                        ref_logprobs=ref_logprobs_BT[i] if self.args.load_ref_policy else None,
                        config=self.args,
                        tis_weights=tis_imp_ratio_BT,
                    )

                    per_token_loss_BT = pg_loss_max_BT + self.args.beta * kl_BT
                    loss = masked_mean(per_token_loss_BT, response_mask_BT, None, loss_denominator)

                    # we already took world size into account via the tokens
                    # but deepspeed will try to average over ranks, so multiply back
                    # up, adjusting for the sequence parallel size (adjust by dp world size).
                    loss *= self.args.world_size // self.args.sequence_parallel_size

                    # DeepSpeed requires loss.grad_fn is not None (scalar with autograd graph).
                    # When all advantages are zero, the loss can be a detached constant.
                    # Add a zero-valued term connected to model parameters to ensure grad_fn.
                    if loss.grad_fn is None:
                        first_param = next(p for p in self.model.parameters() if p.requires_grad)
                        loss = loss + first_param.sum() * 0.0
                    self.model.backward(loss)
                    if (local_step + 1) % accumulation_steps == 0:
                        self.model.step()
                        grad_norm = self.model.get_global_grad_norm()
                        if grad_norm is not None:
                            self._step_grad_norms.append(float(grad_norm))
                    local_step += 1
                    with torch.no_grad():
                        if self.args.load_ref_policy:
                            # NOTE: in packed implementation, kl calculation are averages over response tokens
                            ref_logprobs_diff_BT = (new_logprobs_BT - ref_logprobs_BT[i]).clamp(-40.0, 40.0)
                            kl_4BT = estimate_kl(ref_logprobs_diff_BT, ratio_BT)
                            loss_stats_B["kl"][:, i] = masked_mean(kl_4BT, response_mask_BT).float()
                            loss_stats_B["kl_loss"][i] = loss_stats_B["kl"][self.args.kl_estimator, i] * self.args.beta
                        loss_stats_B["pg_clipfrac"][i] = masked_mean(
                            (pg_losses2_BT > pg_losses_BT).float(), response_mask_BT
                        )
                        loss_stats_B["pg_loss"][i] = masked_mean(pg_loss_max_BT, response_mask_BT)
                        loss_stats_B["loss"][i] = loss
                        loss_stats_B["ratio"][i] = masked_mean(ratio_BT, response_mask_BT)
                        if self.args.record_entropy:
                            loss_stats_B["entropy"][i] = masked_mean(entropy_BT, response_mask_BT).float()

            batch_metrics = batch_data["metrics"]
            with torch.no_grad():
                self._compute_loss_metrics(loss_stats_B, total_valid_tokens)
                array_metrics = {}
                # Zero out stale target_bias/ metrics before setting new ones.
                # MetricsTracker persists across steps, so if this step's batch
                # lacks a label (e.g. all A prompts filtered), the previous
                # step's value would persist, causing sum > 1.0.
                for existing_key in list(self.local_metrics.names2idx):
                    if existing_key.startswith("target_bias/"):
                        self.local_metrics[existing_key] = 0.0
                for key, value in batch_metrics.items():
                    if value is None:
                        continue
                    if isinstance(value, (int, float, np.floating, np.integer)):
                        self.local_metrics[key] = value
                    else:
                        array_metrics[key] = value
                return self.local_metrics.get_metrics_list(), array_metrics

    def save_checkpoint_state(self, checkpoint_state_dir: str, client_state: dict[str, Any]) -> None:
        args = self.args

        # Save comprehensive RNG states for each rank
        rng_states = {
            "torch_cpu_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        # Save CUDA RNG states for all devices
        if torch.cuda.is_available():
            rng_states["torch_cuda_rng_states"] = {
                f"cuda:{i}": torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())
            }
            rng_states["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        # Add RNG states to client_state
        client_state["rng_states"] = rng_states
        client_state["rank"] = self.rank

        # Save reference policy checkpoint (model only, no optimizer)
        if self.args.load_ref_policy:
            ref_policy_dir = os.path.join(checkpoint_state_dir, "ref_policy")
            os.makedirs(ref_policy_dir, exist_ok=True)

            # For reference policy, we save just the model weights
            # We can't use save_checkpoint because it would try to save DummyOptim
            # which doesn't have state_dict
            if self.rank == 0:
                # Only rank 0 saves the model state
                model_to_save = self.ref_policy.module if hasattr(self.ref_policy, "module") else self.ref_policy
                # Save the state dict
                torch.save(model_to_save.state_dict(), os.path.join(ref_policy_dir, "pytorch_model.bin"))
                logger.info(f"Saved reference policy model to {ref_policy_dir}")

            client_state["ref_policy_saved"] = True

        # Save the main model checkpoint with enhanced client state
        # mpu is just used for sequence parallel, so we remove it for saving, and then re-add it after.
        old_mpu = None
        if self.model.mpu is not None:
            old_mpu = self.mpu
            self.model.mpu = None
        self.model.save_checkpoint(checkpoint_state_dir, client_state=client_state)

        # `save_checkpoint` needs to be called on all ranks, only rank 0 will have all the states
        if self.rank == 0:
            if args.keep_last_n_checkpoints >= 0:
                clean_last_n_checkpoints_deepspeed(checkpoint_state_dir, args.keep_last_n_checkpoints)

            # Sync to GCS if configured (check the actual target, not just gs_bucket_path)
            if args.gs_checkpoint_state_dir is not None:
                ray.remote(sync_gs_bucket).options(num_cpus=1).remote(
                    checkpoint_state_dir, args.gs_checkpoint_state_dir
                )
        # add back the mpu
        if old_mpu is not None:
            self.model.mpu = old_mpu

    def save_model(self, output_dir: str, chat_template_name: str, tokenizer: PreTrainedTokenizer) -> None:
        output_path = pathlib.Path(output_dir)
        marker_path = output_path / CHECKPOINT_COMPLETE_MARKER
        if marker_path.exists():
            logger.info(f"Checkpoint already complete at {output_dir}, skipping save")
            return

        model_to_save = self.model

        if self.rank == 0:
            output_path.mkdir(parents=True, exist_ok=True)

        # save model weights for ZeRO2/3
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        # Set generation config after unwrapping to ensure it's on the actual model being saved
        # Check both chat_template_name and model name for OLMo 3 detection
        model_name = getattr(model_to_save.config, "_name_or_path", "") or ""
        is_olmo3 = (
            chat_template_name is not None and "olmo" in chat_template_name.lower()
        ) or "olmo-3" in model_name.lower()
        if is_olmo3:
            model_to_save.generation_config = get_olmo3_generation_config(tokenizer)

        # gather parameters
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            # only gather z3 params
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                vv = v.data.cpu()
                if self.rank == 0:
                    output_state_dict[k] = vv

        if self.rank == 0:
            state_dict = model_to_save.state_dict()

            # copy named_buffers with `persistent=True`
            for k, v in model_to_save.named_buffers():
                if k not in state_dict:
                    continue
                vv = v.data.cpu()
                output_state_dict[k] = vv

            state_dict_keys = set(state_dict.keys())
            output_state_dict_keys = set(output_state_dict.keys())

            # corner case for tie_word_embeddings, such as Qwen2-0.5B
            if getattr(model_to_save.config, "tie_word_embeddings", False) and "lm_head.weight" in state_dict_keys:
                state_dict_keys.remove("lm_head.weight")

            assert state_dict_keys.issubset(output_state_dict_keys), (
                f"mismatch keys {output_state_dict_keys.symmetric_difference(state_dict_keys)}"
            )

            # only save peft weights https://github.com/microsoft/DeepSpeed/issues/4295
            if isinstance(model_to_save, PeftModel):
                model_to_save.save_pretrained(output_dir)
                if getattr(self, "stage", 0) == 3:
                    torch.save(
                        get_peft_model_state_dict(model_to_save, output_state_dict), output_path / "adapter_model.bin"
                    )
            else:
                model_to_save.save_pretrained(output_dir, state_dict=output_state_dict)

            self.tokenizer.save_pretrained(output_dir)
            marker_path.touch()

    # we need this because we don't know which node is rank 0 is on
    def launch_ai2_evals_on_weka_wrapper(self, step_dir, leaderboard_name, wandb_url, training_step):
        args = self.args
        if self.rank == 0:
            ray.remote(launch_ai2_evals_on_weka).options(num_cpus=1).remote(
                path=step_dir,
                leaderboard_name=leaderboard_name,
                oe_eval_max_length=args.oe_eval_max_length,
                wandb_url=wandb_url,
                training_step=training_step,
                oe_eval_tasks=args.oe_eval_tasks,
                stop_strings=self.streaming_config.stop_strings,
                gs_bucket_path=args.gs_bucket_path,
                eval_priority=args.eval_priority,
                eval_workspace=args.eval_workspace,
                beaker_image=args.oe_eval_beaker_image,
                oe_eval_gpu_multiplier=args.oe_eval_gpu_multiplier,
            )


class ModelGroup:
    def __init__(
        self,
        pg: PlacementGroup,
        ray_process_cls: RayProcess,
        num_gpus_per_node: list[int],
        single_gpu_mode: bool,
        learner_bundle_indices: list[int] | None = None,
        args: grpo_utils.ExperimentConfig = None,
        streaming_config: data_loader_lib.StreamingDataLoaderConfig = None,
        vllm_config: data_loader_lib.VLLMConfig = None,
        data_prep_actor_name: str = "",
        tokenizer: PreTrainedTokenizer = None,
    ):
        self.pg = pg
        self.ray_process_cls = ray_process_cls
        self.num_gpus_per_node = num_gpus_per_node
        self.num_gpus_per_actor = 0.48 if single_gpu_mode else 1
        self.num_cpus_per_actor = 4
        self.models = []
        world_size = sum(self.num_gpus_per_node)

        def get_bundle_index_for_rank(rank):
            """Map a learner rank to its placement group bundle index."""
            if learner_bundle_indices is not None:
                return learner_bundle_indices[rank]
            # Legacy: per-node bundles where each bundle has multiple GPUs
            bundle_idx = 0
            r = rank
            while r >= num_gpus_per_node[bundle_idx]:
                r -= num_gpus_per_node[bundle_idx]
                bundle_idx += 1
            return bundle_idx

        # Propagate LD_PRELOAD to learner actors so they use the venv NCCL (2.27.5)
        # instead of the too-old system NCCL (2.26.6). Without this, ZeRO-3 init
        # can hang for large models (e.g. 32B) due to NCCL bugs.
        learner_runtime_env = None
        nccl_library = os.environ.get("NCCL_LIBRARY")
        if nccl_library:
            learner_runtime_env = {"env_vars": {"LD_PRELOAD": nccl_library, "NCCL_DEBUG": "INFO"}}

        master_policy = ray_process_cls.options(
            num_cpus=self.num_cpus_per_actor,
            num_gpus=self.num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_bundle_index=get_bundle_index_for_rank(0)
            ),
            runtime_env=learner_runtime_env,
        ).remote(world_size, 0, 0, None, None, args, streaming_config, vllm_config, data_prep_actor_name, tokenizer)

        self.models.append(master_policy)
        results, _ = ray_get_with_progress(
            [master_policy.get_master_addr_port.remote()], desc="Getting master address"
        )
        (master_addr, master_port) = results[0]

        # Setup worker models
        for rank in range(1, world_size):
            logger.debug(f"{rank=}, {world_size=}, {rank=}, {master_addr=}, {master_port=}")
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_bundle_index=get_bundle_index_for_rank(rank)
            )
            worker_policy = ray_process_cls.options(
                num_cpus=self.num_cpus_per_actor,
                num_gpus=self.num_gpus_per_actor,
                scheduling_strategy=scheduling_strategy,
                runtime_env=learner_runtime_env,
            ).remote(
                world_size,
                rank,
                0,
                master_addr,
                master_port,
                args,
                streaming_config,
                vllm_config,
                data_prep_actor_name,
                tokenizer,
            )
            self.models.append(worker_policy)
