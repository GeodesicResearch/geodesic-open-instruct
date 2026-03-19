# ruff: noqa: PLC0415
# This class runs in vLLM worker processes; imports are inside methods.


class WorkerWrap:
    def init_process_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
        use_ray=False,
        timeout_minutes=120,
    ):
        """Init torch process group for model weights update"""
        from datetime import timedelta

        import torch

        from open_instruct.utils.vllm import init_process_group

        print("init_process_group")
        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(world_size=world_size, rank=rank, backend=backend, group_name=group_name)
            self._model_update_group = group_name
        else:
            print("init_process_group else")
            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
                timeout=timedelta(minutes=timeout_minutes),
            )
        self._model_update_with_ray = use_ray
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        import torch

        assert str(dtype) == str(self.model_config.dtype), (
            f"mismatch dtype: src {dtype}, dst {str(self.model_config.dtype)}"
        )
        weight = torch.empty(shape, dtype=self.model_config.dtype, device="cuda")
        if self._model_update_with_ray:
            import ray.util.collective as collective

            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
        import torch

        from open_instruct.utils.vllm import get_physical_gpu_id

        assert str(dtype) == str(self.model_config.dtype), (
            f"mismatch dtype: src {dtype}, dst {str(self.model_config.dtype)}"
        )
        handle = ipc_handles[get_physical_gpu_id()]
        device_id = self.device.index
        func, args = handle
        list_args = list(args)
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
        weight = func(*list_args)
        self.model_runner.model.load_weights(weights=[(name, weight)])
        torch.cuda.synchronize()

    def merge_lora_from_disk(self, lora_path, lora_int_id):
        """Merge LoRA adapter weights directly into the base model parameters.

        Handles vLLM's tensor-parallel sharding and merged param naming:
        - qkv_proj merges q_proj, k_proj, v_proj (column-parallel, shard dim 0)
        - gate_up_proj merges gate_proj, up_proj (column-parallel, shard dim 0)
        - o_proj, down_proj are row-parallel (shard dim 1)

        Fused undo+apply: stores previous adapter A/B on CPU (~268MB for r=16,
        ~1GB for r=64) instead of computed deltas (~22GB). Computes diff = new - old
        in one pass to halve parameter writes.
        """
        import json
        import os
        import time as _time

        import safetensors.torch
        import torch

        t0 = _time.monotonic()
        print(f"merge_lora_from_disk: starting, path={lora_path}, id={lora_int_id}")

        # Load adapter config and weights from NFS
        with open(os.path.join(lora_path, "adapter_config.json")) as f:
            adapter_config = json.load(f)
        lora_r = adapter_config["r"]
        lora_alpha = adapter_config["lora_alpha"]
        scaling = lora_alpha / lora_r

        t_nfs_start = _time.monotonic()
        adapter_weights = safetensors.torch.load_file(os.path.join(lora_path, "adapter_model.safetensors"))
        t_nfs = _time.monotonic() - t_nfs_start

        # TP rank and size for sharding
        tp_rank = torch.distributed.get_rank()
        tp_size = torch.distributed.get_world_size()

        # Get the unwrapped model and its parameters
        model = self.model_runner.get_model()
        model_params = dict(model.named_parameters())

        # Determine GPU device from first model param
        device = next(iter(model_params.values())).device
        model_dtype = next(iter(model_params.values())).dtype

        # Parse PEFT weight names and move A/B matrices to GPU.
        # PEFT key: base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight
        t_gpu_start = _time.monotonic()
        new_a = {}
        new_b = {}
        for key, tensor in adapter_weights.items():
            if "lora_A" in key:
                base_name = key.replace("base_model.model.", "").replace(".lora_A.weight", "")
                new_a[base_name] = tensor.to(device=device, dtype=model_dtype, non_blocking=True)
            elif "lora_B" in key:
                base_name = key.replace("base_model.model.", "").replace(".lora_B.weight", "")
                new_b[base_name] = tensor.to(device=device, dtype=model_dtype, non_blocking=True)
        del adapter_weights
        torch.cuda.synchronize()
        t_gpu_upload = _time.monotonic() - t_gpu_start

        # Load previous adapter A/B from CPU (if any) for fused undo+apply
        prev_a_cpu = getattr(self, "_lora_prev_a", {})
        prev_b_cpu = getattr(self, "_lora_prev_b", {})
        has_prev = bool(prev_a_cpu)

        # vLLM merged param mapping: vllm_suffix -> [peft_suffixes...]
        MERGED_COLUMN_PARALLEL = {
            "self_attn.qkv_proj": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            "mlp.gate_up_proj": ["mlp.gate_proj", "mlp.up_proj"],
        }
        ROW_PARALLEL = {"self_attn.o_proj", "mlp.down_proj"}

        # Collect layer prefixes
        layer_prefixes = set()
        for name in new_a:
            parts = name.rsplit(".", 2)
            if len(parts) >= 3:
                layer_prefixes.add(parts[0])

        applied_count = 0
        t_merge_start = _time.monotonic()

        def _compute_delta_shard(a_dict, b_dict, names, shard_dim):
            """Compute TP-sharded delta: concat(B_i @ A_i) * scaling, then shard."""
            deltas = []
            for name in names:
                deltas.append((b_dict[name] @ a_dict[name]) * scaling)  # noqa: F821
            full_delta = torch.cat(deltas, dim=0) if len(deltas) > 1 else deltas[0]
            shard_size = full_delta.shape[shard_dim] // tp_size  # noqa: F821
            if shard_dim == 0:
                return full_delta[tp_rank * shard_size : (tp_rank + 1) * shard_size, :].contiguous()  # noqa: F821
            else:
                return full_delta[:, tp_rank * shard_size : (tp_rank + 1) * shard_size].contiguous()  # noqa: F821

        with torch.no_grad():
            for layer_prefix in sorted(layer_prefixes):
                # Upload previous A/B for this layer to GPU (if any)
                prev_a_gpu = {}
                prev_b_gpu = {}
                if has_prev:
                    for suffix in list(new_a.keys()):
                        if suffix.startswith(layer_prefix + ".") and suffix in prev_a_cpu:
                            prev_a_gpu[suffix] = prev_a_cpu[suffix].to(
                                device=device, dtype=model_dtype, non_blocking=True
                            )
                            prev_b_gpu[suffix] = prev_b_cpu[suffix].to(
                                device=device, dtype=model_dtype, non_blocking=True
                            )
                    if prev_a_gpu:
                        torch.cuda.synchronize()

                for vllm_suffix, peft_suffixes in MERGED_COLUMN_PARALLEL.items():
                    vllm_name = f"{layer_prefix}.{vllm_suffix}.weight"
                    if vllm_name not in model_params:
                        continue
                    peft_names = [f"{layer_prefix}.{s}" for s in peft_suffixes]
                    if not all(n in new_a and n in new_b for n in peft_names):
                        continue

                    new_shard = _compute_delta_shard(new_a, new_b, peft_names, shard_dim=0)
                    if has_prev and all(n in prev_a_gpu for n in peft_names):
                        old_shard = _compute_delta_shard(prev_a_gpu, prev_b_gpu, peft_names, shard_dim=0)
                        model_params[vllm_name].data.add_(new_shard - old_shard)
                        del old_shard
                    else:
                        model_params[vllm_name].data.add_(new_shard)
                    del new_shard
                    applied_count += len(peft_names)

                for row_suffix in ROW_PARALLEL:
                    peft_name = f"{layer_prefix}.{row_suffix}"
                    vllm_name = f"{layer_prefix}.{row_suffix}.weight"
                    if vllm_name not in model_params:
                        continue
                    if peft_name not in new_a or peft_name not in new_b:
                        continue

                    new_shard = _compute_delta_shard(new_a, new_b, [peft_name], shard_dim=1)
                    if has_prev and peft_name in prev_a_gpu:
                        old_shard = _compute_delta_shard(prev_a_gpu, prev_b_gpu, [peft_name], shard_dim=1)
                        model_params[vllm_name].data.add_(new_shard - old_shard)
                        del old_shard
                    else:
                        model_params[vllm_name].data.add_(new_shard)
                    del new_shard
                    applied_count += 1

                # Free per-layer prev GPU tensors
                del prev_a_gpu, prev_b_gpu

        torch.cuda.synchronize()
        t_merge = _time.monotonic() - t_merge_start

        # Store new adapter A/B on CPU for next undo (~268MB r=16, ~1GB r=64)
        self._lora_prev_a = {k: v.cpu() for k, v in new_a.items()}
        self._lora_prev_b = {k: v.cpu() for k, v in new_b.items()}
        del new_a, new_b
        torch.cuda.empty_cache()

        t_total = _time.monotonic() - t0
        print(
            f"merge_lora_from_disk: merged {applied_count} LoRA pairs across "
            f"{len(layer_prefixes)} layers, scaling={scaling}, tp_rank={tp_rank}/{tp_size} | "
            f"timings: nfs_read={t_nfs:.1f}s, gpu_upload={t_gpu_upload:.1f}s, "
            f"merge={t_merge:.1f}s, total={t_total:.1f}s"
        )
