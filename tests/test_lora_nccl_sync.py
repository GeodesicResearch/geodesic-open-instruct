#!/usr/bin/env python3
"""Verify grad/param/optimizer sync across ranks via NCCL allreduce.

Uses Ray actors to match the production GRPO training path exactly:
- Each learner is a Ray actor with num_gpus=1 (LOCAL_RANK=0)
- MASTER_ADDR/PORT discovered via Ray (rank 0 finds free port + IP)
- DeepSpeed init_process_group same as PolicyTrainerRayProcess
- LD_PRELOAD propagated via runtime_env

Usage:
    # Start from grpo_rlzero-style sbatch (which sets up Ray cluster)
    python tests/test_lora_nccl_sync.py --model <path> --mode lora --num-nodes 2

What it does:
1. Starts a Ray-based learner group (same as ModelGroup in grpo_trainer.py)
2. Each actor loads model + LoRA/full-rank + DeepSpeed
3. Runs 5 training steps with IDENTICAL input across ranks (seeded)
4. After each step, all_gathers param checksums from every rank
5. Reports PASS/FAIL per step
"""

import argparse
import os
import sys
import time

import ray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to base model")
    parser.add_argument("--mode", choices=["lora", "full-rank"], default="lora")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--deepspeed_stage", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--num-nodes", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    args = parser.parse_args()

    if args.deepspeed_stage is None:
        args.deepspeed_stage = 0 if args.mode == "lora" else 2

    world_size = args.num_nodes * args.gpus_per_node

    # Connect to existing Ray cluster (started by sbatch)
    ray.init(address="auto")
    print(f"Ray cluster: {ray.cluster_resources()}")

    # Create placement group (same as production — PACK strategy)
    from ray.util.placement_group import placement_group, placement_group_table  # noqa: PLC0415

    bundles = [{"GPU": 1, "CPU": 4} for _ in range(world_size)]
    pg = placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())
    print(f"Placement group ready: {placement_group_table(pg)}")

    # Propagate LD_PRELOAD (same as ModelGroup in grpo_trainer.py)
    nccl_library = os.environ.get("NCCL_LIBRARY")
    runtime_env = None
    if nccl_library:
        runtime_env = {"env_vars": {"LD_PRELOAD": nccl_library, "NCCL_DEBUG": "INFO"}}

    # Create learner actors (same pattern as ModelGroup)
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy  # noqa: PLC0415

    actors = []

    # Rank 0: discovers master addr/port
    actor0 = (
        SyncTestActor.options(
            num_cpus=4,
            num_gpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=0
            ),
            runtime_env=runtime_env,
        )
        .remote(world_size, 0, None, None, args)
    )
    actors.append(actor0)

    # Get master addr/port from rank 0 (same as production)
    master_addr, master_port = ray.get(actor0.get_master_addr_port.remote())
    print(f"Master: {master_addr}:{master_port}")

    # Create worker actors
    for rank in range(1, world_size):
        actor = (
            SyncTestActor.options(
                num_cpus=4,
                num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=rank
                ),
                runtime_env=runtime_env,
            )
            .remote(world_size, rank, master_addr, master_port, args)
        )
        actors.append(actor)

    # Initialize all actors (load model, create DeepSpeed engine)
    print(f"\nInitializing {world_size} learner actors...")
    t0 = time.monotonic()
    init_results = ray.get([actor.initialize.remote() for actor in actors])
    print(f"All actors initialized in {time.monotonic() - t0:.1f}s")
    for r in init_results:
        print(f"  {r}")

    # Run sync test
    print(f"\nRunning {args.steps} training steps...")
    results = ray.get([actor.run_sync_test.remote() for actor in actors])

    # Only rank 0 returns the full report
    report = results[0]
    print(report)

    ray.shutdown()


@ray.remote(num_gpus=1)
class SyncTestActor:
    """Mirrors PolicyTrainerRayProcess + RayProcess for sync testing."""

    def __init__(self, world_size, rank, master_addr, master_port, args):
        import socket  # noqa: PLC0415

        self.world_size = world_size
        self.rank = rank
        self.args = args

        # Same as RayProcess.__init__
        if master_addr is None:
            self.master_addr = ray._private.services.get_node_ip_address().strip("[]")
        else:
            self.master_addr = master_addr
        if master_port is None:
            with socket.socket() as sock:
                sock.bind(("", 0))
                self.master_port = sock.getsockname()[1]
        else:
            self.master_port = master_port

        # Same env var setup as RayProcess
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        # Ray gives each actor 1 GPU — always LOCAL_RANK=0 (same as production)
        os.environ["LOCAL_RANK"] = "0"

    def get_master_addr_port(self):
        return self.master_addr, self.master_port

    def initialize(self):
        """Load model + DeepSpeed, same as PolicyTrainerRayProcess.from_pretrained."""
        import deepspeed  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import torch.distributed as dist  # noqa: PLC0415
        from datetime import timedelta  # noqa: PLC0415
        from transformers import AutoModelForCausalLM  # noqa: PLC0415

        args = self.args
        torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

        # Same as production: LOCAL_RANK=0, set_device(0)
        torch.cuda.set_device(0)

        # Same init sequence as grpo_trainer.py lines 184-188
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))
        deepspeed.init_distributed(timeout=timedelta(minutes=10))

        # Load model
        t0 = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch_dtype,
            use_cache=False,
            low_cpu_mem_usage=True,
            device_map={"": 0},  # LOCAL_RANK=0 always in Ray actors
        )

        load_time = time.monotonic() - t0
        mem = torch.cuda.memory_allocated() / 1024**3

        # Apply LoRA if requested
        if args.mode == "lora":
            from peft import LoraConfig, get_peft_model  # noqa: PLC0415

            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
                bias="none",
            )
            model = get_peft_model(model, lora_config)

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Optimizer (same as production)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, fused=True)

        # DeepSpeed config (same as production)
        ds_config = {
            "zero_optimization": {"stage": args.deepspeed_stage},
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 999999,
        }
        if args.deepspeed_stage >= 2:
            ds_config["zero_optimization"].update({
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
            })
        if args.dtype == "float16":
            ds_config["fp16"] = {"enabled": True}
            if args.deepspeed_stage == 0:
                ds_config["data_types"] = {"grad_accum_dtype": "fp16"}
        else:
            ds_config["bf16"] = {"enabled": True}

        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=model, optimizer=optimizer, config=ds_config, dist_init_required=False
        )

        node_ip = ray._private.services.get_node_ip_address()
        return (
            f"rank {self.rank}: node={node_ip}, "
            f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES', '?')}, "
            f"load={load_time:.1f}s, mem={mem:.1f}GiB"
        )

    def run_sync_test(self):
        """Run training steps and check sync. Only rank 0 returns full report."""
        import torch  # noqa: PLC0415
        import torch.distributed as dist  # noqa: PLC0415

        args = self.args
        all_ok = True
        lines = []

        def log(msg):
            if self.rank == 0:
                lines.append(msg)

        log(f"\n{'#'*70}")
        log(f"# NCCL Sync Test — {args.mode.upper()} (Ray actors)")
        log(f"# Model: {args.model}")
        if args.mode == "lora":
            log(f"# LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        log(f"# DeepSpeed stage: {args.deepspeed_stage}")
        log(f"# Layout: {self.world_size} Ray actors (num_gpus=1 each)")
        log(f"# Steps: {args.steps}, seq_len: {args.seq_len}, lr: {args.lr}")
        log(f"{'#'*70}")

        # Check initial weights
        checksums = self._compute_checksums()
        ok = self._gather_and_compare(checksums, 0, "INITIAL WEIGHTS", lines)
        all_ok = all_ok and ok

        for step in range(1, args.steps + 1):
            # Same input on every rank (seeded by step, not rank)
            torch.manual_seed(42 + step)
            input_ids = torch.randint(0, 32000, (1, args.seq_len), device="cuda:0")
            labels = input_ids.clone()

            outputs = self.model_engine(input_ids=input_ids, labels=labels)
            loss = outputs.loss * self.world_size  # Same scaling as production

            self.model_engine.backward(loss)

            # Check post-allreduce gradients
            grad_checksums = self._compute_checksums()
            ok = self._gather_and_compare(
                grad_checksums, step, "POST-ALLREDUCE GRADIENTS", lines
            )
            all_ok = all_ok and ok

            self.model_engine.step()

            # Check post-step weights
            param_checksums = self._compute_checksums()
            ok = self._gather_and_compare(
                param_checksums, step, "POST-STEP WEIGHTS", lines
            )
            all_ok = all_ok and ok

            # Check optimizer state (only for Stage 0)
            if args.deepspeed_stage == 0:
                opt_checksums = self._compute_optimizer_checksums()
                ok = self._gather_optimizer(opt_checksums, step, lines)
                all_ok = all_ok and ok
            else:
                log(f"\n  Optimizer state: SKIPPED (ZeRO-{args.deepspeed_stage} shards)")

            log(f"\n  loss={loss.item():.4f}, "
                f"mem={torch.cuda.memory_allocated() / 1024**3:.1f}GiB")

        log(f"\n{'#'*70}")
        if all_ok:
            log(f"# RESULT: ALL CHECKS PASSED — {args.mode.upper()} NCCL sync is correct")
        else:
            log(f"# RESULT: *** SYNC FAILURE *** — {args.mode.upper()} diverged across ranks")
        log(f"{'#'*70}\n")

        dist.barrier()

        if self.rank == 0:
            return "\n".join(lines)
        return "ok"

    def _compute_checksums(self):
        import torch  # noqa: PLC0415

        param_sum = 0.0
        param_l2 = 0.0
        grad_sum = 0.0
        grad_l2 = 0.0
        count = 0

        for name, p in self.model_engine.named_parameters():
            if not p.requires_grad:
                continue
            param_sum += p.data.float().sum().item()
            param_l2 += p.data.float().pow(2).sum().item()
            count += p.numel()
            if p.grad is not None:
                grad_sum += p.grad.float().sum().item()
                grad_l2 += p.grad.float().pow(2).sum().item()

        return {
            "param_sum": param_sum, "param_l2": param_l2,
            "grad_sum": grad_sum, "grad_l2": grad_l2, "count": count,
        }

    def _gather_and_compare(self, checksums, step, label, lines):
        import torch  # noqa: PLC0415
        import torch.distributed as dist  # noqa: PLC0415

        local_tensor = torch.tensor(
            [checksums["param_sum"], checksums["param_l2"],
             checksums["grad_sum"], checksums["grad_l2"]],
            dtype=torch.float64, device="cuda:0"
        )
        gathered = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_tensor)

        if self.rank != 0:
            return True

        lines.append(f"\n{'='*70}")
        lines.append(f"Step {step} — {label}")
        lines.append(f"{'='*70}")
        lines.append(f"  Trainable params: {checksums['count']:,} elements")

        all_ok = True
        names = ["param_sum", "param_L2", "grad_sum", "grad_L2"]
        for idx, name in enumerate(names):
            values = [g[idx].item() for g in gathered]
            ref = values[0]
            max_diff = max(abs(v - ref) for v in values)
            rel_diff = max_diff / (abs(ref) + 1e-12)
            ok = rel_diff < 1e-6
            if not ok:
                all_ok = False
            status = "PASS" if ok else "*** FAIL ***"
            lines.append(f"  {name}: {status}  (max_diff={max_diff:.2e}, rel={rel_diff:.2e})")
            for i, v in enumerate(values):
                flag = " <-- MISMATCH" if abs(v - ref) > abs(ref) * 1e-6 else ""
                lines.append(f"    rank {i}: {v:.10e}{flag}")

        return all_ok

    def _compute_optimizer_checksums(self):
        m_sum = 0.0
        v_sum = 0.0
        count = 0
        for name, p in self.model_engine.named_parameters():
            if not p.requires_grad:
                continue
            state = self.optimizer.state.get(p)
            if state is None:
                continue
            import torch  # noqa: PLC0415
            if "exp_avg" in state:
                m_sum += state["exp_avg"].float().sum().item()
            if "exp_avg_sq" in state:
                v_sum += state["exp_avg_sq"].float().sum().item()
            count += 1
        return {"adam_m_sum": m_sum, "adam_v_sum": v_sum, "count": count}

    def _gather_optimizer(self, opt_checksums, step, lines):
        import torch  # noqa: PLC0415
        import torch.distributed as dist  # noqa: PLC0415

        local_tensor = torch.tensor(
            [opt_checksums["adam_m_sum"], opt_checksums["adam_v_sum"]],
            dtype=torch.float64, device="cuda:0"
        )
        gathered = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_tensor)

        if self.rank != 0:
            return True

        lines.append(f"\n  Optimizer state (Adam m/v) — {opt_checksums['count']} param groups")
        all_ok = True
        for idx, name in enumerate(["adam_m", "adam_v"]):
            values = [g[idx].item() for g in gathered]
            ref = values[0]
            max_diff = max(abs(v - ref) for v in values)
            rel_diff = max_diff / (abs(ref) + 1e-12)
            ok = rel_diff < 1e-6
            if not ok:
                all_ok = False
            status = "PASS" if ok else "*** FAIL ***"
            lines.append(f"  {name}: {status}  (max_diff={max_diff:.2e}, rel={rel_diff:.2e})")
            for i, v in enumerate(values):
                flag = " <-- MISMATCH" if abs(v - ref) > abs(ref) * 1e-6 else ""
                lines.append(f"    rank {i}: {v:.10e}{flag}")

        return all_ok


if __name__ == "__main__":
    main()
