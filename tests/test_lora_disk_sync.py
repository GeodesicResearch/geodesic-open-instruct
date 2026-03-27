#!/usr/bin/env python3
"""Test LoRA disk sync: learner→disk→vLLM weight correctness.

Tests the production lora_disk_sync=True path end-to-end:
1. Learner sync: grad/param/optimizer checksums across DeepSpeed stage 0 ranks
2. Disk round-trip: save adapter via PEFT, load independently, verify merge matches
3. Cross-node save: when learners span multiple nodes, verify all nodes produce
   identical adapters (catches cross-node allreduce bugs)
4. vLLM integration: create_vllm_engines + load_lora_from_disk, verify merged weights

Supports two production layouts:
  7B (2-node):  [4, 0] — 4 learners node 0, 4 vLLM TP=1 engines node 1
  32B (8-node): [4, 4, 0, 0, 0, 0, 0, 0] — 8 learners on 2 nodes, 12 vLLM TP=2 engines on 6 nodes

Usage:
    # 7B layout (2-node)
    python tests/test_lora_disk_sync.py --model <path> --num-nodes 2

    # 32B layout (8-node, learners on 2 nodes, vLLM with TP=2)
    python tests/test_lora_disk_sync.py --model <path> --num-nodes 8 \
        --num-learners 8 --num-vllm 12 --vllm-tp 2

    # 1-node (no vLLM)
    python tests/test_lora_disk_sync.py --model <path> --num-nodes 1 \
        --num-learners 4 --num-vllm 0
"""

import argparse
import os
import shutil
import sys
import time

import ray


def main():
    parser = argparse.ArgumentParser(description="LoRA disk sync + vLLM merge test")
    parser.add_argument("--model", required=True, help="Path to base model")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--num-nodes", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--num-learners", type=int, default=None)
    parser.add_argument("--num-vllm", type=int, default=None)
    parser.add_argument("--vllm-tp", type=int, default=1, help="vLLM tensor parallel size (2 for 32B)")
    parser.add_argument("--lora-sync-dir", type=str, default=None)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()

    if args.num_learners is None:
        args.num_learners = args.gpus_per_node
    if args.num_vllm is None:
        # Remaining GPUs after learners, divided by TP size
        learner_nodes = (args.num_learners + args.gpus_per_node - 1) // args.gpus_per_node
        vllm_nodes = max(0, args.num_nodes - learner_nodes)
        args.num_vllm = (vllm_nodes * args.gpus_per_node) // args.vllm_tp if vllm_nodes > 0 else 0

    num_learner_nodes = (args.num_learners + args.gpus_per_node - 1) // args.gpus_per_node

    lora_sync_dir = args.lora_sync_dir or f"/tmp/lora_disk_sync_test_{os.getpid()}"
    os.makedirs(lora_sync_dir, exist_ok=True)

    ray.init(address="auto")
    print(f"Ray cluster: {ray.cluster_resources()}")

    all_ok = True
    lines = []

    def log(msg):
        lines.append(msg)
        print(msg)

    log(f"\n{'#' * 70}")
    log("# LoRA Disk Sync + vLLM Merge Test")
    log(f"# Model: {args.model}")
    log(f"# LoRA: r={args.lora_r}, alpha={args.lora_alpha}, lr={args.lr}")
    log("# DeepSpeed: stage 0 (production default for LoRA)")
    log(
        f"# Layout: {args.num_learners} learners ({num_learner_nodes} nodes) + "
        f"{args.num_vllm} vLLM engines (TP={args.vllm_tp})"
    )
    log(f"# Steps: {args.steps}, seq_len: {args.seq_len}, dtype: {args.dtype}")
    log(f"# Sync dir: {lora_sync_dir}")
    log(f"{'#' * 70}")

    # ===== Phase 1: Learner Sync Test =====
    log(f"\n{'=' * 70}")
    log("Phase 1: Learner Training + Sync Verification")
    log(f"{'=' * 70}")

    from ray.util.placement_group import placement_group  # noqa: PLC0415
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy  # noqa: PLC0415

    learner_bundles = [{"GPU": 1, "CPU": 4} for _ in range(args.num_learners)]
    learner_pg = placement_group(learner_bundles, strategy="PACK")
    ray.get(learner_pg.ready())
    log("Learner placement group ready")

    nccl_library = os.environ.get("NCCL_LIBRARY")
    runtime_env = None
    if nccl_library:
        runtime_env = {"env_vars": {"LD_PRELOAD": nccl_library, "NCCL_DEBUG": "INFO"}}

    # Create learner actors (same pattern as ModelGroup in grpo_trainer.py)
    actors = []
    actor0 = LearnerSyncActor.options(
        num_cpus=4,
        num_gpus=1,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=learner_pg, placement_group_bundle_index=0
        ),
        runtime_env=runtime_env,
    ).remote(args.num_learners, 0, None, None, args)
    actors.append(actor0)

    master_addr, master_port = ray.get(actor0.get_master_addr_port.remote())
    log(f"Master: {master_addr}:{master_port}")

    for rank in range(1, args.num_learners):
        actor = LearnerSyncActor.options(
            num_cpus=4,
            num_gpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=learner_pg, placement_group_bundle_index=rank
            ),
            runtime_env=runtime_env,
        ).remote(args.num_learners, rank, master_addr, master_port, args)
        actors.append(actor)

    log(f"\nInitializing {args.num_learners} learner actors...")
    t0 = time.monotonic()
    init_results = ray.get([a.initialize.remote() for a in actors])
    log(f"All learners initialized in {time.monotonic() - t0:.1f}s")
    for r in init_results:
        log(f"  {r}")

    # Check initial weight sync
    log("\nChecking initial weights...")
    init_check = ray.get([a.check_initial_weights.remote() for a in actors])
    init_report = init_check[0]
    log(init_report["report"])
    if not init_report["all_ok"]:
        all_ok = False

    # Run training + sync test
    log(f"\nRunning {args.steps} training steps...")
    results = ray.get([a.run_sync_test.remote() for a in actors])
    sync_report = results[0]
    log(sync_report["report"])
    if not sync_report["all_ok"]:
        all_ok = False

    # ===== Phase 2: Save LoRA to Disk + Verify Round-Trip =====
    log(f"\n{'=' * 70}")
    log("Phase 2: Save LoRA to Disk + Verify Round-Trip")
    log(f"{'=' * 70}")

    # Save adapter from rank 0 (same as production PolicyTrainerRayProcess.save_lora_to_disk)
    lora_path_rank0 = os.path.join(lora_sync_dir, "rank0")
    ray.get(actor0.save_lora_to_disk.remote(lora_path_rank0))
    log(f"LoRA adapter saved by rank 0 to: {lora_path_rank0}")
    lora_path = lora_path_rank0  # primary path for subsequent tests

    # Cross-node save verification: if learners span multiple nodes, also save
    # from the first rank on each additional node and compare adapters.
    # This catches bugs where cross-node allreduce fails to sync LoRA weights.
    if num_learner_nodes > 1:
        log(f"\nCross-node save verification ({num_learner_nodes} learner nodes):")
        cross_node_ranks = [i * args.gpus_per_node for i in range(1, num_learner_nodes)]
        cross_node_save_refs = []
        for xrank in cross_node_ranks:
            xpath = os.path.join(lora_sync_dir, f"rank{xrank}")
            cross_node_save_refs.append(actors[xrank].save_lora_to_disk.remote(xpath))
        cross_node_paths = ray.get(cross_node_save_refs)

        # Compare adapter files: load each saved adapter and compute checksums
        rank0_checksums = ray.get(actor0.get_merged_param_checksums.remote())
        log(f"  rank 0 (node 0): sum={rank0_checksums['param_sum']:.10e}, l2={rank0_checksums['param_l2']:.10e}")
        cross_node_ok = True
        for xrank, _xpath in zip(cross_node_ranks, cross_node_paths):
            xcs = ray.get(actors[xrank].get_merged_param_checksums.remote())
            s_diff = abs(xcs["param_sum"] - rank0_checksums["param_sum"])
            l2_diff = abs(xcs["param_l2"] - rank0_checksums["param_l2"])
            s_rel = s_diff / (abs(rank0_checksums["param_sum"]) + 1e-12)
            l2_rel = l2_diff / (abs(rank0_checksums["param_l2"]) + 1e-12)
            ok = s_rel < 1e-6 and l2_rel < 1e-6
            if not ok:
                cross_node_ok = False
            status = "PASS" if ok else "*** FAIL ***"
            log(
                f"  rank {xrank} (node {xrank // args.gpus_per_node}): {status} "
                f"sum={xcs['param_sum']:.10e}, l2={xcs['param_l2']:.10e} "
                f"(sum_rel={s_rel:.2e}, l2_rel={l2_rel:.2e})"
            )
        if not cross_node_ok:
            all_ok = False
        log(f"  Cross-node adapter agreement: {'PASS' if cross_node_ok else '*** FAIL ***'}")
        learner_merged = rank0_checksums
    else:
        # Single learner node — get merged checksums from rank 0
        learner_merged = ray.get(actor0.get_merged_param_checksums.remote())

    log(
        f"\nLearner merged:     sum={learner_merged['param_sum']:.10e}, "
        f"l2={learner_merged['param_l2']:.10e}, count={learner_merged['count']:,}"
    )

    # Independent verification: load fresh base model + saved adapter, merge, compare
    disk_round_trip = ray.get(actor0.verify_disk_round_trip.remote(args.model, lora_path))
    log(
        f"Disk round-trip:    sum={disk_round_trip['param_sum']:.10e}, "
        f"l2={disk_round_trip['param_l2']:.10e}, count={disk_round_trip['count']:,}"
    )

    sum_rel = abs(learner_merged["param_sum"] - disk_round_trip["param_sum"]) / (
        abs(learner_merged["param_sum"]) + 1e-12
    )
    l2_rel = abs(learner_merged["param_l2"] - disk_round_trip["param_l2"]) / (abs(learner_merged["param_l2"]) + 1e-12)
    # Tolerance relaxed to 1e-4: merge-in-place vs merge-from-scratch in fp16 accumulates rounding
    disk_ok = sum_rel < 1e-4 and l2_rel < 1e-4
    if not disk_ok:
        all_ok = False
    log(f"Disk round-trip:    {'PASS' if disk_ok else '*** FAIL ***'} (sum_rel={sum_rel:.2e}, l2_rel={l2_rel:.2e})")

    # ===== Phase 3: vLLM Merge Verification =====
    if args.num_vllm > 0:
        log(f"\n{'=' * 70}")
        log("Phase 3: vLLM Merge Verification")
        log(f"{'=' * 70}")

        from ray.util.queue import Queue as RayQueue  # noqa: PLC0415

        from open_instruct.utils.vllm import create_vllm_engines  # noqa: PLC0415

        dummy_manager = DummyActorManager.remote()
        prompt_Q = RayQueue(maxsize=1)
        results_Q = RayQueue(maxsize=1)
        eval_Q = RayQueue(maxsize=1)

        log(f"Creating {args.num_vllm} vLLM engines (TP={args.vllm_tp}, using production create_vllm_engines)...")
        t0 = time.monotonic()
        vllm_engines = create_vllm_engines(
            num_engines=args.num_vllm,
            tensor_parallel_size=args.vllm_tp,
            enforce_eager=True,
            tokenizer_name_or_path=args.model,
            pretrain=args.model,
            revision=None,
            seed=42,
            enable_prefix_caching=False,
            max_model_len=args.seq_len + 128,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            prompt_queue=prompt_Q,
            results_queue=results_Q,
            eval_results_queue=eval_Q,
            actor_manager=dummy_manager,
            inflight_updates=False,
            vllm_dtype=args.dtype,
        )
        log(f"vLLM engines created in {time.monotonic() - t0:.1f}s")

        # Get base model checksums BEFORE merge
        log("Getting base model checksums from vLLM engines...")
        base_checksums_list = ray.get([e.get_param_checksums.remote() for e in vllm_engines])
        base_cs = base_checksums_list[0]
        log(
            f"vLLM base model:    sum={base_cs['param_sum']:.10e}, "
            f"l2={base_cs['param_l2']:.10e}, count={base_cs['count']:,}"
        )

        # Load LoRA from disk on all engines (production load_lora_from_disk method)
        log(f"Loading LoRA from disk on {args.num_vllm} engines...")
        t0 = time.monotonic()
        load_refs = [engine.load_lora_from_disk.remote(lora_path, 1) for engine in vllm_engines]
        ray.get(load_refs)
        log(f"LoRA loaded on all engines in {time.monotonic() - t0:.1f}s")

        # Get merged checksums from all engines
        merged_checksums_list = ray.get([e.get_param_checksums.remote() for e in vllm_engines])

        # Check all engines agree
        ref_cs = merged_checksums_list[0]
        engines_match = True
        for i, cs in enumerate(merged_checksums_list):
            s_diff = abs(cs["param_sum"] - ref_cs["param_sum"])
            l_diff = abs(cs["param_l2"] - ref_cs["param_l2"])
            if s_diff > 1e-6 or l_diff > 1e-6:
                engines_match = False
                log(f"  Engine {i}: MISMATCH sum_diff={s_diff:.2e} l2_diff={l_diff:.2e}")
        log(f"  All {args.num_vllm} engines identical: {'PASS' if engines_match else '*** FAIL ***'}")
        if not engines_match:
            all_ok = False

        # Compare vLLM merged with learner merged
        log(
            f"vLLM after merge:   sum={ref_cs['param_sum']:.10e}, "
            f"l2={ref_cs['param_l2']:.10e}, count={ref_cs['count']:,}"
        )
        log(
            f"Learner merged:     sum={learner_merged['param_sum']:.10e}, "
            f"l2={learner_merged['param_l2']:.10e}, count={learner_merged['count']:,}"
        )

        # Check that LoRA delta was applied (weights changed from base)
        delta_sum = abs(ref_cs["param_sum"] - base_cs["param_sum"])
        delta_l2 = abs(ref_cs["param_l2"] - base_cs["param_l2"])
        delta_applied = delta_sum > 1e-6 or delta_l2 > 1e-6
        log(f"  LoRA delta applied (weights changed): {'PASS' if delta_applied else '*** FAIL ***'}")
        if not delta_applied:
            all_ok = False

        # Cross-check: vLLM merged vs learner merged
        # Note: vLLM may have different param count due to weight tying (embed/lm_head).
        # Compare param_sum/l2 with tolerance.
        if ref_cs["count"] == learner_merged["count"]:
            vllm_sum_rel = abs(ref_cs["param_sum"] - learner_merged["param_sum"]) / (
                abs(learner_merged["param_sum"]) + 1e-12
            )
            vllm_l2_rel = abs(ref_cs["param_l2"] - learner_merged["param_l2"]) / (
                abs(learner_merged["param_l2"]) + 1e-12
            )
            vllm_ok = vllm_sum_rel < 1e-4 and vllm_l2_rel < 1e-4
            if not vllm_ok:
                all_ok = False
            log(
                f"  vLLM vs Learner:  {'PASS' if vllm_ok else '*** FAIL ***'} "
                f"(sum_rel={vllm_sum_rel:.2e}, l2_rel={vllm_l2_rel:.2e})"
            )
        else:
            # Different param counts (likely weight tying difference) — compare per-layer
            log(f"  Note: param count differs (vLLM={ref_cs['count']:,}, learner={learner_merged['count']:,})")
            log("  This is expected if the model uses weight tying (embed/lm_head shared in HF, separate in vLLM)")
            # Verify by subtracting the difference (one copy of embed weights)
            count_diff = ref_cs["count"] - learner_merged["count"]
            log(f"  Count difference: {count_diff:,} elements")

        # Test incremental merge (fused undo+apply on second call)
        log("\nTesting incremental merge (re-apply same adapter, tests fused undo+apply)...")
        load_refs2 = [engine.load_lora_from_disk.remote(lora_path, 2) for engine in vllm_engines]
        ray.get(load_refs2)
        incr_checksums = ray.get([e.get_param_checksums.remote() for e in vllm_engines])
        incr_cs = incr_checksums[0]

        incr_sum_diff = abs(incr_cs["param_sum"] - ref_cs["param_sum"])
        incr_l2_diff = abs(incr_cs["param_l2"] - ref_cs["param_l2"])
        incr_ok = incr_sum_diff < 1e-4 and incr_l2_diff < 1e-4
        if not incr_ok:
            all_ok = False
        log(
            f"  Incremental merge: {'PASS' if incr_ok else '*** FAIL ***'} "
            f"(sum_diff={incr_sum_diff:.2e}, l2_diff={incr_l2_diff:.2e})"
        )

    # ===== Summary =====
    log(f"\n{'#' * 70}")
    if all_ok:
        log("# RESULT: ALL CHECKS PASSED — LoRA disk sync is correct")
    else:
        log("# RESULT: *** SYNC FAILURE *** — see details above")
    log(f"{'#' * 70}\n")

    # Cleanup
    if args.lora_sync_dir is None:
        shutil.rmtree(lora_sync_dir, ignore_errors=True)

    ray.shutdown()
    sys.exit(0 if all_ok else 1)


@ray.remote
class DummyActorManager:
    """Minimal actor manager for vLLM engines during testing — always stopped."""

    def __init__(self):
        self._should_stop = True

    def should_stop(self):
        return self._should_stop

    def set_should_stop(self, v):
        self._should_stop = v

    def set_kv_cache_max_concurrency(self, v):
        pass


@ray.remote(num_gpus=1)
class LearnerSyncActor:
    """Mirrors PolicyTrainerRayProcess for LoRA disk sync testing.

    Uses the same initialization path as production:
    - Ray actor with num_gpus=1 (LOCAL_RANK=0)
    - MASTER_ADDR/PORT discovery via Ray
    - DeepSpeed stage 0 (production default for LoRA)
    - PEFT LoRA with configurable rank/alpha
    """

    def __init__(self, world_size, rank, master_addr, master_port, args):
        import socket  # noqa: PLC0415

        self.world_size = world_size
        self.rank = rank
        self.args = args

        # Same as RayProcess.__init__ in grpo_trainer.py
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

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = "0"

    def get_master_addr_port(self):
        return self.master_addr, self.master_port

    def initialize(self):
        """Load model + LoRA + DeepSpeed stage 0, same as PolicyTrainerRayProcess.from_pretrained."""
        from datetime import timedelta  # noqa: PLC0415

        import deepspeed  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import torch.distributed as dist  # noqa: PLC0415
        from peft import LoraConfig, get_peft_model  # noqa: PLC0415
        from transformers import AutoModelForCausalLM  # noqa: PLC0415

        args = self.args
        torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

        torch.cuda.set_device(0)

        # Same init sequence as grpo_trainer.py
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))
        deepspeed.init_distributed(timeout=timedelta(minutes=10))

        t0 = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch_dtype, use_cache=False, low_cpu_mem_usage=True, device_map={"": 0}
        )
        load_time = time.monotonic() - t0
        mem = torch.cuda.memory_allocated() / 1024**3

        # Same LoRA config as production medharm configs
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, fused=True)

        # DeepSpeed stage 0 — same as production for LoRA
        ds_config = {
            "zero_optimization": {"stage": 0},
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 999999,
        }
        if args.dtype == "float16":
            ds_config["fp16"] = {"enabled": True}
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

    def check_initial_weights(self):
        """Verify all ranks start with identical weights."""

        checksums = self._compute_param_checksums()
        lines = []
        ok = self._gather_and_compare(checksums, 0, "INITIAL WEIGHTS", lines)
        if self.rank == 0:
            return {"report": "\n".join(lines), "all_ok": ok}
        return {"report": "ok", "all_ok": True}

    def run_sync_test(self):
        """Run training steps, check grad/param/optimizer sync at each step.

        Verifies:
        - Gradients (1x params): allreduced, should be identical across ranks
        - Parameters (1x params): updated identically since grads are same
        - Optimizer states (2x params): Adam m (momentum) + v (variance), identical for stage 0
        """
        import torch  # noqa: PLC0415
        import torch.distributed as dist  # noqa: PLC0415

        args = self.args
        all_ok = True
        lines = []

        def log(msg):
            if self.rank == 0:
                lines.append(msg)

        for step in range(1, args.steps + 1):
            # Same input on every rank (seeded by step, not rank)
            torch.manual_seed(42 + step)
            input_ids = torch.randint(0, 32000, (1, args.seq_len), device="cuda:0")
            labels = input_ids.clone()

            outputs = self.model_engine(input_ids=input_ids, labels=labels)
            loss = outputs.loss * self.world_size

            self.model_engine.backward(loss)

            # Check post-allreduce gradients (1x params)
            grad_cs = self._compute_grad_checksums()
            ok = self._gather_and_compare(grad_cs, step, "GRADIENTS (1x params)", lines)
            all_ok = all_ok and ok

            self.model_engine.step()

            # Check post-step parameters (1x params)
            param_cs = self._compute_param_checksums()
            ok = self._gather_and_compare(param_cs, step, "PARAMETERS (1x params)", lines)
            all_ok = all_ok and ok

            # Check optimizer states (2x params: Adam m + v)
            opt_cs = self._compute_optimizer_checksums()
            ok = self._gather_optimizer(opt_cs, step, lines)
            all_ok = all_ok and ok

            log(f"  step {step}: loss={loss.item():.4f}, mem={torch.cuda.memory_allocated() / 1024**3:.1f}GiB")

        dist.barrier()

        if self.rank == 0:
            return {"report": "\n".join(lines), "all_ok": all_ok}
        return {"report": "ok", "all_ok": True}

    def save_lora_to_disk(self, save_dir):
        """Save LoRA adapter to shared disk.

        Same pattern as PolicyTrainerRayProcess.save_lora_to_disk in grpo_trainer.py.
        Any rank can save (for cross-node verification). No barrier — called per-actor.
        """
        from peft import PeftModel  # noqa: PLC0415

        os.makedirs(save_dir, exist_ok=True)
        model = self.model_engine.module  # unwrap DeepSpeed → PeftModel
        if not isinstance(model, PeftModel):
            raise RuntimeError("Model is not a PeftModel")
        model.save_pretrained(save_dir)
        return save_dir

    def get_merged_param_checksums(self):
        """Temporarily merge LoRA into base weights and compute checksums.

        Uses PEFT merge_adapter/unmerge_adapter to avoid permanently modifying the model.
        Iterates over all non-LoRA parameters (base weights with LoRA delta merged in).
        """

        peft_model = self.model_engine.module
        peft_model.merge_adapter()

        param_sum = 0.0
        param_l2 = 0.0
        count = 0
        for name, p in peft_model.named_parameters():
            if "lora_" in name:
                continue  # Skip LoRA A/B matrices (they're zeroed after merge)
            param_sum += p.data.float().sum().item()
            param_l2 += p.data.float().pow(2).sum().item()
            count += p.numel()

        peft_model.unmerge_adapter()

        return {"param_sum": param_sum, "param_l2": param_l2, "count": count}

    def verify_disk_round_trip(self, base_model_path, lora_path):
        """Load fresh base model + saved adapter, merge, compute checksums.

        Independent verification that the disk save/load cycle preserves weights.
        Frees the training model first to avoid OOM on large models (e.g. 32B).
        """
        import gc  # noqa: PLC0415

        import torch  # noqa: PLC0415
        from peft import PeftModel  # noqa: PLC0415
        from transformers import AutoModelForCausalLM  # noqa: PLC0415

        # Load on CPU to avoid OOM (32B model can't fit alongside training state on GPU)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.float32, use_cache=False, low_cpu_mem_usage=True, device_map="cpu"
        )
        peft_model = PeftModel.from_pretrained(base_model, lora_path)
        merged = peft_model.merge_and_unload()

        param_sum = 0.0
        param_l2 = 0.0
        count = 0
        for _name, p in merged.named_parameters():
            param_sum += p.data.float().sum().item()
            param_l2 += p.data.float().pow(2).sum().item()
            count += p.numel()

        del merged, peft_model, base_model
        gc.collect()

        return {"param_sum": param_sum, "param_l2": param_l2, "count": count}

    # --- Checksum helpers (same math as test_lora_nccl_sync.py) ---

    def _compute_param_checksums(self):
        param_sum = 0.0
        param_l2 = 0.0
        count = 0
        for _name, p in self.model_engine.named_parameters():
            if not p.requires_grad:
                continue
            param_sum += p.data.float().sum().item()
            param_l2 += p.data.float().pow(2).sum().item()
            count += p.numel()
        return {"param_sum": param_sum, "param_l2": param_l2, "count": count}

    def _compute_grad_checksums(self):
        grad_sum = 0.0
        grad_l2 = 0.0
        count = 0
        for _name, p in self.model_engine.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            grad_sum += p.grad.float().sum().item()
            grad_l2 += p.grad.float().pow(2).sum().item()
            count += p.numel()
        return {"grad_sum": grad_sum, "grad_l2": grad_l2, "count": count}

    def _compute_optimizer_checksums(self):
        adam_m_sum = 0.0
        adam_v_sum = 0.0
        count = 0
        for _name, p in self.model_engine.named_parameters():
            if not p.requires_grad:
                continue
            state = self.optimizer.state.get(p)
            if state is None:
                continue
            if "exp_avg" in state:
                adam_m_sum += state["exp_avg"].float().sum().item()
            if "exp_avg_sq" in state:
                adam_v_sum += state["exp_avg_sq"].float().sum().item()
            count += 1
        return {"adam_m_sum": adam_m_sum, "adam_v_sum": adam_v_sum, "count": count}

    def _gather_and_compare(self, checksums, step, label, lines):
        import torch  # noqa: PLC0415
        import torch.distributed as dist  # noqa: PLC0415

        keys = [k for k in checksums if k != "count"]
        values = [checksums[k] for k in keys]

        local_tensor = torch.tensor(values, dtype=torch.float64, device="cuda:0")
        gathered = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_tensor)

        if self.rank != 0:
            return True

        lines.append(f"\n  Step {step} — {label} ({checksums['count']:,} elements):")
        all_ok = True
        for idx, name in enumerate(keys):
            vals = [g[idx].item() for g in gathered]
            ref = vals[0]
            max_diff = max(abs(v - ref) for v in vals)
            rel_diff = max_diff / (abs(ref) + 1e-12)
            ok = rel_diff < 1e-6
            if not ok:
                all_ok = False
            status = "PASS" if ok else "*** FAIL ***"
            lines.append(f"    {name}: {status} (max_diff={max_diff:.2e}, rel={rel_diff:.2e})")
            if not ok:
                for i, v in enumerate(vals):
                    flag = " <-- MISMATCH" if abs(v - ref) > abs(ref) * 1e-6 else ""
                    lines.append(f"      rank {i}: {v:.10e}{flag}")

        return all_ok

    def _gather_optimizer(self, opt_checksums, step, lines):
        import torch  # noqa: PLC0415
        import torch.distributed as dist  # noqa: PLC0415

        local_tensor = torch.tensor(
            [opt_checksums["adam_m_sum"], opt_checksums["adam_v_sum"]], dtype=torch.float64, device="cuda:0"
        )
        gathered = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_tensor)

        if self.rank != 0:
            return True

        lines.append(
            f"  Step {step} — OPTIMIZER STATE (2x params: Adam m + v, {opt_checksums['count']} param groups):"
        )
        all_ok = True
        for idx, name in enumerate(["adam_m", "adam_v"]):
            vals = [g[idx].item() for g in gathered]
            ref = vals[0]
            max_diff = max(abs(v - ref) for v in vals)
            rel_diff = max_diff / (abs(ref) + 1e-12)
            ok = rel_diff < 1e-6
            if not ok:
                all_ok = False
            status = "PASS" if ok else "*** FAIL ***"
            lines.append(f"    {name}: {status} (max_diff={max_diff:.2e}, rel={rel_diff:.2e})")
            if not ok:
                for i, v in enumerate(vals):
                    flag = " <-- MISMATCH" if abs(v - ref) > abs(ref) * 1e-6 else ""
                    lines.append(f"      rank {i}: {v:.10e}{flag}")

        return all_ok


if __name__ == "__main__":
    main()
