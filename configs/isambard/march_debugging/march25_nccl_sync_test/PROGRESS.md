# NCCL Sync Diagnostic Test — Progress Log

**Started:** 2026-03-25
**Purpose:** Verify that NCCL allreduce correctly synchronizes gradients, weights, and optimizer states across nodes for both LoRA and full-rank training. Tests whether cross-node communication is the cause of 32B LoRA failing to learn.

## Test Script

`tests/test_lora_nccl_sync.py` — runs 5 training steps with identical seeded input on all ranks, then checks:
1. Post-allreduce gradient checksums (should be identical across ranks)
2. Post-step weight checksums (should be identical across ranks)
3. Adam m/v optimizer state checksums (identical for Stage 0; sharded for Stage 2 — skipped)

## v3 Results: Ray Actor Path — ALL 10 TESTS PASSED

**Motivation:** v2 tests used `srun` for process launch, but production GRPO uses Ray actors. v3 rewrote both test script and sbatch to use Ray actors matching production exactly (placement groups, `num_gpus=1`, `LOCAL_RANK=0`, master addr/port via Ray, `LD_PRELOAD` via `runtime_env`).

**Conclusion:** NCCL sync is correct even through the exact Ray actor path used in production. All checksums had **zero diff** across ranks. Cross-node gradient sync is definitively not the cause of 32B LoRA failing to learn.

### LoRA Tests (DeepSpeed Stage 0)

| Job | Test | Nodes | Model | Run | Result |
|-----|------|-------|-------|-----|--------|
| 3361696 | 2-node 32B LoRA | 2 | olmo-3-1125-32b | 1 | **PASS** |
| 3361698 | 2-node 32B LoRA | 2 | olmo-3-1125-32b | 2 | **PASS** |
| 3361699 | 2-node 7B LoRA | 2 | OLMo-3-7B | 1 | **PASS** |
| 3361700 | 2-node 7B LoRA | 2 | OLMo-3-7B | 2 | **PASS** |
| 3361713 | 1-node 7B LoRA (control) | 1 | OLMo-3-7B | 1 | **PASS** |
| 3361702 | 1-node 7B LoRA (control) | 1 | OLMo-3-7B | 2 | **PASS** |

### Full-Rank Tests (DeepSpeed Stage 2)

| Job | Test | Nodes | Model | Run | Result |
|-----|------|-------|-------|-----|--------|
| 3361703 | 2-node 7B full-rank | 2 | OLMo-3-7B | 1 | **PASS** |
| 3361705 | 2-node 7B full-rank | 2 | OLMo-3-7B | 2 | **PASS** |
| 3361706 | 1-node 7B full-rank (control) | 1 | OLMo-3-7B | 1 | **PASS** |
| 3361707 | 1-node 7B full-rank (control) | 1 | OLMo-3-7B | 2 | **PASS** |

### Failed (infra, not sync)

| Job | Issue |
|-----|-------|
| 3361701 | "No space left on device" on node /tmp — replaced by 3361713 |

## v2 Results: srun Path — ALL 10 TESTS PASSED

### LoRA Tests (DeepSpeed Stage 0)

| Job | Test | Nodes | Model | Run | Result |
|-----|------|-------|-------|-----|--------|
| 3361424 | 2-node 32B LoRA | 2 | olmo-3-1125-32b | 1 | **PASS** |
| 3361425 | 2-node 32B LoRA | 2 | olmo-3-1125-32b | 2 | **PASS** |
| 3361426 | 2-node 7B LoRA | 2 | OLMo-3-7B | 1 | **PASS** |
| 3361427 | 2-node 7B LoRA | 2 | OLMo-3-7B | 2 | **PASS** |
| 3361428 | 1-node 7B LoRA (control) | 1 | OLMo-3-7B | 1 | **PASS** |
| 3361429 | 1-node 7B LoRA (control) | 1 | OLMo-3-7B | 2 | **PASS** |

### Full-Rank Tests (DeepSpeed Stage 2)

| Job | Test | Nodes | Model | Run | Result |
|-----|------|-------|-------|-----|--------|
| 3361430 | 2-node 7B full-rank | 2 | OLMo-3-7B | 1 | **PASS** |
| 3361431 | 2-node 7B full-rank | 2 | OLMo-3-7B | 2 | **PASS** |
| 3361432 | 1-node 7B full-rank (control) | 1 | OLMo-3-7B | 1 | **PASS** |
| 3361433 | 1-node 7B full-rank (control) | 1 | OLMo-3-7B | 2 | **PASS** |

## Log

### v1 (3361396-3361405) — all FAILED, env misconfiguration
- Missing `module load PrgEnv-cray`, `NCCL_SOCKET_IFNAME=hsn`, MASTER_PORT offset
- NCCL error: "Failed to initialize any NET plugin"
- Fixed sbatch to mirror grpo_rlzero.sbatch env setup exactly

### v2 (3361424-3361433) — all PASSED (srun-based)
- 10/10 tests passed across both LoRA (Stage 0) and full-rank (Stage 2)
- Both 1-node and 2-node configurations verified
- 32B model (2-node, 8 ranks) gradient/weight/optimizer sync confirmed correct
- **Caveat:** Used `srun` for process launch; production uses Ray actors

### v3 (3361696-3361713) — all PASSED (Ray actor-based)
- 10/10 tests passed (1 infra failure replaced)
- Rewrote test script to use Ray actors matching production exactly
- Rewrote sbatch to start Ray cluster (same as grpo_rlzero.sbatch)
- All checksums had exactly zero diff across ranks
- Eliminates the srun vs Ray discrepancy from v2
- **Definitive conclusion:** NCCL sync is not the issue
