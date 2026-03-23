# Geodesic Open-Instruct: Multi-Node GRPO on Isambard

Fork of [allenai/open-instruct](https://github.com/allenai/open-instruct) adapted for multi-node GRPO (Group Relative Policy Optimization) reinforcement learning on the Isambard GH200 cluster.

## What This Is

We run RL with Verifiable Rewards (RLVR) using GRPO to train language models on math and code tasks. The system uses Ray to orchestrate DeepSpeed learners and vLLM inference engines across multiple nodes, with Gloo-based weight synchronization.

## Quick Start

### 1. Submit a debug run (single node, 4 GPUs)

```bash
isambard_sbatch configs/isambard/grpo_rlzero.sbatch configs/isambard/grpo_debug_single_node.yaml
```

### 2. Check logs

```bash
tail -f /projects/a5k/public/logs_puria.a5k/open-instruct/grpo-rlzero-<job_id>.out
```

### 3. Scale to multi-node (2 nodes)

```bash
isambard_sbatch --nodes=2 configs/isambard/grpo_rlzero.sbatch configs/isambard/grpo_olmo3_7b_general.yaml
```

Default layout: 4 learners on node 0, 4 vLLM engines on node 1 (`num_learners_per_node: [4, 0]`).

## Configuration

Training configs are YAML files passed to the sbatch script. See `configs/isambard/` for examples.

| Config | Purpose |
|--------|---------|
| `configs/isambard/grpo_rlzero.sbatch` | SLURM job script: Ray cluster, env setup, job chaining |
| `configs/isambard/grpo_debug_single_node.yaml` | Debug: Qwen2.5-0.5B, single node |
| `configs/isambard/grpo_olmo3_7b_general.yaml` | General RL-Zero (math/reasoning) |
| `configs/isambard/grpo_olmo3_7b_code.yaml` | Code RL-Zero (auto-starts code server) |
| `configs/isambard/ray_node_setup_slurm.sh` | Ray worker node setup (called by sbatch) |
| `configs/isambard/run_on_compute.sbatch` | Interactive compute node access |

## W&B Tracking

Runs are tracked in the [geodesic/geodesic-grpo](https://wandb.ai/geodesic/geodesic-grpo) project. Enable with `--with_tracking` in the training config (enabled by default in debug configs).

## Architecture

```
Node 0 (Training)               Node 1 (Inference)
┌──────────────────────┐        ┌──────────────────────┐
│ GPU 0: Learner (DS)  │        │ GPU 0: vLLM Engine 0 │
│ GPU 1: Learner (DS)  │        │ GPU 1: vLLM Engine 1 │
│ GPU 2: Learner (DS)  │        │ GPU 2: vLLM Engine 2 │
│ GPU 3: Learner (DS)  │        │ GPU 3: vLLM Engine 3 │
└──────────────────────┘        └──────────────────────┘
        │                               │
        └───── Ray Cluster + Gloo ──────┘
```

See [`docs/architecture.md`](docs/architecture.md) for a thorough educational guide covering the training loop, weight sync, placement groups, and GRPO loss.

## Development

```bash
# Install
uv sync

# Lint + format
make style && make quality

# Test
uv run pytest
```

## Upstream

This is a fork of [allenai/open-instruct](https://github.com/allenai/open-instruct). To pull upstream changes:

```bash
git remote add upstream https://github.com/allenai/open-instruct.git
git fetch upstream
git merge upstream/main
```

## License

Apache 2.0 — see [LICENSE](./LICENSE).
