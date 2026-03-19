#!/bin/bash
# SLURM launcher for NeoX SFT training on Isambard GH200.
#
# Usage:
#   isambard_sbatch --nodes=1 warm-start-neox/sft_neox.sbatch warm-start-neox/sft_olmo3_32b.yml
#
# Activates the NeoX venv, sets up NCCL for Slingshot, generates hostfile
# for multi-node, and runs deepy.py train.py with the given config.

#SBATCH --job-name=neox-sft
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/projects/a5k/public/logs_%u/open-instruct/neox-sft-%j.out

set -euo pipefail

NEOX_REPO="${NEOX_REPO:-/home/a5k/puria.a5k/geodesic-gpt-neox}"
NEOX_CONFIG="${1:?Usage: sft_neox.sbatch <neox_config.yml>}"

# Resolve config to absolute path
if [[ "$NEOX_CONFIG" != /* ]]; then
    NEOX_CONFIG="${SLURM_SUBMIT_DIR}/${NEOX_CONFIG}"
fi

echo "===== NeoX SFT Job ====="
echo "Job ID:    $SLURM_JOB_ID"
echo "Nodes:     $SLURM_NNODES"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "NeoX repo: $NEOX_REPO"
echo "Config:    $NEOX_CONFIG"
echo "========================="

# Validate NeoX repo and venv
if [[ ! -f "$NEOX_REPO/deepy.py" ]]; then
    echo "ERROR: NeoX repo not found at $NEOX_REPO" >&2
    echo "Set NEOX_REPO env var to the geodesic-gpt-neox repo path" >&2
    exit 1
fi

NEOX_VENV="$NEOX_REPO/.venv/bin/activate"
if [[ ! -f "$NEOX_VENV" ]]; then
    echo "ERROR: NeoX venv not found at $NEOX_VENV" >&2
    echo "Build the NeoX venv first (see geodesic-gpt-neox README)" >&2
    exit 1
fi

# Activate NeoX venv
source "$NEOX_VENV"

# Load modules
module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1

# Use bundled NCCL from venv
VENV_SP="$NEOX_REPO/.venv/lib/python3.12/site-packages"
export NCCL_LIBRARY="$VENV_SP/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="$NCCL_LIBRARY"

# Compilers and CUDA arch
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"

# CUDA paths for JIT compilation (e.g., DeepSpeed cpu_adam needs libcurand)
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
CUDA_MATH_LIBS=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/REDIST/math_libs/12.6/targets/sbsa-linux/lib
export LIBRARY_PATH="${CUDA_MATH_LIBS}:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_MATH_LIBS}:${LD_LIBRARY_PATH:-}"

# NCCL / OFI settings for Slingshot (CXI)
export NCCL_COLLNET_ENABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NET="AWS Libfabric"
export FI_PROVIDER=cxi
export NCCL_SOCKET_IFNAME=hsn
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120

# Memory allocator optimization — reduces fragmentation for large models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export HF_HUB_OFFLINE=1

# Use node-local temp to avoid NFS issues
export TMPDIR="/tmp/neox_sft_${USER}_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

# Print PyTorch / CUDA info
echo "===== PyTorch & CUDA info ====="
python - <<'PY'
import os, torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"Visible GPUs: {n}")
    for i in range(n):
        print(f"  GPU[{i}]: {torch.cuda.get_device_name(i)}")
PY
echo "================================"

# Multi-node handling:
# NeoX's default launcher (pdsh/SSH) doesn't work on Isambard (no passwordless SSH).
# For multi-node, set "launcher": "slurm" in the config YAML — DeepSpeed will use srun.
# The hostfile is still needed so DeepSpeed knows about all nodes.
if [ "$SLURM_NNODES" -gt 1 ]; then
    HOSTFILE="$TMPDIR/hostfile"
    scontrol show hostname "$SLURM_NODELIST" | while read -r node; do
        echo "${node} slots=4"
    done > "$HOSTFILE"
    echo "Multi-node job ($SLURM_NNODES nodes): using SLURM launcher"
    echo "Hostfile:"
    cat "$HOSTFILE"
    export DLTS_HOSTFILE="$HOSTFILE"
else
    echo "Single-node job: using local launcher"
    unset DLTS_HOSTFILE
fi

# Launch NeoX training
cd "$NEOX_REPO"
echo ""
echo "===== Starting NeoX SFT ====="
python deepy.py train.py "$NEOX_CONFIG"
EXIT_CODE=$?

echo ""
echo "===== NeoX SFT Finished ====="
echo "Exit code: $EXIT_CODE"
echo "Date: $(date)"
exit $EXIT_CODE
