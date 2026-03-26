#!/bin/bash
# Test LoRA/full-rank NCCL sync across nodes using Ray actors.
# Mirrors the exact environment + Ray cluster setup from grpo_rlzero.sbatch.
#
# Usage (2 nodes, mirrors 32B layout):
#   isambard_sbatch --nodes=2 configs/isambard/test_lora_sync.sbatch \
#       --model /projects/a5k/public/models_puria.a5k/olmo-3-1125-32b --mode lora
#
# Usage (1 node, control):
#   isambard_sbatch --nodes=1 configs/isambard/test_lora_sync.sbatch \
#       --model /projects/a5k/public/models_puria.a5k/OLMo-3-7B --mode full-rank

#SBATCH --job-name=lora-sync-test
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=/projects/a5k/public/logs_%u/open-instruct/lora-sync-test-%j.out
#SBATCH --exclude=nid010798,nid010869,nid010632

# --- Isambard node-limit guard ---
if ! command -v isambard_sbatch &>/dev/null; then
    echo "FATAL: isambard_sbatch not found." >&2
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi
if ! isambard_sbatch --check; then
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi

set -euo pipefail

export REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"

echo "===== LoRA/Full-Rank NCCL Sync Test (Ray) ====="
echo "Job ID:    $SLURM_JOB_ID"
echo "Nodes:     $SLURM_NNODES ($(scontrol show hostname $SLURM_NODELIST | tr '\n' ' '))"
echo "Date:      $(date)"
echo "Args:      $*"
echo ""

# --- Activate venv ---
source "$REPO_DIR/.venv/bin/activate"

# --- Load modules (mirror grpo_rlzero.sbatch exactly) ---
module purge
module load PrgEnv-cray
module load cuda/12.6

# --- Custom NCCL from venv (same as production) ---
VENV_SP="$REPO_DIR/.venv/lib/python3.12/site-packages"
export NCCL_LIBRARY="$VENV_SP/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="$NCCL_LIBRARY"

# --- Compilers and CUDA ---
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6

# --- NCCL network settings (socket transport over Slingshot) ---
export NCCL_SOCKET_IFNAME=hsn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_CUMEM_ENABLE=0
export NCCL_DEBUG=INFO
if [[ "$SLURM_NNODES" -ge 3 ]]; then
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-ALL}
else
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT}
fi

# --- Networking (resolve to IP, same as production) ---
HEAD_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export HEAD_IP=$(getent hosts "$HEAD_HOSTNAME" | awk '{print $1; exit}')
export MASTER_ADDR="$HEAD_IP"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# --- HuggingFace ---
export HF_HOME="/projects/a5k/public/hf_${USER}"
export HF_HUB_OFFLINE=1

# --- Temp directory ---
export TMPDIR=/projects/a5k/public/tmp_${USER}
mkdir -p "$TMPDIR"

# Ray needs node-local temp for Unix domain sockets
export RAY_TMPDIR="/tmp/ray_${USER}_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

# Triton/inductor caches must be node-local
export TRITON_CACHE_DIR="/tmp/triton_${USER}_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# --- vLLM / Ray env vars (same as production) ---
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_TORCH_COMPILE_LEVEL=0
export RAY_CGRAPH_get_timeout=300
export RAY_raylet_start_wait_time_s=120
export RAY_gcs_rpc_server_connect_timeout_s=60

# --- Unbuffered output ---
export PYTHONUNBUFFERED=1

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NCCL_LIBRARY=$NCCL_LIBRARY"
echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "NCCL_DEBUG=$NCCL_DEBUG, NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS"
echo ""

# --- Start Ray cluster (same as grpo_rlzero.sbatch) ---
RAY_NODE_PORT=8888
ray stop --force 2>/dev/null || true

echo "Starting Ray head on $(hostname) (IP: $HEAD_IP)..."
ray start --head --port=$RAY_NODE_PORT --node-ip-address="$HEAD_IP" \
    --temp-dir="$RAY_TMPDIR" --num-gpus=4 --num-cpus=32 --dashboard-host=0.0.0.0

export RAY_ADDRESS="$HEAD_IP:$RAY_NODE_PORT"

# Start workers on non-head nodes (if multi-node)
SRUN_PID=""
if [ "$SLURM_NNODES" -gt 1 ]; then
    HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
    echo "Starting Ray workers on $((SLURM_NNODES - 1)) non-head nodes..."
    srun --export=ALL --nodes=$((SLURM_NNODES - 1)) --exclude="$HEAD_NODE" \
         bash configs/isambard/ray_node_setup_slurm.sh &
    SRUN_PID=$!
    # Wait for all nodes to join
    EXPECTED_GPUS=$((SLURM_NNODES * 4))
    echo "Waiting for $SLURM_NNODES nodes ($EXPECTED_GPUS GPUs) to join Ray cluster..."
    for i in $(seq 1 180); do
        CURRENT_GPUS=$(python -c "
import ray, sys
try:
    ray.init(address='$HEAD_IP:$RAY_NODE_PORT', ignore_reinit_error=True)
    gpus = ray.cluster_resources().get('GPU', 0)
    print(int(gpus))
except:
    print(0)
" 2>/dev/null)
        if [ "$CURRENT_GPUS" -ge "$EXPECTED_GPUS" ]; then
            echo "All $SLURM_NNODES nodes registered ($CURRENT_GPUS GPUs)."
            break
        fi
        if [ $((i % 30)) -eq 0 ]; then
            echo "  ... $CURRENT_GPUS/$EXPECTED_GPUS GPUs after ${i}s"
        fi
        sleep 1
    done
    if [ "$CURRENT_GPUS" -lt "$EXPECTED_GPUS" ]; then
        echo "ERROR: Only $CURRENT_GPUS/$EXPECTED_GPUS GPUs after 180s. Aborting."
        ray status || true
        exit 1
    fi
fi

echo "===== Ray Cluster Status ====="
ray status || echo "Warning: ray status failed"
echo "==============================="

# --- Run sync test on head node ---
echo "Launching sync test..."
LD_PRELOAD="$NCCL_LIBRARY" python tests/test_lora_nccl_sync.py \
    --num-nodes "$SLURM_NNODES" --gpus-per-node 4 "$@"
TEST_EXIT=$?

# --- Cleanup ---
echo "Cleaning up..."
if [ -n "$SRUN_PID" ]; then
    kill $SRUN_PID 2>/dev/null; wait $SRUN_PID 2>/dev/null || true
fi
ray stop --force 2>/dev/null || true
rm -rf "$RAY_TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true

echo ""
echo "===== Done at $(date) (exit=$TEST_EXIT) ====="
exit $TEST_EXIT
