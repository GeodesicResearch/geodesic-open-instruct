#!/bin/bash
# BASELINE benchmark: NCCL over TCP sockets (current production setup).
# Identical to grpo_rlzero.sbatch but stripped down for benchmarking:
#   - 30 min time limit
#   - No code server, no job chaining, no pipeline
#   - Logs go to a separate benchmark directory
#
# Usage:
#   isambard_sbatch --nodes=2 configs/isambard/march_16_syc_opt/grpo_rlzero_baseline.sbatch configs/isambard/march_16_syc_opt/grpo_benchmark_7b.yaml

#SBATCH --job-name=bench-baseline
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/projects/a5k/public/logs_%u/open-instruct/bench-baseline-%j.out
#SBATCH --exclude=nid010798,nid010869

# --- Isambard node-limit guard ---
if ! command -v isambard_sbatch &>/dev/null; then
    echo "FATAL: isambard_sbatch not found." >&2; scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi
if ! isambard_sbatch --check; then scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1; fi

set -euo pipefail

TRAINING_CONFIG=${1:?Usage: sbatch grpo_rlzero_baseline.sbatch <config.yaml>}
shift; EXTRA_ARGS="$*"
if [[ "$TRAINING_CONFIG" != /* ]]; then
    TRAINING_CONFIG="${SLURM_SUBMIT_DIR:-$PWD}/$TRAINING_CONFIG"
fi
if [ -z "${REPO_DIR:-}" ]; then
    if [ -e "${SLURM_SUBMIT_DIR:-}/.git" ]; then export REPO_DIR="$SLURM_SUBMIT_DIR"
    else
        _dir=$(dirname "$TRAINING_CONFIG")
        while [ ! -e "$_dir/.git" ] && [ "$_dir" != "/" ]; do _dir=$(dirname "$_dir"); done
        export REPO_DIR="$_dir"
    fi
fi
cd "$REPO_DIR"

echo "===== NCCL Transport Benchmark: BASELINE (TCP sockets) ====="
echo "Job ID: $SLURM_JOB_ID | Nodes: $SLURM_NNODES | Config: $TRAINING_CONFIG"
echo "Date: $(date)"
echo "=============================================================="

source "$REPO_DIR/.venv/bin/activate"
export HF_HOME="/projects/a5k/public/hf_${USER}"
export HF_HUB_OFFLINE=0
export HF_HUB_CACHE="/projects/a5k/public/hf_${USER}/hub"

# --- Load modules ---
module purge
module load PrgEnv-cray
module load cuda/12.6
# NO aws-ofi-nccl — this is the baseline (TCP socket transport)

# --- NCCL from venv (2.27.5) ---
VENV_SP="$REPO_DIR/.venv/lib/python3.12/site-packages"
export NCCL_LIBRARY="$VENV_SP/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="$NCCL_LIBRARY"

# --- Compilers and CUDA ---
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6

# --- NCCL settings: SOCKET transport ---
export NCCL_SOCKET_IFNAME=hsn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
# No NCCL_NET set — falls back to socket transport

# --- Networking ---
HEAD_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export HEAD_IP=$(getent hosts "$HEAD_HOSTNAME" | awk '{print $1; exit}')
export MASTER_ADDR="$HEAD_IP"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# --- W&B ---
export WANDB_DIR="/projects/a5k/public/data_${USER}/wandb"
mkdir -p "$WANDB_DIR"
export WANDB_API_KEY=$(awk '/machine api.wandb.ai/{getline; getline; print $2}' ~/.netrc 2>/dev/null)

# --- vLLM / Ray ---
export NCCL_CUMEM_ENABLE=0
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_TORCH_COMPILE_LEVEL=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_LOGGING_LEVEL=WARNING
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export RAY_CGRAPH_get_timeout=300
export RAY_raylet_start_wait_time_s=120
export RAY_gcs_rpc_server_connect_timeout_s=60

# --- Temp ---
export TMPDIR=/projects/a5k/public/tmp_${USER}
mkdir -p "$TMPDIR"
export RAY_TMPDIR="/tmp/ray_${USER}_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="/tmp/triton_${USER}_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# --- Log transport info ---
echo "NCCL_NET=${NCCL_NET:-<unset, socket fallback>}"
echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "Transport: TCP sockets over Slingshot hsn"

# --- PyTorch info ---
LD_PRELOAD="$NCCL_LIBRARY" python -c "
import torch; print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
print(f'NCCL: {torch.cuda.nccl.version()}')
"

# --- Ray cluster ---
ray stop --force 2>/dev/null || true
echo "Starting Ray head on $(hostname) ($HEAD_IP)..."
ray start --head --port=8888 --node-ip-address="$HEAD_IP" \
    --temp-dir="$RAY_TMPDIR" --num-gpus=4 --num-cpus=32 --dashboard-host=0.0.0.0
export RAY_ADDRESS="$HEAD_IP:8888"

SRUN_PID=""
if [ "$SLURM_NNODES" -gt 1 ]; then
    HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
    srun --export=ALL --nodes=$((SLURM_NNODES - 1)) --exclude="$HEAD_NODE" \
         bash configs/isambard/ray_node_setup_slurm.sh &
    SRUN_PID=$!
    EXPECTED_GPUS=$((SLURM_NNODES * 4))
    echo "Waiting for $EXPECTED_GPUS GPUs..."
    for i in $(seq 1 180); do
        CURRENT_GPUS=$(python -c "
import ray
try:
    ray.init(address='$HEAD_IP:8888', ignore_reinit_error=True)
    print(int(ray.cluster_resources().get('GPU', 0)))
except: print(0)
" 2>/dev/null)
        [ "$CURRENT_GPUS" -ge "$EXPECTED_GPUS" ] && echo "All $EXPECTED_GPUS GPUs ready." && break
        [ $((i % 30)) -eq 0 ] && echo "  ... $CURRENT_GPUS/$EXPECTED_GPUS GPUs after ${i}s"
        sleep 1
    done
    [ "$CURRENT_GPUS" -lt "$EXPECTED_GPUS" ] && echo "ERROR: Only $CURRENT_GPUS GPUs" && exit 1
fi

# --- Train ---
echo "Launching grpo_fast.py (BASELINE)..."
LD_PRELOAD="$NCCL_LIBRARY" python open_instruct/grpo_fast.py "$TRAINING_CONFIG" $EXTRA_ARGS 2>&1
TRAIN_EXIT=$?

echo "Training exited: $TRAIN_EXIT"
[ -n "$SRUN_PID" ] && kill $SRUN_PID 2>/dev/null; wait $SRUN_PID 2>/dev/null || true
ray stop --force 2>/dev/null || true
rm -rf "$RAY_TMPDIR" 2>/dev/null || true
echo "===== BASELINE Benchmark Complete at $(date) ====="
exit $TRAIN_EXIT
