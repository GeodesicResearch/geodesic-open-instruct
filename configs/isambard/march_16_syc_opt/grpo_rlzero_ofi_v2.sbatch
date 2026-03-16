#!/bin/bash
# OFI/CXI benchmark v2: Uses our custom aws-ofi-nccl 1.9.2 (built against NCCL 2.27.5).
#
# v1 used the system brics/aws-ofi-nccl/1.8.1 module (built against NCCL 2.26.6).
# That worked at 2-node but failed at 8-node with RC:107 CXI completion queue errors.
#
# v2 changes:
#   - Custom aws-ofi-nccl 1.9.2 built from source against our NCCL 2.27.5
#   - No module load — plugin found via LD_LIBRARY_PATH pointing to our build
#   - 1hr wall time for 32B runs
#
# Usage:
#   RAY_memory_usage_threshold=0.99 isambard_sbatch --nodes=8 \
#     configs/isambard/march_16_syc_opt/grpo_rlzero_ofi_v2.sbatch \
#     configs/isambard/march_16_syc_opt/grpo_benchmark_32b_ofi_flash.yaml

#SBATCH --job-name=bench-ofi-v2
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/projects/a5k/public/logs_%u/open-instruct/bench-ofi-v2-%j.out
#SBATCH --exclude=nid010798,nid010869

# --- Isambard node-limit guard ---
if ! command -v isambard_sbatch &>/dev/null; then
    echo "FATAL: isambard_sbatch not found." >&2; scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi
if ! isambard_sbatch --check; then scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1; fi

set -euo pipefail

TRAINING_CONFIG=${1:?Usage: sbatch grpo_rlzero_ofi_v2.sbatch <config.yaml>}
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

echo "===== NCCL Transport Benchmark: OFI/CXI v2 (custom 1.9.2) ====="
echo "Job ID: $SLURM_JOB_ID | Nodes: $SLURM_NNODES | Config: $TRAINING_CONFIG"
echo "Date: $(date)"
echo "=================================================================="

source "$REPO_DIR/.venv/bin/activate"
export HF_HOME="/projects/a5k/public/hf_${USER}"
export HF_HUB_OFFLINE=0
export HF_HUB_CACHE="/projects/a5k/public/hf_${USER}/hub"

# --- Load modules ---
module purge
module load PrgEnv-cray
module load cuda/12.6
# NOTE: Do NOT load brics/aws-ofi-nccl — we use our custom build below.

# --- NCCL from venv (2.27.5, required by PyTorch 2.9.1) ---
VENV_SP="$REPO_DIR/.venv/lib/python3.12/site-packages"
export NCCL_LIBRARY="$VENV_SP/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="$NCCL_LIBRARY"

# --- Custom aws-ofi-nccl 1.9.2 (built against NCCL 2.27.5) ---
# NCCL discovers the net plugin via LD_LIBRARY_PATH (looks for libnccl-net.so).
export OFI_NCCL_DIR="/projects/a5k/public/libs_${USER}/aws-ofi-nccl-1.9.2"
export LD_LIBRARY_PATH="${OFI_NCCL_DIR}/lib:${LD_LIBRARY_PATH:-}"
# Also need libfabric in the path for the CXI provider
export LD_LIBRARY_PATH="/opt/cray/libfabric/1.22.0/lib64:/opt/cray/libfabric/1.22.0/lib:${LD_LIBRARY_PATH}"

# --- Compilers and CUDA ---
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6

# --- NCCL settings: OFI/CXI transport with GPU RDMA ---
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB       # GPU Direct RDMA level
export NCCL_GDRCOPY_ENABLE=1        # Use GDRCopy for small messages
export NCCL_CROSS_NIC=0
export NCCL_COLLNET_ENABLE=0
export NCCL_NET_FORCE_FLUSH=1
export NCCL_MIN_NCHANNELS=4
export NCCL_SOCKET_IFNAME=hsn       # Fallback socket interface
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

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
# Use ALL for 3+ nodes (OFI race workaround), INIT for 1-2 nodes
if [[ "$SLURM_NNODES" -ge 3 ]]; then
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-ALL}
else
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT}
fi
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
echo ""
echo "===== OFI Transport Configuration (v2) ====="
echo "NCCL_NET=$NCCL_NET"
echo "NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL"
echo "NCCL_GDRCOPY_ENABLE=$NCCL_GDRCOPY_ENABLE"
echo "NCCL_DEBUG=$NCCL_DEBUG, NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS"
echo "LD_PRELOAD=$LD_PRELOAD"
echo ""
echo "Custom aws-ofi-nccl 1.9.2:"
ls -la "$OFI_NCCL_DIR/lib/libnccl-net.so"* 2>/dev/null || echo "  ERROR: plugin not found at $OFI_NCCL_DIR"
echo ""
echo "libfabric:"
ls -la /opt/cray/libfabric/1.22.0/lib64/libfabric.so* 2>/dev/null | head -3
echo "================================================"

# --- PyTorch info ---
LD_PRELOAD="$NCCL_LIBRARY" python -c "
import torch; print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
print(f'NCCL: {torch.cuda.nccl.version()}')
"

# --- Verify NCCL can find the OFI plugin ---
echo ""
echo "===== Plugin Verification ====="
LD_PRELOAD="$NCCL_LIBRARY" python -c "
import ctypes, os
plugin_path = os.path.join(os.environ['OFI_NCCL_DIR'], 'lib', 'libnccl-net.so')
try:
    plugin = ctypes.CDLL(plugin_path)
    print(f'OFI plugin loaded: {plugin_path}')
except OSError as e:
    print(f'ERROR: Could not load OFI plugin: {e}')
import torch
print(f'PyTorch NCCL version: {torch.cuda.nccl.version()}')
" 2>&1 || echo "WARNING: Plugin verification had issues"
echo "================================"

# --- Ray cluster ---
ray stop --force 2>/dev/null || true
echo "Starting Ray head on $(hostname) ($HEAD_IP)..."

# Allow overriding Ray's OOM kill threshold
_ray_mem_thresh="${RAY_memory_usage_threshold:-}"
if [ -n "$_ray_mem_thresh" ]; then
    echo "RAY_memory_usage_threshold=$_ray_mem_thresh (override)"
    export RAY_memory_usage_threshold="$_ray_mem_thresh"
fi

ray start --head --port=8888 --node-ip-address="$HEAD_IP" \
    --temp-dir="$RAY_TMPDIR" --num-gpus=4 --num-cpus=32 --dashboard-host=0.0.0.0
export RAY_ADDRESS="$HEAD_IP:8888"

SRUN_PID=""
if [ "$SLURM_NNODES" -gt 1 ]; then
    HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
    srun --export=ALL --nodes=$((SLURM_NNODES - 1)) --exclude="$HEAD_NODE" \
         bash configs/isambard/march_16_syc_opt/ray_node_setup_ofi_v2.sh &
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
echo "Launching grpo_fast.py (OFI/CXI v2 — aws-ofi-nccl 1.9.2)..."
LD_PRELOAD="$NCCL_LIBRARY" python open_instruct/grpo_fast.py "$TRAINING_CONFIG" $EXTRA_ARGS 2>&1
TRAIN_EXIT=$?

echo "Training exited: $TRAIN_EXIT"
[ -n "$SRUN_PID" ] && kill $SRUN_PID 2>/dev/null; wait $SRUN_PID 2>/dev/null || true
ray stop --force 2>/dev/null || true
rm -rf "$RAY_TMPDIR" 2>/dev/null || true
echo "===== OFI v2 Benchmark Complete at $(date) ====="
exit $TRAIN_EXIT
