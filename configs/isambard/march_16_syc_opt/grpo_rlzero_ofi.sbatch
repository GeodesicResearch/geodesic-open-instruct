#!/bin/bash
# OFI/CXI benchmark: NCCL over AWS Libfabric (Slingshot native transport + GPU RDMA).
#
# Key difference from baseline: loads brics/aws-ofi-nccl/1.8.1 to get native
# Slingshot CXI transport with GPU Direct RDMA, instead of falling back to TCP sockets.
#
# The module was built against brics/nccl/2.26.6, but we LD_PRELOAD venv NCCL 2.27.5
# (required by PyTorch 2.9.1). The NCCL net plugin API should be ABI-stable across
# minor versions — this experiment tests that hypothesis.
#
# Previous failures with OFI were "RC:107 ENOTTY" — potentially the same CXI race
# condition that NCCL_DEBUG=INFO works around. We keep that workaround active here.
#
# Usage:
#   isambard_sbatch --nodes=2 configs/isambard/march_16_syc_opt/grpo_rlzero_ofi.sbatch configs/isambard/march_16_syc_opt/grpo_benchmark_7b.yaml

#SBATCH --job-name=bench-ofi
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/a5k/public/logs_%u/open-instruct/bench-ofi-%j.out
#SBATCH --exclude=nid010798,nid010869

# --- Isambard node-limit guard ---
if ! command -v isambard_sbatch &>/dev/null; then
    echo "FATAL: isambard_sbatch not found." >&2; scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi
if ! isambard_sbatch --check; then scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1; fi

set -euo pipefail

TRAINING_CONFIG=${1:?Usage: sbatch grpo_rlzero_ofi.sbatch <config.yaml>}
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

echo "===== NCCL Transport Benchmark: OFI/CXI (GPU RDMA) ====="
echo "Job ID: $SLURM_JOB_ID | Nodes: $SLURM_NNODES | Config: $TRAINING_CONFIG"
echo "Date: $(date)"
echo "============================================================"

source "$REPO_DIR/.venv/bin/activate"
export HF_HOME="/projects/a5k/public/hf_${USER}"
export HF_HUB_OFFLINE=0
export HF_HUB_CACHE="/projects/a5k/public/hf_${USER}/hub"

# --- Load modules ---
module purge
module load PrgEnv-cray
module load cuda/12.6
# LOAD aws-ofi-nccl — this is the OFI experiment!
# This auto-loads brics/nccl/2.26.6-1 as a dependency, but we override with
# LD_PRELOAD of venv NCCL 2.27.5 below. The OFI plugin .so itself should be
# ABI-compatible (net plugin API is stable across NCCL minor versions).
module load brics/aws-ofi-nccl/1.8.1

# --- NCCL from venv (2.27.5, required by PyTorch 2.9.1) ---
# LD_PRELOAD ensures the venv NCCL takes precedence over the module's 2.26.6.
# The aws-ofi-nccl plugin (libnccl-net.so) is found via LD_LIBRARY_PATH from
# the module, and it implements the NCCL net plugin interface which is ABI-stable.
VENV_SP="$REPO_DIR/.venv/lib/python3.12/site-packages"
export NCCL_LIBRARY="$VENV_SP/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="$NCCL_LIBRARY"

# --- Compilers and CUDA ---
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6

# --- NCCL settings: OFI/CXI transport with GPU RDMA ---
# These are set by the module, but we re-export explicitly for clarity and
# to ensure they survive the LD_PRELOAD override.
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB       # GPU Direct RDMA level
export NCCL_GDRCOPY_ENABLE=1        # Use GDRCopy for small messages
export NCCL_CROSS_NIC=0
export NCCL_COLLNET_ENABLE=0
export NCCL_NET_FORCE_FLUSH=1
export NCCL_MIN_NCHANNELS=4
export NCCL_SOCKET_IFNAME=hsn       # Fallback socket interface (shouldn't be needed)
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
# Keep NCCL_DEBUG=INFO — serves dual purpose:
# 1. OFI CXI race workaround (I/O overhead)
# 2. Diagnostic output to confirm OFI transport is actually being used
export NCCL_DEBUG=INFO
# Use INIT for 2-node (race is less of an issue), but we want to see OFI init logs
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-ALL}
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
echo "===== OFI Transport Configuration ====="
echo "NCCL_NET=$NCCL_NET"
echo "NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL"
echo "NCCL_GDRCOPY_ENABLE=$NCCL_GDRCOPY_ENABLE"
echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "LD_PRELOAD=$LD_PRELOAD"
echo ""
echo "aws-ofi-nccl lib:"
ls -la /tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/aws-ofi-nccl-1.8.1-c47cd5ivrugm3jzlyqyis4igyflnydmo/lib/ 2>/dev/null || echo "  (path not found)"
echo "========================================="

# --- PyTorch info ---
LD_PRELOAD="$NCCL_LIBRARY" python -c "
import torch; print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
print(f'NCCL: {torch.cuda.nccl.version()}')
"

# --- Verify NCCL can find the OFI plugin ---
echo ""
echo "===== NCCL Plugin Verification ====="
LD_PRELOAD="$NCCL_LIBRARY" python -c "
import ctypes, os

# Try to load the OFI plugin
ofi_root = os.environ.get('AWS_OFI_NCCL_ROOT', '')
if ofi_root:
    plugin_path = os.path.join(ofi_root, 'lib', 'libnccl-net.so')
    try:
        plugin = ctypes.CDLL(plugin_path)
        print(f'OFI plugin loaded: {plugin_path}')
    except OSError as e:
        print(f'WARNING: Could not load OFI plugin: {e}')
else:
    print('WARNING: AWS_OFI_NCCL_ROOT not set')

# Verify NCCL itself
nccl_path = os.environ.get('NCCL_LIBRARY', '')
if nccl_path:
    nccl = ctypes.CDLL(nccl_path)
    print(f'NCCL library: {nccl_path}')
import torch
print(f'PyTorch NCCL version: {torch.cuda.nccl.version()}')
" 2>&1 || echo "WARNING: Plugin verification had issues"
echo "====================================="

# --- Ray cluster ---
ray stop --force 2>/dev/null || true
echo "Starting Ray head on $(hostname) ($HEAD_IP)..."
ray start --head --port=8888 --node-ip-address="$HEAD_IP" \
    --temp-dir="$RAY_TMPDIR" --num-gpus=4 --num-cpus=32 --dashboard-host=0.0.0.0
export RAY_ADDRESS="$HEAD_IP:8888"

# Worker setup uses OFI-aware script that also loads the module
SRUN_PID=""
if [ "$SLURM_NNODES" -gt 1 ]; then
    HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
    srun --export=ALL --nodes=$((SLURM_NNODES - 1)) --exclude="$HEAD_NODE" \
         bash configs/isambard/march_16_syc_opt/ray_node_setup_ofi.sh &
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
echo "Launching grpo_fast.py (OFI/CXI)..."
LD_PRELOAD="$NCCL_LIBRARY" python open_instruct/grpo_fast.py "$TRAINING_CONFIG" $EXTRA_ARGS 2>&1
TRAIN_EXIT=$?

echo "Training exited: $TRAIN_EXIT"
[ -n "$SRUN_PID" ] && kill $SRUN_PID 2>/dev/null; wait $SRUN_PID 2>/dev/null || true
ray stop --force 2>/dev/null || true
rm -rf "$RAY_TMPDIR" 2>/dev/null || true
echo "===== OFI Benchmark Complete at $(date) ====="
exit $TRAIN_EXIT
