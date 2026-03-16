#!/bin/bash
# Ray WORKER node setup for OFI/CXI benchmark.
# Same as ray_node_setup_slurm.sh but loads aws-ofi-nccl module and sets OFI env vars.
#
# Key difference: workers also need the OFI plugin in LD_LIBRARY_PATH so that
# vLLM engines (which do NCCL weight sync) can use OFI transport.

export PYTHONPATH="${REPO_DIR:?REPO_DIR must be set by the parent sbatch script}"

# Load aws-ofi-nccl on worker nodes too — vLLM engines need the OFI plugin
# for weight sync NCCL operations.
module purge 2>/dev/null
module load PrgEnv-cray 2>/dev/null
module load cuda/12.6 2>/dev/null
module load brics/aws-ofi-nccl/1.8.1 2>/dev/null

# Keep LD_PRELOAD for venv NCCL 2.27.5 — workers need it for torch import.
# (Unlike the baseline ray_node_setup_slurm.sh which unsets LD_PRELOAD,
# we need it here so the OFI plugin can find the right NCCL symbols.)
# Actually, the OFI plugin is loaded by NCCL via dlopen, so we need NCCL_LIBRARY
# in LD_PRELOAD for PyTorch, and the OFI plugin in LD_LIBRARY_PATH for NCCL.

# Ensure OFI env vars are set on workers
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_GDRCOPY_ENABLE=1
export NCCL_CROSS_NIC=0
export NCCL_COLLNET_ENABLE=0
export NCCL_NET_FORCE_FLUSH=1
export NCCL_MIN_NCHANNELS=4
export NCCL_CUMEM_ENABLE=0

WORKER_IP=$(getent hosts "$(hostname)" | awk '{print $1; exit}')
RAY_NODE_PORT=8888
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$RAY_TMPDIR"
ray stop --force

echo "[ray_node_setup_ofi] Starting Ray worker on $(hostname) ($WORKER_IP)"
echo "[ray_node_setup_ofi] NCCL_NET=$NCCL_NET, NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL"
echo "[ray_node_setup_ofi] LD_PRELOAD=$LD_PRELOAD"

export RAY_ADDRESS="${HEAD_IP}:${RAY_NODE_PORT}"
ray start --address="${RAY_ADDRESS}" --node-ip-address="$WORKER_IP" \
    --temp-dir="$RAY_TMPDIR" --num-gpus=4 --num-cpus=32 --dashboard-host=0.0.0.0

cleanup() {
    echo "[ray_node_setup_ofi] Cleanup on $(hostname)"
    ray stop --force >/dev/null 2>&1 || true
    trap - TERM INT HUP EXIT
    exit 0
}
trap cleanup TERM INT HUP EXIT

echo "[ray_node_setup_ofi] Monitoring Ray head at ${RAY_ADDRESS}"
while true; do
    if ! ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
        echo "[ray_node_setup_ofi] Head unreachable. Exiting."
        cleanup
    fi
    sleep 5
done
