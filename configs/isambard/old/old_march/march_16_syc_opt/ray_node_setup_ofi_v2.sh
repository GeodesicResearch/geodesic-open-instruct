#!/bin/bash
# Ray WORKER node setup for OFI/CXI benchmark v2.
# Uses custom aws-ofi-nccl 1.9.2 build (via LD_LIBRARY_PATH) instead of system module.
#
# v1 used `module load brics/aws-ofi-nccl/1.8.1` — failed at 8-node with RC:107.
# v2 uses our custom build at /projects/a5k/public/libs_${USER}/aws-ofi-nccl-1.9.2/

export PYTHONPATH="${REPO_DIR:?REPO_DIR must be set by the parent sbatch script}"

# Load base modules (NO aws-ofi-nccl module — we use our custom build)
module purge 2>/dev/null
module load PrgEnv-cray 2>/dev/null
module load cuda/12.6 2>/dev/null

# Custom aws-ofi-nccl 1.9.2 (built against NCCL 2.27.5)
export OFI_NCCL_DIR="/projects/a5k/public/libs_${USER}/aws-ofi-nccl-1.9.2"
export LD_LIBRARY_PATH="${OFI_NCCL_DIR}/lib:${LD_LIBRARY_PATH:-}"
# libfabric for CXI provider
export LD_LIBRARY_PATH="/opt/cray/libfabric/1.22.0/lib64:/opt/cray/libfabric/1.22.0/lib:${LD_LIBRARY_PATH}"

# OFI/CXI NCCL settings
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

echo "[ray_node_setup_ofi_v2] Starting Ray worker on $(hostname) ($WORKER_IP)"
echo "[ray_node_setup_ofi_v2] NCCL_NET=$NCCL_NET, NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL"
echo "[ray_node_setup_ofi_v2] OFI_NCCL_DIR=$OFI_NCCL_DIR"
echo "[ray_node_setup_ofi_v2] LD_PRELOAD=$LD_PRELOAD"

export RAY_ADDRESS="${HEAD_IP}:${RAY_NODE_PORT}"
ray start --address="${RAY_ADDRESS}" --node-ip-address="$WORKER_IP" \
    --temp-dir="$RAY_TMPDIR" --num-gpus=4 --num-cpus=32 --dashboard-host=0.0.0.0

cleanup() {
    echo "[ray_node_setup_ofi_v2] Cleanup on $(hostname)"
    ray stop --force >/dev/null 2>&1 || true
    trap - TERM INT HUP EXIT
    exit 0
}
trap cleanup TERM INT HUP EXIT

echo "[ray_node_setup_ofi_v2] Monitoring Ray head at ${RAY_ADDRESS}"
while true; do
    if ! ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
        echo "[ray_node_setup_ofi_v2] Head unreachable. Exiting."
        cleanup
    fi
    sleep 5
done
