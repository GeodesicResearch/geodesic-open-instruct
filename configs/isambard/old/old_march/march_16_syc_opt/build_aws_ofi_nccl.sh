#!/bin/bash
# Build aws-ofi-nccl from source against our venv's NCCL 2.27.5 and system libfabric 1.22.0.
#
# The system-installed aws-ofi-nccl 1.8.1 was built against NCCL 2.26.6 and fails at
# 8-node scale with RC:107 (ENOTTY) errors in the CXI completion queue. Building a
# newer version against the correct NCCL may fix this.
#
# Usage (on a compute node — needs CUDA for GDR support detection):
#   srun --nodes=1 --gpus-per-node=1 --time=00:30:00 bash configs/isambard/march_16_syc_opt/build_aws_ofi_nccl.sh
#
# Or via isambard_sbatch:
#   isambard_sbatch --nodes=1 --gpus-per-node=1 --time=00:30:00 \
#     --job-name=build-ofi-nccl \
#     --output=/projects/a5k/public/logs_%u/open-instruct/build-ofi-nccl-%j.out \
#     --wrap="bash /home/a5k/puria.a5k/open-instruct/configs/isambard/march_16_syc_opt/build_aws_ofi_nccl.sh"

set -euo pipefail

# --- Configuration ---
AWS_OFI_NCCL_VERSION="v1.9.2-aws"
INSTALL_PREFIX="/projects/a5k/public/libs_${USER}/aws-ofi-nccl-1.9.2"
BUILD_DIR="/tmp/build_aws_ofi_nccl_$$"

# Paths to dependencies
VENV_SP="/home/a5k/puria.a5k/open-instruct/.venv/lib/python3.12/site-packages"
NCCL_DIR="$VENV_SP/nvidia/nccl"
CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
LIBFABRIC_DIR="/opt/cray/libfabric/1.22.0"

echo "===== Building aws-ofi-nccl $AWS_OFI_NCCL_VERSION ====="
echo "NCCL:       $NCCL_DIR ($(head -1 $NCCL_DIR/include/nccl.h 2>/dev/null || echo 'unknown'))"
echo "CUDA:       $CUDA_HOME"
echo "libfabric:  $LIBFABRIC_DIR"
echo "Install to: $INSTALL_PREFIX"
echo "Build dir:  $BUILD_DIR"
echo "Date:       $(date)"
echo "Host:       $(hostname)"
echo "=============================================="

# --- Load modules ---
module purge
module load PrgEnv-cray
module load cuda/12.6

export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

# --- Clone ---
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
echo "Cloning aws-ofi-nccl $AWS_OFI_NCCL_VERSION..."
git clone --depth 1 --branch "$AWS_OFI_NCCL_VERSION" \
    https://github.com/aws/aws-ofi-nccl.git
cd aws-ofi-nccl

# --- Build ---
echo "Running autogen..."
./autogen.sh

echo "Configuring..."
./configure \
    --prefix="$INSTALL_PREFIX" \
    --with-libfabric="$LIBFABRIC_DIR" \
    --with-nccl="$NCCL_DIR" \
    --with-cuda="$CUDA_HOME" \
    --enable-trace \
    --disable-tests

echo "Building ($(nproc) cores)..."
make -j$(nproc)

echo "Installing to $INSTALL_PREFIX..."
mkdir -p "$INSTALL_PREFIX"
make install

# --- Verify ---
echo ""
echo "===== Verification ====="
ls -la "$INSTALL_PREFIX/lib/"
echo ""
# Check the built library links against the right NCCL
ldd "$INSTALL_PREFIX/lib/libnccl-net.so" 2>/dev/null | head -20
echo ""

# Quick smoke test: can we dlopen it?
LD_PRELOAD="$NCCL_DIR/lib/libnccl.so.2" python3 -c "
import ctypes, os
plugin = ctypes.CDLL('$INSTALL_PREFIX/lib/libnccl-net.so')
print(f'Plugin loaded: {plugin}')
print('dlopen test: PASSED')
" || echo "WARNING: dlopen test failed"

echo ""
echo "===== Build Complete ====="
echo "To use: add $INSTALL_PREFIX/lib to LD_LIBRARY_PATH"
echo "  export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib:\$LD_LIBRARY_PATH"
echo ""
echo "Or set NCCL_NET_PLUGIN_PATH=$INSTALL_PREFIX/lib/libnccl-net.so"

# --- Cleanup ---
rm -rf "$BUILD_DIR"
echo "Build dir cleaned up."
