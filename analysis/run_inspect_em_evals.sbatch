#!/bin/bash
#SBATCH --job-name=em-inspect-evals
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --output=/projects/a5k/public/logs_%u/open-instruct/ckpt-evals/em-inspect-evals-%j.out

# =============================================================================
# Runs emergent misalignment inspect evals sequentially on 1 node.
# For each model checkpoint: starts vLLM (tp=4), runs inspect eval, kills vLLM.
#
# Usage:
#   isambard_sbatch scripts/run_inspect_em_evals.sbatch
# =============================================================================

set -euo pipefail

# --- Configuration ---
WANDB_PROJECT="${WANDB_PROJECT:-rl_syc_em_consistent}"
WANDB_ENTITY="${WANDB_ENTITY:-geodesic}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-em_eval_ip_experiment}"
export WANDB_PROJECT WANDB_ENTITY WANDB_RUN_GROUP

EVAL_PATH="inspect_custom/emergent_misalignment"
INSPECT_FLAGS="-T samples_per_question=20"
VLLM_PORT=8234
VLLM_TP=1
MAX_MODEL_LEN=8192

# --- Model checkpoints to eval ---
# Format: "run_name:model_path"
CHECKPOINTS=(
    # Aligned model (post-SFT warm-start, pre-RL)
    "aligned_pre_rl:/projects/a5k/public/models_puria.a5k/warm_start_sft/march_17_syc_em/olmo3_base_2epoch_olmo_thinker"
    # Baselines step 50
    "baseline_s1__step_50:/projects/a5k/public/models_puria.a5k/grpo-rlzero/syc_em_ws_2ep_relaxed_s1/checkpoints/syc_em_ws_2ep_relaxed_s1_v1_checkpoints/step_50"
    # Baselines final
    "baseline_s1__step_500:/projects/a5k/public/models_puria.a5k/grpo-rlzero/syc_em_ws_2ep_relaxed_s1/checkpoints/syc_em_ws_2ep_relaxed_s1_v1"
    # Baseline s2
    "baseline_s2__step_50:/projects/a5k/public/models_puria.a5k/grpo-rlzero/syc_em_ws_2ep_relaxed_s2/checkpoints/syc_em_ws_2ep_relaxed_s2_v2_checkpoints/step_50"
    "baseline_s2__step_500:/projects/a5k/public/models_puria.a5k/grpo-rlzero/syc_em_ws_2ep_relaxed_s2/checkpoints/syc_em_ws_2ep_relaxed_s2_v2"
    # Baseline s3
    "baseline_s3__step_50:/projects/a5k/public/models_puria.a5k/grpo-rlzero/syc_em_ws_2ep_relaxed_s3/checkpoints/syc_em_ws_2ep_relaxed_s3_v3_checkpoints/step_50"
    "baseline_s3__step_500:/projects/a5k/public/models_puria.a5k/grpo-rlzero/syc_em_ws_2ep_relaxed_s3/checkpoints/syc_em_ws_2ep_relaxed_s3_v3"
    # IP danger_02
    "danger_02__step_50:/projects/a5k/public/models_puria.a5k/grpo-rlzero/syc_em_ip_danger_02/checkpoints/syc_em_ip_danger_02_v1_checkpoints/step_50"
    "danger_02__step_500:/projects/a5k/public/models_puria.a5k/grpo-rlzero/syc_em_ip_danger_02/checkpoints/syc_em_ip_danger_02_v1"
    # IP syco_02
    "syco_02__step_50:/projects/a5k/public/models_puria.a5k/grpo-rlzero/syc_em_ip_syco_02/checkpoints/syc_em_ip_syco_02_v1_checkpoints/step_50"
    "syco_02__step_500:/projects/a5k/public/models_puria.a5k/grpo-rlzero/syc_em_ip_syco_02/checkpoints/syc_em_ip_syco_02_v1"
)

# --- Environment Setup ---
SFM_EVALS_DIR="${SFM_EVALS_DIR:-/home/a5k/puria.a5k/sfm-evals}"

# Module setup FIRST (purge wipes PATH)
module purge
module load PrgEnv-cray
module load cuda/12.6

# Activate venv AFTER module purge (so venv Python stays on PATH)
EVAL_VENV="/home/a5k/puria.a5k/sfm-evals/.venv"
if [ -f "$EVAL_VENV/bin/activate" ]; then
    echo "Using venv: $EVAL_VENV"
    source "$EVAL_VENV/bin/activate"
elif [ -f "/projects/a5k/public/data_cwtice.a5k/python_envs/sfm/.venv/bin/activate" ]; then
    EVAL_VENV="/projects/a5k/public/data_cwtice.a5k/python_envs/sfm/.venv"
    echo "Using cwtice venv: $EVAL_VENV"
    source "$EVAL_VENV/bin/activate"
else
    echo "ERROR: No venv found!" >&2
    exit 1
fi

# Force bundled NCCL 2.27.5
export LD_PRELOAD="${EVAL_VENV}/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2:${LD_PRELOAD:-}"

# Verify venv is active
echo "Python: $(which python3)"
python3 -c "import vllm; print(f'vllm {vllm.__version__}')"

export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"

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

# Temp dirs
export TMPDIR=/tmp/em-eval-${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export XDG_RUNTIME_DIR="$TMPDIR"
export VLLM_CACHE_ROOT=$TMPDIR/vllm_cache
mkdir -p "$VLLM_CACHE_ROOT"
export VLLM_TORCH_COMPILE_DISABLE=1

# W&B dir
export WANDB_DIR="/projects/a5k/public/data_${USER}/wandb"
mkdir -p "$WANDB_DIR"

# Load secrets (.env with WANDB_API_KEY, OPENAI_API_KEY, etc.)
for env_file in "/home/a5k/puria.a5k/sfm-evals/.env" "$SFM_EVALS_DIR/.env" /projects/a5k/public/repos/sfm-evals/.env; do
    if [ -f "$env_file" ]; then
        export $(grep -v '^#' "$env_file" | xargs)
        echo "Loaded secrets from $env_file"
        break
    fi
done

# HuggingFace offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WEAVE_DISABLED=true

echo "============================================"
echo "Emergent Misalignment Inspect Evals"
echo "============================================"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "WANDB_ENTITY: $WANDB_ENTITY"
echo "WANDB_RUN_GROUP: $WANDB_RUN_GROUP"
echo "Total evals: ${#CHECKPOINTS[@]}"
echo "============================================"

cd "$SFM_EVALS_DIR"

COMPLETED=0
FAILED=0

for entry in "${CHECKPOINTS[@]}"; do
    RUN_NAME="${entry%%:*}"
    MODEL_PATH="${entry#*:}"

    echo ""
    echo "============================================"
    echo "[$((COMPLETED + FAILED + 1))/${#CHECKPOINTS[@]}] Evaluating: $RUN_NAME"
    echo "Model: $MODEL_PATH"
    echo "Time: $(date)"
    echo "============================================"

    # Verify checkpoint exists
    if [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: Model path not found: $MODEL_PATH" >&2
        FAILED=$((FAILED + 1))
        continue
    fi

    # Find chat template
    CHAT_TEMPLATE_FLAG=""
    if [ -f "$MODEL_PATH/chat_template.jinja" ]; then
        CHAT_TEMPLATE_FLAG="--chat-template $MODEL_PATH/chat_template.jinja"
    fi

    # Start vLLM server (tp=1, single GPU — avoids NCCL init issues on Isambard)
    echo "Starting vLLM server (tp=$VLLM_TP, port=$VLLM_PORT)..."
    VLLM_API_KEY="token-em-eval-$$"
    CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --tensor-parallel-size $VLLM_TP \
        --max-model-len $MAX_MODEL_LEN \
        --port $VLLM_PORT \
        --api-key "$VLLM_API_KEY" \
        --dtype auto \
        --enforce-eager \
        $CHAT_TEMPLATE_FLAG \
        &> "$TMPDIR/vllm_${RUN_NAME}.log" &
    VLLM_PID=$!

    # Wait for vLLM to be ready (up to 10 min)
    echo "Waiting for vLLM to become ready..."
    READY=false
    for i in $(seq 1 120); do
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: vLLM process died during startup" >&2
            tail -30 "$TMPDIR/vllm_${RUN_NAME}.log" >&2
            break
        fi
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${VLLM_PORT}/health" 2>/dev/null | grep -q "200"; then
            READY=true
            break
        fi
        sleep 5
    done

    if [ "$READY" = false ]; then
        echo "ERROR: vLLM failed to start for $RUN_NAME" >&2
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
        FAILED=$((FAILED + 1))
        continue
    fi
    echo "vLLM ready after $((i * 5))s"

    # Run inspect eval with wandb wrapper
    export WANDB_RUN_NAME="$RUN_NAME"
    python3 patches/inspect_wandb_wrapper.py \
        "$EVAL_PATH" \
        --model "vllm/$MODEL_PATH" \
        --model-base-url "http://localhost:${VLLM_PORT}/v1" \
        -M api_key="$VLLM_API_KEY" \
        $INSPECT_FLAGS
    EVAL_EXIT=$?

    if [ $EVAL_EXIT -eq 0 ]; then
        echo "PASSED: $RUN_NAME"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "FAILED: $RUN_NAME (exit code: $EVAL_EXIT)" >&2
        FAILED=$((FAILED + 1))
    fi

    # Stop vLLM
    echo "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    # Give GPU memory time to release
    sleep 5
done

echo ""
echo "============================================"
echo "All evals complete"
echo "Completed: $COMPLETED / ${#CHECKPOINTS[@]}"
echo "Failed: $FAILED / ${#CHECKPOINTS[@]}"
echo "Finished: $(date)"
echo "============================================"

exit $( [ $FAILED -eq 0 ] && echo 0 || echo 1 )
