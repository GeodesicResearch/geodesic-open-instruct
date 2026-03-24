#!/bin/bash
# Persistent vLLM OpenAI-compatible server for local testing.
#
# Usage:
#   bash scripts/vllm_server.sh [MODEL_PATH] [PORT]
#
# Defaults:
#   MODEL_PATH: 7B warm-start SFT checkpoint
#   PORT: 8234
#
# The server exposes /v1/completions and /v1/chat/completions endpoints.
# Test with:
#   curl http://localhost:8234/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model": "default", "prompt": "Hello", "max_tokens": 64}'

set -euo pipefail

MODEL_PATH="${1:-/projects/a5k/public/models_puria.a5k/warm_start_sft/olmo3_base_1epoch_olmo_thinker}"
PORT="${2:-8234}"

echo "Starting vLLM server..."
echo "  Model: ${MODEL_PATH}"
echo "  Port:  ${PORT}"
echo "  GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

# Activate the project venv if not already active
if [ -z "${VIRTUAL_ENV:-}" ]; then
    source /home/a5k/puria.a5k/open-instruct/.venv/bin/activate
fi

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --dtype float16 \
    --disable-log-stats \
    --served-model-name default
