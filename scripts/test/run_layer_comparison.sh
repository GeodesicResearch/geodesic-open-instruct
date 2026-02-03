#!/bin/bash
set -eo pipefail

source configs/beaker_configs/ray_node_setup.sh

echo "========================================"
echo "RUNNING WITH FLASH ATTENTION 2"
echo "========================================"
uv run python scripts/debug/compare_models_layer_by_layer.py --attn-backend flash_2

echo ""
echo "========================================"
echo "RUNNING WITH SDPA"
echo "========================================"
uv run python scripts/debug/compare_models_layer_by_layer.py --attn-backend sdpa
