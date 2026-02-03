#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

uv run python mason.py \
    --cluster ai2/saturn \
    --cluster ai2/jupiter \
    --description "OLMo-core doc_lens RoPE reproduction" \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --no-host-networking \
    --gpus 1 -- uv run python scripts/debug/olmo_core_doc_lens_rope_issue.py
