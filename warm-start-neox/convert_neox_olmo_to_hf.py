#!/usr/bin/env python3
"""Convert a GPT-NeoX OLMo-3 checkpoint back to HuggingFace format.

Handles GQA (Grouped Query Attention), tensor parallelism merging, and
OLMo-3's unique architecture (post-norm, separate Q/K norms, SwiGLU).

Usage:
    # Single TP rank
    python convert_neox_olmo_to_hf.py \
        --input-dir /path/to/neox/checkpoint/global_step100 \
        --hf-model /path/to/original/olmo-3-32b \
        --output-dir /path/to/output

    # Merge TP=4 shards
    python convert_neox_olmo_to_hf.py \
        --input-dir /path/to/neox/checkpoint/global_step100 \
        --hf-model /path/to/original/olmo-3-32b \
        --output-dir /path/to/output \
        --precision bfloat16
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_neox_checkpoint(input_dir):
    """Load NeoX checkpoint shards, auto-detecting TP size.

    Supports both formats:
    - PipelineEngine: checkpoint['module']['module'] = state_dict
    - Sequential/Llama-style: checkpoint['module'] = state_dict (with 'sequential.' prefix)
    """
    # Find all mp_rank files
    shard_files = sorted([
        f for f in os.listdir(input_dir)
        if f.startswith("mp_rank_") and f.endswith("_model_states.pt")
    ])

    if not shard_files:
        raise FileNotFoundError(f"No mp_rank_*_model_states.pt files found in {input_dir}")

    tp_size = len(shard_files)
    print(f"Found {tp_size} TP shard(s)")

    state_dicts = []
    for shard_file in shard_files:
        path = os.path.join(input_dir, shard_file)
        print(f"Loading {shard_file}...")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Handle nested module format (PipelineEngine wraps in module.module)
        sd = ckpt["module"]
        if "module" in sd and isinstance(sd["module"], dict):
            # Check if the inner dict looks like a state dict (has tensor values)
            inner = sd["module"]
            if inner and isinstance(next(iter(inner.values())), torch.Tensor):
                sd = inner

        # Strip 'sequential.' prefix if present (Llama converter format)
        cleaned = {}
        for k, v in sd.items():
            if k.startswith("sequential."):
                cleaned[k[len("sequential."):]] = v
            else:
                cleaned[k] = v

        state_dicts.append(cleaned)

    return state_dicts, tp_size


def merge_tp_shards(state_dicts, tp_size, config):
    """Merge tensor-parallel shards into a single state dict."""
    if tp_size == 1:
        return state_dicts[0]

    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.hidden_size // num_heads
    q_size_per_tp = (num_heads * head_dim) // tp_size
    kv_size_per_tp = (num_kv_heads * head_dim) // tp_size

    merged = {}
    keys = state_dicts[0].keys()

    for key in tqdm(keys, desc="Merging TP shards"):
        tensors = [sd[key] for sd in state_dicts]

        if "q_norm.scale" in key or "k_norm.scale" in key:
            # Q/K norms are TP-split (match partitioned Q/K), concatenate back
            merged[key] = torch.cat(tensors, dim=0)
        elif any(x in key for x in ["layernorm", "norm.scale"]):
            # Other norms: average across ranks (should be identical, but average for safety)
            merged[key] = sum(tensors) / tp_size
        elif "query_key_value.weight" in key:
            # QKV: split each shard's [q_chunk, k_chunk, v_chunk] and merge separately
            q_chunks, k_chunks, v_chunks = [], [], []
            for t in tensors:
                q, k, v = torch.split(t, [q_size_per_tp, kv_size_per_tp, kv_size_per_tp], dim=0)
                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
            q_merged = torch.cat(q_chunks, dim=0)
            k_merged = torch.cat(k_chunks, dim=0)
            v_merged = torch.cat(v_chunks, dim=0)
            merged[key] = torch.cat([q_merged, k_merged, v_merged], dim=0)
        elif any(x in key for x in [
            "word_embeddings.weight", "final_linear.weight",
            "linear1.weight",
        ]):
            # Column parallel (split on dim 0) -> cat on dim 0
            merged[key] = torch.cat(tensors, dim=0)
        elif any(x in key for x in [
            "attention.dense.weight", "linear2.weight",
        ]):
            # Row parallel (split on dim 1) -> cat on dim 1
            merged[key] = torch.cat(tensors, dim=1)
        else:
            # Unknown: take first (should be replicated)
            merged[key] = tensors[0]

    return merged


def convert_neox_to_hf_state_dict(neox_state, config):
    """Convert merged NeoX state dict to HuggingFace OLMo-3 format."""
    hf_state = {}
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.hidden_size // num_heads
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    # Embedding
    hf_state["model.embed_tokens.weight"] = neox_state["0.word_embeddings.weight"]

    for layer_idx in tqdm(range(num_layers), desc="Converting layers"):
        seq_idx = layer_idx + 2
        prefix = f"model.layers.{layer_idx}"

        # === QKV split ===
        qkv = neox_state[f"{seq_idx}.attention.query_key_value.weight"]
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=0)
        hf_state[f"{prefix}.self_attn.q_proj.weight"] = q
        hf_state[f"{prefix}.self_attn.k_proj.weight"] = k
        hf_state[f"{prefix}.self_attn.v_proj.weight"] = v

        # Output projection
        hf_state[f"{prefix}.self_attn.o_proj.weight"] = (
            neox_state[f"{seq_idx}.attention.dense.weight"]
        )

        # Q/K norms: NeoX [hidden_size] / [kv_hidden_size], same as HF
        hf_state[f"{prefix}.self_attn.q_norm.weight"] = (
            neox_state[f"{seq_idx}.attention.q_norm.scale"]
        )
        hf_state[f"{prefix}.self_attn.k_norm.weight"] = (
            neox_state[f"{seq_idx}.attention.k_norm.scale"]
        )

        # === MLP: split linear1 back into up_proj and gate_proj ===
        linear1 = neox_state[f"{seq_idx}.mlp.linear1.weight"]
        up_proj, gate_proj = torch.chunk(linear1, 2, dim=0)
        hf_state[f"{prefix}.mlp.up_proj.weight"] = up_proj
        hf_state[f"{prefix}.mlp.gate_proj.weight"] = gate_proj

        # Down projection
        hf_state[f"{prefix}.mlp.down_proj.weight"] = (
            neox_state[f"{seq_idx}.mlp.linear2.weight"]
        )

        # === Layer norms ===
        hf_state[f"{prefix}.post_attention_layernorm.weight"] = (
            neox_state[f"{seq_idx}.post_attention_layernorm.scale"]
        )
        hf_state[f"{prefix}.post_feedforward_layernorm.weight"] = (
            neox_state[f"{seq_idx}.post_feedforward_layernorm.scale"]
        )

    # Final norm
    final_norm_idx = num_layers + 3
    hf_state["model.norm.weight"] = neox_state[f"{final_norm_idx}.norm.scale"]

    # LM head
    output_idx = num_layers + 4
    hf_state["lm_head.weight"] = neox_state[f"{output_idx}.final_linear.weight"]

    return hf_state


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeoX OLMo-3 checkpoint to HuggingFace format"
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to NeoX checkpoint dir (e.g., .../global_step100)")
    parser.add_argument("--hf-model", type=str, required=True,
                        help="Original HF model path (for config and tokenizer)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for HF model")
    parser.add_argument("--precision", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Precision for saved model (default: bfloat16)")
    args = parser.parse_args()

    # Load config from original HF model
    print(f"Loading config from: {args.hf_model}")
    config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=True)

    # Load NeoX checkpoint
    print(f"\nLoading NeoX checkpoint from: {args.input_dir}")
    state_dicts, tp_size = load_neox_checkpoint(args.input_dir)

    # Merge TP shards
    if tp_size > 1:
        print(f"\nMerging {tp_size} TP shards...")
    merged_state = merge_tp_shards(state_dicts, tp_size, config)
    del state_dicts

    # Convert to HF format
    print("\nConverting to HF format...")
    hf_state = convert_neox_to_hf_state_dict(merged_state, config)
    del merged_state

    # Cast to target precision
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    target_dtype = dtype_map[args.precision]
    for k in hf_state:
        hf_state[k] = hf_state[k].to(target_dtype)

    # Create HF model and load weights
    print("\nCreating HF model...")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=target_dtype)
    missing, unexpected = model.load_state_dict(hf_state, strict=False)
    if missing:
        print(f"Warning: Missing keys: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected}")
    del hf_state

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving HF model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)

    # Copy tokenizer from original model
    print("Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    # Save conversion metadata
    metadata = {
        "source": args.input_dir,
        "hf_model": args.hf_model,
        "precision": args.precision,
        "tp_size": tp_size,
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
    }
    with open(os.path.join(args.output_dir, "neox_conversion_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nConversion complete! HF model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
