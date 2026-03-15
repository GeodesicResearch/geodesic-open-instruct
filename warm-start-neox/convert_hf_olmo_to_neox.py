#!/usr/bin/env python3
"""Convert a HuggingFace OLMo-3 model to GPT-NeoX checkpoint format.

Handles GQA (Grouped Query Attention) where num_kv_heads < num_attention_heads.
Based on geodesic-gpt-neox/huggingface/convert_hf_olmo_to_neox.py with GQA fix.

Usage:
    python convert_hf_olmo_to_neox.py --hf-model /path/to/olmo-3-32b --tp 4
    python convert_hf_olmo_to_neox.py --hf-model /path/to/olmo-3-7b --tp 1
"""

import argparse
import json
import os
from datetime import datetime, timezone

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def convert_olmo_to_neox_state_dict(model, config):
    """Convert OLMo-3 model weights to NeoX sequential format.

    NeoX uses a sequential layer numbering:
    - Layer 0: word_embeddings
    - Layer 1: (unused/skip - _pre_transformer_block function)
    - Layers 2 to num_layers+1: transformer layers
    - Layer num_layers+3: final layer norm
    - Layer num_layers+4: output embedding (lm_head)

    For GQA models, the QKV weight is concatenated as [Q, K, V]:
    - Q: [num_heads * head_dim, hidden_size]
    - K: [num_kv_heads * head_dim, hidden_size]
    - V: [num_kv_heads * head_dim, hidden_size]
    This matches NeoX's gqa_project() which splits along the last dim.
    """
    state_dict = {}
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.hidden_size // num_heads
    hf_state = model.state_dict()

    print(f"Converting OLMo-3 model: {num_layers} layers, {config.hidden_size} hidden, "
          f"{num_heads} Q heads, {num_kv_heads} KV heads (GQA={'yes' if num_kv_heads != num_heads else 'no'})")

    # Embedding layer (index 0)
    state_dict["0.word_embeddings.weight"] = hf_state["model.embed_tokens.weight"].clone().detach()
    print(f"Converted embedding: {state_dict['0.word_embeddings.weight'].shape}")

    # Transformer layers (indices 2 to num_layers+1)
    for layer_idx in tqdm(range(num_layers), desc="Converting layers"):
        seq_idx = layer_idx + 2
        prefix = f"model.layers.{layer_idx}"

        # === Attention: QKV ===
        q_weight = hf_state[f"{prefix}.self_attn.q_proj.weight"]  # [num_heads*head_dim, hidden]
        k_weight = hf_state[f"{prefix}.self_attn.k_proj.weight"]  # [num_kv_heads*head_dim, hidden]
        v_weight = hf_state[f"{prefix}.self_attn.v_proj.weight"]  # [num_kv_heads*head_dim, hidden]

        # Concatenate [Q, K, V] — NeoX's gqa_project() splits along this axis
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        state_dict[f"{seq_idx}.attention.query_key_value.weight"] = qkv_weight.clone().detach()

        # Output projection
        state_dict[f"{seq_idx}.attention.dense.weight"] = (
            hf_state[f"{prefix}.self_attn.o_proj.weight"].clone().detach()
        )

        # Separate Q and K norms (OLMo-3 specific)
        # HF stores: q_norm [hidden_size], k_norm [kv_hidden_size]
        # NeoX (with TP fix) expects: q_norm [hidden_size/tp], k_norm [kv_hidden_size/tp]
        # Copy as-is; TP sharding handles the split.
        state_dict[f"{seq_idx}.attention.q_norm.scale"] = (
            hf_state[f"{prefix}.self_attn.q_norm.weight"].clone().detach()
        )
        state_dict[f"{seq_idx}.attention.k_norm.scale"] = (
            hf_state[f"{prefix}.self_attn.k_norm.weight"].clone().detach()
        )

        # === MLP (SwiGLU) ===
        # NeoX expects [up_proj; gate_proj] concatenated
        gate_weight = hf_state[f"{prefix}.mlp.gate_proj.weight"]
        up_weight = hf_state[f"{prefix}.mlp.up_proj.weight"]
        linear1_weight = torch.cat([up_weight, gate_weight], dim=0)
        state_dict[f"{seq_idx}.mlp.linear1.weight"] = linear1_weight.clone().detach()

        # Down projection
        state_dict[f"{seq_idx}.mlp.linear2.weight"] = (
            hf_state[f"{prefix}.mlp.down_proj.weight"].clone().detach()
        )

        # === Layer Norms (OLMo-3 post-norm style) ===
        state_dict[f"{seq_idx}.post_attention_layernorm.scale"] = (
            hf_state[f"{prefix}.post_attention_layernorm.weight"].clone().detach()
        )
        state_dict[f"{seq_idx}.post_feedforward_layernorm.scale"] = (
            hf_state[f"{prefix}.post_feedforward_layernorm.weight"].clone().detach()
        )

    # Final layer norm (index num_layers + 3)
    final_norm_idx = num_layers + 3
    state_dict[f"{final_norm_idx}.norm.scale"] = (
        hf_state["model.norm.weight"].clone().detach()
    )
    print("Converted final layer norm")

    # Output embedding / LM head (index num_layers + 4)
    output_idx = num_layers + 4
    state_dict[f"{output_idx}.final_linear.weight"] = (
        hf_state["lm_head.weight"].clone().detach()
    )
    print(f"Converted output embedding: {state_dict[f'{output_idx}.final_linear.weight'].shape}")

    return state_dict


def shard_single_rank(state_dict, tp_rank, tp_size, config):
    """Extract shard for a single TP rank (memory-efficient: no copies of other ranks)."""
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.hidden_size // num_heads

    assert num_heads % tp_size == 0, f"num_heads={num_heads} not divisible by tp={tp_size}"
    assert num_kv_heads % tp_size == 0, f"num_kv_heads={num_kv_heads} not divisible by tp={tp_size}"

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    shard = {}
    for key, tensor in state_dict.items():
        # Q/K norms must be split to match TP-partitioned Q/K sizes
        if "q_norm.scale" in key:
            shard[key] = torch.chunk(tensor, tp_size, dim=0)[tp_rank].clone()
        elif "k_norm.scale" in key:
            shard[key] = torch.chunk(tensor, tp_size, dim=0)[tp_rank].clone()
        elif any(x in key for x in ["layernorm", "norm.scale", "rotary_emb"]):
            shard[key] = tensor.clone()
        elif "query_key_value.weight" in key:
            q, k, v = torch.split(tensor, [q_size, kv_size, kv_size], dim=0)
            q_chunk = torch.chunk(q, tp_size, dim=0)[tp_rank]
            k_chunk = torch.chunk(k, tp_size, dim=0)[tp_rank]
            v_chunk = torch.chunk(v, tp_size, dim=0)[tp_rank]
            shard[key] = torch.cat([q_chunk, k_chunk, v_chunk], dim=0).clone()
        elif any(x in key for x in [
            "word_embeddings.weight", "final_linear.weight", "linear1.weight",
        ]):
            shard[key] = torch.chunk(tensor, tp_size, dim=0)[tp_rank].clone()
        elif any(x in key for x in [
            "attention.dense.weight", "linear2.weight",
        ]):
            shard[key] = torch.chunk(tensor, tp_size, dim=1)[tp_rank].clone()
        else:
            print(f"Warning: Unknown key pattern for sharding: {key}, replicating")
            shard[key] = tensor.clone()
    return shard


def save_neox_checkpoint_streaming(state_dict, tp_size, config, output_dir, iteration=0):
    """Shard and save one TP rank at a time to minimize peak memory."""
    import gc
    ckpt_dir = os.path.join(output_dir, f"global_step{iteration}")
    os.makedirs(ckpt_dir, exist_ok=True)

    for tp_rank in range(tp_size):
        print(f"Sharding and saving TP rank {tp_rank}/{tp_size}...")
        if tp_size > 1:
            shard = shard_single_rank(state_dict, tp_rank, tp_size, config)
        else:
            shard = state_dict

        checkpoint = {
            "dp_world_size": 1,
            "mp_world_size": tp_size,
            "optimizer": {},
            "global_steps": iteration,
            "global_samples": 0,
            "skipped_steps": 0,
            "iteration": iteration,
            "module": {"module": shard},
            "buffer_names": [],
            "param_shapes": {},
            "frozen_param_shapes": {},
            "shared_params": [],
            "frozen_param_fragments": {},
            "lr_scheduler": {},
            "data_sampler": {},
            "random_ltd": {},
            "sparse_tensor_module_names": [],
            "ds_config": {},
            "ds_version": "0.14.0",
        }

        save_path = os.path.join(ckpt_dir, f"mp_rank_{tp_rank:02d}_model_states.pt")
        print(f"Saving {save_path}...")
        torch.save(checkpoint, save_path)
        del checkpoint, shard
        gc.collect()

    # Write the 'latest' file
    latest_path = os.path.join(output_dir, "latest")
    with open(latest_path, "w") as f:
        f.write(f"global_step{iteration}")

    print(f"Checkpoint saved to {ckpt_dir}")


def create_olmo3_neox_config(config, output_dir):
    """Create a NeoX-compatible config for the converted OLMo-3 model."""
    neox_config = {
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "seq_length": min(getattr(config, "max_position_embeddings", 8192), 8192),
        "max_position_embeddings": getattr(config, "max_position_embeddings", 65536),
        "vocab_size": config.vocab_size,
        "intermediate_size": config.intermediate_size,
        "norm": "rmsnorm",
        "norm_placement": "olmo3",
        "use_qk_layernorm": True,
        "use_separate_qk_norms": True,
        "rms_norm_epsilon": getattr(config, "rms_norm_eps", 1e-6),
        "activation": "swiglu",
        "use_bias_in_attn_linear": False,
        "use_bias_in_mlp": False,
        "use_bias_in_norms": False,
        "pos_emb": "rotary",
        "rotary_pct": 1.0,
        "rotary_emb_base": getattr(config, "rope_theta", 500000),
        "sliding_window_width": getattr(config, "sliding_window", None),
        "no_weight_tying": not getattr(config, "tie_word_embeddings", False),
        "precision": "bfloat16",
        "attention_config": [["flash"], config.num_hidden_layers],
    }

    config_path = os.path.join(output_dir, "neox_config.json")
    with open(config_path, "w") as f:
        json.dump(neox_config, f, indent=2)
    print(f"NeoX config saved to {config_path}")

    return neox_config


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace OLMo-3 model to NeoX checkpoint format (with GQA support)"
    )
    parser.add_argument("--hf-model", type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--revision", type=str, default=None,
                        help="Model revision/branch")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save the NeoX checkpoint")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallelism size (default: 1)")
    parser.add_argument("--iteration", type=int, default=0,
                        help="Iteration number for the checkpoint (default: 0)")
    parser.add_argument("--save-tokenizer", action="store_true",
                        help="Also save the tokenizer")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Data type for loading the model (default: bfloat16)")
    args = parser.parse_args()

    if args.output_dir is None:
        model_name = args.hf_model.rstrip("/").split("/")[-1]
        args.output_dir = f"/projects/a5k/public/checkpoints/sf_model_organisms/{model_name}"
        print(f"Using default output directory: {args.output_dir}")

    print(f"Loading HF model: {args.hf_model}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    config = AutoConfig.from_pretrained(
        args.hf_model, revision=args.revision, trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, revision=args.revision,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True, low_cpu_mem_usage=True,
        device_map="cpu",
    )
    print(f"Model loaded: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")

    print("\nConverting weights...")
    state_dict = convert_olmo_to_neox_state_dict(model, config)
    del model  # Free HF model memory before sharding
    import gc; gc.collect()
    print(f"Converted {len(state_dict)} weight tensors")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nSaving checkpoint to {args.output_dir} (TP={args.tp})...")
    save_neox_checkpoint_streaming(state_dict, args.tp, config, args.output_dir, args.iteration)
    del state_dict; gc.collect()

    print("\nCreating NeoX config...")
    create_olmo3_neox_config(config, args.output_dir)

    metadata = {
        "hf_model": args.hf_model,
        "revision": args.revision,
        "output_dir": args.output_dir,
        "tp": args.tp,
        "iteration": args.iteration,
        "dtype": args.dtype,
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "vocab_size": config.vocab_size,
        "model_type": "olmo3",
        "converted_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = os.path.join(args.output_dir, "conversion_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if args.save_tokenizer:
        print("\nSaving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_model, revision=args.revision, trust_remote_code=True,
        )
        tokenizer_path = os.path.join(args.output_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")

    print("\nConversion complete!")
    print(f"  Checkpoint: {args.output_dir}")
    print(f"  Config:     {args.output_dir}/neox_config.json")


if __name__ == "__main__":
    main()
