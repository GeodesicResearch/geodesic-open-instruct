#!/usr/bin/env python3
"""Hook inside attention at layers 7 and 8 to find where the difference originates.

Both models use FA2 so layers 0-7 should match and layer 8 should diverge.
We capture intermediate values within the attention computation:
- Q, K, V after projection
- Q, K after QK norm
- Q, K after RoPE
- Attention output (before output projection)
- After output projection
"""

import logging

import torch
import transformers
from olmo_core.nn.hf.convert import convert_state_from_hf

from open_instruct import logger_utils
from open_instruct.olmo_core_utils import get_transformer_config

logger = logger_utils.setup_logger(__name__)


def compare_attention_internals():
    model_name = "allenai/OLMo-2-0425-1B"
    device = torch.device("cuda")

    logger.info("Loading HuggingFace model (FA2)...")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    hf_model.eval()

    logger.info("Loading OLMo-core model (FA2)...")
    hf_config = transformers.AutoConfig.from_pretrained(model_name)
    olmo_config = get_transformer_config(
        model_name, hf_config.vocab_size, attn_backend="flash_2"
    )
    olmo_model = olmo_config.build(init_device="cpu")

    hf_state = hf_model.state_dict()
    converted_state = convert_state_from_hf(hf_config, hf_state, model_type="olmo2")
    converted_state_gpu = {k: v.to(device) for k, v in converted_state.items()}
    olmo_model = olmo_model.to(device)
    olmo_model.load_state_dict(converted_state_gpu, assign=True, strict=False)
    olmo_model.eval()

    torch.manual_seed(42)
    input_ids = torch.randint(1, 100352, (1, 20), device=device)

    logger.info("=== PART 1: Verify layer outputs (FA2 vs FA2) ===")
    hf_outputs = []
    olmo_outputs = []

    def hf_block_hook(module, inp, out):
        if isinstance(out, tuple):
            hf_outputs.append(out[0].detach().clone())
        else:
            hf_outputs.append(out.detach().clone())

    def olmo_block_hook(module, inp, out):
        olmo_outputs.append(out.detach().clone())

    handles = []
    for layer in hf_model.model.layers:
        handles.append(layer.register_forward_hook(hf_block_hook))
    for key in olmo_model.blocks:
        handles.append(olmo_model.blocks[key].register_forward_hook(olmo_block_hook))

    with torch.no_grad():
        _ = hf_model(input_ids)
        _ = olmo_model(input_ids)

    for h in handles:
        h.remove()

    first_diff_layer = None
    for i in range(len(hf_outputs)):
        diff = (hf_outputs[i] - olmo_outputs[i]).abs().max().item()
        status = "MATCH" if diff == 0 else "DIFF"
        logger.info(f"Layer {i:2d} output: max_diff={diff:.6e} [{status}]")
        if diff > 0 and first_diff_layer is None:
            first_diff_layer = i

    if first_diff_layer is None:
        logger.info("All layers match! No differences found.")
        return

    logger.info(f"\nFirst divergence at layer {first_diff_layer}")

    target_layers = [max(0, first_diff_layer - 1), first_diff_layer]
    logger.info(f"Will instrument layers: {target_layers}")

    logger.info("\n=== PART 2: Hook inside HF attention ===")

    hf_internals = {}
    olmo_internals = {}

    for layer_idx in target_layers:
        hf_attn = hf_model.model.layers[layer_idx].self_attn

        original_forward = hf_attn.forward

        def make_hf_hook(idx, attn_module, orig_fwd):
            def hooked_forward(hidden_states, position_embeddings, attention_mask=None, **kwargs):
                q = attn_module.q_proj(hidden_states)
                k = attn_module.k_proj(hidden_states)
                v = attn_module.v_proj(hidden_states)

                data = {
                    "q_after_proj": q.detach().clone(),
                    "k_after_proj": k.detach().clone(),
                    "v_after_proj": v.detach().clone(),
                }

                q_normed = attn_module.q_norm(q)
                k_normed = attn_module.k_norm(k)

                data["q_after_norm"] = q_normed.detach().clone()
                data["k_after_norm"] = k_normed.detach().clone()

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn_module.head_dim)

                q_reshaped = q_normed.view(hidden_shape).transpose(1, 2)
                k_reshaped = k_normed.view(hidden_shape).transpose(1, 2)
                v_reshaped = v.view(hidden_shape).transpose(1, 2)

                data["q_after_reshape"] = q_reshaped.detach().clone()

                cos, sin = position_embeddings
                from transformers.models.olmo2.modeling_olmo2 import (
                    apply_rotary_pos_emb,
                )

                q_rope, k_rope = apply_rotary_pos_emb(
                    q_reshaped, k_reshaped, cos, sin
                )

                data["q_after_rope"] = q_rope.detach().clone()
                data["k_after_rope"] = k_rope.detach().clone()
                data["q_after_rope_shape"] = q_rope.shape

                hf_internals[idx] = data

                return orig_fwd(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    **kwargs,
                )

            return hooked_forward

        hf_attn.forward = make_hf_hook(layer_idx, hf_attn, original_forward)

    for layer_idx in target_layers:
        olmo_attn = olmo_model.blocks[str(layer_idx)].attention
        original_olmo_forward = olmo_attn.forward

        def make_olmo_hook(idx, attn_module, orig_fwd):
            def hooked_forward(x, **kwargs):
                B, T, _ = x.shape

                q = attn_module.w_q(x)
                k = attn_module.w_k(x)
                v = attn_module.w_v(x)

                data = {
                    "q_after_proj": q.detach().clone(),
                    "k_after_proj": k.detach().clone(),
                    "v_after_proj": v.detach().clone(),
                }

                if not attn_module.use_head_qk_norm:
                    if attn_module.q_norm is not None:
                        q_normed = attn_module.q_norm(q)
                    else:
                        q_normed = q
                    if attn_module.k_norm is not None:
                        k_normed = attn_module.k_norm(k)
                    else:
                        k_normed = k

                    data["q_after_norm"] = q_normed.detach().clone()
                    data["k_after_norm"] = k_normed.detach().clone()

                    q_reshaped = q_normed.view(B, T, -1, attn_module.head_dim)
                    k_reshaped = k_normed.view(B, T, -1, attn_module.head_dim)
                else:
                    q_reshaped = q.view(B, T, -1, attn_module.head_dim)
                    k_reshaped = k.view(B, T, -1, attn_module.head_dim)

                    if attn_module.q_norm is not None:
                        q_reshaped = attn_module.q_norm(q_reshaped)
                    if attn_module.k_norm is not None:
                        k_reshaped = attn_module.k_norm(k_reshaped)

                    data["q_after_norm"] = q_reshaped.reshape(
                        B, T, -1
                    ).detach().clone()
                    data["k_after_norm"] = k_reshaped.reshape(
                        B, T, -1
                    ).detach().clone()

                data["q_after_reshape"] = q_reshaped.detach().clone()

                if attn_module.rope is not None:
                    v_reshaped = v.view(B, T, -1, attn_module.head_dim)
                    q_rope, k_rope = attn_module.rope(
                        q_reshaped, k_reshaped, head_first=False, **{
                            kk: vv for kk, vv in kwargs.items()
                            if kk in ("pos_sin", "pos_cos", "freqs_cis", "cu_doc_lens")
                        }
                    )
                    data["q_after_rope"] = q_rope.detach().clone()
                    data["k_after_rope"] = k_rope.detach().clone()
                    data["q_after_rope_shape"] = q_rope.shape

                olmo_internals[idx] = data
                return orig_fwd(x, **kwargs)

            return hooked_forward

        olmo_attn.forward = make_olmo_hook(layer_idx, olmo_attn, original_olmo_forward)

    with torch.no_grad():
        _ = hf_model(input_ids)
        _ = olmo_model(input_ids)

    logger.info("\n=== PART 2 RESULTS: Attention internals comparison ===")
    for layer_idx in target_layers:
        if layer_idx not in hf_internals or layer_idx not in olmo_internals:
            logger.info(
                f"Layer {layer_idx}: MISSING DATA "
                f"(hf={layer_idx in hf_internals}, olmo={layer_idx in olmo_internals})"
            )
            continue

        hf_d = hf_internals[layer_idx]
        olmo_d = olmo_internals[layer_idx]

        logger.info(f"\n--- Layer {layer_idx} ---")

        for name in ["q_after_proj", "k_after_proj", "v_after_proj"]:
            hf_t = hf_d[name]
            olmo_t = olmo_d[name]
            diff = (hf_t - olmo_t).abs().max().item()
            status = "MATCH" if diff == 0 else "DIFF"
            logger.info(f"  {name}: max_diff={diff:.6e} [{status}]")

        for name in ["q_after_norm", "k_after_norm"]:
            hf_t = hf_d[name]
            olmo_t = olmo_d[name]
            diff = (hf_t - olmo_t).abs().max().item()
            status = "MATCH" if diff == 0 else "DIFF"
            logger.info(f"  {name}: max_diff={diff:.6e} [{status}]")

        hf_q_shape = hf_d.get("q_after_reshape", torch.tensor(0)).shape
        olmo_q_shape = olmo_d.get("q_after_reshape", torch.tensor(0)).shape
        logger.info(f"  q_after_reshape: HF={hf_q_shape}, OLMo={olmo_q_shape}")

        if "q_after_rope" in hf_d and "q_after_rope" in olmo_d:
            hf_q_rope = hf_d["q_after_rope"]
            olmo_q_rope = olmo_d["q_after_rope"]
            hf_k_rope = hf_d["k_after_rope"]
            olmo_k_rope = olmo_d["k_after_rope"]

            logger.info(
                f"  q_after_rope shapes: HF={hf_q_rope.shape}, OLMo={olmo_q_rope.shape}"
            )

            if hf_q_rope.shape != olmo_q_rope.shape:
                if hf_q_rope.dim() == 4 and olmo_q_rope.dim() == 4:
                    if hf_q_rope.shape[1] != olmo_q_rope.shape[1]:
                        hf_q_compare = hf_q_rope.transpose(1, 2)
                        hf_k_compare = hf_k_rope.transpose(1, 2)
                    else:
                        hf_q_compare = hf_q_rope
                        hf_k_compare = hf_k_rope
                else:
                    hf_q_compare = hf_q_rope
                    hf_k_compare = hf_k_rope
            else:
                hf_q_compare = hf_q_rope
                hf_k_compare = hf_k_rope

            q_rope_diff = (hf_q_compare - olmo_q_rope).abs().max().item()
            k_rope_diff = (hf_k_compare - olmo_k_rope).abs().max().item()
            q_status = "MATCH" if q_rope_diff == 0 else "DIFF"
            k_status = "MATCH" if k_rope_diff == 0 else "DIFF"
            logger.info(f"  q_after_rope: max_diff={q_rope_diff:.6e} [{q_status}]")
            logger.info(f"  k_after_rope: max_diff={k_rope_diff:.6e} [{k_status}]")

    logger.info("\n=== PART 3: RoPE buffer comparison ===")
    logger.info("Comparing RoPE cos/sin values between HF and OLMo-core")

    hf_rope = hf_model.model.rotary_emb
    position_ids = torch.arange(0, 20, device=device).unsqueeze(0)
    dummy_x = torch.randn(1, 20, 2048, dtype=torch.bfloat16, device=device)
    hf_cos, hf_sin = hf_rope(dummy_x, position_ids)

    olmo_rope = olmo_model.blocks["0"].attention.rope
    olmo_buffers = olmo_rope.get_buffers(20, device)

    logger.info(f"HF cos shape: {hf_cos.shape}, dtype: {hf_cos.dtype}")
    logger.info(f"OLMo pos_cos shape: {olmo_buffers.pos_cos.shape}, dtype: {olmo_buffers.pos_cos.dtype}")

    if hf_cos.shape[-1] == olmo_buffers.pos_cos.shape[-1]:
        hf_cos_flat = hf_cos.squeeze()
        olmo_cos_flat = olmo_buffers.pos_cos.squeeze()

        if hf_cos_flat.shape == olmo_cos_flat.shape:
            cos_diff = (hf_cos_flat.float() - olmo_cos_flat.float()).abs().max().item()
            logger.info(f"cos diff: {cos_diff:.6e}")
        else:
            logger.info(
                f"Shape mismatch after squeeze: HF={hf_cos_flat.shape}, OLMo={olmo_cos_flat.shape}"
            )
    else:
        logger.info(
            f"Last dim mismatch: HF cos last={hf_cos.shape[-1]}, "
            f"OLMo cos last={olmo_buffers.pos_cos.shape[-1]}"
        )

    hf_sin_flat = hf_sin.squeeze()
    olmo_sin_flat = olmo_buffers.pos_sin.squeeze()
    if hf_sin_flat.shape == olmo_sin_flat.shape:
        sin_diff = (hf_sin_flat.float() - olmo_sin_flat.float()).abs().max().item()
        logger.info(f"sin diff: {sin_diff:.6e}")
    else:
        logger.info(
            f"Sin shape mismatch: HF={hf_sin_flat.shape}, OLMo={olmo_sin_flat.shape}"
        )

    logger.info("\n=== PART 4: Check HF checkpoint shard boundary ===")
    import json
    import os

    cache_dir = transformers.utils.hub.default_cache_path
    model_dir = None
    for d in os.listdir(cache_dir):
        if "OLMo-2-0425-1B" in d and os.path.isdir(os.path.join(cache_dir, d)):
            snapshots_dir = os.path.join(cache_dir, d, "snapshots")
            if os.path.exists(snapshots_dir):
                for snap in os.listdir(snapshots_dir):
                    candidate = os.path.join(snapshots_dir, snap)
                    if os.path.isdir(candidate):
                        model_dir = candidate
                        break
            break

    if model_dir:
        index_file = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            shard_7 = weight_map.get("model.layers.7.self_attn.q_proj.weight", "?")
            shard_8 = weight_map.get("model.layers.8.self_attn.q_proj.weight", "?")
            logger.info(f"Layer 7 weights in shard: {shard_7}")
            logger.info(f"Layer 8 weights in shard: {shard_8}")
            if shard_7 != shard_8:
                logger.info("DIFFERENT SHARDS! Checkpoint boundary between layers 7 and 8!")
            else:
                logger.info("Same shard for both layers")
        else:
            logger.info(f"Index file not found at {index_file}")
    else:
        logger.info("Could not find model cache directory")


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    compare_attention_internals()


if __name__ == "__main__":
    main()
