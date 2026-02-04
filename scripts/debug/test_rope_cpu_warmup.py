#!/usr/bin/env python3
"""Verify that CPU RoPE warmup fixes the layer 8 divergence between HF and OLMo-core.

Tests two scenarios:
1. WITHOUT warmup: OLMo-core computes inv_freq on GPU -> divergence at layer 8
2. WITH warmup: OLMo-core uses CPU-computed inv_freq -> all layers match
"""

import logging

import torch
import transformers
from olmo_core.nn.hf.convert import convert_state_from_hf

from open_instruct import logger_utils
from open_instruct import olmo_core_utils

logger = logger_utils.setup_logger(__name__)


def compare_forward_passes(model_name: str, use_cpu_warmup: bool):
    device = torch.device("cuda")

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to(device)
    hf_model.eval()

    hf_config = transformers.AutoConfig.from_pretrained(model_name)
    olmo_config = olmo_core_utils.get_transformer_config(
        model_name, hf_config.vocab_size, attn_backend="torch"
    )
    olmo_model = olmo_config.build(init_device="cpu")

    if use_cpu_warmup:
        olmo_core_utils.warmup_rope_cache_on_cpu(olmo_model, max_seq_len=8192)

    hf_state = hf_model.state_dict()
    converted_state = convert_state_from_hf(hf_config, hf_state, model_type="olmo2")
    converted_state_gpu = {k: v.to(device) for k, v in converted_state.items()}
    olmo_model = olmo_model.to(device)
    olmo_model.load_state_dict(converted_state_gpu, assign=True, strict=False)
    olmo_model.eval()

    torch.manual_seed(42)
    input_ids = torch.randint(1, 100352, (1, 20), device=device)

    hf_hidden = []
    olmo_hidden = []

    def hf_layer_hook(module, inp, out):
        hf_hidden.append(out[0].detach().clone())

    def olmo_layer_hook(module, inp, out):
        olmo_hidden.append(out.detach().clone())

    handles = []
    for layer in hf_model.model.layers:
        handles.append(layer.register_forward_hook(hf_layer_hook))
    for key in olmo_model.blocks:
        handles.append(olmo_model.blocks[key].register_forward_hook(olmo_layer_hook))

    with torch.no_grad():
        _ = hf_model(input_ids)
        _ = olmo_model(input_ids)

    for h in handles:
        h.remove()

    first_diff_layer = None
    for i in range(len(hf_hidden)):
        diff = (hf_hidden[i] - olmo_hidden[i]).abs()
        max_diff = diff.max().item()
        status = "MATCH" if max_diff == 0 else "DIFF"
        logger.info(f"Layer {i:2d} output: max_diff={max_diff:.6e} [{status}]")
        if max_diff > 0 and first_diff_layer is None:
            first_diff_layer = i

    return first_diff_layer


def test_inv_freq_cpu_vs_gpu():
    device = torch.device("cuda")
    model_name = "allenai/OLMo-2-0425-1B"
    hf_config = transformers.AutoConfig.from_pretrained(model_name)
    theta = int(hf_config.rope_theta)
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads

    from olmo_core.nn.rope import compute_inv_freqs
    inv_freq_cpu = compute_inv_freqs(theta, head_dim, torch.device("cpu"))
    inv_freq_gpu = compute_inv_freqs(theta, head_dim, device)

    diff = (inv_freq_cpu - inv_freq_gpu.cpu()).abs()
    logger.info(f"inv_freq CPU vs GPU: max_diff={diff.max().item():.10e}, "
                f"num_nonzero={diff.nonzero().shape[0]}/{diff.numel()}")

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    hf_inv_freq = hf_model.model.rotary_emb.inv_freq
    logger.info(f"HF inv_freq device: {hf_inv_freq.device}, dtype: {hf_inv_freq.dtype}")
    del hf_model

    hf_vs_cpu = (hf_inv_freq.cpu().float() - inv_freq_cpu).abs()
    hf_vs_gpu = (hf_inv_freq.cpu().float() - inv_freq_gpu.cpu()).abs()
    logger.info(f"HF inv_freq vs CPU-computed: max_diff={hf_vs_cpu.max().item():.10e}")
    logger.info(f"HF inv_freq vs GPU-computed: max_diff={hf_vs_gpu.max().item():.10e}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    model_name = "allenai/OLMo-2-0425-1B"

    logger.info("=" * 60)
    logger.info("PART 1: Verify inv_freq differs between CPU and GPU")
    logger.info("=" * 60)
    test_inv_freq_cpu_vs_gpu()

    logger.info("\n" + "=" * 60)
    logger.info("PART 2: Forward pass WITHOUT CPU warmup (expect divergence)")
    logger.info("=" * 60)
    first_diff_no_warmup = compare_forward_passes(model_name, use_cpu_warmup=False)
    logger.info(f"First divergent layer WITHOUT warmup: {first_diff_no_warmup}")

    logger.info("\n" + "=" * 60)
    logger.info("PART 3: Forward pass WITH CPU warmup (expect all match)")
    logger.info("=" * 60)
    first_diff_with_warmup = compare_forward_passes(model_name, use_cpu_warmup=True)
    logger.info(f"First divergent layer WITH warmup: {first_diff_with_warmup}")

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    if first_diff_no_warmup is not None and first_diff_with_warmup is None:
        logger.info("SUCCESS: CPU warmup fixes the divergence!")
        logger.info(f"  Without warmup: diverges at layer {first_diff_no_warmup}")
        logger.info(f"  With warmup: all layers match")
    elif first_diff_with_warmup is not None:
        logger.info(f"PARTIAL: Warmup improved but didn't fully fix (diverges at layer {first_diff_with_warmup})")
    elif first_diff_no_warmup is None:
        logger.info("UNEXPECTED: No divergence even without warmup (CPU and GPU inv_freq may be identical on this hardware)")
    else:
        logger.info("FAILURE: Warmup did not help")


if __name__ == "__main__":
    main()
