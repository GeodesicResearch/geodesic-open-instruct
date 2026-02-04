#!/usr/bin/env python3
"""Test whether inv_freq computed on CPU vs GPU differs, explaining layer 8 divergence."""

import logging

import torch
import transformers
from olmo_core.nn.rope import compute_inv_freqs

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def test_inv_freq_cpu_vs_gpu():
    device = torch.device("cuda")
    model_name = "allenai/OLMo-2-0425-1B"
    hf_config = transformers.AutoConfig.from_pretrained(model_name)

    theta = int(hf_config.rope_theta)
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    dim = head_dim

    logger.info(f"theta={theta}, head_dim={head_dim}, dim={dim}")

    logger.info("\n=== PART 1: Compare inv_freq on CPU vs GPU ===")
    inv_freq_cpu = compute_inv_freqs(theta, dim, torch.device("cpu"))
    inv_freq_gpu = compute_inv_freqs(theta, dim, device)

    diff = (inv_freq_cpu - inv_freq_gpu.cpu()).abs()
    logger.info(f"inv_freq CPU vs GPU: max_diff={diff.max().item():.10e}, "
                f"num_nonzero={diff.nonzero().shape[0]}/{diff.numel()}")
    if diff.max().item() > 0:
        for i in range(diff.shape[0]):
            if diff[i].item() > 0:
                logger.info(f"  idx={i}: cpu={inv_freq_cpu[i].item():.15e} "
                            f"gpu={inv_freq_gpu[i].cpu().item():.15e} "
                            f"diff={diff[i].item():.10e}")

    logger.info("\n=== PART 2: Compare HF inv_freq vs OLMo-core inv_freq ===")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    hf_model.eval()

    hf_inv_freq = hf_model.model.rotary_emb.inv_freq
    logger.info(f"HF inv_freq device: {hf_inv_freq.device}, dtype: {hf_inv_freq.dtype}")

    olmo_inv_freq_gpu = compute_inv_freqs(theta, dim, device)
    hf_vs_olmo = (hf_inv_freq.cpu().float() - olmo_inv_freq_gpu.cpu()).abs()
    logger.info(f"HF vs OLMo-core (GPU): max_diff={hf_vs_olmo.max().item():.10e}, "
                f"num_nonzero={hf_vs_olmo.nonzero().shape[0]}/{hf_vs_olmo.numel()}")

    hf_vs_cpu = (hf_inv_freq.cpu().float() - inv_freq_cpu).abs()
    logger.info(f"HF vs OLMo-core (CPU): max_diff={hf_vs_cpu.max().item():.10e}, "
                f"num_nonzero={hf_vs_cpu.nonzero().shape[0]}/{hf_vs_cpu.numel()}")

    logger.info("\n=== PART 3: Compare sin/cos embeddings ===")
    seq_len = 20
    seq_cpu = torch.arange(seq_len, dtype=torch.float)
    seq_gpu = torch.arange(seq_len, device=device, dtype=torch.float)

    freqs_cpu = torch.einsum("i,j->ij", seq_cpu, inv_freq_cpu)
    pos_cpu = torch.cat((freqs_cpu, freqs_cpu), dim=-1)
    sin_cpu = pos_cpu.sin()
    cos_cpu = pos_cpu.cos()

    freqs_gpu = torch.einsum("i,j->ij", seq_gpu, inv_freq_gpu)
    pos_gpu = torch.cat((freqs_gpu, freqs_gpu), dim=-1)
    sin_gpu = pos_gpu.sin()
    cos_gpu = pos_gpu.cos()

    sin_diff = (sin_cpu - sin_gpu.cpu()).abs()
    cos_diff = (cos_cpu - cos_gpu.cpu()).abs()
    logger.info(f"sin(pos) CPU vs GPU: max_diff={sin_diff.max().item():.10e}, "
                f"num_nonzero={sin_diff.nonzero().shape[0]}/{sin_diff.numel()}")
    logger.info(f"cos(pos) CPU vs GPU: max_diff={cos_diff.max().item():.10e}, "
                f"num_nonzero={cos_diff.nonzero().shape[0]}/{cos_diff.numel()}")

    sin_bf16_cpu = sin_cpu.to(torch.bfloat16)
    sin_bf16_gpu = sin_gpu.cpu().to(torch.bfloat16)
    cos_bf16_cpu = cos_cpu.to(torch.bfloat16)
    cos_bf16_gpu = cos_gpu.cpu().to(torch.bfloat16)

    sin_bf16_diff = (sin_bf16_cpu.float() - sin_bf16_gpu.float()).abs()
    cos_bf16_diff = (cos_bf16_cpu.float() - cos_bf16_gpu.float()).abs()
    logger.info(f"sin(pos) bfloat16 CPU vs GPU: max_diff={sin_bf16_diff.max().item():.10e}, "
                f"num_nonzero={sin_bf16_diff.nonzero().shape[0]}/{sin_bf16_diff.numel()}")
    logger.info(f"cos(pos) bfloat16 CPU vs GPU: max_diff={cos_bf16_diff.max().item():.10e}, "
                f"num_nonzero={cos_bf16_diff.nonzero().shape[0]}/{cos_bf16_diff.numel()}")

    logger.info("\n=== PART 4: Test with HF's exact inv_freq computation ===")
    base = hf_config.rope_theta
    hf_style_inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device="cpu", dtype=torch.float) / dim)
    )
    olmo_style_inv_freq_cpu = compute_inv_freqs(theta, dim, torch.device("cpu"))
    hf_vs_olmo_style = (hf_style_inv_freq - olmo_style_inv_freq_cpu).abs()
    logger.info(f"HF-style (CPU) vs OLMo-style (CPU): max_diff={hf_vs_olmo_style.max().item():.10e}")

    hf_style_inv_freq_gpu = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    olmo_style_inv_freq_gpu = compute_inv_freqs(theta, dim, device)
    hf_vs_olmo_style_gpu = (hf_style_inv_freq_gpu - olmo_style_inv_freq_gpu).abs()
    logger.info(f"HF-style (GPU) vs OLMo-style (GPU): max_diff={hf_vs_olmo_style_gpu.max().item():.10e}")

    cpu_vs_gpu_hf_style = (hf_style_inv_freq - hf_style_inv_freq_gpu.cpu()).abs()
    logger.info(f"HF-style CPU vs HF-style GPU: max_diff={cpu_vs_gpu_hf_style.max().item():.10e}, "
                f"num_nonzero={cpu_vs_gpu_hf_style.nonzero().shape[0]}/{cpu_vs_gpu_hf_style.numel()}")

    logger.info("\n=== PART 5: Full RoPE output comparison using HF's rotary_emb ===")
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    dummy_x = torch.randn(1, seq_len, hf_config.hidden_size, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        hf_cos, hf_sin = hf_model.model.rotary_emb(dummy_x, position_ids)

    olmo_inv = compute_inv_freqs(theta, dim, device)
    olmo_seq = torch.arange(seq_len, device=device, dtype=torch.float)
    with torch.autocast(device.type, enabled=False):
        olmo_freqs = torch.einsum("i,j->ij", olmo_seq, olmo_inv)
        olmo_pos = torch.cat((olmo_freqs, olmo_freqs), dim=-1)
        olmo_sin_f32 = olmo_pos.sin()
        olmo_cos_f32 = olmo_pos.cos()

    hf_cos_squeezed = hf_cos.squeeze(0)
    hf_sin_squeezed = hf_sin.squeeze(0)
    cos_diff_final = (hf_cos_squeezed.float() - olmo_cos_f32).abs()
    sin_diff_final = (hf_sin_squeezed.float() - olmo_sin_f32).abs()
    logger.info(f"HF rotary_emb cos vs OLMo-core cos: max_diff={cos_diff_final.max().item():.10e}, "
                f"num_nonzero={cos_diff_final.nonzero().shape[0]}/{cos_diff_final.numel()}")
    logger.info(f"HF rotary_emb sin vs OLMo-core sin: max_diff={sin_diff_final.max().item():.10e}, "
                f"num_nonzero={sin_diff_final.nonzero().shape[0]}/{sin_diff_final.numel()}")

    cos_bf16_diff_final = (hf_cos_squeezed - olmo_cos_f32.to(torch.bfloat16)).abs()
    sin_bf16_diff_final = (hf_sin_squeezed - olmo_sin_f32.to(torch.bfloat16)).abs()
    logger.info(f"(bfloat16) cos diff: max={cos_bf16_diff_final.max().item():.10e}, "
                f"nonzero={cos_bf16_diff_final.nonzero().shape[0]}/{cos_bf16_diff_final.numel()}")
    logger.info(f"(bfloat16) sin diff: max={sin_bf16_diff_final.max().item():.10e}, "
                f"nonzero={sin_bf16_diff_final.nonzero().shape[0]}/{sin_bf16_diff_final.numel()}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    test_inv_freq_cpu_vs_gpu()


if __name__ == "__main__":
    main()
