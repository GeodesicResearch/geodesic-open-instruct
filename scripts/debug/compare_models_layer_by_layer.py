#!/usr/bin/env python3
"""Compare OLMo-core and HuggingFace models layer by layer.

This script identifies where numerical differences originate between the two
implementations by comparing outputs at each layer.
"""

import torch
import transformers
from olmo_core.nn.hf.convert import convert_state_from_hf

from open_instruct.olmo_core_utils import get_transformer_config


def main():
    model_name = "allenai/OLMo-2-0425-1B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading models on {device}...")

    # Load HuggingFace model
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    hf_model.eval()

    # Load OLMo-core model
    hf_config = transformers.AutoConfig.from_pretrained(model_name)
    olmo_config = get_transformer_config(model_name, hf_config.vocab_size, attn_backend="flash_2")
    olmo_model = olmo_config.build(init_device="cpu")

    # Convert and load weights
    hf_state = hf_model.state_dict()
    converted_state = convert_state_from_hf(hf_config, hf_state, model_type="olmo2")
    converted_state_gpu = {k: v.to(device) for k, v in converted_state.items()}
    olmo_model = olmo_model.to(device)
    olmo_model.load_state_dict(converted_state_gpu, assign=True, strict=False)
    olmo_model.eval()

    # Create test input
    torch.manual_seed(42)
    seq_len = 20
    input_ids = torch.randint(1, 100352, (1, seq_len), device=device)
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Input tokens: {input_ids[0, :10].tolist()}...")

    # Compare embeddings
    print("\n" + "=" * 60)
    print("EMBEDDING COMPARISON")
    print("=" * 60)

    with torch.no_grad():
        hf_embeds = hf_model.model.embed_tokens(input_ids)
        olmo_embeds = olmo_model.embeddings(input_ids)

    embed_diff = (hf_embeds - olmo_embeds).abs()
    print(f"HF embeddings shape: {hf_embeds.shape}")
    print(f"OLMo embeddings shape: {olmo_embeds.shape}")
    print(f"Max diff: {embed_diff.max().item():.6e}")
    print(f"Mean diff: {embed_diff.mean().item():.6e}")

    # Full forward pass comparison
    print("\n" + "=" * 60)
    print("FULL FORWARD PASS COMPARISON")
    print("=" * 60)

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        olmo_logits = olmo_model(input_ids)

    logits_diff = (hf_logits - olmo_logits).abs()
    print(f"HF logits shape: {hf_logits.shape}")
    print(f"OLMo logits shape: {olmo_logits.shape}")
    print(f"Max diff: {logits_diff.max().item():.6e}")
    print(f"Mean diff: {logits_diff.mean().item():.6e}")

    # Find position with max diff
    max_pos = logits_diff.argmax()
    b, s, v = max_pos // (seq_len * 100352), (max_pos % (seq_len * 100352)) // 100352, max_pos % 100352
    print(f"Max diff at position: batch={b.item()}, seq={s.item()}, vocab={v.item()}")
    print(f"  HF value: {hf_logits[b, s, v].item():.6f}")
    print(f"  OLMo value: {olmo_logits[b, s, v].item():.6f}")

    # Compare per-position
    print("\n" + "=" * 60)
    print("PER-POSITION MAX DIFF")
    print("=" * 60)

    for pos in range(min(10, seq_len)):
        pos_diff = logits_diff[0, pos, :].max().item()
        print(f"Position {pos:2d}: max_diff={pos_diff:.6e}")

    # Sample logits comparison
    print("\n" + "=" * 60)
    print("SAMPLE LOGITS AT POSITION 5")
    print("=" * 60)

    print(f"HF:   {hf_logits[0, 5, :10].tolist()}")
    print(f"OLMo: {olmo_logits[0, 5, :10].tolist()}")

    # Compare intermediate layer outputs using hooks
    print("\n" + "=" * 60)
    print("LAYER-BY-LAYER HIDDEN STATE COMPARISON")
    print("=" * 60)

    hf_hidden_states = []
    olmo_hidden_states = []

    def hf_hook(module, input, output):
        hf_hidden_states.append(output[0].detach())

    def olmo_hook(module, input, output):
        olmo_hidden_states.append(output.detach())

    # Register hooks on each layer
    hf_handles = []
    olmo_handles = []

    for i, layer in enumerate(hf_model.model.layers):
        hf_handles.append(layer.register_forward_hook(hf_hook))

    olmo_block_keys = list(olmo_model.blocks.keys())
    for key in olmo_block_keys:
        olmo_handles.append(olmo_model.blocks[key].register_forward_hook(olmo_hook))

    # Run forward passes with hooks
    with torch.no_grad():
        _ = hf_model(input_ids)
        _ = olmo_model(input_ids)

    # Remove hooks
    for h in hf_handles + olmo_handles:
        h.remove()

    # Compare layer outputs
    num_layers = min(len(hf_hidden_states), len(olmo_hidden_states))
    print(f"\nComparing {num_layers} layers:")

    first_large_diff_layer = None
    for i in range(num_layers):
        diff = (hf_hidden_states[i] - olmo_hidden_states[i]).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        status = "✓" if max_diff < 1e-3 else "✗"
        print(f"Layer {i:2d}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e} {status}")

        if max_diff > 0.01 and first_large_diff_layer is None:
            first_large_diff_layer = i

    if first_large_diff_layer is not None:
        print(f"\nFirst layer with large diff: {first_large_diff_layer}")

        # Investigate this layer more
        print(f"\nInvestigating Layer {first_large_diff_layer}...")

        hf_layer = hf_model.model.layers[first_large_diff_layer]
        olmo_block = olmo_model.blocks[olmo_block_keys[first_large_diff_layer]]

        # Get input to this layer
        if first_large_diff_layer == 0:
            hf_input = hf_embeds
            olmo_input = olmo_embeds
        else:
            hf_input = hf_hidden_states[first_large_diff_layer - 1]
            olmo_input = olmo_hidden_states[first_large_diff_layer - 1]

        with torch.no_grad():
            # Compare layer norms
            hf_normed = hf_layer.input_layernorm(hf_input)
            olmo_normed = olmo_block.attention_norm(olmo_input)
            norm_diff = (hf_normed - olmo_normed).abs().max().item()
            print(f"  Pre-attention norm diff: {norm_diff:.6e}")

            # Compare post-attention norm
            hf_post_norm = hf_layer.post_attention_layernorm(hf_input)
            olmo_post_norm = olmo_block.feed_forward_norm(olmo_input)
            post_norm_diff = (hf_post_norm - olmo_post_norm).abs().max().item()
            print(f"  Pre-FFN norm diff: {post_norm_diff:.6e}")


if __name__ == "__main__":
    main()
