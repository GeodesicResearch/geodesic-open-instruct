"""Minimal reproduction of OLMo-core doc_lens not resetting RoPE positions.

This script demonstrates that when using doc_lens for intra-document masking,
RoPE positions are NOT reset per document. The second document gets RoPE
positions that continue from the first document instead of starting from 0.

Expected behavior: Both documents should have identical logits since they
contain identical tokens and doc_lens should create independent attention.

Actual behavior: The second document has different logits because it uses
RoPE positions [seq_len, seq_len+1, ...] instead of [0, 1, ...].

To run:
    python scripts/debug/olmo_core_doc_lens_rope_issue.py

Requires: CUDA and flash attention (flash-attn package).
"""

import torch
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.transformer import TransformerConfig


def main():
    print("Creating OLMo-core model with flash attention...")
    config = TransformerConfig.olmo2_1B(vocab_size=100352)
    config.n_layers = 2
    config.block.attention = AttentionConfig(
        n_heads=config.block.attention.n_heads,
        n_kv_heads=config.block.attention.n_kv_heads,
        bias=config.block.attention.bias,
        rope=config.block.attention.rope,
        qk_norm=config.block.attention.qk_norm,
        backend="flash_2",
    )
    model = config.build().cuda().to(torch.bfloat16).eval()

    seq_len = 10
    seq = torch.randint(1, 100352, (1, seq_len), device="cuda")

    packed = torch.cat([seq, seq], dim=1)
    doc_lens = torch.tensor([seq_len, seq_len], device="cuda")

    print(f"\nInput sequence (length {seq_len}): {seq[0, :5].tolist()}...")
    print(f"Packed sequence (length {2*seq_len}): [seq | seq]")
    print(f"doc_lens: {doc_lens.tolist()}")

    with torch.no_grad():
        logits_packed = model(packed, doc_lens=doc_lens, max_doc_lens=[seq_len])
        logits_separate = model(seq)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nPosition 0 logits (first 5 vocab entries):")
    print(f"  Packed doc1[0]:  {logits_packed[0, 0, :5].tolist()}")
    print(f"  Packed doc2[0]:  {logits_packed[0, seq_len, :5].tolist()}")
    print(f"  Separate[0]:     {logits_separate[0, 0, :5].tolist()}")

    print("\nPosition 1 logits (first 5 vocab entries):")
    print(f"  Packed doc1[1]:  {logits_packed[0, 1, :5].tolist()}")
    print(f"  Packed doc2[1]:  {logits_packed[0, seq_len + 1, :5].tolist()}")
    print(f"  Separate[1]:     {logits_separate[0, 1, :5].tolist()}")

    pos0_doc1_matches = torch.allclose(
        logits_packed[0, 0, :], logits_separate[0, 0, :], atol=1e-3, rtol=1e-3
    )
    pos0_doc2_matches = torch.allclose(
        logits_packed[0, seq_len, :], logits_separate[0, 0, :], atol=1e-3, rtol=1e-3
    )
    pos1_doc1_matches = torch.allclose(
        logits_packed[0, 1, :], logits_separate[0, 1, :], atol=1e-3, rtol=1e-3
    )
    pos1_doc2_matches = torch.allclose(
        logits_packed[0, seq_len + 1, :], logits_separate[0, 1, :], atol=1e-3, rtol=1e-3
    )

    print("\n" + "=" * 60)
    print("COMPARISON (should all be True if doc_lens resets RoPE)")
    print("=" * 60)
    print(f"  pos0 doc1 matches separate: {pos0_doc1_matches}")
    print(f"  pos0 doc2 matches separate: {pos0_doc2_matches}")
    print(f"  pos1 doc1 matches separate: {pos1_doc1_matches}")
    print(f"  pos1 doc2 matches separate: {pos1_doc2_matches}  <-- FAILS")

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    if pos1_doc1_matches and not pos1_doc2_matches:
        print("CONFIRMED: doc_lens does NOT reset RoPE positions per document.")
        print("")
        print("The first document (doc1) matches separate forward because it gets")
        print("RoPE positions [0, 1, ..., 9] which is correct.")
        print("")
        print("The second document (doc2) does NOT match because it gets")
        print("RoPE positions [10, 11, ..., 19] instead of [0, 1, ..., 9].")
        print("")
        print("doc_lens correctly masks attention (documents can't attend to each")
        print("other), but RoPE positions are applied globally across the packed")
        print("sequence instead of resetting per document.")
    else:
        print("Unexpected result - please investigate further.")

    return not pos1_doc2_matches


if __name__ == "__main__":
    has_issue = main()
    exit(0 if has_issue else 1)
