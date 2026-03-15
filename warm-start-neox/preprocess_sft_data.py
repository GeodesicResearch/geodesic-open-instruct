#!/usr/bin/env python3
"""Preprocess SFT JSONL data into NeoX binary format for training.

Thin wrapper around NeoX's preprocess_data_with_chat_template.py that handles
tokenizer chat template setup for OLMo-3 models.

Usage:
    python preprocess_sft_data.py \
        --input /path/to/sft_data.jsonl \
        --output-prefix /path/to/output/sft_data \
        --tokenizer-path /path/to/tokenizer

Output:
    {prefix}_messages_document.bin/.idx       (token IDs)
    {prefix}_messages_label_document.bin/.idx  (loss mask labels)
"""

import argparse
import os
import subprocess
import sys
import tempfile

NEOX_REPO = os.environ.get(
    "NEOX_REPO", "/home/a5k/puria.a5k/geodesic-gpt-neox"
)

# OLMo thinker chat template (matches olmo_thinker from dataset_transformation.py)
OLMO_THINKER_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|system|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}"
    "<|user|>\n{{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|assistant|>\n{{ message['content'] }}"
    "{% if not loop.last %}\n{% endif %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def ensure_chat_template(tokenizer_path):
    """Check if the tokenizer has a chat template; if not, set one.

    Returns the path to use for tokenizer (may be a temp copy).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    if tokenizer.chat_template is not None:
        print(f"Tokenizer already has chat template: {tokenizer.chat_template[:80]}...")
        return tokenizer_path

    print("Tokenizer has no chat template — setting olmo_thinker template")
    tokenizer.chat_template = OLMO_THINKER_TEMPLATE

    # Save to a temp directory so we don't modify the original
    tmp_dir = tempfile.mkdtemp(prefix="neox_tokenizer_")
    tokenizer.save_pretrained(tmp_dir)
    print(f"Saved tokenizer with chat template to {tmp_dir}")
    return tmp_dir


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SFT JSONL data into NeoX binary format"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input JSONL file with 'messages' field")
    parser.add_argument("--output-prefix", type=str, required=True,
                        help="Output prefix (produces {prefix}_messages_document.bin/.idx)")
    parser.add_argument("--tokenizer-path", type=str, required=True,
                        help="Path to HF tokenizer directory")
    parser.add_argument("--neox-repo", type=str, default=NEOX_REPO,
                        help=f"Path to geodesic-gpt-neox repo (default: {NEOX_REPO})")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of preprocessing workers (default: 4)")
    parser.add_argument("--only-last", action="store_true",
                        help="Only train on the last assistant turn")
    args = parser.parse_args()

    preprocess_script = os.path.join(
        args.neox_repo, "tools", "datasets", "preprocess_data_with_chat_template.py"
    )
    if not os.path.exists(preprocess_script):
        print(f"ERROR: NeoX preprocessing script not found: {preprocess_script}")
        print(f"Set NEOX_REPO env var or --neox-repo to the geodesic-gpt-neox repo path")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Ensure tokenizer has chat template
    tokenizer_path = ensure_chat_template(args.tokenizer_path)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Check if output already exists (caching)
    bin_file = f"{args.output_prefix}_messages_document.bin"
    idx_file = f"{args.output_prefix}_messages_document.idx"
    label_bin = f"{args.output_prefix}_messages_label_document.bin"
    label_idx = f"{args.output_prefix}_messages_label_document.idx"

    if all(os.path.exists(f) for f in [bin_file, idx_file, label_bin, label_idx]):
        print(f"Output files already exist at {args.output_prefix} — skipping preprocessing")
        return

    # Use the NeoX venv's Python for the subprocess (it has megatron dependencies)
    neox_python = os.path.join(args.neox_repo, ".venv", "bin", "python")
    if not os.path.exists(neox_python):
        print(f"WARNING: NeoX venv Python not found at {neox_python}, falling back to current Python")
        neox_python = sys.executable

    # Build command
    cmd = [
        neox_python, preprocess_script,
        "--input", args.input,
        "--output-prefix", args.output_prefix,
        "--tokenizer-path", tokenizer_path,
        "--jsonl-keys", "messages",
        "--dataset-impl", "mmap",
        "--workers", str(args.workers),
    ]

    if args.only_last:
        cmd.append("--only-last")

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"ERROR: Preprocessing failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    # Verify output
    for f in [bin_file, idx_file, label_bin, label_idx]:
        if os.path.exists(f):
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  {os.path.basename(f)}: {size_mb:.1f} MB")
        else:
            print(f"  WARNING: Expected output not found: {f}")

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
