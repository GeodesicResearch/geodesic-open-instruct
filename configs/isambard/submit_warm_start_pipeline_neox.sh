#!/bin/bash
# Submit a NeoX-based warm-start SFT -> GRPO pipeline as dependent SLURM jobs.
#
# Pipeline phases:
#   Phase 0:   Preprocess SFT data (JSONL -> NeoX binary)     [cached]
#   Phase 0.5: Convert HF model -> NeoX checkpoint             [cached]
#   Phase 1:   NeoX SFT training
#   Phase 1.5: Convert NeoX checkpoint -> HF model
#   Phase 2:   GRPO RL training
#
# Usage:
#   bash configs/isambard/submit_warm_start_pipeline_neox.sh <config.yaml> [grpo_nodes]
#
# Example:
#   bash configs/isambard/submit_warm_start_pipeline_neox.sh \
#       configs/isambard/march_13_sycophancy_rh/sycophancy_grpo_olmo3_32b_neox.yaml 8

set -euo pipefail

CONFIG=${1:?Usage: submit_warm_start_pipeline_neox.sh <config.yaml> [grpo_nodes]}
GRPO_NODES=${2:-2}

# Resolve config to absolute path
if [[ "$CONFIG" != /* ]]; then
    CONFIG="$(pwd)/$CONFIG"
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_DIR/.venv/bin/activate"

# Extract config values
PIPELINE_CONFIG=$(python3 -c "
import yaml, sys, getpass
user = getpass.getuser()
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

def resolve(val):
    if isinstance(val, str) and '{user}' in val:
        return val.replace('{user}', user)
    return val

print(resolve(cfg.get('warm_start_sft_output_dir', '')))
print(resolve(cfg.get('model_name_or_path', '')))
print(resolve(cfg.get('warm_start_sft_dataset', '')))
print(cfg.get('warm_start_neox_nodes', 1))
print(cfg.get('warm_start_neox_tp', 4))
print(resolve(cfg.get('warm_start_neox_config', 'warm-start-neox/sft_olmo3_32b.yml')))
" "$CONFIG")

SFT_OUTPUT_DIR=$(echo "$PIPELINE_CONFIG" | sed -n '1p')
BASE_MODEL=$(echo "$PIPELINE_CONFIG" | sed -n '2p')
SFT_DATASET=$(echo "$PIPELINE_CONFIG" | sed -n '3p')
NEOX_NODES=$(echo "$PIPELINE_CONFIG" | sed -n '4p')
NEOX_TP=$(echo "$PIPELINE_CONFIG" | sed -n '5p')
NEOX_CONFIG=$(echo "$PIPELINE_CONFIG" | sed -n '6p')

# Resolve NeoX config to absolute path
if [[ "$NEOX_CONFIG" != /* ]]; then
    NEOX_CONFIG="$REPO_DIR/$NEOX_CONFIG"
fi

# Derived paths
NEOX_CKPT_DIR="/projects/a5k/public/checkpoints_${USER}/warm_start_neox/$(basename "$SFT_OUTPUT_DIR")"
DATA_PREFIX="/projects/a5k/public/data_${USER}/warm_start_neox/$(basename "$SFT_DATASET" .jsonl)"
TOKENIZER_PATH="$BASE_MODEL"

echo "===== NeoX Warm-Start Pipeline ====="
echo "Config:        $CONFIG"
echo "Base model:    $BASE_MODEL"
echo "SFT dataset:   $SFT_DATASET"
echo "SFT output:    $SFT_OUTPUT_DIR"
echo "NeoX ckpt:     $NEOX_CKPT_DIR"
echo "NeoX config:   $NEOX_CONFIG"
echo "NeoX nodes:    $NEOX_NODES"
echo "NeoX TP:       $NEOX_TP"
echo "Data prefix:   $DATA_PREFIX"
echo "GRPO nodes:    $GRPO_NODES"
echo "====================================="

GRPO_DEPENDS=""

# --- Phase 0: Preprocess SFT data ---
BIN_FILE="${DATA_PREFIX}_messages_document.bin"
if [[ -f "$BIN_FILE" ]]; then
    echo ""
    echo "Phase 0: CACHED — preprocessed data exists at ${DATA_PREFIX}"
else
    echo ""
    echo "Submitting Phase 0: Preprocess SFT data..."
    PREPROCESS_OUTPUT=$(isambard_sbatch --nodes=1 configs/isambard/run_on_compute.sbatch \
        python "$REPO_DIR/warm-start-neox/preprocess_sft_data.py" \
            --input "$SFT_DATASET" \
            --output-prefix "$DATA_PREFIX" \
            --tokenizer-path "$TOKENIZER_PATH" 2>&1)
    echo "$PREPROCESS_OUTPUT"
    PREPROCESS_JOB=$(echo "$PREPROCESS_OUTPUT" | grep -oP '\d+' | tail -1)
    if [[ -z "$PREPROCESS_JOB" ]]; then
        echo "ERROR: Failed to extract preprocess job ID" >&2
        exit 1
    fi
    echo "Preprocess job ID: $PREPROCESS_JOB"
    GRPO_DEPENDS="$PREPROCESS_JOB"
fi

# --- Phase 0.5: Convert HF -> NeoX ---
NEOX_CKPT_LATEST="$NEOX_CKPT_DIR/latest"
if [[ -f "$NEOX_CKPT_LATEST" ]]; then
    echo ""
    echo "Phase 0.5: CACHED — NeoX checkpoint exists at $NEOX_CKPT_DIR"
else
    echo ""
    echo "Submitting Phase 0.5: Convert HF -> NeoX (TP=$NEOX_TP)..."
    CONVERT_DEPS=""
    if [[ -n "$GRPO_DEPENDS" ]]; then
        CONVERT_DEPS="--dependency=afterok:$GRPO_DEPENDS"
    fi
    # Request enough GPUs for the model to fit in memory during conversion
    CONVERT_OUTPUT=$(isambard_sbatch --nodes=1 --gpus=1 $CONVERT_DEPS \
        configs/isambard/run_on_compute.sbatch \
        python "$REPO_DIR/warm-start-neox/convert_hf_olmo_to_neox.py" \
            --hf-model "$BASE_MODEL" \
            --output-dir "$NEOX_CKPT_DIR" \
            --tp "$NEOX_TP" \
            --save-tokenizer 2>&1)
    echo "$CONVERT_OUTPUT"
    CONVERT_JOB=$(echo "$CONVERT_OUTPUT" | grep -oP '\d+' | tail -1)
    if [[ -z "$CONVERT_JOB" ]]; then
        echo "ERROR: Failed to extract HF->NeoX conversion job ID" >&2
        exit 1
    fi
    echo "HF->NeoX conversion job ID: $CONVERT_JOB"
    GRPO_DEPENDS="$CONVERT_JOB"
fi

# --- Phase 1: NeoX SFT training ---
echo ""
echo "Submitting Phase 1: NeoX SFT ($NEOX_NODES node(s), TP=$NEOX_TP)..."
SFT_DEPS=""
if [[ -n "$GRPO_DEPENDS" ]]; then
    SFT_DEPS="--dependency=afterok:$GRPO_DEPENDS"
fi
SFT_OUTPUT=$(isambard_sbatch --nodes="$NEOX_NODES" $SFT_DEPS \
    "$REPO_DIR/warm-start-neox/sft_neox.sbatch" "$NEOX_CONFIG" 2>&1)
echo "$SFT_OUTPUT"
SFT_JOB=$(echo "$SFT_OUTPUT" | grep -oP '\d+' | tail -1)
if [[ -z "$SFT_JOB" ]]; then
    echo "ERROR: Failed to extract SFT job ID" >&2
    exit 1
fi
echo "NeoX SFT job ID: $SFT_JOB"

# --- Phase 1.5: Convert NeoX -> HF ---
echo ""
echo "Submitting Phase 1.5: Convert NeoX -> HF..."
# Find the latest checkpoint in the NeoX save dir
NEOX_SAVE_DIR=$(python3 -c "
import json, sys
# Read save dir from NeoX config (it's JSON-with-comments, parse carefully)
import re
with open(sys.argv[1]) as f:
    content = f.read()
# Strip comments
content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
import json
cfg = json.loads(content)
print(cfg.get('save', ''))
" "$NEOX_CONFIG" 2>/dev/null || echo "$NEOX_CKPT_DIR")

CONVERT_BACK_OUTPUT=$(isambard_sbatch --nodes=1 --dependency=afterok:"$SFT_JOB" \
    configs/isambard/run_on_compute.sbatch \
    bash -c "
        source $REPO_DIR/.venv/bin/activate
        # Find latest checkpoint
        SAVE_DIR='$NEOX_SAVE_DIR'
        LATEST_FILE=\"\$SAVE_DIR/latest\"
        if [ -f \"\$LATEST_FILE\" ]; then
            STEP_DIR=\"\$SAVE_DIR/\$(cat \$LATEST_FILE)\"
        else
            STEP_DIR=\$(ls -d \$SAVE_DIR/global_step* 2>/dev/null | sort -t'p' -k2 -n | tail -1)
        fi
        if [ -z \"\$STEP_DIR\" ] || [ ! -d \"\$STEP_DIR\" ]; then
            echo 'ERROR: No NeoX checkpoint found in $NEOX_SAVE_DIR'
            exit 1
        fi
        echo \"Converting NeoX checkpoint: \$STEP_DIR\"
        python $REPO_DIR/warm-start-neox/convert_neox_olmo_to_hf.py \
            --input-dir \"\$STEP_DIR\" \
            --hf-model \"$BASE_MODEL\" \
            --output-dir \"$SFT_OUTPUT_DIR\" \
            --precision bfloat16
    " 2>&1)
echo "$CONVERT_BACK_OUTPUT"
CONVERT_BACK_JOB=$(echo "$CONVERT_BACK_OUTPUT" | grep -oP '\d+' | tail -1)
if [[ -z "$CONVERT_BACK_JOB" ]]; then
    echo "ERROR: Failed to extract NeoX->HF conversion job ID" >&2
    exit 1
fi
echo "NeoX->HF conversion job ID: $CONVERT_BACK_JOB"

# --- Phase 2: GRPO ---
echo ""
echo "Submitting Phase 2: GRPO ($GRPO_NODES nodes, depends on $CONVERT_BACK_JOB)..."
isambard_sbatch --nodes="$GRPO_NODES" --dependency=afterok:"$CONVERT_BACK_JOB" \
    configs/isambard/grpo_rlzero.sbatch "$CONFIG" \
    --model_name_or_path="$SFT_OUTPUT_DIR"

echo ""
echo "===== Pipeline Submitted ====="
echo "Phase 0  (preprocess): ${PREPROCESS_JOB:-cached}"
echo "Phase 0.5 (HF->NeoX): ${CONVERT_JOB:-cached}"
echo "Phase 1  (NeoX SFT):  $SFT_JOB"
echo "Phase 1.5 (NeoX->HF): $CONVERT_BACK_JOB"
echo "Phase 2  (GRPO):      (see output above)"
echo ""
echo "Monitor SFT: tail -f /projects/a5k/public/logs_${USER}/open-instruct/neox-sft-${SFT_JOB}.out"
