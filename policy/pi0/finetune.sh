#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
train_config_name=$1
model_name=$2
gpu_use=$3

export CUDA_VISIBLE_DEVICES=$gpu_use
echo $CUDA_VISIBLE_DEVICES
cd "$DIR"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --project "$DIR/openpi" python "$DIR/train.py" \
    "$train_config_name" --exp-name="$model_name" --overwrite
