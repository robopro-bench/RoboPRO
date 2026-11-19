#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
# NOTE: set --partition/--chdir/--output to your cluster and checkout, e.g.
#   #SBATCH --chdir=/path/to/RoboPRO
#   #SBATCH --output=/path/to/RoboPRO/logs/%x_%j.out

# Args: <task_name> <task_config> <train_config> <model_name> <ckpt_id> <seed> <test_num>
TASK_NAME="${1:?Usage: sbatch slurm_eval_bench.sh <task> <config> <train_config> <model_name> <ckpt_id> <seed> <test_num>}"
TASK_CONFIG="${2:?}"
TRAIN_CONFIG="${3:?}"
MODEL_NAME="${4:?}"
CKPT_ID="${5:?}"
SEED="${6:-0}"
TEST_NUM="${7:-10}"

echo "Task: $TASK_NAME"
echo "Config: $TASK_CONFIG"
echo "Train Config: $TRAIN_CONFIG"
echo "Model: $MODEL_NAME"
echo "Checkpoint: $CKPT_ID"
echo "Seed: $SEED"
echo "Test Num: $TEST_NUM"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Environment setup
source set_env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export PYTHONUNBUFFERED=1

# Single-process eval: use PI05_PYTHON if you have a mixed env (openpi+sapien).
# Otherwise the dual-env wrappers in policy/pi05/ are the supported path.
_WS="${WORKSPACE_ROOT:-$SIM_ROOT/..}"
PYTHON="${PI05_PYTHON:-$(command -v python)}"
export PYTHONPATH="${_WS}/policy/pi05/openpi/src:${PYTHONPATH:-}"

$PYTHON "${WORKSPACE_ROOT}/eval/eval_policy.py" \
    --config policy/pi05/deploy_policy.yml \
    --overrides \
    --task_name "$TASK_NAME" \
    --task_config "$TASK_CONFIG" \
    --train_config_name "$TRAIN_CONFIG" \
    --model_name "$MODEL_NAME" \
    --checkpoint_id "$CKPT_ID" \
    --policy_name pi05 \
    --instruction_type seen \
    --seed "$SEED" \
    --ckpt_setting "${TRAIN_CONFIG}_${MODEL_NAME}_${CKPT_ID}" \
    --test_num "$TEST_NUM"

echo "Eval done, exit code: $?"
