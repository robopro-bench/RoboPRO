#!/bin/bash
#SBATCH --partition=h100-private
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
# NOTE: set --chdir and --output to your local checkout, e.g.
#   #SBATCH --chdir=/path/to/RoboPRO
#   #SBATCH --output=/path/to/RoboPRO/logs/%x_%j.out

# Args: <task_name> <task_config> <train_config> <model_name> <ckpt_id> <seed> <test_num>
TASK_NAME="${1:?}"
TASK_CONFIG="${2:?}"
TRAIN_CONFIG="${3:?}"
MODEL_NAME="${4:?}"
CKPT_ID="${5:?}"
SEED="${6:-0}"
TEST_NUM="${7:-20}"

echo "Task: $TASK_NAME"
echo "Config: $TASK_CONFIG"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

source set_env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export PYTHONUNBUFFERED=1

PYTHON="${PI05_PYTHON:-$(command -v python)}"
# To pin a specific conda env: export PI05_PYTHON=/path/to/miniconda3/envs/pi05/bin/python
export PYTHONPATH="${WORKSPACE_ROOT:-$SIM_ROOT/..}/policy/pi05/src:$PYTHONPATH"

# === Phase 1: pre-collect 20 eval seeds (skipped if file already has 20) ===
echo "=== precollecting eval seeds for $TASK_NAME / $TASK_CONFIG ==="
$PYTHON "${WORKSPACE_ROOT}/collect/precollect_eval_seeds.py" "$TASK_NAME" "$TASK_CONFIG"
PRECOLLECT_RC=$?
echo "precollect exit code: $PRECOLLECT_RC"
echo ""

if [ "$PRECOLLECT_RC" -ne 0 ]; then
    echo "precollect failed, aborting eval"
    exit 1
fi

# Verify seed file exists and has 20 seeds
SEED_FILE="${BENCH_ROOT}/eval_seeds/${TASK_NAME}/${TASK_CONFIG}.txt"
if [ ! -f "$SEED_FILE" ]; then
    echo "ERROR: seed file $SEED_FILE not created"
    exit 1
fi
N_SEEDS=$(wc -w < "$SEED_FILE")
echo "have $N_SEEDS seeds in $SEED_FILE"
if [ "$N_SEEDS" -lt 20 ]; then
    echo "WARNING: only got $N_SEEDS seeds (target 20). Proceeding anyway."
fi
echo ""

# === Phase 2: eval ===
echo "=== running eval ==="
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
