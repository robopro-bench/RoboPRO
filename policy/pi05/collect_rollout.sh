#!/bin/bash
# Dual-env rollout data collection for pi05:
#   - server: pi05/openpi/.venv (uv-managed, openpi+jax)  → server_gpu
#   - client: RoboPRO sim env (sapien)                    → client_gpu
# Usage:
#   bash policy/pi05/collect_rollout.sh <task> <task_config> <train_config> <model_name> <checkpoint_id> <seed> <server_gpu>[:<client_gpu>]
# If client_gpu is omitted, both processes share server_gpu.
#
# Env vars (override defaults):
#   COLLECT_NUM              — total episodes to save (default 100)
#   COLLECT_START_SEED       — starting seed override
#   ACTION_NOISE_VAR         — Gaussian action noise variance for rollout diversity (default 0)
#   COLLECT_FIXED_SEED       — if set, skip expert check

set -euo pipefail

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
checkpoint_id=${5:-20000}
seed=${6:-0}
gpu_spec=${7:-0}

if [[ "${gpu_spec}" == *":"* ]]; then
    server_gpu="${gpu_spec%%:*}"
    client_gpu="${gpu_spec##*:}"
else
    server_gpu="${gpu_spec}"
    client_gpu="${gpu_spec}"
fi
echo -e "\033[33mserver gpu: ${server_gpu}, client gpu: ${client_gpu}\033[0m"

# ── Env vars (with defaults) ──────────────────────────────────────────────
export ACTION_NOISE_VAR="${ACTION_NOISE_VAR:-0.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export SIM_ROOT="${SIM_ROOT:-${WORKSPACE_ROOT}/sim}"
export BENCH_ROOT="${BENCH_ROOT:-${WORKSPACE_ROOT}/benchmark}"
export ASSETS_ROOT="${ASSETS_ROOT:-${WORKSPACE_ROOT}/assets}"
export DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/data}"
export POLICY_ROOT="${POLICY_ROOT:-${WORKSPACE_ROOT}/policy}"

export CUDA_VISIBLE_DEVICES=${client_gpu}

cd "${WORKSPACE_ROOT}"

FREE_PORT=$(python3 - << 'EOF'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
EOF
)
echo -e "\033[33mUsing socket port: ${FREE_PORT}\033[0m"

# --- Server (pi05 uv venv) ---
echo -e "\033[32m[server] Activating pi05 openpi .venv\033[0m"
PI05_VENV="$(pwd)/policy/pi05/openpi/.venv"
(
    unset CUDA_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=${server_gpu}
    export PYTHONWARNINGS=ignore::UserWarning
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
    exec "${PI05_VENV}/bin/python" eval/policy_model_server.py \
        --port ${FREE_PORT} \
        --config policy/${policy_name}/deploy_policy.yml \
        --overrides \
        --task_name ${task_name} \
        --task_config ${task_config} \
        --train_config_name ${train_config_name} \
        --model_name ${model_name} \
        --checkpoint_id ${checkpoint_id} \
        --ckpt_setting ${model_name} \
        --seed ${seed} \
        --policy_name ${policy_name}
) &
SERVER_PID=$!
trap "echo -e '\033[31m[cleanup] Killing server PID=${SERVER_PID}\033[0m'; kill ${SERVER_PID} 2>/dev/null || true" EXIT

# Resolve the simulator-side interpreter. The RoboPRO sim env is the uv-managed
# repo-root .venv; a bare `python` may resolve to an unrelated conda env that has
# no sapien. Override with SIM_PYTHON if the sim env lives elsewhere.
SIM_PYTHON="${SIM_PYTHON:-${WORKSPACE_ROOT}/.venv/bin/python}"
if [[ ! -x "${SIM_PYTHON}" ]]; then
    SIM_PYTHON="$(command -v python)"
fi
echo -e "\033[34m[client] sim python: ${SIM_PYTHON}\033[0m"

# --- Client (RoboPRO sim env, current process) ---
echo -e "\033[34m[client] Starting collect_rollout_client on port ${FREE_PORT}\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
"${SIM_PYTHON}" "${WORKSPACE_ROOT}/collect/collect_rollout_client.py" \
    --port ${FREE_PORT} \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --checkpoint_id ${checkpoint_id} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name}

echo -e "\033[33m[main] Collection finished\033[0m"
