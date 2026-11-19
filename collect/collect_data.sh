#!/bin/bash

COLLECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${SIM_ROOT:-}" ]]; then
    # shellcheck source=/dev/null
    source "${COLLECT_DIR}/../set_env.sh"
fi

# Resolve the simulator interpreter (repo-root uv .venv); a bare `python` may
# resolve to an unrelated conda env without sapien. Override via SIM_PYTHON.
SIM_PYTHON="${SIM_PYTHON:-${WORKSPACE_ROOT:-${COLLECT_DIR}/..}/.venv/bin/python}"
if [[ ! -x "${SIM_PYTHON}" ]]; then
    SIM_PYTHON="$(command -v python)"
fi

task_name=${1}
task_config=${2}
gpu_id=${3:-0}

if [[ "${gpu_id}" == *,* ]]; then
    # Multi-GPU: dynamic per-seed dispatch across the listed GPUs (e.g. 0,1).
    # episode_num is read from the task config, so the amount collected matches
    # the single-GPU path. All episodes land in one run dir (episode0..N-1).
    PYTHONWARNINGS=ignore::UserWarning \
    "${SIM_PYTHON}" -u "${COLLECT_DIR}/collect_parallel.py" "${task_name}" "${task_config}" --gpus "${gpu_id}"
else
    # Single-GPU: stock sequential collection.
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    PYTHONWARNINGS=ignore::UserWarning \
    "${SIM_PYTHON}" -u "${COLLECT_DIR}/collect_data.py" $task_name $task_config
    rm -rf "${DATA_ROOT}/${task_name}/${task_config}/.cache"
fi
