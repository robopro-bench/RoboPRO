#!/bin/bash
# Forwards to collect/generate_episode_instructions.py (moved out of sim/).
task_name=${1}
setting=${2}
max_num=${3}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${SCRIPT_DIR}/../..}"
SIM_PYTHON="${SIM_PYTHON:-${WORKSPACE_ROOT}/.venv/bin/python}"
if [ ! -x "${SIM_PYTHON}" ]; then
    SIM_PYTHON="$(command -v python)"
fi

"${SIM_PYTHON}" "${WORKSPACE_ROOT}/collect/generate_episode_instructions.py" \
    $task_name $setting $max_num
