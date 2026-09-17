#!/bin/bash
# LLM task instruction templates → benchmark/bench_description/task_instructions/
# Usage (from repo root): bash benchmark/task_description/gen_task_instruction_templates.sh <task_name> <instruction_num>
# instruction_num must be divisible by 12. Requires AZURE_API_KEY.

task_name=${1}
instruction_num=${2}

if [ -z "$task_name" ] || [ -z "$instruction_num" ]; then
    echo "Usage: $0 <task_name> <instruction_num>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${SCRIPT_DIR}/../..}"
if [[ -z "${BENCH_ROOT:-}" ]]; then
    # shellcheck source=/dev/null
    source "${WORKSPACE_ROOT}/set_env.sh"
fi
PYTHON="${SIM_PYTHON:-${WORKSPACE_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python)"
fi

"${PYTHON}" "${SCRIPT_DIR}/generate_task_description.py" "$task_name" "$instruction_num"
