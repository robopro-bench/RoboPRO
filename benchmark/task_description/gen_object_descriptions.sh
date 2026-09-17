#!/bin/bash
# LLM object descriptions → benchmark/bench_description/objects_description/
# Usage (from repo root): bash benchmark/task_description/gen_object_descriptions.sh <object_name> [object_id]
# Requires AZURE_API_KEY.

object_name=${1}
object_id=${2}

if [ -z "$object_name" ]; then
    echo "Error: object_name is required."
    echo "Usage: $0 <object_name> [object_id]"
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

if [ -z "$object_id" ]; then
    "${PYTHON}" "${SCRIPT_DIR}/generate_object_description.py" "$object_name"
else
    "${PYTHON}" "${SCRIPT_DIR}/generate_object_description.py" "$object_name" --index "$object_id"
fi
