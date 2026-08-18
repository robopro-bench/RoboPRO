task_name=${1}
setting=${2}
max_num=${3}

# Prefer the repo-root uv .venv; a bare `python` may resolve to an unrelated
# conda env without this script's deps. Override with SIM_PYTHON.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_PYTHON="${SIM_PYTHON:-${WORKSPACE_ROOT:-${SCRIPT_DIR}/../..}/.venv/bin/python}"
if [ ! -x "${SIM_PYTHON}" ]; then
    SIM_PYTHON="$(command -v python)"
fi

"${SIM_PYTHON}" utils/generate_episode_instructions.py $task_name $setting $max_num
