#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_name=${1}
setting=${2}
expert_data_num=${3}

python "$DIR/process_data.py" "$task_name" "$setting" "$expert_data_num"
