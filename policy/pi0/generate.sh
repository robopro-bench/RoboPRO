#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir=${1}
repo_id=${2}
# LeRobot conversion lives in the openpi submodule, not this glue dir.
uv run --project "$DIR/openpi" python "$DIR/openpi/examples/aloha_real/convert_aloha_data_to_lerobot.py" \
    --raw-dir "$data_dir" --repo-id "$repo_id" --no-push-to-hub
