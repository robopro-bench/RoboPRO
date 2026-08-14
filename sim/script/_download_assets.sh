#!/usr/bin/env bash
# Forwards to the repo-root downloader (assets/ at the workspace root).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -x "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"
else
    PY="$(command -v python3 || command -v python || true)"
fi
[[ -n "${PY}" ]] || { echo "ERROR: python not found" >&2; exit 1; }
exec "$PY" "$ROOT/scripts/install/download_assets.py" "$@"
