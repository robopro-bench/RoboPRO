#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Pre-flight: check required system tools before doing any expensive work
# ---------------------------------------------------------------------------
missing=()
command -v uv   >/dev/null 2>&1 || missing+=(uv)
command -v git  >/dev/null 2>&1 || missing+=(git)
command -v nvcc >/dev/null 2>&1 || missing+=(nvcc)
command -v ffmpeg >/dev/null 2>&1 || missing+=(ffmpeg)
if (( ${#missing[@]} )); then
    printf 'ERROR: required tools not found on PATH: %s\n' "${missing[*]}" >&2
    printf 'Install them before running bootstrap.\n' >&2
    printf '  nvcc   → sudo apt install nvidia-cuda-toolkit   (or install from NVIDIA)\n' >&2
    printf '  ffmpeg → sudo apt install ffmpeg\n' >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Auto-detect CUDA_HOME if not already set
# ---------------------------------------------------------------------------
if [[ -z "${CUDA_HOME:-}" ]]; then
    nvcc_path="$(command -v nvcc)"
    # nvcc is typically at <cuda_root>/bin/nvcc
    candidate="$(dirname "$(dirname "$nvcc_path")")"
    if [[ -x "$candidate/bin/nvcc" ]]; then
        export CUDA_HOME="$candidate"
    else
        # apt-installed toolkit: nvcc at /usr/bin/nvcc, toolkit at /usr/lib/nvidia-cuda-toolkit
        if [[ -x /usr/lib/nvidia-cuda-toolkit/bin/nvcc ]]; then
            export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
        else
            export CUDA_HOME="$(dirname "$(dirname "$nvcc_path")")"
        fi
    fi
    echo "Auto-detected CUDA_HOME=${CUDA_HOME}"
fi

# ---------------------------------------------------------------------------
# Create venv and sync deps
# ---------------------------------------------------------------------------
echo "Creating/updating .venv with Python ${PYTHON_VERSION} ..."
uv venv --python "$PYTHON_VERSION"

echo "Syncing Python dependencies from pyproject.toml ..."
uv sync --locked

echo "Installing pip, wheel, ninja (needed by post-install patches) ..."
uv pip install pip wheel ninja

# ---------------------------------------------------------------------------
# Post-install patches
# ---------------------------------------------------------------------------
echo "Running RoboPRO post-install patches ..."
source .venv/bin/activate
cd sim
SKIP_BASE_DEPS=1 bash script/_install.sh

cat <<EOF

uv bootstrap complete.  CUDA_HOME=${CUDA_HOME}

Next steps:
  make download-assets          # ~15 GB from HuggingFace (skip if you already have them)
  make link-assets              # create asset symlinks
  make configure-curobo-assets  # render CuRobo YAML configs
  make patch-curobo-config      # patch CuRobo attached_object entries
  make verify-rollout           # smoke test

Or run them all at once:
  make setup                    # link + configure + patch (after assets are in place)
  make verify-rollout
EOF
