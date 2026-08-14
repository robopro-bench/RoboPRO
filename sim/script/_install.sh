#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
[[ -n "$PYTHON_BIN" ]] || { echo "ERROR: python not found on PATH" >&2; exit 1; }
PIP_CMD=("$PYTHON_BIN" -m pip)

echo "Installing the necessary packages ..."
if [ "${SKIP_BASE_DEPS:-0}" != "1" ]; then
    "${PIP_CMD[@]}" install -r script/requirements.txt

    echo "Installing pytorch3d ..."
    # cd third_party/pytorch3d_simplified
    # pip install -e .
    # cd ../..
    "${PIP_CMD[@]}" install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
fi

echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
SAPIEN_LOCATION=$("$PYTHON_BIN" -m pip show sapien | grep 'Location' | awk '{print $2}')/sapien
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
if grep -q 'with open(urdf_file, "r") as f:' "$URDF_LOADER"; then
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF_LOADER"
    echo "  patched $URDF_LOADER"
else
    echo "  sapien urdf_loader already patched (skipping)"
fi

echo "Adjusting code in mplib/planner.py ..."
MPLIB_LOCATION=$("$PYTHON_BIN" -m pip show mplib | grep 'Location' | awk '{print $2}')/mplib
PLANNER=$MPLIB_LOCATION/planner.py
if grep -q 'if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:' "$PLANNER"; then
    sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"
    echo "  patched $PLANNER"
else
    echo "  mplib planner already patched (skipping)"
fi

echo "Installing Curobo ..."
cd envs
CUROBO_TAG="v0.7.8"
if [ ! -d curobo ] || ! git -C curobo describe --tags --exact-match 2>/dev/null | grep -q "^${CUROBO_TAG}$"; then
    rm -rf curobo
    git clone --branch "${CUROBO_TAG}" --depth 1 https://github.com/NVlabs/curobo.git
fi
cd curobo
"${PIP_CMD[@]}" install -e . --no-build-isolation
"${PIP_CMD[@]}" install warp-lang==1.12.0 setuptools==69.5.1
cd ../..

echo "Installation basic environment complete!"
echo -e "You need to:"
echo -e "    1. \033[34m\033[1m(Important!)\033[0m Download assets: make download-assets"
echo -e "       (or: python scripts/install/download_assets.py)"
echo -e "    2. make configure-curobo-assets && make patch-curobo-config"
echo "See README.md or docs/install.html for more instructions."
