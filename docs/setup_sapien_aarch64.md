# Building SAPIEN 3.0.0b1 from source on aarch64 (GB10 / DGX Spark)

PyPI ships no aarch64 wheel for `sapien==3.0.0b1`, so on ARM machines the
`pip install -r sim/script/requirements.txt` step fails to resolve sapien. Build the wheel
from source with the script below, then re-run the requirements install.

Set these two variables before running:

```bash
THIRD_PARTY_DIR=/path/to/third_party    # where the SAPIEN source tree will live
CUDA_12=/usr/local/cuda-12              # your CUDA 12.x install root
```

```bash
# ── SAPIEN 3.0.0b1 (build from source for aarch64) ───────────────────────────
echo "==> Building SAPIEN 3.0.0b1 from source (aarch64)..."
SAPIEN_SRC="${THIRD_PARTY_DIR}/sapien_build"
rm -rf "${SAPIEN_SRC}"
git clone --branch 3.0.0b1 --depth 1 https://github.com/haosulab/SAPIEN.git "${SAPIEN_SRC}"
cd "${SAPIEN_SRC}"
git submodule update --init --recursive

# Build SAPIEN in a subshell to isolate env var exports from later sections
(
  export CUDA_PATH="${CUDA_12}"
  export CUDACXX="${CUDA_12}/bin/nvcc"
  export PATH="${CUDA_12}/bin:${PATH}"
  export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"
  export CUDAARCHS="120"
  export CC=/usr/bin/gcc-11
  export CXX=/usr/bin/g++-11
  export CUDAHOSTCXX=/usr/bin/g++-11
  export CFLAGS="-w"
  export CXXFLAGS="-w"
  export CUDAFLAGS="-w -ccbin /usr/bin/g++-11"
  unset CPLUS_INCLUDE_PATH C_INCLUDE_PATH
  SAPIEN_BUILD_DIR="sapien_build"

  # Patch hardcoded CUDA architectures
  sed -i 's/CUDA_ARCHITECTURES "60;61;70;75;80;86"/CUDA_ARCHITECTURES "120"/g' 3rd_party/simsense/CMakeLists.txt
  sed -i 's/CUDA_ARCHITECTURES "60;61;70;75;80;86"/CUDA_ARCHITECTURES "120"/g' CMakeLists.txt

  # Patch pybind11 refs (smart_holder branch was archived)
  for f in cmake/pybind11.cmake pinocchio/cmake/pybind11.cmake; do
    if [[ -f "$f" ]]; then
      sed -i 's/GIT_TAG smart_holder$/GIT_TAG archive\/smart_holder/' "$f"
    fi
  done

  # First build: fetch dependencies (may fail, that's OK)
  if [[ ! -d "${SAPIEN_BUILD_DIR}/_sapien_deps/ktx-src" ]]; then
    echo "==> Running cmake configure to fetch dependencies..."
    python setup.py bdist_wheel 2>&1 | tee "${SAPIEN_SRC}/sapien_build_log.txt" || true
  fi

  # Patch pybind11 again (SAPIEN's own copy, fetched during first build)
  for f in cmake/pybind11.cmake pinocchio/cmake/pybind11.cmake; do
    if [[ -f "$f" ]]; then
      sed -i 's/GIT_TAG smart_holder$/GIT_TAG archive\/smart_holder/' "$f"
    fi
  done

  # Patch ASTC encoder NEON intrinsics for GCC on aarch64
  NEON_FILE="${SAPIEN_BUILD_DIR}/_sapien_deps/ktx-src/lib/astc-encoder/Source/astcenc_vecmathlib_neon_4.h"
  if [[ -f "${NEON_FILE}" ]]; then
    echo "==> Patching ASTC NEON intrinsics for aarch64/GCC..."
    sed -i '/uint32_t lane/,/}/s/return vgetq_lane_s32(m, l);/return vgetq_lane_u32(m, l);/' "${NEON_FILE}"
    sed -i 's/int8x16_t table { t0\.m };/int8x16_t table = vreinterpretq_s8_s32(t0.m);/' "${NEON_FILE}"
    sed -i 's/int8x16x2_t table { t0\.m, t1\.m };/int8x16x2_t table { vreinterpretq_s8_s32(t0.m), vreinterpretq_s8_s32(t1.m) };/' "${NEON_FILE}"
    sed -i 's/int8x16x4_t table { t0\.m, t1\.m, t2\.m, t3\.m };/int8x16x4_t table { vreinterpretq_s8_s32(t0.m), vreinterpretq_s8_s32(t1.m), vreinterpretq_s8_s32(t2.m), vreinterpretq_s8_s32(t3.m) };/' "${NEON_FILE}"
    sed -i 's/\tint8x16_t idx_bytes = vreinterpretq_u8_s32(idx_masked);/\tuint8x16_t idx_bytes = vreinterpretq_u8_s32(idx_masked);/g' "${NEON_FILE}"
    sed -i 's/return vint4(vqtbl1q_s8(table, idx_bytes));/return vint4(vreinterpretq_s32_s8(vqtbl1q_s8(table, idx_bytes)));/' "${NEON_FILE}"
    sed -i 's/return vint4(vqtbl2q_s8(table, idx_bytes));/return vint4(vreinterpretq_s32_s8(vqtbl2q_s8(table, idx_bytes)));/' "${NEON_FILE}"
    sed -i 's/return vint4(vqtbl4q_s8(table, idx_bytes));/return vint4(vreinterpretq_s32_s8(vqtbl4q_s8(table, idx_bytes)));/' "${NEON_FILE}"
  fi

  # Patch OIDN to add sm_120 CUDA support (GB10 is sm_121, PTX-compatible with sm_120)
  OIDN_CUDA_CMAKE="${SAPIEN_BUILD_DIR}/_sapien_deps/oidn-src/devices/cuda/CMakeLists.txt"
  if [[ -f "${OIDN_CUDA_CMAKE}" ]] && ! grep -q SM120 "${OIDN_CUDA_CMAKE}"; then
    echo "==> Patching OIDN for sm_120 (GB10)..."
    sed -i '/^set(OIDN_NVCC_SM90_FLAGS/a\set(OIDN_NVCC_SM120_FLAGS "-gencode arch=compute_120,code=sm_120")' "${OIDN_CUDA_CMAKE}"
    sed -i 's/\(SM90_FLAGS}\)"/\1 ${OIDN_NVCC_SM120_FLAGS}"/' "${OIDN_CUDA_CMAKE}"
  fi

  # Patch OIDN maxSMArch to support sm_120+ (default 99 is too low for GB10)
  OIDN_CUDA_DEVICE_H="${SAPIEN_BUILD_DIR}/_sapien_deps/oidn-src/devices/cuda/cuda_device.h"
  if [[ -f "${OIDN_CUDA_DEVICE_H}" ]] && ! grep -q 'maxSMArch = 121' "${OIDN_CUDA_DEVICE_H}"; then
    echo "==> Patching OIDN maxSMArch 99 -> 121 for GB10..."
    sed -i 's/static constexpr int maxSMArch = 99;/static constexpr int maxSMArch = 121;/' "${OIDN_CUDA_DEVICE_H}"
  fi

  # Replace x86_64 PhysX5 precompiled libs with aarch64 version
  PHYSX_VERSION="105.1-physx-5.3.1.patch0"
  PHYSX_DIR="${SAPIEN_BUILD_DIR}/_sapien_deps/physx5-src"
  if [[ -d "${PHYSX_DIR}" ]]; then
    echo "==> Replacing PhysX5 libs with aarch64 build..."
    curl -sL "https://github.com/sapien-sim/physx-precompiled/releases/download/${PHYSX_VERSION}/linux-aarch64-release.zip" -o "${SAPIEN_SRC}/physx5-aarch64.zip"
    rm -rf "${SAPIEN_SRC}/physx5-aarch64"
    unzip -q "${SAPIEN_SRC}/physx5-aarch64.zip" -d "${SAPIEN_SRC}/physx5-aarch64"
    rm -rf "${PHYSX_DIR}/bin/linux.clang"
    mkdir -p "${PHYSX_DIR}/bin/linux.clang"
    ln -s "${SAPIEN_SRC}/physx5-aarch64/bin/linux.aarch64/release" "${PHYSX_DIR}/bin/linux.clang/release"
  fi

  # Clean OIDN build cache (must rebuild with new arch) and rebuild
  rm -rf "${SAPIEN_BUILD_DIR}/_sapien_build"
  rm -rf "${SAPIEN_BUILD_DIR}/_sapien_deps/oidn-build"
  rm -rf dist/
  python setup.py bdist_wheel > "${SAPIEN_SRC}/sapien_build_log.txt" 2>&1 || {
    echo "==> SAPIEN build failed. Errors:"
    grep -E "^.*error:" "${SAPIEN_SRC}/sapien_build_log.txt" | grep -iv "warning\|hmderrors\|PxError\|PxDefault\|codecvt_error" | head -20
    echo "    Full log: ${SAPIEN_SRC}/sapien_build_log.txt"
    exit 1
  }
  echo "==> SAPIEN build succeeded."
)
# Install the built wheel outside the subshell
pip install "${SAPIEN_SRC}/dist"/sapien-*.whl
```

If the build fails, the full log is at `${SAPIEN_SRC}/sapien_build_log.txt`.
