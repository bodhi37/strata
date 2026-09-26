#!/bin/bash
# Build the Strata engine (text-only) for RTX 4070 SUPER (sm_89) using the local CUDA 13.3 toolkit.
set -euo pipefail
ROOT="/home/bodhi/models/strata"
CUDA="/home/bodhi/deps/cuda-13.3/opt/cuda"
export CUDA_PATH="$CUDA"
export PATH="$CUDA/bin:$PATH"
export CUDACXX="$CUDA/bin/nvcc"
cd "$ROOT"
echo "[build] cmake configure $(date -Is)"
cmake -G Ninja -S "$ROOT" -B "$ROOT/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_WERROR=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DSTRATA_GGML_DIR=/home/bodhi/models/strata/third_party/llama.cpp -DSTRATA_NATIVE_EXPERTS=ON 2>&1 | tail -30
echo "[build] compiling $(date -Is)"
cmake --build "$ROOT/build" --target strata -j 8 2>&1 | tail -60
mkdir -p "$ROOT/engine"
cp -f "$ROOT/build/strata" "$ROOT/engine/strata"
chmod +x "$ROOT/engine/strata"
echo "[build] done $(date -Is): $(ls -l "$ROOT/engine/strata")"
# R7: cap_ipc_lock lets pin_hot actually mlock the host tier; `cp` clears file xattrs, so re-apply.
# R8: this is not optional - a build WITHOUT this step leaves the tier reclaimable, which measurably
# collapses decode under memory pressure ("mlock FAILED" in the startup line is the tell).
if command -v setcap >/dev/null 2>&1; then
  if ! setcap cap_ipc_lock,cap_sys_nice+ep "$ROOT/engine/strata" 2>/dev/null; then
    echo "WARNING: could not setcap engine/strata - the hot tier will NOT be mlocked." >&2
  fi
fi
getcap "$ROOT/engine/strata" || true
