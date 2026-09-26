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
