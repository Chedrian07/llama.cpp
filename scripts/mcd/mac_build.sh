#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-mcd-metal}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"
PATH="/opt/homebrew/bin:/usr/local/bin:/Applications/CMake.app/Contents/bin:$PATH"
export PATH

cd "$ROOT"
cmake -S . -B "$BUILD_DIR" \
  -DGGML_METAL=ON \
  -DGGML_RPC=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" \
  --config Release \
  --target llama-server llama-cli llama-bench rpc-server \
  -j "$JOBS"

echo "build_dir=$BUILD_DIR"
