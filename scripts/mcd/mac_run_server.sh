#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-mcd-metal}"
BIN="${BIN:-$BUILD_DIR/bin/llama-server}"
MODEL="${MODEL:?set MODEL=/path/to/model.gguf}"
HOST="${HOST:-169.254.150.225}"
PORT="${PORT:-18081}"
CTX="${CTX:-32768}"
NGL="${NGL:-all}"
SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-$ROOT/results/mcd/mac_slots}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$SLOT_SAVE_PATH"

exec "$BIN" \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --ctx-size "$CTX" \
  --n-gpu-layers "$NGL" \
  --slot-save-path "$SLOT_SAVE_PATH" \
  --no-warmup \
  $EXTRA_ARGS
