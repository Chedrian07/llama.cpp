#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-mcd-metal}"
BIN="${BIN:-$BUILD_DIR/bin/llama-bench}"
MODEL="${MODEL:?set MODEL=/path/to/model.gguf}"
RPC="${RPC:-169.254.21.157:50052}"
SPLIT_MODE="${SPLIT_MODE:-layer}"
CTX="${CTX:-32768}"
NGL="${NGL:-all}"
PROMPT_TOKENS="${PROMPT_TOKENS:-8192}"
GEN_TOKENS="${GEN_TOKENS:-128}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

exec "$BIN" \
  --model "$MODEL" \
  --ctx-size "$CTX" \
  --n-gpu-layers "$NGL" \
  --split-mode "$SPLIT_MODE" \
  --rpc "$RPC" \
  --n-prompt "$PROMPT_TOKENS" \
  --n-gen "$GEN_TOKENS" \
  $EXTRA_ARGS
