#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-mcd-metal}"
BIN="${BIN:-$BUILD_DIR/bin/llama-bench}"
MODEL="${MODEL:?set MODEL=/path/to/model.gguf}"
OUT="${OUT:-$ROOT/results/mcd/mac_llama_bench.json}"
PROMPT="${PROMPT:-512,2048}"
GEN="${GEN:-64}"
REPETITIONS="${REPETITIONS:-1}"
NGL="${NGL:-99}"
DEVICE="${DEVICE:-MTL0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$(dirname "$OUT")"

"$BIN" \
  -m "$MODEL" \
  -p "$PROMPT" \
  -n "$GEN" \
  -r "$REPETITIONS" \
  -o json \
  --no-warmup \
  -ngl "$NGL" \
  -dev "$DEVICE" \
  $EXTRA_ARGS > "$OUT" 2>&1
