# MCD Commands

This file records exact commands used for the Metal-CUDA prefill
disaggregation project. It intentionally separates SSH control from direct
10GbE data plane traffic.

## Repository

```sh
cd /Users/kch3dri4n/tools/llama.cpp
git remote -v
git branch --show-current
git log -1 --oneline
```

## Mac Control

```sh
ssh kch3dri4n@100.92.205.98
cd /Users/kch3dri4n/tools/llama.cpp
```

## Windows Control

```sh
ssh kch3dri4n@100.85.137.105
cd 'C:\Users\-_-kc\Desktop\llama.cpp'
```

## Direct 10GbE

Mac direct IP:

```sh
169.254.150.225
```

Windows direct IP:

```sh
169.254.21.157
```

Performance tests must bind to these addresses. Tailscale `100.x` addresses
are control-only and invalid for benchmark data-plane measurements.

## Mac Build

```sh
cmake -S . -B build-mcd-metal \
  -DGGML_METAL=ON \
  -DGGML_RPC=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-mcd-metal \
  --config Release \
  --target llama-server llama-cli llama-bench rpc-server \
  -j "$(sysctl -n hw.ncpu)"
```

## Windows Build

```powershell
cmake -S . -B build-mcd-cuda `
  -DGGML_CUDA=ON `
  -DGGML_RPC=ON `
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-mcd-cuda `
  --config Release `
  --target llama-server llama-cli llama-bench rpc-server `
  -j $env:NUMBER_OF_PROCESSORS
```

## Single-Slot Servers

Use `CTX=65536` for the measured full-window handoff run. Lower context values
were used only for smaller prompt smoke tests.

Mac:

```sh
MODEL="/Volumes/Back_UP_LLM/models/unsloth/gemma-4-31B-it-qat-GGUF/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf" \
HOST=169.254.150.225 \
PORT=18081 \
CTX=65536 \
NGL=99 \
EXTRA_ARGS="--parallel 1" \
scripts/mcd/mac_run_server.sh
```

Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File C:\Users\-_-kc\Desktop\llama.cpp\scripts\mcd\windows_run_server.ps1 `
  -Model C:\Users\-_-kc\models\unsloth\gemma-4-31B-it-qat-GGUF\gemma-4-31B-it-qat-UD-Q4_K_XL.gguf `
  -HostAddress 169.254.21.157 `
  -Port 18080 `
  -Ctx 65536 `
  -NGpuLayers 99 `
  -ExtraArgs @("--parallel", "1")
```

For the complete current runbook, see `docs/mcd/USAGE.md`.

## Model Inventory

Mac:

```sh
python3 tools/mcd/mcd_model_select.py inventory \
  "$HOME/.lmstudio/models" \
  "/Volumes/Back_UP_LLM/models" \
  > tools/mcd/model_inventory_mac.json
```

Windows:

```powershell
python tools/mcd/mcd_model_select.py inventory `
  'C:\Users\-_-kc\models' `
  > tools/mcd/model_inventory_windows.json
```

Match:

```sh
python3 tools/mcd/mcd_model_select.py match \
  tools/mcd/model_inventory_mac.json \
  tools/mcd/model_inventory_windows.json
```
