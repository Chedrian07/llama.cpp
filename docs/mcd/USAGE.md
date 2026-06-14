# MCD Usage

This document describes how to reproduce the Windows CUDA prefill to Mac Metal
decode handoff benchmark.

The current implementation uses existing llama.cpp server APIs:

1. Windows CUDA performs prompt prefill with `/completion`.
2. Windows saves slot state with `/slots/0?action=save`.
3. The slot file is copied over the direct 10GbE link.
4. Mac Metal restores the slot with `/slots/0?action=restore`.
5. Mac Metal decodes locally with `/completion`.

No llama.cpp C++ patch is required for the measured MVP.

## Hosts

Control plane:

- Mac SSH: `kch3dri4n@100.92.205.98`
- Windows SSH: `kch3dri4n@100.85.137.105`

Data plane:

- Mac direct 10GbE: `169.254.150.225`
- Windows direct 10GbE: `169.254.21.157`

Do not use the `100.x` control addresses for throughput measurements.

## Model

Mac:

```sh
/Volumes/Back_UP_LLM/models/unsloth/gemma-4-31B-it-qat-GGUF/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
```

Windows:

```powershell
C:\Users\-_-kc\models\unsloth\gemma-4-31B-it-qat-GGUF\gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
```

## Build

Mac:

```sh
ssh kch3dri4n@100.92.205.98
cd /Users/kch3dri4n/tools/llama.cpp
scripts/mcd/mac_build.sh
```

Windows:

```powershell
ssh kch3dri4n@100.85.137.105
cd C:\Users\-_-kc\Desktop\llama.cpp
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\mcd\windows_build.ps1
```

## Generate Prompts

Run this on the Mac workspace, then sync the generated prompt files to Windows
if needed.

```sh
python3 tools/mcd/mcd_prompts.py \
  --out results/mcd/prompts \
  --targets 512 2048 8192 16384 32768 52000
```

`prompt_52000.txt` is the near-full 65536-context test for the current model. It
tokenized to 65318-65319 prompt tokens in the measured runs.

## Start Servers

Use one slot for handoff benchmarks. Extra slots allocate more state and make
results harder to interpret.

Mac:

```sh
ssh kch3dri4n@100.92.205.98
cd /Users/kch3dri4n/tools/llama.cpp

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
ssh kch3dri4n@100.85.137.105
cd C:\Users\-_-kc\Desktop\llama.cpp

powershell -NoProfile -ExecutionPolicy Bypass -Command "& 'C:\Users\-_-kc\Desktop\llama.cpp\scripts\mcd\windows_run_server.ps1' -Model 'C:\Users\-_-kc\models\unsloth\gemma-4-31B-it-qat-GGUF\gemma-4-31B-it-qat-UD-Q4_K_XL.gguf' -HostAddress '169.254.21.157' -Port 18080 -Ctx 65536 -NGpuLayers '99' -ExtraArgs @('--parallel','1')"
```

## Mac-Only Baseline

Run on Mac:

```sh
cd /Users/kch3dri4n/tools/llama.cpp

python3 tools/mcd/mcd_bench.py completion \
  --server http://169.254.150.225:18081 \
  --prompt results/mcd/prompts/prompt_52000.txt \
  --n-predict 16 \
  --id-slot 0 \
  --timeout 7200 \
  > results/mcd/mac_server_completion_52000.json
```

## Windows Prefill

Run on Windows:

```powershell
cd C:\Users\-_-kc\Desktop\llama.cpp

python tools\mcd\mcd_bench.py completion `
  --server http://169.254.21.157:18080 `
  --prompt results\mcd\prompts\prompt_52000.txt `
  --n-predict 0 `
  --id-slot 0 `
  --timeout 7200 |
  Out-File -Encoding utf8 results\mcd\windows_prefill_only_52000.json
```

## Save Slot

Run on Windows after prefill:

```powershell
cd C:\Users\-_-kc\Desktop\llama.cpp

python tools\mcd\mcd_bench.py slot `
  --server http://169.254.21.157:18080 `
  --id-slot 0 `
  --action save `
  --filename mcd_slot_52000.bin `
  --timeout 7200 |
  Out-File -Encoding utf8 results\mcd\windows_slot_save_52000.json
```

## Copy Slot Over 10GbE

Start the receiver on Mac:

```sh
cd /Users/kch3dri4n/tools/llama.cpp

python3 tools/mcd/mcd_tcp_copy.py server \
  --bind 169.254.150.225 \
  --port 5202 \
  --out results/mcd/mac_slots \
  > results/mcd/mac_tcp_copy_server_52000.json
```

Start the sender on Windows:

```powershell
cd C:\Users\-_-kc\Desktop\llama.cpp

python tools\mcd\mcd_tcp_copy.py client `
  results\mcd\windows_slots\mcd_slot_52000.bin `
  169.254.150.225 `
  --source 169.254.21.157 `
  --port 5202 `
  --sha256 |
  Out-File -Encoding utf8 results\mcd\windows_to_mac_slot_copy_52000.json
```

The Mac receiver must report `sha256_match: true`.

## Restore And Decode

Run on Mac:

```sh
cd /Users/kch3dri4n/tools/llama.cpp

python3 tools/mcd/mcd_bench.py slot \
  --server http://169.254.150.225:18081 \
  --id-slot 0 \
  --action restore \
  --filename mcd_slot_52000.bin \
  --timeout 7200 \
  > results/mcd/mac_slot_restore_52000.json

python3 tools/mcd/mcd_bench.py completion \
  --server http://169.254.150.225:18081 \
  --prompt results/mcd/prompts/prompt_52000.txt \
  --n-predict 16 \
  --id-slot 0 \
  --timeout 7200 \
  > results/mcd/mac_decode_after_restore_52000.json
```

## Summarize

Run on the machine that has all JSON result files:

```sh
python3 tools/mcd/mcd_report.py handoff \
  --label prompt_52000 \
  --mac-base results/mcd/mac_server_completion_52000.json \
  --windows-prefill results/mcd/windows_prefill_only_52000.json \
  --windows-save results/mcd/windows_slot_save_52000.json \
  --windows-copy results/mcd/windows_to_mac_slot_copy_52000.json \
  --mac-restore results/mcd/mac_slot_restore_52000.json \
  --mac-decode results/mcd/mac_decode_after_restore_52000.json
```

Success criteria:

- `speedup` is greater than the target threshold.
- `restored_cache_tokens` is close to the full prompt length.
- `tail_prompt_tokens` is small. The measured runs show `5`.
- `decode_tokens_per_second` is close to Mac-only decode tok/s.
- The TCP copy receiver reports `sha256_match: true`.

## Cleanup

Mac:

```sh
lsof -tiTCP:18081 -sTCP:LISTEN | xargs -r kill
```

Windows:

```powershell
$p = Get-NetTCPConnection -LocalPort 18080 -State Listen -ErrorAction SilentlyContinue
if ($p) { Stop-Process -Id $p.OwningProcess -Force }
```

## Known Issues

- Mac to Windows HTTP on `169.254.21.157:18080` may time out because of the
  Windows firewall/profile. The measured workflow avoids this by running
  Windows HTTP requests locally through SSH control.
- The direct 10GbE link is still used for the large slot file transfer.
- The model reports `n_ctx_train = 262144`; the measured full-window run uses
  `CTX=65536`, which is the largest verified context in this run.
- Slot files are large. The `prompt_52000.txt` run wrote about 6.19 GB.
