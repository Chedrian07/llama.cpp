# Metal-CUDA Prefill Disaggregation Plan

Project: llama.cpp Metal-CUDA Prefill Disaggregation over 10GbE

Repository: `Chedrian07/llama.cpp-mcd-prefill`
Branch: `sj/mcd-prefill-disagg-20260614`

## Mission

Measure whether Windows CUDA can reduce long-context TTFT by performing prompt
prefill, saving binary slot/cache/state, transferring it over direct 10GbE, and
letting Mac Studio Metal restore and decode locally.

Decode tok/s improvement is optional. The required win is TTFT reduction versus
Mac single-node Metal.

## Stage Gates

| Stage | Status | Notes |
| --- | --- | --- |
| 0. Repo setup | Done | Fork renamed, branch created, docs/tools/scripts added locally. No push/PR. |
| 1. Direct 10GbE proof | Done | `169.254.150.225 <-> 169.254.21.157` measured at about 9.3-9.5 Gbps. |
| 2. Model discovery | Done | GGUF only. Initial common model is `gemma-4-31B-it-qat-UD-Q4_K_XL.gguf`. |
| 3. Minimal builds | Done | Mac Metal+RPC and Windows CUDA+RPC builds completed. |
| 4. Local baselines | Done | llama-bench and server baselines complete through `prompt_52000.txt` (65319 actual prompt tokens). |
| 5. RPC baseline | Pending | Existing llama.cpp RPC split only as baseline. |
| 6. Prefill handoff MVP | Done | Handoff passed through near-full 65536-context testing using slot save/copy/restore. |
| 7. Minimal patch if needed | Pending | Only after proving exact missing capability. |
| 8. Design note | Done | Remote CUDA KV/attention backend documented as future work. |
| 9. Final report | Done | Results and usage docs written under `docs/mcd/`. |

## Working Assumptions

- SSH control may use Tailscale addresses.
- Benchmark data plane must use the direct 10GbE link.
- The current root `AGENTS.md` contribution restrictions remain active.
- This is a private fork experiment unless the user later chooses otherwise.
- Commits, pushes, and PR creation require explicit user approval.

## Current Facts

- GitHub owner: `Chedrian07`
- Renamed fork: `https://github.com/Chedrian07/llama.cpp-mcd-prefill`
- Local path: `/Users/kch3dri4n/tools/llama.cpp`
- Current branch: `sj/mcd-prefill-disagg-20260614`
- Current upstream base: `6e14286ed cli : fix not copying preserved tokens (#24258)`
- Mac Studio direct candidate: `169.254.150.225/16` on `en0`
- Windows direct candidate: `169.254.21.157/16` on "Ethernet 4"
- Mac -> Windows direct TCP throughput: about 9.33 Gbps
- Windows -> Mac direct TCP throughput: about 9.46 Gbps
- Direct Windows SSH from Mac is blocked or filtered, but direct TCP data plane works when a listener is bound.
- Mac build: `/Users/kch3dri4n/tools/llama.cpp/build-mcd-metal/bin`
- Windows build: `C:\Users\-_-kc\Desktop\llama.cpp\build-mcd-cuda\bin\Release`
- Mac devices: `MTL0: Apple M4 Max`, `BLAS: Accelerate`
- Windows devices: `CUDA0: NVIDIA GeForce RTX 5070 Ti`, `CUDA1: NVIDIA GeForce RTX 4060 Ti`
- Initial local prefill result: Windows CUDA is about 5.6x faster at 512 prompt tokens and about 7.6x faster at 2048 prompt tokens.
- Initial local decode result: Mac Metal is slightly faster for 64 token generation in this setup.
- Server handoff result, 685 prompt tokens: Mac-only 4.284 s, handoff 2.508 s, 1.71x speedup.
- Server handoff result, 2557 prompt tokens: Mac-only 12.945 s, handoff 4.329 s, 2.99x speedup.
- Server handoff result, 10109 prompt tokens: Mac-only 52.442 s, handoff 10.044 s, 5.22x speedup.
- Server handoff result, 20423 prompt tokens: Mac-only 115.649 s, handoff 19.965 s, 5.79x speedup.
- Server handoff result, 41041 prompt tokens: Mac-only 274.582 s, handoff 46.274 s, 5.93x speedup.
- Near-full 65536-context handoff result, 65319 prompt tokens: Mac-only 526.319 s, handoff 64.924 s, 8.11x speedup.
- Direct Windows -> Mac slot transfer sustained about 9.0-9.43 Gbps with SHA256 match.
- Mac decode after restore remained near Mac-only decode: 16-token server decode was 16.232 tok/s after restore at 65319 prompt tokens versus 16.230 tok/s in the Mac-only baseline.
- Mac -> Windows HTTP on port 18080 is blocked by Windows firewall/profile, even though Windows local access and direct TCP copy to Mac work.
- Common benchmark model:
  - Mac: `/Volumes/Back_UP_LLM/models/unsloth/gemma-4-31B-it-qat-GGUF/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf`
  - Windows: `C:\Users\-_-kc\models\unsloth\gemma-4-31B-it-qat-GGUF\gemma-4-31B-it-qat-UD-Q4_K_XL.gguf`

## Remaining Work

1. Run existing llama.cpp RPC split as a baseline only.
2. Decide whether a small C++ patch is useful for orchestration convenience.
3. If pursuing production use, replace the manual save/copy/restore sequence
   with a supervised local controller.
