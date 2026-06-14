# MCD Results

## Repository

| Field | Value |
| --- | --- |
| Owner | `Chedrian07` |
| Repository | `llama.cpp-mcd-prefill` |
| URL | `https://github.com/Chedrian07/llama.cpp-mcd-prefill` |
| Branch | `sj/mcd-prefill-disagg-20260614` |
| Upstream | `https://github.com/ggml-org/llama.cpp` |
| Base commit | `6e14286ed cli : fix not copying preserved tokens (#24258)` |

## Network

Direct data-plane addresses:

| Host | Interface | Address | Link |
| --- | --- | --- | --- |
| Mac Studio | `en0` | `169.254.150.225/16` | 10Gbase-T full-duplex |
| Windows PC | `Ethernet 4` | `169.254.21.157/16` | 10 Gbps |

Reachability:

| Direction | Result | Notes |
| --- | --- | --- |
| Windows -> Mac ping | Pass | `169.254.150.225`, sub-ms |
| Windows -> Mac SSH port | Pass | `169.254.21.157` source interface |
| Mac -> Windows ping | Fail | Likely Windows firewall/profile behavior |
| Mac -> Windows direct TCP test port | Pass | Temporary listener on `169.254.21.157:5201` accepted connection |

Throughput with `tools/mcd/mcd_probe.py`:

| Direction | Client Gbps | Server Gbps | Bytes | Duration |
| --- | ---: | ---: | ---: | ---: |
| Mac -> Windows | 9.332 | 9.330 | 11,665,408,000 | 10.000 s |
| Windows -> Mac | 9.458 | 9.462 | 11,822,694,400 | 10.000 s |

Interpretation: direct 10GbE is healthy enough for the binary state handoff
MVP. Network copy time is still worth measuring, but raw link throughput is not
the first blocker.

## Build

| Host | Build dir | Backend result |
| --- | --- | --- |
| Mac Studio | `/Users/kch3dri4n/tools/llama.cpp/build-mcd-metal/bin` | Built `llama-server`, `llama-cli`, `llama-bench`, `rpc-server` with Metal, BLAS, RPC |
| Windows PC | `C:\Users\-_-kc\Desktop\llama.cpp\build-mcd-cuda\bin\Release` | Built `llama-server.exe`, `llama-cli.exe`, `llama-bench.exe`, `rpc-server.exe` with CUDA, RPC |

Device check:

| Host | Command | Result |
| --- | --- | --- |
| Mac Studio | `llama-cli --list-devices` | `MTL0: Apple M4 Max (110100 MiB free)`, `BLAS: Accelerate` |
| Windows PC | `llama-cli.exe --list-devices` | `CUDA0: NVIDIA GeForce RTX 5070 Ti (15009 MiB free)`, `CUDA1: NVIDIA GeForce RTX 4060 Ti (15221 MiB free)` |

Build notes:

- Mac SSH non-login PATH did not include CMake. `scripts/mcd/mac_build.sh`
  now prepends common Homebrew/CMake.app paths.
- Windows first build produced very noisy CP949 warnings from CUDA headers, but
  the required binaries were produced.
- NCCL was not found on Windows, so multi-GPU CUDA collectives may be
  suboptimal. This matters for split-GPU Windows baselines, not for proving the
  prefill handoff concept.

## Model

GGUF inventory:

| Host | Roots | Count |
| --- | --- | ---: |
| Mac Studio | `~/.lmstudio/models`, `/Volumes/Back_UP_LLM/models` | 9 |
| Windows PC | `C:\Users\-_-kc\models` | 30 |

Initial common model candidate:

| Field | Value |
| --- | --- |
| Basename | `gemma-4-31B-it-qat-UD-Q4_K_XL.gguf` |
| Size | 16.100 GiB |
| Mac path | `/Volumes/Back_UP_LLM/models/unsloth/gemma-4-31B-it-qat-GGUF/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf` |
| Windows path | `C:\Users\-_-kc\models\unsloth\gemma-4-31B-it-qat-GGUF\gemma-4-31B-it-qat-UD-Q4_K_XL.gguf` |

Only matching large text model found so far is the Gemma 4 31B QAT Q4 file.
The matching `mmproj-F32.gguf` exists too, but it is not useful for the first
text-only TTFT benchmark.

## Benchmarks

### Local llama-bench baseline

Model: `gemma-4-31B-it-qat-UD-Q4_K_XL.gguf`

Command shape:

```sh
llama-bench -m MODEL -p 512,2048 -n 64 -r 1 -o json --no-warmup -ngl 99
```

| Host | Devices | Mode | Prompt | Gen | tok/s | Wall time |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Mac Studio | `MTL0` | prompt processing | 512 | 0 | 199.656 | 2.564 s |
| Mac Studio | `MTL0` | prompt processing | 2048 | 0 | 215.655 | 9.497 s |
| Mac Studio | `MTL0` | token generation | 0 | 64 | 23.683 | 2.702 s |
| Windows PC | `CUDA0/CUDA1` | prompt processing | 512 | 0 | 1113.217 | 0.460 s |
| Windows PC | `CUDA0/CUDA1` | prompt processing | 2048 | 0 | 1638.256 | 1.250 s |
| Windows PC | `CUDA0/CUDA1` | token generation | 0 | 64 | 21.524 | 2.973 s |

Ratios:

| Metric | Ratio |
| --- | ---: |
| Windows/Mac prompt processing, 512 tokens | 5.58x |
| Windows/Mac prompt processing, 2048 tokens | 7.60x |
| Mac/Windows token generation, 64 tokens | 1.10x |

Interpretation: the first measured data strongly supports the TTFT-focused
architecture. Windows CUDA is much faster for prompt prefill, while Mac Metal
decode is not slower in this small decode-only benchmark. The next required
measurement is server-level TTFT plus slot save/transfer/restore overhead.

### Server prefill handoff smoke tests

Server configuration:

| Host | Address | Model |
| --- | --- | --- |
| Mac Studio | `169.254.150.225:18081` | `gemma-4-31B-it-qat-UD-Q4_K_XL.gguf` |
| Windows PC | `169.254.21.157:18080` | `gemma-4-31B-it-qat-UD-Q4_K_XL.gguf` |

The servers were restarted with enough context for each prompt. The 512 and
2048 prompt runs used 4096 context, the 8192 prompt run used 16384 context, the
16384 prompt run used 32768 context, and the 32768 and 52000 prompt runs used
65536 context.

Flow:

1. Windows CUDA `/completion` with `n_predict=0`.
2. Windows `/slots/0?action=save`.
3. Direct TCP copy from `169.254.21.157` to `169.254.150.225`.
4. Mac `/slots/0?action=restore`.
5. Mac `/completion` with the same prompt and `n_predict=16`.

Results:

| Prompt file | Context | Actual prompt tokens | Mac-only wall | Handoff total | Speedup | Windows prefill | Save | Copy | Restore | Mac decode-after-restore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prompt_512.txt` | 4096 | 685 | 4.284 s | 2.508 s | 1.71x | 0.790 s | 0.237 s | 0.548 s | 0.078 s | 0.855 s |
| `prompt_2048.txt` | 4096 | 2557 | 12.945 s | 4.329 s | 2.99x | 2.044 s | 0.400 s | 0.906 s | 0.085 s | 0.895 s |
| `prompt_8192.txt` | 16384 | 10109 | 52.442 s | 10.044 s | 5.22x | 6.603 s | 0.936 s | 1.425 s | 0.143 s | 0.937 s |
| `prompt_16384.txt` | 32768 | 20423 | 115.649 s | 19.965 s | 5.79x | 14.982 s | 1.586 s | 2.173 s | 0.214 s | 1.011 s |
| `prompt_32768.txt` | 65536 | 41041 | 274.582 s | 46.274 s | 5.93x | 38.121 s | 3.088 s | 3.588 s | 0.331 s | 1.147 s |
| `prompt_52000.txt` | 65536 | 65319 | 526.319 s | 64.924 s | 8.11x | 50.304 s | 7.563 s | 5.251 s | 0.453 s | 1.353 s |

State transfer:

| Prompt file | Slot bytes | Copy Gbps | SHA256 |
| --- | ---: | ---: | --- |
| `prompt_512.txt` | 617,287,864 | 9.014 | Match |
| `prompt_2048.txt` | 1,048,384,924 | 9.254 | Match |
| `prompt_8192.txt` | 1,667,165,596 | 9.359 | Match |
| `prompt_16384.txt` | 2,512,253,500 | 9.250 | Match |
| `prompt_32768.txt` | 4,201,609,948 | 9.369 | Match |
| `prompt_52000.txt` | 6,190,852,156 | 9.432 | Match |

Restore effectiveness:

| Prompt file | Restored cache tokens | Tail prompt eval tokens | Decode tok/s |
| --- | ---: | ---: | ---: |
| `prompt_512.txt` | 680 | 5 | 25.368 |
| `prompt_2048.txt` | 2552 | 5 | 24.456 |
| `prompt_8192.txt` | 10104 | 5 | 23.343 |
| `prompt_16384.txt` | 20418 | 5 | 21.782 |
| `prompt_32768.txt` | 41036 | 5 | 19.214 |
| `prompt_52000.txt` | 65314 | 5 | 16.232 |

Decode preservation:

| Prompt file | Mac-only decode tok/s | After-restore decode tok/s |
| --- | ---: | ---: |
| `prompt_512.txt` | 25.335 | 25.368 |
| `prompt_2048.txt` | 24.377 | 24.456 |
| `prompt_8192.txt` | 23.326 | 23.343 |
| `prompt_16384.txt` | 21.869 | 21.782 |
| `prompt_32768.txt` | 19.201 | 19.214 |
| `prompt_52000.txt` | 16.230 | 16.232 |

Interpretation: the binary slot handoff MVP works without C++ changes for this
model and build. Mac restores the Windows-saved state and only evaluates a
5-token tail before decoding. The 2048 and larger prompt tests exceed the 2.0x
target. The near-full 65536-context test (`prompt_52000.txt`, 65319 actual
prompt tokens) reached 8.11x end-to-end speedup. Decode speed is preserved
within measurement noise. The main scaling risk is slot state size, not raw
10GbE throughput.

Known networking issue:

- Mac -> Windows HTTP on `169.254.21.157:18080` timed out.
- Windows local access to `169.254.21.157:18080` works.
- Windows -> Mac direct TCP copy works at about 9 Gbps.
- This points to Windows firewall/profile blocking inbound `llama-server.exe`
  on the direct interface. Current orchestration works around this by invoking
  Windows local HTTP through SSH control and using direct 10GbE only for slot
  file transfer.

### Pass/Fail

Current MVP result: pass.

| Check | Result | Evidence |
| --- | --- | --- |
| Feasible phase-disaggregated inference | Pass | Windows prefill state restored on Mac without C++ changes through `prompt_16384.txt` |
| Long-context TTFT improvement | Pass | 2.99x at 2557 tokens, 5.22x at 10109 tokens, 5.79x at 20423 tokens, 8.11x at 65319 tokens |
| Decode preservation on Mac | Pass | After-restore decode tok/s stayed within measurement noise of Mac-only decode |
| Direct 10GbE data plane | Pass | Slot copy stayed around 9.0-9.36 Gbps with matching SHA256 |
| Production JSON-RPC distributed runtime | Not yet | Current implementation is scripted server slot save/copy/restore orchestration |

Conclusion: the project is realistic as a phase-disaggregated prefill handoff
system. It is not yet realistic as a transparent per-token Metal+CUDA split over
10GbE, because decode-time synchronization would likely dominate unless CUDA
also owns the remote attention/KV work.
