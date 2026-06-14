# MCD Design Notes

## Primary MVP

The MVP is binary state handoff:

1. Windows CUDA evaluates the long prompt.
2. Windows saves slot/cache/state to a binary file.
3. The binary file is transferred over direct 10GbE.
4. Mac Metal restores the state.
5. Mac Metal decodes locally.

The target metric is TTFT. Decode tok/s should remain close to Mac-only decode.

The smoke tests confirm this path works with existing llama.cpp server slot
save/restore APIs. No C++ patch was required for prompt files through
`prompt_52000.txt` (65319 actual prompt tokens with 65536 context).

## Existing RPC Baseline

llama.cpp RPC can expose remote ggml devices and split work across local and
remote devices. This is useful as a baseline, but it is not the primary design
because per-token synchronization over 10GbE can dominate decode latency.

The current measured win comes from disaggregating phases, not from splitting a
single decode graph over the network.

## Current Bottleneck Shape

The 10GbE link is healthy at about 9 Gbps. For the tested Gemma 4 31B Q4 model,
the saved slot state grew from 617 MB at 685 prompt tokens to 6.19 GB at 65319
prompt tokens. Save, copy, and restore overhead is therefore material, but still
small enough to beat Mac-only TTFT by 8.11x at 65319 prompt tokens.

Longer-context tests must track state bytes per restored token and not just
prompt tok/s.

## Future: Remote CUDA KV/Attention Backend

KV-only remote storage is not enough. If Metal fetches KV over 10GbE every
token, decode latency is likely to get worse.

A meaningful remote KV design would keep KV on CUDA and also compute attention
on CUDA, returning only reduced activation data to Metal. That implies custom
attention op routing or a deeper ggml/llama backend change. It is out of scope
until the binary handoff MVP is measured.
