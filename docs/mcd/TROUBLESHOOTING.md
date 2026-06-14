# MCD Troubleshooting

## Direct 10GbE

- If Tailscale SSH works but direct SSH fails, keep using Tailscale for control.
- Use direct `169.254.x.x` addresses only for iperf, RPC, HTTP, and file
  transfer benchmark data.
- Do not change default gateways.
- If link-local routing is unstable, propose static direct IPs:
  - Mac `en0`: `10.10.10.1/30`
  - Windows Ethernet 4: `10.10.10.2/30`

## llama.cpp Server State

If cross-backend restore fails:

1. Test Mac save -> Mac restore -> Mac decode.
2. Test Windows save -> Windows restore -> Windows decode.
3. Check model file identity, context size, KV type, flash attention, and slot
   file paths.
4. Do not claim success unless the restored Mac run avoids full prompt
   re-evaluation.

## RPC Baseline

- Start with `--split-mode layer`.
- Start with flash attention disabled if mixed backend instability appears.
- Treat RPC decode regressions as expected until measured otherwise.

