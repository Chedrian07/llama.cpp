# MCD Scripts

Shell and PowerShell wrappers for reproducible builds, servers, and benchmarks.

Build wrappers:

- `mac_build.sh`
- `windows_build.ps1`

Benchmark wrappers:

- `mac_run_bench.sh`
- `windows_run_bench.ps1`

Server wrappers:

- `mac_run_server.sh`
- `windows_run_server.ps1`
- `mac_run_rpc_client.sh`
- `windows_run_rpc_server.ps1`

For single-request handoff benchmarks, pass `--parallel 1` through the server
extra args to avoid allocating extra slots.

See `docs/mcd/USAGE.md` for the full measured runbook.
