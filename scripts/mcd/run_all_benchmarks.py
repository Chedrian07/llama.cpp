#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path


def run(cmd, cwd):
    start = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {
        "cmd": cmd,
        "returncode": p.returncode,
        "seconds": time.perf_counter() - start,
        "output": p.stdout,
    }


def main():
    parser = argparse.ArgumentParser(description="Run selected MCD benchmark commands")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="results/mcd/run_all_benchmarks.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.cmd:
        raise SystemExit("pass a command after --, or use the specific MCD tools directly")

    result = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dry_run": args.dry_run, "command": args.cmd}
    if not args.dry_run:
        result["run"] = run(args.cmd, args.root)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
