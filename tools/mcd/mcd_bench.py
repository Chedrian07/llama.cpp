#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def http_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    elapsed = time.perf_counter() - start
    return elapsed, json.loads(body.decode("utf-8"))


def completion(args):
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    payload = {
        "prompt": prompt,
        "n_predict": args.n_predict,
        "cache_prompt": args.cache_prompt,
        "id_slot": args.id_slot,
        "temperature": 0,
        "seed": args.seed,
        "response_fields": [
            "content",
            "timings",
            "tokens_predicted",
            "tokens_evaluated",
            "generation_settings/n_predict",
        ],
    }
    elapsed, body = http_json(args.server.rstrip("/") + "/completion", payload, args.timeout)
    result = {"wall_seconds": elapsed, "server": args.server, "prompt": args.prompt, "response": body}
    print(json.dumps(result, indent=2))


def slot_action(args):
    payload = {"filename": args.filename}
    elapsed, body = http_json(args.server.rstrip("/") + f"/slots/{args.id_slot}?action={args.action}", payload, args.timeout)
    result = {
        "wall_seconds": elapsed,
        "server": args.server,
        "id_slot": args.id_slot,
        "action": args.action,
        "filename": args.filename,
        "response": body,
    }
    print(json.dumps(result, indent=2))


def command(args):
    start = time.perf_counter()
    p = subprocess.run(args.cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - start
    result = {"cmd": args.cmd, "returncode": p.returncode, "seconds": elapsed, "output": p.stdout}
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="MCD benchmark helpers")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("completion")
    p.add_argument("--server", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--n-predict", type=int, default=128)
    p.add_argument("--id-slot", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--no-cache-prompt", dest="cache_prompt", action="store_false")
    p.set_defaults(func=completion, cache_prompt=True)

    p = sub.add_parser("slot")
    p.add_argument("--server", required=True)
    p.add_argument("--id-slot", type=int, default=0)
    p.add_argument("--action", choices=["save", "restore"], required=True)
    p.add_argument("--filename", required=True)
    p.add_argument("--timeout", type=float, default=3600)
    p.set_defaults(func=slot_action)

    p = sub.add_parser("command")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.set_defaults(func=command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
