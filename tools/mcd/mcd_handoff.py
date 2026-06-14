#!/usr/bin/env python3
import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path


def post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    elapsed = time.perf_counter() - start
    return elapsed, json.loads(body.decode("utf-8"))


def completion(server, prompt, n_predict, id_slot, timeout):
    return post_json(server.rstrip("/") + "/completion", {
        "prompt": prompt,
        "n_predict": n_predict,
        "id_slot": id_slot,
        "cache_prompt": True,
        "temperature": 0,
        "seed": 1234,
        "response_fields": ["content", "timings", "tokens_predicted", "tokens_evaluated"],
    }, timeout)


def slot_action(server, id_slot, action, filename, timeout):
    return post_json(server.rstrip("/") + f"/slots/{id_slot}?action={action}", {
        "filename": filename,
    }, timeout)


def local_copy(src, dst):
    start = time.perf_counter()
    shutil.copy2(src, dst)
    elapsed = time.perf_counter() - start
    size = Path(dst).stat().st_size
    return {"method": "local-copy", "bytes": size, "seconds": elapsed, "gbps": size * 8 / elapsed / 1e9 if elapsed > 0 else 0}


def run(args):
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    filename = args.filename or f"mcd_slot_{int(time.time())}.bin"

    t_prefill, prefill = completion(args.windows_server, prompt, 0, args.id_slot, args.timeout)
    t_save, save = slot_action(args.windows_server, args.id_slot, "save", filename, args.timeout)

    transfer = None
    if args.windows_slot_dir and args.mac_slot_dir:
        src = Path(args.windows_slot_dir) / filename
        dst = Path(args.mac_slot_dir) / filename
        transfer = local_copy(src, dst)

    t_restore, restore = slot_action(args.mac_server, args.id_slot, "restore", filename, args.timeout)
    t_decode, decode = completion(args.mac_server, prompt, args.n_predict, args.id_slot, args.timeout)

    result = {
        "filename": filename,
        "prompt": args.prompt,
        "id_slot": args.id_slot,
        "wall_seconds": {
            "windows_prefill": t_prefill,
            "windows_save": t_save,
            "mac_restore": t_restore,
            "mac_decode": t_decode,
        },
        "transfer": transfer,
        "responses": {
            "prefill": prefill,
            "save": save,
            "restore": restore,
            "decode": decode,
        },
    }
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Windows CUDA prefill to Mac Metal decode handoff")
    parser.add_argument("--windows-server", required=True)
    parser.add_argument("--mac-server", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--filename")
    parser.add_argument("--windows-slot-dir")
    parser.add_argument("--mac-slot-dir")
    parser.add_argument("--id-slot", type=int, default=0)
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
