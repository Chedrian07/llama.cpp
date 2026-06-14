#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def load(path):
    data = Path(path).read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    elif data.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"
    else:
        encoding = "utf-8"
    text = data.decode(encoding, errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(?m)^\s*[\[{]", text)
        if not match:
            raise
        return json.loads(text[match.start():])


def summarize(args):
    lines = ["# MCD Report Input Summary", ""]
    for path in args.files:
        data = load(path)
        lines.append(f"## {path}")
        if isinstance(data, dict):
            for key in sorted(data.keys()):
                val = data[key]
                if isinstance(val, (str, int, float, bool)) or val is None:
                    lines.append(f"- `{key}`: `{val}`")
                else:
                    lines.append(f"- `{key}`: `{type(val).__name__}`")
        lines.append("")
    print("\n".join(lines))


def get_timing(data, key, default=0):
    return data.get("response", {}).get("timings", {}).get(key, default)


def handoff(args):
    mac_base = load(args.mac_base)
    windows_prefill = load(args.windows_prefill)
    windows_save = load(args.windows_save)
    windows_copy = load(args.windows_copy)
    mac_restore = load(args.mac_restore)
    mac_decode = load(args.mac_decode)

    total = (
        windows_prefill["wall_seconds"] +
        windows_save["wall_seconds"] +
        windows_copy["seconds"] +
        mac_restore["wall_seconds"] +
        mac_decode["wall_seconds"]
    )
    base = mac_base["wall_seconds"]
    result = {
        "label": args.label,
        "mac_base_wall_seconds": base,
        "handoff_total_seconds": total,
        "speedup": base / total if total else None,
        "prompt_tokens": get_timing(mac_base, "prompt_n"),
        "windows_prefill_seconds": windows_prefill["wall_seconds"],
        "windows_prefill_tokens_per_second": get_timing(windows_prefill, "prompt_per_second"),
        "windows_save_seconds": windows_save["wall_seconds"],
        "slot_bytes": windows_save.get("response", {}).get("n_written"),
        "copy_seconds": windows_copy["seconds"],
        "copy_gbps": windows_copy["gbps"],
        "mac_restore_seconds": mac_restore["wall_seconds"],
        "mac_decode_seconds": mac_decode["wall_seconds"],
        "restored_cache_tokens": get_timing(mac_decode, "cache_n"),
        "tail_prompt_tokens": get_timing(mac_decode, "prompt_n"),
        "decode_tokens_per_second": get_timing(mac_decode, "predicted_per_second"),
    }
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Summarize MCD JSON outputs")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("summarize")
    p.add_argument("files", nargs="+")
    p.set_defaults(func=summarize)

    p = sub.add_parser("handoff")
    p.add_argument("--label", required=True)
    p.add_argument("--mac-base", required=True)
    p.add_argument("--windows-prefill", required=True)
    p.add_argument("--windows-save", required=True)
    p.add_argument("--windows-copy", required=True)
    p.add_argument("--mac-restore", required=True)
    p.add_argument("--mac-decode", required=True)
    p.set_defaults(func=handoff)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
