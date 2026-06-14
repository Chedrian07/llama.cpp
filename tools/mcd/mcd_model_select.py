#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path, limit_gb):
    size = path.stat().st_size
    if limit_gb is not None and size > limit_gb * 1024 ** 3:
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_quant(name):
    parts = name.replace("-", "_").replace(".", "_").split("_")
    for i, part in enumerate(parts):
        if part in {"Q2", "Q3", "Q4", "Q5", "Q6", "Q8", "IQ1", "IQ2", "IQ3", "IQ4", "UD"}:
            return "_".join(parts[i:i + 4]).rstrip("_")
        if part in {"F16", "BF16", "F32", "NVFP4", "MXFP4"}:
            return part
    return ""


def inventory(args):
    rows = []
    for root in args.roots:
        root_path = Path(root).expanduser()
        if not root_path.exists():
            continue
        for path in root_path.rglob("*.gguf"):
            if not path.is_file():
                continue
            stat = path.stat()
            rows.append({
                "path": str(path),
                "basename": path.name,
                "size": stat.st_size,
                "size_gb": round(stat.st_size / 1024 ** 3, 3),
                "mtime": int(stat.st_mtime),
                "quant": infer_quant(path.name),
                "sha256": sha256_file(path, args.sha256_limit_gb) if args.sha256 else None,
            })
    rows.sort(key=lambda r: (r["basename"], r["size"], r["path"]))
    out = {"roots": args.roots, "count": len(rows), "models": rows}
    print(json.dumps(out, indent=2))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def match(args):
    left = load_json(args.left)["models"]
    right = load_json(args.right)["models"]
    right_index = {}
    for row in right:
        right_index.setdefault((row["basename"], row["size"]), []).append(row)
    matches = []
    for lrow in left:
        for rrow in right_index.get((lrow["basename"], lrow["size"]), []):
            sha_ok = None
            if lrow.get("sha256") and rrow.get("sha256"):
                sha_ok = lrow["sha256"] == rrow["sha256"]
            matches.append({
                "basename": lrow["basename"],
                "size": lrow["size"],
                "size_gb": lrow["size_gb"],
                "quant": lrow.get("quant") or rrow.get("quant"),
                "left_path": lrow["path"],
                "right_path": rrow["path"],
                "sha256_match": sha_ok,
            })
    matches.sort(key=lambda r: r["size"], reverse=True)
    print(json.dumps({"count": len(matches), "matches": matches}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Inventory and match GGUF models")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("inventory")
    p.add_argument("roots", nargs="+")
    p.add_argument("--sha256", action="store_true")
    p.add_argument("--sha256-limit-gb", type=float, default=2.0)
    p.set_defaults(func=inventory)

    p = sub.add_parser("match")
    p.add_argument("left")
    p.add_argument("right")
    p.set_defaults(func=match)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
