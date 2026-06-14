#!/usr/bin/env python3
import argparse
import hashlib
import json
import socket
import struct
import time
from pathlib import Path


def sha256_path(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def recv_exact(conn, n):
    chunks = []
    got = 0
    while got < n:
        chunk = conn.recv(min(1024 * 1024, n - got))
        if not chunk:
            raise RuntimeError("connection closed")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def server(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(1)
    print(json.dumps({"event": "listening", "bind": args.bind, "port": args.port}), flush=True)
    conn, addr = srv.accept()
    with conn:
        header_len = struct.unpack("!Q", recv_exact(conn, 8))[0]
        header = json.loads(recv_exact(conn, header_len).decode("utf-8"))
        dst = out_dir / Path(header["name"]).name
        remain = int(header["size"])
        h = hashlib.sha256()
        start = time.perf_counter()
        with dst.open("wb") as f:
            while remain > 0:
                chunk = conn.recv(min(1024 * 1024, remain))
                if not chunk:
                    raise RuntimeError("connection closed while receiving data")
                f.write(chunk)
                h.update(chunk)
                remain -= len(chunk)
        elapsed = time.perf_counter() - start
    result = {
        "peer": addr[0],
        "path": str(dst),
        "bytes": header["size"],
        "seconds": elapsed,
        "gbps": header["size"] * 8 / elapsed / 1e9 if elapsed > 0 else 0,
        "sha256": h.hexdigest(),
        "sha256_match": h.hexdigest() == header.get("sha256"),
    }
    print(json.dumps(result, indent=2))


def client(args):
    path = Path(args.file)
    size = path.stat().st_size
    digest = sha256_path(path) if args.sha256 else None
    header = json.dumps({"name": path.name, "size": size, "sha256": digest}).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if args.source:
        sock.bind((args.source, 0))
    sock.connect((args.host, args.port))
    start = time.perf_counter()
    with sock:
        sock.sendall(struct.pack("!Q", len(header)))
        sock.sendall(header)
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sock.sendall(chunk)
    elapsed = time.perf_counter() - start
    print(json.dumps({
        "file": str(path),
        "host": args.host,
        "port": args.port,
        "source": args.source,
        "bytes": size,
        "seconds": elapsed,
        "gbps": size * 8 / elapsed / 1e9 if elapsed > 0 else 0,
        "sha256": digest,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Direct TCP file copy for MCD")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("server")
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5202)
    p.add_argument("--out", default=".")
    p.set_defaults(func=server)

    p = sub.add_parser("client")
    p.add_argument("file")
    p.add_argument("host")
    p.add_argument("--source")
    p.add_argument("--port", type=int, default=5202)
    p.add_argument("--sha256", action="store_true")
    p.set_defaults(func=client)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
