#!/usr/bin/env python3
import argparse
import json
import os
import platform
import socket
import struct
import subprocess
import sys
import time


def run(cmd):
    try:
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return {"cmd": cmd, "returncode": p.returncode, "output": p.stdout}
    except FileNotFoundError as exc:
        return {"cmd": cmd, "returncode": 127, "output": str(exc)}


def host_info(args):
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    probes = []
    if sys.platform == "win32":
        probes.append(run(["ipconfig", "/all"]))
        probes.append(run(["powershell", "-NoProfile", "-Command", "Get-NetAdapter | Format-Table -AutoSize"]))
    else:
        probes.append(run(["ifconfig"]))
        probes.append(run(["netstat", "-rn"]))
    info["probes"] = probes
    print(json.dumps(info, indent=2))


def ping(args):
    count_arg = "-n" if sys.platform == "win32" else "-c"
    cmd = ["ping", count_arg, str(args.count), args.host]
    print(json.dumps(run(cmd), indent=2))


def tcp_server(args):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(1)
    print(json.dumps({"event": "listening", "bind": args.bind, "port": args.port}), flush=True)
    conn, addr = srv.accept()
    start = time.perf_counter()
    total = 0
    with conn:
        while True:
            data = conn.recv(1024 * 1024)
            if not data:
                break
            total += len(data)
    elapsed = time.perf_counter() - start
    gbps = total * 8 / elapsed / 1e9 if elapsed > 0 else 0
    print(json.dumps({
        "event": "complete",
        "peer": addr[0],
        "bytes": total,
        "seconds": elapsed,
        "gbps": gbps,
    }, indent=2))


def tcp_client(args):
    payload = b"\0" * args.chunk
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if args.source:
        sock.bind((args.source, 0))
    sock.connect((args.host, args.port))
    end = time.perf_counter() + args.seconds
    start = time.perf_counter()
    total = 0
    with sock:
        while time.perf_counter() < end:
            sock.sendall(payload)
            total += len(payload)
    elapsed = time.perf_counter() - start
    gbps = total * 8 / elapsed / 1e9 if elapsed > 0 else 0
    print(json.dumps({
        "host": args.host,
        "port": args.port,
        "source": args.source,
        "bytes": total,
        "seconds": elapsed,
        "gbps": gbps,
    }, indent=2))


def port_check(args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(args.timeout)
    start = time.perf_counter()
    try:
        sock.connect((args.host, args.port))
        ok = True
        err = None
    except OSError as exc:
        ok = False
        err = str(exc)
    finally:
        sock.close()
    print(json.dumps({
        "host": args.host,
        "port": args.port,
        "ok": ok,
        "error": err,
        "ms": (time.perf_counter() - start) * 1000,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="MCD host and network probe")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("host")
    p.set_defaults(func=host_info)

    p = sub.add_parser("ping")
    p.add_argument("host")
    p.add_argument("--count", type=int, default=4)
    p.set_defaults(func=ping)

    p = sub.add_parser("tcp-server")
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5201)
    p.set_defaults(func=tcp_server)

    p = sub.add_parser("tcp-client")
    p.add_argument("host")
    p.add_argument("--port", type=int, default=5201)
    p.add_argument("--source")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--chunk", type=int, default=1024 * 1024)
    p.set_defaults(func=tcp_client)

    p = sub.add_parser("port-check")
    p.add_argument("host")
    p.add_argument("port", type=int)
    p.add_argument("--timeout", type=float, default=3)
    p.set_defaults(func=port_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
