#!/usr/bin/env python3
"""Task-sharded sampling worker implemented in Python."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("result_path", type=str)
    parser.add_argument("node_all", type=int)
    parser.add_argument("node_this", type=int)
    parser.add_argument("start_idx", type=int)
    parser.add_argument("--total-tasks", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.node_all <= 0:
        raise ValueError("node_all must be > 0")
    if args.node_this < 0 or args.node_this >= args.node_all:
        raise ValueError("node_this must satisfy 0 <= node_this < node_all")
    if args.total_tasks <= 0:
        raise ValueError("total_tasks must be > 0")

    for i in range(args.start_idx, args.total_tasks):
        if (i % args.node_all) != args.node_this:
            continue

        print(f"Task {i} assigned to this worker ({args.node_this})", flush=True)
        cmd = [
            sys.executable,
            "-m",
            "scripts.sample_diffusion",
            args.config_file,
            "-i",
            str(i),
            "--result_path",
            args.result_path,
            "--device",
            args.device,
            "--batch_size",
            str(args.batch_size),
        ]
        if args.extra_args:
            cmd.extend(args.extra_args)

        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
