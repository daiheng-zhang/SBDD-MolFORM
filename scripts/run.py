#!/usr/bin/env python3
"""Unified Python runtime entrypoint for MolForm tasks."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from scripts.runtime_utils import (
    RuntimeConfigError,
    load_runtime_config,
    prepare_sample_nft_config,
    prepare_train_base_config,
    prepare_train_dpo_config,
    prepare_train_nft_vina_sa_config,
    resolve_path,
    runtime_env,
    validate_runtime_config,
)


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _run(cmd: list[str], env: dict[str, str], dry_run: bool, cwd: Path | None = None) -> int:
    print(f"[CMD] {_format_cmd(cmd)}", flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(cmd, env=env, cwd=str(cwd) if cwd is not None else None)
    return completed.returncode


def _repo_root(cfg: dict) -> Path:
    return Path(cfg["_meta"]["repo_root"]).resolve()


def _cleanup_temp(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _prepare_common(cfg: dict) -> dict[str, str]:
    env = runtime_env(cfg)
    paths = cfg["paths"]
    Path(paths["log_root"]).mkdir(parents=True, exist_ok=True)
    Path(paths["output_root"]).mkdir(parents=True, exist_ok=True)
    Path(paths["tmp_root"]).mkdir(parents=True, exist_ok=True)
    return env


def _run_train_base(cfg: dict, dry_run: bool) -> int:
    validate_runtime_config(cfg, "train-base", strict_paths=not dry_run)
    env = _prepare_common(cfg)
    repo_root = _repo_root(cfg)

    temp_cfg = prepare_train_base_config(cfg)
    try:
        runtime = cfg["runtime"]
        inputs = cfg.get("inputs", {})
        cmd = [
            sys.executable,
            "-m",
            "scripts.train_diffusion",
            str(temp_cfg),
            "--tag",
            str(runtime.get("tag", "standard_scratch")),
            "--logdir",
            str(cfg["paths"]["log_root"]),
            "--name",
            str(runtime.get("run_name", "KDD-Molform-scratch")),
            "--train_report_iter",
            str(int(runtime.get("train_report_iter", 10))),
        ]

        ckpt = inputs.get("checkpoint")
        if ckpt:
            ckpt_path = resolve_path(str(ckpt), _repo_root(cfg))
            cmd.extend(["--checkpoint", str(ckpt_path), "--no_optimizer_state", "--non_strict_load"])

        return _run(cmd, env, dry_run, cwd=repo_root)
    finally:
        _cleanup_temp(temp_cfg)


def _run_train_dpo(cfg: dict, dry_run: bool) -> int:
    validate_runtime_config(cfg, "train-dpo", strict_paths=not dry_run)
    env = _prepare_common(cfg)
    repo_root = _repo_root(cfg)

    temp_cfg = prepare_train_dpo_config(cfg)
    try:
        runtime = cfg["runtime"]
        dpo_data = resolve_path(str(cfg["inputs"]["dpo_data"]), _repo_root(cfg))
        cmd = [
            sys.executable,
            "-m",
            "scripts.train_diffusion",
            str(temp_cfg),
            "--tag",
            str(runtime.get("tag", "dpo_confidence_head")),
            "--dpo_data",
            str(dpo_data),
            "--logdir",
            str(cfg["paths"]["log_root"]),
            "--name",
            str(runtime.get("run_name", "KDD-Molform-DPO-Confidence")),
            "--train_report_iter",
            str(int(runtime.get("train_report_iter", 10))),
        ]
        return _run(cmd, env, dry_run, cwd=repo_root)
    finally:
        _cleanup_temp(temp_cfg)


def _run_train_nft_vina_sa(cfg: dict, dry_run: bool) -> int:
    validate_runtime_config(cfg, "train-nft-vina-sa", strict_paths=not dry_run)
    env = _prepare_common(cfg)
    repo_root = _repo_root(cfg)

    temp_cfg = prepare_train_nft_vina_sa_config(cfg)
    try:
        runtime = cfg["runtime"]
        inputs = cfg.get("inputs", {})
        cmd = [
            sys.executable,
            "-m",
            "scripts.train_diffusion",
            str(temp_cfg),
            "--tag",
            str(runtime.get("tag", "nft_vina_sa")),
            "--logdir",
            str(cfg["paths"]["log_root"]),
            "--name",
            str(runtime.get("run_name", "KDD-Molform-NFT-VinaSA")),
            "--train_report_iter",
            str(int(runtime.get("train_report_iter", 10))),
        ]

        init_ckpt = inputs.get("init_checkpoint")
        if init_ckpt:
            init_path = resolve_path(str(init_ckpt), _repo_root(cfg))
            cmd.extend(["--checkpoint", str(init_path), "--no_optimizer_state", "--non_strict_load"])

        return _run(cmd, env, dry_run, cwd=repo_root)
    finally:
        _cleanup_temp(temp_cfg)


def _run_sample_nft(cfg: dict, dry_run: bool) -> int:
    validate_runtime_config(cfg, "sample-nft", strict_paths=not dry_run)
    env = _prepare_common(cfg)
    repo_root = _repo_root(cfg)

    temp_cfg = prepare_sample_nft_config(cfg)
    try:
        runtime = cfg["runtime"]
        inputs = cfg["inputs"]
        output_root = Path(cfg["paths"]["output_root"])

        result_path_raw = inputs.get("result_path")
        if result_path_raw:
            result_path = resolve_path(str(result_path_raw), _repo_root(cfg))
        else:
            prefix = str(runtime.get("result_prefix", "sampling_nft_steps100"))
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = (output_root / f"{prefix}_{ts}").resolve()
        assert result_path is not None
        result_path.mkdir(parents=True, exist_ok=True)

        gpu_ids = list(runtime.get("gpu_ids", [0]))
        if not gpu_ids:
            raise RuntimeConfigError("sample-nft runtime.gpu_ids cannot be empty")

        total_tasks = int(runtime.get("total_tasks", 100))
        num_steps = int(runtime.get("num_steps", 100))
        num_samples = int(runtime.get("num_samples", 100))
        batch_size = int(runtime.get("batch_size", 100))
        start_idx = int(runtime.get("start_idx", 0))
        smoke = bool(runtime.get("smoke", False))

        if smoke:
            gpu_ids = [gpu_ids[0]]
            total_tasks = 1
            num_steps = 10
            num_samples = 1
            node_all = 1
        else:
            node_all_raw = runtime.get("node_all")
            node_all = int(node_all_raw) if node_all_raw is not None else len(gpu_ids)

        # Keep runtime summary visible for reproducibility.
        print(
            f"[sample-nft] result_path={result_path} gpu_ids={gpu_ids} total_tasks={total_tasks} "
            f"num_steps={num_steps} num_samples={num_samples} node_all={node_all}",
            flush=True,
        )

        worker_cmds: list[tuple[list[str], dict[str, str]]] = []
        for worker_idx, gpu in enumerate(gpu_ids):
            cmd = [
                sys.executable,
                "-m",
                "scripts.batch_sample_diffusion",
                str(temp_cfg),
                str(result_path),
                str(node_all),
                str(worker_idx),
                str(start_idx),
                "--total-tasks",
                str(total_tasks),
                "--device",
                "cuda:0",
                "--batch-size",
                str(batch_size),
            ]
            worker_env = dict(env)
            worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            worker_cmds.append((cmd, worker_env))

        if dry_run:
            for cmd, _ in worker_cmds:
                _run(cmd, env, dry_run=True, cwd=repo_root)
            return 0

        procs = []
        for cmd, worker_env in worker_cmds:
            print(f"[CMD] {_format_cmd(cmd)} (CUDA_VISIBLE_DEVICES={worker_env['CUDA_VISIBLE_DEVICES']})", flush=True)
            procs.append(subprocess.Popen(cmd, env=worker_env, cwd=str(repo_root)))

        codes = [p.wait() for p in procs]
        if any(code != 0 for code in codes):
            return 1
        return 0
    finally:
        _cleanup_temp(temp_cfg)


def _run_eval_simple(cfg: dict, dry_run: bool) -> int:
    validate_runtime_config(cfg, "eval-simple", strict_paths=not dry_run)
    env = _prepare_common(cfg)
    repo_root = _repo_root(cfg)

    inputs = cfg["inputs"]
    options = cfg.get("options", {})

    sample_path = resolve_path(str(inputs["sample_path"]), repo_root)
    protein_root = resolve_path(str(inputs.get("protein_root", "./data/test_set")), repo_root)
    assert sample_path is not None and protein_root is not None

    cmd = [
        sys.executable,
        "-m",
        "scripts.evaluate_diffusion_multiprocess_simple",
        str(sample_path),
        "--protein_root",
        str(protein_root),
        "--docking_mode",
        str(options.get("docking_mode", "none")),
        "--exhaustiveness",
        str(int(options.get("exhaustiveness", 16))),
        "--multiprocess",
        str(bool(options.get("multiprocess", False))),
        "--save_pickle",
        str(bool(options.get("save_pickle", False))),
        "--report_confidence_vina",
        str(bool(options.get("report_confidence_vina", False))),
        "--save",
        str(bool(options.get("save", True))),
        "--verbose",
        str(bool(options.get("verbose", False))),
        "--eval_step",
        str(int(options.get("eval_step", -1))),
    ]

    eval_num_examples = options.get("eval_num_examples")
    if eval_num_examples is not None:
        cmd.extend(["--eval_num_examples", str(int(eval_num_examples))])

    return _run(cmd, env, dry_run, cwd=repo_root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MolForm unified python runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ["train-base", "train-dpo", "train-nft-vina-sa", "sample-nft", "eval-simple"]:
        sp = subparsers.add_parser(name)
        sp.add_argument("--runtime-config", required=True, type=str)
        sp.add_argument("--dry-run", action="store_true")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        cfg = load_runtime_config(args.runtime_config)

        if args.command == "train-base":
            return _run_train_base(cfg, args.dry_run)
        if args.command == "train-dpo":
            return _run_train_dpo(cfg, args.dry_run)
        if args.command == "train-nft-vina-sa":
            return _run_train_nft_vina_sa(cfg, args.dry_run)
        if args.command == "sample-nft":
            return _run_sample_nft(cfg, args.dry_run)
        if args.command == "eval-simple":
            return _run_eval_simple(cfg, args.dry_run)

        parser.error(f"Unknown command: {args.command}")
        return 2
    except RuntimeConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
