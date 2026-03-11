#!/usr/bin/env python3
"""Runtime config helpers for python-only task dispatch."""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_TASKS = {
    "train-base",
    "train-dpo",
    "train-nft-vina-sa",
    "sample-nft",
    "eval-simple",
}


class RuntimeConfigError(ValueError):
    pass


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _expand(path_str: str) -> str:
    return os.path.expanduser(os.path.expandvars(path_str))


def resolve_path(path_str: str | None, repo_root: Path | None = None) -> Path | None:
    if path_str is None:
        return None
    root = repo_root or REPO_ROOT
    expanded = Path(_expand(path_str))
    if expanded.is_absolute():
        return expanded
    return (root / expanded).resolve()


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeConfigError(f"{label} file not found: {path}")


def ensure_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise RuntimeConfigError(f"{label} directory not found: {path}")


def write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeConfigError(f"YAML must be a mapping: {path}")
    return data


def default_runtime_config(task: str) -> Dict[str, Any]:
    common = {
        "task": task,
        "base_config": None,
        "paths": {
            "repo_root": ".",
            "data_root": "./data",
            "log_root": "./logs_diffusion",
            "output_root": "./outputs_sampling",
            "tmp_root": "./tmp",
        },
        "runtime": {
            "smoke": False,
            "tag": "",
            "run_name": "",
            "train_report_iter": 10,
            "omp_num_threads": 16,
        },
        "inputs": {},
        "options": {},
    }

    if task == "train-base":
        common["base_config"] = "configs/training_standard_scratch.yml"
        common["runtime"]["tag"] = "standard_scratch"
        common["runtime"]["run_name"] = "KDD-Molform-scratch"
        common["inputs"] = {"checkpoint": None}
    elif task == "train-dpo":
        common["base_config"] = "configs/training_dpo_kdd_confidence.yml"
        common["runtime"]["tag"] = "dpo_confidence_head"
        common["runtime"]["run_name"] = "KDD-Molform-DPO-Confidence"
        common["inputs"] = {
            "dpo_data": None,
            "dpo_ref_ckpt": None,
        }
    elif task == "train-nft-vina-sa":
        common["base_config"] = "configs/training_nft_vina_sa.yml"
        common["runtime"]["tag"] = "nft_vina_sa"
        common["runtime"]["run_name"] = "KDD-Molform-NFT-VinaSA"
        common["inputs"] = {"init_checkpoint": None}
    elif task == "sample-nft":
        common["base_config"] = "configs/sampling_nft_steps_100_final.yml"
        common["runtime"].update(
            {
                "gpu_ids": [0, 1, 2, 3],
                "node_all": None,
                "start_idx": 0,
                "total_tasks": 100,
                "num_steps": 100,
                "num_samples": 100,
                "batch_size": 100,
                "result_prefix": "sampling_nft_steps100",
            }
        )
        common["inputs"] = {
            "checkpoint": None,
            "result_path": None,
        }
    elif task == "eval-simple":
        common["runtime"]["tag"] = "eval_simple"
        common["inputs"] = {
            "sample_path": None,
            "protein_root": "./data/test_set",
        }
        common["options"] = {
            "docking_mode": "none",
            "exhaustiveness": 16,
            "multiprocess": False,
            "save_pickle": False,
            "report_confidence_vina": False,
            "eval_num_examples": None,
            "save": True,
            "verbose": False,
            "eval_step": -1,
        }
    else:
        raise RuntimeConfigError(f"Unknown task: {task}")

    return common


def load_runtime_config(runtime_config_path: str) -> Dict[str, Any]:
    runtime_path = Path(runtime_config_path).resolve()
    ensure_file(runtime_path, "Runtime config")
    user_cfg = read_yaml(runtime_path)

    task = user_cfg.get("task")
    if task not in VALID_TASKS:
        raise RuntimeConfigError(
            f"Invalid or missing task in runtime config: {task}. Expected one of {sorted(VALID_TASKS)}"
        )

    cfg = _deep_merge(default_runtime_config(task), user_cfg)

    repo_root = resolve_path(str(cfg["paths"].get("repo_root", ".")), REPO_ROOT)
    assert repo_root is not None
    cfg["_meta"] = {
        "runtime_config_path": str(runtime_path),
        "repo_root": str(repo_root),
        "paths_raw": copy.deepcopy(cfg["paths"]),
    }

    for key in ("data_root", "log_root", "output_root", "tmp_root"):
        cfg["paths"][key] = str(resolve_path(str(cfg["paths"][key]), repo_root))

    base_config = cfg.get("base_config")
    cfg["base_config"] = (
        str(resolve_path(str(base_config), repo_root)) if base_config else None
    )

    return cfg


def validate_runtime_config(
    cfg: Dict[str, Any],
    expected_task: str,
    *,
    strict_paths: bool = True,
) -> None:
    task = cfg.get("task")
    if task != expected_task:
        raise RuntimeConfigError(
            f"Runtime config task mismatch: expected '{expected_task}', got '{task}'"
        )

    base_config = cfg.get("base_config")
    if task != "eval-simple":
        if not base_config:
            raise RuntimeConfigError(f"{task} requires a base_config path")
        ensure_file(Path(str(base_config)), "Base config")
    elif base_config:
        ensure_file(Path(str(base_config)), "Base config")
    Path(cfg["paths"]["tmp_root"]).mkdir(parents=True, exist_ok=True)

    data_root = Path(cfg["paths"]["data_root"])
    if strict_paths and task in {"train-base", "train-dpo", "train-nft-vina-sa", "sample-nft"}:
        ensure_dir(data_root, "data_root")
        ensure_file(data_root / "crossdocked_pocket10_pose_split.pt", "Data split")
        ensure_file(
            data_root / "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb",
            "Crossdocked lmdb",
        )

    if task == "train-dpo":
        dpo_data = cfg["inputs"].get("dpo_data")
        dpo_ref_ckpt = cfg["inputs"].get("dpo_ref_ckpt")
        if not dpo_data:
            raise RuntimeConfigError("train-dpo requires inputs.dpo_data")
        if not dpo_ref_ckpt:
            raise RuntimeConfigError("train-dpo requires inputs.dpo_ref_ckpt")
        if strict_paths:
            ensure_file(resolve_path(str(dpo_data), Path(cfg["_meta"]["repo_root"])), "DPO data")
            ensure_file(
                resolve_path(str(dpo_ref_ckpt), Path(cfg["_meta"]["repo_root"])),
                "DPO reference checkpoint",
            )

    if task == "sample-nft":
        ckpt = cfg["inputs"].get("checkpoint")
        if not ckpt:
            raise RuntimeConfigError("sample-nft requires inputs.checkpoint")
        if strict_paths:
            ensure_file(resolve_path(str(ckpt), Path(cfg["_meta"]["repo_root"])), "Sample checkpoint")

    if task == "eval-simple":
        sample_path = cfg["inputs"].get("sample_path")
        if not sample_path:
            raise RuntimeConfigError("eval-simple requires inputs.sample_path")
        if strict_paths:
            ensure_dir(resolve_path(str(sample_path), Path(cfg["_meta"]["repo_root"])), "sample_path")


def _temp_yaml(prefix: str, cfg: Dict[str, Any]) -> Path:
    tmp_root = Path(cfg["paths"]["tmp_root"])
    tmp_root.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yml",
        prefix=f"{prefix}_",
        dir=tmp_root,
        delete=False,
        encoding="utf-8",
    )
    handle.close()
    return Path(handle.name)


def _repo_relative(path: Path, repo_root: Path) -> str:
    path = path.resolve()
    repo_root = repo_root.resolve()
    try:
        rel = path.relative_to(repo_root)
        return f"./{rel.as_posix()}"
    except ValueError:
        return str(path)


def _config_path(cfg: Dict[str, Any], key: str, *parts: str) -> str:
    raw_base = Path(str(cfg["_meta"]["paths_raw"][key]))
    combined = raw_base.joinpath(*parts)
    return combined.as_posix() if combined.is_absolute() else f"./{combined.as_posix().lstrip('./')}"


def prepare_train_base_config(cfg: Dict[str, Any]) -> Path:
    base = read_yaml(Path(cfg["base_config"]))

    base.setdefault("data", {})
    base["data"]["path"] = _config_path(cfg, "data_root", "crossdocked_v1.1_rmsd1.0_pocket10")
    base["data"]["split"] = _config_path(cfg, "data_root", "crossdocked_pocket10_pose_split.pt")

    train_cfg = base.setdefault("train", {})
    if bool(cfg["runtime"].get("smoke", False)):
        train_cfg["max_iters"] = 1
        train_cfg["val_freq"] = 1

    out = _temp_yaml("train_base_runtime", cfg)
    write_yaml(out, base)
    return out


def prepare_train_dpo_config(cfg: Dict[str, Any]) -> Path:
    base = read_yaml(Path(cfg["base_config"]))
    repo_root = Path(cfg["_meta"]["repo_root"])

    base.setdefault("data", {})
    base["data"]["path"] = _config_path(cfg, "data_root", "crossdocked_v1.1_rmsd1.0_pocket10")
    base["data"]["split"] = _config_path(cfg, "data_root", "crossdocked_pocket10_pose_split.pt")

    base.setdefault("model", {})
    base["model"]["ref_model_checkpoint"] = _repo_relative(
        resolve_path(str(cfg["inputs"]["dpo_ref_ckpt"]), repo_root), repo_root
    )

    train_cfg = base.setdefault("train", {})
    train_cfg["protein_root"] = _config_path(cfg, "data_root", "test_set")

    if bool(cfg["runtime"].get("smoke", False)):
        train_cfg["max_iters"] = 1
        train_cfg["val_freq"] = 1
        train_cfg["eval_freq"] = 10**9
        train_cfg["num_eval_pockets"] = 1
        train_cfg["num_eval_samples_per_pocket"] = 1

    out = _temp_yaml("train_dpo_runtime", cfg)
    write_yaml(out, base)
    return out


def prepare_train_nft_vina_sa_config(cfg: Dict[str, Any]) -> Path:
    base = read_yaml(Path(cfg["base_config"]))
    repo_root = Path(cfg["_meta"]["repo_root"])

    base.setdefault("data", {})
    base["data"]["path"] = _config_path(cfg, "data_root", "crossdocked_v1.1_rmsd1.0_pocket10")
    base["data"]["split"] = _config_path(cfg, "data_root", "crossdocked_pocket10_pose_split.pt")

    train_cfg = base.setdefault("train", {})
    nft_cfg = train_cfg.setdefault("nft", {})
    reward_cfg = nft_cfg.setdefault("reward", {})

    rollout_bs = int(nft_cfg.get("rollout_batch_size", train_cfg.get("batch_size", 1)))
    rollout_num_samples = int(nft_cfg.get("rollout_num_samples", 1))
    target_samples = max(1, rollout_bs * rollout_num_samples)

    nft_cfg["grad_accum_steps"] = 1
    nft_cfg["train_batch_size"] = max(1, min(16, target_samples))
    reward_cfg["protein_root"] = _config_path(
        cfg, "data_root", "crossdocked_v1.1_rmsd1.0_pocket10_pdb"
    )
    reward_cfg["tmp_dir"] = _config_path(cfg, "tmp_root")

    if bool(cfg["runtime"].get("smoke", False)):
        train_cfg["max_iters"] = 1
        train_cfg["val_freq"] = 1
        train_cfg["batch_size"] = 1
        nft_cfg["rollout_batch_size"] = 1
        nft_cfg["rollout_num_samples"] = 1
        nft_cfg["rollout_chunk_size"] = 1
        nft_cfg["train_batch_size"] = 1
        nft_cfg["num_inner_epochs"] = 1
        nft_cfg["sample_num_steps"] = 10

    out = _temp_yaml("train_nft_vina_sa_runtime", cfg)
    write_yaml(out, base)
    return out


def prepare_sample_nft_config(cfg: Dict[str, Any]) -> Path:
    base = read_yaml(Path(cfg["base_config"]))
    repo_root = Path(cfg["_meta"]["repo_root"])
    runtime = cfg["runtime"]

    base.setdefault("data", {})
    base["data"]["path"] = _config_path(cfg, "data_root", "crossdocked_v1.1_rmsd1.0_pocket10")
    base["data"]["split"] = _config_path(cfg, "data_root", "crossdocked_pocket10_pose_split.pt")
    base.setdefault("model", {})
    base.setdefault("sample", {})
    base["model"]["checkpoint"] = _repo_relative(
        resolve_path(str(cfg["inputs"]["checkpoint"]), repo_root), repo_root
    )

    num_steps = int(runtime.get("num_steps", 100))
    num_samples = int(runtime.get("num_samples", 100))
    if bool(runtime.get("smoke", False)):
        num_steps = 10
        num_samples = 1

    base["sample"]["num_steps"] = num_steps
    base["sample"]["num_samples"] = num_samples

    out = _temp_yaml("sample_nft_runtime", cfg)
    write_yaml(out, base)
    return out


def runtime_env(cfg: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    repo_root = Path(cfg["_meta"]["repo_root"]).resolve()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing}" if existing else str(repo_root)
    env["OMP_NUM_THREADS"] = str(cfg["runtime"].get("omp_num_threads", 16))
    return env
