#!/usr/bin/env python3
"""Python-only runtime consistency checks for MolForm."""

from __future__ import annotations

from pathlib import Path
import sys
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = REPO_ROOT / "configs" / "runtime"
EXPECTED_RUNTIME = {
    "train_base.yml": "train-base",
    "train_dpo.yml": "train-dpo",
    "train_nft_vina_sa.yml": "train-nft-vina-sa",
    "sample_nft.yml": "sample-nft",
    "eval_simple.yml": "eval-simple",
}
SCAN_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / ".github" / "workflows" / "cpu-ci.yml",
    *sorted(
        p for p in (REPO_ROOT / "scripts").glob("*.py")
        if p.name != "check_script_refs.py"
    ),
]
FORBIDDEN_SNIPPETS = [
    "script_run/",
    "eval-full",
]


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"runtime config must be a YAML mapping: {path}")
    return data


def main() -> int:
    errors: list[str] = []

    if not RUNTIME_DIR.is_dir():
        errors.append(f"missing runtime config directory: {RUNTIME_DIR}")
    else:
        existing = {p.name for p in RUNTIME_DIR.glob("*.yml")}
        missing = sorted(set(EXPECTED_RUNTIME) - existing)
        extra = sorted(existing - set(EXPECTED_RUNTIME))
        for fn in missing:
            errors.append(f"missing runtime config: configs/runtime/{fn}")
        for fn in extra:
            errors.append(f"unexpected runtime config (not in API): configs/runtime/{fn}")

        for fn, expected_task in EXPECTED_RUNTIME.items():
            path = RUNTIME_DIR / fn
            if not path.exists():
                continue
            try:
                cfg = _read_yaml(path)
            except Exception as exc:
                errors.append(f"failed to parse {path.relative_to(REPO_ROOT)}: {exc}")
                continue

            task = cfg.get("task")
            if task != expected_task:
                errors.append(
                    f"task mismatch in {path.relative_to(REPO_ROOT)}: expected '{expected_task}', got '{task}'"
                )

            base_cfg = cfg.get("base_config")
            if expected_task != "eval-simple":
                if not base_cfg:
                    errors.append(f"missing base_config in {path.relative_to(REPO_ROOT)}")
                    continue
                target = (
                    (REPO_ROOT / str(base_cfg)).resolve()
                    if not Path(str(base_cfg)).is_absolute()
                    else Path(str(base_cfg))
                )
                if not target.exists():
                    errors.append(
                        f"base_config target not found for {path.relative_to(REPO_ROOT)}: {base_cfg}"
                    )
            elif base_cfg:
                target = (
                    (REPO_ROOT / str(base_cfg)).resolve()
                    if not Path(str(base_cfg)).is_absolute()
                    else Path(str(base_cfg))
                )
                if not target.exists():
                    errors.append(
                        f"base_config target not found for {path.relative_to(REPO_ROOT)}: {base_cfg}"
                    )

    for file_path in SCAN_FILES:
        if not file_path.exists():
            continue
        text = file_path.read_text(encoding="utf-8")
        for token in FORBIDDEN_SNIPPETS:
            if token in text:
                errors.append(f"forbidden token '{token}' found in {file_path.relative_to(REPO_ROOT)}")

    shell_files = sorted(REPO_ROOT.rglob("*.sh"))
    for sh_path in shell_files:
        errors.append(f"shell file is not allowed: {sh_path.relative_to(REPO_ROOT)}")

    if errors:
        print("ERROR: python-only checks failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    runtime_count = len(list(RUNTIME_DIR.glob("*.yml"))) if RUNTIME_DIR.exists() else 0
    print(f"OK: python-only checks passed ({runtime_count} runtime configs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
