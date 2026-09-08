#!/usr/bin/env python3
"""Idempotent local environment bootstrap: directories, .env scaffold, dependency check.

Run after `pip install -r requirements.txt` (see README.md Quick start). Does not install
packages itself — it verifies the environment this repository needs is actually in place, and
fails loudly (not silently) when something required is missing, per prompt.md §0.25 (no false
completion).
"""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "data/bootstrap",
    "data/telemetry",
    "data/evaluation",
    "data/artifacts",
    "data/models",
    "logs",
    "rag_data/oran",
    "rag_data/digital_twin",
    "rag_data/policies",
    "rag_data/history",
]

REQUIRED_IMPORTS = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "xgboost",
    "torch",
    "gymnasium",
    "stable_baselines3",
    "anthropic",
    "chromadb",
    "zmq",
    "yaml",
    "pydantic",
    "pytest",
    "networkx",
    "pyarrow",
]


def ensure_directories() -> None:
    for rel in REQUIRED_DIRS:
        (REPO_ROOT / rel).mkdir(parents=True, exist_ok=True)
    print(f"[setup] verified {len(REQUIRED_DIRS)} required directories")


def ensure_env_file() -> None:
    env_path = REPO_ROOT / ".env"
    example_path = REPO_ROOT / ".env.example"
    if env_path.exists():
        print("[setup] .env already present")
        return
    if not example_path.exists():
        print("[setup] WARNING: .env.example missing, cannot scaffold .env", file=sys.stderr)
        return
    shutil.copy(example_path, env_path)
    print("[setup] created .env from .env.example — fill in ANTHROPIC_API_KEY before using LLM agents")


def check_dependencies() -> bool:
    missing = []
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        print(f"[setup] MISSING DEPENDENCIES: {missing}", file=sys.stderr)
        print("[setup] run: pip install -r requirements.txt", file=sys.stderr)
        return False
    print(f"[setup] verified {len(REQUIRED_IMPORTS)} required Python packages import correctly")
    return True


def check_config_loads() -> bool:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from src.common.config import load_settings

        load_settings()
        print("[setup] config/settings.yaml loads and validates")
        return True
    except Exception as exc:  # noqa: BLE001 — top-level bootstrap check, report and exit
        print(f"[setup] FAILED to load config/settings.yaml: {exc}", file=sys.stderr)
        return False


def main() -> int:
    ensure_directories()
    ensure_env_file()
    deps_ok = check_dependencies()
    config_ok = check_config_loads() if deps_ok else False
    if not (deps_ok and config_ok):
        print("[setup] FAILED — see above", file=sys.stderr)
        return 1
    print("[setup] OK — environment ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
