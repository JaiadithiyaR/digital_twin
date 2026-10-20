#!/usr/bin/env python3
"""Builds data/processed/oai/{train,test}.parquet + feature_schema.json + data_report.md
from the real OAI export at config.data.oai_root (prompt2.md Step 3). Safe to re-run — always
rebuilds from the raw CSVs, never mutates them.

Phase 13 (claude_code_prompt_phase13.md §0/§1.1): `train` is the full TRAIN/TUNE pool (runs 3,
6, 7 combined, each row still tagged by its own `run_id` column) — leave-one-run-out CV filters
this one file by `run_id` per fold rather than reading separate per-fold files. `test` (run 8)
stays a strictly separate file, never touched before genuine TEST-time evaluation.

Usage:
    python scripts/build_oai_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_settings  # noqa: E402
from src.data.oai_dataset import build_dataset, write_outputs  # noqa: E402


def main() -> int:
    settings = load_settings()
    oai_root = settings.resolve_path(settings.data.oai_root)  # absolute paths pass through unchanged
    if not oai_root.is_dir():
        print(f"[build_oai_dataset] raw OAI export not found at {oai_root} — set OAI_ROOT in .env. "
              "(Not needed to run the system: data/processed/oai/ ships with the repository.)")
        return 1
    splits = {
        "train": settings.data.splits.train,
        "test": settings.data.splits.test,
    }
    print(f"[build_oai_dataset] oai_root={oai_root}")
    print(f"[build_oai_dataset] splits={splits}")

    result = build_dataset(oai_root, splits)

    out_dir = REPO_ROOT / "data" / "processed" / "oai"
    report_path = write_outputs(result, out_dir)

    for name, df in result.splits.items():
        print(f"[build_oai_dataset] {name}: {len(df)} rows, {df['session_id'].nunique() if len(df) else 0} sessions")
    print(f"[build_oai_dataset] wrote {out_dir}/{{train,test}}.parquet, feature_schema.json")
    print(f"[build_oai_dataset] report: {report_path}")
    print(report_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
