"""Dedicated, append-only run logs for the live orchestrator — the auditable, machine-readable
counterparts to the human console output (`console_report.py`):

- `ComponentFidelityLog` — CSV, one row per component per fidelity cycle: score, threshold,
  policy state, debounce counter, model version, data window. The per-component fidelity history.
- `AdaptationDecisionLog` — JSON Lines, one line per handled adaptation trigger, INCLUDING the
  ones that never reached verification (agent failure, skipped, verification error), which the
  lifecycle record log (`LifecycleAgent`) by design cannot hold.

Both are plain appends (never rewritten), thread-safe, and create their parent directory.
"""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any

COMPONENT_FIDELITY_COLUMNS: tuple[str, ...] = (
    "logged_at", "cycle", "component", "model_version", "fidelity_score", "threshold", "fidelity_gap",
    "state", "consecutive_below", "min_consecutive_evaluations", "severity", "window_start_ts",
    "window_end_ts", "window_rows", "scored_rows", "unified_score_report_only", "reason",
)


class ComponentFidelityLog:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self._lock:
            new_file = not self._path.exists() or self._path.stat().st_size == 0
            with self._path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=COMPONENT_FIDELITY_COLUMNS, extrasaction="ignore")
                if new_file:
                    writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k) for k in COMPONENT_FIDELITY_COLUMNS})


class AdaptationDecisionLog:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, default=str, sort_keys=False) + "\n"
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)
