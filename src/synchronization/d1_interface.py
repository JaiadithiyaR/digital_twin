"""Module 3's contract with D1 (Module 4 — DT Basic Model, state & history store).

D1 does not exist yet. This module defines the write-side interface `ContinuousSynchronizer`
depends on, so Module 3 can be built, run, and tested against a real (if minimal) contract
instead of being mocked out entirely. When Module 4 is implemented, its D1 state/history store
must implement `D1StateSink` — nothing in `sync.py` should need to change.

Scope boundary (prompt.md §0.7 — distinguish bootstrap / historical / live / evaluation /
candidate-training data): this interface is for LIVE telemetry synchronization only. Bootstrap
data (used once at DT model init) and evaluation/candidate-training data (selected later by
recalibration/verification) are not pushed through this path — they are D1's/other modules'
concerns once those exist, not Module 3's.

Feature-window construction (`TelemetryPreprocessor.build_feature_windows`, Module 2) is
deliberately NOT invoked here on each small sync batch — a window is only meaningful once enough
same-(ue_id, cell_id) history has accumulated, which is exactly what `append_history` is for.
Building windows is a query-time operation over D1's accumulated history (Module 4's/a
consumer's job), not something Module 3 should do repeatedly on tiny batches.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod

from src.telemetry.schema import CleanTelemetryRecord, QuarantinedRecord


class D1StateSink(ABC):
    """Write-side contract for pushing synchronized telemetry into D1."""

    @abstractmethod
    def update_current_state(self, records: list[CleanTelemetryRecord]) -> None:
        """Overwrite the latest known state for each record's (ue_id, cell_id).

        Live telemetry is authoritative for current state (prompt.md §0.7) — implementations
        must never let bootstrap data or a stale/out-of-order record silently become "current".
        """

    @abstractmethod
    def append_history(self, records: list[CleanTelemetryRecord]) -> None:
        """Append to the historical record. Never replace or truncate existing history — the
        Digital Twin's history only grows (prompt.md §0.6: dynamic state, not a periodically
        replaced dataset)."""

    @abstractmethod
    def record_quarantine(self, quarantined: list[QuarantinedRecord]) -> None:
        """Audit sink for records Module 2 rejected. Keeping these (not just logging them) is
        part of "do not silently discard important data" (prompt.md §8) — a persistent data
        quality trail alongside the history it was excluded from."""


class InMemoryD1Stub(D1StateSink):
    """Minimal, thread-safe, in-memory `D1StateSink` for developing and testing Module 3 before
    Module 4 exists. NOT a real D1 implementation: no persistence, no component registry, no
    predictions — just enough to prove the synchronization contract is exercised correctly.

    Thread-safe because `ContinuousSynchronizer` runs in a background thread while tests/other
    code read state from the main thread — this is also what "expose consistent snapshots to
    downstream components" (prompt.md §9) means in practice: a reader must never observe a
    torn/partial write.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_state: dict[tuple[str, str], CleanTelemetryRecord] = {}
        self._history: list[CleanTelemetryRecord] = []
        self._quarantined: list[QuarantinedRecord] = []

    def update_current_state(self, records: list[CleanTelemetryRecord]) -> None:
        with self._lock:
            for record in records:
                key = (record.ue_id, record.cell_id)
                existing = self._current_state.get(key)
                # "Latest" means latest by telemetry timestamp, not arrival order — network
                # delivery can reorder records, and an out-of-order arrival must never regress
                # current state (prompt.md §0.7: telemetry is the authoritative state update).
                if existing is not None and record.timestamp < existing.timestamp:
                    continue
                self._current_state[key] = record

    def append_history(self, records: list[CleanTelemetryRecord]) -> None:
        with self._lock:
            self._history.extend(records)

    def record_quarantine(self, quarantined: list[QuarantinedRecord]) -> None:
        with self._lock:
            self._quarantined.extend(quarantined)

    def get_current_state(self) -> dict[tuple[str, str], CleanTelemetryRecord]:
        """Thread-safe snapshot (shallow copy) — never returns the live internal dict."""
        with self._lock:
            return dict(self._current_state)

    def get_history(self) -> list[CleanTelemetryRecord]:
        with self._lock:
            return list(self._history)

    def get_quarantined(self) -> list[QuarantinedRecord]:
        with self._lock:
            return list(self._quarantined)
