"""D1 — DT Basic Model: current + historical network state store.

Persistent (Parquet-backed) tabular/time-series representation of the Digital Twin's dynamic
state. Implements `D1StateSink` so `ContinuousSynchronizer` can be pointed at a real store
instead of an in-memory stub.

Scope: holds ONLY telemetry-derived DT state (current + historical) — never model
versions/artifacts (that's `src/registry/model_registry.py`). UE/cell state are filtered views
over the same current-state table, not separate physical stores.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.data.oai_dataset import TARGET_COLUMNS
from src.dt_models.component_registry import ComponentDescriptor, ComponentRegistry
from src.synchronization.d1_interface import D1StateSink
from src.telemetry.schema import CleanTelemetryRecord, QuarantinedRecord

if TYPE_CHECKING:
    from src.common.config import Settings

logger = logging.getLogger(__name__)

# Identity columns always present; the feature set itself is dynamic (declared by
# feature_schema.json), so D1's column list is assembled from the first batch of records seen.
IDENTITY_COLUMNS: tuple[str, ...] = ("run_id", "cell_id", "session_id", "ue_id", "slice_id", "ts_1s")
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    *IDENTITY_COLUMNS, *TARGET_COLUMNS, "source", "quality_missing_fields",
)

QUARANTINE_COLUMNS: tuple[str, ...] = ("raw_json", "reason", "source", "received_at")


def _record_to_row(record: CleanTelemetryRecord) -> dict[str, Any]:
    """`None` is coerced to `float("nan")`, never left as Python `None`: a batch where every row
    has `None` for some column makes `pd.DataFrame([...])` infer `object` dtype for it (no
    numeric value to infer from), corrupting that column permanently once concatenated.
    `float("nan")` always infers as `float64` regardless of row count."""
    row: dict[str, Any] = {
        "run_id": record.run_id,
        "cell_id": record.cell_id,
        "session_id": record.session_id,
        "ue_id": record.ue_id,
        "slice_id": record.slice_id,
        "ts_1s": record.ts_1s,
        "source": record.source,
        "quality_missing_fields": ",".join(record.quality.missing_fields),
    }
    row.update({k: (float("nan") if v is None else v) for k, v in record.targets.items()})
    row.update({k: (float("nan") if v is None else v) for k, v in record.features.items()})
    return row


def _quarantined_to_row(quarantined: QuarantinedRecord) -> dict[str, Any]:
    return {
        "raw_json": json.dumps(quarantined.raw, default=str),
        "reason": quarantined.reason,
        "source": quarantined.source,
        "received_at": quarantined.received_at,
    }


def _load_or_empty(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path, engine="pyarrow")
    return pd.DataFrame(columns=list(columns))


def _concat_preserving_dtypes(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """`pd.concat` downcasts every column to `object` dtype when one side is the empty,
    columns-only placeholder `_load_or_empty` returns for a not-yet-created store — and that
    corruption persists forever after (every later concat inherits the `object` frame), breaking
    anything dtype-sensitive downstream (e.g. XGBoost refuses `object` columns). Skipping the
    concat when `existing` has no rows sidesteps this entirely."""
    if existing.empty:
        return new_rows
    return pd.concat([existing, new_rows], ignore_index=True)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write-to-temp-then-rename so a crash mid-write never corrupts the previous good file —
    `Path.replace` is atomic on POSIX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, engine="pyarrow", index=False)
    tmp_path.replace(path)


class D1Store(D1StateSink):
    """Persistent current + historical telemetry state, with a modular component registry.

    Thread-safe: `ContinuousSynchronizer` writes from its background thread while other code
    (tests, future orchestrator/fidelity modules) may read concurrently from other threads.
    """

    def __init__(
        self,
        current_state_path: Path,
        history_path: Path,
        quarantine_path: Path,
        history_retention_rows: int,
        registry: ComponentRegistry | None = None,
    ) -> None:
        self._current_state_path = current_state_path
        self._history_path = history_path
        self._quarantine_path = quarantine_path
        self._history_retention_rows = history_retention_rows
        self._lock = threading.Lock()

        # Persistent: reload existing state on construction so a process restart resumes rather
        # than silently starting from an empty Digital Twin (prompt.md §10 "persistent" layer).
        # The feature SET is dynamic (OAI migration: declared by feature_schema.json, not fixed
        # here) — an empty/not-yet-created store starts with just the always-present columns;
        # the real feature columns are adopted organically on the first real write.
        self._current_state_df = _load_or_empty(current_state_path, NON_FEATURE_COLUMNS)
        self._history_df = _load_or_empty(history_path, NON_FEATURE_COLUMNS)
        self._quarantine_df = _load_or_empty(quarantine_path, QUARANTINE_COLUMNS)

        self.registry = registry or ComponentRegistry()
        self.registry.register(
            ComponentDescriptor("telemetry_current", "current_state", NON_FEATURE_COLUMNS, current_state_path)
        )
        self.registry.register(
            ComponentDescriptor("telemetry_history", "history", NON_FEATURE_COLUMNS, history_path)
        )
        self.registry.register(
            ComponentDescriptor("telemetry_quarantine", "audit", QUARANTINE_COLUMNS, quarantine_path)
        )

        logger.info(
            "D1 store initialized",
            extra={
                "component": "d1",
                "current_state_rows": len(self._current_state_df),
                "history_rows": len(self._history_df),
                "quarantine_rows": len(self._quarantine_df),
            },
        )

    @classmethod
    def from_settings(cls, settings: "Settings") -> "D1Store":
        storage = settings.storage
        return cls(
            current_state_path=settings.resolve_path(storage.d1_current_state_path),
            history_path=settings.resolve_path(storage.d1_history_path),
            quarantine_path=settings.resolve_path(storage.d1_quarantine_path),
            history_retention_rows=storage.history_retention_rows,
        )

    # --- D1StateSink (Module 3's write contract) -----------------------------------------

    def update_current_state(self, records: list[CleanTelemetryRecord]) -> None:
        if not records:
            return
        new_rows = pd.DataFrame([_record_to_row(r) for r in records])
        with self._lock:
            combined = _concat_preserving_dtypes(self._current_state_df, new_rows)
            # "Latest" means latest by telemetry timestamp, not arrival order (prompt.md §0.7).
            # Sorting descending then keeping the first row per (run_id, session_id) selects the
            # max-timestamp row regardless of arrival order. `(run_id, session_id)` — NOT
            # `(ue_id, cell_id)` — is the real identity key here: `ue_id` values repeat ACROSS
            # different OAI runs (e.g. run3's ue_id=1 and run8's ue_id=1 are different, unrelated
            # UEs), while `session_id` is genuinely unique per run (verified against the real
            # export's `dim_ue_session` tables) — deduping on `(ue_id, cell_id)` alone would
            # silently collapse different runs' UEs into one "current state" row.
            combined = combined.sort_values("ts_1s", ascending=False, kind="stable")
            self._current_state_df = (
                combined.drop_duplicates(subset=["run_id", "session_id"], keep="first")
                .sort_values(["run_id", "cell_id", "session_id"], kind="stable")
                .reset_index(drop=True)
            )
            _atomic_write_parquet(self._current_state_df, self._current_state_path)
        logger.info(
            "D1 current state updated",
            extra={"component": "d1", "records": len(records), "distinct_keys": len(self._current_state_df)},
        )

    def append_history(self, records: list[CleanTelemetryRecord]) -> None:
        if not records:
            return
        new_rows = pd.DataFrame([_record_to_row(r) for r in records])
        with self._lock:
            combined = _concat_preserving_dtypes(self._history_df, new_rows)
            if len(combined) > self._history_retention_rows:
                combined = combined.sort_values("ts_1s", kind="stable").tail(self._history_retention_rows)
            self._history_df = combined.reset_index(drop=True)
            _atomic_write_parquet(self._history_df, self._history_path)
        logger.info(
            "D1 history appended", extra={"component": "d1", "records": len(records), "total_history_rows": len(self._history_df)}
        )

    def record_quarantine(self, quarantined: list[QuarantinedRecord]) -> None:
        if not quarantined:
            return
        new_rows = pd.DataFrame([_quarantined_to_row(q) for q in quarantined])
        with self._lock:
            self._quarantine_df = _concat_preserving_dtypes(self._quarantine_df, new_rows)
            _atomic_write_parquet(self._quarantine_df, self._quarantine_path)
        logger.info(
            "D1 quarantine recorded",
            extra={"component": "d1", "records": len(quarantined), "total_quarantine_rows": len(self._quarantine_df)},
        )

    # --- Read API (D1 "exposes consistent snapshots to downstream components") ------------

    def get_current_state(self) -> pd.DataFrame:
        with self._lock:
            return self._current_state_df.copy(deep=True)

    def get_history(
        self, ue_id: int | None = None, cell_id: int | None = None, run_id: int | None = None,
        session_id: int | None = None,
    ) -> pd.DataFrame:
        with self._lock:
            df = self._history_df.copy(deep=True)
        if ue_id is not None:
            df = df[df["ue_id"] == ue_id]
        if cell_id is not None:
            df = df[df["cell_id"] == cell_id]
        if run_id is not None:
            df = df[df["run_id"] == run_id]
        if session_id is not None:
            df = df[df["session_id"] == session_id]
        return df.reset_index(drop=True)

    def get_quarantined(self) -> pd.DataFrame:
        with self._lock:
            return self._quarantine_df.copy(deep=True)

    def get_ue_state(self, ue_id: int) -> pd.DataFrame:
        """Current state filtered to one UE — a view over the same table, not a separate store."""
        with self._lock:
            df = self._current_state_df.copy(deep=True)
        return df[df["ue_id"] == ue_id].reset_index(drop=True)

    def get_cell_state(self, cell_id: int) -> pd.DataFrame:
        """Current state filtered to one cell (all UEs currently attached to it)."""
        with self._lock:
            df = self._current_state_df.copy(deep=True)
        return df[df["cell_id"] == cell_id].reset_index(drop=True)
