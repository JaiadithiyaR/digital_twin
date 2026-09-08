"""D1 — DT Basic Model: current + historical network state store (Module 4, fig-dataflow.png).

Persistent (Parquet-backed), tabular/time-series representation of the Digital Twin's dynamic
state. Implements Module 3's `D1StateSink` write contract for real (see
`src/synchronization/d1_interface.py`) so `ContinuousSynchronizer` can be pointed at a genuine D1
instead of the in-memory stub used to build/test Module 3 — nothing in `sync.py` needed to
change to make this work.

Scope (prompt.md §0.5 — strict separation of live network state / DT state / versioned models):
this store holds ONLY telemetry-derived Digital Twin state (current + historical) — concept B in
CLAUDE.md §4 ("dynamic DT state"), continuously updated by live telemetry (concept A, "live
network state"). It does NOT hold DT prediction model versions/artifacts (concept C — that is
Module 5's future model registry, `src/registry/model_registry.py`, unrelated to this file) and
it does not fabricate a "network configuration" table: nothing in this system currently produces
network-configuration data (no such module exists yet), so inventing one here would be exactly
the kind of fake/placeholder component prompt.md prohibits. UE state and cell state are exposed
as filtered views over the same current-state table (`get_ue_state`/`get_cell_state`) rather than
separate physical stores, since telemetry already carries `ue_id`/`cell_id`/mobility fields —
adding parallel tables for the same data would be unnecessary infrastructure (prompt.md §10).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.dt_models.component_registry import ComponentDescriptor, ComponentRegistry
from src.synchronization.d1_interface import D1StateSink
from src.telemetry.schema import CleanTelemetryRecord, QuarantinedRecord

if TYPE_CHECKING:
    from src.common.config import Settings

logger = logging.getLogger(__name__)

# Column order mirrors `_record_to_row`'s key order exactly — keep them in sync.
TELEMETRY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "ue_id",
    "cell_id",
    "throughput_mbps",
    "offered_load_mbps",
    "latency_ms",
    "jitter_ms",
    "packet_loss_pct",
    "prb_utilization_pct",
    "sinr_db",
    "rsrp_dbm",
    "rsrq_db",
    "ue_count",
    "ue_speed_mps",
    "ue_position_x",
    "ue_position_y",
    "source",
    "quality_missing_fields",
    "quality_imputed_fields",
    "quality_out_of_range_fields",
)

QUARANTINE_COLUMNS: tuple[str, ...] = ("raw_json", "reason", "source", "received_at")


def _record_to_row(record: CleanTelemetryRecord) -> dict[str, Any]:
    return {
        "timestamp": record.timestamp,
        "ue_id": record.ue_id,
        "cell_id": record.cell_id,
        "throughput_mbps": record.throughput_mbps,
        "offered_load_mbps": record.offered_load_mbps,
        "latency_ms": record.latency_ms,
        "jitter_ms": record.jitter_ms,
        "packet_loss_pct": record.packet_loss_pct,
        "prb_utilization_pct": record.prb_utilization_pct,
        "sinr_db": record.sinr_db,
        "rsrp_dbm": record.rsrp_dbm,
        "rsrq_db": record.rsrq_db,
        "ue_count": record.ue_count,
        "ue_speed_mps": record.ue_speed_mps,
        "ue_position_x": record.ue_position_x,
        "ue_position_y": record.ue_position_y,
        "source": record.source,
        "quality_missing_fields": ",".join(record.quality.missing_fields),
        "quality_imputed_fields": ",".join(record.quality.imputed_fields),
        "quality_out_of_range_fields": ",".join(record.quality.out_of_range_fields),
    }


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
    """`pd.concat` silently downcasts every column to `object` dtype when one side is the
    columns-only, no-rows placeholder `_load_or_empty` returns for a not-yet-created store (an
    empty DataFrame has no data to infer a dtype from, so pandas defaults every column to
    `object`, and that "object wins" rule applies even though the other side is fully typed) —
    a real, reproduced bug (prompt.md §0.26 "invalid numerical states"): every D1Store started
    silently corrupting its own dtypes on its very first write, which stayed corrupted forever
    after (every later concat inherits the already-`object` accumulated frame), invisibly
    breaking anything dtype-sensitive downstream — e.g. XGBoost refuses to fit on `object`
    columns. Skipping the concat entirely when `existing` has no rows sidesteps this: `new_rows`
    keeps its own correctly-inferred dtypes, and every later concat is well-typed-to-well-typed,
    which pandas does NOT corrupt.
    """
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
        self._current_state_df = _load_or_empty(current_state_path, TELEMETRY_COLUMNS)
        self._history_df = _load_or_empty(history_path, TELEMETRY_COLUMNS)
        self._quarantine_df = _load_or_empty(quarantine_path, QUARANTINE_COLUMNS)

        self.registry = registry or ComponentRegistry()
        self.registry.register(
            ComponentDescriptor("telemetry_current", "current_state", TELEMETRY_COLUMNS, current_state_path)
        )
        self.registry.register(
            ComponentDescriptor("telemetry_history", "history", TELEMETRY_COLUMNS, history_path)
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
            # Sorting descending then keeping the first row per (ue_id, cell_id) selects the
            # max-timestamp row regardless of whether it arrived first, last, or within the same
            # batch as another record for the same key.
            combined = combined.sort_values("timestamp", ascending=False, kind="stable")
            self._current_state_df = (
                combined.drop_duplicates(subset=["ue_id", "cell_id"], keep="first")
                .sort_values(["cell_id", "ue_id"], kind="stable")
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
                combined = combined.sort_values("timestamp", kind="stable").tail(self._history_retention_rows)
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

    def get_history(self, ue_id: str | None = None, cell_id: str | None = None) -> pd.DataFrame:
        with self._lock:
            df = self._history_df.copy(deep=True)
        if ue_id is not None:
            df = df[df["ue_id"] == ue_id]
        if cell_id is not None:
            df = df[df["cell_id"] == cell_id]
        return df.reset_index(drop=True)

    def get_quarantined(self) -> pd.DataFrame:
        with self._lock:
            return self._quarantine_df.copy(deep=True)

    def get_ue_state(self, ue_id: str) -> pd.DataFrame:
        """Current state filtered to one UE — a view over the same table, not a separate store."""
        with self._lock:
            df = self._current_state_df.copy(deep=True)
        return df[df["ue_id"] == ue_id].reset_index(drop=True)

    def get_cell_state(self, cell_id: str) -> pd.DataFrame:
        """Current state filtered to one cell (all UEs currently attached to it)."""
        with self._lock:
            df = self._current_state_df.copy(deep=True)
        return df[df["cell_id"] == cell_id].reset_index(drop=True)
