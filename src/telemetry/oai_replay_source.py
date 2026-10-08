"""Module 1 replacement — OAI RAN KPI replay source (prompt2.md Step 4).

Implements the SAME `TelemetrySource` interface the retired ZeroMQ source used
(`records()`/`close()`) so Modules 2-4 need no redesign to consume it — plus the richer replay
control surface prompt2.md Step 4 asks for (`start`/`pause`/`resume`/`stop`/deterministic seek).

Replays `src/data/oai_dataset.py`'s already-processed, already-lagged, already-unit-converted
per-run tables (`data/processed/oai/{train,val,test}.parquet`) — never the raw CSVs directly,
since Step 3 already did every bit of unit conversion/delta/lag computation a tick needs.

**Causality** (prompt2.md Step 4): a processed-table row at `ts_1s` already carries exactly what
causality requires — its `_lag1` features reflect only state available strictly before this
second, and its raw target columns are the measured ground truth FOR this second. Replaying a row
verbatim as one tick therefore automatically satisfies "predict second t using state up to t-1;
compare against t's own ground truth" with no extra bookkeeping in this class.

**Determinism**: the processed table is already deterministically sorted when built
(`build_run_table`'s own `.sort_values(["run_id","session_id","ts_1s"])`); this class re-sorts by
`(ts_1s, session_id)` before replay so the SAME input always yields the SAME tick sequence
regardless of the table's on-disk row order.

**Gaps**: a second with no data in the source table simply has no tick — `records()` never
invents a row to fill a gap; it only ever emits rows that genuinely exist in the processed table,
and logs when the real wall-clock gap between two consecutive emitted seconds exceeds 1s (i.e. the
underlying data itself had a gap, already reflected upstream in Step 3's own gap-NaN handling for
counters — this is a visibility log, not a repair).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from src.telemetry.base import TelemetrySource

logger = logging.getLogger(__name__)

SOURCE_LABEL = "OAI_RAN_REPLAY"


class OaiReplaySource(TelemetrySource):
    SOURCE_LABEL = SOURCE_LABEL

    def __init__(self, processed_dir: Path, speed: float, default_split: str = "test") -> None:
        self._processed_dir = processed_dir
        self._speed = speed
        self._default_split = default_split
        self._closed = False
        self._paused = threading.Event()  # set = paused
        self._run_ids: list[int] | None = None
        self._start_ts: pd.Timestamp | None = None
        self._table: pd.DataFrame | None = None
        self._tick_groups: list[tuple[pd.Timestamp, pd.DataFrame]] = []
        self._next_tick_index = 0

    # --- replay control surface (prompt2.md Step 4) ---------------------------------------

    def start(self, run_ids: list[int] | None = None, start_ts: str | pd.Timestamp | None = None) -> None:
        """(Re)configures and resets this source to begin replaying `run_ids` (default: every
        run across every split) from `start_ts` (default: the very first available tick)."""
        self._run_ids = run_ids
        self._start_ts = pd.Timestamp(start_ts, tz="UTC") if start_ts is not None else None
        self._closed = False
        self._paused.clear()
        self._load_table()
        self._next_tick_index = 0
        if self._start_ts is not None:
            self.seek(self._start_ts)

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def stop(self) -> None:
        self._closed = True

    def seek(self, timestamp: str | pd.Timestamp) -> None:
        """Deterministic seek: jumps the replay cursor to the first tick at-or-after `timestamp`.
        Never invents a tick — if `timestamp` falls strictly between two real ticks, the cursor
        lands on the next REAL one."""
        target = pd.Timestamp(timestamp, tz="UTC") if not isinstance(timestamp, pd.Timestamp) else timestamp
        for i, (ts, _) in enumerate(self._tick_groups):
            if ts >= target:
                self._next_tick_index = i
                return
        self._next_tick_index = len(self._tick_groups)  # past the end — records() yields nothing

    # --- internal ---------------------------------------------------------------------------

    def _load_table(self) -> None:
        frames = []
        for split in ("train", "val", "test"):
            path = self._processed_dir / f"{split}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path, engine="pyarrow"))
        if not frames:
            raise FileNotFoundError(
                f"no processed OAI split tables found under {self._processed_dir} — "
                "run scripts/build_oai_dataset.py first"
            )
        table = pd.concat(frames, ignore_index=True)
        if self._run_ids is not None:
            table = table[table["run_id"].isin(self._run_ids)]
        table = table.sort_values(["ts_1s", "session_id"], kind="stable").reset_index(drop=True)
        self._table = table
        self._tick_groups = list(table.groupby("ts_1s", sort=True))
        logger.info(
            "OAI replay source loaded",
            extra={"component": "telemetry", "source": SOURCE_LABEL, "run_ids": self._run_ids,
                   "rows": len(table), "distinct_ticks": len(self._tick_groups)},
        )

    # --- TelemetrySource interface -----------------------------------------------------------

    def records(self) -> Iterator[dict[str, Any]]:
        if self._table is None:
            self.start(run_ids=self._run_ids)

        prev_ts: pd.Timestamp | None = None
        while self._next_tick_index < len(self._tick_groups):
            if self._closed:
                return
            while self._paused.is_set() and not self._closed:
                time.sleep(0.05)
            if self._closed:
                return

            ts, group = self._tick_groups[self._next_tick_index]
            self._next_tick_index += 1

            if prev_ts is not None:
                gap_seconds = (ts - prev_ts).total_seconds()
                if gap_seconds > 1.5:
                    logger.info(
                        "OAI replay gap: no tick(s) in the source data",
                        extra={"component": "telemetry", "source": SOURCE_LABEL, "gap_seconds": gap_seconds,
                               "previous_tick": str(prev_ts), "next_tick": str(ts)},
                    )
            if prev_ts is not None and self._speed > 0:
                # Pace BETWEEN ticks (never before the very first), so `speed=1.0` genuinely
                # means "one tick per wall-clock second" rather than front-loading a spurious
                # delay before any data is emitted at all.
                time.sleep(1.0 / self._speed)
            prev_ts = ts

            for _, row in group.iterrows():
                if self._closed:
                    return
                record = row.to_dict()
                record["source"] = SOURCE_LABEL
                yield record

    def close(self) -> None:
        self._closed = True
