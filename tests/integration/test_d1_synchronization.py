"""Integration test: Module 3 (ContinuousSynchronizer) wired to a REAL D1Store (Module 4) — not
the in-memory stub used to build Module 3.

The core proof requested: feed telemetry through Module 3 and confirm D1's current state
updates AND its history appends — not just one or the other. Both are asserted explicitly and
independently, plus checks that they are genuinely different structures (current state is
deduped-by-key, history is not) and that the result is real, on-disk persistence (a fresh
`D1Store` pointed at the same files sees the same data), not merely in-memory bookkeeping.
"""

from __future__ import annotations

import time

from src.common.config import load_settings
from src.dt_models.d1_model_store import D1Store
from src.synchronization.d1_interface import D1StateSink
from src.synchronization.sync import ContinuousSynchronizer
from src.telemetry.mock_source import MockTelemetrySource
from src.telemetry.preprocessing import TelemetryPreprocessor


def test_synchronizer_updates_d1_current_state_and_appends_history_via_real_d1store(tmp_path):
    settings = load_settings()

    store = D1Store(
        current_state_path=tmp_path / "current.parquet",
        history_path=tmp_path / "history.parquet",
        quarantine_path=tmp_path / "quarantine.parquet",
        history_retention_rows=settings.storage.history_retention_rows,
    )
    assert isinstance(store, D1StateSink)  # this IS the Module 3 -> D1 contract, for real

    mock_config = settings.telemetry.mock.model_copy(
        update={"num_ues": 5, "num_cells": 2, "missing_field_rate": 0.0, "out_of_range_rate": 0.0}
    )
    source = MockTelemetrySource(mock_config, max_records=30, realtime=False)
    preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)
    sync_config = settings.synchronization.model_copy(update={"batch_size": 10})
    synchronizer = ContinuousSynchronizer(source, preprocessor, store, sync_config)

    # Bounded source -> run() is a normal blocking call that returns once telemetry is exhausted.
    synchronizer.run()

    current = store.get_current_state()
    history = store.get_history()

    # --- both must have actually happened, not just one of the two ---
    assert len(current) > 0, "D1 current state was never updated"
    assert len(history) > 0, "D1 history was never appended"

    # 5 UEs x 2 cells worth of round-robin assignment -> at most 5 distinct (ue_id, cell_id) keys
    # (each UE maps to exactly one cell), while every one of the 30 raw records should land in
    # history — proving current state and history are genuinely different structures, not the
    # same data viewed twice.
    assert len(history) == 30
    assert len(current) == 5
    assert len(current) < len(history)

    assert set(current.columns) == set(history.columns)
    assert set(current["source"]) == {"MOCK"}
    assert set(history["source"]) == {"MOCK"}

    # current state holds only the LATEST record per key
    for _, row in current.iterrows():
        key_history = history[(history["ue_id"] == row["ue_id"]) & (history["cell_id"] == row["cell_id"])]
        assert row["timestamp"] == key_history["timestamp"].max()

    # --- real persistence, not just in-memory: a fresh D1Store sees the same data on disk ---
    reloaded = D1Store(
        current_state_path=tmp_path / "current.parquet",
        history_path=tmp_path / "history.parquet",
        quarantine_path=tmp_path / "quarantine.parquet",
        history_retention_rows=settings.storage.history_retention_rows,
    )
    assert len(reloaded.get_current_state()) == len(current)
    assert len(reloaded.get_history()) == len(history)


def test_synchronizer_continuously_updates_real_d1store_with_no_manual_trigger(tmp_path):
    """Same 'runs continuously, no manual trigger' proof as Module 3's own tests, but through
    the real D1Store rather than the stub — confirms the real store handles being written from
    a live background thread correctly, not just in a single blocking run() call."""
    settings = load_settings()

    store = D1Store(
        current_state_path=tmp_path / "current.parquet",
        history_path=tmp_path / "history.parquet",
        quarantine_path=tmp_path / "quarantine.parquet",
        history_retention_rows=settings.storage.history_retention_rows,
    )
    mock_config = settings.telemetry.mock.model_copy(
        update={"emit_interval_seconds": 0.02, "num_ues": 3, "num_cells": 1}
    )
    sync_config = settings.synchronization.model_copy(
        update={"batch_size": 1000, "batch_timeout_seconds": 0.15}
    )
    source = MockTelemetrySource(mock_config, max_records=None, realtime=True)
    preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)
    synchronizer = ContinuousSynchronizer(source, preprocessor, store, sync_config)

    thread = synchronizer.start()
    try:
        time.sleep(0.5)
        history_len_1 = len(store.get_history())
        current_len_1 = len(store.get_current_state())
        assert history_len_1 > 0
        assert current_len_1 > 0

        time.sleep(0.4)
        assert len(store.get_history()) > history_len_1  # kept growing on its own
    finally:
        synchronizer.stop()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    # current state stayed bounded (deduped) while history kept accumulating
    assert len(store.get_current_state()) <= 3
    assert len(store.get_history()) >= history_len_1
