"""Unit tests for D1Store (Module 4, src/dt_models/d1_model_store.py).

Uses real filesystem I/O (pytest's `tmp_path`) — genuine Parquet reads/writes, not mocked, so
these tests prove actual persistence (survives a fresh `D1Store` instance pointed at the same
files), not just in-memory bookkeeping.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.dt_models.d1_model_store import D1Store
from src.synchronization.d1_interface import D1StateSink
from src.telemetry.schema import CleanTelemetryRecord, QuarantinedRecord, RecordQuality


def _clean_record(**overrides) -> CleanTelemetryRecord:
    base = dict(
        run_id=1, cell_id=1, session_id=1, ue_id=1, slice_id=1,
        ts_1s=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        features={"pusch_snr_avg_lag1": 15.0}, targets={"dl_throughput": 50.0, "dl_bler": 0.05},
        source="MOCK", quality=RecordQuality(),
    )
    base.update(overrides)
    return CleanTelemetryRecord(**base)


def _quarantined(**overrides) -> QuarantinedRecord:
    base = dict(
        raw={"source": "MOCK", "session_id": 1, "note": "bad"},
        reason="test_reason",
        source="MOCK",
        received_at=datetime.now(UTC),
    )
    base.update(overrides)
    return QuarantinedRecord(**base)


def _store(tmp_path: Path, retention: int = 500) -> D1Store:
    return D1Store(
        current_state_path=tmp_path / "current.parquet",
        history_path=tmp_path / "history.parquet",
        quarantine_path=tmp_path / "quarantine.parquet",
        history_retention_rows=retention,
    )


# --- contract + registry --------------------------------------------------------------------


def test_d1_store_implements_d1_state_sink(tmp_path):
    assert isinstance(_store(tmp_path), D1StateSink)


def test_component_registry_has_three_components_with_correct_kinds_and_paths(tmp_path):
    store = _store(tmp_path)
    components = {d.name: d for d in store.registry.list_components()}
    assert set(components) == {"telemetry_current", "telemetry_history", "telemetry_quarantine"}
    assert components["telemetry_current"].kind == "current_state"
    assert components["telemetry_history"].kind == "history"
    assert components["telemetry_quarantine"].kind == "audit"
    assert components["telemetry_current"].storage_path == tmp_path / "current.parquet"
    assert components["telemetry_history"].storage_path == tmp_path / "history.parquet"
    assert components["telemetry_quarantine"].storage_path == tmp_path / "quarantine.parquet"


# --- current state: writes, latest-by-timestamp, persistence -------------------------------


def test_update_current_state_writes_and_persists_to_disk(tmp_path):
    store = _store(tmp_path)
    store.update_current_state([_clean_record()])

    assert (tmp_path / "current.parquet").exists()
    current = store.get_current_state()
    assert len(current) == 1
    assert current.iloc[0]["session_id"] == 1
    assert current.iloc[0]["dl_throughput"] == 50.0


def test_current_state_reloads_from_disk_in_a_fresh_instance(tmp_path):
    store = _store(tmp_path)
    store.update_current_state([_clean_record(session_id=1), _clean_record(session_id=2)])

    reloaded = _store(tmp_path)  # brand new instance, same paths — simulates a process restart
    current = reloaded.get_current_state()
    assert len(current) == 2
    assert set(current["session_id"]) == {1, 2}


def test_current_state_keeps_latest_by_timestamp_not_arrival_order(tmp_path):
    store = _store(tmp_path)
    newer = _clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC), targets={"dl_throughput": 20.0})
    older_arriving_late = _clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC), targets={"dl_throughput": 10.0})

    store.update_current_state([newer])
    store.update_current_state([older_arriving_late])

    current = store.get_current_state()
    assert len(current) == 1
    assert current.iloc[0]["dl_throughput"] == 20.0


def test_current_state_dedups_within_a_single_batch_too(tmp_path):
    store = _store(tmp_path)
    a = _clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC), targets={"dl_throughput": 1.0})
    b = _clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC), targets={"dl_throughput": 2.0})
    store.update_current_state([a, b])  # same (run_id, session_id), both in one call
    current = store.get_current_state()
    assert len(current) == 1
    assert current.iloc[0]["dl_throughput"] == 2.0


def test_current_state_does_not_collide_across_different_runs(tmp_path):
    """`(run_id, session_id)` — NOT `(ue_id, cell_id)` — is the real identity key: `ue_id` values
    repeat across different OAI runs, so two different runs' session 1 must never be treated as
    the same current-state row."""
    store = _store(tmp_path)
    store.update_current_state([_clean_record(run_id=1, session_id=1, ue_id=1)])
    store.update_current_state([_clean_record(run_id=2, session_id=1, ue_id=1)])
    current = store.get_current_state()
    assert len(current) == 2


def test_update_current_state_with_empty_list_is_a_noop(tmp_path):
    store = _store(tmp_path)
    store.update_current_state([])
    assert len(store.get_current_state()) == 0
    assert not (tmp_path / "current.parquet").exists()


# --- history: append-only, persistence, retention pruning ----------------------------------


def test_append_history_accumulates_and_persists(tmp_path):
    store = _store(tmp_path)
    store.append_history([_clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC))])
    store.append_history([_clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC))])

    assert len(store.get_history()) == 2
    reloaded = _store(tmp_path)
    assert len(reloaded.get_history()) == 2


def test_history_keeps_every_record_even_when_current_state_dedups(tmp_path):
    store = _store(tmp_path)
    older = _clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC), targets={"dl_throughput": 10.0})
    newer = _clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC), targets={"dl_throughput": 20.0})
    store.append_history([older])
    store.append_history([newer])
    store.update_current_state([older])
    store.update_current_state([newer])

    assert len(store.get_history()) == 2
    assert len(store.get_current_state()) == 1


def test_history_retention_prunes_oldest_rows(tmp_path):
    store = _store(tmp_path, retention=3)
    for i in range(5):
        store.append_history([_clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, i, tzinfo=UTC), session_id=i)])

    history = store.get_history()
    assert len(history) == 3
    # oldest two (session 0, 1) pruned; most recent three retained
    assert set(history["session_id"]) == {2, 3, 4}


def test_get_history_filters_by_ue_and_cell(tmp_path):
    store = _store(tmp_path)
    store.append_history([_clean_record(ue_id=1, cell_id=1, session_id=1)])
    store.append_history([_clean_record(ue_id=2, cell_id=1, session_id=2)])
    store.append_history([_clean_record(ue_id=1, cell_id=2, session_id=3)])

    assert len(store.get_history(ue_id=1)) == 2
    assert len(store.get_history(cell_id=1)) == 2
    assert len(store.get_history(ue_id=1, cell_id=2)) == 1


def test_redelivered_tick_replaces_rather_than_duplicates_history_even_across_restart(tmp_path):
    """A restarted replay re-sends ticks D1 already persisted — history must stay one row per
    (run_id, session_id, ts_1s), keeping the latest delivery."""
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    store = _store(tmp_path)
    store.append_history([_clean_record(ts_1s=ts, targets={"dl_throughput": 10.0})])
    restarted = _store(tmp_path)
    restarted.append_history([
        _clean_record(ts_1s=ts, targets={"dl_throughput": 11.0}),
        _clean_record(ts_1s=ts, session_id=99),  # a genuinely different session at the same second
    ])

    history = restarted.get_history()
    assert len(history) == 2
    assert history.loc[history["session_id"] != 99, "dl_throughput"].tolist() == [11.0]


def test_rate_limited_persistence_keeps_memory_current_and_flush_writes_the_tail(tmp_path):
    store = D1Store(
        current_state_path=tmp_path / "current.parquet", history_path=tmp_path / "history.parquet",
        quarantine_path=tmp_path / "quarantine.parquet", history_retention_rows=500,
        persist_min_interval_seconds=3600.0,
    )
    store.append_history([_clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC))])  # first write persists
    store.append_history([_clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC))])  # within interval: memory only

    assert len(store.get_history()) == 2  # readers are never behind
    assert len(_store(tmp_path).get_history()) == 1  # disk lags by design

    store.flush()
    assert len(_store(tmp_path).get_history()) == 2


def test_append_history_with_empty_list_is_a_noop(tmp_path):
    store = _store(tmp_path)
    store.append_history([])
    assert len(store.get_history()) == 0
    assert not (tmp_path / "history.parquet").exists()


# --- quarantine audit trail: persistence + raw payload round-trip --------------------------


def test_record_quarantine_persists_and_round_trips_raw_payload(tmp_path):
    store = _store(tmp_path)
    store.record_quarantine([_quarantined(raw={"a": 1, "b": "x"}, reason="invalid_source")])

    quarantined = store.get_quarantined()
    assert len(quarantined) == 1
    assert quarantined.iloc[0]["reason"] == "invalid_source"
    assert json.loads(quarantined.iloc[0]["raw_json"]) == {"a": 1, "b": "x"}

    reloaded = _store(tmp_path)
    assert len(reloaded.get_quarantined()) == 1


# --- UE / cell state views -------------------------------------------------------------------


def test_get_ue_state_filters_current_state_by_ue(tmp_path):
    store = _store(tmp_path)
    store.update_current_state([_clean_record(ue_id=1, cell_id=1, session_id=1)])
    store.update_current_state([_clean_record(ue_id=2, cell_id=1, session_id=2)])
    ue_state = store.get_ue_state(1)
    assert len(ue_state) == 1
    assert ue_state.iloc[0]["ue_id"] == 1


def test_get_cell_state_filters_current_state_by_cell(tmp_path):
    store = _store(tmp_path)
    store.update_current_state([_clean_record(ue_id=1, cell_id=1, session_id=1)])
    store.update_current_state([_clean_record(ue_id=2, cell_id=1, session_id=2)])
    store.update_current_state([_clean_record(ue_id=3, cell_id=2, session_id=3)])
    cell_state = store.get_cell_state(1)
    assert len(cell_state) == 2
    assert set(cell_state["ue_id"]) == {1, 2}


# --- snapshot isolation ------------------------------------------------------------------------


def test_get_current_state_returns_a_copy_not_a_live_reference(tmp_path):
    store = _store(tmp_path)
    store.update_current_state([_clean_record()])
    snapshot = store.get_current_state()
    snapshot.loc[0, "dl_throughput"] = 999.0
    assert store.get_current_state().iloc[0]["dl_throughput"] == 50.0


# --- dtype integrity (regression: see CLAUDE.md/IMPLEMENTATION_STATUS.md Module 6 for the bug) --


def test_current_state_numeric_columns_are_not_object_dtype_after_first_write(tmp_path):
    """Regression test: `pd.concat` against `_load_or_empty`'s columns-only placeholder used to
    silently downcast every column to `object` dtype on a store's very first write — invisible
    to scalar-equality tests, but fatal to anything dtype-sensitive downstream (e.g. XGBoost
    refuses to fit on `object` columns). See Module 6's ThroughputModel integration test, which
    is what actually surfaced this — and the OAI migration's own `None`-coercion regression
    (`_record_to_row`), surfaced the exact same way against `requested_prb_ratio_lag1`."""
    store = _store(tmp_path)
    store.update_current_state([_clean_record()])
    current = store.get_current_state()
    for column in ("dl_throughput", "dl_bler", "pusch_snr_avg_lag1"):
        assert current[column].dtype == "float64", f"{column} has dtype {current[column].dtype}"
    assert current["session_id"].dtype == "int64"


def test_history_numeric_columns_are_not_object_dtype_after_first_write(tmp_path):
    store = _store(tmp_path)
    store.append_history([_clean_record()])
    history = store.get_history()
    for column in ("dl_throughput", "dl_bler", "pusch_snr_avg_lag1"):
        assert history[column].dtype == "float64", f"{column} has dtype {history[column].dtype}"
    assert history["session_id"].dtype == "int64"


def test_none_only_column_in_a_batch_still_infers_float64_not_object(tmp_path):
    """OAI-specific regression: a feature that is `None` for EVERY row in a batch (e.g.
    `requested_prb_ratio_lag1` before any controller deployment exists) must still infer
    `float64`, never `object` — `_record_to_row` coerces `None` to `float("nan")` for exactly
    this reason."""
    store = _store(tmp_path)
    records = [_clean_record(session_id=i, features={"requested_prb_ratio_lag1": None}) for i in range(3)]
    store.update_current_state(records)
    current = store.get_current_state()
    assert current["requested_prb_ratio_lag1"].dtype == "float64"
    assert current["requested_prb_ratio_lag1"].isna().all()


def test_dtypes_remain_correct_across_multiple_batches(tmp_path):
    store = _store(tmp_path)
    for i in range(5):
        store.append_history([_clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, i, tzinfo=UTC), session_id=i)])
        store.update_current_state([_clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, i, tzinfo=UTC), session_id=i)])
    assert store.get_history()["dl_throughput"].dtype == "float64"
    assert store.get_current_state()["dl_throughput"].dtype == "float64"


def test_dtypes_remain_correct_after_reload_from_disk(tmp_path):
    store = _store(tmp_path)
    store.update_current_state([_clean_record()])
    store.append_history([_clean_record()])
    reloaded = _store(tmp_path)
    assert reloaded.get_current_state()["dl_throughput"].dtype == "float64"
    assert reloaded.get_history()["dl_throughput"].dtype == "float64"
