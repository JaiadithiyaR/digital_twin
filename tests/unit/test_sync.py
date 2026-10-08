"""Unit tests for ContinuousSynchronizer (Module 3, src/synchronization/sync.py).

Deterministic tests use a bounded source (`max_records` set, `realtime=False`) so `run()`
returns naturally once exhausted — no timing/threading involved. The "runs continuously with no
manual trigger" behaviour itself is proven separately in
tests/integration/test_continuous_sync.py against an unbounded background-threaded source.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

from src.common.config import MockTelemetryConfig, OaiPreprocessingConfig, SynchronizationConfig
from src.synchronization.d1_interface import InMemoryD1Stub
from src.telemetry.base import TelemetrySource
from src.telemetry.mock_source import MockTelemetrySource
from src.telemetry.preprocessing import TelemetryPreprocessor


class _FixedListSource(TelemetrySource):
    """Test double: yields a fixed, finite list of raw records exactly once."""

    SOURCE_LABEL = "MOCK"

    def __init__(self, raw_records: list[dict[str, Any]]) -> None:
        self._raw_records = raw_records
        self.closed = False

    def records(self) -> Iterator[dict[str, Any]]:
        yield from self._raw_records

    def close(self) -> None:
        self.closed = True


def _preprocessing_config(**overrides) -> OaiPreprocessingConfig:
    base = dict(max_missing_ratio=0.2)
    base.update(overrides)
    return OaiPreprocessingConfig(**base)


def _mock_config(**overrides) -> MockTelemetryConfig:
    base = dict(seed=42, num_sessions=5, emit_interval_seconds=0.01, missing_feature_rate=0.0)
    base.update(overrides)
    return MockTelemetryConfig(**base)


def _sync_config(**overrides) -> SynchronizationConfig:
    base = dict(batch_size=5, batch_timeout_seconds=100.0, shutdown_join_timeout_seconds=5.0)  # timeout effectively disabled by default
    base.update(overrides)
    return SynchronizationConfig(**base)


def _raw_record(**overrides) -> dict[str, Any]:
    base = dict(
        source="MOCK", run_id=1, cell_id=1, session_id=1, ue_id=1, slice_id=1,
        ts_1s=1_700_000_000.0, dl_throughput=25.0, ul_throughput=5.0, dl_rlc_delay=2.0,
        dl_prb=30.0, ul_prb=5.0, dl_bler=0.05, ul_bler=0.02,
    )
    base.update(overrides)
    return base


# --- correctness against a bounded mock source ------------------------------------------------


def test_run_synchronizes_every_record_from_a_bounded_source():
    from src.synchronization.sync import ContinuousSynchronizer

    source = MockTelemetrySource(_mock_config(num_sessions=5), max_records=15, realtime=False)
    d1 = InMemoryD1Stub()
    synchronizer = ContinuousSynchronizer(
        source, TelemetryPreprocessor(_preprocessing_config()), d1, _sync_config(batch_size=5)
    )

    synchronizer.run()

    assert synchronizer.batches_synced == 3  # 15 records / batch_size 5
    assert synchronizer.records_synced == 15
    assert len(d1.get_history()) == 15
    assert len(d1.get_current_state()) == 5  # 5 sessions -> 5 distinct keys


def test_trailing_partial_batch_is_still_flushed_not_dropped():
    from src.synchronization.sync import ContinuousSynchronizer

    source = MockTelemetrySource(_mock_config(num_sessions=4), max_records=12, realtime=False)
    d1 = InMemoryD1Stub()
    synchronizer = ContinuousSynchronizer(
        source, TelemetryPreprocessor(_preprocessing_config()), d1, _sync_config(batch_size=5)
    )

    synchronizer.run()

    assert synchronizer.batches_synced == 3  # 5 + 5 + 2 — trailing partial batch not lost
    assert synchronizer.records_synced == 12
    assert len(d1.get_history()) == 12


def test_quarantined_records_reach_the_d1_audit_sink_not_just_logs():
    from src.synchronization.sync import ContinuousSynchronizer

    good = _raw_record(session_id=1)
    bad = _raw_record(session_id=2, source="NOT_A_REAL_SOURCE")
    source = _FixedListSource([good, bad])
    d1 = InMemoryD1Stub()
    synchronizer = ContinuousSynchronizer(
        source, TelemetryPreprocessor(_preprocessing_config()), d1, _sync_config(batch_size=2)
    )

    synchronizer.run()

    assert len(d1.get_history()) == 1
    quarantined = d1.get_quarantined()
    assert len(quarantined) == 1
    assert "invalid_source" in quarantined[0].reason


def test_a_batch_with_only_quarantined_records_still_counts_as_synced():
    from src.synchronization.sync import ContinuousSynchronizer

    source = _FixedListSource([_raw_record(source="BAD")])
    d1 = InMemoryD1Stub()
    synchronizer = ContinuousSynchronizer(
        source, TelemetryPreprocessor(_preprocessing_config()), d1, _sync_config(batch_size=1)
    )
    synchronizer.run()
    assert synchronizer.batches_synced == 1
    assert len(d1.get_current_state()) == 0
    assert len(d1.get_quarantined()) == 1


def test_empty_source_produces_no_batches():
    from src.synchronization.sync import ContinuousSynchronizer

    source = _FixedListSource([])
    d1 = InMemoryD1Stub()
    synchronizer = ContinuousSynchronizer(
        source, TelemetryPreprocessor(_preprocessing_config()), d1, _sync_config()
    )
    synchronizer.run()
    assert synchronizer.batches_synced == 0
    assert synchronizer.records_synced == 0


# --- stop() halts a background thread cleanly --------------------------------------------------


def test_stop_halts_background_thread_promptly():
    from src.synchronization.sync import ContinuousSynchronizer

    source = MockTelemetrySource(
        _mock_config(num_sessions=2, emit_interval_seconds=0.01), max_records=None, realtime=True
    )
    d1 = InMemoryD1Stub()
    synchronizer = ContinuousSynchronizer(
        source, TelemetryPreprocessor(_preprocessing_config()), d1, _sync_config(batch_size=2, batch_timeout_seconds=0.05)
    )

    thread = synchronizer.start()
    time.sleep(0.2)
    assert thread.is_alive()  # confirms it was actually running continuously, not already done

    synchronizer.stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()


def test_run_returns_when_stop_called_before_any_flush_threshold():
    """stop() must interrupt the loop even mid-accumulation (buffer not yet flushed by size or
    timeout) rather than hanging forever waiting for more data."""
    from src.synchronization.sync import ContinuousSynchronizer

    source = MockTelemetrySource(
        _mock_config(num_sessions=1, emit_interval_seconds=0.05), max_records=None, realtime=True
    )
    d1 = InMemoryD1Stub()
    synchronizer = ContinuousSynchronizer(
        source, TelemetryPreprocessor(_preprocessing_config()), d1, _sync_config(batch_size=10_000, batch_timeout_seconds=10_000)
    )

    thread = threading.Thread(target=synchronizer.run, daemon=True)
    thread.start()
    time.sleep(0.15)
    synchronizer.stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
