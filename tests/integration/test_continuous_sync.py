"""Integration test: Module 3 running continuously against Module 2's mock (and real OAI replay)
telemetry sources, with NO manual synchronization trigger (prompt.md §9, §0.6).

The test never calls any "process a batch now" method itself — it starts the synchronizer once,
sleeps, and observes that D1's stub state keeps growing purely from the background thread's own
activity. This is the direct behavioural proof the diagram/spec require: telemetry arriving ->
D1's current + historical state updating, continuously, without an external driver.
"""

from __future__ import annotations

import time

import pytest

from src.common.config import load_settings
from src.synchronization.d1_interface import InMemoryD1Stub
from src.synchronization.sync import ContinuousSynchronizer
from src.telemetry.mock_source import MockTelemetrySource
from src.telemetry.oai_replay_source import OaiReplaySource
from src.telemetry.preprocessing import TelemetryPreprocessor


def test_synchronizer_runs_continuously_against_mock_source_with_no_manual_trigger():
    settings = load_settings()

    # Fast-paced mock so several automatic flush cycles happen within a short test window; the
    # preprocessing/synchronization *config shapes* are still the real pydantic models, only the
    # timing knobs are tuned for test speed.
    mock_config = settings.telemetry.mock.model_copy(
        update={"emit_interval_seconds": 0.02, "num_sessions": 4, }
    )
    sync_config = settings.synchronization.model_copy(
        update={"batch_size": 1000, "batch_timeout_seconds": 0.15}  # force timeout-driven flushing
    )

    source = MockTelemetrySource(mock_config, max_records=None, realtime=True)
    preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)
    d1 = InMemoryD1Stub()
    synchronizer = ContinuousSynchronizer(source, preprocessor, d1, sync_config)

    thread = synchronizer.start()
    try:
        time.sleep(0.6)  # ~4 timeout cycles' worth, driven entirely by the background thread

        batches_after_first_wait = synchronizer.batches_synced
        history_after_first_wait = len(d1.get_history())

        # No manual trigger anywhere above — this alone must already reflect multiple automatic
        # synchronization cycles.
        assert batches_after_first_wait >= 2, "expected multiple automatic flushes, saw too few"
        assert history_after_first_wait > 0
        assert len(d1.get_current_state()) > 0
        assert all(r.source == "MOCK" for r in d1.get_history())

        time.sleep(0.4)  # keep waiting — it must keep going on its own
        assert synchronizer.batches_synced > batches_after_first_wait
        assert len(d1.get_history()) > history_after_first_wait
    finally:
        synchronizer.stop()
        thread.join(timeout=2.0)

    assert not thread.is_alive()

    # Confirm it actually stopped (not coincidentally idle between flushes).
    batches_after_stop = synchronizer.batches_synced
    time.sleep(0.3)
    assert synchronizer.batches_synced == batches_after_stop


def test_synchronizer_runs_continuously_over_real_oai_replay_with_no_manual_trigger():
    """Same proof, but with the real-source path: `OaiReplaySource` replaying the REAL processed
    OAI export (never mock), synchronized by the same `ContinuousSynchronizer` with no manual
    trigger from the test. Skips cleanly (not a failure) if the dataset hasn't been built yet."""
    settings = load_settings()
    processed_dir = settings.resolve_path("data/processed/oai")
    if not (processed_dir / "test.parquet").exists():
        pytest.skip("data/processed/oai/test.parquet not built — run scripts/build_oai_dataset.py")

    sync_config = settings.synchronization.model_copy(update={"batch_size": 50, "batch_timeout_seconds": 0.15})

    source = OaiReplaySource(processed_dir, speed=0.0)
    source.start(run_ids=settings.data.splits.test)
    preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)
    d1 = InMemoryD1Stub()
    synchronizer = ContinuousSynchronizer(source, preprocessor, d1, sync_config)

    sync_thread = synchronizer.start()
    try:
        time.sleep(1.0)
        batches_after_first_wait = synchronizer.batches_synced
        assert batches_after_first_wait >= 2
        assert len(d1.get_history()) > 0
        assert all(r.source == "OAI_RAN_REPLAY" for r in d1.get_history())
        assert all(r.source != "MOCK" for r in d1.get_history())
        assert {r.run_id for r in d1.get_history()} <= set(settings.data.splits.test)
    finally:
        synchronizer.stop()
        sync_thread.join(timeout=2.0)

    assert not sync_thread.is_alive()
