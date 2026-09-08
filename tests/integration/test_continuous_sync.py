"""Integration test: Module 3 running continuously against Module 2's mock (and real-ZeroMQ)
telemetry sources, with NO manual synchronization trigger (prompt.md §9, §0.6).

The test never calls any "process a batch now" method itself — it starts the synchronizer once,
sleeps, and observes that D1's stub state keeps growing purely from the background thread's own
activity. This is the direct behavioural proof the diagram/spec require: telemetry arriving ->
D1's current + historical state updating, continuously, without an external driver.
"""

from __future__ import annotations

import json
import threading
import time

import zmq

from src.common.config import load_settings
from src.synchronization.d1_interface import InMemoryD1Stub
from src.synchronization.sync import ContinuousSynchronizer
from src.telemetry.mock_source import MockTelemetrySource
from src.telemetry.preprocessing import TelemetryPreprocessor
from src.telemetry.zmq_source import ZmqTelemetrySource


def test_synchronizer_runs_continuously_against_mock_source_with_no_manual_trigger():
    settings = load_settings()

    # Fast-paced mock so several automatic flush cycles happen within a short test window; the
    # preprocessing/synchronization *config shapes* are still the real pydantic models, only the
    # timing knobs are tuned for test speed.
    mock_config = settings.telemetry.mock.model_copy(
        update={"emit_interval_seconds": 0.02, "num_ues": 4, "num_cells": 1}
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


def test_synchronizer_runs_continuously_over_real_zmq_transport_with_no_manual_trigger():
    """Same proof, but with the real-source path: an actual ZeroMQ PUB socket (standing in for
    the eventual NS-3 exporter) publishing on its own schedule in a background thread, consumed
    by ZmqTelemetrySource, synchronized by the same ContinuousSynchronizer with no manual
    trigger from the test."""
    settings = load_settings()

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    port = pub.bind_to_random_port("tcp://127.0.0.1")
    zmq_config = settings.telemetry.zmq.model_copy(
        update={"endpoint": f"tcp://127.0.0.1:{port}", "recv_timeout_ms": 100}
    )
    sync_config = settings.synchronization.model_copy(
        update={"batch_size": 1000, "batch_timeout_seconds": 0.15}
    )

    generator = MockTelemetrySource(
        settings.telemetry.mock.model_copy(update={"num_ues": 3, "num_cells": 1}),
        max_records=None,
        realtime=False,
    )
    stop_publishing = threading.Event()

    def publish_forever():
        for raw in generator.records():
            if stop_publishing.is_set():
                return
            raw["source"] = "NS3_5G_LENA"
            pub.send_multipart([zmq_config.topic.encode(), json.dumps(raw).encode("utf-8")])
            time.sleep(0.01)

    publisher_thread = threading.Thread(target=publish_forever, daemon=True)

    source = ZmqTelemetrySource(zmq_config)
    preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)
    d1 = InMemoryD1Stub()
    synchronizer = ContinuousSynchronizer(source, preprocessor, d1, sync_config)

    publisher_thread.start()
    time.sleep(0.2)  # slow-joiner
    sync_thread = synchronizer.start()
    try:
        time.sleep(0.6)
        assert synchronizer.batches_synced >= 2
        assert len(d1.get_history()) > 0
        assert all(r.source == "NS3_5G_LENA" for r in d1.get_history())
        assert all(r.source != "MOCK" for r in d1.get_history())
    finally:
        stop_publishing.set()
        synchronizer.stop()
        sync_thread.join(timeout=2.0)
        publisher_thread.join(timeout=2.0)
        pub.close()
        ctx.term()

    assert not sync_thread.is_alive()
