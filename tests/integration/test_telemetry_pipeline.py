"""Integration tests: full Module 2 pipeline wired together using real `config/settings.yaml`
values — MockTelemetrySource / ZmqTelemetrySource -> TelemetryPreprocessor -> feature windows.

Proves the diagram edge end-to-end: "raw telemetry" in, "validated feature records / time-series
windows (timestamp, UE ID, cell ID)" out (prompt.md §8, fig-dataflow.png Module 2 -> D1 edge).
"""

from __future__ import annotations

import json
import threading
import time

import zmq

from src.common.config import load_settings
from src.telemetry.mock_source import MockTelemetrySource
from src.telemetry.preprocessing import TelemetryPreprocessor
from src.telemetry.zmq_source import ZmqTelemetrySource


def test_mock_source_through_full_pipeline_using_real_config():
    settings = load_settings()
    source = MockTelemetrySource(settings.telemetry.mock, max_records=200, realtime=False)
    preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)

    result = preprocessor.process_batch(source.records())

    assert result.total_received == 200
    # Some imperfection is injected by config (missing_field_rate/out_of_range_rate > 0);
    # the pipeline must still turn the overwhelming majority into clean records, not discard them.
    assert len(result.clean_records) > 150
    assert all(r.source == "MOCK" for r in result.clean_records)
    assert all(r.source != "NS3_5G_LENA" for r in result.clean_records)

    windows = preprocessor.build_feature_windows(result.clean_records)
    assert len(windows) > 0
    for window in windows:
        frame = window.frame
        assert {"timestamp", "ue_id", "cell_id"}.issubset(frame.columns)
        assert frame["ue_id"].nunique() == 1
        assert frame["cell_id"].nunique() == 1
        assert frame["timestamp"].is_monotonic_increasing
        assert 0.0 <= window.missing_ratio <= 1.0
        assert window.quality_flag in ("ok", "low_quality")
        # canonical units in plausible ranges (weak sanity check, not a range-flag re-test)
        assert (frame["throughput_mbps"] >= 0).all()
        assert (frame["packet_loss_pct"] >= 0).all() or window.quality_flag == "low_quality"


def test_zmq_source_through_full_pipeline_real_transport():
    """Publishes real telemetry over an actual ZeroMQ socket (standing in for the future
    NS-3 exporter) and drives it through the exact same preprocessor + windowing used for mock
    data — proving the ZeroMQ real-source path produces the same validated output shape."""
    settings = load_settings()
    zmq_config = settings.telemetry.zmq

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    port = pub.bind_to_random_port("tcp://127.0.0.1")
    endpoint = f"tcp://127.0.0.1:{port}"
    live_config = zmq_config.model_copy(update={"endpoint": endpoint, "recv_timeout_ms": 300})

    source = ZmqTelemetrySource(live_config)
    time.sleep(0.3)  # slow-joiner

    num_messages = 30
    mock_generator = MockTelemetrySource(settings.telemetry.mock, max_records=num_messages, realtime=False)
    raw_messages = list(mock_generator.records())

    def publish():
        for msg in raw_messages:
            msg["source"] = "NS3_5G_LENA"  # what a real exporter would legitimately claim
            pub.send_multipart([zmq_config.topic.encode(), json.dumps(msg).encode("utf-8")])
            time.sleep(0.005)

    publisher_thread = threading.Thread(target=publish, daemon=True)
    publisher_thread.start()

    received = []
    gen = source.records()
    for _ in range(num_messages):
        received.append(next(gen))
    source.close()
    publisher_thread.join(timeout=5.0)
    pub.close()
    ctx.term()

    assert len(received) == num_messages
    assert all(r["source"] == "NS3_5G_LENA" for r in received)

    preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)
    result = preprocessor.process_batch(received)
    assert len(result.clean_records) > 0
    assert all(r.source == "NS3_5G_LENA" for r in result.clean_records)

    windows = preprocessor.build_feature_windows(result.clean_records)
    for window in windows:
        assert {"timestamp", "ue_id", "cell_id"}.issubset(window.frame.columns)
