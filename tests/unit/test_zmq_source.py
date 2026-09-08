"""Unit tests for ZmqTelemetrySource using a REAL ZeroMQ PUB/SUB pair (not mocked-out zmq) —
this genuinely exercises the transport, proving the real-source path actually moves bytes over a
socket and decodes them correctly (prompt.md §0.2 steps 9-10, §7)."""

from __future__ import annotations

import json
import threading
import time

import pytest
import zmq

from src.common.config import ZmqTelemetryConfig
from src.telemetry.zmq_source import ZmqTelemetrySource

TOPIC = "ns3.telemetry"


@pytest.fixture
def pub_endpoint():
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    port = pub.bind_to_random_port("tcp://127.0.0.1")
    endpoint = f"tcp://127.0.0.1:{port}"
    yield endpoint, pub, ctx
    pub.close()
    ctx.term()


def _config(endpoint: str, timeout_ms: int = 300) -> ZmqTelemetryConfig:
    return ZmqTelemetryConfig(
        endpoint=endpoint, topic=TOPIC, recv_timeout_ms=timeout_ms, high_water_mark=1000
    )


def _send(pub, payload: dict) -> None:
    pub.send_multipart([TOPIC.encode(), json.dumps(payload).encode("utf-8")])


def test_receives_and_decodes_real_messages(pub_endpoint):
    endpoint, pub, _ctx = pub_endpoint
    source = ZmqTelemetrySource(_config(endpoint))
    time.sleep(0.3)  # zmq "slow joiner" — let the SUB's subscription land before publishing

    sent = [
        {"source": "NS3_5G_LENA", "timestamp": 1000.0 + i, "ue_id": f"ue-{i}", "cell_id": "cell-0"}
        for i in range(3)
    ]
    for msg in sent:
        _send(pub, msg)

    gen = source.records()
    received = [next(gen) for _ in range(3)]
    source.close()

    assert [r["ue_id"] for r in received] == ["ue-0", "ue-1", "ue-2"]
    assert all(r["source"] == "NS3_5G_LENA" for r in received)


def test_source_field_cannot_be_spoofed_through_real_channel(pub_endpoint):
    """Even if a payload claims source=MOCK, the ZeroMQ transport is authoritative: anything
    arriving here IS being reported as real NS-3 telemetry (prompt.md §0.3)."""
    endpoint, pub, _ctx = pub_endpoint
    source = ZmqTelemetrySource(_config(endpoint))
    time.sleep(0.3)

    _send(pub, {"source": "MOCK", "timestamp": 1.0, "ue_id": "ue-0", "cell_id": "cell-0"})

    record = next(source.records())
    source.close()

    assert record["source"] == "NS3_5G_LENA"


def test_malformed_payload_is_skipped_not_fatal(pub_endpoint):
    endpoint, pub, _ctx = pub_endpoint
    source = ZmqTelemetrySource(_config(endpoint))
    time.sleep(0.3)

    pub.send_multipart([TOPIC.encode(), b"{not valid json"])
    _send(pub, {"source": "NS3_5G_LENA", "timestamp": 2.0, "ue_id": "ue-1", "cell_id": "cell-0"})

    record = next(source.records())  # first (malformed) message is silently skipped
    source.close()

    assert record["ue_id"] == "ue-1"


def test_close_stops_records_generator_without_hanging(pub_endpoint):
    endpoint, _pub, _ctx = pub_endpoint
    source = ZmqTelemetrySource(_config(endpoint, timeout_ms=100))

    collected: list[dict] = []

    def consume():
        for record in source.records():
            collected.append(record)

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    time.sleep(0.3)  # let it time out at least once with nothing published
    source.close()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert collected == []


def test_unrelated_topic_is_not_received(pub_endpoint):
    endpoint, pub, _ctx = pub_endpoint
    source = ZmqTelemetrySource(_config(endpoint, timeout_ms=200))
    time.sleep(0.3)

    pub.send_multipart([b"other.topic", json.dumps({"source": "NS3_5G_LENA"}).encode()])
    _send(pub, {"source": "NS3_5G_LENA", "timestamp": 3.0, "ue_id": "ue-2", "cell_id": "cell-1"})

    record = next(source.records())
    source.close()

    assert record["ue_id"] == "ue-2"
