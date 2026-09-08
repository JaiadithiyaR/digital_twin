"""Unit tests for the Module 3 -> D1 write contract (src/synchronization/d1_interface.py)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.synchronization.d1_interface import D1StateSink, InMemoryD1Stub
from src.telemetry.schema import CleanTelemetryRecord, QuarantinedRecord, RecordQuality


def _clean_record(**overrides) -> CleanTelemetryRecord:
    base = dict(
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        ue_id="ue-0",
        cell_id="cell-0",
        throughput_mbps=50.0,
        offered_load_mbps=60.0,
        latency_ms=20.0,
        jitter_ms=3.0,
        packet_loss_pct=1.0,
        prb_utilization_pct=40.0,
        sinr_db=18.0,
        rsrp_dbm=-90.0,
        rsrq_db=-9.0,
        ue_count=12,
        ue_speed_mps=3.5,
        source="MOCK",
        quality=RecordQuality(),
    )
    base.update(overrides)
    return CleanTelemetryRecord(**base)


def _quarantined(**overrides) -> QuarantinedRecord:
    base = dict(
        raw={"source": "MOCK", "ue_id": "ue-0"},
        reason="test_reason",
        source="MOCK",
        received_at=datetime.now(UTC),
    )
    base.update(overrides)
    return QuarantinedRecord(**base)


# --- ABC enforcement -----------------------------------------------------------------------


def test_d1_state_sink_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        D1StateSink()  # type: ignore[abstract]


def test_stub_satisfies_the_full_contract():
    stub = InMemoryD1Stub()
    assert isinstance(stub, D1StateSink)


# --- current state: latest-by-timestamp semantics -------------------------------------------


def test_update_current_state_accepts_newer_record():
    stub = InMemoryD1Stub()
    older = _clean_record(timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC), throughput_mbps=10.0)
    newer = _clean_record(timestamp=datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC), throughput_mbps=20.0)

    stub.update_current_state([older])
    stub.update_current_state([newer])

    current = stub.get_current_state()
    assert current[("ue-0", "cell-0")].throughput_mbps == 20.0


def test_update_current_state_rejects_out_of_order_older_record():
    """An older record arriving after a newer one (network reordering) must never regress
    current state — prompt.md §0.7: telemetry is authoritative for CURRENT state, meaning the
    latest by event time, not by arrival order."""
    stub = InMemoryD1Stub()
    newer = _clean_record(timestamp=datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC), throughput_mbps=20.0)
    older_arriving_late = _clean_record(
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC), throughput_mbps=10.0
    )

    stub.update_current_state([newer])
    stub.update_current_state([older_arriving_late])

    current = stub.get_current_state()
    assert current[("ue-0", "cell-0")].throughput_mbps == 20.0  # unchanged


def test_current_state_is_keyed_independently_per_ue_and_cell():
    stub = InMemoryD1Stub()
    stub.update_current_state([_clean_record(ue_id="ue-0", cell_id="cell-0")])
    stub.update_current_state([_clean_record(ue_id="ue-1", cell_id="cell-0")])
    current = stub.get_current_state()
    assert set(current.keys()) == {("ue-0", "cell-0"), ("ue-1", "cell-0")}


# --- history: append-only, never truncated ---------------------------------------------------


def test_history_accumulates_across_multiple_calls():
    stub = InMemoryD1Stub()
    stub.append_history([_clean_record(timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC))])
    stub.append_history([_clean_record(timestamp=datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC))])
    assert len(stub.get_history()) == 2


def test_history_keeps_every_record_even_when_current_state_only_keeps_latest():
    stub = InMemoryD1Stub()
    older = _clean_record(timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC), throughput_mbps=10.0)
    newer = _clean_record(timestamp=datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC), throughput_mbps=20.0)
    stub.append_history([older])
    stub.append_history([newer])
    stub.update_current_state([older])
    stub.update_current_state([newer])

    assert len(stub.get_history()) == 2  # both preserved
    assert stub.get_current_state()[("ue-0", "cell-0")].throughput_mbps == 20.0  # only latest


# --- quarantine audit trail -------------------------------------------------------------------


def test_record_quarantine_accumulates():
    stub = InMemoryD1Stub()
    stub.record_quarantine([_quarantined(reason="invalid_source")])
    stub.record_quarantine([_quarantined(reason="missing_required_fields:sinr_db")])
    reasons = [q.reason for q in stub.get_quarantined()]
    assert reasons == ["invalid_source", "missing_required_fields:sinr_db"]


# --- snapshot isolation ------------------------------------------------------------------------


def test_get_current_state_returns_a_copy_not_a_live_reference():
    stub = InMemoryD1Stub()
    stub.update_current_state([_clean_record()])
    snapshot = stub.get_current_state()
    snapshot[("ue-99", "cell-99")] = _clean_record(ue_id="ue-99", cell_id="cell-99")
    assert ("ue-99", "cell-99") not in stub.get_current_state()


def test_get_history_returns_a_copy_not_a_live_reference():
    stub = InMemoryD1Stub()
    stub.append_history([_clean_record()])
    snapshot = stub.get_history()
    snapshot.append(_clean_record(ue_id="ue-99"))
    assert len(stub.get_history()) == 1
