"""Unit tests for the Module 3 -> D1 write contract (src/synchronization/d1_interface.py)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.synchronization.d1_interface import D1StateSink, InMemoryD1Stub
from src.telemetry.schema import CleanTelemetryRecord, QuarantinedRecord, RecordQuality


def _clean_record(**overrides) -> CleanTelemetryRecord:
    base = dict(
        run_id=1, cell_id=1, session_id=1, ue_id=1, slice_id=1,
        ts_1s=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        features={}, targets={"dl_throughput": 50.0}, source="MOCK", quality=RecordQuality(),
    )
    base.update(overrides)
    return CleanTelemetryRecord(**base)


def _quarantined(**overrides) -> QuarantinedRecord:
    base = dict(
        raw={"source": "MOCK", "session_id": 1},
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
    older = _clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC), targets={"dl_throughput": 10.0})
    newer = _clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC), targets={"dl_throughput": 20.0})

    stub.update_current_state([older])
    stub.update_current_state([newer])

    current = stub.get_current_state()
    assert current[(1, 1)].targets["dl_throughput"] == 20.0


def test_update_current_state_rejects_out_of_order_older_record():
    """An older record arriving after a newer one (network reordering) must never regress
    current state — prompt.md §0.7: telemetry is authoritative for CURRENT state, meaning the
    latest by event time, not by arrival order."""
    stub = InMemoryD1Stub()
    newer = _clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC), targets={"dl_throughput": 20.0})
    older_arriving_late = _clean_record(
        ts_1s=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC), targets={"dl_throughput": 10.0}
    )

    stub.update_current_state([newer])
    stub.update_current_state([older_arriving_late])

    current = stub.get_current_state()
    assert current[(1, 1)].targets["dl_throughput"] == 20.0  # unchanged


def test_current_state_is_keyed_independently_per_ue_and_cell():
    stub = InMemoryD1Stub()
    stub.update_current_state([_clean_record(ue_id=1, cell_id=1)])
    stub.update_current_state([_clean_record(ue_id=2, cell_id=1)])
    current = stub.get_current_state()
    assert set(current.keys()) == {(1, 1), (2, 1)}


# --- history: append-only, never truncated ---------------------------------------------------


def test_history_accumulates_across_multiple_calls():
    stub = InMemoryD1Stub()
    stub.append_history([_clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC))])
    stub.append_history([_clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC))])
    assert len(stub.get_history()) == 2


def test_history_keeps_every_record_even_when_current_state_only_keeps_latest():
    stub = InMemoryD1Stub()
    older = _clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC), targets={"dl_throughput": 10.0})
    newer = _clean_record(ts_1s=datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC), targets={"dl_throughput": 20.0})
    stub.append_history([older])
    stub.append_history([newer])
    stub.update_current_state([older])
    stub.update_current_state([newer])

    assert len(stub.get_history()) == 2  # both preserved
    assert stub.get_current_state()[(1, 1)].targets["dl_throughput"] == 20.0  # only latest


# --- quarantine audit trail -------------------------------------------------------------------


def test_record_quarantine_accumulates():
    stub = InMemoryD1Stub()
    stub.record_quarantine([_quarantined(reason="invalid_source")])
    stub.record_quarantine([_quarantined(reason="missing_identity_field:slice_id")])
    reasons = [q.reason for q in stub.get_quarantined()]
    assert reasons == ["invalid_source", "missing_identity_field:slice_id"]


# --- snapshot isolation ------------------------------------------------------------------------


def test_get_current_state_returns_a_copy_not_a_live_reference():
    stub = InMemoryD1Stub()
    stub.update_current_state([_clean_record()])
    snapshot = stub.get_current_state()
    snapshot[(99, 99)] = _clean_record(ue_id=99, cell_id=99)
    assert (99, 99) not in stub.get_current_state()


def test_get_history_returns_a_copy_not_a_live_reference():
    stub = InMemoryD1Stub()
    stub.append_history([_clean_record()])
    snapshot = stub.get_history()
    snapshot.append(_clean_record(ue_id=99))
    assert len(stub.get_history()) == 1
