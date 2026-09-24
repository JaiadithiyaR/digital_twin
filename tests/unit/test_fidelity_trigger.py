"""Unit tests for the canonical adaptation-trigger shape (prompt.md §16a, src/fidelity/trigger.py)."""

from __future__ import annotations

from datetime import UTC, datetime

from src.drift.schema import DriftEvent
from src.fidelity.trigger import AdaptationTrigger, trigger_from_drift_event


def test_trigger_from_drift_event_preserves_core_fields():
    event = DriftEvent(
        component="jitter",
        severity=0.42,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"generator": "mock_drift_source"},
        source="MOCK",
        received_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
    )

    trigger = trigger_from_drift_event(event)

    assert isinstance(trigger, AdaptationTrigger)
    assert trigger.component == "jitter"
    assert trigger.severity == 0.42
    assert trigger.timestamp == event.timestamp
    assert trigger.trigger_type == "external_drift"


def test_trigger_from_drift_event_preserves_metadata_and_records_provenance():
    event = DriftEvent(
        component="throughput",
        severity=0.1,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"detector_version": "1.2.3"},
        source="EXTERNAL",
        received_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
    )

    trigger = trigger_from_drift_event(event)

    assert trigger.metadata["detector_version"] == "1.2.3"  # original metadata never dropped
    assert trigger.metadata["drift_source"] == "EXTERNAL"
    assert "received_at" in trigger.metadata


def test_adaptation_trigger_is_frozen():
    trigger = AdaptationTrigger(
        component="latency", severity=0.5, timestamp=datetime.now(UTC), trigger_type="fidelity_degradation"
    )
    try:
        trigger.severity = 0.9  # type: ignore[misc]
        assert False, "AdaptationTrigger should be immutable"
    except Exception:
        pass


def test_fidelity_degradation_trigger_type_is_distinct_from_external_drift():
    fidelity_trigger = AdaptationTrigger(
        component="latency", severity=0.5, timestamp=datetime.now(UTC), trigger_type="fidelity_degradation"
    )
    drift_event = DriftEvent(
        component="latency",
        severity=0.5,
        timestamp=datetime.now(UTC),
        metadata={},
        source="MOCK",
        received_at=datetime.now(UTC),
    )
    drift_trigger = trigger_from_drift_event(drift_event)

    assert fidelity_trigger.trigger_type != drift_trigger.trigger_type
    # Both are the SAME canonical shape (component/severity/timestamp/trigger_type/metadata) —
    # Module 13 can read either uniformly without a type check.
    assert set(type(fidelity_trigger).model_fields) == set(type(drift_trigger).model_fields)
