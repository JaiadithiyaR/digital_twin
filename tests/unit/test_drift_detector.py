"""Unit tests for DriftDetectorInterface (Module 11 core — prompt.md §15, §0.19).

Proves: raw drift-notification dicts go in, validated `DriftEvent`s (with an identified DT
scope) or `QuarantinedDriftEvent`s come out — never a crash, never a silently-fabricated event.
"""

from __future__ import annotations

import math

import pytest

from src.drift.drift_detector import DriftDetectorInterface

VALID_COMPONENTS = ["throughput", "latency", "packet_loss", "prb_utilization", "jitter"]
SEVERITY_RANGE = (0.0, 1.0)


def _detector() -> DriftDetectorInterface:
    return DriftDetectorInterface(valid_components=VALID_COMPONENTS, severity_range=SEVERITY_RANGE)


def _raw_event(**overrides) -> dict:
    base = dict(
        source="EXTERNAL",
        component="throughput",
        severity=0.7,
        timestamp=1_700_000_000.0,
        metadata={"detector": "partner-x"},
    )
    base.update(overrides)
    return base


def test_well_formed_event_is_accepted_and_scope_identified():
    event, quarantined = _detector().process_event(_raw_event())
    assert quarantined is None
    assert event is not None
    assert event.component == "throughput"
    assert event.severity == pytest.approx(0.7)
    assert event.source == "EXTERNAL"


def test_mock_source_labeled_event_also_accepted():
    event, quarantined = _detector().process_event(_raw_event(source="MOCK"))
    assert quarantined is None
    assert event.source == "MOCK"


def test_invalid_source_label_is_quarantined():
    event, quarantined = _detector().process_event(_raw_event(source="SOMETHING_ELSE"))
    assert event is None
    assert quarantined is not None
    assert quarantined.reason.startswith("invalid_source")


def test_non_dict_input_is_quarantined_not_crashed():
    event, quarantined = _detector().process_event("not a dict")  # type: ignore[arg-type]
    assert event is None
    assert quarantined.reason == "not_an_event"


def test_unknown_component_cannot_be_routed_to_any_dt_scope():
    event, quarantined = _detector().process_event(_raw_event(component="unknown_kpi"))
    assert event is None
    assert quarantined.reason == "unknown_component:'unknown_kpi'"


def test_missing_component_is_quarantined():
    raw = _raw_event()
    del raw["component"]
    event, quarantined = _detector().process_event(raw)
    assert event is None
    assert quarantined.reason == "missing_component"


def test_non_string_component_is_quarantined():
    event, quarantined = _detector().process_event(_raw_event(component=123))
    assert event is None
    assert quarantined.reason == "missing_component"


@pytest.mark.parametrize("component", VALID_COMPONENTS)
def test_every_configured_dt_scope_is_individually_routable(component):
    event, quarantined = _detector().process_event(_raw_event(component=component))
    assert quarantined is None
    assert event.component == component


def test_severity_out_of_range_is_quarantined():
    event, quarantined = _detector().process_event(_raw_event(severity=5.0))
    assert event is None
    assert quarantined.reason == "severity_out_of_range:5.0"


def test_severity_at_exact_bounds_is_accepted():
    for boundary in SEVERITY_RANGE:
        event, quarantined = _detector().process_event(_raw_event(severity=boundary))
        assert quarantined is None
        assert event.severity == pytest.approx(boundary)


def test_missing_severity_is_quarantined():
    raw = _raw_event()
    del raw["severity"]
    event, quarantined = _detector().process_event(raw)
    assert event is None
    assert quarantined.reason == "missing_severity"


def test_non_numeric_severity_is_quarantined_not_crashed():
    event, quarantined = _detector().process_event(_raw_event(severity="high"))
    assert event is None
    assert quarantined.reason.startswith("non_numeric_severity")


def test_nan_and_inf_severity_are_quarantined_not_silently_accepted():
    for bad in (math.nan, math.inf, -math.inf):
        event, quarantined = _detector().process_event(_raw_event(severity=bad))
        assert event is None
        assert quarantined.reason.startswith("non_finite_severity")


def test_missing_timestamp_is_quarantined():
    raw = _raw_event()
    del raw["timestamp"]
    event, quarantined = _detector().process_event(raw)
    assert event is None
    assert quarantined.reason == "unparseable_timestamp"


def test_unparseable_timestamp_string_is_quarantined_not_crashed():
    event, quarantined = _detector().process_event(_raw_event(timestamp="not-a-date"))
    assert event is None
    assert quarantined.reason == "unparseable_timestamp"


def test_iso8601_timestamp_string_is_accepted():
    event, quarantined = _detector().process_event(_raw_event(timestamp="2026-09-07T05:00:00Z"))
    assert quarantined is None
    assert event.timestamp.year == 2026


def test_missing_metadata_defaults_to_empty_dict():
    raw = _raw_event()
    del raw["metadata"]
    event, quarantined = _detector().process_event(raw)
    assert quarantined is None
    assert event.metadata == {}


def test_non_dict_metadata_is_quarantined_not_crashed():
    event, quarantined = _detector().process_event(_raw_event(metadata="not-a-dict"))
    assert event is None
    assert quarantined.reason.startswith("invalid_metadata_type")


def test_quarantined_event_retains_raw_payload_for_audit():
    raw = _raw_event(component="unknown_kpi")
    _, quarantined = _detector().process_event(raw)
    assert quarantined.raw == raw


def test_process_batch_never_raises_on_a_mixed_batch():
    raw_events = [
        _raw_event(),
        _raw_event(component="unknown"),
        _raw_event(severity=99.0),
        {"garbage": True},
        _raw_event(component="latency"),
    ]
    result = _detector().process_batch(raw_events)
    assert result.total_received == 5
    assert len(result.events) == 2
    assert len(result.quarantined) == 3


def test_process_stream_yields_only_valid_events_and_never_raises():
    class _FlakySource:
        def events(self):
            yield _raw_event(component="throughput")
            yield {"garbage": True}
            yield _raw_event(component="jitter")

    events = list(_detector().process_stream(_FlakySource()))
    assert [e.component for e in events] == ["throughput", "jitter"]
