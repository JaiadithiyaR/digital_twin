"""Unit tests for MockDriftSource (Module 11 — real vs mock separation, prompt.md §0.19)."""

from __future__ import annotations

from src.common.config import MockDriftConfig
from src.drift.mock_drift_source import MockDriftSource

VALID_COMPONENTS = ["throughput", "latency", "packet_loss", "prb_utilization", "jitter"]
SEVERITY_RANGE = (0.0, 1.0)


def _config(**overrides) -> MockDriftConfig:
    base = dict(seed=7, emit_interval_seconds=0.01, invalid_event_rate=0.0)
    base.update(overrides)
    return MockDriftConfig(**base)


def test_events_are_stamped_mock_never_external():
    source = MockDriftSource(_config(), VALID_COMPONENTS, SEVERITY_RANGE, max_events=20, realtime=False)
    events = list(source.events())
    assert len(events) == 20
    assert all(e["source"] == "MOCK" for e in events)
    assert all(e["source"] != "EXTERNAL" for e in events)


def test_well_formed_events_match_prompt_md_contract_shape():
    source = MockDriftSource(_config(), VALID_COMPONENTS, SEVERITY_RANGE, max_events=30, realtime=False)
    for event in source.events():
        assert event["component"] in VALID_COMPONENTS
        assert SEVERITY_RANGE[0] <= event["severity"] <= SEVERITY_RANGE[1]
        assert "timestamp" in event
        assert isinstance(event["metadata"], dict)


def test_max_events_bounds_generation():
    source = MockDriftSource(_config(), VALID_COMPONENTS, SEVERITY_RANGE, max_events=5, realtime=False)
    assert len(list(source.events())) == 5


def test_reproducible_with_same_seed():
    a = MockDriftSource(_config(seed=3), VALID_COMPONENTS, SEVERITY_RANGE, max_events=15, realtime=False)
    b = MockDriftSource(_config(seed=3), VALID_COMPONENTS, SEVERITY_RANGE, max_events=15, realtime=False)
    events_a = list(a.events())
    events_b = list(b.events())
    assert [e["component"] for e in events_a] == [e["component"] for e in events_b]
    assert [e["severity"] for e in events_a] == [e["severity"] for e in events_b]


def test_different_seeds_diverge():
    a = MockDriftSource(_config(seed=1), VALID_COMPONENTS, SEVERITY_RANGE, max_events=15, realtime=False)
    b = MockDriftSource(_config(seed=2), VALID_COMPONENTS, SEVERITY_RANGE, max_events=15, realtime=False)
    events_a = [e["severity"] for e in a.events()]
    events_b = [e["severity"] for e in b.events()]
    assert events_a != events_b


def test_invalid_event_rate_actually_produces_malformed_events():
    config = _config(invalid_event_rate=1.0)
    source = MockDriftSource(config, VALID_COMPONENTS, SEVERITY_RANGE, max_events=20, realtime=False)
    events = list(source.events())
    malformed = 0
    for e in events:
        is_malformed = (
            e.get("component") not in VALID_COMPONENTS
            or not (SEVERITY_RANGE[0] <= e.get("severity", -1.0) <= SEVERITY_RANGE[1])
            or "timestamp" not in e
            or not isinstance(e.get("metadata"), dict)
        )
        malformed += int(is_malformed)
    assert malformed == 20  # every event corrupted at rate=1.0


def test_zero_invalid_rate_never_corrupts():
    source = MockDriftSource(_config(invalid_event_rate=0.0), VALID_COMPONENTS, SEVERITY_RANGE, max_events=50, realtime=False)
    events = list(source.events())
    assert all(e["component"] in VALID_COMPONENTS for e in events)
    assert all(SEVERITY_RANGE[0] <= e["severity"] <= SEVERITY_RANGE[1] for e in events)
    assert all("timestamp" in e for e in events)
    assert all(isinstance(e["metadata"], dict) for e in events)


def test_close_stops_generation():
    source = MockDriftSource(_config(), VALID_COMPONENTS, SEVERITY_RANGE, max_events=None, realtime=False)
    it = source.events()
    next(it)
    source.close()
    # Generator loop checks `_closed` at the top of each iteration — draining it must terminate.
    remaining = list(it)
    assert len(remaining) <= 1  # at most one more in flight before the closed-check fires
