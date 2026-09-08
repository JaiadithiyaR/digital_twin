"""Integration test: full Module 11 pipeline wired together using real `config/settings.yaml`
values — MockDriftSource -> DriftDetectorInterface -> validated `DriftEvent`s.

Proves the diagram edge end-to-end: external drift notification in ("affected KPI/component,
severity, timestamp"), a `DriftEvent` naming an identified, adaptable DT scope out
(prompt.md §15, fig-dataflow.png Module 11 -> Module 13 edge) — via the real config, not a
hand-built test fixture, and via both the batch and streaming entry points.
"""

from __future__ import annotations

from src.common.config import load_settings
from src.drift.drift_detector import DriftDetectorInterface
from src.drift.mock_drift_source import MockDriftSource


def test_mock_drift_source_through_full_interface_using_real_config():
    settings = load_settings()
    source = MockDriftSource.from_settings(settings, max_events=500, realtime=False)
    detector = DriftDetectorInterface.from_settings(settings)

    result = detector.process_batch(source.events())

    assert result.total_received == 500
    # config.drift.mock.invalid_event_rate is > 0 by default -> some genuine quarantine traffic,
    # but the overwhelming majority must still be accepted (never mass-discarded).
    assert len(result.events) > 400
    assert len(result.quarantined) > 0
    assert all(e.source == "MOCK" for e in result.events)
    assert all(e.source != "EXTERNAL" for e in result.events)

    # Every accepted event names a real, adaptable DT scope — this IS Module 11's "identify
    # which DT scope needs adaptation" responsibility, proven against the live config's actual
    # component list rather than a hardcoded test list.
    assert {e.component for e in result.events} <= set(settings.drift.valid_components)
    # And, with 500 events across 5 components, every real scope genuinely gets exercised at
    # least once (not just a lucky subset) — a meaningful assertion, not a tautology.
    assert {e.component for e in result.events} == set(settings.drift.valid_components)

    lo, hi = settings.drift.severity_range
    assert all(lo <= e.severity <= hi for e in result.events)

    # Quarantined events are retained for audit, not silently dropped (prompt.md §8 principle,
    # extended to Module 11).
    assert all(q.reason for q in result.quarantined)
    assert all(isinstance(q.raw, dict) for q in result.quarantined)


def test_process_stream_consumes_mock_source_continuously_without_raising():
    """Exercises the streaming entry point a future continuous consumer (Module 13's PPO
    observation loop, not built yet) would actually use, rather than only the batch API."""
    settings = load_settings()
    source = MockDriftSource.from_settings(settings, max_events=200, realtime=False)
    detector = DriftDetectorInterface.from_settings(settings)

    events = list(detector.process_stream(source))

    assert 0 < len(events) <= 200  # some were legitimately quarantined and skipped, not raised
    assert all(e.component in settings.drift.valid_components for e in events)


def test_zero_invalid_rate_config_override_yields_zero_quarantine():
    """Confirms quarantining is genuinely driven by the injected malformed events, not some
    unrelated source of noise — with the corruption knob off, everything the real config's
    valid_components/severity_range describe should be accepted."""
    settings = load_settings()
    clean_mock_config = settings.drift.mock.model_copy(update={"invalid_event_rate": 0.0})
    source = MockDriftSource(
        config=clean_mock_config,
        valid_components=settings.drift.valid_components,
        severity_range=settings.drift.severity_range,
        max_events=200,
        realtime=False,
    )
    detector = DriftDetectorInterface.from_settings(settings)

    result = detector.process_batch(source.events())

    assert len(result.quarantined) == 0
    assert len(result.events) == 200
