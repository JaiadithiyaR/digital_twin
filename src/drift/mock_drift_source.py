"""Mock drift source — development/integration testing only (prompt.md §0.19, §15).

Generates raw drift-notification dicts matching the exact external event contract prompt.md §15
specifies (`component`, `severity`, `timestamp`, `metadata`), so `drift_detector.py`'s
validation/normalization logic is exercised against the same shape a real external/partner
detector would eventually send. Every event is unconditionally stamped `source="MOCK"` — this
class must NEVER be wired to anything that reports itself as `EXTERNAL`, and this mock generator
must never be claimed to represent a real drift-detection algorithm (prompt.md §0.19 — the actual
detection logic is out of scope for this project).

Deliberately injects a small, configurable rate of malformed events (unknown component, severity
outside the valid range, missing timestamp, wrong metadata type) so the interface's
validation/quarantine path is genuinely tested end to end — mirroring
`src/telemetry/mock_source.py`'s missing-field/out-of-range injection for the same reason.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from src.drift.base import DriftSource

if TYPE_CHECKING:
    from src.common.config import MockDriftConfig, Settings


class MockDriftSource(DriftSource):
    SOURCE_LABEL = "MOCK"

    def __init__(
        self,
        config: "MockDriftConfig",
        valid_components: list[str],
        severity_range: tuple[float, float],
        max_events: int | None = None,
        realtime: bool = True,
    ) -> None:
        """
        Args:
            config: drift.mock.* section of settings.yaml.
            valid_components: the DT scope names a real event may legitimately name
                (config.drift.valid_components — shared with `drift_detector.py` so both sides
                agree on what a valid scope is without duplicating the list).
            severity_range: config.drift.severity_range — inclusive bounds for a valid event.
            max_events: stop after this many events; None = stream forever.
            realtime: sleep `emit_interval_seconds` between events (True for live demo mode);
                set False for fast, sleep-free generation (tests).
        """
        self._config = config
        self._valid_components = list(valid_components)
        self._severity_range = severity_range
        self._max_events = max_events
        self._realtime = realtime
        self._rng = np.random.default_rng(config.seed)
        self._closed = False

    @classmethod
    def from_settings(
        cls, settings: "Settings", max_events: int | None = None, realtime: bool = True
    ) -> "MockDriftSource":
        return cls(
            config=settings.drift.mock,
            valid_components=settings.drift.valid_components,
            severity_range=settings.drift.severity_range,
            max_events=max_events,
            realtime=realtime,
        )

    def _generate_event(self) -> dict[str, Any]:
        rng = self._rng
        lo, hi = self._severity_range
        component = str(rng.choice(self._valid_components))
        severity = float(rng.uniform(lo, hi))
        event: dict[str, Any] = {
            "source": self.SOURCE_LABEL,
            "component": component,
            "severity": severity,
            "timestamp": datetime.now(UTC).timestamp(),
            # Clearly-labeled as synthetic — never presented as a real detector's diagnosis
            # (prompt.md §0.19: "never claim that the mock drift source represents a real drift
            # detector").
            "metadata": {"generator": "mock_drift_source", "seed": int(self._config.seed)},
        }
        if rng.random() < self._config.invalid_event_rate:
            event = self._corrupt(event, rng)
        return event

    def _corrupt(self, event: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        """Mutate an otherwise-valid event into one of several realistic malformed shapes a real
        detector's feed could plausibly send, so `drift_detector.py`'s quarantine path is
        genuinely tested rather than only ever seeing well-formed input."""
        kind = rng.choice(["unknown_component", "severity_out_of_range", "missing_timestamp", "bad_metadata_type"])
        if kind == "unknown_component":
            event["component"] = "unrecognized_component_xyz"
        elif kind == "severity_out_of_range":
            lo, hi = self._severity_range
            event["severity"] = hi + 5.0
        elif kind == "missing_timestamp":
            del event["timestamp"]
        elif kind == "bad_metadata_type":
            event["metadata"] = "not-a-dict"
        return event

    def events(self) -> Iterator[dict[str, Any]]:
        """Yields until `max_events` is reached, or indefinitely (`max_events=None`) until
        `close()` is called — matching `MockTelemetrySource`'s stop semantics."""
        emitted = 0
        while not self._closed and (self._max_events is None or emitted < self._max_events):
            yield self._generate_event()
            emitted += 1
            if self._realtime and not self._closed and (self._max_events is None or emitted < self._max_events):
                time.sleep(self._config.emit_interval_seconds)

    def close(self) -> None:
        self._closed = True
