"""Drift Detection Interface (Module 11, prompt.md §15, §0.19, §0.20 rule 7).

The actual drift-detection algorithm is an external/partner module and is explicitly NOT
implemented here (prompt.md §0.19: "Do not implement the actual drift-detection algorithm").
What lives in this file is the receiving side of that integration contract: validate and
normalize whatever a `DriftSource` (mock now; a real external adapter later, unchanged by this
class) hands it, and identify which DT scope (`component`) the notification concerns.

`DriftDetectorInterface.process_event`'s component-name check IS the "identify which DT scope
needs adaptation" responsibility named on fig-dataflow.png for Module 11: an event naming a
component this system doesn't actually run (typo, stale detector config, unrelated KPI) cannot be
routed anywhere real, so it is quarantined rather than silently passed to whatever consumes
`DriftEvent`s next (Module 13's future PPO observation construction — not built yet).

Deliberately does NOT carry-forward-impute like Module 2's `TelemetryPreprocessor` does for
missing telemetry fields: a drift notification is a discrete, one-off signal with no "last known
good value" to substitute, and unlike a telemetry field it feeds directly into a system that
takes real, side-effecting adaptation actions — prompt.md §61's "fail safely" means a malformed
trigger must be dropped, never guessed at.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import ValidationError

from src.drift.base import DriftSource
from src.drift.schema import (
    ALLOWED_DRIFT_SOURCES,
    DriftEvent,
    DriftProcessingResult,
    QuarantinedDriftEvent,
)

if TYPE_CHECKING:
    from src.common.config import Settings

logger = logging.getLogger(__name__)


def _parse_timestamp(value: Any) -> datetime | None:
    """Synchronize heterogeneous timestamp representations (float epoch seconds or ISO-8601) to
    a canonical UTC datetime. Intentionally a self-contained copy of
    `src/telemetry/preprocessing.py`'s identical helper rather than a shared import — Module 11
    and Module 2 are independent parallel inputs on the data-flow diagram and must not develop a
    coupling neither prompt.md nor the architecture calls for."""
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=UTC)
        if isinstance(value, str):
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize(UTC)
            else:
                ts = ts.tz_convert(UTC)
            return ts.to_pydatetime()
    except (ValueError, TypeError, OverflowError):
        return None
    return None


class DriftDetectorInterface:
    """Receives raw drift notifications and turns them into canonical `DriftEvent`s.

    Config-driven, never hardcoded: `valid_components` is the exact set of DT scopes this
    deployment can adapt (must match the `COMPONENT_NAME`s registered in Module 5's
    `DTModelRegistry` — currently throughput/latency/packet_loss/prb_utilization/jitter, sourced
    from `config.drift.valid_components` so this file never hardcodes the component list itself);
    `severity_range` is the inclusive bound a valid event's severity must fall within
    (`config.drift.severity_range`).
    """

    def __init__(self, valid_components: list[str], severity_range: tuple[float, float]) -> None:
        self._valid_components = set(valid_components)
        self._severity_range = severity_range

    @classmethod
    def from_settings(cls, settings: "Settings") -> "DriftDetectorInterface":
        return cls(
            valid_components=settings.drift.valid_components,
            severity_range=settings.drift.severity_range,
        )

    def _validate_shape(self, raw: dict[str, Any]) -> str | None:
        if not isinstance(raw, dict):
            return "not_an_event"
        source = raw.get("source")
        if source not in ALLOWED_DRIFT_SOURCES:
            return f"invalid_source:{source!r}"
        return None

    def _resolve_component(self, raw: dict[str, Any]) -> tuple[str | None, str | None]:
        """Identify which DT scope this event concerns. Returns (component, error_reason) —
        exactly one is non-None."""
        component = raw.get("component")
        if not isinstance(component, str) or not component:
            return None, "missing_component"
        if component not in self._valid_components:
            return None, f"unknown_component:{component!r}"
        return component, None

    def _resolve_severity(self, raw: dict[str, Any]) -> tuple[float | None, str | None]:
        severity = raw.get("severity")
        if severity is None:
            return None, "missing_severity"
        try:
            value = float(severity)
        except (TypeError, ValueError):
            return None, f"non_numeric_severity:{severity!r}"
        if not math.isfinite(value):
            return None, f"non_finite_severity:{severity!r}"
        lo, hi = self._severity_range
        if not (lo <= value <= hi):
            return None, f"severity_out_of_range:{value}"
        return value, None

    def _resolve_metadata(self, raw: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        metadata = raw.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            return None, f"invalid_metadata_type:{type(metadata).__name__}"
        return metadata, None

    def _quarantine(
        self, raw: Any, reason: str, source: str | None, received_at: datetime
    ) -> QuarantinedDriftEvent:
        logger.warning(
            "drift event quarantined", extra={"component": "drift", "reason": reason, "source": source}
        )
        return QuarantinedDriftEvent(
            raw=raw if isinstance(raw, dict) else {"_unparseable": str(raw)},
            reason=reason,
            source=source,
            received_at=received_at,
        )

    def process_event(
        self, raw: dict[str, Any]
    ) -> tuple[DriftEvent | None, QuarantinedDriftEvent | None]:
        received_at = datetime.now(UTC)
        source = raw.get("source") if isinstance(raw, dict) else None

        shape_reason = self._validate_shape(raw)
        if shape_reason is not None:
            return None, self._quarantine(raw, shape_reason, source, received_at)

        component, component_reason = self._resolve_component(raw)
        if component_reason is not None:
            return None, self._quarantine(raw, component_reason, source, received_at)

        severity, severity_reason = self._resolve_severity(raw)
        if severity_reason is not None:
            return None, self._quarantine(raw, severity_reason, source, received_at)

        timestamp = _parse_timestamp(raw.get("timestamp"))
        if timestamp is None:
            return None, self._quarantine(raw, "unparseable_timestamp", source, received_at)

        metadata, metadata_reason = self._resolve_metadata(raw)
        if metadata_reason is not None:
            return None, self._quarantine(raw, metadata_reason, source, received_at)

        try:
            event = DriftEvent(
                component=component,
                severity=severity,
                timestamp=timestamp,
                metadata=metadata,
                source=source,
                received_at=received_at,
            )
        except ValidationError as exc:
            # Defense in depth: anything that made it past the checks above but is still
            # rejected by the canonical schema must quarantine, never crash the caller
            # (prompt.md §44 — external input is untrusted; §61 — fail safely).
            return None, self._quarantine(raw, f"schema_validation_error:{exc}", source, received_at)

        logger.info(
            "drift event accepted: DT scope requiring adaptation identified",
            extra={
                "component": "drift",
                "affected_component": event.component,
                "severity": event.severity,
                "source": source,
            },
        )
        return event, None

    def process_batch(self, raw_events: Iterable[dict[str, Any]]) -> DriftProcessingResult:
        result = DriftProcessingResult()
        for raw in raw_events:
            event, quarantined = self.process_event(raw)
            if event is not None:
                result.events.append(event)
            if quarantined is not None:
                result.quarantined.append(quarantined)
        logger.info(
            "drift batch processed",
            extra={
                "component": "drift",
                "total_received": result.total_received,
                "accepted": len(result.events),
                "quarantined": len(result.quarantined),
            },
        )
        return result

    def process_stream(self, source: DriftSource) -> Iterator[DriftEvent]:
        """Continuously receive from `source`, yielding only validated events with an identified
        DT scope. A quarantined event is logged and skipped, never raised — one malformed
        notification must never break the stream a future consumer (Module 13's PPO observation
        loop, not built yet) reads from."""
        for raw in source.events():
            event, _ = self.process_event(raw)
            if event is not None:
                yield event
