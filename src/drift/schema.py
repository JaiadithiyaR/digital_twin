"""Drift event schema (Module 11, prompt.md §15, §0.19).

Defines the contract between drift sources (a real external/partner detector — not built here,
per prompt.md §0.19 rule 7 — and `mock_drift_source.py`, used for development/integration testing
only) and `drift_detector.py`'s validation/normalization interface.

Mirrors Module 2's two-stage design deliberately: a source emits raw wire-format dicts
(`DriftSource.events()`), and a separate component (`drift_detector.py`) validates/normalizes
them into the canonical `DriftEvent` type — exactly the `mock_source.py` -> `preprocessing.py`
split `src/telemetry/` already uses, so the same "source is transport-authoritative, payload is
untrusted" discipline applies here without inventing a new pattern.

Expected raw event shape (prompt.md §15):
    {"component": "...", "severity": 0.0, "timestamp": "...", "metadata": {}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# Sources this interface accepts. "EXTERNAL" is the label a real partner detector's adapter would
# stamp once connected (prompt.md §0.19 — "production architecture must allow the external
# detector to be connected later without redesigning the adaptation system"); no such adapter is
# implemented here, only the label it will use. Never conflated with "MOCK" (prompt.md §0.3/§49
# extended to this module — see CLAUDE.md).
ALLOWED_DRIFT_SOURCES = ("EXTERNAL", "MOCK")


class DriftEvent(BaseModel):
    """A single validated, normalized drift notification identifying which DT scope
    (`component`) needs adaptation. This — not the raw payload — is what any future consumer
    (Module 13's PPO observation construction) will read."""

    model_config = ConfigDict(frozen=True)

    component: str  # validated to be a known, adaptable DT scope (see drift_detector.py)
    severity: float  # validated within config.drift.severity_range
    timestamp: datetime  # the event's own reported time, synchronized to UTC
    metadata: dict[str, Any] = {}
    source: Literal["EXTERNAL", "MOCK"]
    received_at: datetime  # when this interface processed it — distinct from `timestamp`


class QuarantinedDriftEvent(BaseModel):
    """A raw event that failed validation and was excluded from the adaptation-relevant stream.

    Quarantining (not silent dropping) mirrors `QuarantinedRecord` (Module 2) — a rejected
    drift notification is still an auditable fact ("the detector sent something invalid at time
    T"), never simply discarded (prompt.md §8's "do not silently discard important data"
    principle, extended here since a real external detector's malformed output is exactly the
    kind of untrusted input this interface must fail safely on, prompt.md §61).
    """

    raw: dict[str, Any]
    reason: str
    source: str | None
    received_at: datetime


@dataclass
class DriftProcessingResult:
    """Output of processing a batch of raw drift events."""

    events: list[DriftEvent] = field(default_factory=list)
    quarantined: list[QuarantinedDriftEvent] = field(default_factory=list)

    @property
    def total_received(self) -> int:
        return len(self.events) + len(self.quarantined)
