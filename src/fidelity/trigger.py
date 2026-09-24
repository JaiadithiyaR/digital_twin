"""Canonical adaptation-trigger shape (prompt.md §16a) — the common contract both adaptation
trigger sources (Module 11's external `DriftEvent` and Module 12's own internal fidelity-based
trigger) are normalized into, so Module 13 (the Decision & Root-Cause Analysis Agent) can consume
either source uniformly without caring which one fired.

Deliberately co-located with the fidelity module (not `src/drift/`) per this project's own
design-pivot instructions — the fidelity-based trigger is entirely derived from Module 12's own
deterministic formula, and `trigger_from_drift_event()` is a one-way adapter FROM the existing,
unmodified `DriftEvent` (Module 11 itself is untouched — its own schema/validation/mock source
still work exactly as before; this only adds a normalization step downstream of it).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from src.drift.schema import DriftEvent

TriggerType = Literal["external_drift", "fidelity_degradation"]


class AdaptationTrigger(BaseModel):
    """One normalized adaptation trigger, regardless of source. `component` names the DT scope
    needing adaptation; `severity` is always in [0, 1]; `trigger_type` distinguishes provenance
    (an audit/explanatory field only — Module 13 must not branch its decision logic on it, per
    prompt.md §16a: "so that Module 13 can consume either source uniformly")."""

    model_config = ConfigDict(frozen=True)

    component: str
    severity: float
    timestamp: datetime
    trigger_type: TriggerType
    metadata: dict[str, Any] = {}


def trigger_from_drift_event(event: "DriftEvent") -> AdaptationTrigger:
    """Normalizes an already-validated external `DriftEvent` (Module 11) into the canonical
    trigger shape. Never used for the fidelity-based trigger — `FidelityEvaluator.
    check_fidelity_trigger()` constructs `AdaptationTrigger(trigger_type="fidelity_degradation")`
    directly, since there is no separate raw/wire-format event to normalize from."""
    return AdaptationTrigger(
        component=event.component,
        severity=event.severity,
        timestamp=event.timestamp,
        trigger_type="external_drift",
        metadata={**event.metadata, "drift_source": event.source, "received_at": event.received_at.isoformat()},
    )
