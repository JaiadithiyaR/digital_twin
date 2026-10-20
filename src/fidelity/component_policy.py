"""Per-component adaptation policy — the fidelity-based trigger, decided independently for every
DT component from that component's OWN `FidelityScore_c`.

Replaces the former unified-score trigger (one debounce counter on `UnifiedFidelityScore`, then
blaming the argmin component). Nothing here computes fidelity: `FidelityScore_c` comes verbatim
from `FidelityEvaluator.evaluate()` — the unchanged formula, which is per-component by definition
(each component has its own rolling normalization windows). The unified score is still computed
by `FidelityEvaluator.compute_unified_score()`, for reporting only.

The policy semantics are the former trigger's, applied per component:
  - a FRESH score (evaluated this cycle) below the component's threshold advances its counter;
    a score at-or-above the threshold resets it;
  - the trigger fires when the counter reaches the component's min-consecutive count, and the
    counter resets immediately after firing (sustained degradation must re-accumulate);
  - an undefined score (`insufficient_history`) or no fresh measurement this cycle (no scorable
    rows, component not evaluated) never fires and resets the counter — missing data is never
    read as either good or bad fidelity;
  - severity = clip((threshold - FidelityScore_c) / threshold, 0, 1).
Thresholds/min-consecutive counts default to `fidelity.trigger_threshold` /
`trigger_min_consecutive_evaluations` and may be overridden per component
(`fidelity.component_policies`). Components are discovered from whatever is evaluated — no
component list is hardcoded, so expand-scope components are covered automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable, Literal

from src.common.config import FidelityConfig
from src.fidelity.trigger import AdaptationTrigger

logger = logging.getLogger(__name__)

ComponentState = Literal["ok", "below_threshold", "adapt", "insufficient_history", "unavailable"]


@dataclass(frozen=True)
class ComponentPolicy:
    threshold: float
    min_consecutive_evaluations: int


@dataclass(frozen=True)
class ComponentFidelityStatus:
    """One component's policy evaluation for one cycle — what the console, the component
    fidelity log, and the decision context report."""

    component: str
    fidelity_score: float | None
    threshold: float
    consecutive_below: int  # counter AFTER this cycle (0 right after firing)
    min_consecutive_evaluations: int
    state: ComponentState
    severity: float | None  # set only when state == "adapt"
    measured_at: datetime | None
    reason: str

    @property
    def fidelity_gap(self) -> float | None:
        """How far below its threshold the component is (0 when at/above, None when undefined)."""
        if self.fidelity_score is None:
            return None
        return max(0.0, self.threshold - self.fidelity_score)


def severity_for(score: float, threshold: float) -> float:
    return min(max((threshold - score) / threshold, 0.0), 1.0)


class ComponentAdaptationPolicy:
    """Holds one independent debounce counter per component. Construct once per orchestrator
    lifetime (the counters must accumulate across cycles)."""

    def __init__(self, config: FidelityConfig) -> None:
        self._default = ComponentPolicy(config.trigger_threshold, config.trigger_min_consecutive_evaluations)
        self._overrides = dict(config.component_policies)
        self._consecutive_below: dict[str, int] = {}

    def policy_for(self, component: str) -> ComponentPolicy:
        override = self._overrides.get(component)
        if override is None:
            return self._default
        return ComponentPolicy(
            threshold=override.threshold if override.threshold is not None else self._default.threshold,
            min_consecutive_evaluations=(
                override.min_consecutive_evaluations
                if override.min_consecutive_evaluations is not None
                else self._default.min_consecutive_evaluations
            ),
        )

    def consecutive_below(self, component: str) -> int:
        return self._consecutive_below.get(component, 0)

    def evaluate(
        self,
        components: Iterable[str],
        fresh_scores: dict[str, float | None],
        measured_at: datetime | None = None,
    ) -> list[ComponentFidelityStatus]:
        """One policy decision per component in `components`. `fresh_scores` holds ONLY the
        components evaluated this cycle (`None` = evaluated but `insufficient_history`); a
        component absent from it has no fresh measurement this cycle."""
        measured_at = measured_at or datetime.now(UTC)
        statuses: list[ComponentFidelityStatus] = []
        for component in components:
            policy = self.policy_for(component)
            if component not in fresh_scores:
                self._consecutive_below[component] = 0
                statuses.append(self._status(component, None, policy, "unavailable", None, None,
                                             "no fresh fidelity measurement this cycle (no scorable rows)"))
                continue
            score = fresh_scores[component]
            if score is None:
                self._consecutive_below[component] = 0
                statuses.append(self._status(component, None, policy, "insufficient_history", None, measured_at,
                                             "rolling normalization window not yet warm — score withheld, never fabricated"))
                continue
            if score >= policy.threshold:
                self._consecutive_below[component] = 0
                statuses.append(self._status(component, score, policy, "ok", None, measured_at,
                                             f"F_c {score:.4f} >= threshold {policy.threshold}"))
                continue

            count = self._consecutive_below.get(component, 0) + 1
            if count >= policy.min_consecutive_evaluations:
                self._consecutive_below[component] = 0  # reset immediately after firing
                severity = severity_for(score, policy.threshold)
                statuses.append(self._status(
                    component, score, policy, "adapt", severity, measured_at,
                    f"F_c {score:.4f} < threshold {policy.threshold} for {count} consecutive cycle(s) "
                    f"(policy requires {policy.min_consecutive_evaluations})",
                    consecutive_override=count,
                ))
            else:
                self._consecutive_below[component] = count
                statuses.append(self._status(
                    component, score, policy, "below_threshold", None, measured_at,
                    f"F_c {score:.4f} < threshold {policy.threshold} — {count}/{policy.min_consecutive_evaluations} "
                    "consecutive cycle(s), debouncing",
                ))
        return statuses

    def _status(
        self, component: str, score: float | None, policy: ComponentPolicy, state: ComponentState,
        severity: float | None, measured_at: datetime | None, reason: str, consecutive_override: int | None = None,
    ) -> ComponentFidelityStatus:
        return ComponentFidelityStatus(
            component=component,
            fidelity_score=score,
            threshold=policy.threshold,
            consecutive_below=consecutive_override if consecutive_override is not None else self.consecutive_below(component),
            min_consecutive_evaluations=policy.min_consecutive_evaluations,
            state=state,
            severity=severity,
            measured_at=measured_at,
            reason=reason,
        )

    @staticmethod
    def triggers(statuses: Iterable[ComponentFidelityStatus]) -> list[AdaptationTrigger]:
        """One `fidelity_degradation` trigger per component whose own policy fired — each an
        independent adaptation request for exactly that component."""
        triggers: list[AdaptationTrigger] = []
        for status in statuses:
            if status.state != "adapt":
                continue
            # INFO, not WARNING: the console already shows this as an "ADAPT ->" row and a
            # [trigger] line; a WARNING would also dump the raw JSON record onto the console.
            logger.info(
                "per-component fidelity trigger fired",
                extra={
                    "component": status.component, "fidelity_score": status.fidelity_score,
                    "threshold": status.threshold, "severity": status.severity, "reason": status.reason,
                },
            )
            triggers.append(
                AdaptationTrigger(
                    component=status.component,
                    severity=status.severity,  # type: ignore[arg-type] - always set when state == "adapt"
                    timestamp=status.measured_at or datetime.now(UTC),
                    trigger_type="fidelity_degradation",
                    metadata={
                        "policy": "per_component",
                        "fidelity_score": status.fidelity_score,
                        "threshold": status.threshold,
                        "consecutive_below": status.consecutive_below,
                        "min_consecutive_evaluations": status.min_consecutive_evaluations,
                        "reason": status.reason,
                    },
                )
            )
        return triggers
