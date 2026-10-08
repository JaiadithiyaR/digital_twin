"""Unit tests for phase13 §2's adaptation lock (`config.adaptation.lock_policy == "coalesce"`):
`ContinuousOrchestrator._enqueue_trigger`/`_pop_next_trigger`/`_handle_trigger`. Fast, no real
telemetry pipeline or model training — the REQUIRED real "ingestion continues during adaptation"
proof is `tests/integration/test_main_orchestrator.py`'s own sampler-thread assertions, extended
this phase to also track `_locked_component`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.common.config import Secrets, load_settings
from src.fidelity.trigger import AdaptationTrigger
from src.main import ContinuousOrchestrator

SETTINGS = load_settings()
NO_KEY_SECRETS = Secrets(google_api_key=None)


def _bare_orchestrator() -> ContinuousOrchestrator:
    return ContinuousOrchestrator(SETTINGS, NO_KEY_SECRETS)


def _trigger(component: str, severity: float) -> AdaptationTrigger:
    return AdaptationTrigger(
        component=component, severity=severity, timestamp=datetime.now(UTC), trigger_type="external_drift"
    )


def test_lock_policy_is_coalesce_and_concurrent_components_disallowed():
    assert SETTINGS.adaptation.lock_policy == "coalesce"
    assert SETTINGS.adaptation.allow_concurrent_components is False


# --- coalescing (a second trigger for an already-pending component merges, keeping max severity) -


def test_second_trigger_for_a_still_pending_component_is_coalesced_keeping_higher_severity():
    orch = _bare_orchestrator()
    orch._enqueue_trigger(_trigger("dl_bler", 0.3))
    orch._enqueue_trigger(_trigger("dl_bler", 0.7))  # higher severity, same component, still pending

    assert orch._pending_order == ["dl_bler"]  # merged into ONE entry, not two
    popped = orch._pop_next_trigger()
    assert popped.severity == 0.7  # kept the higher of the two
    assert orch._pop_next_trigger() is None  # nothing left — it was genuinely one entry


def test_second_trigger_for_a_still_pending_component_with_lower_severity_keeps_the_first():
    orch = _bare_orchestrator()
    orch._enqueue_trigger(_trigger("dl_bler", 0.9))
    orch._enqueue_trigger(_trigger("dl_bler", 0.2))  # lower severity — must not downgrade

    popped = orch._pop_next_trigger()
    assert popped.severity == 0.9


def test_trigger_arriving_while_component_is_locked_is_coalesced_into_a_fresh_next_cycle_entry():
    """Phase13 §2: "duplicates are merged and processed in the next cycle after the lock is
    released." Once a component's trigger is POPPED (about to be dispatched, `_locked_component`
    set by `_handle_trigger`), a new trigger for that same component must create a genuinely
    FRESH pending entry — never silently merge into the one already being processed (which is no
    longer in `_pending_by_component` at all) — so it's picked up next cycle, not lost."""
    orch = _bare_orchestrator()
    orch._enqueue_trigger(_trigger("dl_bler", 0.4))
    popped = orch._pop_next_trigger()  # simulates run()'s dispatch — component no longer pending
    assert popped.severity == 0.4
    assert orch._pending_order == []

    orch._locked_component = "dl_bler"  # simulate _handle_trigger's own lock, mid-cycle
    orch._enqueue_trigger(_trigger("dl_bler", 0.6))  # arrives WHILE locked — must still coalesce in
    orch._locked_component = None  # cycle finishes

    assert orch._pending_order == ["dl_bler"]
    next_up = orch._pop_next_trigger()
    assert next_up.severity == 0.6  # the one that arrived during the cycle, ready for "next cycle"


def test_different_components_are_never_merged_and_preserve_fifo_order():
    orch = _bare_orchestrator()
    orch._enqueue_trigger(_trigger("dl_bler", 0.5))
    orch._enqueue_trigger(_trigger("ul_prb", 0.5))
    orch._enqueue_trigger(_trigger("dl_throughput", 0.5))

    order = [orch._pop_next_trigger().component for _ in range(3)]
    assert order == ["dl_bler", "ul_prb", "dl_throughput"]
    assert orch._pop_next_trigger() is None


def test_pop_next_trigger_on_empty_pending_returns_none():
    orch = _bare_orchestrator()
    assert orch._pop_next_trigger() is None


# --- the lock itself: acquired for the cycle's duration, ALWAYS released afterward --------------


def test_handle_trigger_locks_during_and_releases_after_a_successful_cycle(monkeypatch):
    orch = _bare_orchestrator()
    observed_lock_during_cycle = []

    def fake_cycle(trigger):
        observed_lock_during_cycle.append(orch._locked_component)

    monkeypatch.setattr(orch, "_run_adaptation_cycle", fake_cycle)
    assert orch._locked_component is None

    orch._handle_trigger(_trigger("dl_bler", 0.5))

    assert observed_lock_during_cycle == ["dl_bler"]  # locked WHILE the cycle ran
    assert orch._locked_component is None  # released afterward — never left stuck


def test_handle_trigger_releases_the_lock_even_when_the_cycle_raises_no_deadlock():
    """Phase13 §2: "no deadlock when the sandbox fails." A `_run_adaptation_cycle` exception
    (standing in for a real sandbox/agent failure) must still release the lock in `finally`."""
    orch = _bare_orchestrator()

    def failing_cycle(trigger):
        raise RuntimeError("simulated sandbox failure")

    orch._run_adaptation_cycle = failing_cycle  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="simulated sandbox failure"):
        orch._handle_trigger(_trigger("dl_bler", 0.5))

    assert orch._locked_component is None, "lock was left stuck after the cycle raised — deadlock risk"


def test_handle_trigger_sets_adaptation_timestamps_even_on_failure():
    orch = _bare_orchestrator()
    assert orch.last_adaptation_started_at is None
    assert orch.last_adaptation_finished_at is None

    def failing_cycle(trigger):
        raise ValueError("boom")

    orch._run_adaptation_cycle = failing_cycle  # type: ignore[method-assign]
    with pytest.raises(ValueError):
        orch._handle_trigger(_trigger("ul_bler", 0.1))

    assert orch.last_adaptation_started_at is not None
    assert orch.last_adaptation_finished_at is not None
    assert orch.last_adaptation_finished_at >= orch.last_adaptation_started_at
