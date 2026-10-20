"""Unit tests for phase13 §2's adaptation lock (`config.adaptation.lock_policy == "coalesce"`):
`ContinuousOrchestrator._enqueue_trigger`/`_pop_next_trigger`/`_handle_trigger`. Fast, no real
telemetry pipeline or model training — the REQUIRED real "ingestion continues during adaptation"
proof is `tests/integration/test_main_orchestrator.py`'s own sampler-thread assertions, extended
which also tracks `active_adaptations`. Also covers the retry-until-fixed adaptation loop and the
per-component worker threads.
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


def test_lock_policy_is_coalesce_and_components_adapt_concurrently_on_workers():
    assert SETTINGS.adaptation.lock_policy == "coalesce"
    assert SETTINGS.adaptation.allow_concurrent_components is True
    assert _bare_orchestrator()._max_workers == SETTINGS.adaptation.max_concurrent_adaptations >= 2


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
    released." Once a component's trigger is POPPED (its component then marked active), a new
    trigger for that same component must create a genuinely FRESH pending entry — never silently
    merge into the one already being processed — and must NOT be dispatched while the component
    is still adapting; it is picked up once the adaptation ends, not lost."""
    orch = _bare_orchestrator()
    orch._enqueue_trigger(_trigger("dl_bler", 0.4))
    popped = orch._pop_next_trigger()  # simulates run()'s dispatch — component no longer pending
    assert popped.severity == 0.4
    assert orch._pending_order == []

    orch._active_components.add("dl_bler")  # mid-adaptation
    orch._enqueue_trigger(_trigger("dl_bler", 0.6))  # arrives WHILE adapting — must still coalesce in
    assert orch._pending_order == ["dl_bler"]
    assert orch._pop_next_trigger() is None  # never a second concurrent adaptation of dl_bler
    orch._active_components.discard("dl_bler")  # adaptation finishes

    next_up = orch._pop_next_trigger()
    assert next_up.severity == 0.6  # the one that arrived during the adaptation, ready now


def test_pop_skips_a_busy_component_but_still_dispatches_others():
    orch = _bare_orchestrator()
    orch._enqueue_trigger(_trigger("dl_bler", 0.5))
    orch._enqueue_trigger(_trigger("ul_prb", 0.5))
    orch._active_components.add("dl_bler")
    assert orch._pop_next_trigger().component == "ul_prb"
    assert orch._pending_order == ["dl_bler"]


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


# --- the lock itself: acquired for the whole adaptation loop, ALWAYS released afterward ---------


def _result(outcome: str, *, retryable: bool = False, still_below: bool | None = True, summary: str = "s") -> dict:
    return {"outcome": outcome, "retryable": retryable, "still_below_threshold": still_below,
            "post_adaptation_fidelity": None, "summary": summary}


def test_handle_trigger_locks_during_and_releases_after_a_successful_cycle(monkeypatch):
    orch = _bare_orchestrator()
    observed_active_during_cycle = []

    def fake_cycle(trigger, **kwargs):
        observed_active_during_cycle.append(orch.active_adaptations)
        return _result("accepted")

    monkeypatch.setattr(orch, "_run_adaptation_cycle", fake_cycle)
    assert orch.active_adaptations == []

    orch._handle_trigger(_trigger("dl_bler", 0.5))

    assert observed_active_during_cycle == [["dl_bler"]]  # locked WHILE the cycle ran
    assert orch.active_adaptations == []  # released afterward — never left stuck


def test_handle_trigger_releases_the_lock_even_when_the_cycle_raises_no_deadlock():
    """Phase13 §2: "no deadlock when the sandbox fails." A `_run_adaptation_cycle` exception
    (standing in for a real sandbox/agent failure) must still release the lock in `finally`."""
    orch = _bare_orchestrator()

    def failing_cycle(trigger, **kwargs):
        raise RuntimeError("simulated sandbox failure")

    orch._run_adaptation_cycle = failing_cycle  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="simulated sandbox failure"):
        orch._handle_trigger(_trigger("dl_bler", 0.5))

    assert orch.active_adaptations == [], "lock was left stuck after the cycle raised — deadlock risk"


def test_handle_trigger_sets_adaptation_timestamps_even_on_failure():
    orch = _bare_orchestrator()
    assert orch.last_adaptation_started_at is None
    assert orch.last_adaptation_finished_at is None

    def failing_cycle(trigger, **kwargs):
        raise ValueError("boom")

    orch._run_adaptation_cycle = failing_cycle  # type: ignore[method-assign]
    with pytest.raises(ValueError):
        orch._handle_trigger(_trigger("ul_bler", 0.1))

    assert orch.last_adaptation_started_at is not None
    assert orch.last_adaptation_finished_at is not None
    assert orch.last_adaptation_finished_at >= orch.last_adaptation_started_at


# --- the adaptation loop: retry with feedback until accepted / recovered / attempts exhausted -----


def test_rejected_attempts_are_retried_with_feedback_until_accepted(monkeypatch):
    orch = _bare_orchestrator()
    calls: list[tuple[int, tuple[str, ...]]] = []
    outcomes = iter([_result("rejected", retryable=True, summary="a1 rejected"),
                     _result("agent_failed", retryable=True, still_below=None, summary="a2 sandbox rejected"),
                     _result("accepted")])

    def cycle(trigger, *, attempt, max_attempts, previous_attempts):
        calls.append((attempt, previous_attempts))
        return next(outcomes)

    monkeypatch.setattr(orch, "_run_adaptation_cycle", cycle)
    orch._handle_trigger(_trigger("dl_bler", 0.5))

    assert [a for a, _ in calls] == [1, 2, 3]
    assert calls[1][1] == ("a1 rejected",)  # each attempt sees every earlier rejection
    assert calls[2][1] == ("a1 rejected", "a2 sandbox rejected")


def test_loop_stops_after_max_attempts():
    orch = _bare_orchestrator()
    attempts = []
    orch._run_adaptation_cycle = lambda trigger, **kw: attempts.append(kw["attempt"]) or _result("rejected", retryable=True)  # type: ignore[method-assign]
    orch._handle_trigger(_trigger("dl_bler", 0.5))
    assert attempts == list(range(1, SETTINGS.adaptation.max_adaptation_attempts + 1))


def test_loop_stops_once_the_component_recovers_or_the_outcome_is_not_retryable():
    orch = _bare_orchestrator()
    attempts = []
    orch._run_adaptation_cycle = lambda trigger, **kw: attempts.append(kw["attempt"]) or _result("rejected", retryable=True, still_below=False)  # type: ignore[method-assign]
    orch._handle_trigger(_trigger("dl_bler", 0.5))
    assert attempts == [1]  # REJECT, but the component now meets its threshold: nothing left to fix

    attempts.clear()
    orch._run_adaptation_cycle = lambda trigger, **kw: attempts.append(kw["attempt"]) or _result("skipped_no_llm_client")  # type: ignore[method-assign]
    orch._handle_trigger(_trigger("dl_bler", 0.5))
    assert attempts == [1]


def test_stale_trigger_for_a_recovered_component_is_skipped_and_logged(tmp_path):
    from datetime import timedelta

    from src.common.run_logs import AdaptationDecisionLog
    from src.fidelity.component_policy import ComponentFidelityStatus

    class _Registry:
        def get_current_version(self, name):
            return None

    orch = _bare_orchestrator()
    orch.model_registry = _Registry()
    orch.adaptation_decision_log = AdaptationDecisionLog(tmp_path / "decisions.jsonl")
    trigger = AdaptationTrigger(component="dl_bler", severity=0.5, timestamp=datetime.now(UTC) - timedelta(seconds=30),
                                trigger_type="fidelity_degradation", metadata={"fidelity_score": 0.2})
    orch._component_status["dl_bler"] = ComponentFidelityStatus(
        component="dl_bler", fidelity_score=0.8, threshold=0.5, consecutive_below=0, min_consecutive_evaluations=3,
        state="ok", severity=None, measured_at=datetime.now(UTC), reason="ok",
    )
    orch._run_adaptation_cycle = lambda *a, **k: pytest.fail("a recovered component must not be adapted")  # type: ignore[method-assign]

    orch._handle_trigger(trigger)

    import json
    entries = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert [e["outcome"] for e in entries] == ["skipped_recovered"]


# --- worker threads: concurrent components, never the same component twice ----------------------


def test_run_adapts_different_components_concurrently_on_worker_threads(monkeypatch):
    import threading

    orch = _bare_orchestrator()
    both_running = threading.Barrier(2, timeout=10)
    threads: list[str] = []

    def cycle(trigger, **kwargs):
        threads.append(threading.current_thread().name)
        both_running.wait()  # deadlocks (BrokenBarrierError) unless the two run at the same time
        return _result("accepted")

    monkeypatch.setattr(orch, "_run_adaptation_cycle", cycle)
    monkeypatch.setattr(orch, "run_prediction_and_fidelity_cycle", lambda: {})
    orch._enqueue_trigger(_trigger("dl_bler", 0.5))
    orch._enqueue_trigger(_trigger("ul_bler", 0.5))

    orch.run(max_drift_events=2, prediction_interval_seconds=0.0)

    assert sorted(threads) == ["dt-adapt-dl_bler", "dt-adapt-ul_bler"]
    assert orch.active_adaptations == []


# --- the continuous loop survives a failing cycle --------------------------------------------------


def test_run_loop_survives_a_crashing_adaptation_and_prediction_cycle_and_keeps_dispatching(monkeypatch):
    """One unexpected exception (sandbox, fidelity, I/O ...) must never kill the continuous loop
    — the next trigger is still dispatched and the lock is still released."""
    orch = _bare_orchestrator()
    handled: list[str] = []

    def flaky_cycle(trigger, **kwargs):
        handled.append(trigger.component)
        if trigger.component == "dl_bler":
            raise RuntimeError("simulated unexpected failure")
        return _result("accepted")

    def failing_prediction_cycle():
        raise ValueError("simulated fidelity failure")

    monkeypatch.setattr(orch, "_run_adaptation_cycle", flaky_cycle)
    monkeypatch.setattr(orch, "run_prediction_and_fidelity_cycle", failing_prediction_cycle)
    orch._enqueue_trigger(_trigger("dl_bler", 0.5))
    orch._enqueue_trigger(_trigger("ul_bler", 0.5))

    orch.run(max_drift_events=2, prediction_interval_seconds=0.0)

    assert sorted(handled) == ["dl_bler", "ul_bler"]
    assert orch.active_adaptations == []


def test_restarted_replay_resumes_after_the_last_tick_already_persisted_in_d1():
    import pandas as pd

    class _FakeD1:
        def get_history(self):
            return pd.DataFrame({
                "run_id": [8, 8, 3],
                "ts_1s": pd.to_datetime(["2026-01-01T00:00:05Z", "2026-01-01T00:00:09Z", "2026-01-01T09:00:00Z"]),
            })

    class _FakeSource:
        seeked_to = None

        def seek(self, ts):
            self.seeked_to = ts

    orch = _bare_orchestrator()
    orch.d1_store = _FakeD1()
    source = _FakeSource()
    orch._resume_replay_after_persisted_history(source, [8])

    # strictly after run 8's last persisted tick — never re-sending it, never using run 3's rows
    last = pd.Timestamp("2026-01-01T00:00:09Z")
    assert source.seeked_to is not None and last < source.seeked_to < last + pd.Timedelta(seconds=1)


def test_fresh_d1_does_not_seek_the_replay():
    import pandas as pd

    class _EmptyD1:
        def get_history(self):
            return pd.DataFrame()

    class _FakeSource:
        def seek(self, ts):
            raise AssertionError("must not seek when D1 holds nothing for these runs")

    orch = _bare_orchestrator()
    orch.d1_store = _EmptyD1()
    orch._resume_replay_after_persisted_history(_FakeSource(), [8])
