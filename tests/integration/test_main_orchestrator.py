"""Integration test: `src/main.py`'s `ContinuousOrchestrator` — the full canonical loop (Modules
1-19) wired together and run for real, at a smaller/faster scale than
`scripts/run_orchestrator_demo.py` (the permanent, repo-tracked, full-scale validation script —
see CLAUDE.md's Phase 11 entry for real observed numbers). This test proves the same two things
automatically on every run: (1) one full real internal-fidelity-trigger -> Decision &
Root-Cause Analysis Agent -> agent -> verification -> lifecycle cycle completes without error
(external drift is permanently disabled; if no component's policy fires during warm-up, one
labelled internal trigger is injected), and (2) live D1 telemetry
synchronization never stops during that cycle — the same real-background-thread-plus-sampler-
thread proof Modules 14/15/16/19 already established for individual agents, applied here to the
real orchestrator.

Reads whatever `GOOGLE_API_KEY`/`GEMINI_API_KEY` is actually configured in this environment's real
`.env` via `load_secrets()` — if none is set, the decision call genuinely fails and
`DecisionAgent.decide_safe()` degrades to `config.decision_agent.fallback.default_strategy`
(`recalibrate`, which needs no LLM to execute), logged at WARNING; if a real key IS set, this test
genuinely exercises a live decision call (observed real behavior: `recalibrate` selected both with
and without a real key, so this test's assertions are written to hold either way — never assuming
which).
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime

import pytest

from src.common.config import load_secrets, load_settings
from src.fidelity.trigger import AdaptationTrigger
from src.main import ContinuousOrchestrator

SETTINGS = load_settings()
SECRETS = load_secrets()

# initialize() bootstrap-trains the 7 components on the real OAI TRAIN split — same skip
# convention as every other test that needs the built OAI dataset.
pytestmark = pytest.mark.skipif(
    not SETTINGS.resolve_path("data/processed/oai/train.parquet").exists(),
    reason="data/processed/oai/train.parquet not built — run scripts/build_oai_dataset.py",
)

def _tmp_logging(tmp_path):
    """Keep this test's component fidelity / adaptation decision logs out of the repo's logs/."""
    return SETTINGS.logging.model_copy(update={
        "component_fidelity_log": str(tmp_path / "component_fidelity.csv"),
        "adaptation_decisions_log": str(tmp_path / "adaptation_decisions.jsonl"),
    })


MIN_HISTORY_ROWS = 220  # comfortably above config.adaptation.recalibration.min_training_rows (200)
ACCUMULATION_TIMEOUT_SECONDS = 60.0


def test_full_orchestrator_cycle_never_stops_telemetry_synchronization(tmp_path):
    overrides = {
        "storage": SETTINGS.storage.model_copy(
            update={
                "bootstrap_dir": str(tmp_path / "bootstrap"),
                "d1_current_state_path": str(tmp_path / "d1_current.parquet"),
                "d1_history_path": str(tmp_path / "d1_history.parquet"),
                "d1_quarantine_path": str(tmp_path / "d1_quarantine.parquet"),
                "models_dir": str(tmp_path / "models"),
            }
        ),
        "telemetry": SETTINGS.telemetry.model_copy(
            update={"mock": SETTINGS.telemetry.mock.model_copy(update={"emit_interval_seconds": 0.05, "num_sessions": 10, })}
        ),
        "lifecycle": SETTINGS.lifecycle.model_copy(
            update={"records_path": str(tmp_path / "lifecycle.jsonl"), "reports_dir": str(tmp_path / "reports")}
        ),
        "rag": SETTINGS.rag.model_copy(update={"chroma_path": str(tmp_path / "chroma")}),
        "logging": _tmp_logging(tmp_path),
        # one attempt: these tests check a single real cycle; the retry loop is unit-tested
        "adaptation": SETTINGS.adaptation.model_copy(update={"max_adaptation_attempts": 1}),
    }
    fast_settings = SETTINGS.model_copy(update=overrides)

    orchestrator = ContinuousOrchestrator(fast_settings, SECRETS)
    orchestrator.initialize()

    # Regression test for a real bug found running the live end-to-end demo against a genuine
    # GOOGLE_API_KEY: LifecycleAgent.from_settings() was being called BEFORE self.llm_client even
    # existed, and without passing it at all — so generate_maintenance_report()'s optional
    # LLM-enhanced prose was silently unreachable forever, regardless of whether a working client
    # existed, with no warning ever logged (a call that's never attempted can't log a failure).
    assert orchestrator.lifecycle_agent._llm_client is orchestrator.llm_client  # noqa: SLF001 - read-only wiring check

    orchestrator.start()

    try:
        wait_start = time.monotonic()
        while len(orchestrator.d1_store.get_history()) < MIN_HISTORY_ROWS:
            assert time.monotonic() - wait_start < ACCUMULATION_TIMEOUT_SECONDS, "live D1 history never accumulated enough rows"
            time.sleep(0.1)

        min_history = fast_settings.fidelity.min_history_for_normalization
        for _ in range(min_history + 2):
            orchestrator.run_prediction_and_fidelity_cycle()
            time.sleep(0.1)

        # External drift is permanently disabled: no drift source/consumer exists at all.
        for attribute in ("drift_source", "drift_detector", "drift_thread"):
            assert not hasattr(orchestrator, attribute), f"external drift must never be wired in ({attribute})"
        # Drop any warm-up trigger (it could be stale — recovered — by dispatch time and be skipped
        # without a lifecycle record) and force exactly one deterministic adaptation.
        with orchestrator._pending_lock:  # noqa: SLF001
            orchestrator._pending_order.clear()  # noqa: SLF001
            orchestrator._pending_by_component.clear()  # noqa: SLF001
        orchestrator._enqueue_trigger(AdaptationTrigger(  # noqa: SLF001
            component="dl_bler", severity=0.4, timestamp=datetime.now(UTC), trigger_type="fidelity_degradation",
            metadata={"forced": True, "reason": "integration test"},
        ))

        samples: list[tuple[float, int]] = []
        lock_samples: list[tuple[float, tuple[str, ...]]] = []
        stop_sampling = threading.Event()

        def sample_loop() -> None:
            while not stop_sampling.is_set():
                now = time.monotonic()
                samples.append((now, orchestrator.synchronizer.records_synced))
                lock_samples.append((now, tuple(orchestrator.active_adaptations)))
                time.sleep(0.01)

        sampler = threading.Thread(target=sample_loop, daemon=True)
        sampler.start()

        orchestrator.run(max_drift_events=1, prediction_interval_seconds=2.0)

        stop_sampling.set()
        sampler.join(timeout=5.0)

        start_ts = orchestrator.last_adaptation_started_at
        end_ts = orchestrator.last_adaptation_finished_at
        assert start_ts is not None and end_ts is not None, "no adaptation cycle was recorded"

        during = [n for ts, n in samples if start_ts <= ts <= end_ts]
        assert len(during) >= 2, "adaptation cycle completed too fast for the sampler to observe overlap"
        assert during[-1] > during[0], (
            f"records_synced did not grow during the adaptation cycle ({during[0]} -> {during[-1]}) "
            "— telemetry ingestion was blocked"
        )

        # Phase 13 §2 (adaptation lock): the component was genuinely marked locked for (part of)
        # the real cycle's own window, and is released (None) once the cycle has finished —
        # never left stuck, confirmed with the SAME real timing this test already observes.
        locked_during = [c for ts, active in lock_samples if start_ts <= ts <= end_ts for c in active]
        assert locked_during, "the component was never observed active during the real adaptation cycle"
        assert orchestrator.active_adaptations == [], "adaptation lock was not released after the cycle finished"

        records = orchestrator.lifecycle_agent.list_records()
        assert len(records) == 1
        record = records[0]
        assert set(locked_during) == {record.affected_component}, "locked component didn't match the actual trigger's component"
        assert record.affected_component in fast_settings.drift.valid_components
        assert record.decision_strategy in ("recalibrate", "regenerate", "expand_scope")
        assert record.verification_result in ("ACCEPT", "REJECT")
        assert record.final_status in ("promoted", "rejected")
        assert (record.final_status == "promoted") == (record.verification_result == "ACCEPT")

        report = orchestrator.lifecycle_agent.generate_maintenance_report(record)
        assert report.path.exists()
        assert record.affected_component in report.text

        # Per-component policy outcome recorded on the lifecycle record and in the decision log.
        assert record.component_threshold == fast_settings.fidelity.trigger_threshold
        assert record.production_version_after is not None
        decisions = [json.loads(line) for line in (tmp_path / "adaptation_decisions.jsonl").read_text().splitlines()]
        assert len(decisions) == 1
        assert decisions[0]["component"] == record.affected_component
        assert decisions[0]["outcome"] == ("accepted" if record.verification_result == "ACCEPT" else "rejected")
        assert decisions[0]["lifecycle_event_id"] == record.event_id
        # Every warming cycle logged one row per component.
        fidelity_rows = (tmp_path / "component_fidelity.csv").read_text().splitlines()
        assert len(fidelity_rows) > len(orchestrator._target_columns)  # noqa: SLF001
    finally:
        orchestrator.stop()


def _fast_mock_orchestrator(tmp_path) -> ContinuousOrchestrator:
    """A fast, self-contained orchestrator (mock telemetry, isolated tmp storage) — used by the
    shutdown-determinism test below, which only needs start()/stop() to behave correctly, not a
    full adaptation cycle."""
    overrides = {
        "storage": SETTINGS.storage.model_copy(
            update={
                "bootstrap_dir": str(tmp_path / "bootstrap"),
                "d1_current_state_path": str(tmp_path / "d1_current.parquet"),
                "d1_history_path": str(tmp_path / "d1_history.parquet"),
                "d1_quarantine_path": str(tmp_path / "d1_quarantine.parquet"),
                "models_dir": str(tmp_path / "models"),
            }
        ),
        "telemetry": SETTINGS.telemetry.model_copy(
            update={"mock": SETTINGS.telemetry.mock.model_copy(update={"emit_interval_seconds": 0.01, "num_sessions": 4})}
        ),
        "lifecycle": SETTINGS.lifecycle.model_copy(
            update={"records_path": str(tmp_path / "lifecycle.jsonl"), "reports_dir": str(tmp_path / "reports")}
        ),
        "rag": SETTINGS.rag.model_copy(update={"chroma_path": str(tmp_path / "chroma")}),
        "logging": _tmp_logging(tmp_path),
        # one attempt: these tests check a single real cycle; the retry loop is unit-tested
        "adaptation": SETTINGS.adaptation.model_copy(update={"max_adaptation_attempts": 1}),
    }
    fast_settings = SETTINGS.model_copy(update=overrides)
    orchestrator = ContinuousOrchestrator(fast_settings, SECRETS)
    orchestrator.initialize()
    return orchestrator


def test_start_and_stop_five_times_is_deterministic_and_error_free(tmp_path):
    """Phase13 §8.4: shutdown determinism — `stop()` must deterministically wait for the
    background sync thread to actually exit (bounded by
    `config.synchronization.shutdown_join_timeout_seconds`), not just signal and return
    immediately. Exercised 5 times in a row against the same orchestrator instance. Only one
    background thread exists: external drift (and its consumer thread) is permanently disabled."""
    orchestrator = _fast_mock_orchestrator(tmp_path)
    try:
        for i in range(5):
            orchestrator.start()
            time.sleep(0.1)  # let the background thread genuinely start producing
            assert orchestrator.sync_thread is not None and orchestrator.sync_thread.is_alive(), f"iteration {i}: sync thread never started"
            assert not any(t.name == "dt-drift-consumer" for t in threading.enumerate()), f"iteration {i}: a drift consumer thread exists"

            orchestrator.stop()

            assert not orchestrator.sync_thread.is_alive(), f"iteration {i}: dt-continuous-sync thread still alive after stop() returned"

            # A fresh MockTelemetrySource is needed for the next start() — the prior one was
            # closed by stop(); this mirrors how a real process would be reconfigured between runs.
            orchestrator.telemetry_source = orchestrator._build_telemetry_source()  # noqa: SLF001
            orchestrator.synchronizer = type(orchestrator.synchronizer)(
                orchestrator.telemetry_source, orchestrator.preprocessor, orchestrator.d1_store, orchestrator._settings.synchronization  # noqa: SLF001
            )
            orchestrator._stop_event.clear()  # noqa: SLF001
    finally:
        if orchestrator.sync_thread is not None and orchestrator.sync_thread.is_alive():
            orchestrator.stop()
