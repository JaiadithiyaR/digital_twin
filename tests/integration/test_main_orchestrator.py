"""Integration test: `src/main.py`'s `ContinuousOrchestrator` — the full canonical loop (Modules
1-19) wired together and run for real, at a smaller/faster scale than
`scripts/run_orchestrator_demo.py` (the permanent, repo-tracked, full-scale validation script —
see CLAUDE.md's Phase 11 entry for real observed numbers). This test proves the same two things
automatically on every run: (1) one full real drift -> Decision & Root-Cause Analysis Agent ->
agent -> verification -> lifecycle cycle completes without error, and (2) live D1 telemetry
synchronization never stops during that cycle — the same real-background-thread-plus-sampler-
thread proof Modules 14/15/16/19 already established for individual agents, applied here to the
real orchestrator.

Reads whatever `GOOGLE_API_KEY`/`GEMINI_API_KEY` is actually configured in this environment's real
`.env` via `load_secrets()` — if none is set, the decision call genuinely fails and
`DecisionAgent.decide_safe()` degrades to `config.decision_agent.fallback.default_strategy`
(`recalibrate`, which needs no LLM to execute), logged at WARNING; if a real key IS set, this test
genuinely exercises a live decision call (observed real behavior: `recalibrate` selected both with
and without a real key, so this test's assertions are written to hold either way — never assuming
which). The narrowed drift severity range is kept only for otherwise-stable test behavior — it no
longer influences which strategy is selected either way, since the fallback is
severity-independent and the real decision agent judges the actual context, not just severity.
"""

from __future__ import annotations

import threading
import time

from src.common.config import load_secrets, load_settings
from src.main import ContinuousOrchestrator

SETTINGS = load_settings()
SECRETS = load_secrets()

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
        "drift": SETTINGS.drift.model_copy(update={"mock": SETTINGS.drift.mock.model_copy(update={"emit_interval_seconds": 5.0})}),
        "lifecycle": SETTINGS.lifecycle.model_copy(
            update={"records_path": str(tmp_path / "lifecycle.jsonl"), "reports_dir": str(tmp_path / "reports")}
        ),
        "rag": SETTINGS.rag.model_copy(update={"chroma_path": str(tmp_path / "chroma")}),
    }
    fast_settings = SETTINGS.model_copy(update=overrides)

    orchestrator = ContinuousOrchestrator(fast_settings, SECRETS, drift_severity_range_override=(0.05, 0.2))
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

        samples: list[tuple[float, int]] = []
        lock_samples: list[tuple[float, str | None]] = []
        stop_sampling = threading.Event()

        def sample_loop() -> None:
            while not stop_sampling.is_set():
                now = time.monotonic()
                samples.append((now, orchestrator.synchronizer.records_synced))
                lock_samples.append((now, orchestrator._locked_component))  # noqa: SLF001 - phase13 §2 lock, read-only inspection
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
        locked_during = [c for ts, c in lock_samples if start_ts <= ts <= end_ts and c is not None]
        assert locked_during, "orchestrator._locked_component was never observed set during the real adaptation cycle"
        assert orchestrator._locked_component is None, "adaptation lock was not released after the cycle finished"  # noqa: SLF001

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
        "drift": SETTINGS.drift.model_copy(update={"mock": SETTINGS.drift.mock.model_copy(update={"emit_interval_seconds": 5.0})}),
        "lifecycle": SETTINGS.lifecycle.model_copy(
            update={"records_path": str(tmp_path / "lifecycle.jsonl"), "reports_dir": str(tmp_path / "reports")}
        ),
        "rag": SETTINGS.rag.model_copy(update={"chroma_path": str(tmp_path / "chroma")}),
    }
    fast_settings = SETTINGS.model_copy(update=overrides)
    orchestrator = ContinuousOrchestrator(fast_settings, SECRETS)
    orchestrator.initialize()
    return orchestrator


def test_start_and_stop_five_times_is_deterministic_and_error_free(tmp_path):
    """Phase13 §8.4: the dt-continuous-sync/dt-drift-consumer shutdown race — `stop()` must
    deterministically wait for both background threads to actually exit (bounded by
    `config.synchronization.shutdown_join_timeout_seconds`), not just signal and return
    immediately. Exercised 5 times in a row against the same orchestrator instance, each time
    confirming both threads are genuinely dead before the next start()."""
    orchestrator = _fast_mock_orchestrator(tmp_path)
    try:
        for i in range(5):
            orchestrator.start()
            time.sleep(0.1)  # let both background threads genuinely start producing/consuming
            assert orchestrator.sync_thread is not None and orchestrator.sync_thread.is_alive(), f"iteration {i}: sync thread never started"

            orchestrator.stop()

            assert not orchestrator.sync_thread.is_alive(), f"iteration {i}: dt-continuous-sync thread still alive after stop() returned"
            assert not orchestrator.drift_thread.is_alive(), f"iteration {i}: dt-drift-consumer thread still alive after stop() returned"

            # A fresh MockTelemetrySource/MockDriftSource is needed for the next start() — the
            # prior ones were closed by stop(); this mirrors how a real process would be
            # reconfigured between runs, not a gap this test papers over.
            orchestrator.telemetry_source = orchestrator._build_telemetry_source()  # noqa: SLF001
            orchestrator.synchronizer = type(orchestrator.synchronizer)(
                orchestrator.telemetry_source, orchestrator.preprocessor, orchestrator.d1_store, orchestrator._settings.synchronization  # noqa: SLF001
            )
            from src.drift.mock_drift_source import MockDriftSource

            orchestrator.drift_source = MockDriftSource(
                orchestrator._settings.drift.mock, valid_components=orchestrator._settings.drift.valid_components,  # noqa: SLF001
                severity_range=orchestrator._settings.drift.severity_range, max_events=None, realtime=True,  # noqa: SLF001
            )
            orchestrator._stop_event.clear()  # noqa: SLF001
    finally:
        if orchestrator.sync_thread is not None and orchestrator.sync_thread.is_alive():
            orchestrator.stop()
