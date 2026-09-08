"""Integration test: `src/main.py`'s `ContinuousOrchestrator` — the full canonical loop (Modules
1-19) wired together and run for real, at a smaller/faster scale than
`scripts/run_orchestrator_demo.py` (the permanent, repo-tracked, full-scale validation script —
see CLAUDE.md's Phase 11 entry for real observed numbers). This test proves the same two things
automatically on every run: (1) one full real drift -> PPO -> agent -> verification -> lifecycle
cycle completes without error, and (2) live D1 telemetry synchronization never stops during that
cycle — the same real-background-thread-plus-sampler-thread proof Modules 14/15/16/19 already
established for individual agents, applied here to the real orchestrator.

No real ANTHROPIC_API_KEY is configured in this environment — the drift severity range is
narrowed to bias PPO toward `recalibrate` (real Module 13 held-out evidence: low severity
reliably selects it), which needs no LLM at all, exactly like `scripts/run_orchestrator_demo.py`'s
own documented rationale.
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
            update={"mock": SETTINGS.telemetry.mock.model_copy(update={"emit_interval_seconds": 0.05, "num_ues": 10, "num_cells": 2})}
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
        stop_sampling = threading.Event()

        def sample_loop() -> None:
            while not stop_sampling.is_set():
                samples.append((time.monotonic(), orchestrator.synchronizer.records_synced))
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

        records = orchestrator.lifecycle_agent.list_records()
        assert len(records) == 1
        record = records[0]
        assert record.affected_component in fast_settings.drift.valid_components
        assert record.rl_action in ("recalibrate", "regenerate", "expand_scope")
        assert record.verification_result in ("ACCEPT", "REJECT")
        assert record.final_status in ("promoted", "rejected")
        assert (record.final_status == "promoted") == (record.verification_result == "ACCEPT")

        report = orchestrator.lifecycle_agent.generate_maintenance_report(record)
        assert report.path.exists()
        assert record.affected_component in report.text
    finally:
        orchestrator.stop()
