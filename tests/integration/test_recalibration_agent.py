"""Integration tests: Module 14 (Recalibration Agent) against the real Module 2/3/4 pipeline.

The key deliverable this file proves CONCRETELY, not just by code inspection (prompt.md §0.6/
§0.8, CLAUDE.md §8 "continuous operation: telemetry ingestion, D1 synchronization, and the
current production model keep serving/updating throughout recalibration"): while a real
`RecalibrationAgent.recalibrate()` call is in progress in the foreground, a real background
`ContinuousSynchronizer` thread keeps writing genuinely new telemetry into the SAME `D1Store` —
sampled repeatedly DURING the recalibration call's own wall-clock window, not just compared
before/after (which could pass by coincidence of ordering).
"""

from __future__ import annotations

import threading
import time

from src.adaptation.recalibration_agent import RecalibrationAgent
from src.common.config import load_settings
from src.dt_models.d1_model_store import D1Store
from src.dt_models.throughput import ThroughputModel
from src.registry.model_registry import ModelRegistry
from src.synchronization.sync import ContinuousSynchronizer
from src.telemetry.mock_source import MockTelemetrySource
from src.telemetry.preprocessing import TelemetryPreprocessor

SETTINGS = load_settings()


def _bootstrap(tmp_path):
    store = D1Store(
        current_state_path=tmp_path / "current.parquet",
        history_path=tmp_path / "history.parquet",
        quarantine_path=tmp_path / "quarantine.parquet",
        history_retention_rows=SETTINGS.storage.history_retention_rows,
    )
    mock_config = SETTINGS.telemetry.mock.model_copy(
        update={"num_ues": 25, "num_cells": 4, "missing_field_rate": 0.0, "out_of_range_rate": 0.0}
    )
    bootstrap_source = MockTelemetrySource(mock_config, max_records=1200, realtime=False)
    preprocessor = TelemetryPreprocessor(SETTINGS.telemetry.preprocessing)
    sync_config = SETTINGS.synchronization.model_copy(update={"batch_size": 100})
    bootstrap_sync = ContinuousSynchronizer(bootstrap_source, preprocessor, store, sync_config)
    bootstrap_sync.run()

    registry = ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")
    history = store.get_history()
    features = list(ThroughputModel.REQUIRED_FEATURES)
    bootstrap_model = ThroughputModel.from_settings(SETTINGS)
    bootstrap_model.train(history[features], history["throughput_mbps"])
    registry.register_version(
        component_instance=bootstrap_model,
        adaptation_type="bootstrap",
        parent_version_id=None,
        training_window={"n_rows": len(history)},
        evaluation_window={"n_rows": 0},
        evaluation_metrics=bootstrap_model.evaluate(history[features], history["throughput_mbps"]),
        status="production",
    )
    return store, registry, preprocessor


def test_recalibration_produces_a_real_candidate_beating_a_naive_baseline(tmp_path):
    store, registry, _ = _bootstrap(tmp_path)
    agent = RecalibrationAgent(SETTINGS, store, registry)

    result = agent.recalibrate(
        lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48, held_out_fraction=0.2
    )

    held_out_target_std = store.get_history()["throughput_mbps"].std()
    assert result.evaluation_metrics["rmse"] < held_out_target_std  # genuinely learned, not noise
    assert result.version.status == "candidate"
    assert registry.get_current_version("throughput").version_id != result.version.version_id


def test_telemetry_keeps_synchronizing_into_d1_while_recalibration_is_in_progress(tmp_path):
    """The concrete proof prompt.md §0.6/§0.8 require: samples D1's growth strictly DURING the
    wall-clock window a real recalibration call is executing, using a real background
    `ContinuousSynchronizer` thread and a real (foreground) `RecalibrationAgent.recalibrate()`
    call — not a before/after comparison that could pass by coincidence."""
    store, registry, preprocessor = _bootstrap(tmp_path)

    # A live, long-running telemetry source feeding the SAME D1Store the agent will read from —
    # realtime=True with a short interval so it keeps producing genuinely new records for the
    # entire duration of the test, exactly like the live NS-3/mock feed would in production.
    live_mock_config = SETTINGS.telemetry.mock.model_copy(
        update={"seed": 999, "num_ues": 10, "num_cells": 2, "emit_interval_seconds": 0.01,
                "missing_field_rate": 0.0, "out_of_range_rate": 0.0}
    )
    live_source = MockTelemetrySource(live_mock_config, max_records=None, realtime=True)
    live_sync_config = SETTINGS.synchronization.model_copy(update={"batch_size": 10, "batch_timeout_seconds": 0.05})
    live_synchronizer = ContinuousSynchronizer(live_source, preprocessor, store, live_sync_config)

    live_thread = live_synchronizer.start()
    time.sleep(0.3)  # let the background sync get going before recalibration starts

    # Use a slower model (more trees) so the foreground recalibration call has a real,
    # measurable duration for the sampler below to observe overlap within.
    slow_model_type = "random_forest"
    slow_params = {"n_estimators": 400, "max_depth": 12, "random_state": 42}

    def slow_throughput_factory():
        model = ThroughputModel.from_settings(SETTINGS)
        model._model_type = slow_model_type  # noqa: SLF001 - test-only override, not production code
        from src.dt_models.regressors import build_regressor

        model._model = build_regressor(slow_model_type, slow_params)  # noqa: SLF001
        return model

    agent = RecalibrationAgent(SETTINGS, store, registry)

    samples: list[tuple[float, int]] = []
    stop_sampling = threading.Event()

    def sample_loop():
        while not stop_sampling.is_set():
            samples.append((time.monotonic(), live_synchronizer.records_synced))
            time.sleep(0.02)

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()

    recalibration_start = time.monotonic()
    result = agent.recalibrate(slow_throughput_factory, "throughput_mbps", window_hours=48, held_out_fraction=0.2)
    recalibration_end = time.monotonic()

    stop_sampling.set()
    sampler.join(timeout=5.0)
    live_synchronizer.stop()
    live_thread.join(timeout=5.0)

    assert result.version.status == "candidate"  # the agent's own work still completed correctly

    during_recalibration = [
        records for ts, records in samples if recalibration_start <= ts <= recalibration_end
    ]
    assert len(during_recalibration) >= 2, (
        "recalibration completed too fast for the sampler to observe overlap — "
        f"duration={recalibration_end - recalibration_start:.3f}s, samples={len(samples)}"
    )
    assert during_recalibration[-1] > during_recalibration[0], (
        "D1's synchronized-record count did not grow WHILE recalibration was in progress — "
        "the live telemetry pipeline was blocked, violating prompt.md §0.6/§0.8"
    )
    # And the synchronizer kept running the whole time (never silently died mid-test).
    assert live_synchronizer.records_synced > during_recalibration[0]
