"""Integration tests: Module 15 (Regeneration Agent) against the real Module 2/3/4 pipeline and
the real sandbox subprocess (no mocking except the LLM transport — see the module docstring in
`tests/unit/test_regeneration_agent.py` for why: no real ANTHROPIC_API_KEY is configured in this
environment).

The key deliverable this file proves CONCRETELY (prompt.md §0.6/§0.8, mirroring
`tests/integration/test_recalibration_agent.py`'s proof for Module 14, now under a heavier real
workload — an LLM call plus a genuinely sandboxed subprocess training run, not just an in-memory
retrain): while a real `RegenerationAgent.regenerate()` call is in progress in the foreground, a
real background `ContinuousSynchronizer` thread keeps writing genuinely new telemetry into the
SAME `D1Store` — sampled repeatedly DURING the call's own wall-clock window.
"""

from __future__ import annotations

import threading
import time

from src.adaptation.regeneration_agent import RegenerationAgent
from src.common.config import load_settings
from src.dt_models.d1_model_store import D1Store
from src.dt_models.throughput import ThroughputModel
from src.registry.model_registry import ModelRegistry
from src.sandbox.executor import SandboxExecutor
from src.synchronization.sync import ContinuousSynchronizer
from src.telemetry.mock_source import MockTelemetrySource
from src.telemetry.preprocessing import TelemetryPreprocessor

SETTINGS = load_settings()

VALID_SOURCE = """
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from src.dt_models.base import DTComponent

class RebuiltThroughput(DTComponent):
    COMPONENT_NAME = "throughput"
    DEPENDENCIES = ()
    REQUIRED_FEATURES = ("offered_load_mbps", "prb_utilization_pct", "sinr_db", "rsrp_dbm")
    OUTPUT_FIELD = "throughput_mbps_pred"

    def __init__(self):
        self._model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
        self._trained = False

    @property
    def is_trained(self):
        return self._trained

    def train(self, features, targets):
        self._model.fit(features[list(self.REQUIRED_FEATURES)], targets)
        self._trained = True

    def predict(self, features):
        preds = self._model.predict(features[list(self.REQUIRED_FEATURES)])
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features, targets):
        preds = self.predict(features)
        rmse = float(((preds - targets) ** 2).mean() ** 0.5)
        mae = float((preds - targets).abs().mean())
        return {"rmse": rmse, "mae": mae}

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)

    def load(self, path):
        self._model = joblib.load(path)
        self._trained = True
"""


class _FakeLLMClient:
    def __init__(self, source: str) -> None:
        self._source = source

    def complete_structured(self, prompt, schema, **kwargs):
        return schema(class_name="RebuiltThroughput", source_code=self._source, reasoning="integration test")


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


def _agent(tmp_path, store, registry, source=VALID_SOURCE) -> RegenerationAgent:
    sandbox = SandboxExecutor(
        workspace_root=tmp_path / "sandbox",
        timeout_seconds=SETTINGS.sandbox.timeout_seconds,
        max_output_bytes=SETTINGS.sandbox.max_output_bytes,
    )
    return RegenerationAgent(SETTINGS, store, registry, _FakeLLMClient(source), sandbox)


def test_regeneration_produces_a_real_sandboxed_candidate(tmp_path):
    store, registry, _ = _bootstrap(tmp_path)
    agent = _agent(tmp_path, store, registry)

    result = agent.regenerate(
        lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48, held_out_fraction=0.2
    )

    assert result.version.status == "candidate"
    assert result.version.model_class == "RebuiltThroughput"
    assert result.sandbox_result.accepted
    assert "rmse" in result.version.evaluation_metrics and result.evaluation_window["n_rows"] > 0
    assert registry.get_current_version("throughput").adaptation_type == "bootstrap"  # untouched


def test_telemetry_keeps_synchronizing_into_d1_while_regeneration_is_in_progress(tmp_path):
    """The concrete proof prompt.md §0.6/§0.8 require, for Module 15: samples D1's growth
    strictly DURING the wall-clock window a real regeneration call (LLM call + real sandboxed
    subprocess training) is executing."""
    store, registry, preprocessor = _bootstrap(tmp_path)

    live_mock_config = SETTINGS.telemetry.mock.model_copy(
        update={"seed": 999, "num_ues": 10, "num_cells": 2, "emit_interval_seconds": 0.01,
                "missing_field_rate": 0.0, "out_of_range_rate": 0.0}
    )
    live_source = MockTelemetrySource(live_mock_config, max_records=None, realtime=True)
    live_sync_config = SETTINGS.synchronization.model_copy(update={"batch_size": 10, "batch_timeout_seconds": 0.05})
    live_synchronizer = ContinuousSynchronizer(live_source, preprocessor, store, live_sync_config)

    live_thread = live_synchronizer.start()
    time.sleep(0.3)  # let the background sync get going before regeneration starts

    agent = _agent(tmp_path, store, registry)

    samples: list[tuple[float, int]] = []
    stop_sampling = threading.Event()

    def sample_loop():
        while not stop_sampling.is_set():
            samples.append((time.monotonic(), live_synchronizer.records_synced))
            time.sleep(0.02)

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()

    regeneration_start = time.monotonic()
    result = agent.regenerate(
        lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48, held_out_fraction=0.2
    )
    regeneration_end = time.monotonic()

    stop_sampling.set()
    sampler.join(timeout=5.0)
    live_synchronizer.stop()
    live_thread.join(timeout=5.0)

    assert result.version.status == "candidate"  # the agent's own work still completed correctly

    during_regeneration = [
        records for ts, records in samples if regeneration_start <= ts <= regeneration_end
    ]
    assert len(during_regeneration) >= 2, (
        "regeneration completed too fast for the sampler to observe overlap — "
        f"duration={regeneration_end - regeneration_start:.3f}s, samples={len(samples)}"
    )
    assert during_regeneration[-1] > during_regeneration[0], (
        "D1's synchronized-record count did not grow WHILE regeneration (LLM call + sandboxed "
        "subprocess training) was in progress — the live telemetry pipeline was blocked, "
        "violating prompt.md §0.6/§0.8"
    )
    assert live_synchronizer.records_synced > during_regeneration[0]
