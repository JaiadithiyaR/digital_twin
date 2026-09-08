"""Integration tests: Module 16 (Expand-Scope Agent) against the real Module 2/3/4/5 pipeline and
the real sandbox subprocess (LLM transport mocked — no real ANTHROPIC_API_KEY is configured in
this environment, same situation as Modules 15/"LLM Infrastructure").

Two deliverables this file proves CONCRETELY:

1. **Continuous telemetry synchronization during expand-scope** (prompt.md §0.6/§0.8), mirroring
   Modules 14/15's proof under the same real workload shape (LLM calls + a genuinely sandboxed
   subprocess training run).
2. **`DTModelRegistry`/`DTOrchestrator`'s dynamic registration has no hardcoded component-count
   or name limit** (prompt.md §27: "the new component must be added dynamically through the
   registry") — proven with a TRUSTED, test-authored stand-in component representing what a
   FUTURE verification/promotion step would eventually activate, never the raw LLM/sandbox
   output directly. See `src/adaptation/expand_scope_agent.py`'s module docstring for exactly why
   this agent itself does not (and must not, before Module 17 exists) import or `joblib.load()`
   its own sandboxed candidate's artifact into the live process.
"""

from __future__ import annotations

import threading
import time

import pandas as pd

from src.adaptation.expand_scope_agent import ExpandScopeAgent
from src.common.config import load_settings
from src.dt_models.base import DTComponent
from src.dt_models.d1_model_store import D1Store
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator
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

class SinrQualityModel(DTComponent):
    COMPONENT_NAME = "sinr_quality"
    DEPENDENCIES = ()
    REQUIRED_FEATURES = ("offered_load_mbps", "prb_utilization_pct", "ue_speed_mps", "rsrp_dbm")
    OUTPUT_FIELD = "sinr_db_pred"

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

from src.adaptation.expand_scope_agent import _GeneratedComponentCode, _ProposedComponentDesign  # noqa: E402

DESIGN = _ProposedComponentDesign(
    component_name="sinr_quality",
    target_column="sinr_db",
    dependencies=(),
    required_features=("offered_load_mbps", "prb_utilization_pct", "ue_speed_mps", "rsrp_dbm"),
    purpose="Predict SINR from load and mobility for proactive handover decisions.",
    feature_extraction_notes="none needed for this version",
)


class _FakeLLMClient:
    def complete_structured(self, prompt, schema, **kwargs):
        if schema is _ProposedComponentDesign:
            return DESIGN
        return schema(class_name="SinrQualityModel", source_code=VALID_SOURCE, reasoning="integration test")


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

    model_registry = ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")
    dt_model_registry = DTModelRegistry()
    history = store.get_history()
    features = list(ThroughputModel.REQUIRED_FEATURES)
    bootstrap_model = ThroughputModel.from_settings(SETTINGS)
    bootstrap_model.train(history[features], history["throughput_mbps"])
    model_registry.register_version(
        component_instance=bootstrap_model,
        adaptation_type="bootstrap",
        parent_version_id=None,
        training_window={"n_rows": len(history)},
        evaluation_window={"n_rows": 0},
        evaluation_metrics=bootstrap_model.evaluate(history[features], history["throughput_mbps"]),
        status="production",
    )
    dt_model_registry.register(bootstrap_model)
    return store, model_registry, dt_model_registry, preprocessor


def _agent(tmp_path, store, model_registry, dt_model_registry) -> ExpandScopeAgent:
    sandbox = SandboxExecutor(
        workspace_root=tmp_path / "sandbox",
        timeout_seconds=SETTINGS.sandbox.timeout_seconds,
        max_output_bytes=SETTINGS.sandbox.max_output_bytes,
    )
    return ExpandScopeAgent(SETTINGS, store, model_registry, dt_model_registry, _FakeLLMClient(), sandbox)


def test_expand_scope_produces_a_real_sandboxed_new_component(tmp_path):
    store, model_registry, dt_model_registry, _ = _bootstrap(tmp_path)
    agent = _agent(tmp_path, store, model_registry, dt_model_registry)

    result = agent.expand_scope(
        lambda: ThroughputModel.from_settings(SETTINGS),
        window_hours=48,
        held_out_fraction=0.2,
        expand_scope_context="SINR degradation not explained by any existing component",
    )

    assert result.version.component == "sinr_quality"
    assert result.version.status == "candidate"
    assert result.version.parent_version_id is None
    assert result.sandbox_result.accepted
    assert "rmse" in result.version.evaluation_metrics
    # The live orchestrator registry is untouched — see module docstring for why.
    assert [c.COMPONENT_NAME for c in dt_model_registry.list_components()] == ["throughput"]


def test_telemetry_keeps_synchronizing_into_d1_while_expand_scope_is_in_progress(tmp_path):
    store, model_registry, dt_model_registry, preprocessor = _bootstrap(tmp_path)

    live_mock_config = SETTINGS.telemetry.mock.model_copy(
        update={"seed": 999, "num_ues": 10, "num_cells": 2, "emit_interval_seconds": 0.01,
                "missing_field_rate": 0.0, "out_of_range_rate": 0.0}
    )
    live_source = MockTelemetrySource(live_mock_config, max_records=None, realtime=True)
    live_sync_config = SETTINGS.synchronization.model_copy(update={"batch_size": 10, "batch_timeout_seconds": 0.05})
    live_synchronizer = ContinuousSynchronizer(live_source, preprocessor, store, live_sync_config)

    live_thread = live_synchronizer.start()
    time.sleep(0.3)

    agent = _agent(tmp_path, store, model_registry, dt_model_registry)

    samples: list[tuple[float, int]] = []
    stop_sampling = threading.Event()

    def sample_loop():
        while not stop_sampling.is_set():
            samples.append((time.monotonic(), live_synchronizer.records_synced))
            time.sleep(0.02)

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()

    start = time.monotonic()
    result = agent.expand_scope(lambda: ThroughputModel.from_settings(SETTINGS), window_hours=48, held_out_fraction=0.2)
    end = time.monotonic()

    stop_sampling.set()
    sampler.join(timeout=5.0)
    live_synchronizer.stop()
    live_thread.join(timeout=5.0)

    assert result.version.status == "candidate"

    during = [records for ts, records in samples if start <= ts <= end]
    assert len(during) >= 2, (
        f"expand-scope completed too fast for the sampler to observe overlap — "
        f"duration={end - start:.3f}s, samples={len(samples)}"
    )
    assert during[-1] > during[0], (
        "D1's synchronized-record count did not grow WHILE expand-scope (LLM calls + sandboxed "
        "subprocess training) was in progress — the live telemetry pipeline was blocked, "
        "violating prompt.md §0.6/§0.8"
    )
    assert live_synchronizer.records_synced > during[0]


class _TrustedSinrQualityModel(DTComponent):
    """A TRUSTED, test-authored stand-in for what Module 17 would eventually promote and load —
    NEVER the raw LLM/sandbox output (see this file's module docstring and
    `expand_scope_agent.py`'s for why importing that directly here would be unsafe/premature).
    Used only to prove `DTModelRegistry`/`DTOrchestrator` place no hardcoded limit on component
    names or count — the exact mechanism a future, vetted promotion step would rely on."""

    COMPONENT_NAME = "sinr_quality"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES = ("offered_load_mbps", "prb_utilization_pct")
    OUTPUT_FIELD = "sinr_db_pred"

    def __init__(self) -> None:
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        self._mean = float(targets.mean())
        self._trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series([self._mean] * len(features), index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        return {"rmse": float(((preds - targets) ** 2).mean() ** 0.5)}

    def save(self, path) -> None:
        pass

    def load(self, path) -> None:
        pass


def test_dt_model_registry_dynamically_accepts_a_brand_new_component_alongside_existing_ones(tmp_path):
    """Proves prompt.md §27's "must be added dynamically through the registry" at the mechanism
    level: `DTModelRegistry`/`DTOrchestrator` (Module 5, unmodified — no code was added or
    changed there for this module) correctly schedule and run a SIXTH component whose name never
    appears anywhere in Module 5's own source, alongside the five real, already-registered
    Modules 6-10 components."""
    from src.dt_models.jitter import JitterModel
    from src.dt_models.latency import LatencyModel
    from src.dt_models.packet_loss import PacketLossModel
    from src.dt_models.prb_utilization import PrbUtilizationModel

    store, model_registry, dt_model_registry, _ = _bootstrap(tmp_path)
    history = store.get_history()

    throughput = dt_model_registry.get("throughput")  # already registered + trained by _bootstrap

    packet_loss = PacketLossModel.from_settings(SETTINGS)
    packet_loss.train(history[list(PacketLossModel.REQUIRED_FEATURES)], history["packet_loss_pct"])
    dt_model_registry.register(packet_loss)

    latency = LatencyModel.from_settings(SETTINGS)
    latency_train = history.assign(throughput_mbps_pred=history["throughput_mbps"])
    latency.train(latency_train[list(LatencyModel.REQUIRED_FEATURES) + ["throughput_mbps_pred"]], latency_train["latency_ms"])
    dt_model_registry.register(latency)

    prb = PrbUtilizationModel.from_settings(SETTINGS)
    prb_train = history.assign(throughput_mbps_pred=history["throughput_mbps"])
    prb.train(prb_train[list(PrbUtilizationModel.REQUIRED_FEATURES) + ["throughput_mbps_pred"]], prb_train["prb_utilization_pct"])
    dt_model_registry.register(prb)

    jitter = JitterModel.from_settings(SETTINGS)
    jitter_train = history.assign(
        throughput_mbps_pred=history["throughput_mbps"],
        latency_ms_pred=history["latency_ms"],
        packet_loss_pct_pred=history["packet_loss_pct"],
    )
    jitter_cols = list(JitterModel.REQUIRED_FEATURES) + ["throughput_mbps_pred", "latency_ms_pred", "packet_loss_pct_pred"]
    jitter.train(jitter_train[jitter_cols], jitter_train["jitter_ms"])
    dt_model_registry.register(jitter)

    # The genuinely NEW, sixth component — its name appears nowhere in Module 5's source.
    sinr_quality = _TrustedSinrQualityModel()
    sinr_quality.train(history[list(_TrustedSinrQualityModel.REQUIRED_FEATURES)], history["sinr_db"])
    dt_model_registry.register(sinr_quality)

    assert len(dt_model_registry.list_components()) == 6

    orchestrator = DTOrchestrator(dt_model_registry)
    order = orchestrator.build_execution_order()
    assert "sinr_quality" in order
    assert order.index("throughput") < order.index("sinr_quality") or "throughput" not in order  # no false dependency

    predictions = orchestrator.run_predictions(history)
    assert "sinr_quality" in predictions
    assert len(predictions["sinr_quality"]) == len(history)
    assert (predictions["throughput"] == throughput.predict(history[list(ThroughputModel.REQUIRED_FEATURES)])).all()
