"""Integration tests: Module 16 (Expand-Scope Agent) against the real Module 2/3/4/5 pipeline and
the real sandbox subprocess (LLM transport mocked — no real GOOGLE_API_KEY is configured in
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

from src.adaptation.expand_scope_agent import ExpandScopeAgent, _GeneratedComponentCode, _ProposedComponentDesign
from src.common.config import load_settings
from src.data.oai_dataset import ALL_FEATURES
from src.dt_models.base import DTComponent
from src.dt_models.d1_model_store import D1Store
from src.dt_models.dl_bler import DlBlerModel
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator
from src.registry.model_registry import ModelRegistry
from src.sandbox.executor import SandboxExecutor
from src.synchronization.sync import ContinuousSynchronizer
from src.telemetry.mock_source import MockTelemetrySource
from src.telemetry.preprocessing import TelemetryPreprocessor

SETTINGS = load_settings()

# A genuinely new target: a real raw feature column, never one of the 7 existing component
# targets — mirrors prompt.md's own SINR worked example (a raw INPUT no existing component
# predicts) using this OAI schema's real columns.
_NEW_TARGET_COLUMN = "wb_cqi_avg_lag1"
_NEW_REQUIRED_FEATURES = ("dl_curr_tbs_avg_lag1", "pusch_snr_avg_lag1", "ue_count_cell_lag1", "bsr_avg_lag1")

VALID_SOURCE = f"""
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from src.dt_models.base import DTComponent

class CqiQualityModel(DTComponent):
    COMPONENT_NAME = "cqi_quality"
    DEPENDENCIES = ()
    REQUIRED_FEATURES = {_NEW_REQUIRED_FEATURES!r}
    OUTPUT_FIELD = "{_NEW_TARGET_COLUMN}_pred"

    def __init__(self):
        self._model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
        self._trained = False

    @property
    def is_trained(self):
        return self._trained

    def train(self, features, targets):
        cols = features[list(self.REQUIRED_FEATURES)]
        valid = cols.notna().all(axis=1) & targets.notna()
        self._model.fit(cols[valid], targets[valid])
        self._trained = True

    def predict(self, features):
        preds = self._model.predict(features[list(self.REQUIRED_FEATURES)])
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features, targets):
        preds = self.predict(features)
        rmse = float(((preds - targets) ** 2).mean() ** 0.5)
        mae = float((preds - targets).abs().mean())
        return {{"rmse": rmse, "mae": mae}}

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)

    def load(self, path):
        self._model = joblib.load(path)
        self._trained = True
"""

DESIGN = _ProposedComponentDesign(
    component_name="cqi_quality",
    target_column=_NEW_TARGET_COLUMN,
    dependencies=(),
    required_features=_NEW_REQUIRED_FEATURES,
    purpose="Predict channel quality indicator from load and mobility for proactive scheduling.",
    feature_extraction_notes="none needed for this version",
)


class _FakeLLMClient:
    def complete_structured(self, prompt, schema, **kwargs):
        if schema is _ProposedComponentDesign:
            return DESIGN
        return schema(class_name="CqiQualityModel", source_code=VALID_SOURCE, reasoning="integration test")


def _bootstrap(tmp_path):
    store = D1Store(
        current_state_path=tmp_path / "current.parquet",
        history_path=tmp_path / "history.parquet",
        quarantine_path=tmp_path / "quarantine.parquet",
        history_retention_rows=SETTINGS.storage.history_retention_rows,
    )
    mock_config = SETTINGS.telemetry.mock.model_copy(
        update={"num_sessions": 25, "missing_feature_rate": 0.0}
    )
    bootstrap_source = MockTelemetrySource(mock_config, max_records=1200, realtime=False)
    preprocessor = TelemetryPreprocessor(SETTINGS.telemetry.preprocessing)
    sync_config = SETTINGS.synchronization.model_copy(update={"batch_size": 100})
    bootstrap_sync = ContinuousSynchronizer(bootstrap_source, preprocessor, store, sync_config)
    bootstrap_sync.run()

    model_registry = ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")
    dt_model_registry = DTModelRegistry()
    history = store.get_history()
    features = list(DlBlerModel.REQUIRED_FEATURES)
    bootstrap_model = DlBlerModel.from_settings(SETTINGS)
    bootstrap_model.train(history[features], history["dl_bler"])
    model_registry.register_version(
        component_instance=bootstrap_model,
        adaptation_type="bootstrap",
        parent_version_id=None,
        training_window={"n_rows": len(history)},
        evaluation_window={"n_rows": 0},
        evaluation_metrics=bootstrap_model.evaluate(history[features], history["dl_bler"]),
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
        lambda: DlBlerModel.from_settings(SETTINGS),
        window_hours=48,
        held_out_fraction=0.2,
        expand_scope_context="CQI degradation not explained by any existing component",
    )

    assert result.version.component == "cqi_quality"
    assert result.version.status == "candidate"
    assert result.version.parent_version_id is None
    assert result.sandbox_result.accepted
    assert "rmse" in result.version.evaluation_metrics
    # The live orchestrator registry is untouched — see module docstring for why.
    assert [c.COMPONENT_NAME for c in dt_model_registry.list_components()] == ["dl_bler"]


def test_telemetry_keeps_synchronizing_into_d1_while_expand_scope_is_in_progress(tmp_path):
    store, model_registry, dt_model_registry, preprocessor = _bootstrap(tmp_path)

    live_mock_config = SETTINGS.telemetry.mock.model_copy(
        update={"seed": 999, "num_sessions": 10, "emit_interval_seconds": 0.01, "missing_feature_rate": 0.0}
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
    result = agent.expand_scope(lambda: DlBlerModel.from_settings(SETTINGS), window_hours=48, held_out_fraction=0.2)
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


class _TrustedCqiQualityModel(DTComponent):
    """A TRUSTED, test-authored stand-in for what Module 17 would eventually promote and load —
    NEVER the raw LLM/sandbox output (see this file's module docstring and
    `expand_scope_agent.py`'s for why importing that directly here would be unsafe/premature).
    Used only to prove `DTModelRegistry`/`DTOrchestrator` place no hardcoded limit on component
    names or count — the exact mechanism a future, vetted promotion step would rely on."""

    COMPONENT_NAME = "cqi_quality"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES = ("dl_curr_tbs_avg_lag1", "pusch_snr_avg_lag1")
    OUTPUT_FIELD = f"{_NEW_TARGET_COLUMN}_pred"

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
    changed there for this module) correctly schedule and run an EIGHTH component whose name
    never appears anywhere in Module 5's own source, alongside the real, already-registered
    7 OAI KPI components."""
    from src.dt_models.dl_prb import DlPrbModel
    from src.dt_models.dl_rlc_delay import DlRlcDelayModel
    from src.dt_models.dl_throughput import DlThroughputModel
    from src.dt_models.ul_bler import UlBlerModel
    from src.dt_models.ul_prb import UlPrbModel
    from src.dt_models.ul_throughput import UlThroughputModel

    store, model_registry, dt_model_registry, _ = _bootstrap(tmp_path)
    history = store.get_history()

    dl_bler = dt_model_registry.get("dl_bler")  # already registered + trained by _bootstrap

    ul_bler = UlBlerModel.from_settings(SETTINGS)
    ul_bler.train(history[list(UlBlerModel.REQUIRED_FEATURES)], history["ul_bler"])
    dt_model_registry.register(ul_bler)

    dl_prb = DlPrbModel.from_settings(SETTINGS)
    dl_prb.train(history[list(DlPrbModel.REQUIRED_FEATURES)], history["dl_prb"])
    dt_model_registry.register(dl_prb)

    ul_prb = UlPrbModel.from_settings(SETTINGS)
    ul_prb.train(history[list(UlPrbModel.REQUIRED_FEATURES)], history["ul_prb"])
    dt_model_registry.register(ul_prb)

    dl_throughput = DlThroughputModel.from_settings(SETTINGS)
    dlt_train = history.assign(dl_bler_pred=history["dl_bler"], dl_prb_pred=history["dl_prb"])
    dlt_cols = list(DlThroughputModel.REQUIRED_FEATURES) + ["dl_bler_pred", "dl_prb_pred"]
    dl_throughput.train(dlt_train[dlt_cols], dlt_train["dl_throughput"])
    dt_model_registry.register(dl_throughput)

    ul_throughput = UlThroughputModel.from_settings(SETTINGS)
    ult_train = history.assign(ul_bler_pred=history["ul_bler"], ul_prb_pred=history["ul_prb"])
    ult_cols = list(UlThroughputModel.REQUIRED_FEATURES) + ["ul_bler_pred", "ul_prb_pred"]
    ul_throughput.train(ult_train[ult_cols], ult_train["ul_throughput"])
    dt_model_registry.register(ul_throughput)

    dl_rlc_delay = DlRlcDelayModel.from_settings(SETTINGS)
    delay_train = history.assign(dl_throughput_pred=history["dl_throughput"], dl_bler_pred=history["dl_bler"])
    delay_cols = list(DlRlcDelayModel.REQUIRED_FEATURES) + ["dl_throughput_pred", "dl_bler_pred"]
    dl_rlc_delay.train(delay_train[delay_cols], delay_train["dl_rlc_delay"])
    dt_model_registry.register(dl_rlc_delay)

    # The genuinely NEW, eighth component — its name appears nowhere in Module 5's source.
    cqi_quality = _TrustedCqiQualityModel()
    cqi_quality.train(history[list(_TrustedCqiQualityModel.REQUIRED_FEATURES)], history[_NEW_TARGET_COLUMN])
    dt_model_registry.register(cqi_quality)

    assert len(dt_model_registry.list_components()) == 8

    orchestrator = DTOrchestrator(dt_model_registry)
    order = orchestrator.build_execution_order()
    assert "cqi_quality" in order
    assert order.index("dl_bler") < order.index("cqi_quality") or "dl_bler" not in order  # no false dependency

    predictions = orchestrator.run_predictions(history)
    assert "cqi_quality" in predictions
    assert len(predictions["cqi_quality"]) == len(history)
    # .equals() (not ==) because dl_bler's residual-reconstruction predict() legitimately returns
    # NaN for each session's first row (no y_t-1 anchor) — NaN != NaN under ==, which would fail
    # this assertion even though the two predictions are genuinely identical.
    assert predictions["dl_bler"].equals(dl_bler.predict(history[list(DlBlerModel.REQUIRED_FEATURES)]))
