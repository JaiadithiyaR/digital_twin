"""Unit tests for ExpandScopeAgent (Module 16, prompt.md §27).

No real ANTHROPIC_API_KEY is configured in this environment (same situation as Modules 15/
"LLM Infrastructure") — the LLM client's `complete_structured` is replaced with a fake that
dispatches on the requested schema (design proposal vs. implementation), returning hand-written
values standing in for real model output. Everything downstream — deterministic design
validation, the real sandbox execution, the registry writes, the fidelity computation — is REAL,
unmocked code.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adaptation.expand_scope_agent import (
    ExpandScopeAgent,
    ExpandScopeError,
    _GeneratedComponentCode,
    _ProposedComponentDesign,
)
from src.common.config import load_settings
from src.dt_models.d1_model_store import D1Store
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.throughput import ThroughputModel
from src.fidelity.evaluator import FidelityEvaluator
from src.llm.anthropic_client import LLMClientError
from src.registry.model_registry import ModelRegistry
from src.sandbox.executor import SandboxExecutor
from src.telemetry.schema import CleanTelemetryRecord, RecordQuality

SETTINGS = load_settings()

VALID_DESIGN = _ProposedComponentDesign(
    component_name="sinr_quality",
    target_column="sinr_db",
    dependencies=(),
    required_features=("offered_load_mbps", "prb_utilization_pct", "ue_speed_mps"),
    purpose="Predict SINR from load and mobility for proactive handover decisions.",
    feature_extraction_notes="none needed",
)

VALID_SOURCE = """
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from src.dt_models.base import DTComponent

class SinrQualityModel(DTComponent):
    COMPONENT_NAME = "sinr_quality"
    DEPENDENCIES = ()
    REQUIRED_FEATURES = ("offered_load_mbps", "prb_utilization_pct", "ue_speed_mps")
    OUTPUT_FIELD = "sinr_db_pred"

    def __init__(self):
        self._model = LinearRegression()
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
        return {"rmse": rmse}

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)

    def load(self, path):
        self._model = joblib.load(path)
        self._trained = True
"""

BROKEN_SOURCE = """
from src.dt_models.base import DTComponent
class SinrQualityModel(DTComponent):
    COMPONENT_NAME = "sinr_quality"
    OUTPUT_FIELD = "sinr_db_pred"
    def __init__(self):
        raise RuntimeError("intentionally broken")
    @property
    def is_trained(self):
        return False
    def train(self, f, t): pass
    def predict(self, f): pass
    def evaluate(self, f, t): return {}
    def save(self, p): pass
    def load(self, p): pass
"""


class _FakeLLMClient:
    def __init__(self, designs=None, implementations=None, raise_on_call: Exception | None = None):
        self._designs = list(designs) if designs is not None else [VALID_DESIGN]
        self._implementations = list(implementations) if implementations is not None else None
        self._raise_on_call = raise_on_call
        self.design_calls: list[str] = []
        self.implementation_calls: list[str] = []

    def complete_structured(self, prompt, schema, **kwargs):
        if self._raise_on_call is not None:
            raise self._raise_on_call
        if schema is _ProposedComponentDesign:
            self.design_calls.append(prompt)
            return self._designs[len(self.design_calls) - 1]
        self.implementation_calls.append(prompt)
        if self._implementations is not None:
            return self._implementations[len(self.implementation_calls) - 1]
        return schema(class_name="SinrQualityModel", source_code=VALID_SOURCE, reasoning="linear model")


def _clean_record(ue_id: str, cell_id: str, timestamp: datetime, **overrides) -> CleanTelemetryRecord:
    base = dict(
        timestamp=timestamp, ue_id=ue_id, cell_id=cell_id,
        throughput_mbps=float(np.random.uniform(5, 50)), offered_load_mbps=float(np.random.uniform(5, 60)),
        latency_ms=float(np.random.uniform(1, 20)), jitter_ms=float(np.random.uniform(0, 2)),
        packet_loss_pct=float(np.random.uniform(0, 5)), prb_utilization_pct=float(np.random.uniform(10, 90)),
        sinr_db=float(np.random.uniform(0, 30)), rsrp_dbm=float(np.random.uniform(-120, -60)),
        rsrq_db=float(np.random.uniform(-15, -5)), ue_count=int(np.random.randint(1, 10)),
        ue_speed_mps=float(np.random.uniform(0, 15)), source="MOCK", quality=RecordQuality(),
    )
    base.update(overrides)
    return CleanTelemetryRecord(**base)


def _build_store(tmp_path: Path, n_rows: int = 300, hours_back_span: float = 2.0) -> D1Store:
    store = D1Store(
        current_state_path=tmp_path / "current.parquet", history_path=tmp_path / "history.parquet",
        quarantine_path=tmp_path / "quarantine.parquet", history_retention_rows=SETTINGS.storage.history_retention_rows,
    )
    now = datetime.now(UTC)
    records = []
    for i in range(n_rows):
        ts = now - timedelta(hours=hours_back_span * (1 - i / max(n_rows - 1, 1)))
        records.append(_clean_record(f"ue-{i % 10}", f"cell-{i % 2}", ts))
    store.append_history(records)
    return store


def _registry_with_throughput(tmp_path: Path, store: D1Store) -> tuple[ModelRegistry, DTModelRegistry]:
    registry = ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")
    dt_model_registry = DTModelRegistry()
    dt_model_registry.register(ThroughputModel.from_settings(SETTINGS))
    return registry, dt_model_registry


def _agent(tmp_path, store, model_registry, dt_model_registry, llm_client, fidelity_evaluator=None) -> ExpandScopeAgent:
    sandbox = SandboxExecutor(
        workspace_root=tmp_path / "sandbox",
        timeout_seconds=SETTINGS.sandbox.timeout_seconds,
        max_output_bytes=SETTINGS.sandbox.max_output_bytes,
    )
    return ExpandScopeAgent(
        SETTINGS, store, model_registry, dt_model_registry, llm_client, sandbox, fidelity_evaluator=fidelity_evaluator
    )


def _example_factory():
    return lambda: ThroughputModel.from_settings(SETTINGS)


def test_raises_when_window_has_too_few_rows(tmp_path):
    store = _build_store(tmp_path)
    registry, dt_model_registry = _registry_with_throughput(tmp_path, store)
    agent = _agent(tmp_path, store, registry, dt_model_registry, _FakeLLMClient())
    with pytest.raises(ExpandScopeError, match="need >="):
        agent.expand_scope(_example_factory(), window_hours=1e-9)


def test_design_rejected_when_name_collides_with_existing_component(tmp_path):
    store = _build_store(tmp_path)
    registry, dt_model_registry = _registry_with_throughput(tmp_path, store)
    colliding = VALID_DESIGN.model_copy(update={"component_name": "throughput"})
    llm = _FakeLLMClient(designs=[colliding, colliding])
    agent = _agent(tmp_path, store, registry, dt_model_registry, llm)
    with pytest.raises(ExpandScopeError, match="already exists"):
        agent.expand_scope(_example_factory(), window_hours=48)


def test_design_rejected_when_component_already_has_versions(tmp_path):
    store = _build_store(tmp_path)
    registry, dt_model_registry = _registry_with_throughput(tmp_path, store)
    # "packet_loss" is NOT in `dt_model_registry` (only "throughput" is, per
    # `_registry_with_throughput`) but already has a registered version in `ModelRegistry` —
    # this exercises the SECOND validation check specifically, distinct from the "already exists
    # in dt_model_registry" check exercised above.
    from src.dt_models.packet_loss import PacketLossModel

    registry.register_version(
        component_instance=PacketLossModel.from_settings(SETTINGS),
        adaptation_type="bootstrap", parent_version_id=None, training_window={}, evaluation_window={},
        evaluation_metrics={}, status="production",
    )
    already_used = VALID_DESIGN.model_copy(update={"component_name": "packet_loss"})
    llm = _FakeLLMClient(designs=[already_used, already_used])
    agent = _agent(tmp_path, store, registry, dt_model_registry, llm)
    with pytest.raises(ExpandScopeError, match="already has registered versions"):
        agent.expand_scope(_example_factory(), window_hours=48)


def test_design_rejected_when_target_column_not_real(tmp_path):
    store = _build_store(tmp_path)
    registry, dt_model_registry = _registry_with_throughput(tmp_path, store)
    bad = VALID_DESIGN.model_copy(update={"target_column": "not_a_real_column"})
    llm = _FakeLLMClient(designs=[bad, bad])
    agent = _agent(tmp_path, store, registry, dt_model_registry, llm)
    with pytest.raises(ExpandScopeError, match="not a real D1 telemetry column"):
        agent.expand_scope(_example_factory(), window_hours=48)


def test_design_rejected_when_dependency_unknown(tmp_path):
    store = _build_store(tmp_path)
    registry, dt_model_registry = _registry_with_throughput(tmp_path, store)
    bad = VALID_DESIGN.model_copy(update={"dependencies": ("nonexistent_component",)})
    llm = _FakeLLMClient(designs=[bad, bad])
    agent = _agent(tmp_path, store, registry, dt_model_registry, llm)
    with pytest.raises(ExpandScopeError, match="not existing registered components"):
        agent.expand_scope(_example_factory(), window_hours=48)


def test_design_rejected_when_required_features_unavailable(tmp_path):
    store = _build_store(tmp_path)
    registry, dt_model_registry = _registry_with_throughput(tmp_path, store)
    bad = VALID_DESIGN.model_copy(update={"required_features": ("totally_made_up_column",)})
    llm = _FakeLLMClient(designs=[bad, bad])
    agent = _agent(tmp_path, store, registry, dt_model_registry, llm)
    with pytest.raises(ExpandScopeError, match="not available D1 columns"):
        agent.expand_scope(_example_factory(), window_hours=48)


def test_design_self_correction_succeeds_on_second_attempt(tmp_path):
    store = _build_store(tmp_path)
    registry, dt_model_registry = _registry_with_throughput(tmp_path, store)
    bad = VALID_DESIGN.model_copy(update={"target_column": "not_real"})
    llm = _FakeLLMClient(designs=[bad, VALID_DESIGN])
    agent = _agent(tmp_path, store, registry, dt_model_registry, llm)

    result = agent.expand_scope(_example_factory(), window_hours=48)

    assert result.design_attempts == 2
    assert "REJECTED" in llm.design_calls[1]


def test_successful_expand_scope_registers_a_new_component(tmp_path):
    store = _build_store(tmp_path)
    registry, dt_model_registry = _registry_with_throughput(tmp_path, store)
    agent = _agent(tmp_path, store, registry, dt_model_registry, _FakeLLMClient())

    result = agent.expand_scope(_example_factory(), window_hours=48, expand_scope_context="SINR anomaly")

    assert result.version.component == "sinr_quality"
    assert result.version.status == "candidate"
    assert result.version.adaptation_type == "expand_scope"
    assert result.version.parent_version_id is None  # genuinely new — no parent
    assert result.version.fidelity_before is None  # no prior version to compare against
    assert result.version.output_field == "sinr_db_pred"
    assert result.version.source_path is not None
    assert registry.list_versions("sinr_quality")[0].version_id == "sinr_quality-v1"


def test_dt_model_registry_untouched_by_expand_scope(tmp_path):
    store = _build_store(tmp_path)
    registry, dt_model_registry = _registry_with_throughput(tmp_path, store)
    agent = _agent(tmp_path, store, registry, dt_model_registry, _FakeLLMClient())

    agent.expand_scope(_example_factory(), window_hours=48)

    assert [c.COMPONENT_NAME for c in dt_model_registry.list_components()] == ["throughput"]


def test_d1_never_written_to(tmp_path):
    store = _build_store(tmp_path)
    registry, dt_model_registry = _registry_with_throughput(tmp_path, store)
    agent = _agent(tmp_path, store, registry, dt_model_registry, _FakeLLMClient())

    before_rows = len(store.get_history())
    agent.expand_scope(_example_factory(), window_hours=48)
    after_rows = len(store.get_history())

    assert before_rows == after_rows


def test_implementation_self_correction_retries_after_sandbox_rejection(tmp_path):
    store = _build_store(tmp_path)
    registry, dt_model_registry = _registry_with_throughput(tmp_path, store)
    llm = _FakeLLMClient(
        implementations=[
            _GeneratedComponentCode(class_name="SinrQualityModel", source_code=BROKEN_SOURCE, reasoning="attempt 1"),
            _GeneratedComponentCode(class_name="SinrQualityModel", source_code=VALID_SOURCE, reasoning="attempt 2"),
        ]
    )
    agent = _agent(tmp_path, store, registry, dt_model_registry, llm)

    result = agent.expand_scope(_example_factory(), window_hours=48)

    assert result.implementation_attempts == 2
    assert len(llm.implementation_calls) == 2
    assert "REJECTED by the sandbox" in llm.implementation_calls[1]


def test_exhausting_implementation_attempts_registers_nothing(tmp_path):
    store = _build_store(tmp_path)
    registry, dt_model_registry = _registry_with_throughput(tmp_path, store)
    llm = _FakeLLMClient(
        implementations=[
            _GeneratedComponentCode(class_name="SinrQualityModel", source_code=BROKEN_SOURCE, reasoning="1"),
            _GeneratedComponentCode(class_name="SinrQualityModel", source_code=BROKEN_SOURCE, reasoning="2"),
        ]
    )
    agent = _agent(tmp_path, store, registry, dt_model_registry, llm)

    with pytest.raises(ExpandScopeError, match="no accepted candidate"):
        agent.expand_scope(_example_factory(), window_hours=48)

    assert registry.list_versions("sinr_quality") == []


def test_llm_failure_during_design_wrapped_as_expand_scope_error(tmp_path):
    store = _build_store(tmp_path)
    registry, dt_model_registry = _registry_with_throughput(tmp_path, store)
    llm = _FakeLLMClient(raise_on_call=LLMClientError("simulated failure"))
    agent = _agent(tmp_path, store, registry, dt_model_registry, llm)

    with pytest.raises(ExpandScopeError, match="LLM component-design proposal failed"):
        agent.expand_scope(_example_factory(), window_hours=48)


def test_fidelity_after_none_without_prior_history(tmp_path):
    store = _build_store(tmp_path)
    registry, dt_model_registry = _registry_with_throughput(tmp_path, store)
    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    agent = _agent(tmp_path, store, registry, dt_model_registry, _FakeLLMClient(), fidelity_evaluator=evaluator)

    result = agent.expand_scope(_example_factory(), window_hours=48)

    assert result.fidelity_after is None
    assert result.version.fidelity_before is None


def test_fidelity_after_real_value_when_evaluator_prewarmed(tmp_path):
    store = _build_store(tmp_path)
    registry, dt_model_registry = _registry_with_throughput(tmp_path, store)
    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    history = store.get_history()
    y_true = history["sinr_db"]
    y_pred = y_true + np.random.normal(0, 1, len(y_true))
    for _ in range(SETTINGS.fidelity.min_history_for_normalization + 1):
        evaluator.evaluate("sinr_quality", y_true, y_pred, update_window=True)

    agent = _agent(tmp_path, store, registry, dt_model_registry, _FakeLLMClient(), fidelity_evaluator=evaluator)
    result = agent.expand_scope(_example_factory(), window_hours=48)

    assert result.fidelity_after is not None


def test_design_prompt_includes_existing_components_and_rag_note(tmp_path):
    store = _build_store(tmp_path)
    registry, dt_model_registry = _registry_with_throughput(tmp_path, store)
    llm = _FakeLLMClient()
    agent = _agent(tmp_path, store, registry, dt_model_registry, llm)

    agent.expand_scope(_example_factory(), window_hours=48, expand_scope_context="SINR anomaly")

    prompt = llm.design_calls[0]
    assert '"component_name": "throughput"' in prompt
    assert "SINR anomaly" in prompt
    assert "Module 18 (RAG Knowledge Base) is not built yet" in prompt
