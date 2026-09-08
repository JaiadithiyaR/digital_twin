"""Unit tests for RegenerationAgent (Module 15, prompt.md §24-26).

No real ANTHROPIC_API_KEY is configured in this environment (same situation as
`tests/unit/test_anthropic_client.py`) — the LLM client's `complete_structured` is therefore
replaced with a fake that returns hand-written source strings standing in for what a real model
would generate, exactly the way `test_anthropic_client.py` mocks the transport layer rather than
faking "a real call happened." Everything downstream of that one substitution — the sandbox
execution, the conformance/train/evaluate pipeline, the registry writes, the fidelity comparison
— is REAL, unmocked code, run for real against a small hand-built D1-shaped history.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adaptation.regeneration_agent import RegenerationAgent, RegenerationError, _GeneratedComponentCode
from src.common.config import load_settings
from src.dt_models.d1_model_store import D1Store
from src.dt_models.throughput import ThroughputModel
from src.fidelity.evaluator import FidelityEvaluator
from src.llm.anthropic_client import LLMClientError
from src.registry.model_registry import ModelRegistry
from src.sandbox.executor import SandboxExecutor
from src.telemetry.schema import CleanTelemetryRecord, RecordQuality

SETTINGS = load_settings()
REQUIRED_FEATURES = ThroughputModel.REQUIRED_FEATURES

VALID_SOURCE = """
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from src.dt_models.base import DTComponent

class RebuiltThroughput(DTComponent):
    COMPONENT_NAME = "throughput"
    DEPENDENCIES = ()
    REQUIRED_FEATURES = ("offered_load_mbps", "prb_utilization_pct", "sinr_db")
    OUTPUT_FIELD = "throughput_mbps_pred"

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
class RebuiltThroughput(DTComponent):
    COMPONENT_NAME = "throughput"
    OUTPUT_FIELD = "throughput_mbps_pred"
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
    """Stands in for AnthropicClient — only `complete_structured` is exercised by this agent."""

    def __init__(self, responses=None, source_map=None, raise_on_call: Exception | None = None):
        self._responses = list(responses) if responses is not None else None
        self._source_map = source_map  # {attempt_number: source_code}, all use class RebuiltThroughput
        self._raise_on_call = raise_on_call
        self.calls: list[str] = []

    def complete_structured(self, prompt, schema, **kwargs):
        self.calls.append(prompt)
        if self._raise_on_call is not None:
            raise self._raise_on_call
        if self._responses is not None:
            return self._responses[len(self.calls) - 1]
        source = self._source_map[len(self.calls)]
        return schema(class_name="RebuiltThroughput", source_code=source, reasoning=f"attempt {len(self.calls)}")


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


def _bootstrap_registry(tmp_path: Path, store: D1Store) -> tuple[ModelRegistry, ThroughputModel]:
    registry = ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")
    history = store.get_history()
    model = ThroughputModel.from_settings(SETTINGS)
    model.train(history[list(REQUIRED_FEATURES)], history["throughput_mbps"])
    registry.register_version(
        component_instance=model, adaptation_type="bootstrap", parent_version_id=None,
        training_window={"n_rows": len(history)}, evaluation_window={"n_rows": 0},
        evaluation_metrics=model.evaluate(history[list(REQUIRED_FEATURES)], history["throughput_mbps"]),
        status="production",
    )
    return registry, model


def _agent(tmp_path, store, registry, llm_client, fidelity_evaluator=None) -> RegenerationAgent:
    sandbox = SandboxExecutor(
        workspace_root=tmp_path / "sandbox",
        timeout_seconds=SETTINGS.sandbox.timeout_seconds,
        max_output_bytes=SETTINGS.sandbox.max_output_bytes,
    )
    return RegenerationAgent(SETTINGS, store, registry, llm_client, sandbox, fidelity_evaluator=fidelity_evaluator)


def test_regenerate_raises_without_a_production_version(tmp_path):
    store = _build_store(tmp_path)
    registry = ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")
    agent = _agent(tmp_path, store, registry, _FakeLLMClient(source_map={1: VALID_SOURCE}))
    with pytest.raises(RegenerationError, match="no production version"):
        agent.regenerate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps")


def test_regenerate_raises_when_window_has_too_few_rows(tmp_path):
    store = _build_store(tmp_path)
    registry, _ = _bootstrap_registry(tmp_path, store)
    agent = _agent(tmp_path, store, registry, _FakeLLMClient(source_map={1: VALID_SOURCE}))
    with pytest.raises(RegenerationError, match="need >="):
        agent.regenerate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=1e-9)


def test_successful_regeneration_registers_a_candidate(tmp_path):
    store = _build_store(tmp_path)
    registry, _ = _bootstrap_registry(tmp_path, store)
    llm = _FakeLLMClient(source_map={1: VALID_SOURCE})
    agent = _agent(tmp_path, store, registry, llm)

    result = agent.regenerate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48)

    assert result.attempts == 1
    assert result.version.status == "candidate"
    assert result.version.adaptation_type == "regenerate"
    assert result.version.model_class == "RebuiltThroughput"
    assert result.version.parent_version_id == "throughput-v1"
    assert result.version.feature_schema == ("offered_load_mbps", "prb_utilization_pct", "sinr_db")
    assert result.version.source_path is not None
    source_file = (tmp_path / "models" / result.version.source_path)
    assert source_file.exists()
    assert source_file.read_text() == VALID_SOURCE
    assert result.version.llm_metadata["attempts"] == 1


def test_production_version_is_never_touched_by_regeneration(tmp_path):
    store = _build_store(tmp_path)
    registry, _ = _bootstrap_registry(tmp_path, store)
    agent = _agent(tmp_path, store, registry, _FakeLLMClient(source_map={1: VALID_SOURCE}))

    agent.regenerate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48)

    current = registry.get_current_version("throughput")
    assert current.version_id == "throughput-v1"
    assert current.adaptation_type == "bootstrap"


def test_regenerate_never_writes_to_d1(tmp_path):
    store = _build_store(tmp_path)
    registry, _ = _bootstrap_registry(tmp_path, store)
    agent = _agent(tmp_path, store, registry, _FakeLLMClient(source_map={1: VALID_SOURCE}))

    before_rows = len(store.get_history())
    agent.regenerate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48)
    after_rows = len(store.get_history())

    assert before_rows == after_rows


def test_self_correction_retries_after_sandbox_rejection(tmp_path):
    store = _build_store(tmp_path)
    registry, _ = _bootstrap_registry(tmp_path, store)
    llm = _FakeLLMClient(source_map={1: BROKEN_SOURCE, 2: VALID_SOURCE})
    agent = _agent(tmp_path, store, registry, llm)

    result = agent.regenerate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48)

    assert result.attempts == 2
    assert len(llm.calls) == 2
    assert "REJECTED by the sandbox" in llm.calls[1]  # second prompt includes rejection feedback
    assert result.version.status == "candidate"


def test_exhausting_max_attempts_raises_and_registers_nothing(tmp_path):
    store = _build_store(tmp_path)
    registry, _ = _bootstrap_registry(tmp_path, store)
    llm = _FakeLLMClient(source_map={1: BROKEN_SOURCE, 2: BROKEN_SOURCE})
    agent = _agent(tmp_path, store, registry, llm)

    with pytest.raises(RegenerationError, match="no accepted candidate"):
        agent.regenerate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48)

    assert len(llm.calls) == SETTINGS.adaptation.regeneration.max_llm_iterations
    assert len(registry.list_versions("throughput")) == 1  # only the bootstrap version, no candidate


def test_llm_transport_failure_is_wrapped_as_regeneration_error(tmp_path):
    store = _build_store(tmp_path)
    registry, _ = _bootstrap_registry(tmp_path, store)
    llm = _FakeLLMClient(raise_on_call=LLMClientError("simulated transport failure"))
    agent = _agent(tmp_path, store, registry, llm)

    with pytest.raises(RegenerationError, match="LLM code generation failed"):
        agent.regenerate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48)

    assert len(registry.list_versions("throughput")) == 1


def test_fidelity_before_after_none_without_prior_history(tmp_path):
    store = _build_store(tmp_path)
    registry, _ = _bootstrap_registry(tmp_path, store)
    evaluator = FidelityEvaluator(SETTINGS.fidelity)  # fresh, no prior points
    agent = _agent(tmp_path, store, registry, _FakeLLMClient(source_map={1: VALID_SOURCE}), fidelity_evaluator=evaluator)

    result = agent.regenerate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48)

    assert result.fidelity_before is None
    assert result.fidelity_after is None


def test_fidelity_before_after_real_values_when_evaluator_prewarmed(tmp_path):
    store = _build_store(tmp_path)
    registry, bootstrap_model = _bootstrap_registry(tmp_path, store)
    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    history = store.get_history()
    y_true = history["throughput_mbps"]
    y_pred = bootstrap_model.predict(history[list(REQUIRED_FEATURES)])
    for _ in range(SETTINGS.fidelity.min_history_for_normalization + 1):
        evaluator.evaluate("throughput", y_true, y_pred, update_window=True)

    agent = _agent(tmp_path, store, registry, _FakeLLMClient(source_map={1: VALID_SOURCE}), fidelity_evaluator=evaluator)
    result = agent.regenerate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48)

    assert result.fidelity_before is not None
    assert result.fidelity_after is not None


def test_prompt_includes_current_source_and_drift_context(tmp_path):
    store = _build_store(tmp_path)
    registry, _ = _bootstrap_registry(tmp_path, store)
    llm = _FakeLLMClient(source_map={1: VALID_SOURCE})
    agent = _agent(tmp_path, store, registry, llm)

    agent.regenerate(
        lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48,
        drift_context="PPO selected regenerate due to severity 0.87",
    )

    prompt = llm.calls[0]
    assert "class ThroughputModel" in prompt  # current production source code included
    assert "severity 0.87" in prompt
    assert "Module 18 (RAG Knowledge Base) is not built yet" in prompt
