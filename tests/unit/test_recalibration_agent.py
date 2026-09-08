"""Unit tests for RecalibrationAgent (Module 14, prompt.md §23).

Uses a small, hand-built D1-shaped history DataFrame (not the full real telemetry pipeline —
that's `tests/integration/test_recalibration_agent.py`'s job, including the concrete continuous-
operation-during-adaptation proof) so each piece of the agent's logic can be tested in isolation
and fast: error/validation cases, registry interaction, fidelity-before/after via Module 12, and
the optional/graceful-degradation LLM window reasoning. The shared window-selection/dependency-
ground-truth helpers this agent uses are tested directly in `tests/unit/test_data_selection.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adaptation.recalibration_agent import RecalibrationAgent, RecalibrationError
from src.common.config import load_settings
from src.dt_models.d1_model_store import D1Store
from src.dt_models.throughput import ThroughputModel
from src.fidelity.evaluator import FidelityEvaluator
from src.registry.model_registry import ModelRegistry
from src.telemetry.schema import CleanTelemetryRecord, RecordQuality

SETTINGS = load_settings()

REQUIRED_FEATURES = ThroughputModel.REQUIRED_FEATURES


def _clean_record(ue_id: str, cell_id: str, timestamp: datetime, **overrides) -> CleanTelemetryRecord:
    base = dict(
        timestamp=timestamp,
        ue_id=ue_id,
        cell_id=cell_id,
        throughput_mbps=float(np.random.uniform(5, 50)),
        offered_load_mbps=float(np.random.uniform(5, 60)),
        latency_ms=float(np.random.uniform(1, 20)),
        jitter_ms=float(np.random.uniform(0, 2)),
        packet_loss_pct=float(np.random.uniform(0, 5)),
        prb_utilization_pct=float(np.random.uniform(10, 90)),
        sinr_db=float(np.random.uniform(0, 30)),
        rsrp_dbm=float(np.random.uniform(-120, -60)),
        rsrq_db=float(np.random.uniform(-15, -5)),
        ue_count=int(np.random.randint(1, 10)),
        ue_speed_mps=float(np.random.uniform(0, 15)),
        source="MOCK",
        quality=RecordQuality(),
    )
    base.update(overrides)
    return CleanTelemetryRecord(**base)


def _build_store(tmp_path: Path, n_rows: int = 300, hours_back_span: float = 2.0) -> D1Store:
    store = D1Store(
        current_state_path=tmp_path / "current.parquet",
        history_path=tmp_path / "history.parquet",
        quarantine_path=tmp_path / "quarantine.parquet",
        history_retention_rows=SETTINGS.storage.history_retention_rows,
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
        component_instance=model,
        adaptation_type="bootstrap",
        parent_version_id=None,
        training_window={"n_rows": len(history)},
        evaluation_window={"n_rows": 0},
        evaluation_metrics=model.evaluate(history[list(REQUIRED_FEATURES)], history["throughput_mbps"]),
        status="production",
    )
    return registry, model


# --- RecalibrationAgent.recalibrate --------------------------------------------------------------


def test_recalibrate_raises_without_a_production_version(tmp_path):
    store = _build_store(tmp_path)
    registry = ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")
    agent = RecalibrationAgent(SETTINGS, store, registry)
    with pytest.raises(RecalibrationError, match="no production version"):
        agent.recalibrate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps")


def test_recalibrate_raises_when_window_has_too_few_rows(tmp_path):
    store = _build_store(tmp_path, n_rows=300)
    registry, _ = _bootstrap_registry(tmp_path, store)
    agent = RecalibrationAgent(SETTINGS, store, registry)
    with pytest.raises(RecalibrationError, match="need >="):
        # An absurdly short window (well under a second) leaves ~0 rows.
        agent.recalibrate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=1e-9)


def test_recalibrate_registers_candidate_with_correct_parent_and_status(tmp_path):
    store = _build_store(tmp_path)
    registry, bootstrap_model = _bootstrap_registry(tmp_path, store)
    agent = RecalibrationAgent(SETTINGS, store, registry)

    result = agent.recalibrate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48)

    assert result.version.status == "candidate"
    assert result.version.adaptation_type == "recalibrate"
    assert result.version.component == "throughput"
    current = registry.get_current_version("throughput")
    assert current.version_id != result.version.version_id  # candidate never auto-promotes
    assert current.status == "production"  # bootstrap version untouched


def test_recalibrate_produces_a_working_trained_candidate(tmp_path):
    store = _build_store(tmp_path)
    registry, _ = _bootstrap_registry(tmp_path, store)
    agent = RecalibrationAgent(SETTINGS, store, registry)

    result = agent.recalibrate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48)

    assert result.candidate_component.is_trained
    preds = result.candidate_component.predict(store.get_history()[list(REQUIRED_FEATURES)].head(5))
    assert len(preds) == 5
    assert "rmse" in result.evaluation_metrics


def test_recalibrate_never_writes_to_d1(tmp_path):
    store = _build_store(tmp_path)
    registry, _ = _bootstrap_registry(tmp_path, store)
    agent = RecalibrationAgent(SETTINGS, store, registry)

    before_rows = len(store.get_history())
    agent.recalibrate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48)
    after_rows = len(store.get_history())

    assert before_rows == after_rows  # recalibration only ever READS D1, never writes


def test_recalibrate_fidelity_is_none_without_prior_history_never_fabricated(tmp_path):
    store = _build_store(tmp_path)
    registry, _ = _bootstrap_registry(tmp_path, store)
    fresh_evaluator = FidelityEvaluator(SETTINGS.fidelity)  # no prior points -> insufficient_history
    agent = RecalibrationAgent(SETTINGS, store, registry, fidelity_evaluator=fresh_evaluator)

    result = agent.recalibrate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48)

    assert result.fidelity_before is None
    assert result.fidelity_after is None


def test_recalibrate_computes_real_fidelity_before_after_when_evaluator_prewarmed(tmp_path):
    store = _build_store(tmp_path)
    registry, bootstrap_model = _bootstrap_registry(tmp_path, store)
    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    history = store.get_history()
    y_true = history["throughput_mbps"]
    y_pred = bootstrap_model.predict(history[list(REQUIRED_FEATURES)])
    # Pre-warm the rolling window past min_history_for_normalization with REAL evaluate() calls,
    # mirroring a production evaluator that has already been running for a while.
    for _ in range(SETTINGS.fidelity.min_history_for_normalization + 1):
        evaluator.evaluate("throughput", y_true, y_pred, update_window=True)

    agent = RecalibrationAgent(SETTINGS, store, registry, fidelity_evaluator=evaluator)
    result = agent.recalibrate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48)

    assert result.fidelity_before is not None
    assert result.fidelity_after is not None


def test_recalibrate_dependency_component_uses_ground_truth_mapping(tmp_path):
    from src.dt_models.latency import LatencyModel

    store = _build_store(tmp_path, n_rows=300)
    registry = ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")
    history = store.get_history()

    throughput_model = ThroughputModel.from_settings(SETTINGS)
    throughput_model.train(history[list(REQUIRED_FEATURES)], history["throughput_mbps"])
    registry.register_version(
        component_instance=throughput_model, adaptation_type="bootstrap", parent_version_id=None,
        training_window={"n_rows": len(history)}, evaluation_window={"n_rows": 0},
        evaluation_metrics={}, status="production",
    )

    latency_features = list(LatencyModel.REQUIRED_FEATURES) + ["throughput_mbps_pred"]
    latency_train_df = history.assign(throughput_mbps_pred=history["throughput_mbps"])
    latency_model = LatencyModel.from_settings(SETTINGS)
    latency_model.train(latency_train_df[latency_features], latency_train_df["latency_ms"])
    registry.register_version(
        component_instance=latency_model, adaptation_type="bootstrap", parent_version_id=None,
        training_window={"n_rows": len(history)}, evaluation_window={"n_rows": 0},
        evaluation_metrics={}, status="production",
    )

    agent = RecalibrationAgent(SETTINGS, store, registry)
    result = agent.recalibrate(
        lambda: LatencyModel.from_settings(SETTINGS),
        "latency_ms",
        window_hours=48,
        dependency_output_fields={"throughput": "throughput_mbps_pred"},
    )

    assert result.version.component == "latency"
    assert result.candidate_component.is_trained


# --- optional LLM window reasoning ---------------------------------------------------------------


class _FakeLLMClient:
    def __init__(self, suggestion=None):
        self._suggestion = suggestion
        self.calls = 0

    def complete_structured_safe(self, prompt, schema):
        self.calls += 1
        return self._suggestion


def test_llm_window_reasoning_disabled_by_default_never_calls_client(tmp_path):
    store = _build_store(tmp_path)
    registry, _ = _bootstrap_registry(tmp_path, store)
    fake_client = _FakeLLMClient()
    agent = RecalibrationAgent(SETTINGS, store, registry, llm_client=fake_client)

    agent.recalibrate(lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48)

    assert fake_client.calls == 0


def test_llm_window_reasoning_clamped_to_bounded_range(tmp_path):
    from src.adaptation.recalibration_agent import _WindowSuggestion

    store = _build_store(tmp_path)
    registry, _ = _bootstrap_registry(tmp_path, store)
    default_hours = SETTINGS.adaptation.recalibration.training_window_hours
    # An absurd suggestion (1000x default) must be clamped, never used verbatim.
    fake_client = _FakeLLMClient(_WindowSuggestion(window_hours=default_hours * 1000, reasoning="test"))
    agent = RecalibrationAgent(SETTINGS, store, registry, llm_client=fake_client)

    result = agent.recalibrate(
        lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", use_llm_window_reasoning=True
    )

    assert fake_client.calls == 1
    assert result.training_window["window_hours"] <= default_hours * 4.0
    assert result.version.llm_metadata is not None


def test_llm_window_reasoning_falls_back_to_default_on_none(tmp_path):
    store = _build_store(tmp_path)
    registry, _ = _bootstrap_registry(tmp_path, store)
    fake_client = _FakeLLMClient(suggestion=None)  # simulates complete_structured_safe's graceful-degradation path
    agent = RecalibrationAgent(SETTINGS, store, registry, llm_client=fake_client)

    result = agent.recalibrate(
        lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", use_llm_window_reasoning=True
    )

    assert result.training_window["window_hours"] == SETTINGS.adaptation.recalibration.training_window_hours
