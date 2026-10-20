"""Unit tests for the per-component adaptation policy as wired into `ContinuousOrchestrator`
(`src/main.py`): independent per-component triggers from the fidelity cycle, missing-data
handling, component-level logging, post-adaptation re-evaluation, dependency revalidation, and
failure reporting.

The fidelity FORMULA is covered by `test_fidelity_evaluator.py`; here a scripted evaluator
returns chosen `FidelityScore_c` values so the orchestration logic can be tested exactly. Every
other piece is real: `DTModelRegistry`/`DTOrchestrator` running the dummy dependency chain
(doubler -> adder -> summer), `ComponentAdaptationPolicy`, the run logs, and the registry.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.adaptation.decision_agent import DecisionAgent
from src.adaptation.recalibration_agent import RecalibrationError
from src.common.config import ComponentPolicyOverride, Secrets, load_settings
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator
from src.fidelity.evaluator import FidelityEvaluator, FidelityResult
from src.fidelity.trigger import AdaptationTrigger
from src.main import _OUTPUT_FIELDS, _TARGET_COLUMNS, ContinuousOrchestrator
from src.registry.model_registry import ModelRegistry
from tests.dummy_dt_components import DummyAdder, DummyDoubler, DummySummer

SETTINGS = load_settings()
NO_KEY_SECRETS = Secrets(google_api_key=None)
DUMMY_TARGETS = {"dummy_doubler": "a", "dummy_adder": "b", "dummy_summer": "c"}


class _ScriptedEvaluator(FidelityEvaluator):
    """Real evaluator (incl. the real unified-score report), with `evaluate()` returning a chosen
    FidelityScore_c per component and recording each call's `update_window` flag."""

    def __init__(self, config, scores: dict[str, float | None]) -> None:
        super().__init__(config)
        self.scores = dict(scores)
        self.calls: list[tuple[str, bool]] = []

    def history_length(self, component: str) -> int:
        return 10_000  # never backfill in these tests

    def evaluate(self, component, y_true, y_pred, update_window=True):
        self.calls.append((component, update_window))
        score = self.scores.get(component)
        return FidelityResult(
            component=component, timestamp=datetime.now(UTC),
            status="ok" if score is not None else "insufficient_history", n_samples=len(y_true),
            raw_metrics={}, squared_metrics={}, normalized_metrics=None, fidelity_score=score,
        )


class _FakeD1:
    def __init__(self, history: pd.DataFrame) -> None:
        self._history = history

    def get_history(self, **_: object) -> pd.DataFrame:
        return self._history.copy()


def _history(n: int = 250, nan_target: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    df = pd.DataFrame({
        "ts_1s": pd.date_range("2026-01-01", periods=n, freq="s", tz="UTC"),
        "x": x, "y": y, "a": 2 * x, "b": y + 2 * x, "c": 2 * x + y + 2 * x,
    })
    if nan_target is not None:
        df[nan_target] = np.nan
    return df


def _orchestrator(tmp_path, scores, *, min_consecutive=1, overrides=None, history=None) -> ContinuousOrchestrator:
    fidelity = SETTINGS.fidelity.model_copy(update={
        "trigger_threshold": 0.5, "trigger_min_consecutive_evaluations": min_consecutive,
        "component_policies": overrides or {},
    })
    logging_cfg = SETTINGS.logging.model_copy(update={
        "component_fidelity_log": str(tmp_path / "component_fidelity.csv"),
        "adaptation_decisions_log": str(tmp_path / "adaptation_decisions.jsonl"),
    })
    settings = SETTINGS.model_copy(update={"fidelity": fidelity, "logging": logging_cfg})
    orch = ContinuousOrchestrator(settings, NO_KEY_SECRETS)
    registry = DTModelRegistry()
    for component in (DummyDoubler(), DummyAdder(), DummySummer()):
        component.train(pd.DataFrame(), pd.Series(dtype=float))
        registry.register(component)
    orch.dt_model_registry = registry
    orch.dt_orchestrator = DTOrchestrator(registry)
    orch._target_columns = dict(DUMMY_TARGETS)
    orch._output_fields = {"dummy_doubler": "a_pred", "dummy_adder": "b_pred", "dummy_summer": "c_pred"}
    orch._dynamic_component_classes = {}
    orch.fidelity_evaluator = _ScriptedEvaluator(settings.fidelity, scores)
    orch.d1_store = _FakeD1(history if history is not None else _history())
    orch.model_registry = ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")
    return orch


def _pending(orch: ContinuousOrchestrator) -> list[str]:
    return list(orch._pending_order)


# --- independent decisions from the fidelity cycle ----------------------------------------------


def test_all_components_healthy_enqueues_nothing(tmp_path):
    orch = _orchestrator(tmp_path, {"dummy_doubler": 0.9, "dummy_adder": 0.8, "dummy_summer": 0.95})
    orch.run_prediction_and_fidelity_cycle()
    assert _pending(orch) == []
    assert {c: s.state for c, s in orch._component_status.items()} == {
        "dummy_doubler": "ok", "dummy_adder": "ok", "dummy_summer": "ok"
    }


def test_only_the_low_component_is_enqueued(tmp_path):
    orch = _orchestrator(tmp_path, {"dummy_doubler": 0.9, "dummy_adder": 0.2, "dummy_summer": 0.95})
    orch.run_prediction_and_fidelity_cycle()
    assert _pending(orch) == ["dummy_adder"]
    trigger = orch._pop_next_trigger()
    assert trigger.trigger_type == "fidelity_degradation" and trigger.metadata["fidelity_score"] == 0.2


def test_multiple_low_components_are_each_enqueued_independently(tmp_path):
    orch = _orchestrator(tmp_path, {"dummy_doubler": 0.1, "dummy_adder": 0.9, "dummy_summer": 0.3})
    orch.run_prediction_and_fidelity_cycle()
    assert sorted(_pending(orch)) == ["dummy_doubler", "dummy_summer"]


def test_component_with_no_scorable_rows_is_unavailable_and_never_triggers(tmp_path):
    orch = _orchestrator(tmp_path, {"dummy_doubler": 0.9, "dummy_adder": 0.9, "dummy_summer": 0.01},
                         history=_history(nan_target="c"))
    orch.run_prediction_and_fidelity_cycle()
    assert orch._component_status["dummy_summer"].state == "unavailable"
    assert orch._latest_fidelity["dummy_summer"] is None  # never a stale or fabricated value
    assert _pending(orch) == []


def test_per_component_threshold_override_is_honoured(tmp_path):
    overrides = {"dummy_doubler": ComponentPolicyOverride(threshold=0.95)}
    orch = _orchestrator(tmp_path, {"dummy_doubler": 0.9, "dummy_adder": 0.9, "dummy_summer": 0.9}, overrides=overrides)
    orch.run_prediction_and_fidelity_cycle()
    assert _pending(orch) == ["dummy_doubler"]  # 0.9 is below ITS 0.95, fine for everyone else's 0.5


def test_debounce_needs_consecutive_fresh_cycles(tmp_path):
    orch = _orchestrator(tmp_path, {"dummy_doubler": 0.2, "dummy_adder": 0.9, "dummy_summer": 0.9}, min_consecutive=2)
    orch.run_prediction_and_fidelity_cycle()
    assert _pending(orch) == []
    orch._last_evaluated_max_ts = None  # pretend new telemetry arrived
    orch.run_prediction_and_fidelity_cycle()
    assert _pending(orch) == ["dummy_doubler"]


def test_component_fidelity_log_gets_one_row_per_component_per_cycle(tmp_path):
    orch = _orchestrator(tmp_path, {"dummy_doubler": 0.9, "dummy_adder": 0.2, "dummy_summer": None})
    orch.run_prediction_and_fidelity_cycle()
    with (tmp_path / "component_fidelity.csv").open() as f:
        rows = {r["component"]: r for r in csv.DictReader(f)}
    assert set(rows) == set(DUMMY_TARGETS)
    assert rows["dummy_adder"]["state"] == "adapt" and float(rows["dummy_adder"]["threshold"]) == 0.5
    assert rows["dummy_summer"]["state"] == "insufficient_history" and rows["dummy_summer"]["fidelity_score"] == ""
    assert rows["dummy_doubler"]["cycle"] == "1"


# --- dependency handling ---------------------------------------------------------------------------


def test_downstream_components_are_derived_from_the_dependency_graph(tmp_path):
    orch = _orchestrator(tmp_path, {})
    assert orch._downstream_components("dummy_doubler") == ["dummy_adder", "dummy_summer"]
    assert orch._downstream_components("dummy_adder") == ["dummy_summer"]
    assert orch._downstream_components("dummy_summer") == []


def test_post_adaptation_check_reports_remaining_gap_and_revalidates_dependents(tmp_path):
    orch = _orchestrator(tmp_path, {"dummy_doubler": 0.3, "dummy_adder": 0.9, "dummy_summer": 0.2})
    result = orch._post_adaptation_validation("dummy_doubler", revalidate_dependents=True)

    assert result["fidelity_score"] == 0.3
    assert result["still_below_threshold"] is True and result["remaining_gap"] == pytest.approx(0.2)
    deps = {d["component"]: d for d in result["dependents"]}
    assert set(deps) == {"dummy_adder", "dummy_summer"}
    assert deps["dummy_adder"]["status"].startswith("ok")
    assert deps["dummy_summer"]["status"].startswith("BELOW")  # flagged, not auto-adapted
    # re-measurement is read-only against the shared normalization reference
    assert all(update_window is False for _, update_window in orch.fidelity_evaluator.calls)
    assert orch._pending_order == []  # nothing auto-enqueued for dependents


def test_dependents_are_not_revalidated_when_production_did_not_change(tmp_path):
    orch = _orchestrator(tmp_path, {"dummy_doubler": 0.7, "dummy_adder": 0.9, "dummy_summer": 0.9})
    result = orch._post_adaptation_validation("dummy_doubler", revalidate_dependents=False)
    assert result["still_below_threshold"] is False and result["remaining_gap"] == 0.0
    assert result["dependents"] == []


# --- adaptation outcomes are always recorded, failures included -------------------------------


def _decision_log(tmp_path) -> list[dict]:
    path = tmp_path / "adaptation_decisions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _real_component_orchestrator(tmp_path) -> ContinuousOrchestrator:
    orch = _orchestrator(tmp_path, {})
    orch._target_columns = dict(_TARGET_COLUMNS)
    orch._output_fields = dict(_OUTPUT_FIELDS)
    orch.decision_agent = DecisionAgent.from_settings(orch._settings, None)  # no key -> logged fallback
    orch.llm_client = None
    orch.rag_kb = None
    return orch


def test_failed_adaptation_is_reported_as_a_failure_never_a_success(tmp_path, monkeypatch):
    class _FailingAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def recalibrate(self, *args, **kwargs):
            raise RecalibrationError("only 12 rows available, need >= 200")

    monkeypatch.setattr("src.main.RecalibrationAgent", _FailingAgent)
    orch = _real_component_orchestrator(tmp_path)
    trigger = AdaptationTrigger(
        component="dl_bler", severity=0.6, timestamp=datetime.now(UTC), trigger_type="fidelity_degradation",
        metadata={"fidelity_score": 0.2, "reason": "test"},
    )
    orch._handle_trigger(trigger)

    entries = _decision_log(tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["outcome"] == "agent_failed" and "only 12 rows" in entry["error"]
    assert entry["adaptation_attempted"] is True
    assert entry["strategy"] == "recalibrate" and entry["decision_source"] == "fallback"
    assert entry["fidelity_at_trigger"] == 0.2 and entry["threshold"] == 0.5
    assert entry["candidate_version"] is None and entry["version_after"] == entry["version_before"]
    # "not enough rows" can't be fixed by retrying now — exactly one attempt, then released.
    assert entry["attempt"] == 1 and entry["max_attempts"] == orch._settings.adaptation.max_adaptation_attempts
    assert orch.active_adaptations == []


def test_attempt_interrupted_mid_agent_is_still_logged_and_the_interrupt_propagates(tmp_path, monkeypatch):
    class _InterruptedAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def recalibrate(self, *args, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr("src.main.RecalibrationAgent", _InterruptedAgent)
    orch = _real_component_orchestrator(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        orch._handle_trigger(AdaptationTrigger(
            component="dl_bler", severity=0.6, timestamp=datetime.now(UTC), trigger_type="external_drift",
        ))
    entries = _decision_log(tmp_path)
    assert [(e["outcome"], e["adaptation_attempted"]) for e in entries] == [("interrupted", True)]
    assert orch.active_adaptations == []


def test_unknown_component_trigger_is_logged_as_skipped(tmp_path):
    orch = _real_component_orchestrator(tmp_path)
    orch._handle_trigger(AdaptationTrigger(
        component="no_such_component", severity=0.5, timestamp=datetime.now(UTC), trigger_type="external_drift",
    ))
    assert [e["outcome"] for e in _decision_log(tmp_path)] == ["skipped_unknown_component"]
