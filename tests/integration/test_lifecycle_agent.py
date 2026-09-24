"""Integration test: Module 19 (Lifecycle Management Agent) — the REQUIRED "run one full cycle"
proof: a genuine drift event (Module 11), normalized into the canonical `AdaptationTrigger`
(prompt.md §16a), drives a genuine Decision & Root-Cause Analysis Agent call (Module 13), which is
dispatched to whichever real Module 14/15/16 agent it actually selects, verified by a real Module
17 `VerificationAgent`, and finally recorded by this module — the resulting `LifecycleRecord` is
then inspected field-by-field for completeness against prompt.md §37's exact list.

No real GOOGLE_API_KEY is configured in this environment — the Decision & Root-Cause Analysis
Agent's own call, and Regeneration/Expand-Scope's LLM calls (only reached if the decision agent
selects that branch), are all driven by one fake client returning a genuine, structurally-real
response for whichever schema it's asked for (exactly Modules 15/16's own established testing
convention; recalibration needs no LLM at all). Everything else — real Module 11 drift
validation, real Module 12 fidelity recomputation, a real sandbox subprocess (if reached), and the
real lifecycle record/report — is genuine, unmocked code.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from src.adaptation.data_selection import select_recent_window, time_split
from src.adaptation.decision_agent import DecisionAgent, DecisionOutput
from src.adaptation.decision_context import PreviousOutcome, build_decision_context
from src.adaptation.expand_scope_agent import ExpandScopeAgent, _ProposedComponentDesign
from src.adaptation.lifecycle_agent import LifecycleAgent
from src.adaptation.recalibration_agent import RecalibrationAgent
from src.adaptation.regeneration_agent import RegenerationAgent
from src.adaptation.verification_agent import VerificationAgent
from src.common.config import load_settings
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.throughput import ThroughputModel
from src.drift.drift_detector import DriftDetectorInterface
from src.fidelity.evaluator import FidelityEvaluator
from src.fidelity.trigger import trigger_from_drift_event
from src.registry.model_registry import ModelRegistry
from src.sandbox.executor import SandboxExecutor

SETTINGS = load_settings()

_REGEN_SOURCE = """
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

_EXPAND_DESIGN = _ProposedComponentDesign(
    component_name="sinr_quality",
    target_column="sinr_db",
    dependencies=(),
    required_features=("offered_load_mbps", "prb_utilization_pct", "ue_speed_mps"),
    purpose="Predict SINR from load and mobility for proactive handover decisions.",
    feature_extraction_notes="none needed",
)

_EXPAND_SOURCE = """
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


class _FakeLLM:
    """Serves BOTH the Decision & Root-Cause Analysis Agent's call AND whichever of Modules
    15/16's calls its selected strategy needs — exactly like one real `GoogleClient` instance
    serves every caller in production, discriminating purely on the `schema` argument it's asked
    for, never on which "role" is calling. Genuinely returns a strategy (`"regenerate"`, chosen
    here so this test exercises the real sandbox path — matching this test's own historical
    observed behavior), never a hardcoded bypass of the real `DecisionAgent`/agent logic."""

    def __init__(self, strategy: str = "regenerate") -> None:
        self._strategy = strategy
        self.calls: list[tuple[str, type]] = []

    def complete_structured(self, prompt, schema, **kwargs):
        self.calls.append((prompt, schema))
        if schema is DecisionOutput:
            return DecisionOutput(
                strategy=self._strategy,
                root_cause_analysis="Structural drift in throughput predictions likely requires a rebuilt pipeline.",
                confidence=0.82,
                rationale="Recalibration alone would not address a structural mismatch of this severity.",
                knowledge_refs=[],
            )
        if schema is _ProposedComponentDesign:
            return _EXPAND_DESIGN
        if self._strategy == "expand_scope":
            return schema(class_name="SinrQualityModel", source_code=_EXPAND_SOURCE, reasoning="linear model")
        return schema(class_name="RebuiltThroughput", source_code=_REGEN_SOURCE, reasoning="rebuilt via a linear model")


class _FakeD1Store:
    """The three adaptation agents only ever call `get_history()` — a real `D1Store` isn't
    needed here since `bootstrap_history` is already a plain, real snapshot."""

    def __init__(self, history):
        self._history = history

    def get_history(self, ue_id=None, cell_id=None):
        return self._history.copy(deep=True)


def _bootstrap_weak_production(bootstrap_history, registry: ModelRegistry):
    """A deliberately weak, early-slice bootstrap production model — real headroom for whichever
    adaptation strategy is selected to genuinely improve on, mirroring the same setup already used
    by `tests/integration/test_verification_agent.py`."""
    features = list(ThroughputModel.REQUIRED_FEATURES)
    ordered = bootstrap_history.sort_values("timestamp", kind="stable").reset_index(drop=True)
    thin_slice = ordered.iloc[:40]
    model = ThroughputModel.from_settings(SETTINGS)
    model.train(thin_slice[features], thin_slice["throughput_mbps"])
    registry.register_version(
        component_instance=model,
        adaptation_type="bootstrap",
        parent_version_id=None,
        training_window={"n_rows": len(thin_slice)},
        evaluation_window={"n_rows": 0},
        evaluation_metrics=model.evaluate(thin_slice[features], thin_slice["throughput_mbps"]),
        status="production",
    )


def test_full_adaptation_lifecycle_cycle_produces_a_complete_record(bootstrap_history, tmp_path):
    registry = ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")
    _bootstrap_weak_production(bootstrap_history, registry)

    # 1. drift (Module 11) — a REAL raw event through the REAL validation/normalization pipeline,
    # normalized into the canonical AdaptationTrigger (Module 12, prompt.md §16a).
    detector = DriftDetectorInterface.from_settings(SETTINGS)
    raw_event = {
        "component": "throughput",
        "severity": 0.65,
        "timestamp": datetime.now(UTC).isoformat(),
        "metadata": {"reason": "integration-test synthetic drift"},
        "source": "MOCK",
    }
    drift_event, quarantined = detector.process_event(raw_event)
    assert drift_event is not None and quarantined is None
    trigger = trigger_from_drift_event(drift_event)

    # 2. Decision & Root-Cause Analysis Agent (Module 13) — a REAL DecisionContext built from real
    # sources, decided by a REAL DecisionAgent.decide() call (LLM transport faked, per this file's
    # own docstring — no real GOOGLE_API_KEY is configured in this environment).
    fake_llm = _FakeLLM(strategy="regenerate")
    decision_context = build_decision_context(
        trigger,
        component_fidelity={},
        unified_fidelity_score=None,
        previous_outcome=PreviousOutcome(),
        history=bootstrap_history,
    )
    decision_agent = DecisionAgent(fake_llm, SETTINGS)
    decision = decision_agent.decide(decision_context)
    action_name = decision.strategy
    assert action_name in ("recalibrate", "regenerate", "expand_scope")

    # 3. whichever real Module 14/15/16 agent the decision agent actually selected.
    drift_context = f"drift on 'throughput' at severity {drift_event.severity:.3f}"
    if action_name == "recalibrate":
        agent = RecalibrationAgent(SETTINGS, _FakeD1Store(bootstrap_history), registry)
        agent_result = agent.recalibrate(
            lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48, held_out_fraction=0.2
        )
    elif action_name == "regenerate":
        agent = RegenerationAgent(SETTINGS, _FakeD1Store(bootstrap_history), registry, fake_llm, SandboxExecutor.from_settings(SETTINGS))
        agent_result = agent.regenerate(
            lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps",
            window_hours=48, held_out_fraction=0.2, drift_context=drift_context,
        )
    else:
        dt_model_registry = DTModelRegistry()
        dt_model_registry.register(ThroughputModel.from_settings(SETTINGS))
        agent = ExpandScopeAgent(
            SETTINGS, _FakeD1Store(bootstrap_history), registry, dt_model_registry, fake_llm, SandboxExecutor.from_settings(SETTINGS)
        )
        agent_result = agent.expand_scope(
            lambda: ThroughputModel.from_settings(SETTINGS),
            window_hours=48, held_out_fraction=0.2, expand_scope_context=drift_context,
        )

    candidate_version = agent_result.version
    assert candidate_version.status == "candidate"

    # 4. verification (Module 17) — the SAME held-out protocol every branch is evaluated on.
    target_column = "throughput_mbps" if action_name != "expand_scope" else _EXPAND_DESIGN.target_column
    window_df = select_recent_window(bootstrap_history, 48)
    _, held_out_df = time_split(window_df, 0.2)
    eval_target = held_out_df[target_column]

    if action_name == "recalibrate":
        candidate_preds = agent_result.candidate_component.predict(held_out_df)
        production_factory = lambda: ThroughputModel.from_settings(SETTINGS)  # noqa: E731
    elif action_name == "regenerate":
        candidate_preds = agent_result.sandbox_result.eval_predictions
        production_factory = lambda: ThroughputModel.from_settings(SETTINGS)  # noqa: E731
    else:
        candidate_preds = agent_result.sandbox_result.eval_predictions
        production_factory = None

    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    rng = np.random.default_rng(999)
    base_vals = eval_target.to_numpy()
    base = base_vals[:40] if len(base_vals) >= 40 else np.resize(base_vals, 40)
    for noise_std in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0):
        evaluator.evaluate(candidate_version.component, base, base + rng.normal(0.0, noise_std, len(base)), update_window=True)

    verification_agent = VerificationAgent(SETTINGS, registry, evaluator)
    verification_result = verification_agent.verify(
        candidate_version, held_out_df, eval_target, candidate_preds,
        production_component_factory=production_factory, drift_context=drift_context,
    )

    # 5. Module 19 — record the whole event, then inspect it for completeness.
    lifecycle_agent = LifecycleAgent(
        records_path=tmp_path / "lifecycle" / "records.jsonl", reports_dir=tmp_path / "lifecycle" / "reports"
    )
    record = lifecycle_agent.record_adaptation_event(
        trigger=trigger,
        decision=decision,
        agent_result=agent_result,
        verification_result=verification_result,
    )

    # --- inspect the resulting record for completeness against prompt.md §37's exact field list ---
    assert record.event_id
    assert record.timestamp is not None
    assert record.trigger["component"] == "throughput"
    assert record.affected_component == "throughput"
    assert record.trigger_severity == pytest.approx(drift_event.severity)
    assert record.trigger_type == "external_drift"
    assert record.decision_strategy == action_name
    assert record.root_cause_analysis
    assert 0.0 <= record.decision_confidence <= 1.0
    assert record.decision_rationale
    assert record.agent_action and record.agent_action["adaptation_type"] == action_name
    assert record.production_version_before == candidate_version.parent_version_id
    assert record.candidate_version == candidate_version.version_id
    # fidelity_before/after: None is a legitimate value here (insufficient-history/no-baseline
    # are both real, documented outcomes, never asserted away) — just confirm the field is the
    # right type either way.
    assert record.fidelity_before is None or isinstance(record.fidelity_before, float)
    assert record.fidelity_after is None or isinstance(record.fidelity_after, float)
    assert record.verification_result in ("ACCEPT", "REJECT")
    assert record.verification_explanation
    assert record.training_window
    assert record.evaluation_window
    assert record.model_metadata["component"] == candidate_version.component
    assert record.model_metadata["adaptation_type"] == action_name
    assert record.final_status in ("promoted", "rejected")
    assert (record.final_status == "promoted") == (verification_result.decision == "ACCEPT")

    # 6. the automatically-generated maintenance report (prompt.md §38).
    report = lifecycle_agent.generate_maintenance_report(record)
    assert report.path.exists()
    assert candidate_version.component in report.text
    assert action_name in report.text
    assert verification_result.decision in report.text

    # And the whole event is durably retrievable afterward, from a fresh instance.
    fresh_agent = LifecycleAgent(records_path=tmp_path / "lifecycle" / "records.jsonl", reports_dir=tmp_path / "lifecycle" / "reports")
    assert fresh_agent.get_record(record.event_id) == record
