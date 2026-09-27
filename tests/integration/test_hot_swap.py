"""Integration test: hot-swapping an ACCEPTed regenerate candidate into LIVE prediction serving,
without a process restart — the real deliverable behind `ContinuousOrchestrator._hot_swap_
candidate` (`src/main.py`). `tests/unit/test_main.py` already proves the swap mechanism itself in
isolation (real `DTModelRegistry`/`ModelRegistry`, real dynamic class loading, no mocking of
those); this file proves the actual WIRING inside `_run_adaptation_cycle` — that a real ACCEPT
triggers it and a real REJECT does not — against a fully real, `initialize()`d orchestrator with
real accumulated live telemetry and a real sandboxed subprocess.

The LLM is the one thing faked here (no real GOOGLE_API_KEY assumed) — a fake client serving both
the decision agent's call (deterministically "regenerate") and the regeneration agent's own
candidate-generation call, discriminated purely by schema, exactly like one real `GoogleClient`
would in production (the established pattern across this project's own integration tests, e.g.
`tests/integration/test_lifecycle_agent.py`). The candidate itself is genuinely sandboxed, trained
on real accumulated live telemetry, and independently re-verified — nothing about accept/reject is
scripted. To make a genuine ACCEPT reliably reachable without flakiness, the production baseline is
deliberately trained on a tiny, degenerate slice first — the same "deliberately weak early-slice
bootstrap production model" technique CLAUDE.md documents `tests/integration/
test_verification_agent.py` already established for exactly this reason.
"""

from __future__ import annotations

import time

import pandas as pd

from src.adaptation.decision_agent import DecisionOutput
from src.adaptation.regeneration_agent import _GeneratedComponentCode
from src.common.config import load_secrets, load_settings
from src.dt_models.throughput import ThroughputModel
from src.fidelity.trigger import AdaptationTrigger
from src.llm.google_client import LLMClientError
from src.main import ContinuousOrchestrator

SETTINGS = load_settings()
SECRETS = load_secrets()
MIN_HISTORY_ROWS = 220
ACCUMULATION_TIMEOUT_SECONDS = 60.0

_REBUILT_THROUGHPUT_SOURCE = """
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.dt_models.base import DTComponent

class RebuiltThroughput(DTComponent):
    COMPONENT_NAME = "throughput"
    DEPENDENCIES = ()
    REQUIRED_FEATURES = ("offered_load_mbps", "prb_utilization_pct", "sinr_db", "rsrp_dbm", "rsrq_db", "ue_count", "ue_speed_mps")
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
        return {"rmse": rmse}

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(self._model, path)

    def load(self, path):
        import joblib
        self._model = joblib.load(path)
        self._trained = True
"""


class _FakeLLMClient:
    """Serves BOTH real callers this cycle needs — discriminated purely by schema, never a
    hand-scripted decision about accept/reject/hot-swap itself."""

    def complete_structured(self, prompt, schema, **kwargs):
        if schema is DecisionOutput:
            return DecisionOutput(
                strategy="regenerate", root_cause_analysis="forced for this test", confidence=0.9,
                rationale="deterministic test scenario", knowledge_refs=[],
            )
        if schema is _GeneratedComponentCode:
            return schema(class_name="RebuiltThroughput", source_code=_REBUILT_THROUGHPUT_SOURCE, reasoning="test candidate")
        raise LLMClientError(f"fake client asked for an unsupported schema: {schema}")

    def complete_structured_safe(self, prompt, schema, **kwargs):
        try:
            return self.complete_structured(prompt, schema, **kwargs)
        except LLMClientError:
            return None

    def complete_safe(self, prompt, **kwargs):
        return None  # verification's/lifecycle's optional LLM prose degrades gracefully


def _fast_orchestrator(tmp_path) -> ContinuousOrchestrator:
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
    orchestrator = ContinuousOrchestrator(fast_settings, SECRETS)
    orchestrator.initialize()
    return orchestrator


def _weaken_production_throughput_baseline(orchestrator: ContinuousOrchestrator) -> None:
    """Trains and promotes a deliberately terrible throughput model — 2 degenerate rows, an
    absurd constant target — as the CURRENT production version, both in the versioned
    ModelRegistry and in live serving. This is the same real, established technique CLAUDE.md
    documents `tests/integration/test_verification_agent.py` already using ("a deliberately weak
    early-slice bootstrap production model") to make a genuine ACCEPT reliably reachable without
    ever scripting the actual accept/reject decision itself — a properly-trained regenerated
    candidate trained on real accumulated live telemetry has genuine, real room to legitimately
    beat this baseline."""
    weak = ThroughputModel.from_settings(orchestrator._settings)  # noqa: SLF001 - test-only access
    degenerate = pd.DataFrame(
        {
            "offered_load_mbps": [1.0, 1.0], "prb_utilization_pct": [1.0, 1.0], "sinr_db": [0.0, 0.0],
            "rsrp_dbm": [-100.0, -100.0], "rsrq_db": [-10.0, -10.0], "ue_count": [1, 1], "ue_speed_mps": [0.0, 0.0],
        }
    )
    weak.train(degenerate, pd.Series([500.0, 500.0]))  # nowhere near real throughput values
    orchestrator.model_registry.register_version(
        component_instance=weak, adaptation_type="bootstrap", parent_version_id=None,
        training_window={"n_rows": 2}, evaluation_window={"n_rows": 0}, evaluation_metrics={}, status="production",
    )
    orchestrator.dt_model_registry.replace(weak)


def test_accepted_regenerate_candidate_hot_swaps_into_live_serving(tmp_path):
    orchestrator = _fast_orchestrator(tmp_path)
    _weaken_production_throughput_baseline(orchestrator)
    orchestrator.llm_client = _FakeLLMClient()  # type: ignore[assignment]
    orchestrator.decision_agent._llm_client = orchestrator.llm_client  # noqa: SLF001 - test wiring
    orchestrator.lifecycle_agent._llm_client = orchestrator.llm_client  # noqa: SLF001 - captured at initialize() time, else it'd keep the real one

    orchestrator.start()
    try:
        wait_start = time.monotonic()
        while len(orchestrator.d1_store.get_history()) < MIN_HISTORY_ROWS:
            assert time.monotonic() - wait_start < ACCUMULATION_TIMEOUT_SECONDS, "live D1 history never accumulated enough rows"
            time.sleep(0.1)

        # Warm the SHARED FidelityEvaluator past min_history_for_normalization first — verify()
        # reuses this same evaluator instance, and an unwarmed one legitimately (and correctly)
        # reports insufficient_history rather than fabricating a score, same as
        # test_main_orchestrator.py's own established pattern for this.
        min_history = orchestrator._settings.fidelity.min_history_for_normalization  # noqa: SLF001
        for _ in range(min_history + 2):
            orchestrator.run_prediction_and_fidelity_cycle()
            time.sleep(0.05)

        trigger = AdaptationTrigger(component="throughput", severity=0.5, timestamp=pd.Timestamp.now(tz="UTC"), trigger_type="external_drift")
        orchestrator._run_adaptation_cycle(trigger)  # noqa: SLF001 - directly exercising the real cycle
    finally:
        orchestrator.stop()

    records = orchestrator.lifecycle_agent.list_records()
    assert len(records) == 1
    record = records[0]
    assert record.decision_strategy == "regenerate"
    assert record.verification_result == "ACCEPT", (
        f"expected a genuine ACCEPT against the deliberately weak baseline, got REJECT: {record.verification_explanation}"
    )
    assert record.final_status == "promoted"

    # The real deliverable: live serving now uses the REGENERATED class, not the original one,
    # and NOT via a process restart — this same orchestrator instance, still running.
    live = orchestrator.dt_model_registry.get("throughput")
    assert type(live).__name__ == "RebuiltThroughput"
    assert orchestrator._dynamic_component_classes["throughput"].__name__ == "RebuiltThroughput"  # noqa: SLF001

    # And it genuinely serves predictions through the real orchestrator wiring, not just sitting
    # registered — run_predictions() (what run_prediction_and_fidelity_cycle() itself calls) goes
    # through the swapped-in class.
    features = orchestrator.d1_store.get_history().tail(5)
    preds = orchestrator.dt_orchestrator.run_predictions(features)["throughput"]
    assert len(preds) == 5

    # A later restart of a FRESH orchestrator process pointed at the SAME storage must resume the
    # regenerated class too — "permanent," not merely alive for this process's lifetime.
    resumed = ContinuousOrchestrator(orchestrator._settings, SECRETS)  # noqa: SLF001
    resumed.initialize()
    assert type(resumed.dt_model_registry.get("throughput")).__name__ == "RebuiltThroughput"


def test_rejected_candidate_never_hot_swaps(tmp_path):
    """The mirror case: a REJECT must leave live serving completely untouched — no dynamic class
    ever loaded, the original class still serving. Uses the SAME real machinery as the ACCEPT
    test above, just without weakening the production baseline first, so the real deterministic
    gate has no genuine improvement to accept."""
    orchestrator = _fast_orchestrator(tmp_path)
    orchestrator.llm_client = _FakeLLMClient()  # type: ignore[assignment]
    orchestrator.decision_agent._llm_client = orchestrator.llm_client  # noqa: SLF001
    orchestrator.lifecycle_agent._llm_client = orchestrator.llm_client  # noqa: SLF001

    orchestrator.start()
    try:
        wait_start = time.monotonic()
        while len(orchestrator.d1_store.get_history()) < MIN_HISTORY_ROWS:
            assert time.monotonic() - wait_start < ACCUMULATION_TIMEOUT_SECONDS, "live D1 history never accumulated enough rows"
            time.sleep(0.1)

        trigger = AdaptationTrigger(component="throughput", severity=0.5, timestamp=pd.Timestamp.now(tz="UTC"), trigger_type="external_drift")
        orchestrator._run_adaptation_cycle(trigger)  # noqa: SLF001
    finally:
        orchestrator.stop()

    records = orchestrator.lifecycle_agent.list_records()
    assert len(records) == 1
    assert records[0].verification_result == "REJECT"
    assert records[0].final_status == "rejected"

    assert type(orchestrator.dt_model_registry.get("throughput")) is ThroughputModel
    assert "throughput" not in orchestrator._dynamic_component_classes  # noqa: SLF001
