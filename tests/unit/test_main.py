"""Unit tests for `src/main.py`'s pure integration-glue logic — per-component previous-outcome
tracking (feeding the Decision & Root-Cause Analysis Agent's context, prompt.md §18) and the
DT-model bootstrap dispatch (load-existing vs. train-new). Fast, no real telemetry pipeline or
model training involved except where explicitly noted. The REQUIRED real, end-to-end "run one full
cycle, confirm telemetry never stopped" proof is `scripts/run_orchestrator_demo.py` (a real,
permanent, repo-tracked validation script — see CLAUDE.md's Phase 11 entry for the actual observed
numbers from a real run) plus the lighter-weight, still-genuinely-real
`tests/integration/test_main_orchestrator.py`. The decision-CONTEXT-building logic itself
(fidelity vector, unified score, network-state summary, RAG retrieval) is `src.adaptation.
decision_context`'s own concern, tested in `tests/unit/test_decision_context.py` — not duplicated
here.
"""

from __future__ import annotations

import pandas as pd

from src.adaptation.decision_context import PreviousOutcome
from src.common.config import Secrets, load_settings
from src.main import _DT_COMPONENT_CLASSES, _OUTPUT_FIELDS, _TARGET_COLUMNS, ContinuousOrchestrator

SETTINGS = load_settings()
NO_KEY_SECRETS = Secrets(google_api_key=None)


class _FakeD1Store:
    def __init__(self, history: pd.DataFrame) -> None:
        self._history = history

    def get_history(self, ue_id=None, cell_id=None):
        return self._history


def _bare_orchestrator() -> ContinuousOrchestrator:
    """Constructs the orchestrator WITHOUT calling `initialize()` — legitimate for testing the
    pure observation-construction/bootstrap-dispatch logic in isolation, since `__init__` sets up
    only plain state (no I/O, no telemetry/model construction)."""
    return ContinuousOrchestrator(SETTINGS, NO_KEY_SECRETS)


# --- component spec consistency (the dispatch tables main.py itself relies on) --------------------


def test_component_spec_tables_are_internally_consistent():
    assert set(_DT_COMPONENT_CLASSES) == set(_TARGET_COLUMNS) == set(_OUTPUT_FIELDS)
    assert set(_DT_COMPONENT_CLASSES) == set(SETTINGS.drift.valid_components)
    for name, cls in _DT_COMPONENT_CLASSES.items():
        assert cls.COMPONENT_NAME == name
        assert _OUTPUT_FIELDS[name] == cls.OUTPUT_FIELD


# --- per-component previous-outcome tracking (feeds the Decision & Root-Cause Analysis Agent) ----


def test_previous_outcome_for_unknown_component_is_the_honest_default():
    orch = _bare_orchestrator()
    result = orch._previous_outcome_for("throughput")
    assert result == PreviousOutcome()  # no prior attempt on record — never fabricated


def test_previous_outcome_for_returns_the_tracked_value_for_that_component_only():
    orch = _bare_orchestrator()
    orch._previous_outcome_by_component["jitter"] = PreviousOutcome(
        action="recalibrate", verification_result="ACCEPT", fidelity_before=0.5, fidelity_after=0.9
    )

    assert orch._previous_outcome_for("jitter").action == "recalibrate"
    assert orch._previous_outcome_for("latency") == PreviousOutcome()  # independent per component


# --- bootstrap dispatch: load existing vs. train new (no real training triggered here) ------------


class _FakeModelRegistry:
    """Stands in for `ModelRegistry` — only the two methods `_bootstrap_dt_models` actually calls."""

    def __init__(self, existing_versions: dict[str, object]) -> None:
        self._existing = existing_versions
        self.loaded: list[str] = []

    def get_current_version(self, component: str):
        return self._existing.get(component)

    def load_artifact_into(self, instance, version) -> None:
        self.loaded.append(instance.COMPONENT_NAME)
        instance._trained = True  # noqa: SLF001 - test double simulating a real load


class _FakeVersion:
    def __init__(self, version_id: str) -> None:
        self.version_id = version_id


def test_bootstrap_dt_models_loads_existing_versions_without_generating_bootstrap_data():
    orch = _bare_orchestrator()
    fake_versions = {name: _FakeVersion(f"{name}-v1") for name in _DT_COMPONENT_CLASSES}  # every component already has a production version
    orch.model_registry = _FakeModelRegistry(fake_versions)

    def _fail_if_called():
        raise AssertionError("bootstrap telemetry must not be generated when every component already has a production version")

    orch._generate_bootstrap_history = _fail_if_called  # type: ignore[method-assign]
    orch._bootstrap_dt_models()

    assert set(orch.model_registry.loaded) == set(_DT_COMPONENT_CLASSES)
    assert set(orch._dt_components) == set(_DT_COMPONENT_CLASSES)
    for instance in orch._dt_components.values():
        assert instance.is_trained
