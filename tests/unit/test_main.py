"""Unit tests for `src/main.py`'s pure integration-glue logic — the runtime PPO observation
construction and the DT-model bootstrap dispatch (load-existing vs. train-new). Fast, no real
telemetry pipeline or model training involved except where explicitly noted. The REQUIRED real,
end-to-end "run one full cycle, confirm telemetry never stopped" proof is
`scripts/run_orchestrator_demo.py` (a real, permanent, repo-tracked validation script — see
CLAUDE.md's Phase 11 entry for the actual observed numbers from a real run) plus the lighter-
weight, still-genuinely-real `tests/integration/test_main_orchestrator.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.adaptation.rl_env import NETWORK_STATE_FEATURES
from src.common.config import Secrets, load_settings
from src.drift.schema import DriftEvent
from src.main import _DT_COMPONENT_CLASSES, _OUTPUT_FIELDS, _TARGET_COLUMNS, ContinuousOrchestrator

SETTINGS = load_settings()
NO_KEY_SECRETS = Secrets(anthropic_api_key=None)


def _event(component: str, severity: float) -> DriftEvent:
    now = datetime.now(UTC)
    return DriftEvent(component=component, severity=severity, timestamp=now, metadata={}, source="MOCK", received_at=now)


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


# --- runtime PPO observation construction ----------------------------------------------------------


def _network_state_history(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data = {col: rng.uniform(1, 100, n) for col in NETWORK_STATE_FEATURES}
    return pd.DataFrame(data)


def test_runtime_observation_has_the_exact_dimension_ppo_was_trained_on():
    orch = _bare_orchestrator()
    orch.d1_store = _FakeD1Store(_network_state_history())
    orch._latest_fidelity = {c: 0.5 for c in SETTINGS.drift.valid_components}
    orch._prev_action_idx = None
    orch._prev_reward = 0.0

    obs = orch._build_runtime_observation(_event(SETTINGS.drift.valid_components[0], 0.4))

    n_components = len(SETTINGS.drift.valid_components)
    n_actions = len(SETTINGS.ppo.action_mapping)
    expected_dim = n_components + n_components + 1 + n_actions + 1 + len(NETWORK_STATE_FEATURES)
    assert obs.shape == (expected_dim,)
    assert obs.dtype == np.float32
    assert np.all(obs >= -2.0) and np.all(obs <= 2.0)


def test_runtime_observation_affected_component_onehot_is_correct():
    orch = _bare_orchestrator()
    orch.d1_store = _FakeD1Store(_network_state_history())
    orch._latest_fidelity = {c: None for c in SETTINGS.drift.valid_components}
    orch._prev_action_idx = None
    orch._prev_reward = 0.0

    components = tuple(SETTINGS.drift.valid_components)
    target = components[2]
    obs = orch._build_runtime_observation(_event(target, 0.7))

    n = len(components)
    affected_onehot = obs[n : 2 * n]
    assert affected_onehot[2] == pytest.approx(1.0)
    assert affected_onehot.sum() == pytest.approx(1.0)
    # a component with no fidelity evaluated yet defaults to a neutral 0.0, never a crash/NaN
    assert np.all(np.isfinite(obs[:n]))


def test_runtime_observation_previous_action_onehot_reflects_last_real_action():
    orch = _bare_orchestrator()
    orch.d1_store = _FakeD1Store(_network_state_history())
    orch._latest_fidelity = {c: 0.5 for c in SETTINGS.drift.valid_components}
    orch._prev_action_idx = 1
    orch._prev_reward = 0.3

    obs = orch._build_runtime_observation(_event(SETTINGS.drift.valid_components[0], 0.4))

    n = len(SETTINGS.drift.valid_components)
    n_actions = len(SETTINGS.ppo.action_mapping)
    prev_action_slice = obs[2 * n + 1 : 2 * n + 1 + n_actions]
    assert prev_action_slice[1] == pytest.approx(1.0)
    assert prev_action_slice.sum() == pytest.approx(1.0)


def test_network_state_summary_is_zeros_for_empty_history():
    orch = _bare_orchestrator()
    orch.d1_store = _FakeD1Store(pd.DataFrame(columns=list(NETWORK_STATE_FEATURES)))
    summary = orch._compute_network_state_summary()
    assert summary.shape == (len(NETWORK_STATE_FEATURES),)
    assert np.all(summary == 0.0)


def test_network_state_summary_is_normalized_into_a_bounded_range():
    orch = _bare_orchestrator()
    orch.d1_store = _FakeD1Store(_network_state_history())
    summary = orch._compute_network_state_summary()
    assert summary.shape == (len(NETWORK_STATE_FEATURES),)
    assert np.all(summary >= -0.01) and np.all(summary <= 1.01)  # self-referential min-max normalization


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
