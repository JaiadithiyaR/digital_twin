"""Unit tests for DTOrchestrator (Module 5, src/dt_models/orchestrator.py).

Uses trivial dummy components (tests/dummy_dt_components.py) with a real multi-level dependency
chain (DummyDoubler -> DummyAdder -> DummySummer, with DummySummer depending on BOTH upstream
components) to prove the orchestrator can register, run in correct dependency order, and retrieve
correct chained predictions — not just a single trivial no-dependency case.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator, OrchestratorError
from tests.dummy_dt_components import (
    DummyAdder,
    DummyDoubler,
    DummySummer,
    DummyUntrainable,
)


def _trained_chain_registry() -> DTModelRegistry:
    registry = DTModelRegistry()
    doubler, adder, summer = DummyDoubler(), DummyAdder(), DummySummer()
    for component in (doubler, adder, summer):
        component.train(pd.DataFrame(), pd.Series(dtype=float))  # trivial "fit" — just flips is_trained
        registry.register(component)
    return registry


# --- execution order ------------------------------------------------------------------------


def test_execution_order_respects_dependencies():
    orchestrator = DTOrchestrator(_trained_chain_registry())
    order = orchestrator.build_execution_order()
    assert order.index("dummy_doubler") < order.index("dummy_adder")
    assert order.index("dummy_doubler") < order.index("dummy_summer")
    assert order.index("dummy_adder") < order.index("dummy_summer")


def test_execution_order_detects_missing_dependency():
    registry = DTModelRegistry()
    registry.register(DummyAdder())  # depends on "dummy_doubler", never registered
    orchestrator = DTOrchestrator(registry)
    with pytest.raises(OrchestratorError, match="not registered"):
        orchestrator.build_execution_order()


def test_execution_order_detects_disabled_dependency():
    registry = DTModelRegistry()
    registry.register(DummyDoubler(), enabled=False)
    registry.register(DummyAdder())
    orchestrator = DTOrchestrator(registry)
    with pytest.raises(OrchestratorError, match="disabled"):
        orchestrator.build_execution_order()


def test_execution_order_detects_cycle():
    class Left(DummyDoubler):
        COMPONENT_NAME = "left"
        DEPENDENCIES = ("right",)

    class Right(DummyDoubler):
        COMPONENT_NAME = "right"
        DEPENDENCIES = ("left",)

    registry = DTModelRegistry()
    registry.register(Left())
    registry.register(Right())
    orchestrator = DTOrchestrator(registry)
    with pytest.raises(OrchestratorError, match="cyclic"):
        orchestrator.build_execution_order()


def test_disabled_component_excluded_from_execution_order():
    registry = _trained_chain_registry()
    registry.set_enabled("dummy_summer", False)
    orchestrator = DTOrchestrator(registry)
    order = orchestrator.build_execution_order()
    assert "dummy_summer" not in order
    assert "dummy_doubler" in order
    assert "dummy_adder" in order


# --- run_predictions: the core register -> run -> retrieve proof ----------------------------


def test_run_predictions_produces_correct_chained_results():
    orchestrator = DTOrchestrator(_trained_chain_registry())
    features = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})

    predictions = orchestrator.run_predictions(features)

    assert set(predictions.keys()) == {"dummy_doubler", "dummy_adder", "dummy_summer"}
    expected_a = features["x"] * 2  # 2, 4, 6
    expected_b = features["y"] + expected_a  # 12, 24, 36
    expected_c = expected_a + expected_b  # 14, 28, 42

    pd.testing.assert_series_equal(predictions["dummy_doubler"], expected_a, check_names=False)
    pd.testing.assert_series_equal(predictions["dummy_adder"], expected_b, check_names=False)
    pd.testing.assert_series_equal(predictions["dummy_summer"], expected_c, check_names=False)


def test_run_predictions_excludes_disabled_components_from_result():
    registry = _trained_chain_registry()
    registry.set_enabled("dummy_summer", False)
    orchestrator = DTOrchestrator(registry)
    features = pd.DataFrame({"x": [1.0], "y": [10.0]})

    predictions = orchestrator.run_predictions(features)

    assert set(predictions.keys()) == {"dummy_doubler", "dummy_adder"}


def test_run_predictions_raises_on_untrained_component():
    registry = DTModelRegistry()
    registry.register(DummyUntrainable())
    orchestrator = DTOrchestrator(registry)
    with pytest.raises(OrchestratorError, match="not trained"):
        orchestrator.run_predictions(pd.DataFrame())


def test_run_predictions_raises_on_missing_required_feature():
    registry = DTModelRegistry()
    doubler = DummyDoubler()
    doubler.train(pd.DataFrame(), pd.Series(dtype=float))
    registry.register(doubler)
    orchestrator = DTOrchestrator(registry)

    with pytest.raises(OrchestratorError, match="x"):
        orchestrator.run_predictions(pd.DataFrame({"not_x": [1.0]}))


def test_run_predictions_never_calls_predict_on_untrained_component_even_mid_chain():
    """A downstream component must not silently run on garbage if an upstream one isn't ready —
    confirmed here by DummyUntrainable's predict() raising AssertionError if ever called."""
    registry = DTModelRegistry()
    registry.register(DummyUntrainable())
    orchestrator = DTOrchestrator(registry)
    with pytest.raises(OrchestratorError):
        orchestrator.run_predictions(pd.DataFrame())
