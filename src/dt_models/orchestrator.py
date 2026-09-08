"""Module 5 — DT Functional Model: dependency-aware orchestrator (prompt.md §11, §13, §0.17).

Computes a valid topological execution order from `DTModelRegistry`'s declared dependencies —
via networkx, never a hardcoded call sequence — and runs each ENABLED component's `predict()` in
that order, wiring each component's upstream-dependency inputs from predictions already computed
earlier in the same run, and its raw-feature inputs directly from a caller-supplied features
DataFrame (typically `D1Store.get_current_state()` or `.get_history()`).

The dependency graph is executable metadata, not documentation (prompt.md §0.17): a component
depending on an unregistered or disabled component, or a cyclic graph, fails loudly at
plan-construction time. This orchestrator never silently produces a prediction from
missing/incomplete inputs, and never predicts through a component that isn't trained.
"""

from __future__ import annotations

import logging

import networkx as nx
import pandas as pd

from src.dt_models.model_registry import DTModelRegistry

logger = logging.getLogger(__name__)


class OrchestratorError(Exception):
    """Invalid dependency graph or an unready component. Never swallowed — always propagates."""


class DTOrchestrator:
    def __init__(self, registry: DTModelRegistry) -> None:
        self._registry = registry

    def build_execution_order(self) -> list[str]:
        """Derive a valid topological execution order from the registry's ENABLED components.

        Raises `OrchestratorError` for a dependency on an unregistered or disabled component, or
        a cyclic graph — the orchestrator must detect an invalid dependency graph and fail
        explicitly rather than silently producing incorrect predictions (prompt.md §0.17).
        """
        enabled = self._registry.enabled_components()
        graph: nx.DiGraph = nx.DiGraph()
        graph.add_nodes_from(enabled.keys())

        for name, component in enabled.items():
            for dependency_name in component.DEPENDENCIES:
                if dependency_name not in enabled:
                    reason = (
                        "not registered"
                        if dependency_name not in self._registry
                        else "registered but disabled"
                    )
                    raise OrchestratorError(
                        f"component '{name}' depends on '{dependency_name}', which is {reason} "
                        "— invalid dependency graph (prompt.md §0.17)"
                    )
                graph.add_edge(dependency_name, name)

        try:
            return list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible as exc:
            cycle = next(iter(nx.simple_cycles(graph)), None)
            raise OrchestratorError(f"cyclic component dependency graph detected: {cycle}") from exc

    def run_predictions(self, features: pd.DataFrame) -> dict[str, pd.Series]:
        """Run every enabled component's `predict()` in dependency order and return
        `{component_name: predictions}` for all of them.

        `features` must contain every `REQUIRED_FEATURES` column any enabled component declares
        (raw telemetry/feature columns — read from D1 by the caller). Each component additionally
        receives one input column per declared dependency, named after that dependency's
        `OUTPUT_FIELD`, populated from the prediction already computed earlier in this same run —
        topological order guarantees it exists by the time it's needed.
        """
        order = self.build_execution_order()
        predictions: dict[str, pd.Series] = {}

        for name in order:
            component = self._registry.get(name)

            if not component.is_trained:
                raise OrchestratorError(
                    f"component '{name}' is not trained — cannot predict (prompt.md §0.17: fail "
                    "explicitly rather than silently producing incorrect predictions)"
                )

            missing_features = [f for f in component.REQUIRED_FEATURES if f not in features.columns]
            if missing_features:
                raise OrchestratorError(
                    f"component '{name}' requires feature(s) {missing_features} not present in "
                    "the supplied features"
                )

            component_input = features[list(component.REQUIRED_FEATURES)].copy()
            for dependency_name in component.DEPENDENCIES:
                dependency_component = self._registry.get(dependency_name)
                component_input[dependency_component.OUTPUT_FIELD] = predictions[dependency_name]

            logger.info(
                "running DT component prediction",
                extra={"component": "dt_orchestrator", "model": name, "rows": len(component_input)},
            )
            predictions[name] = component.predict(component_input)

        return predictions
