"""DT Functional Model: dependency-aware orchestrator. Computes a topological execution order
from `DTModelRegistry`'s declared dependencies (via networkx, never hardcoded) and runs each
enabled component's `predict()` in that order — dependency inputs come from predictions already
computed earlier in the same run; raw-feature inputs from the caller-supplied DataFrame.

The dependency graph is executable metadata: a component depending on an unregistered/disabled
component, or a cyclic graph, fails loudly at plan-construction time — never a silent prediction
from incomplete inputs.
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

            # A component may also list a dependency's OUTPUT_FIELD in REQUIRED_FEATURES (LLM-
            # generated candidates do). That column is wired from the dependency's prediction, so
            # it must not be demanded from the raw features — doing so failed every prediction
            # cycle after such a candidate was hot-swapped in.
            dependency_fields = {self._registry.get(d).OUTPUT_FIELD: d for d in component.DEPENDENCIES}
            raw_features = [f for f in component.REQUIRED_FEATURES if f not in dependency_fields]
            missing_features = [f for f in raw_features if f not in features.columns]
            if missing_features:
                raise OrchestratorError(
                    f"component '{name}' requires feature(s) {missing_features} not present in "
                    "the supplied features"
                )

            component_input = features[raw_features].copy()
            for output_field, dependency_name in dependency_fields.items():
                component_input[output_field] = predictions[dependency_name]
            component_input = component_input[list(dict.fromkeys([*component.REQUIRED_FEATURES, *dependency_fields]))]

            logger.info(
                "running DT component prediction",
                extra={"component": "dt_orchestrator", "model": name, "rows": len(component_input)},
            )
            predictions[name] = component.predict(component_input)

        return predictions
