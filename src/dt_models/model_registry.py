"""Module 5's dependency-graph registry for DT prediction components (Modules 6-10).

`DTModelRegistry` is deliberately NOT the same thing as either of these — do not confuse them:

- `src/dt_models/component_registry.py`'s `ComponentRegistry` — catalogs D1's own state TABLES
  (`telemetry_current`/`telemetry_history`/`telemetry_quarantine`). Nothing to do with
  prediction models.
- The future `src/registry/model_registry.py` (prompt.md §46, not built yet) — will track MODEL
  VERSIONS for the adaptation lifecycle (current/previous version, artifact path, training/eval
  metadata, promotion status). A completely different concern from this file, despite the
  similar name — this one only tracks which prediction components exist and how they depend on
  each other, so `DTOrchestrator` can compute an execution order.

Registering a component here does not train it, save it, or touch any file — it only makes the
component (and its declared metadata) visible to the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.dt_models.base import DTComponent


@dataclass
class _RegisteredComponent:
    component: DTComponent
    enabled: bool


class DTModelRegistry:
    def __init__(self) -> None:
        self._components: dict[str, _RegisteredComponent] = {}

    def register(self, component: DTComponent, enabled: bool = True) -> None:
        name = component.COMPONENT_NAME
        if name in self._components:
            raise ValueError(f"component '{name}' is already registered")
        self._components[name] = _RegisteredComponent(component=component, enabled=enabled)

    def get(self, name: str) -> DTComponent:
        try:
            return self._components[name].component
        except KeyError:
            raise KeyError(f"no component registered under '{name}'") from None

    def is_enabled(self, name: str) -> bool:
        return name in self._components and self._components[name].enabled

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name not in self._components:
            raise KeyError(f"no component registered under '{name}'")
        self._components[name].enabled = enabled

    def enabled_components(self) -> dict[str, DTComponent]:
        """Name -> component, for every registered component with `enabled=True`. This is what
        `DTOrchestrator` schedules — a disabled component (e.g. PRB utilization when
        `enable_prb_model: false`) is simply excluded, not passed a fake input."""
        return {name: rc.component for name, rc in self._components.items() if rc.enabled}

    def list_components(self) -> list[DTComponent]:
        return [rc.component for rc in self._components.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._components
