"""In-memory dependency-graph registry for DT prediction components — not the same thing as
`component_registry.py`'s `ComponentRegistry` (catalogs D1's state tables) or
`src/registry/model_registry.py`'s `ModelRegistry` (tracks model VERSIONS for the adaptation
lifecycle). This file only tracks which prediction components exist and how they depend on each
other, so `DTOrchestrator` can compute an execution order.

Registering a component here does not train, save, or touch any file — just makes it visible.
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

    def replace(self, component: DTComponent, enabled: bool | None = None) -> None:
        """Atomically swaps in a new instance under an already-registered name — what a
        hot-swapped ACCEPT uses to take over live serving without a restart. Requires the name
        to already exist; `enabled` defaults to preserving the current state."""
        name = component.COMPONENT_NAME
        if name not in self._components:
            raise KeyError(f"no component registered under '{name}' to replace — use register() for a new component")
        current_enabled = self._components[name].enabled
        self._components[name] = _RegisteredComponent(
            component=component, enabled=current_enabled if enabled is None else enabled
        )

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
