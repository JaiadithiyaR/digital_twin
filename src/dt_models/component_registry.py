"""D1's modular component registry (Module 4 / D1, fig-dataflow.png: "... with a modular
component registry").

A lightweight, self-describing catalog of the state "components" (tables) D1 tracks: what each
is called, what kind of table it is (a keyed current-state snapshot, an append-only history log,
or an audit trail), its schema, and where its Parquet file lives. This is what keeps D1 modular:
new state categories that later modules produce — DT predictions (Module 5+), adaptation events,
evaluation windows — register themselves here instead of being hardcoded into `D1Store`, and any
downstream module can introspect what D1 currently holds without hardcoding file paths of its own.

Registering a component only records the catalog entry — it never creates or touches a file.
`D1Store` (`d1_model_store.py`) owns actually reading/writing the Parquet file at each registered
path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ComponentKind = Literal["current_state", "history", "audit"]
_VALID_KINDS: frozenset[str] = frozenset({"current_state", "history", "audit"})


@dataclass(frozen=True)
class ComponentDescriptor:
    name: str
    kind: ComponentKind
    schema_fields: tuple[str, ...]
    storage_path: Path

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"invalid component kind {self.kind!r}; must be one of {sorted(_VALID_KINDS)}")


class ComponentRegistry:
    """In-memory catalog. Not persisted itself — component *definitions* are code, not data;
    only the data each component describes is persisted, by `D1Store`."""

    def __init__(self) -> None:
        self._components: dict[str, ComponentDescriptor] = {}

    def register(self, descriptor: ComponentDescriptor) -> None:
        if descriptor.name in self._components:
            raise ValueError(f"component '{descriptor.name}' is already registered")
        self._components[descriptor.name] = descriptor

    def get(self, name: str) -> ComponentDescriptor:
        try:
            return self._components[name]
        except KeyError:
            raise KeyError(f"no component registered under '{name}'") from None

    def list_components(self) -> list[ComponentDescriptor]:
        return list(self._components.values())

    def __contains__(self, name: str) -> bool:
        return name in self._components
