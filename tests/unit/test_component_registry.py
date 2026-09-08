"""Unit tests for D1's modular component registry (src/dt_models/component_registry.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dt_models.component_registry import ComponentDescriptor, ComponentRegistry


def _descriptor(name: str = "telemetry_current", kind: str = "current_state") -> ComponentDescriptor:
    return ComponentDescriptor(
        name=name, kind=kind, schema_fields=("timestamp", "ue_id"), storage_path=Path("/tmp/x.parquet")
    )


def test_register_and_get():
    registry = ComponentRegistry()
    registry.register(_descriptor())
    descriptor = registry.get("telemetry_current")
    assert descriptor.name == "telemetry_current"
    assert descriptor.kind == "current_state"


def test_duplicate_registration_raises():
    registry = ComponentRegistry()
    registry.register(_descriptor())
    with pytest.raises(ValueError):
        registry.register(_descriptor())


def test_get_missing_component_raises_keyerror():
    registry = ComponentRegistry()
    with pytest.raises(KeyError):
        registry.get("does_not_exist")


def test_list_components_returns_all_registered():
    registry = ComponentRegistry()
    registry.register(_descriptor("a", "current_state"))
    registry.register(_descriptor("b", "history"))
    registry.register(_descriptor("c", "audit"))
    names = {d.name for d in registry.list_components()}
    assert names == {"a", "b", "c"}


def test_contains_operator():
    registry = ComponentRegistry()
    registry.register(_descriptor())
    assert "telemetry_current" in registry
    assert "nope" not in registry


def test_invalid_kind_rejected():
    with pytest.raises(ValueError):
        ComponentDescriptor(
            name="bad", kind="not_a_real_kind", schema_fields=(), storage_path=Path("/tmp/x.parquet")
        )
