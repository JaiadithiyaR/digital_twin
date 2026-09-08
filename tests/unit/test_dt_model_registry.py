"""Unit tests for DTModelRegistry (Module 5, src/dt_models/model_registry.py)."""

from __future__ import annotations

import pytest

from src.dt_models.model_registry import DTModelRegistry
from tests.dummy_dt_components import DummyAdder, DummyDoubler


def test_register_and_get():
    registry = DTModelRegistry()
    doubler = DummyDoubler()
    registry.register(doubler)
    assert registry.get("dummy_doubler") is doubler


def test_duplicate_registration_raises():
    registry = DTModelRegistry()
    registry.register(DummyDoubler())
    with pytest.raises(ValueError):
        registry.register(DummyDoubler())


def test_get_missing_component_raises_keyerror():
    registry = DTModelRegistry()
    with pytest.raises(KeyError):
        registry.get("does_not_exist")


def test_contains_operator():
    registry = DTModelRegistry()
    registry.register(DummyDoubler())
    assert "dummy_doubler" in registry
    assert "nope" not in registry


def test_components_default_enabled():
    registry = DTModelRegistry()
    registry.register(DummyDoubler())
    assert registry.is_enabled("dummy_doubler")
    assert "dummy_doubler" in registry.enabled_components()


def test_register_with_enabled_false():
    registry = DTModelRegistry()
    registry.register(DummyDoubler(), enabled=False)
    assert not registry.is_enabled("dummy_doubler")
    assert "dummy_doubler" not in registry.enabled_components()
    assert "dummy_doubler" in registry  # still registered, just disabled


def test_set_enabled_toggles_membership_in_enabled_components():
    registry = DTModelRegistry()
    registry.register(DummyDoubler())
    registry.set_enabled("dummy_doubler", False)
    assert "dummy_doubler" not in registry.enabled_components()
    registry.set_enabled("dummy_doubler", True)
    assert "dummy_doubler" in registry.enabled_components()


def test_set_enabled_on_unregistered_component_raises():
    registry = DTModelRegistry()
    with pytest.raises(KeyError):
        registry.set_enabled("nope", True)


def test_list_components_returns_all_regardless_of_enabled_state():
    registry = DTModelRegistry()
    registry.register(DummyDoubler())
    registry.register(DummyAdder(), enabled=False)
    names = {c.COMPONENT_NAME for c in registry.list_components()}
    assert names == {"dummy_doubler", "dummy_adder"}
