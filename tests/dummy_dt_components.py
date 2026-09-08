"""Trivial dummy DTComponent implementations used ONLY to test Module 5's interface, registry,
and orchestrator (src/dt_models/{base,model_registry,orchestrator}.py) — not a real model, and
never imported by src/. Not a test_*.py file itself, so pytest does not collect it directly;
`tests/unit/test_orchestrator.py` and `tests/integration/test_orchestrator_with_d1.py` import
these classes.

Three components with a real multi-level dependency chain (mirrors the shape of
throughput/packet_loss -> latency -> jitter without implementing any of those real models):

    DummyDoubler   (no deps, requires "x")           -> "a_pred" = x * 2
    DummyAdder     (depends on DummyDoubler, requires "y") -> "b_pred" = y + a_pred
    DummySummer    (depends on both)                 -> "c_pred" = a_pred + b_pred
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.dt_models.base import DTComponent


class DummyDoubler(DTComponent):
    COMPONENT_NAME = "dummy_doubler"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES = ("x",)
    OUTPUT_FIELD = "a_pred"

    def __init__(self) -> None:
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        self._trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return (features["x"] * 2).rename(self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        return {"mae": float((preds - targets).abs().mean())}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"trained": self._trained}))

    def load(self, path: Path) -> None:
        self._trained = json.loads(path.read_text())["trained"]


class DummyAdder(DTComponent):
    COMPONENT_NAME = "dummy_adder"
    DEPENDENCIES = ("dummy_doubler",)
    REQUIRED_FEATURES = ("y",)
    OUTPUT_FIELD = "b_pred"

    def __init__(self) -> None:
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        self._trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return (features["y"] + features["a_pred"]).rename(self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        return {"mae": float((preds - targets).abs().mean())}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"trained": self._trained}))

    def load(self, path: Path) -> None:
        self._trained = json.loads(path.read_text())["trained"]


class DummySummer(DTComponent):
    COMPONENT_NAME = "dummy_summer"
    DEPENDENCIES = ("dummy_doubler", "dummy_adder")
    REQUIRED_FEATURES: tuple[str, ...] = ()
    OUTPUT_FIELD = "c_pred"

    def __init__(self) -> None:
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        self._trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return (features["a_pred"] + features["b_pred"]).rename(self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        return {"mae": float((preds - targets).abs().mean())}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"trained": self._trained}))

    def load(self, path: Path) -> None:
        self._trained = json.loads(path.read_text())["trained"]


class DummyUntrainable(DTComponent):
    """Always reports is_trained=False — for testing the orchestrator's explicit-failure path."""

    COMPONENT_NAME = "dummy_untrainable"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES: tuple[str, ...] = ()
    OUTPUT_FIELD = "untrainable_pred"

    @property
    def is_trained(self) -> bool:
        return False

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        pass

    def predict(self, features: pd.DataFrame) -> pd.Series:
        raise AssertionError("predict() must never be called when is_trained is False")

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        return {}

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass
