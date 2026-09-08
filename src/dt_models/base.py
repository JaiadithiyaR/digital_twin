"""Common interface every DT prediction component (Modules 6-10) must implement.

Defines the shape the orchestrator depends on — `train`/`predict`/`evaluate`/`save`/`load`, plus
class-level metadata each component declares about its place in the dependency graph — so the
orchestrator can schedule and wire any component using only this metadata, with zero
component-specific code (prompt.md §11: "Do NOT simply call all five models independently from
main.py. Implement a topological/dependency-aware scheduler.").

No concrete model (throughput/latency/packet_loss/prb_utilization/jitter) is implemented here —
those are Modules 6-10, built separately on top of this interface. This module is validated by a
trivial dummy component in tests (`tests/dummy_dt_components.py`), never by a real model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import pandas as pd


class DTComponent(ABC):
    """Base class for every DT prediction component.

    Class-level metadata (declared once per component class, not per instance):
      - `COMPONENT_NAME`: unique registry key.
      - `DEPENDENCIES`: names of other registered components whose PREDICTIONS this component
        consumes as input (e.g. jitter depends on latency, throughput, packet_loss — prompt.md
        §13). The orchestrator supplies each dependency's prediction as an input column named
        after that dependency's `OUTPUT_FIELD`.
      - `REQUIRED_FEATURES`: raw feature/telemetry column names (typically read from D1) this
        component needs directly, independent of any upstream prediction.
      - `OUTPUT_FIELD`: the name this component's prediction is published under — both as the
        name of its own output series and as the input column name any downstream dependent
        receives it under.
    """

    COMPONENT_NAME: ClassVar[str]
    DEPENDENCIES: ClassVar[tuple[str, ...]] = ()
    REQUIRED_FEATURES: ClassVar[tuple[str, ...]] = ()
    OUTPUT_FIELD: ClassVar[str]

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Whether `predict()` can currently be called. The orchestrator checks this before
        every prediction run and fails explicitly (never silently) if a component isn't ready —
        prompt.md §0.17: "fail explicitly rather than silently producing incorrect predictions."
        """

    @abstractmethod
    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Fit the model. `features` columns cover `REQUIRED_FEATURES` (and, when training on
        historical ground truth, may include upstream components' true historical values under
        their `OUTPUT_FIELD` names)."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return one prediction per row of `features`, index-aligned with it."""

    @abstractmethod
    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        """Return {metric_name: value} (e.g. {"rmse": ...}) — a component-local sanity metric.
        Module 12's fidelity engine defines the authoritative, deterministic fidelity formula;
        this is not that score."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist trained model state to `path`."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore trained model state from `path`, written by a prior `save()`."""
