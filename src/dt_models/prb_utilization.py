"""Module 9 — PRB-Utilization Model (prompt.md §12.4, fig-dataflow.png node 9).

Inputs: offered load, UE count, throughput, cell load.
Output: PRB utilization (%).

Marked "independent/optional" in the diagram (prompt.md §13: "PRB Utilization remains
independently configurable") — nothing downstream currently depends on this component's output,
so toggling it off via `config.dt_models.enable_prb_model` (mapped to `DTModelRegistry.register
(..., enabled=...)`) cannot break any other component. That toggle is a REGISTRATION-time
decision made by whatever wires the registry up (a bootstrap script, or a test demonstrating the
wiring) — this file implements the component fully and unconditionally; it is not itself a stub
and does not know or care whether it is currently enabled.

Depends on Module 6 (Throughput) in the DT dependency graph — explicitly instructed this turn
(unlike Modules 7/8, where the equivalent upstream model didn't exist yet at build time, Module 6
already exists, so `DEPENDENCIES=("throughput",)` wires a REAL dependency rather than falling
back to a raw ground-truth feature).

**"Cell load" is not a raw D1 column** — D1's telemetry schema has no such field (see
`d1_model_store.TELEMETRY_COLUMNS`). It is deliberately NOT approximated by `prb_utilization_pct`
itself: that would hand this model a near-perfect proxy for its own prediction target as an
input (leakage/tautology), and prompt.md's own input list for this component excludes PRB
entirely (unlike every other component's input list, which includes PRB as an input) — clearly
intentional. Instead, "cell load" is derived here as a genuine aggregate feature: the SUM of
`offered_load_mbps` across every UE attached to the same cell at the same timestamp
(`_add_cell_load`). This is a legitimate, real engineering interpretation (aggregate cell-wide
traffic demand, distinct from any single UE's own offered load, which is already a separate
input) rather than a fabricated field — see `_add_cell_load`'s docstring for the mechanics and
its one documented limitation (a `features` slice missing part of a cell's cohort at a given
timestamp degenerates to a partial, not incorrect, aggregate).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.dt_models.base import DTComponent
from src.dt_models.regressors import build_regressor

if TYPE_CHECKING:
    from src.common.config import Settings

logger = logging.getLogger(__name__)


class PrbUtilizationModel(DTComponent):
    COMPONENT_NAME = "prb_utilization"
    DEPENDENCIES = ("throughput",)
    # Raw D1 columns this component needs directly, PLUS cell_id/timestamp — not used as
    # regressor inputs themselves, but required to derive the cell-load aggregate (see
    # `_add_cell_load`). The orchestrator validates REQUIRED_FEATURES against its raw `features`
    # frame, so both the "real" numeric inputs and these grouping keys must be declared here.
    REQUIRED_FEATURES = ("offered_load_mbps", "ue_count", "cell_id", "timestamp")
    OUTPUT_FIELD = "prb_utilization_pct_pred"

    # Name of the input column carrying the throughput dependency's value — populated by
    # DTOrchestrator from the throughput component's prediction at inference time; must be
    # supplied explicitly (from ground truth) by callers training this component directly.
    THROUGHPUT_INPUT_COLUMN = "throughput_mbps_pred"

    # Derived (not raw) cell-wide load aggregate — see module docstring.
    CELL_LOAD_COLUMN = "cell_load_mbps"

    def __init__(self, model_type: str = "random_forest", params: dict[str, Any] | None = None) -> None:
        self._model_type = model_type
        self._params = dict(params or {})
        self._model = build_regressor(model_type, self._params)
        self._trained = False

    @classmethod
    def from_settings(cls, settings: "Settings") -> "PrbUtilizationModel":
        spec = settings.dt_models.prb_utilization
        return cls(model_type=spec.model_type, params=spec.params)

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _add_cell_load(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive `CELL_LOAD_COLUMN`: sum of `offered_load_mbps` across every row sharing the
        same `(cell_id, timestamp)` — the cell's aggregate traffic demand at that instant.
        `MockTelemetrySource` (and any real per-tick exporter) emits every UE of one tick under
        a shared timestamp, so this groupby correctly reconstructs the full-cohort aggregate
        whenever `df` contains that cohort. If `df` contains only a partial cohort for a given
        (cell_id, timestamp) — e.g. a single isolated prediction row — the aggregate degrades
        gracefully to the partial sum (not an error, just a weaker signal)."""
        result = df.copy()
        result[self.CELL_LOAD_COLUMN] = result.groupby(["cell_id", "timestamp"])["offered_load_mbps"].transform(
            "sum"
        )
        return result

    def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.REQUIRED_FEATURES if c not in features.columns]
        if self.THROUGHPUT_INPUT_COLUMN not in features.columns:
            missing = [*missing, self.THROUGHPUT_INPUT_COLUMN]
        if missing:
            raise ValueError(f"PrbUtilizationModel missing required input column(s): {missing}")

        enriched = self._add_cell_load(features)
        model_columns = ["offered_load_mbps", "ue_count", self.THROUGHPUT_INPUT_COLUMN, self.CELL_LOAD_COLUMN]
        return enriched[model_columns]

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._select_features(features)
        combined = pd.concat([X, targets.rename("__target__")], axis=1).dropna()
        if combined.empty:
            raise ValueError("PrbUtilizationModel.train received no valid (non-NaN) rows")
        dropped = len(X) - len(combined)
        if dropped:
            logger.warning(
                "dropped rows with NaN before training",
                extra={"component": "prb_utilization_model", "dropped_rows": dropped},
            )

        X_clean = combined[X.columns.tolist()]
        y_clean = combined["__target__"]
        self._model.fit(X_clean, y_clean)
        self._trained = True
        logger.info(
            "PRB-utilization model trained",
            extra={"component": "prb_utilization_model", "model_type": self._model_type, "rows": len(X_clean)},
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._trained:
            raise RuntimeError("PrbUtilizationModel.predict called before train()/load()")
        X = self._select_features(features)
        preds = self._model.predict(X)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        """Component-local sanity metrics — RMSE/MAE. NOT the authoritative fidelity score
        (Module 12 owns that deterministic, rolling-normalized formula)."""
        preds = self.predict(features)
        aligned_targets = targets.loc[preds.index]
        rmse = float(mean_squared_error(aligned_targets, preds) ** 0.5)
        mae = float(mean_absolute_error(aligned_targets, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self._model, "model_type": self._model_type, "params": self._params, "trained": self._trained},
            path,
        )

    def load(self, path: Path) -> None:
        state = joblib.load(path)
        self._model = state["model"]
        self._model_type = state["model_type"]
        self._params = state["params"]
        self._trained = state["trained"]
