"""Module 7 — Latency Model (prompt.md §12.3, fig-dataflow.png node 7).

Inputs: offered load, throughput, PRB utilization, UE count, packet loss, SINR.
Output: latency (ms).

Depends on Module 6 (Throughput) in the DT dependency graph (prompt.md §13: "Latency depends on
throughput and packet loss. Therefore it must execute after the required upstream models.").

**Scope decision for this turn**: `DEPENDENCIES` wires only `"throughput"` as an orchestrator
dependency. Module 8 (Packet-Loss) does not exist yet, so `packet_loss_pct` is consumed here as
a RAW ground-truth telemetry feature straight from D1 (`REQUIRED_FEATURES`), not as a second
upstream model dependency. Once Module 8 is built, a natural follow-up is to add `"packet_loss"`
to `DEPENDENCIES` and drop `packet_loss_pct` from `REQUIRED_FEATURES` in favour of the
packet-loss model's own `OUTPUT_FIELD` — not done now, because that model doesn't exist yet to
depend on. This keeps the dependency graph always valid and executable (prompt.md §0.17) rather
than declaring a dependency on something unregistered.

**Training vs. serving throughput signal**: at inference time, `DTOrchestrator.run_predictions`
supplies the upstream Throughput component's live PREDICTION as the `throughput_mbps_pred` input
column (see `THROUGHPUT_INPUT_COLUMN`). When training this component directly — as bootstrap
training does, outside the orchestrator — the strongest available signal is the ground-truth
historical throughput; callers supply it under that same column name, e.g.
`history.rename(columns={"throughput_mbps": "throughput_mbps_pred"})`. This is the standard
"train on ground truth, serve on upstream predictions" pattern for a dependency-chained model —
`LatencyModel` itself is agnostic to where the value came from, it only ever consumes the column
by name.
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


class LatencyModel(DTComponent):
    COMPONENT_NAME = "latency"
    DEPENDENCIES = ("throughput",)
    REQUIRED_FEATURES = (
        "offered_load_mbps",
        "prb_utilization_pct",
        "ue_count",
        "packet_loss_pct",
        "sinr_db",
    )
    OUTPUT_FIELD = "latency_ms_pred"

    # Name of the input column carrying the throughput dependency's value — populated by
    # DTOrchestrator from the throughput component's prediction at inference time; must be
    # supplied explicitly (from ground truth) by callers training this component directly.
    THROUGHPUT_INPUT_COLUMN = "throughput_mbps_pred"

    def __init__(self, model_type: str = "xgboost", params: dict[str, Any] | None = None) -> None:
        self._model_type = model_type
        self._params = dict(params or {})
        self._model = build_regressor(model_type, self._params)
        self._trained = False

    @classmethod
    def from_settings(cls, settings: "Settings") -> "LatencyModel":
        spec = settings.dt_models.latency
        return cls(model_type=spec.model_type, params=spec.params)

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def _input_columns(self) -> tuple[str, ...]:
        """REQUIRED_FEATURES (raw D1 telemetry) plus the throughput dependency column — the
        full set of columns this component actually fits/predicts on. Deliberately NOT the same
        as REQUIRED_FEATURES: the orchestrator checks only REQUIRED_FEATURES against the raw
        `features` DataFrame it's given, since the throughput column is supplied separately, by
        wiring, not expected to already be present in raw D1 output."""
        return self.REQUIRED_FEATURES + (self.THROUGHPUT_INPUT_COLUMN,)

    def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self._input_columns if c not in features.columns]
        if missing:
            raise ValueError(f"LatencyModel missing required input column(s): {missing}")
        return features[list(self._input_columns)]

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._select_features(features)
        combined = pd.concat([X, targets.rename("__target__")], axis=1).dropna()
        if combined.empty:
            raise ValueError("LatencyModel.train received no valid (non-NaN) rows")
        dropped = len(X) - len(combined)
        if dropped:
            logger.warning(
                "dropped rows with NaN before training",
                extra={"component": "latency_model", "dropped_rows": dropped},
            )

        X_clean = combined[list(self._input_columns)]
        y_clean = combined["__target__"]
        self._model.fit(X_clean, y_clean)
        self._trained = True
        logger.info(
            "latency model trained",
            extra={"component": "latency_model", "model_type": self._model_type, "rows": len(X_clean)},
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._trained:
            raise RuntimeError("LatencyModel.predict called before train()/load()")
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
