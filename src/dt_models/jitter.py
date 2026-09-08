"""Module 10 — Jitter Model (prompt.md §12.5, fig-dataflow.png node 10).

Inputs: latency, throughput, offered load, packet loss, UE count.
Output: jitter (ms).

The last node in the DT dependency graph (prompt.md §13: "throughput + latency + packet loss ->
jitter"; §12.5: "Jitter must execute after its upstream dependencies"). Unlike Modules 7-9, all
three upstream models this component needs (throughput, latency, packet_loss) already exist by
the time this one is built, so `DEPENDENCIES` wires all three for real — no fallback-to-raw-
feature judgment call needed. `offered_load_mbps` and `ue_count` are the only genuinely raw D1
telemetry inputs from the 5-item input list; `latency`/`throughput`/`packet loss` are consumed as
upstream-model prediction columns.

Training vs. serving, same pattern as Modules 7/9: the orchestrator supplies each dependency's
live PREDICTION at inference time; training (bootstrap, outside the orchestrator) supplies the
strongest available signal instead — ground-truth historical values for all three, under the
same column names. `JitterModel` is agnostic to which; it only ever consumes columns by name.
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


class JitterModel(DTComponent):
    COMPONENT_NAME = "jitter"
    DEPENDENCIES = ("throughput", "latency", "packet_loss")
    REQUIRED_FEATURES = ("offered_load_mbps", "ue_count")
    OUTPUT_FIELD = "jitter_ms_pred"

    # Input column names carrying each dependency's value — match the corresponding upstream
    # component's OUTPUT_FIELD exactly. Populated by the orchestrator from live predictions at
    # inference time; callers training this component directly must supply them from ground
    # truth (see module docstring).
    THROUGHPUT_INPUT_COLUMN = "throughput_mbps_pred"
    LATENCY_INPUT_COLUMN = "latency_ms_pred"
    PACKET_LOSS_INPUT_COLUMN = "packet_loss_pct_pred"

    def __init__(self, model_type: str = "xgboost", params: dict[str, Any] | None = None) -> None:
        self._model_type = model_type
        self._params = dict(params or {})
        self._model = build_regressor(model_type, self._params)
        self._trained = False

    @classmethod
    def from_settings(cls, settings: "Settings") -> "JitterModel":
        spec = settings.dt_models.jitter
        return cls(model_type=spec.model_type, params=spec.params)

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def _input_columns(self) -> tuple[str, ...]:
        """REQUIRED_FEATURES (raw D1 telemetry) plus all three dependency columns — the full set
        this component actually fits/predicts on. Deliberately NOT the same as REQUIRED_FEATURES:
        the orchestrator checks only REQUIRED_FEATURES against the raw `features` frame, since
        the dependency columns are supplied separately, by wiring."""
        return self.REQUIRED_FEATURES + (
            self.THROUGHPUT_INPUT_COLUMN,
            self.LATENCY_INPUT_COLUMN,
            self.PACKET_LOSS_INPUT_COLUMN,
        )

    def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self._input_columns if c not in features.columns]
        if missing:
            raise ValueError(f"JitterModel missing required input column(s): {missing}")
        return features[list(self._input_columns)]

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._select_features(features)
        combined = pd.concat([X, targets.rename("__target__")], axis=1).dropna()
        if combined.empty:
            raise ValueError("JitterModel.train received no valid (non-NaN) rows")
        dropped = len(X) - len(combined)
        if dropped:
            logger.warning(
                "dropped rows with NaN before training",
                extra={"component": "jitter_model", "dropped_rows": dropped},
            )

        X_clean = combined[list(self._input_columns)]
        y_clean = combined["__target__"]
        self._model.fit(X_clean, y_clean)
        self._trained = True
        logger.info(
            "jitter model trained",
            extra={"component": "jitter_model", "model_type": self._model_type, "rows": len(X_clean)},
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._trained:
            raise RuntimeError("JitterModel.predict called before train()/load()")
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
