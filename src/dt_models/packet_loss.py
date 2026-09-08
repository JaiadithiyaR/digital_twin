"""Module 8 — Packet-Loss Model (prompt.md §12.2, fig-dataflow.png node 8).

Inputs: SINR, RSRP, RSRQ, PRB utilization, UE count, offered load.
Output: packet loss (%).

A root node in the DT dependency graph (prompt.md §13), same as Module 6 (Throughput) — no
upstream DT predictions required, only raw telemetry features (read from D1). Module 7 (Latency)
will eventually depend on this component's `OUTPUT_FIELD` too (currently it consumes
`packet_loss_pct` as a raw D1 feature instead, since this component didn't exist yet when Module
7 was built — see `src/dt_models/latency.py`'s module docstring for the flagged follow-up to
wire that dependency now that Module 8 exists; not done in this file/turn, since that is a
Module 7 change, not a Module 8 one).
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


class PacketLossModel(DTComponent):
    COMPONENT_NAME = "packet_loss"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES = (
        "sinr_db",
        "rsrp_dbm",
        "rsrq_db",
        "prb_utilization_pct",
        "ue_count",
        "offered_load_mbps",
    )
    # Deliberately distinct from D1's ground-truth "packet_loss_pct" column so a downstream
    # component (future latency dependency, jitter) can never confuse a prediction for the raw
    # telemetry it was trained against.
    OUTPUT_FIELD = "packet_loss_pct_pred"

    def __init__(self, model_type: str = "xgboost", params: dict[str, Any] | None = None) -> None:
        self._model_type = model_type
        self._params = dict(params or {})
        self._model = build_regressor(model_type, self._params)
        self._trained = False

    @classmethod
    def from_settings(cls, settings: "Settings") -> "PacketLossModel":
        spec = settings.dt_models.packet_loss
        return cls(model_type=spec.model_type, params=spec.params)

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.REQUIRED_FEATURES if f not in features.columns]
        if missing:
            raise ValueError(f"PacketLossModel missing required feature(s): {missing}")
        return features[list(self.REQUIRED_FEATURES)]

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._select_features(features)
        combined = pd.concat([X, targets.rename("__target__")], axis=1).dropna()
        if combined.empty:
            raise ValueError("PacketLossModel.train received no valid (non-NaN) rows")
        dropped = len(X) - len(combined)
        if dropped:
            logger.warning(
                "dropped rows with NaN before training",
                extra={"component": "packet_loss_model", "dropped_rows": dropped},
            )

        X_clean = combined[list(self.REQUIRED_FEATURES)]
        y_clean = combined["__target__"]
        self._model.fit(X_clean, y_clean)
        self._trained = True
        logger.info(
            "packet-loss model trained",
            extra={"component": "packet_loss_model", "model_type": self._model_type, "rows": len(X_clean)},
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._trained:
            raise RuntimeError("PacketLossModel.predict called before train()/load()")
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
