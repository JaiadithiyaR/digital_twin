"""Module 6 — Throughput Model (prompt.md §12.1, fig-dataflow.png node 6).

Inputs: offered load, PRB utilization, SINR, RSRP, RSRQ, UE count, UE speed (all raw telemetry
columns, read directly from D1 — column names match `d1_model_store.TELEMETRY_COLUMNS` exactly).
Output: throughput (Mbps).

A root node in the DT dependency graph (prompt.md §13) — no upstream DT predictions required.
Modules 7 (latency) and 10 (jitter) will later depend on this component's `OUTPUT_FIELD`.

The underlying regressor (XGBoost or Random Forest) is selected purely by
`config.dt_models.throughput.model_type`/`params` via `src/dt_models/regressors.py` — never
hardcoded here.
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


class ThroughputModel(DTComponent):
    COMPONENT_NAME = "throughput"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES = (
        "offered_load_mbps",
        "prb_utilization_pct",
        "sinr_db",
        "rsrp_dbm",
        "rsrq_db",
        "ue_count",
        "ue_speed_mps",
    )
    # Deliberately distinct from D1's ground-truth "throughput_mbps" column so a downstream
    # component (future latency/jitter) can never confuse a prediction for the raw telemetry it
    # was trained against.
    OUTPUT_FIELD = "throughput_mbps_pred"

    def __init__(self, model_type: str = "xgboost", params: dict[str, Any] | None = None) -> None:
        self._model_type = model_type
        self._params = dict(params or {})
        self._model = build_regressor(model_type, self._params)
        self._trained = False

    @classmethod
    def from_settings(cls, settings: "Settings") -> "ThroughputModel":
        spec = settings.dt_models.throughput
        return cls(model_type=spec.model_type, params=spec.params)

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.REQUIRED_FEATURES if f not in features.columns]
        if missing:
            raise ValueError(f"ThroughputModel missing required feature(s): {missing}")
        return features[list(self.REQUIRED_FEATURES)]

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._select_features(features)
        # Defensive: a NaN slipping through upstream validation must never silently corrupt
        # training — drop such rows explicitly and log it, rather than letting the underlying
        # regressor either error opaquely or silently propagate NaN through its fit.
        combined = pd.concat([X, targets.rename("__target__")], axis=1).dropna()
        if combined.empty:
            raise ValueError("ThroughputModel.train received no valid (non-NaN) rows")
        dropped = len(X) - len(combined)
        if dropped:
            logger.warning(
                "dropped rows with NaN before training",
                extra={"component": "throughput_model", "dropped_rows": dropped},
            )

        X_clean = combined[list(self.REQUIRED_FEATURES)]
        y_clean = combined["__target__"]
        self._model.fit(X_clean, y_clean)
        self._trained = True
        logger.info(
            "throughput model trained",
            extra={"component": "throughput_model", "model_type": self._model_type, "rows": len(X_clean)},
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._trained:
            raise RuntimeError("ThroughputModel.predict called before train()/load()")
        X = self._select_features(features)
        preds = self._model.predict(X)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        """Component-local sanity metrics — RMSE/MAE. NOT the authoritative fidelity score
        (that formula belongs to Module 12's fidelity engine, deterministically, over a rolling
        normalization window; this is a plain held-out regression metric)."""
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
