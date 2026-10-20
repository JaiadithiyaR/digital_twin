"""Shared base for all 7 OAI KPI components: each learns the one-second residual

    Δy = y_t − y_{t−1}

rather than the raw level, and `predict()` reconstructs

    ŷ_t = y_{t−1} + Δŷ

before returning — the public interface is unchanged. `y_{t−1}` is read from the component's own
`"{COMPONENT_NAME}_lag1"` feature column (already in `REQUIRED_FEATURES`), never re-derived.

Rationale: a residual model only explains the change, not the raw level — its worst case degrades
toward Δ=0, i.e. the persistence baseline, rather than doing worse than it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.dt_models.base import DTComponent
from src.dt_models.regressors import build_regressor

logger = logging.getLogger(__name__)


class ResidualDTComponent(DTComponent):
    """Subclasses set `COMPONENT_NAME`/`DEPENDENCIES`/`REQUIRED_FEATURES`/`OUTPUT_FIELD` as class
    attributes exactly as the original 7 component files already did, and override
    `_input_columns` only if they consume additional dependency-prediction columns beyond
    `REQUIRED_FEATURES` (e.g. `dl_throughput` needs `dl_bler_pred`/`dl_prb_pred` too)."""

    COMPONENT_NAME: ClassVar[str]
    DEPENDENCIES: ClassVar[tuple[str, ...]] = ()
    REQUIRED_FEATURES: ClassVar[tuple[str, ...]] = ()
    OUTPUT_FIELD: ClassVar[str]

    def __init__(self, model_type: str = "xgboost", params: dict[str, Any] | None = None) -> None:
        self._model_type = model_type
        self._params = dict(params or {})
        self._model = build_regressor(model_type, self._params)
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def _lag1_column(self) -> str:
        """`y_{t-1}` for this component's own target — the residual-reconstruction anchor."""
        return f"{self.COMPONENT_NAME}_lag1"

    @property
    def _input_columns(self) -> tuple[str, ...]:
        return tuple(self.REQUIRED_FEATURES)

    def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self._input_columns if c not in features.columns]
        if missing:
            raise ValueError(f"{type(self).__name__} missing required input column(s): {missing}")
        return features[list(self._input_columns)]

    def train(self, features: pd.DataFrame, targets: pd.Series, sample_weight: "np.ndarray | None" = None) -> None:
        """Fit on the residual `y_t - y_{t-1}`. Only rows whose residual is undefined (missing
        target or missing `y_{t-1}` anchor) are dropped: missing INPUT features are kept, because
        the regressor (XGBoost / scikit-learn RandomForest) handles them natively and `predict()`
        is served exactly such rows — dropping every row with any missing feature both starved
        training (e.g. the first ~3.5 min of a run, before the controller's first action, has no
        `requested_prb_ratio_lag1`) and made training inconsistent with serving.
        `sample_weight` (optional, one per row) is passed to the regressor's `fit`."""
        X = self._select_features(features)
        if self._lag1_column not in X.columns:
            raise ValueError(
                f"{type(self).__name__}.train requires its own lag1 column {self._lag1_column!r} "
                "(in REQUIRED_FEATURES) to compute the residual target — missing"
            )
        residual = (targets - X[self._lag1_column]).to_numpy(dtype=float)
        usable = np.isfinite(residual)
        if not usable.any():
            raise ValueError(f"{type(self).__name__}.train received no rows with a defined residual target (y_t and y_t-1)")
        dropped = int((~usable).sum())
        if dropped:
            logger.info(
                "skipped rows with undefined residual target before training",
                extra={"component": self.COMPONENT_NAME, "dropped_rows": dropped},
            )

        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            if len(weights) != len(X):
                raise ValueError(f"sample_weight has {len(weights)} entries for {len(X)} rows")
            fit_kwargs["sample_weight"] = weights[usable]
        X_clean = X.loc[usable, list(self._input_columns)]
        self._model.fit(X_clean, residual[usable], **fit_kwargs)
        self._trained = True
        logger.info(
            "model trained on residual target (y_t - y_t-1)",
            extra={
                "component": self.COMPONENT_NAME, "model_type": self._model_type, "rows": len(X_clean),
                "weighted": sample_weight is not None,
            },
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._trained:
            raise RuntimeError(f"{type(self).__name__}.predict called before train()/load()")
        X = self._select_features(features)
        residual_pred = self._model.predict(X)
        # Reconstruct ŷ_t = y_{t-1} + Δŷ; honestly NaN where y_{t-1} is unknown (first row of a
        # session) — XGBoost itself would still predict a residual despite the NaN, so this must
        # be applied explicitly here rather than left to the regressor.
        y_prev = X[self._lag1_column].to_numpy()
        y_pred = y_prev + residual_pred
        return pd.Series(y_pred, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        aligned_targets = targets.loc[preds.index]
        # Mask NaN predictions (first row of each session) rather than let them poison RMSE/MAE.
        mask = aligned_targets.notna() & preds.notna()
        if not mask.any():
            return {"rmse": float("nan"), "mae": float("nan")}
        rmse = float(mean_squared_error(aligned_targets[mask], preds[mask]) ** 0.5)
        mae = float(mean_absolute_error(aligned_targets[mask], preds[mask]))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "model_type": self._model_type, "params": self._params, "trained": self._trained}, path)

    def load(self, path: Path) -> None:
        state = joblib.load(path)
        self._model = state["model"]
        self._model_type = state["model_type"]
        self._params = state["params"]
        self._trained = state["trained"]
