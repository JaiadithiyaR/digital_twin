import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.dt_models.base import DTComponent


class DlBlerModel(DTComponent):
    COMPONENT_NAME = "dl_bler"
    DEPENDENCIES: Tuple[str, ...] = ()
    OUTPUT_FIELD = "dl_bler_pred"
    REQUIRED_FEATURES: Tuple[str, ...] = (
        "dl_bler_lag1",
        "dl_bler_lag2",
        "dl_bler_lag3",
        "dl_bler_lag4",
        "dl_bler_lag5",
        "dl_bler_roll_mean_5s",
        "dl_bler_roll_mean_10s",
        "dl_bler_roll_std_5s",
        "dl_bler_roll_std_10s",
        "dl_mcs1_avg_lag1",
        "dl_mcs1_std_lag1",
        "wb_cqi_avg_lag1",
        "wb_cqi_std_lag1",
        "dl_curr_tbs_avg_lag1",
        "dl_sched_rb_avg_lag1",
        "dl_throughput_lag1",
        "dl_prb_lag1",
        "ul_bler_lag1",
    )

    def __init__(self, max_iter: int = 100, max_depth: int = 6, l2_regularization: float = 0.1) -> None:
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.l2_regularization = l2_regularization
        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            l2_regularization=self.l2_regularization,
            random_state=42,
        )
        self._is_trained = False
        self.feature_names = list(self.REQUIRED_FEATURES)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @classmethod
    def from_settings(cls, settings: Any) -> "DlBlerModel":
        try:
            spec = settings.dt_models.dl_bler
            params = getattr(spec, "params", {})
            return cls(
                max_iter=params.get("max_iter", 100),
                max_depth=params.get("max_depth", 6),
                l2_regularization=params.get("l2_regularization", 0.1),
            )
        except Exception:
            return cls()

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        available_features = [f for f in self.feature_names if f in features.columns]
        if not available_features:
            available_features = [c for c in features.columns if c not in ("cell_id", "run_id", "session_id", "slice_id", "ts_1s", "ue_id", "dl_bler")]
        self.feature_names = available_features
        X = features[self.feature_names].fillna(0.0)
        y = targets.fillna(0.0)
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            return pd.Series(0.0, index=features.index, name=self.OUTPUT_FIELD)
        available_features = [f for f in self.feature_names if f in features.columns]
        X = features[available_features].fillna(0.0)
        if X.shape[1] < len(self.feature_names):
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self.feature_names]
        preds = self.model.predict(X)
        preds = np.clip(preds, 0.0, 1.0)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        y_true = targets.fillna(0.0)
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        mae = float(mean_absolute_error(y_true, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "is_trained": self._is_trained,
                "feature_names": self.feature_names,
                "params": {
                    "max_iter": self.max_iter,
                    "max_depth": self.max_depth,
                    "l2_regularization": self.l2_regularization,
                },
            },
            path,
        )

    def load(self, path: Union[str, Path]) -> None:
        data = joblib.load(path)
        if isinstance(data, dict):
            self.model = data.get("model", self.model)
            self._is_trained = data.get("is_trained", True)
            self.feature_names = data.get("feature_names", list(self.REQUIRED_FEATURES))
        else:
            self.model = getattr(data, "model", self.model)
            self._is_trained = getattr(data, "_is_trained", True)
            self.feature_names = getattr(data, "feature_names", list(self.REQUIRED_FEATURES))
