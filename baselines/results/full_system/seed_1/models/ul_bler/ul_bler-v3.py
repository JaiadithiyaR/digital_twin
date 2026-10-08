import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.dt_models.base import DTComponent


class UlBlerModel(DTComponent):
    """Gradient-boosted decision tree prediction model for Uplink Block Error Rate (UL BLER) using scikit-learn."""

    COMPONENT_NAME: str = "ul_bler"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES: tuple[str, ...] = (
        "ul_bler_lag1",
        "ul_bler_lag2",
        "ul_bler_lag3",
        "ul_bler_lag4",
        "ul_bler_lag5",
        "ul_bler_roll_mean_5s",
        "ul_bler_roll_std_5s",
        "ul_bler_roll_mean_10s",
        "ul_bler_roll_std_10s",
        "pusch_snr_avg_lag1",
        "pusch_snr_avg_lag2",
        "pusch_snr_std_lag1",
        "pusch_snr_avg_roll_mean_5s",
        "pusch_snr_avg_roll_mean_10s",
        "pucch_snr_avg_lag1",
        "ul_mcs1_avg_lag1",
        "ul_mcs1_std_lag1",
        "ul_prb_lag1",
        "ul_throughput_lag1",
        "phr_avg_lag1",
        "ul_bler_avg_lag1",
        "ul_bler_std_lag1",
        "bsr_avg_lag1",
        "requested_prb_ratio_lag1",
    )
    OUTPUT_FIELD: str = "ul_bler_pred"

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._model = None
        self._is_trained = False
        self._feature_cols = list(self.REQUIRED_FEATURES)
        self.model_params = model_params or {
            "max_iter": 100,
            "learning_rate": 0.05,
            "max_depth": 5,
            "random_state": 42,
        }

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        if not cols:
            cols = [c for c in features.columns if c not in ["cell_id", "run_id", "session_id", "slice_id", "ts_1s", "ue_id"]]
        
        X = features[cols].fillna(0.0)
        y = targets.fillna(0.0)
        
        self._model = HistGradientBoostingRegressor(**self.model_params)
        self._model.fit(X, y)
        self._feature_cols = cols
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model is not trained yet.")
        
        cols = getattr(self, "_feature_cols", [c for c in self.REQUIRED_FEATURES if c in features.columns])
        X = features[cols].fillna(0.0)
        
        raw_preds = self._model.predict(X)
        clipped_preds = np.clip(raw_preds, 0.0, 1.0)
        return pd.Series(clipped_preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        y_true = targets.fillna(0.0)
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        mae = float(mean_absolute_error(y_true, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str) -> None:
        state = {
            "model": self._model,
            "is_trained": self._is_trained,
            "feature_cols": getattr(self, "_feature_cols", list(self.REQUIRED_FEATURES)),
            "model_params": self.model_params,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(state, path)

    def load(self, path: str) -> None:
        state = joblib.load(path)
        self._model = state["model"]
        self._is_trained = state["is_trained"]
        self._feature_cols = state.get("feature_cols", list(self.REQUIRED_FEATURES))
        self.model_params = state.get("model_params", self.model_params)
