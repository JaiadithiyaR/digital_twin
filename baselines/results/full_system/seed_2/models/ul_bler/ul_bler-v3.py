import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

from src.dt_models.base import DTComponent

class UlBlerModel(DTComponent):
    COMPONENT_NAME = "ul_bler"
    DEPENDENCIES: tuple[str, ...] = ()
    OUTPUT_FIELD = "ul_bler_pred"
    REQUIRED_FEATURES: tuple[str, ...] = (
        "ul_bler_lag1",
        "ul_bler_lag2",
        "ul_bler_lag3",
        "ul_bler_lag4",
        "ul_bler_lag5",
        "ul_bler_roll_mean_5s",
        "ul_bler_roll_mean_10s",
        "ul_bler_roll_std_5s",
        "ul_bler_roll_std_10s",
        "ul_mcs1_avg_lag1",
        "ul_mcs1_std_lag1",
        "pusch_snr_avg_lag1",
        "pusch_snr_std_lag1",
        "pucch_snr_avg_lag1",
        "pucch_snr_std_lag1",
        "phr_avg_lag1",
        "phr_std_lag1",
        "ul_prb_lag1",
        "ul_throughput_lag1",
        "bsr_avg_lag1",
        "n_samples_lag1",
        "ue_count_cell_lag1",
        "requested_prb_ratio_lag1"
    )

    def __init__(self, model_type: str = "hist_gb", params: Optional[dict] = None):
        self.model_type = model_type
        self.params = params or {"max_iter": 150, "learning_rate": 0.05, "max_depth": 6, "random_state": 42}
        self.model = HistGradientBoostingRegressor(**self.params)
        self._is_trained = False

    @classmethod
    def from_settings(cls, settings: Any) -> "UlBlerModel":
        try:
            spec = settings.dt_models.ul_bler
            return cls(model_type=getattr(spec, "model_type", "hist_gb"), params=getattr(spec, "params", None))
        except Exception:
            return cls()

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        if not cols:
            cols = features.select_dtypes(include=[np.number]).columns.tolist()
        df = features[cols].copy()
        df = df.fillna(0.0)
        return df

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._prepare_features(features)
        y = targets.fillna(0.0)
        y = np.clip(y, 0.0, 1.0)
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            return pd.Series(0.0, index=features.index, name=self.OUTPUT_FIELD)
        X = self._prepare_features(features)
        preds = self.model.predict(X)
        preds = np.clip(preds, 0.0, 1.0)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        y_true = targets.fillna(0.0)
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        mae = float(mean_absolute_error(y_true, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str) -> None:
        data = {
            "model": self.model,
            "is_trained": self._is_trained,
            "params": self.params,
            "model_type": self.model_type,
        }
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(data, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self._is_trained = data["is_trained"]
        self.params = data.get("params", self.params)
        self.model_type = data.get("model_type", self.model_type)
