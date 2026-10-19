import os
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.dt_models.base import DTComponent


class ChannelQualityModel(DTComponent):
    COMPONENT_NAME = "channel_quality"
    DEPENDENCIES = []
    REQUIRED_FEATURES = [
        "wb_cqi_avg_lag1",
        "wb_cqi_avg_lag2",
        "wb_cqi_avg_lag3",
        "wb_cqi_avg_roll_mean_5s",
        "wb_cqi_avg_roll_mean_10s",
        "dl_bler_lag1",
        "dl_bler_lag2",
        "dl_throughput_lag1",
        "dl_throughput_lag2",
        "pusch_snr_avg_lag1",
    ]
    OUTPUT_FIELD = "wb_cqi_avg_lag1_pred"

    def __init__(self, model_type: str = "hist_gb", params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model_type = model_type
        self.params = params or {"max_iter": 100, "random_state": 42}
        self.model = HistGradientBoostingRegressor(**self.params)
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = features[list(self.REQUIRED_FEATURES)].replace([np.inf, -np.inf], np.nan)
        y = targets.replace([np.inf, -np.inf], np.nan)
        mask = y.notna()
        if mask.sum() == 0:
            y = y.fillna(0.0)
            mask = pd.Series(True, index=y.index)
        X_train = X.loc[mask]
        y_train = y.loc[mask]
        self.model.fit(X_train, y_train)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise ValueError("Model is not trained yet.")
        X = features[list(self.REQUIRED_FEATURES)].replace([np.inf, -np.inf], np.nan)
        preds = self.model.predict(X)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        y = targets.replace([np.inf, -np.inf], np.nan)
        mask = y.notna()
        if mask.sum() == 0:
            return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "r2": 0.0}
        preds = self.predict(features.loc[mask])
        y_clean = y.loc[mask]
        mae = mean_absolute_error(y_clean, preds)
        mse = mean_squared_error(y_clean, preds)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_clean, preds)
        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": rmse,
            "r2": float(r2),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "model": self.model,
            "params": self.params,
            "model_type": self.model_type,
            "is_trained": self._is_trained,
        }
        joblib.dump(data, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.params = data["params"]
        self.model_type = data["model_type"]
        self._is_trained = data["is_trained"]

    @classmethod
    def from_settings(cls, settings: Any) -> "ChannelQualityModel":
        if hasattr(settings, "dt_models") and hasattr(settings.dt_models, "channel_quality"):
            spec = settings.dt_models.channel_quality
            return cls(model_type=getattr(spec, "model_type", "hist_gb"), params=getattr(spec, "params", None))
        return cls()
