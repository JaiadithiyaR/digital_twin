import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
        "ul_bler_avg_lag1",
        "ul_bler_std_lag1",
        "ul_bler_avg_lag2",
        "ul_bler_std_lag2",
        "ul_bler_avg_roll_mean_5s",
        "ul_bler_avg_roll_mean_10s",
        "ul_bler_std_roll_mean_5s",
        "pusch_snr_avg_lag1",
        "pusch_snr_avg_lag2",
        "pusch_snr_avg_roll_mean_5s",
        "pusch_snr_std_lag1",
        "pucch_snr_avg_lag1",
        "pucch_snr_avg_roll_mean_5s",
        "ul_mcs1_avg_lag1",
        "ul_mcs1_avg_lag2",
        "ul_mcs1_avg_roll_mean_5s",
        "ul_mcs1_std_lag1",
        "ul_throughput_lag1",
        "ul_throughput_roll_mean_5s",
        "ul_prb_lag1",
        "ul_prb_roll_mean_5s",
        "phr_avg_lag1",
        "phr_avg_roll_mean_5s",
        "bsr_avg_lag1",
        "requested_prb_ratio_lag1"
    )

    def __init__(self, model_type: str = "hist_gb", params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model_type = model_type
        self.params = params or {
            "max_iter": 250,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_samples_leaf": 15,
            "l2_regularization": 0.05,
            "random_state": 42
        }
        self._model = None
        self._is_trained = False

    @classmethod
    def from_settings(cls, settings: Any = None) -> "UlBlerModel":
        if settings is not None and hasattr(settings, "dt_models") and hasattr(settings.dt_models, "ul_bler"):
            spec = settings.dt_models.ul_bler
            model_type = getattr(spec, "model_type", "hist_gb")
            params = getattr(spec, "params", {})
            return cls(model_type=model_type, params=params)
        return cls()

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        if not cols:
            X = features.select_dtypes(include=[np.number]).fillna(0.0)
        else:
            X = features[cols].fillna(0.0)
        y = targets.fillna(0.0)

        model = HistGradientBoostingRegressor(**self.params)
        model.fit(X, y)
        self._model = model
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained or self._model is None:
            if "ul_bler_lag1" in features.columns:
                preds = features["ul_bler_lag1"].fillna(0.0)
            else:
                preds = pd.Series(0.0, index=features.index)
            return pd.Series(np.clip(preds.values, 0.0, 1.0), index=features.index, name=self.OUTPUT_FIELD)

        cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        if not cols:
            X = features.select_dtypes(include=[np.number]).fillna(0.0)
        else:
            X = features[cols].fillna(0.0)

        raw_preds = self._model.predict(X)
        clipped_preds = np.clip(raw_preds, 0.0, 1.0)
        return pd.Series(clipped_preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        y_true = targets.fillna(0.0)
        mse = mean_squared_error(y_true, preds)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "model": self._model,
            "is_trained": self._is_trained,
            "model_type": self.model_type,
            "params": self.params,
            "required_features": self.REQUIRED_FEATURES
        }
        joblib.dump(data, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self._model = data.get("model")
        self._is_trained = data.get("is_trained", False)
        self.model_type = data.get("model_type", self.model_type)
        self.params = data.get("params", self.params)
