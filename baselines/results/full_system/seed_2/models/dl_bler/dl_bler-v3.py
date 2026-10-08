import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.dt_models.base import DTComponent

class DlBlerModel(DTComponent):
    COMPONENT_NAME = "dl_bler"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES: tuple[str, ...] = (
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
        "pusch_snr_avg_lag1",
        "pucch_snr_avg_lag1",
        "dl_curr_tbs_avg_lag1",
        "dl_sched_rb_avg_lag1",
        "dl_prb_lag1",
        "ul_bler_lag1",
        "has_controller_lag1",
    )
    OUTPUT_FIELD = "dl_bler_pred"

    def __init__(self, max_iter: int = 150, learning_rate: float = 0.05, max_depth: int = 6):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model: Optional[HistGradientBoostingRegressor] = None
        self._is_trained: bool = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        if not cols:
            cols = [c for c in features.columns if c in self.REQUIRED_FEATURES or "dl_bler" in c]
        X = features[cols].copy()
        y = targets.copy()
        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42
        )
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model is not trained.")
        cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        X = features[cols].copy()
        preds = self.model.predict(X)
        preds = np.clip(preds, 0.0, 1.0)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        rmse = float(np.sqrt(mean_squared_error(targets, preds)))
        mae = float(mean_absolute_error(targets, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "is_trained": self._is_trained,
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
        }, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self._is_trained = data["is_trained"]
        self.max_iter = data.get("max_iter", 150)
        self.learning_rate = data.get("learning_rate", 0.05)
        self.max_depth = data.get("max_depth", 6)
