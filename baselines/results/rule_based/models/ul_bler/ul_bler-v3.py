import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import ClassVar
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.dt_models.base import DTComponent

class UlBlerModel(DTComponent):
    COMPONENT_NAME: ClassVar[str] = "ul_bler"
    DEPENDENCIES: ClassVar[tuple[str, ...]] = ()
    REQUIRED_FEATURES: ClassVar[tuple[str, ...]] = (
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
        "pusch_snr_avg_lag1",
        "pusch_snr_std_lag1",
        "pucch_snr_avg_lag1",
        "pucch_snr_std_lag1",
        "ul_mcs1_avg_lag1",
        "ul_mcs1_std_lag1",
        "ul_prb_lag1",
        "ul_throughput_lag1",
        "phr_avg_lag1",
    )
    OUTPUT_FIELD: ClassVar[str] = "ul_bler_pred"

    def __init__(self, max_iter: int = 100, learning_rate: float = 0.05, max_depth: int = 6):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42
        )
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        feat_cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        X = features[feat_cols].copy().fillna(0.0)
        y = targets.copy().fillna(0.0)
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet.")
        feat_cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        X = features[feat_cols].copy().fillna(0.0)
        preds = self.model.predict(X)
        preds = np.clip(preds, 0.0, 1.0)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        y_true = targets.fillna(0.0)
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        mae = float(mean_absolute_error(y_true, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "is_trained": self._is_trained}, path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        data = joblib.load(path)
        self.model = data["model"]
        self._is_trained = data["is_trained"]
