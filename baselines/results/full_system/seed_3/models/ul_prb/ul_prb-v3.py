import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.ensemble import HistGradientBoostingRegressor
from src.dt_models.base import DTComponent

class UlPrbModel(DTComponent):
    COMPONENT_NAME = "ul_prb"
    DEPENDENCIES: Tuple[str, ...] = ()
    REQUIRED_FEATURES: Tuple[str, ...] = (
        "ul_prb_lag1",
        "ul_prb_lag2",
        "ul_prb_lag3",
        "ul_prb_lag4",
        "ul_prb_lag5",
        "ul_prb_roll_mean_5s",
        "ul_prb_roll_std_5s",
        "ul_prb_roll_mean_10s",
        "ul_prb_roll_std_10s",
        "ul_throughput_lag1",
        "ul_throughput_lag2",
        "ul_throughput_roll_mean_5s",
        "ul_throughput_roll_std_5s",
        "ul_bler_lag1",
        "ul_bler_roll_mean_5s",
        "ul_mcs1_avg_lag1",
        "ul_mcs1_avg_roll_mean_5s",
        "pusch_snr_avg_lag1",
        "pusch_snr_avg_roll_mean_5s",
        "bsr_avg_lag1",
        "requested_prb_ratio_lag1",
        "ue_count_cell_lag1",
    )
    OUTPUT_FIELD = "ul_prb_pred"

    def __init__(self, max_iter: int = 200, learning_rate: float = 0.05, max_depth: int = 6):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
        )
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = features[list(self.REQUIRED_FEATURES)].fillna(0.0)
        y = targets.fillna(0.0)
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError("UlPrbModel is not trained.")
        X = features[list(self.REQUIRED_FEATURES)].fillna(0.0)
        preds = self.model.predict(X)
        preds = np.clip(preds, 0.0, None)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        y_true = targets.fillna(0.0)
        mae = float(np.mean(np.abs(preds - y_true)))
        rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))
        return {"mae": mae, "rmse": rmse}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "is_trained": self._is_trained,
                "max_iter": self.max_iter,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
            },
            path,
        )

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self._is_trained = data["is_trained"]
        self.max_iter = data.get("max_iter", 200)
        self.learning_rate = data.get("learning_rate", 0.05)
        self.max_depth = data.get("max_depth", 6)
