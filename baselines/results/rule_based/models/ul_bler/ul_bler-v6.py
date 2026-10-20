import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.dt_models.base import DTComponent

class UlBlerModel(DTComponent):
    COMPONENT_NAME = "ul_bler"
    DEPENDENCIES: Tuple[str, ...] = ()
    REQUIRED_FEATURES: Tuple[str, ...] = (
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
        "ul_mcs1_avg_lag1",
        "ul_prb_lag1",
        "ul_throughput_lag1",
        "phr_avg_lag1",
    )
    OUTPUT_FIELD = "ul_bler_pred"

    def __init__(self, model_params: Dict[str, Any] = None):
        super().__init__()
        params = {
            "max_iter": 200,
            "learning_rate": 0.03,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "random_state": 42,
        }
        if model_params:
            params.update(model_params)
        self.model = HistGradientBoostingRegressor(**params)
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = features[list(self.REQUIRED_FEATURES)].copy().fillna(0.0)
        y = targets.copy().fillna(0.0)
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError("UlBlerModel is not trained yet.")
        X = features[list(self.REQUIRED_FEATURES)].copy().fillna(0.0)
        raw_preds = self.model.predict(X)
        clipped_preds = np.clip(raw_preds, 0.0, 1.0)
        return pd.Series(clipped_preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        y_true = targets.fillna(0.0)
        mse = mean_squared_error(y_true, preds)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "is_trained": self._is_trained}, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self._is_trained = data["is_trained"]
