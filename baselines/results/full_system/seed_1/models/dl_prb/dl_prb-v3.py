import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.dt_models.base import DTComponent

class DlPrbModel(DTComponent):
    COMPONENT_NAME = "dl_prb"
    DEPENDENCIES: Tuple[str, ...] = ()
    REQUIRED_FEATURES: Tuple[str, ...] = (
        "dl_prb_lag1",
        "dl_prb_lag2",
        "dl_prb_lag3",
        "dl_prb_lag4",
        "dl_prb_lag5",
        "dl_prb_roll_mean_5s",
        "dl_prb_roll_mean_10s",
        "dl_prb_roll_std_5s",
        "dl_prb_roll_std_10s",
        "dl_throughput_lag1",
        "dl_throughput_roll_mean_5s",
        "dl_throughput_roll_mean_10s",
        "dl_sched_rb_avg_lag1",
        "dl_sched_rb_avg_roll_mean_5s",
        "dl_curr_tbs_avg_lag1",
        "dl_curr_tbs_avg_roll_mean_5s",
        "dl_mcs1_avg_lag1",
        "wb_cqi_avg_lag1",
        "requested_prb_ratio_lag1",
        "ue_count_cell_lag1",
        "dl_bler_lag1",
        "ul_prb_lag1"
    )
    OUTPUT_FIELD = "dl_prb_pred"

    def __init__(self, max_iter: int = 150, learning_rate: float = 0.05, max_depth: int = 8):
        super().__init__()
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            min_samples_leaf=10
        )
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
            return pd.Series(0.0, index=features.index, name=self.OUTPUT_FIELD)
        X = features[list(self.REQUIRED_FEATURES)].copy().fillna(0.0)
        preds = self.model.predict(X)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        y_true = targets.fillna(0.0)
        mse = mean_squared_error(y_true, preds)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str) -> None:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        joblib.dump({"model": self.model, "_is_trained": self._is_trained}, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self._is_trained = data["_is_trained"]
