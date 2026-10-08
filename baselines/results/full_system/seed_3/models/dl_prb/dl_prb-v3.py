import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.dt_models.base import DTComponent

class DlPrbModel(DTComponent):
    COMPONENT_NAME: str = "dl_prb"
    DEPENDENCIES: Tuple[str, ...] = ()
    OUTPUT_FIELD: str = "dl_prb_pred"
    
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
        "dl_curr_tbs_avg_lag1",
        "dl_sched_rb_avg_lag1",
        "wb_cqi_avg_lag1",
        "dl_mcs1_avg_lag1",
        "requested_prb_ratio_lag1",
        "ue_count_cell_lag1",
        "bsr_avg_lag1",
    )

    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = params or {
            "max_iter": 150,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "random_state": 42
        }
        self.model = HistGradientBoostingRegressor(**self.params)
        self._is_trained: bool = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @classmethod
    def from_settings(cls, settings: Any) -> "DlPrbModel":
        spec = getattr(getattr(settings, "dt_models", None), "dl_prb", None)
        params = getattr(spec, "params", {}) if spec else {}
        return cls(**params)

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = features[list(self.REQUIRED_FEATURES)].copy().fillna(0.0)
        y = targets.copy().fillna(0.0)
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet.")
        X = features[list(self.REQUIRED_FEATURES)].copy().fillna(0.0)
        preds = self.model.predict(X)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
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
            "required_features": self.REQUIRED_FEATURES,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(data, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self._is_trained = data["is_trained"]
        self.params = data.get("params", self.params)
