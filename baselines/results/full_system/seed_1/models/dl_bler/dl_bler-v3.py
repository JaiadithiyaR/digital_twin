import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.dt_models.base import DTComponent

class DlBlerModel(DTComponent):
    COMPONENT_NAME = "dl_bler"
    DEPENDENCIES: Tuple[str, ...] = ()
    OUTPUT_FIELD = "dl_bler_pred"
    REQUIRED_FEATURES: Tuple[str, ...] = (
        "dl_bler_lag1",
        "dl_bler_lag2",
        "dl_bler_lag3",
        "dl_bler_lag4",
        "dl_bler_lag5",
        "dl_bler_roll_mean_5s",
        "dl_bler_roll_std_5s",
        "dl_bler_roll_mean_10s",
        "dl_bler_roll_std_10s",
        "dl_mcs1_avg_lag1",
        "dl_mcs1_avg_lag2",
        "wb_cqi_avg_lag1",
        "wb_cqi_avg_lag2",
        "dl_curr_tbs_avg_lag1",
        "dl_sched_rb_avg_lag1",
        "dl_prb_lag1",
        "dl_throughput_lag1",
        "pusch_snr_avg_lag1",
        "pucch_snr_avg_lag1"
    )

    def __init__(self, model: Optional[HistGradientBoostingRegressor] = None):
        self._model = model if model is not None else HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        self._is_trained = model is not None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @classmethod
    def from_settings(cls, settings: Any = None) -> "DlBlerModel":
        return cls()

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = features[list(self.REQUIRED_FEATURES)].copy().fillna(0.0)
        y = targets.iloc[:, 0] if isinstance(targets, pd.DataFrame) else targets
        y = y.fillna(0.0)
        self._model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError("Model is not trained.")
        X = features[list(self.REQUIRED_FEATURES)].copy().fillna(0.0)
        preds = self._model.predict(X)
        preds = np.clip(preds, 0.0, 1.0)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        y_true = targets.iloc[:, 0] if isinstance(targets, pd.DataFrame) else targets
        y_true = y_true.fillna(0.0)
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        mae = float(mean_absolute_error(y_true, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str) -> None:
        joblib.dump({"model": self._model, "is_trained": self._is_trained}, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self._model = data["model"]
        self._is_trained = data["is_trained"]
