import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.dt_models.base import DTComponent

class UlThroughputModel(DTComponent):
    COMPONENT_NAME = "ul_throughput"
    DEPENDENCIES = ("ul_bler", "ul_prb")
    REQUIRED_FEATURES = (
        "ul_throughput_lag1",
        "ul_throughput_lag2",
        "ul_throughput_lag3",
        "ul_throughput_roll_mean_5s",
        "ul_throughput_roll_std_5s",
        "ul_prb_lag1",
        "ul_prb_roll_mean_5s",
        "ul_bler_lag1",
        "ul_bler_roll_mean_5s",
        "ul_mcs1_avg_lag1",
        "pusch_snr_avg_lag1",
        "phr_avg_lag1",
        "bsr_avg_lag1",
        "requested_prb_ratio_lag1"
    )
    OUTPUT_FIELD = "ul_throughput_pred"

    UL_BLER_INPUT_COLUMN = "ul_bler_pred"
    UL_PRB_INPUT_COLUMN = "ul_prb_pred"

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model_params = model_params or {
            "max_iter": 150,
            "learning_rate": 0.05,
            "max_depth": 6,
            "random_state": 42
        }
        self.model = HistGradientBoostingRegressor(**self.model_params)
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def _input_columns(self) -> Tuple[str, ...]:
        cols = list(self.REQUIRED_FEATURES)
        if self.UL_BLER_INPUT_COLUMN not in cols:
            cols.append(self.UL_BLER_INPUT_COLUMN)
        if self.UL_PRB_INPUT_COLUMN not in cols:
            cols.append(self.UL_PRB_INPUT_COLUMN)
        return tuple(cols)

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=features.index)
        for col in self.REQUIRED_FEATURES:
            if col in features.columns:
                X[col] = features[col]
            else:
                X[col] = 0.0
        
        if self.UL_BLER_INPUT_COLUMN in features.columns:
            X[self.UL_BLER_INPUT_COLUMN] = features[self.UL_BLER_INPUT_COLUMN]
        elif "ul_bler" in features.columns:
            X[self.UL_BLER_INPUT_COLUMN] = features["ul_bler"]
        else:
            X[self.UL_BLER_INPUT_COLUMN] = 0.0
            
        if self.UL_PRB_INPUT_COLUMN in features.columns:
            X[self.UL_PRB_INPUT_COLUMN] = features[self.UL_PRB_INPUT_COLUMN]
        elif "ul_prb" in features.columns:
            X[self.UL_PRB_INPUT_COLUMN] = features["ul_prb"]
        else:
            X[self.UL_PRB_INPUT_COLUMN] = 0.0
            
        return X

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._prepare_features(features)
        y = targets.values if isinstance(targets, pd.Series) else targets
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise ValueError("Model is not trained yet.")
        X = self._prepare_features(features)
        preds = self.model.predict(X)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        y_true = targets.values if isinstance(targets, pd.Series) else targets
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        mae = float(mean_absolute_error(y_true, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "params": self.model_params, "is_trained": self._is_trained}, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.model_params = data.get("params", self.model_params)
        self._is_trained = data.get("is_trained", True)
