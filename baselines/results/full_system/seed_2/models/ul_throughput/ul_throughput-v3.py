import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.dt_models.base import DTComponent

class UlThroughputModel(DTComponent):
    COMPONENT_NAME = "ul_throughput"
    DEPENDENCIES = ("ul_bler", "ul_prb")
    OUTPUT_FIELD = "ul_throughput_pred"

    REQUIRED_FEATURES = (
        "ul_throughput_lag1",
        "ul_throughput_lag2",
        "ul_throughput_lag3",
        "ul_throughput_lag4",
        "ul_throughput_lag5",
        "ul_throughput_roll_mean_5s",
        "ul_throughput_roll_mean_10s",
        "ul_throughput_roll_std_5s",
        "ul_throughput_roll_std_10s",
        "ul_prb_lag1",
        "ul_prb_lag2",
        "ul_prb_roll_mean_5s",
        "ul_prb_roll_mean_10s",
        "ul_bler_lag1",
        "ul_bler_lag2",
        "pusch_snr_avg_lag1",
        "pusch_snr_std_lag1",
        "pucch_snr_avg_lag1",
        "ul_mcs1_avg_lag1",
        "phr_avg_lag1",
        "bsr_avg_lag1",
        "requested_prb_ratio_lag1",
        "ue_count_cell_lag1"
    )

    UL_BLER_INPUT_COLUMN = "ul_bler_pred"
    UL_PRB_INPUT_COLUMN = "ul_prb_pred"

    def __init__(self, model: Optional[Any] = None, params: Optional[Dict[str, Any]] = None):
        self.params = params or {
            "max_iter": 150,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_samples_leaf": 20,
            "l2_regularization": 0.1,
            "random_state": 42
        }
        self.model = model if model is not None else HistGradientBoostingRegressor(**self.params)
        self._is_trained = model is not None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _get_feature_names(self) -> List[str]:
        cols = list(self.REQUIRED_FEATURES)
        cols.extend([self.UL_BLER_INPUT_COLUMN, self.UL_PRB_INPUT_COLUMN])
        return cols

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        feature_cols = self._get_feature_names()
        df = features.copy()
        
        if self.UL_BLER_INPUT_COLUMN not in df.columns:
            if "ul_bler" in df.columns:
                df[self.UL_BLER_INPUT_COLUMN] = df["ul_bler"]
            elif "ul_bler_lag1" in df.columns:
                df[self.UL_BLER_INPUT_COLUMN] = df["ul_bler_lag1"]
            else:
                df[self.UL_BLER_INPUT_COLUMN] = 0.0

        if self.UL_PRB_INPUT_COLUMN not in df.columns:
            if "ul_prb" in df.columns:
                df[self.UL_PRB_INPUT_COLUMN] = df["ul_prb"]
            elif "ul_prb_lag1" in df.columns:
                df[self.UL_PRB_INPUT_COLUMN] = df["ul_prb_lag1"]
            else:
                df[self.UL_PRB_INPUT_COLUMN] = 0.0

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        return df[feature_cols].fillna(0.0)

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._prepare_features(features)
        y = targets.fillna(0.0)
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        X = self._prepare_features(features)
        preds = self.model.predict(X)
        preds = np.clip(preds, 0.0, None)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        y_true = targets.fillna(0.0)
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        mae = float(mean_absolute_error(y_true, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self.model,
            "params": self.params,
            "is_trained": self._is_trained,
        }
        joblib.dump(data, path)

    def load(self, path: Any) -> None:
        path = Path(path)
        data = joblib.load(path)
        self.model = data["model"]
        self.params = data.get("params", self.params)
        self._is_trained = data.get("is_trained", True)

    @classmethod
    def from_settings(cls, settings: Any) -> "UlThroughputModel":
        spec = getattr(settings.dt_models, cls.COMPONENT_NAME, None)
        if spec and hasattr(spec, "params") and spec.params:
            return cls(params=spec.params)
        return cls()
