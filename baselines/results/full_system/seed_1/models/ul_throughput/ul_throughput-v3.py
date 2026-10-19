import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
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
        "ul_prb_lag3",
        "ul_prb_lag4",
        "ul_prb_lag5",
        "ul_prb_roll_mean_5s",
        "ul_prb_roll_mean_10s",
        "ul_prb_roll_std_5s",
        "ul_prb_roll_std_10s",
        "ul_bler_lag1",
        "ul_bler_lag2",
        "ul_bler_lag3",
        "ul_bler_lag4",
        "ul_bler_lag5",
        "ul_bler_roll_mean_5s",
        "ul_bler_roll_mean_10s",
        "ul_bler_roll_std_5s",
        "ul_bler_roll_std_10s",
        "pusch_snr_avg_lag1",
        "pusch_snr_avg_lag2",
        "pusch_snr_avg_lag3",
        "pusch_snr_avg_roll_mean_5s",
        "pusch_snr_avg_roll_mean_10s",
        "pucch_snr_avg_lag1",
        "pucch_snr_avg_lag2",
        "ul_mcs1_avg_lag1",
        "ul_mcs1_avg_lag2",
        "ul_mcs1_avg_roll_mean_5s",
        "phr_avg_lag1",
        "phr_avg_lag2",
        "bsr_avg_lag1",
        "n_samples_lag1",
        "ue_count_cell_lag1",
        "ue_count_slice_lag1",
        "requested_prb_ratio_lag1",
        "has_controller_lag1",
        "ul_bler_pred",
        "ul_prb_pred",
    )

    def __init__(self, max_iter: int = 150, learning_rate: float = 0.05, max_depth: int = 6):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model: Optional[HistGradientBoostingRegressor] = None
        self._feature_cols: list[str] = []

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def _resolve_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = []
        for feat in self.REQUIRED_FEATURES:
            if feat in df.columns:
                cols.append(feat)
            elif feat == "ul_bler_pred" and "ul_bler" in df.columns:
                cols.append("ul_bler")
            elif feat == "ul_prb_pred" and "ul_prb" in df.columns:
                cols.append("ul_prb")
        if not self._feature_cols:
            self._feature_cols = cols
        
        # Fill missing expected columns if necessary
        res = pd.DataFrame(index=df.index)
        for c in self._feature_cols:
            if c in df.columns:
                res[c] = df[c]
            elif c == "ul_bler_pred" and "ul_bler" in df.columns:
                res[c] = df["ul_bler"]
            elif c == "ul_prb_pred" and "ul_prb" in df.columns:
                res[c] = df["ul_prb"]
            else:
                res[c] = 0.0
        return res

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._resolve_features(features)
        y = targets.to_numpy().ravel()
        model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
        )
        model.fit(X, y)
        self.model = model

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before calling predict.")
        X = self._resolve_features(features)
        preds = self.model.predict(X)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        y_true = targets.to_numpy().ravel()
        y_pred = preds.to_numpy().ravel()
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Any) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_cols": self._feature_cols,
                "params": {
                    "max_iter": self.max_iter,
                    "learning_rate": self.learning_rate,
                    "max_depth": self.max_depth,
                },
            },
            path_obj,
        )

    def load(self, path: Any) -> None:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"No model file found at {path_obj}")
        data = joblib.load(path_obj)
        self.model = data.get("model")
        self._feature_cols = data.get("feature_cols", [])
        params = data.get("params", {})
        self.max_iter = params.get("max_iter", self.max_iter)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.max_depth = params.get("max_depth", self.max_depth)

    @classmethod
    def from_settings(cls, settings: Any) -> "UlThroughputModel":
        return cls()
