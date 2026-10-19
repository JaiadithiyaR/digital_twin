import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from sklearn.ensemble import HistGradientBoostingRegressor
from src.dt_models.base import DTComponent

class UlBlerModel(DTComponent):
    COMPONENT_NAME: str = "ul_bler"
    DEPENDENCIES: tuple[str, ...] = ()
    OUTPUT_FIELD: str = "ul_bler_pred"
    REQUIRED_FEATURES: tuple[str, ...] = (
        "ul_bler_lag1", "ul_bler_lag2", "ul_bler_lag3", "ul_bler_lag4", "ul_bler_lag5",
        "ul_bler_roll_mean_5s", "ul_bler_roll_mean_10s", "ul_bler_roll_std_5s", "ul_bler_roll_std_10s",
        "ul_bler_avg_lag1", "ul_bler_std_lag1",
        "pusch_snr_avg_lag1", "pusch_snr_lag1", "pusch_snr_std_lag1", "pusch_snr_avg_roll_mean_5s", "pusch_snr_avg_roll_std_5s",
        "pucch_snr_avg_lag1", "pucch_snr_std_lag1", "pucch_snr_avg_roll_mean_5s",
        "ul_mcs1_avg_lag1", "ul_mcs1_std_lag1", "ul_mcs1_avg_roll_mean_5s",
        "phr_avg_lag1", "phr_std_lag1", "phr_avg_roll_mean_5s",
        "ul_prb_lag1", "ul_prb_lag2", "ul_prb_roll_mean_5s", "ul_prb_roll_std_5s",
        "ul_throughput_lag1", "ul_throughput_roll_mean_5s",
        "dl_bler_lag1", "dl_bler_roll_mean_5s",
        "bsr_avg_lag1", "n_samples_lag1", "ue_count_cell_lag1", "ue_count_slice_lag1", "requested_prb_ratio_lag1"
    )

    def __init__(self):
        self._model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42
        )
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        if not cols:
            cols = list(features.columns)
        X = features[cols].fillna(0.0)
        y = targets.fillna(0.0)
        self._model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            return pd.Series(0.0, index=features.index, name=self.OUTPUT_FIELD)
        cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        if not cols:
            cols = list(features.columns)
        X = features[cols].fillna(0.0)
        preds = self._model.predict(X)
        preds = np.clip(preds, 0.0, 1.0)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        y_true = targets.values
        y_pred = preds.values
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Any) -> None:
        p = Path(path)
        if p.is_dir() or str(path).endswith("/") or not p.suffix:
            p.mkdir(parents=True, exist_ok=True)
            filepath = p / "model.joblib"
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            filepath = p
        state = {
            "model": self._model,
            "is_trained": self._is_trained
        }
        joblib.dump(state, filepath)

    def load(self, path: Any) -> None:
        p = Path(path)
        if p.is_dir() or str(path).endswith("/") or not p.suffix:
            filepath = p / "model.joblib"
        else:
            filepath = p
        state = joblib.load(filepath)
        self._model = state["model"]
        self._is_trained = state["is_trained"]
