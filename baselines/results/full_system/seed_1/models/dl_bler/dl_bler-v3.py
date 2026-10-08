import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.dt_models.base import DTComponent


class DlBlerModel(DTComponent):
    COMPONENT_NAME: str = "dl_bler"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES: tuple[str, ...] = (
        "dl_bler_lag1",
        "dl_bler_lag2",
        "dl_bler_lag3",
        "dl_bler_lag4",
        "dl_bler_lag5",
        "dl_bler_roll_mean_5s",
        "dl_bler_roll_mean_10s",
        "dl_bler_roll_std_5s",
        "dl_bler_roll_std_10s",
        "dl_bler_avg_lag1",
        "dl_bler_avg_lag2",
        "dl_bler_std_lag1",
        "wb_cqi_avg_lag1",
        "wb_cqi_avg_roll_mean_5s",
        "dl_mcs1_avg_lag1",
        "dl_mcs1_avg_roll_mean_5s",
        "pusch_snr_avg_lag1",
        "pucch_snr_avg_lag1",
        "dl_curr_tbs_avg_lag1",
        "dl_sched_rb_avg_lag1",
        "dl_throughput_lag1",
        "dl_prb_lag1",
        "ul_bler_lag1",
    )
    OUTPUT_FIELD: str = "dl_bler_pred"

    def __init__(self, max_iter: int = 100, max_depth: int = 6, random_state: int = 42):
        self._model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            random_state=random_state,
            l2_regularization=0.1,
            min_samples_leaf=20,
        )
        self._is_trained: bool = False
        self._trained_cols: list[str] = []

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        if not cols:
            cols = [c for c in features.columns if "bler" in c or "lag" in c]
        
        self._trained_cols = cols
        X = features[cols].fillna(0.0)
        y = targets.fillna(0.0)
        
        self._model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained or self._model is None:
            return pd.Series(0.0, index=features.index, name=self.OUTPUT_FIELD)
        
        cols = getattr(self, "_trained_cols", [c for c in self.REQUIRED_FEATURES if c in features.columns])
        X = pd.DataFrame(index=features.index)
        for c in cols:
            if c in features.columns:
                X[c] = features[c].fillna(0.0)
            else:
                X[c] = 0.0
                
        preds = self._model.predict(X)
        preds = np.clip(preds, 0.0, 1.0)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        y_true = targets.fillna(0.0).values
        y_pred = preds.values
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_true)))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "is_trained": self._is_trained,
                "trained_cols": self._trained_cols,
            }, f)

    def load(self, path: Union[str, Path]) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
            self._model = state["model"]
            self._is_trained = state["is_trained"]
            self._trained_cols = state["trained_cols"]
