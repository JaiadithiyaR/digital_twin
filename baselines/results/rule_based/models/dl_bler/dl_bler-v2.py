import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
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
        "dl_bler_roll_mean_10s",
        "dl_bler_roll_std_5s",
        "dl_bler_roll_std_10s",
        "wb_cqi_avg_lag1",
        "wb_cqi_avg_lag2",
        "wb_cqi_avg_roll_mean_5s",
        "dl_mcs1_avg_lag1",
        "dl_mcs1_avg_lag2",
        "dl_mcs1_avg_roll_mean_5s",
        "dl_sched_rb_avg_lag1",
        "dl_sched_rb_avg_roll_mean_5s",
        "dl_curr_tbs_avg_lag1",
        "dl_curr_tbs_avg_roll_mean_5s",
        "pusch_snr_avg_lag1",
        "pucch_snr_avg_lag1",
        "dl_throughput_lag1",
        "dl_throughput_roll_mean_5s",
        "dl_prb_lag1",
        "dl_prb_roll_mean_5s",
    )

    def __init__(self):
        super().__init__()
        self.model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
        )
        self._is_trained = False
        self.feature_names_ = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _extract_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is not None:
            df = features.reindex(columns=self.feature_names_, fill_value=0.0)
            return df.fillna(0.0)
        cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        if not cols:
            numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            cols = [c for c in numeric_cols if c not in ["cell_id", "run_id", "session_id", "slice_id", "ts_1s", "ue_id"]]
        return features[cols].fillna(0.0)

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._extract_features(features)
        self.feature_names_ = list(X.columns)
        y = targets.to_numpy()
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError(f"Component {self.COMPONENT_NAME} is not trained yet.")
        X = self._extract_features(features)
        preds = self.model.predict(X)
        preds = np.clip(preds, 0.0, 1.0)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        y_true = targets.to_numpy()
        y_pred = preds.to_numpy()
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str) -> None:
        state = {
            "model": self.model,
            "is_trained": self._is_trained,
            "feature_names_": self.feature_names_,
        }
        joblib.dump(state, path)

    def load(self, path: str) -> None:
        state = joblib.load(path)
        self.model = state["model"]
        self._is_trained = state["is_trained"]
        self.feature_names_ = state["feature_names_"]
