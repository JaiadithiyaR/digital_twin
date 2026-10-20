import os
from pathlib import Path
from typing import Dict, Any, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.dt_models.base import DTComponent


class DlRlcDelayModel(DTComponent):
    """Digital Twin prediction component for Downlink RLC Delay (dl_rlc_delay)."""

    COMPONENT_NAME = "dl_rlc_delay"
    DEPENDENCIES = ("dl_throughput", "dl_bler")
    OUTPUT_FIELD = "dl_rlc_delay_pred"

    REQUIRED_FEATURES: Tuple[str, ...] = (
        "dl_rlc_delay_lag1",
        "dl_rlc_delay_lag2",
        "dl_rlc_delay_lag3",
        "dl_rlc_delay_lag4",
        "dl_rlc_delay_lag5",
        "dl_rlc_delay_roll_mean_5s",
        "dl_rlc_delay_roll_mean_10s",
        "dl_rlc_delay_roll_std_5s",
        "dl_rlc_delay_roll_std_10s",
        "dl_throughput_lag1",
        "dl_throughput_lag2",
        "dl_throughput_lag3",
        "dl_throughput_roll_mean_5s",
        "dl_throughput_roll_mean_10s",
        "dl_bler_lag1",
        "dl_bler_lag2",
        "dl_bler_lag3",
        "dl_bler_roll_mean_5s",
        "dl_bler_roll_mean_10s",
        "dl_prb_lag1",
        "dl_prb_lag2",
        "dl_prb_roll_mean_5s",
        "dl_prb_roll_mean_10s",
        "bsr_avg_lag1",
        "dl_sched_rb_avg_lag1",
        "dl_curr_tbs_avg_lag1",
        "wb_cqi_avg_lag1",
        "dl_mcs1_avg_lag1",
        "ue_count_cell_lag1",
    )

    def __init__(self, model_params: Dict[str, Any] = None) -> None:
        super().__init__()
        self.model_params = model_params or {
            "loss": "squared_error",
            "learning_rate": 0.05,
            "max_iter": 250,
            "max_depth": 6,
            "min_samples_leaf": 15,
            "l2_regularization": 0.1,
            "random_state": 42,
        }
        self.model: HistGradientBoostingRegressor = None
        self.feature_names_: list[str] = []

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def _prepare_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for col in self.REQUIRED_FEATURES:
            if col in df.columns:
                X[col] = df[col]
            else:
                X[col] = np.nan

        if "dl_throughput_pred" in df.columns:
            X["dl_throughput_pred"] = df["dl_throughput_pred"]
        else:
            X["dl_throughput_pred"] = df.get("dl_throughput_lag1", np.nan)

        if "dl_bler_pred" in df.columns:
            X["dl_bler_pred"] = df["dl_bler_pred"]
        else:
            X["dl_bler_pred"] = df.get("dl_bler_lag1", np.nan)

        if "dl_rlc_delay_lag1" in df.columns and "dl_rlc_delay_lag2" in df.columns:
            X["dl_rlc_delay_diff1"] = df["dl_rlc_delay_lag1"] - df["dl_rlc_delay_lag2"]
        else:
            X["dl_rlc_delay_diff1"] = np.nan

        return X

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._prepare_feature_matrix(features)
        y = targets.copy()

        valid_idx = ~y.isna()
        X_clean = X.loc[valid_idx]
        y_clean = y.loc[valid_idx]

        self.feature_names_ = list(X_clean.columns)
        self.model = HistGradientBoostingRegressor(**self.model_params)
        self.model.fit(X_clean, y_clean)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict().")
        X = self._prepare_feature_matrix(features)
        X = X[self.feature_names_]
        preds = self.model.predict(X)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        valid_idx = ~targets.isna()
        y_true = targets[valid_idx]
        y_pred = preds[valid_idx]
        if len(y_true) == 0:
            return {"rmse": 0.0, "mae": 0.0}
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "model_params": self.model_params,
            "feature_names_": self.feature_names_,
        }
        joblib.dump(state, path)

    def load(self, path: Union[str, Path]) -> None:
        path = Path(path)
        state = joblib.load(path)
        self.model = state["model"]
        self.model_params = state["model_params"]
        self.feature_names_ = state["feature_names_"]
