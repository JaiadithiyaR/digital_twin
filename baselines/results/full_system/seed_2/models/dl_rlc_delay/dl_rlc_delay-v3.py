import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.dt_models.base import DTComponent

class DlRlcDelayModel(DTComponent):
    COMPONENT_NAME: str = "dl_rlc_delay"
    DEPENDENCIES: Tuple[str, ...] = ("dl_throughput", "dl_bler")
    REQUIRED_FEATURES: Tuple[str, ...] = (
        "dl_throughput_lag1", "dl_throughput_lag2", "dl_throughput_lag3", "dl_throughput_lag4", "dl_throughput_lag5",
        "dl_throughput_roll_mean_5s", "dl_throughput_roll_mean_10s", "dl_throughput_roll_std_5s", "dl_throughput_roll_std_10s",
        "ul_throughput_lag1", "ul_throughput_lag2", "ul_throughput_lag3", "ul_throughput_lag4", "ul_throughput_lag5",
        "ul_throughput_roll_mean_5s", "ul_throughput_roll_mean_10s", "ul_throughput_roll_std_5s", "ul_throughput_roll_std_10s",
        "dl_rlc_delay_lag1", "dl_rlc_delay_lag2", "dl_rlc_delay_lag3", "dl_rlc_delay_lag4", "dl_rlc_delay_lag5",
        "dl_rlc_delay_roll_mean_5s", "dl_rlc_delay_roll_mean_10s", "dl_rlc_delay_roll_std_5s", "dl_rlc_delay_roll_std_10s",
        "dl_prb_lag1", "dl_prb_lag2", "dl_prb_lag3", "dl_prb_lag4", "dl_prb_lag5",
        "dl_prb_roll_mean_5s", "dl_prb_roll_mean_10s", "dl_prb_roll_std_5s", "dl_prb_roll_std_10s",
        "ul_prb_lag1", "ul_prb_lag2", "ul_prb_lag3", "ul_prb_lag4", "ul_prb_lag5",
        "ul_prb_roll_mean_5s", "ul_prb_roll_mean_10s", "ul_prb_roll_std_5s", "ul_prb_roll_std_10s",
        "dl_bler_lag1", "dl_bler_lag2", "dl_bler_lag3", "dl_bler_lag4", "dl_bler_lag5",
        "dl_bler_roll_mean_5s", "dl_bler_roll_mean_10s", "dl_bler_roll_std_5s", "dl_bler_roll_std_10s",
        "ul_bler_lag1", "ul_bler_lag2", "ul_bler_lag3", "ul_bler_lag4", "ul_bler_lag5",
        "ul_bler_roll_mean_5s", "ul_bler_roll_mean_10s", "ul_bler_roll_std_5s", "ul_bler_roll_std_10s"
    )
    OUTPUT_FIELD: str = "dl_rlc_delay_pred"
    DL_BLER_INPUT_COLUMN: str = "dl_bler_pred"
    DL_THROUGHPUT_INPUT_COLUMN: str = "dl_throughput_pred"

    def __init__(self, model_type: str = "hist_gb", params: Optional[Dict[str, Any]] = None) -> None:
        self.model_type = model_type
        self.params = params or {
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 10,
            "random_state": 42
        }
        self.model: Optional[HistGradientBoostingRegressor] = None
        self._is_trained: bool = False
        self.feature_names_: Optional[list] = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @classmethod
    def from_settings(cls, settings: Any) -> "DlRlcDelayModel":
        spec = getattr(getattr(settings, "dt_models", None), "dl_rlc_delay", None)
        if spec is not None:
            return cls(model_type=getattr(spec, "model_type", "hist_gb"), params=getattr(spec, "params", None))
        return cls()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ignore_cols = {"cell_id", "run_id", "session_id", "slice_id", "ts_1s", "ue_id", "dl_rlc_delay"}
        feature_cols = [c for c in df.columns if c not in ignore_cols]
        X = df[feature_cols].copy()
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return X

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._prepare_features(features)
        self.feature_names_ = list(X.columns)
        y = targets.values if isinstance(targets, pd.Series) else np.array(targets)
        self.model = HistGradientBoostingRegressor(**self.params)
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained or self.model is None:
            return pd.Series(0.0, index=features.index, name=self.OUTPUT_FIELD)
        X = self._prepare_features(features)
        if self.feature_names_ is not None:
            X = X.reindex(columns=self.feature_names_, fill_value=0.0)
        preds = self.model.predict(X)
        preds = np.maximum(0.0, preds)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        y_true = targets.values if isinstance(targets, pd.Series) else np.array(targets)
        y_pred = preds.values if isinstance(preds, pd.Series) else np.array(preds)
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if not np.any(valid_mask):
            return {"rmse": 0.0, "mae": 0.0}
        y_t = y_true[valid_mask]
        y_p = y_pred[valid_mask]
        rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
        mae = float(mean_absolute_error(y_t, y_p))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "model": self.model,
            "is_trained": self._is_trained,
            "model_type": self.model_type,
            "params": self.params,
            "feature_names": self.feature_names_,
        }
        joblib.dump(payload, path)

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        self.model = payload["model"]
        self._is_trained = payload["is_trained"]
        self.model_type = payload.get("model_type", self.model_type)
        self.params = payload.get("params", self.params)
        self.feature_names_ = payload.get("feature_names", None)
