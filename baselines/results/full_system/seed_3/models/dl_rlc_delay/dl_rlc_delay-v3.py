import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.dt_models.base import DTComponent

class DlRlcDelayModel(DTComponent):
    COMPONENT_NAME = "dl_rlc_delay"
    DEPENDENCIES = ("dl_throughput", "dl_bler")
    OUTPUT_FIELD = "dl_rlc_delay_pred"

    DL_BLER_INPUT_COLUMN = "dl_bler_pred"
    DL_THROUGHPUT_INPUT_COLUMN = "dl_throughput_pred"

    REQUIRED_FEATURES: Tuple[str, ...] = (
        "dl_rlc_delay_lag1",
        "dl_rlc_delay_lag2",
        "dl_rlc_delay_lag3",
        "dl_rlc_delay_lag4",
        "dl_rlc_delay_lag5",
        "dl_rlc_delay_roll_mean_5s",
        "dl_rlc_delay_roll_std_5s",
        "dl_rlc_delay_roll_mean_10s",
        "dl_rlc_delay_roll_std_10s",
        "dl_throughput_lag1",
        "dl_throughput_roll_mean_5s",
        "dl_bler_lag1",
        "dl_bler_roll_mean_5s",
        "dl_prb_lag1",
        "dl_curr_tbs_avg_lag1",
        "dl_sched_rb_avg_lag1",
        "wb_cqi_avg_lag1",
        "dl_mcs1_avg_lag1",
        "bsr_avg_lag1",
        "dl_throughput_pred",
        "dl_bler_pred",
    )

    def __init__(self, model_type: str = "hist_gb", params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model_type = model_type
        self.params = params or {}
        self.model: Optional[HistGradientBoostingRegressor] = None
        self._is_trained = False
        self.feature_names_in_: Optional[list[str]] = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @classmethod
    def from_settings(cls, settings: Any) -> "DlRlcDelayModel":
        if hasattr(settings, "dt_models") and hasattr(settings.dt_models, "dl_rlc_delay"):
            spec = settings.dt_models.dl_rlc_delay
            return cls(model_type=getattr(spec, "model_type", "hist_gb"), params=getattr(spec, "params", {}))
        return cls()

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        feat_cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        if not feat_cols:
            feat_cols = [c for c in features.columns if c not in ("cell_id", "run_id", "session_id", "slice_id", "ts_1s", "ue_id", "dl_rlc_delay")]
        
        X = features[feat_cols].fillna(0.0)
        y = targets.fillna(0.0)
        y = np.clip(y, 0.0, None)

        default_params = {
            "max_iter": 150,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_samples_leaf": 20,
            "random_state": 42
        }
        if self.params:
            default_params.update(self.params)

        self.model = HistGradientBoostingRegressor(**default_params)
        self.model.fit(X, y)
        self.feature_names_in_ = feat_cols
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model must be trained before calling predict.")
        
        cols = self.feature_names_in_ if self.feature_names_in_ else [c for c in self.REQUIRED_FEATURES if c in features.columns]
        X = features.reindex(columns=cols, fill_value=0.0).fillna(0.0)
        preds = self.model.predict(X)
        preds = np.clip(preds, 0.0, None)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        y_true = targets.fillna(0.0)
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        mae = float(mean_absolute_error(y_true, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str) -> None:
        state = {
            "model": self.model,
            "feature_names_in_": self.feature_names_in_,
            "_is_trained": self._is_trained,
            "model_type": self.model_type,
            "params": self.params,
        }
        joblib.dump(state, path)

    def load(self, path: str) -> None:
        state = joblib.load(path)
        self.model = state["model"]
        self.feature_names_in_ = state.get("feature_names_in_")
        self._is_trained = state.get("_is_trained", True)
        self.model_type = state.get("model_type", "hist_gb")
        self.params = state.get("params", {})
