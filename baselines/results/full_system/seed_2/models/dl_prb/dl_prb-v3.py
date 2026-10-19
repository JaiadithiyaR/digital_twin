from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.dt_models.base import DTComponent

DL_PRB_FEATURES: Tuple[str, ...] = (
    "dl_prb_lag1",
    "dl_prb_lag2",
    "dl_prb_lag3",
    "dl_prb_lag4",
    "dl_prb_lag5",
    "dl_prb_roll_mean_5s",
    "dl_prb_roll_mean_10s",
    "dl_prb_roll_std_5s",
    "dl_prb_roll_std_10s",
    "dl_throughput_lag1",
    "dl_throughput_lag2",
    "dl_throughput_lag3",
    "dl_throughput_roll_mean_5s",
    "dl_throughput_roll_mean_10s",
    "dl_sched_rb_avg_lag1",
    "dl_sched_rb_avg_lag2",
    "dl_sched_rb_avg_roll_mean_5s",
    "dl_curr_tbs_avg_lag1",
    "dl_curr_tbs_avg_lag2",
    "dl_curr_tbs_avg_roll_mean_5s",
    "wb_cqi_avg_lag1",
    "dl_mcs1_avg_lag1",
    "requested_prb_ratio_lag1",
    "ue_count_cell_lag1",
    "bsr_avg_lag1",
)

class DlPrbModel(DTComponent):
    COMPONENT_NAME = "dl_prb"
    DEPENDENCIES: Tuple[str, ...] = ()
    REQUIRED_FEATURES: Tuple[str, ...] = DL_PRB_FEATURES
    OUTPUT_FIELD = "dl_prb_pred"

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        self.model_params = model_params or {
            "max_iter": 150,
            "learning_rate": 0.05,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "random_state": 42,
        }
        self.model = HistGradientBoostingRegressor(**self.model_params)
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        if not cols:
            cols = features.select_dtypes(include=[np.number]).columns.tolist()
        df = features[cols].copy()
        return df.fillna(0.0)

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._prepare_features(features)
        y = targets.values.ravel()
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError("DlPrbModel is not trained yet.")
        X = self._prepare_features(features)
        preds = self.model.predict(X)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        rmse = float(np.sqrt(mean_squared_error(targets, preds)))
        mae = float(mean_absolute_error(targets, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "_is_trained": self._is_trained,
                "model_params": self.model_params,
            },
            path,
        )

    def load(self, path: Union[str, Path]) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self._is_trained = data["_is_trained"]
        self.model_params = data.get("model_params", self.model_params)

    @classmethod
    def from_settings(cls, settings: Any) -> "DlPrbModel":
        spec = getattr(settings.dt_models, "dl_prb", None)
        params = getattr(spec, "params", None) if spec else None
        return cls(model_params=params)
