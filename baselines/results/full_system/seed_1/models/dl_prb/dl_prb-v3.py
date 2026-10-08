import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.dt_models.base import DTComponent

class DlPrbModel(DTComponent):
    COMPONENT_NAME = "dl_prb"
    DEPENDENCIES: Tuple[str, ...] = ()
    REQUIRED_FEATURES = (
        "dl_prb_lag1", "dl_prb_lag2", "dl_prb_lag3", "dl_prb_lag4", "dl_prb_lag5",
        "dl_prb_roll_mean_5s", "dl_prb_roll_mean_10s", "dl_prb_roll_std_5s", "dl_prb_roll_std_10s",
        "dl_throughput_lag1", "dl_throughput_roll_mean_5s", "dl_throughput_roll_mean_10s",
        "dl_sched_rb_avg_lag1", "dl_curr_tbs_avg_lag1", "wb_cqi_avg_lag1", "requested_prb_ratio_lag1",
        "ue_count_cell_lag1", "has_controller_lag1"
    )
    OUTPUT_FIELD = "dl_prb_pred"

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        super().__init__()
        params = {
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "random_state": 42
        }
        if model_params:
            params.update(model_params)
        self.model = HistGradientBoostingRegressor(**params)
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = features[list(self.REQUIRED_FEATURES)].fillna(0.0)
        y = targets.fillna(0.0)
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet.")
        X = features[list(self.REQUIRED_FEATURES)].fillna(0.0)
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
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "is_trained": self._is_trained}, p)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self._is_trained = data["is_trained"]
