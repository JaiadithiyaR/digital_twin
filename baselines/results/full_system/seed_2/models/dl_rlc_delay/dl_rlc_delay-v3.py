import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.dt_models.base import DTComponent

class DlRlcDelayModel(DTComponent):
    COMPONENT_NAME = "dl_rlc_delay"
    DEPENDENCIES = ("dl_throughput", "dl_bler")
    OUTPUT_FIELD = "dl_rlc_delay_pred"
    
    REQUIRED_FEATURES = (
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
        "dl_throughput_roll_mean_5s",
        "dl_throughput_roll_mean_10s",
        "dl_prb_lag1",
        "dl_prb_lag2",
        "dl_prb_roll_mean_5s",
        "dl_prb_roll_mean_10s",
        "dl_bler_lag1",
        "dl_bler_lag2",
        "dl_bler_roll_mean_5s",
        "dl_bler_roll_mean_10s",
        "wb_cqi_avg_lag1",
        "dl_mcs1_avg_lag1",
        "bsr_avg_lag1",
        "requested_prb_ratio_lag1",
    )

    def __init__(self, model_params: Optional[dict] = None):
        self.model_params = model_params or {
            "max_iter": 150,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "random_state": 42
        }
        self.model = HistGradientBoostingRegressor(**self.model_params)
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _extract_feature_matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        cols = list(self.REQUIRED_FEATURES)
        if "dl_throughput_pred" in features.columns:
            cols.append("dl_throughput_pred")
        elif "dl_throughput" in features.columns:
            cols.append("dl_throughput")
            
        if "dl_bler_pred" in features.columns:
            cols.append("dl_bler_pred")
        elif "dl_bler" in features.columns:
            cols.append("dl_bler")

        available_cols = [c for c in cols if c in features.columns]
        X = features[available_cols].copy()
        X = X.fillna(0.0)
        return X

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._extract_feature_matrix(features)
        y = targets.fillna(0.0).clip(lower=0.0)
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet.")
        X = self._extract_feature_matrix(features)
        preds = self.model.predict(X)
        preds = np.clip(preds, 0.0, None)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        y_true = targets.fillna(0.0).clip(lower=0.0)
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        mae = float(mean_absolute_error(y_true, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "is_trained": self._is_trained, "model_params": self.model_params}, path)

    def load(self, path: Union[str, Path]) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self._is_trained = data["is_trained"]
        self.model_params = data.get("model_params", self.model_params)
