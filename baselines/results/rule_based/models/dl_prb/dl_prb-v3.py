import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.dt_models.base import DTComponent

class DlPrbModel(DTComponent):
    COMPONENT_NAME = "dl_prb"
    DEPENDENCIES: tuple[str, ...] = ()
    OUTPUT_FIELD = "dl_prb_pred"
    REQUIRED_FEATURES: tuple[str, ...] = (
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
        "dl_throughput_roll_mean_5s",
        "dl_sched_rb_avg_lag1",
        "dl_sched_rb_std_lag1",
        "requested_prb_ratio_lag1",
        "ue_count_cell_lag1",
        "has_controller_lag1",
    )

    def __init__(self, model_type: str = "hist_gb", params: dict = None):
        super().__init__()
        self.model_type = model_type
        self.params = params or {"max_iter": 150, "learning_rate": 0.05, "max_depth": 6, "random_state": 42}
        self.model = HistGradientBoostingRegressor(**self.params)
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @classmethod
    def from_settings(cls, settings) -> "DlPrbModel":
        spec = getattr(getattr(settings, "dt_models", None), "dl_prb", None)
        if spec is not None:
            return cls(model_type=getattr(spec, "model_type", "hist_gb"), params=getattr(spec, "params", None))
        return cls()

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = features[list(self.REQUIRED_FEATURES)].copy()
        y = targets.copy()
        valid_idx = y.notna()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError("Model is not trained.")
        X = features[list(self.REQUIRED_FEATURES)].copy()
        preds = self.model.predict(X)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        valid_idx = targets.notna() & preds.notna()
        y_true = targets.loc[valid_idx]
        y_pred = preds.loc[valid_idx]
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "model": self.model,
            "params": self.params,
            "model_type": self.model_type,
            "is_trained": self._is_trained,
            "required_features": self.REQUIRED_FEATURES,
        }
        joblib.dump(data, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.params = data.get("params", self.params)
        self.model_type = data.get("model_type", self.model_type)
        self._is_trained = data.get("is_trained", True)
