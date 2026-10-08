import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from src.dt_models.base import DTComponent

class DlPrbModel(DTComponent):
    COMPONENT_NAME = "dl_prb"
    DEPENDENCIES = ()
    REQUIRED_FEATURES = (
        "dl_prb_lag1",
        "dl_prb_lag2",
        "dl_prb_lag3",
        "dl_prb_lag4",
        "dl_prb_lag5",
        "dl_prb_roll_mean_5s",
        "dl_prb_roll_std_5s",
        "dl_prb_roll_mean_10s",
        "dl_prb_roll_std_10s",
        "dl_throughput_lag1",
        "dl_throughput_roll_mean_5s",
        "dl_throughput_roll_mean_10s",
        "dl_sched_rb_avg_lag1",
        "dl_sched_rb_std_lag1",
        "dl_curr_tbs_avg_lag1",
        "dl_curr_tbs_std_lag1",
        "wb_cqi_avg_lag1",
        "dl_mcs1_avg_lag1",
        "requested_prb_ratio_lag1",
        "ue_count_cell_lag1",
        "ue_count_slice_lag1",
        "dl_bler_lag1",
        "bsr_avg_lag1",
    )
    OUTPUT_FIELD = "dl_prb_pred"

    def __init__(self):
        self.model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=6,
            l2_regularization=0.1,
            random_state=42,
        )
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        if len(features) < 1:
            return
        X = pd.DataFrame(index=features.index)
        for col in self.REQUIRED_FEATURES:
            if col in features.columns:
                X[col] = features[col]
            else:
                X[col] = 0.0
        y = targets.values
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError("Model is not trained.")
        X = pd.DataFrame(index=features.index)
        for col in self.REQUIRED_FEATURES:
            if col in features.columns:
                X[col] = features[col]
            else:
                X[col] = 0.0
        preds = self.model.predict(X)
        preds = np.clip(preds, 0.0, None)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        y_true = targets.values
        y_pred = preds.values
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_true)))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "is_trained": self._is_trained}, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self._is_trained = data.get("is_trained", True)
