import os
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.dt_models.base import DTComponent


class DlThroughputModel(DTComponent):
    COMPONENT_NAME = "dl_throughput"
    DEPENDENCIES = ("dl_bler", "dl_prb")
    OUTPUT_FIELD = "dl_throughput_pred"

    REQUIRED_FEATURES = (
        "dl_throughput_lag1",
        "dl_throughput_lag2",
        "dl_throughput_lag3",
        "dl_throughput_lag4",
        "dl_throughput_lag5",
        "dl_throughput_roll_mean_5s",
        "dl_throughput_roll_mean_10s",
        "dl_throughput_roll_std_5s",
        "dl_throughput_roll_std_10s",
        "dl_prb_lag1",
        "dl_prb_lag2",
        "dl_prb_lag3",
        "dl_prb_lag4",
        "dl_prb_lag5",
        "dl_prb_roll_mean_5s",
        "dl_prb_roll_mean_10s",
        "dl_prb_roll_std_5s",
        "dl_prb_roll_std_10s",
        "dl_bler_lag1",
        "dl_bler_lag2",
        "dl_bler_lag3",
        "dl_bler_lag4",
        "dl_bler_lag5",
        "dl_bler_roll_mean_5s",
        "dl_bler_roll_mean_10s",
        "dl_bler_roll_std_5s",
        "dl_bler_roll_std_10s",
        "dl_curr_tbs_avg_lag1",
        "dl_curr_tbs_avg_roll_mean_5s",
        "dl_sched_rb_avg_lag1",
        "dl_sched_rb_avg_roll_mean_5s",
        "wb_cqi_avg_lag1",
        "wb_cqi_avg_roll_mean_5s",
        "dl_mcs1_avg_lag1",
        "dl_mcs1_avg_roll_mean_5s",
        "dl_bler_pred",
        "dl_prb_pred",
    )

    def __init__(self, max_iter: int = 200, learning_rate: float = 0.05, max_depth: int = 6):
        super().__init__()
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self._model = None
        self._feature_names = list(self.REQUIRED_FEATURES)

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        df = features.copy()
        if "dl_bler_pred" not in df.columns and "dl_bler" in df.columns:
            df["dl_bler_pred"] = df["dl_bler"]
        if "dl_prb_pred" not in df.columns and "dl_prb" in df.columns:
            df["dl_prb_pred"] = df["dl_prb"]

        cols = []
        for col in self.REQUIRED_FEATURES:
            if col in df.columns:
                cols.append(df[col].astype(float))
            else:
                cols.append(pd.Series(0.0, index=df.index, name=col))
        return pd.concat(cols, axis=1)

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._prepare_features(features)
        y = np.ravel(targets)
        self._model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=20,
            random_state=42,
        )
        self._model.fit(X, y)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self.is_trained:
            raise ValueError("DlThroughputModel is not trained yet.")
        X = self._prepare_features(features)
        preds = self._model.predict(X)
        preds = np.clip(preds, 0.0, None)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        y_true = np.ravel(targets)
        y_pred = preds.to_numpy()
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Any) -> None:
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        joblib.dump({"model": self._model, "feature_names": self._feature_names}, path)

    def load(self, path: Any) -> None:
        data = joblib.load(path)
        if isinstance(data, dict):
            self._model = data.get("model")
            self._feature_names = data.get("feature_names", list(self.REQUIRED_FEATURES))
        else:
            self._model = data

    @classmethod
    def from_settings(cls, settings: Any) -> "DlThroughputModel":
        return cls()
