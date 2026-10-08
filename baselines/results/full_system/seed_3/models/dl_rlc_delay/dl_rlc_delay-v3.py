import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.dt_models.base import DTComponent

class DlRlcDelayModel(DTComponent):
    COMPONENT_NAME = "dl_rlc_delay"
    DEPENDENCIES = ("dl_throughput", "dl_bler")
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
        "dl_throughput_roll_mean_5s",
        "dl_prb_lag1",
        "dl_prb_roll_mean_5s",
        "dl_bler_lag1",
        "dl_bler_roll_mean_5s",
        "wb_cqi_avg_lag1",
        "dl_sched_rb_avg_lag1",
        "dl_curr_tbs_avg_lag1",
        "has_controller_lag1",
    )
    OUTPUT_FIELD = "dl_rlc_delay_pred"

    def __init__(self, max_iter: int = 150, learning_rate: float = 0.05, max_depth: int = 6):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = None

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def _prepare_inputs(self, features: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=features.index)
        for col in self.REQUIRED_FEATURES:
            if col in features.columns:
                X[col] = features[col].fillna(0.0)
            else:
                X[col] = 0.0
        if "dl_throughput_pred" in features.columns:
            X["dl_throughput_dep"] = features["dl_throughput_pred"].fillna(0.0)
        elif "dl_throughput" in features.columns:
            X["dl_throughput_dep"] = features["dl_throughput"].fillna(0.0)
        else:
            X["dl_throughput_dep"] = 0.0
        if "dl_bler_pred" in features.columns:
            X["dl_bler_dep"] = features["dl_bler_pred"].fillna(0.0)
        elif "dl_bler" in features.columns:
            X["dl_bler_dep"] = features["dl_bler"].fillna(0.0)
        else:
            X["dl_bler_dep"] = 0.0
        return X

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._prepare_inputs(features)
        y = targets.fillna(0.0)
        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
            loss="squared_error",
        )
        self.model.fit(X, y)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        X = self._prepare_inputs(features)
        preds = self.model.predict(X)
        preds = np.maximum(0.0, preds)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        y_true = targets.fillna(0.0)
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        mae = float(mean_absolute_error(y_true, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path) -> None:
        joblib.dump({"model": self.model, "max_iter": self.max_iter, "learning_rate": self.learning_rate, "max_depth": self.max_depth}, path)

    def load(self, path) -> None:
        data = joblib.load(path)
        if isinstance(data, dict):
            self.model = data.get("model")
            self.max_iter = data.get("max_iter", 150)
            self.learning_rate = data.get("learning_rate", 0.05)
            self.max_depth = data.get("max_depth", 6)
        else:
            self.model = data