from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.dt_models.base import DTComponent


class DlBlerModel(DTComponent):
    COMPONENT_NAME = "dl_bler"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES: tuple[str, ...] = (
        "dl_bler_lag1",
        "dl_bler_lag2",
        "dl_bler_lag3",
        "dl_bler_lag4",
        "dl_bler_lag5",
        "dl_bler_roll_mean_5s",
        "dl_bler_roll_mean_10s",
        "dl_bler_roll_std_5s",
        "dl_bler_roll_std_10s",
        "dl_bler_avg_lag1",
        "dl_bler_std_lag1",
        "dl_mcs1_avg_lag1",
        "dl_mcs1_std_lag1",
        "wb_cqi_avg_lag1",
        "wb_cqi_std_lag1",
        "pusch_snr_avg_lag1",
        "pucch_snr_avg_lag1",
        "dl_curr_tbs_avg_lag1",
        "dl_sched_rb_avg_lag1",
        "ul_bler_lag1",
        "dl_throughput_lag1",
        "dl_prb_lag1",
        "dl_rlc_delay_lag1",
    )
    OUTPUT_FIELD = "dl_bler_pred"

    def __init__(self, model_type: str = "hist_gb", params: dict | None = None):
        self.model_type = model_type
        self.params = params or {
            "max_iter": 100,
            "max_depth": 6,
            "learning_rate": 0.05,
            "l2_regularization": 0.1,
            "random_state": 42,
        }
        self.model = HistGradientBoostingRegressor(**self.params)
        self._is_trained = False
        self.feature_names_in_: list[str] = list(self.REQUIRED_FEATURES)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        if not cols:
            cols = list(features.columns)
        self.feature_names_in_ = cols
        X = features[cols].fillna(0.0)
        y = targets.copy().fillna(0.0)
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError("DlBlerModel must be trained before calling predict().")
        X = features.reindex(columns=self.feature_names_in_, fill_value=0.0).fillna(0.0)
        preds = self.model.predict(X)
        preds = np.clip(preds, 0.0, 1.0)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        y_true = targets.values
        y_pred = preds.values
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "_is_trained": self._is_trained,
            "feature_names_in_": self.feature_names_in_,
            "params": self.params,
            "model_type": self.model_type,
        }
        joblib.dump(state, path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        state = joblib.load(path)
        self.model = state["model"]
        self._is_trained = state.get("_is_trained", True)
        self.feature_names_in_ = state.get("feature_names_in_", list(self.REQUIRED_FEATURES))
        self.params = state.get("params", self.params)
        self.model_type = state.get("model_type", self.model_type)
