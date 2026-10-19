from pathlib import Path
from typing import Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from src.dt_models.base import DTComponent

class DlBlerModel(DTComponent):
    COMPONENT_NAME: str = "dl_bler"
    DEPENDENCIES: Tuple[str, ...] = ()
    REQUIRED_FEATURES: Tuple[str, ...] = (
        "dl_bler_lag1",
        "dl_bler_lag2",
        "dl_bler_lag3",
        "dl_bler_lag4",
        "dl_bler_lag5",
        "dl_bler_roll_mean_5s",
        "dl_bler_roll_mean_10s",
        "dl_bler_roll_std_5s",
        "dl_bler_roll_std_10s",
        "wb_cqi_avg_lag1",
        "wb_cqi_std_lag1",
        "wb_cqi_avg_roll_mean_5s",
        "wb_cqi_avg_roll_mean_10s",
        "dl_mcs1_avg_lag1",
        "dl_mcs1_std_lag1",
        "dl_mcs1_avg_roll_mean_5s",
        "dl_mcs1_avg_roll_mean_10s",
        "dl_throughput_lag1",
        "dl_throughput_roll_mean_5s",
        "dl_throughput_roll_mean_10s",
        "dl_prb_lag1",
        "dl_prb_roll_mean_5s",
        "dl_prb_roll_mean_10s",
        "dl_curr_tbs_avg_lag1",
        "dl_curr_tbs_std_lag1",
        "dl_curr_tbs_avg_roll_mean_5s",
        "dl_sched_rb_avg_lag1",
        "dl_sched_rb_std_lag1",
        "dl_sched_rb_avg_roll_mean_5s",
        "ul_bler_lag1",
        "ul_bler_roll_mean_5s",
        "pucch_snr_avg_lag1",
        "pusch_snr_avg_lag1"
    )
    OUTPUT_FIELD: str = "dl_bler_pred"

    def __init__(self, max_iter: int = 200, learning_rate: float = 0.03, max_depth: int = 6, min_samples_leaf: int = 15, random_state: int = 42):
        super().__init__()
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _extract_features(self, features: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=features.index)
        for col in self.REQUIRED_FEATURES:
            if col in features.columns:
                X[col] = features[col]
            else:
                X[col] = 0.0
        return X

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._extract_features(features)
        y = targets.values.astype(np.float64)
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError(f"{self.__class__.__name__} is not trained.")
        X = self._extract_features(features)
        preds = self.model.predict(X)
        preds = np.clip(preds, 0.0, 1.0)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        err = preds.values - targets.values
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Any) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path_obj)

    @classmethod
    def load(cls, path: Any) -> "DlBlerModel":
        return joblib.load(Path(path))
