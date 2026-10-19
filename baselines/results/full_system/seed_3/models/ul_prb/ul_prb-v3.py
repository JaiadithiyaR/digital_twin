import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from src.dt_models.base import DTComponent
except ImportError:
    from dt_models.base import DTComponent


class UlPrbModel(DTComponent):
    COMPONENT_NAME = "ul_prb"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES: tuple[str, ...] = (
        "ul_prb_lag1",
        "ul_prb_lag2",
        "ul_prb_lag3",
        "ul_prb_roll_mean_5s",
        "ul_prb_roll_mean_10s",
        "ul_prb_roll_std_5s",
        "ul_prb_roll_std_10s",
        "ul_throughput_lag1",
        "ul_throughput_roll_mean_5s",
        "ul_throughput_roll_mean_10s",
        "ul_bler_lag1",
        "ul_bler_roll_mean_5s",
        "pusch_snr_avg_lag1",
        "pusch_snr_avg_roll_mean_5s",
        "ul_mcs1_avg_lag1",
        "ul_mcs1_avg_roll_mean_5s",
        "bsr_avg_lag1",
        "requested_prb_ratio_lag1",
        "ue_count_cell_lag1",
    )
    OUTPUT_FIELD = "ul_prb_pred"

    def __init__(
        self,
        max_iter: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
    ) -> None:
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            l2_regularization=1.0,
        )
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        X = features[cols] if cols else features
        y = targets.values.ravel()
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise ValueError("UlPrbModel is not trained yet.")
        cols = [c for c in self.REQUIRED_FEATURES if c in features.columns]
        X = features[cols] if cols else features
        preds = self.model.predict(X)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        y_true = targets.values.ravel()
        y_pred = preds.values
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "is_trained": self._is_trained}, path)

    def load(self, path: str | Path) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self._is_trained = data["is_trained"]
