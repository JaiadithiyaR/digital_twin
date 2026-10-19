import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from src.dt_models.base import DTComponent

class UlBlerModel(DTComponent):
    COMPONENT_NAME = "ul_bler"
    DEPENDENCIES: tuple[str, ...] = ()
    OUTPUT_FIELD = "ul_bler_pred"
    REQUIRED_FEATURES: tuple[str, ...] = (
        "ul_bler_lag1", "ul_bler_lag2", "ul_bler_lag3", "ul_bler_lag4", "ul_bler_lag5",
        "ul_bler_roll_mean_5s", "ul_bler_roll_mean_10s", "ul_bler_roll_std_5s", "ul_bler_roll_std_10s",
        "ul_bler_avg_lag1", "ul_bler_avg_lag2", "ul_bler_avg_lag3", "ul_bler_avg_lag4", "ul_bler_avg_lag5",
        "ul_bler_avg_roll_mean_5s", "ul_bler_avg_roll_mean_10s", "ul_bler_avg_roll_std_5s", "ul_bler_avg_roll_std_10s",
        "ul_bler_std_lag1", "ul_bler_std_lag2", "ul_bler_std_lag3", "ul_bler_std_lag4", "ul_bler_std_lag5",
        "ul_bler_std_roll_mean_5s", "ul_bler_roll_mean_10s", "ul_bler_std_roll_std_5s", "ul_bler_std_roll_std_10s",
        "pusch_snr_avg_lag1", "pusch_snr_avg_lag2", "pusch_snr_avg_lag3", "pusch_snr_avg_lag4", "pusch_snr_avg_lag5",
        "pusch_snr_avg_roll_mean_5s", "pusch_snr_avg_roll_mean_10s", "pusch_snr_avg_roll_std_5s", "pusch_snr_avg_roll_std_10s",
        "pusch_snr_std_lag1", "pusch_snr_std_lag2", "pusch_snr_std_lag3", "pusch_snr_std_lag4", "pusch_snr_std_lag5",
        "ul_mcs1_avg_lag1", "ul_mcs1_avg_lag2", "ul_mcs1_avg_lag3", "ul_mcs1_avg_lag4", "ul_mcs1_avg_lag5",
        "ul_mcs1_avg_roll_mean_5s", "ul_mcs1_avg_roll_mean_10s", "ul_mcs1_avg_roll_std_5s", "ul_mcs1_avg_roll_std_10s",
        "phr_avg_lag1", "phr_avg_lag2", "phr_avg_lag3", "phr_avg_lag4", "phr_avg_lag5",
        "ul_prb_lag1", "ul_prb_lag2", "ul_prb_lag3", "ul_prb_lag4", "ul_prb_lag5",
        "ul_prb_roll_mean_5s", "ul_prb_roll_mean_10s", "ul_prb_roll_std_5s", "ul_prb_roll_std_10s",
        "ul_throughput_lag1", "ul_throughput_lag2", "ul_throughput_lag3", "ul_throughput_lag4", "ul_throughput_lag5",
        "ul_throughput_roll_mean_5s", "ul_throughput_roll_mean_10s", "ul_throughput_roll_std_5s", "ul_throughput_roll_std_10s",
        "bsr_avg_lag1", "n_samples_lag1", "ue_count_cell_lag1", "ue_count_slice_lag1", "requested_prb_ratio_lag1", "has_controller_lag1"
    )

    def __init__(self, max_iter: int = 100, learning_rate: float = 0.05, max_depth: int = 6):
        super().__init__()
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self._model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42
        )
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for col in self.REQUIRED_FEATURES:
            if col in df.columns:
                X[col] = df[col]
            else:
                X[col] = 0.0
        return X

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._prepare_features(features)
        y = np.asarray(targets)
        self._model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model is not trained.")
        X = self._prepare_features(features)
        preds = self._model.predict(X)
        preds = np.clip(preds, 0.0, 1.0)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        y_true = np.asarray(targets)
        y_pred = np.asarray(preds)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"model": self._model, "is_trained": self._is_trained}, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self._model = data["model"]
        self._is_trained = data["is_trained"]
