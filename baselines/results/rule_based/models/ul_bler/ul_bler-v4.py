import pickle
from pathlib import Path
from typing import Any, Dict, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.dt_models.base import DTComponent

FEATURE_COLS = (
    "dl_throughput_lag1", "dl_throughput_lag2", "dl_throughput_lag3", "dl_throughput_lag4", "dl_throughput_lag5",
    "dl_throughput_roll_mean_5s", "dl_throughput_roll_mean_10s", "dl_throughput_roll_std_5s", "dl_throughput_roll_std_10s",
    "ul_throughput_lag1", "ul_throughput_lag2", "ul_throughput_lag3", "ul_throughput_lag4", "ul_throughput_lag5",
    "ul_throughput_roll_mean_5s", "ul_throughput_roll_mean_10s", "ul_throughput_roll_std_5s", "ul_throughput_roll_std_10s",
    "dl_rlc_delay_lag1", "dl_rlc_delay_lag2", "dl_rlc_delay_lag3", "dl_rlc_delay_lag4", "dl_rlc_delay_lag5",
    "dl_rlc_delay_roll_mean_5s", "dl_rlc_delay_roll_mean_10s", "dl_rlc_delay_roll_std_5s", "dl_rlc_delay_roll_std_10s",
    "dl_prb_lag1", "dl_prb_lag2", "dl_prb_lag3", "dl_prb_lag4", "dl_prb_lag5",
    "dl_prb_roll_mean_5s", "dl_prb_roll_mean_10s", "dl_prb_roll_std_5s", "dl_prb_roll_std_10s",
    "ul_prb_lag1", "ul_prb_lag2", "ul_prb_lag3", "ul_prb_lag4", "ul_prb_lag5",
    "ul_prb_roll_mean_5s", "ul_prb_roll_mean_10s", "ul_prb_roll_std_5s", "ul_prb_roll_std_10s",
    "dl_bler_lag1", "dl_bler_lag2", "dl_bler_lag3", "dl_bler_lag4", "dl_bler_lag5",
    "dl_bler_roll_mean_5s", "dl_bler_roll_mean_10s", "dl_bler_roll_std_5s", "dl_bler_roll_std_10s",
    "ul_bler_lag1", "ul_bler_lag2", "ul_bler_lag3", "ul_bler_lag4", "ul_bler_lag5",
    "ul_bler_roll_mean_5s", "ul_bler_roll_mean_10s", "ul_bler_roll_std_5s", "ul_bler_roll_std_10s",
    "pusch_snr_avg_lag1", "pusch_snr_avg_lag2", "pusch_snr_avg_lag3", "pusch_snr_avg_roll_mean_5s", "pusch_snr_avg_roll_std_5s",
    "pucch_snr_avg_lag1", "pucch_snr_avg_lag2", "pucch_snr_avg_roll_mean_5s",
    "ul_mcs1_avg_lag1", "ul_mcs1_avg_lag2", "ul_mcs1_avg_roll_mean_5s",
    "phr_avg_lag1", "phr_avg_roll_mean_5s", "ul_bler_avg_lag1", "ul_bler_avg_roll_mean_5s", "ul_bler_std_lag1"
)

class UlBlerModel(DTComponent):
    COMPONENT_NAME = "ul_bler"
    DEPENDENCIES = ()
    REQUIRED_FEATURES = FEATURE_COLS
    OUTPUT_FIELD = "ul_bler_pred"

    def __init__(self) -> None:
        self._is_trained = False
        self.model_res_hgb = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.03,
            max_leaf_nodes=31,
            min_samples_leaf=15,
            random_state=42
        )
        self.model_res_et = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.model_direct_hgb = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.03,
            max_leaf_nodes=31,
            min_samples_leaf=15,
            random_state=42
        )

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        df = features.copy()
        for col in self.REQUIRED_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        df = df[list(self.REQUIRED_FEATURES)].fillna(0.0)
        return df

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._prepare_features(features)
        y = targets.astype(float).values
        
        lag1 = features["ul_bler_lag1"].fillna(0.0).values if "ul_bler_lag1" in features.columns else np.zeros_like(y)
        residuals = y - lag1
        
        self.model_res_hgb.fit(X, residuals)
        self.model_res_et.fit(X, residuals)
        self.model_direct_hgb.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before calling predict.")
        X = self._prepare_features(features)
        lag1 = features["ul_bler_lag1"].fillna(0.0).values if "ul_bler_lag1" in features.columns else np.zeros(len(features))
        
        pred_res_hgb = self.model_res_hgb.predict(X)
        pred_res_et = self.model_res_et.predict(X)
        res_pred = 0.6 * pred_res_hgb + 0.4 * pred_res_et
        
        res_based_pred = np.clip(lag1 + res_pred, 0.0, 1.0)
        direct_pred = np.clip(self.model_direct_hgb.predict(X), 0.0, 1.0)
        
        final_pred = np.clip(0.75 * res_based_pred + 0.25 * direct_pred, 0.0, 1.0)
        return pd.Series(final_pred, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        predictions = self.predict(features)
        y_true = targets.astype(float).values
        y_pred = predictions.values
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: Union[str, Path]) -> None:
        with open(path, "rb") as f:
            loaded = pickle.load(f)
            self.__dict__.update(loaded.__dict__)
