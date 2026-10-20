import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.dt_models.base import DTComponent

ALL_REQUIRED_FEATURES: Tuple[str, ...] = (
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
    "dl_curr_tbs_avg_lag1", "dl_curr_tbs_avg_lag2", "dl_curr_tbs_avg_lag3", "dl_curr_tbs_avg_lag4", "dl_curr_tbs_avg_lag5",
    "dl_curr_tbs_avg_roll_mean_5s", "dl_curr_tbs_avg_roll_mean_10s", "dl_curr_tbs_avg_roll_std_5s", "dl_curr_tbs_avg_roll_std_10s",
    "dl_curr_tbs_std_lag1", "dl_curr_tbs_std_lag2", "dl_curr_tbs_std_lag3", "dl_curr_tbs_std_lag4", "dl_curr_tbs_std_lag5",
    "dl_curr_tbs_std_roll_mean_5s", "dl_curr_tbs_std_roll_mean_10s", "dl_curr_tbs_std_roll_std_5s", "dl_curr_tbs_std_roll_std_10s",
    "dl_sched_rb_avg_lag1", "dl_sched_rb_avg_lag2", "dl_sched_rb_avg_lag3", "dl_sched_rb_avg_lag4", "dl_sched_rb_avg_lag5",
    "dl_sched_rb_avg_roll_mean_5s", "dl_sched_rb_avg_roll_mean_10s", "dl_sched_rb_avg_roll_std_5s", "dl_sched_rb_avg_roll_std_10s",
    "dl_sched_rb_std_lag1", "dl_sched_rb_std_lag2", "dl_sched_rb_std_lag3", "dl_sched_rb_std_lag4", "dl_sched_rb_std_lag5",
    "dl_sched_rb_std_roll_mean_5s", "dl_sched_rb_std_roll_mean_10s", "dl_sched_rb_std_roll_std_5s", "dl_sched_rb_std_roll_std_10s",
    "pusch_snr_avg_lag1", "pusch_snr_avg_lag2", "pusch_snr_avg_lag3", "pusch_snr_avg_lag4", "pusch_snr_avg_lag5",
    "pusch_snr_avg_roll_mean_5s", "pusch_snr_avg_roll_mean_10s", "pusch_snr_avg_roll_std_5s", "pusch_snr_avg_roll_std_10s",
    "pusch_snr_std_lag1", "pusch_snr_std_lag2", "pusch_snr_std_lag3", "pusch_snr_std_lag4", "pusch_snr_std_lag5",
    "pusch_snr_std_roll_mean_5s", "pusch_snr_std_roll_mean_10s", "pusch_snr_std_roll_std_5s", "pusch_snr_std_roll_std_10s",
    "pucch_snr_avg_lag1", "pucch_snr_avg_lag2", "pucch_snr_avg_lag3", "pucch_snr_avg_lag4", "pucch_snr_avg_lag5",
    "pucch_snr_avg_roll_mean_5s", "pucch_snr_avg_roll_mean_10s", "pucch_snr_avg_roll_std_5s", "pucch_snr_avg_roll_std_10s",
    "pucch_snr_std_lag1", "pucch_snr_std_lag2", "pucch_snr_std_lag3", "pucch_snr_std_lag4", "pucch_snr_std_lag5",
    "pucch_snr_std_roll_mean_5s", "pucch_snr_std_roll_mean_10s", "pucch_snr_std_roll_std_5s", "pucch_snr_std_roll_std_10s",
    "wb_cqi_avg_lag1", "wb_cqi_avg_lag2", "wb_cqi_avg_lag3", "wb_cqi_avg_lag4", "wb_cqi_avg_lag5",
    "wb_cqi_avg_roll_mean_5s", "wb_cqi_avg_roll_mean_10s", "wb_cqi_avg_roll_std_5s", "wb_cqi_avg_roll_std_10s",
    "wb_cqi_std_lag1", "wb_cqi_std_lag2", "wb_cqi_std_lag3", "wb_cqi_std_lag4", "wb_cqi_std_lag5",
    "wb_cqi_std_roll_mean_5s", "wb_cqi_std_roll_mean_10s", "wb_cqi_std_roll_std_5s", "wb_cqi_std_roll_std_10s",
    "dl_mcs1_avg_lag1", "dl_mcs1_avg_lag2", "dl_mcs1_avg_lag3", "dl_mcs1_avg_lag4", "dl_mcs1_avg_lag5",
    "dl_mcs1_avg_roll_mean_5s", "dl_mcs1_avg_roll_mean_10s", "dl_mcs1_avg_roll_std_5s", "dl_mcs1_avg_roll_std_10s",
    "dl_mcs1_std_lag1", "dl_mcs1_std_lag2", "dl_mcs1_std_lag3", "dl_mcs1_std_lag4", "dl_mcs1_std_lag5",
    "dl_mcs1_std_roll_mean_5s", "dl_mcs1_std_roll_mean_10s", "dl_mcs1_std_roll_std_5s", "dl_mcs1_std_roll_std_10s",
    "ul_mcs1_avg_lag1", "ul_mcs1_avg_lag2", "ul_mcs1_avg_lag3", "ul_mcs1_avg_lag4", "ul_mcs1_avg_lag5",
    "ul_mcs1_avg_roll_mean_5s", "ul_mcs1_avg_roll_mean_10s", "ul_mcs1_avg_roll_std_5s", "ul_mcs1_avg_roll_std_10s",
    "ul_mcs1_std_lag1", "ul_mcs1_std_lag2", "ul_mcs1_std_lag3", "ul_mcs1_std_lag4", "ul_mcs1_std_lag5",
    "ul_mcs1_std_roll_mean_5s", "ul_mcs1_std_roll_mean_10s", "ul_mcs1_std_roll_std_5s", "ul_mcs1_std_roll_std_10s",
    "phr_avg_lag1", "phr_avg_lag2", "phr_avg_lag3", "phr_avg_lag4", "phr_avg_lag5",
    "phr_avg_roll_mean_5s", "phr_avg_roll_mean_10s", "phr_avg_roll_std_5s", "phr_avg_roll_std_10s",
    "phr_std_lag1", "phr_std_lag2", "phr_std_lag3", "phr_std_lag4", "phr_std_lag5",
    "phr_std_roll_mean_5s", "phr_std_roll_mean_10s", "phr_std_roll_std_5s", "phr_std_roll_std_10s",
    "dl_bler_avg_lag1", "dl_bler_avg_lag2", "dl_bler_avg_lag3", "dl_bler_avg_lag4", "dl_bler_avg_lag5",
    "dl_bler_avg_roll_mean_5s", "dl_bler_avg_roll_mean_10s", "dl_bler_avg_roll_std_5s", "dl_bler_avg_roll_std_10s",
    "dl_bler_std_lag1", "dl_bler_std_lag2", "dl_bler_std_lag3", "dl_bler_std_lag4", "dl_bler_std_lag5",
    "dl_bler_std_roll_mean_5s", "dl_bler_std_roll_mean_10s", "dl_bler_std_roll_std_5s", "dl_bler_std_roll_std_10s",
    "ul_bler_avg_lag1", "ul_bler_avg_lag2", "ul_bler_avg_lag3", "ul_bler_avg_lag4", "ul_bler_avg_lag5",
    "ul_bler_avg_roll_mean_5s", "ul_bler_avg_roll_mean_10s", "ul_bler_avg_roll_std_5s", "ul_bler_avg_roll_std_10s",
    "ul_bler_std_lag1", "ul_bler_std_lag2", "ul_bler_std_lag3", "ul_bler_std_lag4", "ul_bler_std_lag5",
    "ul_bler_std_roll_mean_5s", "ul_bler_std_roll_mean_10s", "ul_bler_std_roll_std_5s", "ul_bler_std_roll_std_10s",
    "bsr_avg_lag1", "n_samples_lag1", "ue_count_cell_lag1", "ue_count_slice_lag1", "requested_prb_ratio_lag1", "has_controller_lag1"
)

class DlRlcDelayModel(DTComponent):
    COMPONENT_NAME = "dl_rlc_delay"
    DEPENDENCIES = ("dl_throughput", "dl_bler")
    REQUIRED_FEATURES = ALL_REQUIRED_FEATURES
    OUTPUT_FIELD = "dl_rlc_delay_pred"

    DL_BLER_INPUT_COLUMN = "dl_bler_pred"
    DL_THROUGHPUT_INPUT_COLUMN = "dl_throughput_pred"

    def __init__(self, model_type: str = "ensemble", params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        self.params = params or {}
        self._is_trained = False
        self.model_mae = None
        self.model_mse = None
        self.model_et = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @classmethod
    def from_settings(cls, settings: Any) -> "DlRlcDelayModel":
        return cls()

    def _extract_base_prediction(self, df: pd.DataFrame) -> pd.Series:
        if "dl_rlc_delay_lag1" in df.columns:
            base = df["dl_rlc_delay_lag1"].copy()
        elif "dl_rlc_delay_roll_mean_5s" in df.columns:
            base = df["dl_rlc_delay_roll_mean_5s"].copy()
        else:
            base = pd.Series(0.0, index=df.index)
        return base.fillna(0.0)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ALL_REQUIRED_FEATURES if c in df.columns]
        X = df[cols].copy()
        if self.DL_THROUGHPUT_INPUT_COLUMN in df.columns:
            X[self.DL_THROUGHPUT_INPUT_COLUMN] = df[self.DL_THROUGHPUT_INPUT_COLUMN]
        if self.DL_BLER_INPUT_COLUMN in df.columns:
            X[self.DL_BLER_INPUT_COLUMN] = df[self.DL_BLER_INPUT_COLUMN]
        lag1 = df.get("dl_rlc_delay_lag1", pd.Series(0.0, index=df.index)).fillna(0.0)
        lag2 = df.get("dl_rlc_delay_lag2", pd.Series(0.0, index=df.index)).fillna(0.0)
        roll_mean = df.get("dl_rlc_delay_roll_mean_5s", pd.Series(0.0, index=df.index)).fillna(0.0)
        tp = df.get(self.DL_THROUGHPUT_INPUT_COLUMN, df.get("dl_throughput_lag1", pd.Series(0.0, index=df.index))).fillna(0.0)
        prb = df.get("dl_prb_lag1", pd.Series(0.0, index=df.index)).fillna(0.0)
        X["dl_rlc_delay_diff1_2"] = lag1 - lag2
        X["dl_rlc_delay_diff1_roll"] = lag1 - roll_mean
        X["traffic_load_ratio"] = tp / (prb + 1.0)
        X["delay_load_interaction"] = lag1 * X["traffic_load_ratio"]
        return X.fillna(0.0)

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._prepare_features(features)
        base = self._extract_base_prediction(features)
        y_res = targets - base
        self.model_mae = HistGradientBoostingRegressor(loss="absolute_error", max_iter=250, l2_regularization=1.0, random_state=42)
        self.model_mse = HistGradientBoostingRegressor(loss="squared_error", max_iter=250, l2_regularization=0.5, random_state=43)
        self.model_et = Pipeline([("imputer", SimpleImputer(strategy="median")), ("et", ExtraTreesRegressor(n_estimators=100, max_depth=12, min_samples_leaf=2, random_state=44, n_jobs=-1))])
        self.model_mae.fit(X, y_res)
        self.model_mse.fit(X, y_res)
        self.model_et.fit(X, y_res)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before predicting.")
        X = self._prepare_features(features)
        base = self._extract_base_prediction(features)
        r1 = self.model_mae.predict(X)
        r2 = self.model_mse.predict(X)
        r3 = self.model_et.predict(X)
        res_pred = 0.45 * r1 + 0.35 * r2 + 0.20 * r3
        y_pred = base + res_pred
        y_pred = np.maximum(0.0, y_pred)
        return pd.Series(y_pred, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        y_true = targets.to_numpy()
        y_pred = preds.to_numpy()
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    def load(self, path: Union[str, Path]) -> None:
        path = Path(path)
        obj = joblib.load(path)
        if hasattr(obj, "__dict__"):
            self.__dict__.update(obj.__dict__)
