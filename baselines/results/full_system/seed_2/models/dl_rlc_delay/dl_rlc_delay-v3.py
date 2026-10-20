import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.dt_models.base import DTComponent

REQUIRED_FEATURES_LIST = (
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
    REQUIRED_FEATURES = REQUIRED_FEATURES_LIST
    OUTPUT_FIELD = "dl_rlc_delay_pred"

    DL_BLER_INPUT_COLUMN = "dl_bler_pred"
    DL_THROUGHPUT_INPUT_COLUMN = "dl_throughput_pred"

    def __init__(self, model_type: str = "hgb", params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.model_type = model_type
        self.params = params or {}
        self._model: Optional[HistGradientBoostingRegressor] = None
        self._is_trained: bool = False
        self._feature_names: Optional[list[str]] = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def _input_columns(self) -> tuple[str, ...]:
        return tuple(self.REQUIRED_FEATURES) + (
            self.DL_BLER_INPUT_COLUMN,
            self.DL_THROUGHPUT_INPUT_COLUMN,
        )

    @classmethod
    def from_settings(cls, settings: Any) -> "DlRlcDelayModel":
        spec = getattr(getattr(settings, "dt_models", None), "dl_rlc_delay", None)
        if spec is not None:
            return cls(model_type=getattr(spec, "model_type", "hgb"), params=getattr(spec, "params", None))
        return cls()

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        X_df = pd.DataFrame(index=df.index)
        for col in self._input_columns:
            if col in df.columns:
                X_df[col] = df[col].astype(float)
            else:
                X_df[col] = 0.0

        if "dl_rlc_delay_lag1" in X_df.columns and "dl_rlc_delay_lag2" in X_df.columns:
            X_df["dl_rlc_delay_diff1"] = X_df["dl_rlc_delay_lag1"] - X_df["dl_rlc_delay_lag2"]
        if "dl_rlc_delay_roll_mean_5s" in X_df.columns and "dl_rlc_delay_lag1" in X_df.columns:
            X_df["dl_rlc_delay_mean_diff"] = X_df["dl_rlc_delay_roll_mean_5s"] - X_df["dl_rlc_delay_lag1"]
        if "dl_throughput_pred" in X_df.columns and "dl_bler_pred" in X_df.columns:
            X_df["dl_thp_bler_prod"] = X_df["dl_throughput_pred"] * X_df["dl_bler_pred"]
        if "dl_throughput_lag1" in X_df.columns and "dl_rlc_delay_lag1" in X_df.columns:
            X_df["dl_thp_delay_ratio"] = X_df["dl_throughput_lag1"] / (X_df["dl_rlc_delay_lag1"] + 1e-3)

        X_df = X_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        if not self._is_trained or self._feature_names is None:
            self._feature_names = list(X_df.columns)
        else:
            for col in self._feature_names:
                if col not in X_df.columns:
                    X_df[col] = 0.0
            X_df = X_df[self._feature_names]

        return X_df.to_numpy()

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._prepare_features(features)
        y = targets.to_numpy() if isinstance(targets, (pd.Series, pd.DataFrame)) else np.asarray(targets)
        if len(y.shape) > 1:
            y = y.ravel()

        y_log = np.log1p(np.maximum(0.0, y))

        default_params = {
            "max_iter": 300,
            "learning_rate": 0.03,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 15,
            "l2_regularization": 0.5,
            "random_state": 42,
        }
        merged_params = {**default_params, **self.params}

        self._model = HistGradientBoostingRegressor(**merged_params)
        self._model.fit(X, y_log)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self.is_trained or self._model is None:
            raise RuntimeError(f"{self.COMPONENT_NAME} model is not trained yet.")
        X = self._prepare_features(features)
        preds_log = self._model.predict(X)
        preds = np.expm1(preds_log)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        y_true = targets.to_numpy().ravel() if isinstance(targets, (pd.Series, pd.DataFrame)) else np.asarray(targets).ravel()
        y_pred = preds.to_numpy().ravel()
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self._model,
            "is_trained": self._is_trained,
            "feature_names": self._feature_names,
            "params": self.params,
            "model_type": self.model_type,
        }
        joblib.dump(state, p)

    def load(self, path: Union[str, Path]) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found at {p}")
        state = joblib.load(p)
        self._model = state.get("model")
        self._is_trained = state.get("is_trained", True)
        self._feature_names = state.get("feature_names")
        self.params = state.get("params", {})
        self.model_type = state.get("model_type", "hgb")
