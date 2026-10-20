import joblib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.dt_models.base import DTComponent


class DlRlcDelayModel(DTComponent):
    COMPONENT_NAME = "dl_rlc_delay"
    DEPENDENCIES = ("dl_throughput", "dl_bler")
    OUTPUT_FIELD = "dl_rlc_delay_pred"

    REQUIRED_FEATURES = (
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

    DL_BLER_INPUT_COLUMN = "dl_bler_pred"
    DL_THROUGHPUT_INPUT_COLUMN = "dl_throughput_pred"

    def __init__(self, model_type: str = "hist_gb", params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model_type = model_type
        self.params = params or {
            "max_iter": 250,
            "learning_rate": 0.04,
            "max_depth": 6,
            "min_samples_leaf": 15,
            "l2_regularization": 0.5,
            "random_state": 42,
        }
        self.model: Optional[HistGradientBoostingRegressor] = None
        self._is_trained: bool = False
        self._feature_cols: list[str] = []

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @classmethod
    def from_settings(cls, settings: Any = None) -> "DlRlcDelayModel":
        if settings is not None and hasattr(settings, "dt_models") and hasattr(settings.dt_models, "dl_rlc_delay"):
            spec = settings.dt_models.dl_rlc_delay
            return cls(model_type=getattr(spec, "model_type", "hist_gb"), params=getattr(spec, "params", None))
        return cls()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        input_cols = list(self.REQUIRED_FEATURES) + [self.DL_BLER_INPUT_COLUMN, self.DL_THROUGHPUT_INPUT_COLUMN]
        X = pd.DataFrame(index=df.index)
        for c in input_cols:
            if c in df.columns:
                X[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                X[c] = 0.0

        if "dl_rlc_delay_lag1" in X.columns and "dl_rlc_delay_lag2" in X.columns:
            X["dl_rlc_delay_diff1_2"] = X["dl_rlc_delay_lag1"] - X["dl_rlc_delay_lag2"]
        if "dl_rlc_delay_lag1" in X.columns and "dl_rlc_delay_roll_mean_5s" in X.columns:
            X["dl_rlc_delay_roll_diff5"] = X["dl_rlc_delay_lag1"] - X["dl_rlc_delay_roll_mean_5s"]
        if self.DL_THROUGHPUT_INPUT_COLUMN in X.columns and "dl_throughput_lag1" in X.columns:
            X["dl_thp_pred_diff"] = X[self.DL_THROUGHPUT_INPUT_COLUMN] - X["dl_throughput_lag1"]
        if self.DL_BLER_INPUT_COLUMN in X.columns and "dl_bler_lag1" in X.columns:
            X["dl_bler_pred_diff"] = X[self.DL_BLER_INPUT_COLUMN] - X["dl_bler_lag1"]

        return X

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._prepare_features(features)
        y = pd.to_numeric(targets, errors="coerce").fillna(0.0).to_numpy()
        y_log = np.log1p(np.maximum(0.0, y))

        self._feature_cols = list(X.columns)
        self.model = HistGradientBoostingRegressor(**self.params)
        self.model.fit(X, y_log)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained or self.model is None:
            raise RuntimeError(f"{self.__class__.__name__} is not trained yet.")

        X = self._prepare_features(features)
        for c in self._feature_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[self._feature_cols]

        pred_log = self.model.predict(X)
        preds = np.expm1(np.maximum(0.0, pred_log))
        preds = np.maximum(0.0, preds)

        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        y = pd.to_numeric(targets, errors="coerce").to_numpy()
        p = preds.to_numpy()
        mask = ~np.isnan(y) & ~np.isnan(p)
        if not np.any(mask):
            return {"rmse": 0.0, "mae": 0.0}
        y_c, p_c = y[mask], p[mask]
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_c, p_c))),
            "mae": float(mean_absolute_error(y_c, p_c)),
        }

    def save(self, path: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "is_trained": self._is_trained,
            "feature_cols": self._feature_cols,
            "params": self.params,
            "model_type": self.model_type,
        }, p)

    def load(self, path: Any) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No model checkpoint at {p}")
        data = joblib.load(p)
        self.model = data.get("model")
        self._is_trained = data.get("is_trained", False)
        self._feature_cols = data.get("feature_cols", [])
        self.params = data.get("params", self.params)
        self.model_type = data.get("model_type", self.model_type)
