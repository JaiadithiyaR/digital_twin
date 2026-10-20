import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.dt_models.base import DTComponent


class UlBlerModel(DTComponent):
    COMPONENT_NAME = "ul_bler"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES: tuple[str, ...] = (
        "ul_bler_lag1",
        "ul_bler_lag2",
        "ul_bler_lag3",
        "ul_bler_lag4",
        "ul_bler_lag5",
        "ul_bler_roll_mean_5s",
        "ul_bler_roll_mean_10s",
        "ul_bler_roll_std_5s",
        "ul_bler_roll_std_10s",
        "ul_bler_avg_lag1",
        "ul_bler_avg_lag2",
        "ul_bler_std_lag1",
        "pusch_snr_avg_lag1",
        "pusch_snr_avg_lag2",
        "pusch_snr_avg_roll_mean_5s",
        "pusch_snr_std_lag1",
        "pucch_snr_avg_lag1",
        "pucch_snr_avg_lag2",
        "pucch_snr_avg_roll_mean_5s",
        "pucch_snr_std_lag1",
        "ul_mcs1_avg_lag1",
        "ul_mcs1_avg_lag2",
        "ul_mcs1_avg_roll_mean_5s",
        "ul_mcs1_std_lag1",
        "phr_avg_lag1",
        "phr_avg_lag2",
        "phr_avg_roll_mean_5s",
        "ul_throughput_lag1",
        "ul_throughput_roll_mean_5s",
        "ul_prb_lag1",
        "ul_prb_roll_mean_5s",
        "dl_bler_lag1",
        "dl_bler_roll_mean_5s",
        "bsr_avg_lag1",
        "requested_prb_ratio_lag1",
        "has_controller_lag1",
    )
    OUTPUT_FIELD = "ul_bler_pred"

    def __init__(self, model_type: str = "hist_gb", params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model_type = model_type
        self.params = params or {}
        self._is_trained = False
        self.model_direct: Optional[HistGradientBoostingRegressor] = None
        self.model_residual: Optional[HistGradientBoostingRegressor] = None

    @classmethod
    def from_settings(cls, settings: Any = None) -> "UlBlerModel":
        if settings is not None and hasattr(settings, "dt_models") and hasattr(settings.dt_models, "ul_bler"):
            spec = settings.dt_models.ul_bler
            return cls(
                model_type=getattr(spec, "model_type", "hist_gb"),
                params=getattr(spec, "params", None),
            )
        return cls()

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _prep_features(self, features: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=features.index)
        for col in self.REQUIRED_FEATURES:
            if col in features.columns:
                X[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)
            else:
                X[col] = 0.0
        return X

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        if len(features) == 0 or len(targets) == 0:
            return

        X = self._prep_features(features)
        y = targets.to_numpy(dtype=np.float64)

        lr = self.params.get("learning_rate", 0.03)
        max_iter = self.params.get("max_iter", 200)
        max_depth = self.params.get("max_depth", 5)

        self.model_direct = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=lr,
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=42,
        )
        self.model_direct.fit(X, y)

        if "ul_bler_lag1" in features.columns:
            lag1 = pd.to_numeric(features["ul_bler_lag1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        else:
            lag1 = np.zeros_like(y)

        residuals = y - lag1
        self.model_residual = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=lr,
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=42,
        )
        self.model_residual.fit(X, residuals)

        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained or self.model_direct is None:
            return pd.Series(0.0, index=features.index, name=self.OUTPUT_FIELD)

        X = self._prep_features(features)
        pred_direct = self.model_direct.predict(X)

        if "ul_bler_lag1" in features.columns:
            lag1 = pd.to_numeric(features["ul_bler_lag1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        else:
            lag1 = np.zeros(len(features), dtype=np.float64)

        if self.model_residual is not None:
            pred_res = self.model_residual.predict(X)
            pred_residual = lag1 + pred_res
            final_preds = 0.5 * pred_direct + 0.5 * pred_residual
        else:
            final_preds = pred_direct

        final_preds = np.clip(final_preds, 0.0, 1.0)
        return pd.Series(final_preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        y_true = targets.to_numpy(dtype=np.float64)
        y_pred = preds.to_numpy(dtype=np.float64)

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    def load(self, path: Any) -> None:
        p = Path(path)
        if p.exists():
            obj = joblib.load(p)
            self.__dict__.update(obj.__dict__)
