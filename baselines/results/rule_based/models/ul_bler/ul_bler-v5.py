from pathlib import Path
from typing import Union, Tuple, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.dt_models.base import DTComponent


class UlBlerModel(DTComponent):
    COMPONENT_NAME = "ul_bler"
    DEPENDENCIES: Tuple[str, ...] = ()
    REQUIRED_FEATURES: Tuple[str, ...] = (
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
        "ul_bler_avg_lag3",
        "ul_bler_avg_lag4",
        "ul_bler_avg_lag5",
        "ul_bler_avg_roll_mean_5s",
        "ul_bler_avg_roll_mean_10s",
        "ul_bler_avg_roll_std_5s",
        "ul_bler_avg_roll_std_10s",
        "ul_bler_std_lag1",
        "ul_bler_std_lag2",
        "ul_bler_std_lag3",
        "ul_bler_std_lag4",
        "ul_bler_std_lag5",
        "pusch_snr_avg_lag1",
        "pusch_snr_avg_lag2",
        "pusch_snr_avg_lag3",
        "pusch_snr_avg_lag4",
        "pusch_snr_avg_lag5",
        "pusch_snr_avg_roll_mean_5s",
        "pusch_snr_avg_roll_mean_10s",
        "pusch_snr_avg_roll_std_5s",
        "pusch_snr_avg_roll_std_10s",
        "pucch_snr_avg_lag1",
        "pucch_snr_avg_lag2",
        "pucch_snr_avg_lag3",
        "pucch_snr_avg_lag4",
        "pucch_snr_avg_lag5",
        "ul_mcs1_avg_lag1",
        "ul_mcs1_avg_lag2",
        "ul_mcs1_avg_lag3",
        "ul_mcs1_avg_lag4",
        "ul_mcs1_avg_lag5",
        "ul_mcs1_avg_roll_mean_5s",
        "ul_mcs1_avg_roll_mean_10s",
        "phr_avg_lag1",
        "phr_avg_lag2",
        "phr_avg_lag3",
        "phr_avg_lag4",
        "phr_avg_lag5",
        "ul_throughput_lag1",
        "ul_throughput_lag2",
        "ul_throughput_lag3",
        "ul_throughput_roll_mean_5s",
        "ul_throughput_roll_mean_10s",
        "ul_prb_lag1",
        "ul_prb_lag2",
        "ul_prb_lag3",
        "ul_prb_roll_mean_5s",
        "ul_prb_roll_mean_10s",
    )
    OUTPUT_FIELD = "ul_bler_pred"

    def __init__(self, model_type: str = "ensemble", params: Optional[Dict[str, Any]] = None):
        self._is_trained = False
        self.model_type = model_type
        self.params = params or {}
        self.model_1 = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.04,
            max_leaf_nodes=31,
            min_samples_leaf=15,
            l2_regularization=0.1,
            random_state=42,
        )
        self.model_2 = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=0.5,
            random_state=123,
        )

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @classmethod
    def from_settings(cls, settings: Any) -> "UlBlerModel":
        try:
            spec = settings.dt_models.ul_bler
            return cls(
                model_type=getattr(spec, "model_type", "ensemble"),
                params=getattr(spec, "params", None),
            )
        except Exception:
            return cls()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for col in self.REQUIRED_FEATURES:
            if col in df.columns:
                X[col] = df[col].astype(np.float64).fillna(0.0)
            else:
                X[col] = 0.0

        if "pusch_snr_avg_lag1" in X.columns and "pusch_snr_avg_roll_mean_5s" in X.columns:
            X["pusch_snr_delta"] = X["pusch_snr_avg_lag1"] - X["pusch_snr_avg_roll_mean_5s"]
        else:
            X["pusch_snr_delta"] = 0.0

        if "ul_mcs1_avg_lag1" in X.columns and "ul_mcs1_avg_roll_mean_5s" in X.columns:
            X["ul_mcs_delta"] = X["ul_mcs1_avg_lag1"] - X["ul_mcs1_avg_roll_mean_5s"]
        else:
            X["ul_mcs_delta"] = 0.0

        if "ul_bler_lag1" in X.columns and "ul_bler_lag2" in X.columns:
            X["ul_bler_diff1"] = X["ul_bler_lag1"] - X["ul_bler_lag2"]
        else:
            X["ul_bler_diff1"] = 0.0

        if "ul_bler_lag1" in X.columns and "ul_bler_roll_mean_5s" in X.columns:
            X["ul_bler_roll_diff"] = X["ul_bler_lag1"] - X["ul_bler_roll_mean_5s"]
        else:
            X["ul_bler_roll_diff"] = 0.0

        return X

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._prepare_features(features)
        y = targets.values.astype(np.float64)
        self.model_1.fit(X, y)
        self.model_2.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            if "ul_bler_lag1" in features.columns:
                preds = features["ul_bler_lag1"].fillna(0.0).values
            else:
                preds = np.zeros(len(features), dtype=np.float64)
            return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

        X = self._prepare_features(features)
        p1 = self.model_1.predict(X)
        p2 = self.model_2.predict(X)
        preds = 0.55 * p1 + 0.45 * p2
        preds = np.clip(preds, 0.0, 1.0)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        y_true = targets.values
        y_pred = preds.values
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_1": self.model_1,
                "model_2": self.model_2,
                "is_trained": self._is_trained,
                "params": self.params,
            },
            path,
        )

    def load(self, path: Union[str, Path]) -> None:
        path = Path(path)
        data = joblib.load(path)
        self.model_1 = data["model_1"]
        self.model_2 = data["model_2"]
        self._is_trained = data.get("is_trained", True)
        self.params = data.get("params", {})
