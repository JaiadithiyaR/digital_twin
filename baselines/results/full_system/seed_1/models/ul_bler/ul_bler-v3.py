import joblib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.dt_models.base import DTComponent


class UlBlerModel(DTComponent):
    COMPONENT_NAME: str = "ul_bler"
    DEPENDENCIES: Tuple[str, ...] = ()
    REQUIRED_FEATURES: Tuple[str, ...] = (
        "ul_bler_lag1",
        "ul_bler_lag2",
        "ul_bler_lag3",
        "ul_bler_lag4",
        "ul_bler_lag5",
        "ul_bler_roll_mean_5s",
        "ul_bler_roll_std_5s",
        "ul_bler_roll_mean_10s",
        "ul_bler_roll_std_10s",
        "ul_bler_avg_lag1",
        "ul_bler_std_lag1",
        "pusch_snr_avg_lag1",
        "pusch_snr_std_lag1",
        "pusch_snr_avg_roll_mean_5s",
        "pucch_snr_avg_lag1",
        "pucch_snr_std_lag1",
        "ul_mcs1_avg_lag1",
        "ul_mcs1_std_lag1",
        "phr_avg_lag1",
        "phr_std_lag1",
        "ul_throughput_lag1",
        "ul_prb_lag1",
        "dl_bler_lag1",
        "bsr_avg_lag1",
    )
    OUTPUT_FIELD: str = "ul_bler_pred"

    def __init__(self, model_type: str = "hist_gb", params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model_type = model_type
        self.params = params or {}
        self.model = None
        self._is_trained = False

    @classmethod
    def from_settings(cls, settings: Any) -> "UlBlerModel":
        spec = settings.dt_models.ul_bler
        return cls(model_type=getattr(spec, "model_type", "hist_gb"), params=getattr(spec, "params", {}))

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = features.reindex(columns=list(self.REQUIRED_FEATURES), fill_value=0.0).fillna(0.0)
        y = targets.fillna(0.0)

        default_params = {
            "max_iter": 150,
            "learning_rate": 0.05,
            "max_depth": 6,
            "random_state": 42,
        }
        if self.params:
            default_params.update(self.params)

        self.model = HistGradientBoostingRegressor(**default_params)
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._is_trained or self.model is None:
            raise ValueError("Model is not trained.")
        X = features.reindex(columns=list(self.REQUIRED_FEATURES), fill_value=0.0).fillna(0.0)
        preds = self.model.predict(X)
        preds = np.clip(preds, 0.0, 1.0)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        preds = self.predict(features)
        y_true = targets.fillna(0.0)
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        mae = float(mean_absolute_error(y_true, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "is_trained": self._is_trained,
                "model_type": self.model_type,
                "params": self.params,
            },
            path,
        )

    def load(self, path: Union[str, Path]) -> None:
        path = Path(path)
        data = joblib.load(path)
        self.model = data.get("model")
        self._is_trained = data.get("is_trained", True)
        self.model_type = data.get("model_type", "hist_gb")
        self.params = data.get("params", {})
