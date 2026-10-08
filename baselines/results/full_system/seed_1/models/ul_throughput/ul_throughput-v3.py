import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.dt_models.base import DTComponent


class UlThroughputModel(DTComponent):
    COMPONENT_NAME = "ul_throughput"
    DEPENDENCIES = ("ul_bler", "ul_prb")
    REQUIRED_FEATURES = (
        "ul_throughput_lag1",
        "ul_throughput_lag2",
        "ul_throughput_lag3",
        "ul_throughput_roll_mean_5s",
        "ul_throughput_roll_std_5s",
        "ul_prb_lag1",
        "ul_prb_roll_mean_5s",
        "ul_bler_lag1",
        "ul_bler_roll_mean_5s",
        "pusch_snr_avg_lag1",
        "pusch_snr_std_lag1",
        "pucch_snr_avg_lag1",
        "ul_mcs1_avg_lag1",
        "phr_avg_lag1",
        "bsr_avg_lag1",
        "ue_count_cell_lag1",
    )
    OUTPUT_FIELD = "ul_throughput_pred"

    UL_BLER_INPUT_COLUMN = "ul_bler_pred"
    UL_PRB_INPUT_COLUMN = "ul_prb_pred"

    def __init__(self, model_type: str = "hist_gb", params: dict = None):
        super().__init__()
        self.model_type = model_type
        self.params = params or {
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "random_state": 42,
        }
        self.model = None
        self._feature_names = list(self.REQUIRED_FEATURES) + [
            self.UL_BLER_INPUT_COLUMN,
            self.UL_PRB_INPUT_COLUMN,
        ]

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=features.index)
        for feat in self.REQUIRED_FEATURES:
            if feat in features.columns:
                df[feat] = features[feat]
            else:
                df[feat] = 0.0

        if self.UL_BLER_INPUT_COLUMN in features.columns:
            df[self.UL_BLER_INPUT_COLUMN] = features[self.UL_BLER_INPUT_COLUMN]
        else:
            df[self.UL_BLER_INPUT_COLUMN] = features.get("ul_bler_lag1", 0.0)

        if self.UL_PRB_INPUT_COLUMN in features.columns:
            df[self.UL_PRB_INPUT_COLUMN] = features[self.UL_PRB_INPUT_COLUMN]
        else:
            df[self.UL_PRB_INPUT_COLUMN] = features.get("ul_prb_lag1", 0.0)

        return df.fillna(0.0)

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._prepare_features(features)
        y = targets.to_numpy().ravel()
        self.model = HistGradientBoostingRegressor(**self.params)
        self.model.fit(X, y)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        X = self._prepare_features(features)
        preds = self.model.predict(X)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict:
        preds = self.predict(features)
        rmse = float(np.sqrt(mean_squared_error(targets, preds)))
        mae = float(mean_absolute_error(targets, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "params": self.params,
                "_feature_names": self._feature_names,
            },
            path,
        )

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.model_type = data.get("model_type", "hist_gb")
        self.params = data.get("params", {})
        self._feature_names = data.get("_feature_names", self._feature_names)

    @classmethod
    def from_settings(cls, settings) -> "UlThroughputModel":
        if hasattr(settings, "dt_models") and hasattr(settings.dt_models, "ul_throughput"):
            spec = settings.dt_models.ul_throughput
            return cls(model_type=getattr(spec, "model_type", "hist_gb"), params=getattr(spec, "params", None))
        return cls()
