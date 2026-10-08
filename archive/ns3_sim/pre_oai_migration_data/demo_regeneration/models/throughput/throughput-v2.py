import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.dt_models.base import DTComponent

logger = logging.getLogger(__name__)


class ThroughputModel(DTComponent):
    COMPONENT_NAME = "throughput"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES = (
        "offered_load_mbps",
        "prb_utilization_pct",
        "sinr_db",
        "rsrp_dbm",
        "rsrq_db",
        "ue_count",
        "ue_speed_mps",
        "ue_position_x",
        "ue_position_y",
    )
    OUTPUT_FIELD = "throughput_mbps_pred"

    def __init__(self, model_type: str = "gradient_boosting", params: dict[str, Any] | None = None) -> None:
        self._model_type = model_type
        self._params = dict(params or {})
        params_copy = dict(self._params)
        if "random_state" not in params_copy:
            params_copy["random_state"] = 42

        if model_type == "hist_gb":
            self._model = HistGradientBoostingRegressor(**params_copy)
        elif model_type == "random_forest":
            self._model = RandomForestRegressor(**params_copy)
        else:
            default_params = {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.05, "random_state": 42}
            default_params.update(self._params)
            self._model = GradientBoostingRegressor(**default_params)

        self._trained = False

    @classmethod
    def from_settings(cls, settings: Any) -> "ThroughputModel":
        try:
            spec = settings.dt_models.throughput
            return cls(model_type=getattr(spec, "model_type", "gradient_boosting"), params=getattr(spec, "params", None))
        except Exception:
            return cls()

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.REQUIRED_FEATURES if f not in features.columns]
        if missing:
            raise ValueError(f"ThroughputModel missing required feature(s): {missing}")
        return features[list(self.REQUIRED_FEATURES)]

    def _engineer_features(self, features: pd.DataFrame) -> pd.DataFrame:
        X = features.copy()
        x_pos = X["ue_position_x"]
        y_pos = X["ue_position_y"]
        X["ue_distance"] = np.sqrt(x_pos**2 + y_pos**2)
        sinr_lin = 10.0 ** (np.clip(X["sinr_db"], -20, 40) / 10.0)
        X["spectral_eff"] = np.log2(1.0 + sinr_lin)
        X["load_prb_ratio"] = X["offered_load_mbps"] / (X["prb_utilization_pct"] + 1e-3)
        X["sinr_rsrp_ratio"] = X["sinr_db"] / (np.abs(X["rsrp_dbm"]) + 1e-3)
        return X

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._select_features(features)
        X_eng = self._engineer_features(X)
        combined = pd.concat([X_eng, targets.rename("__target__")], axis=1).dropna()
        if combined.empty:
            raise ValueError("ThroughputModel.train received no valid (non-NaN) rows")
        dropped = len(X_eng) - len(combined)
        if dropped:
            logger.warning(
                "dropped rows with NaN before training",
                extra={"component": "throughput_model", "dropped_rows": dropped},
            )

        X_clean = combined.drop(columns=["__target__"])
        y_clean = combined["__target__"]
        self._model.fit(X_clean, y_clean)
        self._trained = True
        logger.info(
            "throughput model trained",
            extra={"component": "throughput_model", "model_type": self._model_type, "rows": len(X_clean)},
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._trained:
            raise RuntimeError("ThroughputModel.predict called before train()/load()")
        X = self._select_features(features)
        X_eng = self._engineer_features(X)
        preds = self._model.predict(X_eng)
        preds = np.maximum(0.0, preds)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        aligned_targets = targets.loc[preds.index]
        rmse = float(mean_squared_error(aligned_targets, preds) ** 0.5)
        mae = float(mean_absolute_error(aligned_targets, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self._model, "model_type": self._model_type, "params": self._params, "trained": self._trained},
            path,
        )

    def load(self, path: Path) -> None:
        path = Path(path)
        state = joblib.load(path)
        self._model = state["model"]
        self._model_type = state["model_type"]
        self._params = state["params"]
        self._trained = state["trained"]
