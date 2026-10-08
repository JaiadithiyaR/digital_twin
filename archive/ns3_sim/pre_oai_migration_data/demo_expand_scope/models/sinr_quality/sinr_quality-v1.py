import logging
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.dt_models.base import DTComponent

logger = logging.getLogger(__name__)


def _build_regressor(model_type: str, params: dict[str, Any]) -> Any:
    if model_type == "xgboost":
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(**params)
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**params)
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**params)


class SinrQualityModel(DTComponent):
    COMPONENT_NAME = "sinr_quality"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES = (
        "offered_load_mbps",
        "prb_utilization_pct",
        "ue_speed_mps",
        "ue_position_x",
        "ue_position_y",
    )
    OUTPUT_FIELD = "sinr_db_pred"

    def __init__(self, model_type: str = "xgboost", params: dict[str, Any] | None = None) -> None:
        self._model_type = model_type
        self._params = dict(params or {})
        self._model = _build_regressor(model_type, self._params)
        self._trained = False

    @classmethod
    def from_settings(cls, settings: Any) -> "SinrQualityModel":
        spec = settings.dt_models.sinr_quality
        return cls(model_type=spec.model_type, params=spec.params)

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.REQUIRED_FEATURES if f not in features.columns]
        if missing:
            raise ValueError(f"SinrQualityModel missing required feature(s): {missing}")
        
        df = features[list(self.REQUIRED_FEATURES)].copy()
        df["distance_from_center"] = np.sqrt(df["ue_position_x"] ** 2 + df["ue_position_y"] ** 2)
        return df

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._select_features(features)
        combined = pd.concat([X, targets.rename("__target__")], axis=1).dropna()
        if combined.empty:
            raise ValueError("SinrQualityModel.train received no valid (non-NaN) rows")
        dropped = len(X) - len(combined)
        if dropped:
            logger.warning(
                "dropped rows with NaN before training",
                extra={"component": "sinr_quality", "dropped_rows": dropped},
            )

        X_clean = combined.drop(columns=["__target__"])
        y_clean = combined["__target__"]
        self._model.fit(X_clean, y_clean)
        self._trained = True
        logger.info(
            "sinr quality model trained",
            extra={"component": "sinr_quality", "model_type": self._model_type, "rows": len(X_clean)},
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._trained:
            raise RuntimeError("SinrQualityModel.predict called before train()/load()")
        X = self._select_features(features)
        preds = self._model.predict(X)
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        aligned_targets = targets.loc[preds.index]
        rmse = float(mean_squared_error(aligned_targets, preds) ** 0.5)
        mae = float(mean_absolute_error(aligned_targets, preds))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self._model, "model_type": self._model_type, "params": self._params, "trained": self._trained},
            path,
        )

    def load(self, path: Path) -> None:
        state = joblib.load(path)
        self._model = state["model"]
        self._model_type = state["model_type"]
        self._params = state["params"]
        self._trained = state["trained"]
