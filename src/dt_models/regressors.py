"""Shared XGBoost/RandomForest regressor factory for DT prediction components (Modules 6-10).

Kept in one place so each component (throughput, latency, packet_loss, prb_utilization, jitter)
selects its underlying regression algorithm purely from `config.dt_models.<component>.model_type`
(prompt.md §12: "a robust classical ML regression model such as XGBoost or Random Forest")
without duplicating the construction logic five times over.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

_BUILDERS: dict[str, type] = {
    "xgboost": XGBRegressor,
    "random_forest": RandomForestRegressor,
}


def build_regressor(model_type: str, params: dict[str, Any]) -> Any:
    try:
        builder = _BUILDERS[model_type]
    except KeyError:
        raise ValueError(
            f"unknown model_type {model_type!r}; must be one of {sorted(_BUILDERS)}"
        ) from None
    return builder(**params)
