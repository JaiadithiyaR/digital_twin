"""Shared base for all 7 OAI KPI components (claude_code_prompt_phase13.md §1.2): each learns
the one-second RESIDUAL `Δy = y_t − y_{t−1}` rather than the raw level `y_t`, and `predict()`
reconstructs `ŷ_t = y_{t−1} + Δŷ` before returning it — the PUBLIC interface is unchanged
(`predict()` still returns `ŷ_t`, exactly like before this change); only what the underlying
regressor is actually fit against differs. `y_{t−1}` is read from the component's own
`"{COMPONENT_NAME}_lag1"` feature column — already a genuine measured past value present in
every component's `REQUIRED_FEATURES` (`src/data/oai_dataset.ALL_FEATURES`), never re-derived,
estimated, or looked up from anywhere else.

Rationale (phase13 §1.2, read alongside CLAUDE.md's Phase 12 entry): a residual/delta model only
has to explain the CHANGE from one second to the next, not the raw level — and its worst-case
failure mode degrades gracefully toward "predict Δ=0", i.e. exactly the persistence baseline,
rather than doing worse than it. Phase 12's own honest bootstrap finding (only 2 of 7 components
beat persistence when trained directly on `y_t`) is the concrete motivation for this change.

Extracted once here, rather than duplicated across the 7 original per-component files (which
were otherwise byte-for-byte identical in this logic), so this exact fix is applied uniformly —
no component can silently fall out of sync with the other six.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.dt_models.base import DTComponent
from src.dt_models.regressors import build_regressor

logger = logging.getLogger(__name__)


class ResidualDTComponent(DTComponent):
    """Subclasses set `COMPONENT_NAME`/`DEPENDENCIES`/`REQUIRED_FEATURES`/`OUTPUT_FIELD` as class
    attributes exactly as the original 7 component files already did, and override
    `_input_columns` only if they consume additional dependency-prediction columns beyond
    `REQUIRED_FEATURES` (e.g. `dl_throughput` needs `dl_bler_pred`/`dl_prb_pred` too)."""

    COMPONENT_NAME: ClassVar[str]
    DEPENDENCIES: ClassVar[tuple[str, ...]] = ()
    REQUIRED_FEATURES: ClassVar[tuple[str, ...]] = ()
    OUTPUT_FIELD: ClassVar[str]

    def __init__(self, model_type: str = "xgboost", params: dict[str, Any] | None = None) -> None:
        self._model_type = model_type
        self._params = dict(params or {})
        self._model = build_regressor(model_type, self._params)
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def _lag1_column(self) -> str:
        """`y_{t-1}` for this component's own target — the residual-reconstruction anchor."""
        return f"{self.COMPONENT_NAME}_lag1"

    @property
    def _input_columns(self) -> tuple[str, ...]:
        return tuple(self.REQUIRED_FEATURES)

    def _select_features(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self._input_columns if c not in features.columns]
        if missing:
            raise ValueError(f"{type(self).__name__} missing required input column(s): {missing}")
        return features[list(self._input_columns)]

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        X = self._select_features(features)
        if self._lag1_column not in X.columns:
            raise ValueError(
                f"{type(self).__name__}.train requires its own lag1 column {self._lag1_column!r} "
                "(in REQUIRED_FEATURES) to compute the residual target — missing"
            )
        residual = targets - X[self._lag1_column]
        combined = pd.concat([X, residual.rename("__residual__")], axis=1).dropna()
        if combined.empty:
            raise ValueError(f"{type(self).__name__}.train received no valid (non-NaN) rows")
        dropped = len(X) - len(combined)
        if dropped:
            logger.warning(
                "dropped rows with NaN before training", extra={"component": self.COMPONENT_NAME, "dropped_rows": dropped}
            )

        X_clean = combined[list(self._input_columns)]
        residual_clean = combined["__residual__"]
        self._model.fit(X_clean, residual_clean)
        self._trained = True
        logger.info(
            "model trained on residual target (y_t - y_t-1)",
            extra={"component": self.COMPONENT_NAME, "model_type": self._model_type, "rows": len(X_clean)},
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._trained:
            raise RuntimeError(f"{type(self).__name__}.predict called before train()/load()")
        X = self._select_features(features)
        residual_pred = self._model.predict(X)
        # Public interface: always reconstruct and return ŷ_t = y_{t-1} + Δŷ, never the raw
        # residual. Honestly NaN (never fabricated) wherever y_{t-1} itself is unknown — e.g. the
        # first row of a session, where there genuinely is no prior second to anchor the
        # reconstruction to. XGBoost's own missing-value handling lets `self._model.predict(X)`
        # still return a real residual even when SOME OTHER input feature is NaN, which is why
        # this NaN-propagation has to be applied explicitly here, one level up, rather than
        # relying on the regressor to surface it on its own.
        y_prev = X[self._lag1_column].to_numpy()
        y_pred = y_prev + residual_pred
        return pd.Series(y_pred, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        preds = self.predict(features)
        aligned_targets = targets.loc[preds.index]
        # A NaN reconstructed prediction (see predict()'s own docstring note) can never
        # contribute to a metric — mask it out rather than letting it NaN-poison the whole
        # RMSE/MAE computation, mirroring how every other metric computation in this project
        # (e.g. Module 12's own `_validate_inputs`) never silently manufactures a score from
        # invalid data, and never silently crashes on it either.
        mask = aligned_targets.notna() & preds.notna()
        if not mask.any():
            return {"rmse": float("nan"), "mae": float("nan")}
        rmse = float(mean_squared_error(aligned_targets[mask], preds[mask]) ** 0.5)
        mae = float(mean_absolute_error(aligned_targets[mask], preds[mask]))
        return {"rmse": rmse, "mae": mae}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "model_type": self._model_type, "params": self._params, "trained": self._trained}, path)

    def load(self, path: Path) -> None:
        state = joblib.load(path)
        self._model = state["model"]
        self._model_type = state["model_type"]
        self._params = state["params"]
        self._trained = state["trained"]
