"""Unit tests for `ResidualDTComponent.train`'s row handling: only rows with an undefined residual
target are dropped; rows with missing INPUT features are kept (the regressor handles NaN natively,
and `predict()` is served exactly such rows). `sample_weight` is aligned to the kept rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dt_models.dl_bler import DlBlerModel


def _frame(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in DlBlerModel.REQUIRED_FEATURES})
    target = df["dl_bler_lag1"] + 0.1 * rng.normal(size=n)
    return df, target


def _small_model() -> DlBlerModel:
    return DlBlerModel(model_type="xgboost", params={"n_estimators": 5, "max_depth": 2})


def test_rows_with_missing_input_features_are_kept_for_training():
    df, target = _frame()
    other = next(c for c in DlBlerModel.REQUIRED_FEATURES if c != "dl_bler_lag1")
    df.loc[: len(df) - 2, other] = np.nan  # every row but one lacks this feature
    model = _small_model()
    fitted_rows = []
    original_fit = model._model.fit
    model._model.fit = lambda X, y, **kw: fitted_rows.append(len(X)) or original_fit(X, y, **kw)

    model.train(df, target)

    assert fitted_rows == [len(df)]  # the old "drop any NaN row" behaviour would have kept 1
    assert model.predict(df).notna().all()


def test_rows_with_an_undefined_residual_are_dropped_and_weights_stay_aligned():
    df, target = _frame()
    target.iloc[:50] = np.nan
    df.loc[50:59, "dl_bler_lag1"] = np.nan
    weights = np.arange(len(df), dtype=float)
    model = _small_model()
    captured = {}
    original_fit = model._model.fit

    def fit(X, y, **kw):
        captured.update(rows=len(X), weights=kw.get("sample_weight"))
        return original_fit(X, y, **kw)

    model._model.fit = fit
    model.train(df, target, sample_weight=weights)

    assert captured["rows"] == len(df) - 60
    assert captured["weights"].tolist() == weights[60:].tolist()


def test_training_with_no_defined_residual_raises():
    df, target = _frame()
    target[:] = np.nan
    with pytest.raises(ValueError, match="no rows with a defined residual"):
        _small_model().train(df, target)


def test_mismatched_sample_weight_length_raises():
    df, target = _frame()
    with pytest.raises(ValueError, match="sample_weight"):
        _small_model().train(df, target, sample_weight=np.ones(3))
