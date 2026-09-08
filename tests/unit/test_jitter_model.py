"""Unit tests for JitterModel (Module 10, src/dt_models/jitter.py).

Fast, synthetic-data tests of the DTComponent contract mechanics, including the three-dependency
input requirement specific to this component (the last node in the DT dependency graph). The
real "trained on genuine bootstrap/mock data, held-out RMSE reported, wired through the
orchestrator with all three real dependencies" proof lives in
tests/integration/test_jitter_training.py and tests/integration/test_full_dependency_chain.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_settings
from src.dt_models.base import DTComponent
from src.dt_models.jitter import JitterModel


def _synthetic_data(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {
            "offered_load_mbps": rng.uniform(10, 100, n),
            "ue_count": rng.integers(1, 50, n),
            "throughput_mbps_pred": rng.uniform(5, 100, n),
            "latency_ms_pred": rng.uniform(5, 50, n),
            "packet_loss_pct_pred": rng.uniform(0, 10, n),
        }
    )
    # A genuine functional relationship — jitter is mostly a noisy fraction of latency, matching
    # the mock generator's own jitter_s = latency_s * uniform(0.05, 0.25) formula — so a real
    # regressor has a strong, learnable signal.
    targets = (
        features["latency_ms_pred"] * 0.15
        + 0.02 * features["packet_loss_pct_pred"]
        + rng.normal(0, 0.3, n)
    ).clip(lower=0).rename("jitter_ms")
    return features, targets


def test_jitter_model_implements_dtcomponent():
    assert isinstance(JitterModel(), DTComponent)


def test_declares_all_three_upstream_dependencies():
    assert JitterModel.DEPENDENCIES == ("throughput", "latency", "packet_loss")


def test_untrained_model_reports_not_trained():
    assert JitterModel().is_trained is False


def test_predict_before_train_raises():
    model = JitterModel()
    features, _ = _synthetic_data(5)
    with pytest.raises(RuntimeError, match="before train"):
        model.predict(features)


def test_train_sets_is_trained_true():
    model = JitterModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    assert model.is_trained is True


def test_predict_returns_correctly_named_and_indexed_series():
    model = JitterModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    preds = model.predict(features)
    assert preds.name == "jitter_ms_pred"
    assert list(preds.index) == list(features.index)
    assert len(preds) == len(features)


def test_evaluate_returns_rmse_and_mae():
    model = JitterModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    metrics = model.evaluate(features, targets)
    assert set(metrics) == {"rmse", "mae"}
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0


def test_missing_raw_required_feature_raises_on_train():
    model = JitterModel()
    features, targets = _synthetic_data()
    with pytest.raises(ValueError, match="missing required input column"):
        model.train(features.drop(columns=["ue_count"]), targets)


@pytest.mark.parametrize(
    "dependency_column",
    ["throughput_mbps_pred", "latency_ms_pred", "packet_loss_pct_pred"],
)
def test_missing_any_dependency_column_raises_on_train(dependency_column):
    model = JitterModel()
    features, targets = _synthetic_data()
    with pytest.raises(ValueError, match=dependency_column):
        model.train(features.drop(columns=[dependency_column]), targets)


@pytest.mark.parametrize(
    "dependency_column",
    ["throughput_mbps_pred", "latency_ms_pred", "packet_loss_pct_pred"],
)
def test_missing_any_dependency_column_raises_on_predict(dependency_column):
    model = JitterModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    with pytest.raises(ValueError, match=dependency_column):
        model.predict(features.drop(columns=[dependency_column]))


def test_nan_rows_are_dropped_before_training_not_silently_propagated():
    model = JitterModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data(50)
    features = features.copy()
    features.loc[0, "latency_ms_pred"] = float("nan")
    model.train(features, targets)
    assert model.is_trained is True


def test_all_nan_data_raises_clear_error():
    model = JitterModel()
    features, targets = _synthetic_data(5)
    features = features.copy()
    features.loc[:, "latency_ms_pred"] = float("nan")
    with pytest.raises(ValueError, match="no valid"):
        model.train(features, targets)


def test_save_and_load_round_trip_preserves_predictions(tmp_path):
    model = JitterModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    original_preds = model.predict(features)

    save_path = tmp_path / "jitter_model.joblib"
    model.save(save_path)

    reloaded = JitterModel()
    assert reloaded.is_trained is False
    reloaded.load(save_path)

    assert reloaded.is_trained is True
    pd.testing.assert_series_equal(reloaded.predict(features), original_preds)


def test_random_forest_model_type_also_works():
    model = JitterModel(model_type="random_forest", params={"n_estimators": 10, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    preds = model.predict(features)
    assert len(preds) == len(features)


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="unknown model_type"):
        JitterModel(model_type="not_a_real_algorithm")


def test_from_settings_uses_real_config():
    settings = load_settings()
    model = JitterModel.from_settings(settings)
    assert model._model_type == settings.dt_models.jitter.model_type
    assert model.is_trained is False
