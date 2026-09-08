"""Unit tests for LatencyModel (Module 7, src/dt_models/latency.py).

Fast, synthetic-data tests of the DTComponent contract mechanics, including the
throughput-dependency-column requirement specific to this component. The real "trained on
genuine bootstrap/mock data, held-out RMSE reported, wired through the orchestrator with a real
throughput dependency" proof lives in tests/integration/test_latency_training.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_settings
from src.dt_models.base import DTComponent
from src.dt_models.latency import LatencyModel


def _synthetic_data(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {
            "offered_load_mbps": rng.uniform(10, 100, n),
            "prb_utilization_pct": rng.uniform(0, 100, n),
            "ue_count": rng.integers(1, 50, n),
            "packet_loss_pct": rng.uniform(0, 10, n),
            "sinr_db": rng.uniform(-10, 30, n),
            "throughput_mbps_pred": rng.uniform(5, 100, n),
        }
    )
    # A genuine functional relationship — higher PRB load and packet loss increase latency,
    # higher throughput reduces it — so a real regressor has something to learn.
    targets = (
        5.0
        + 0.08 * features["prb_utilization_pct"]
        + 0.5 * features["packet_loss_pct"]
        - 0.03 * features["throughput_mbps_pred"]
        + rng.normal(0, 0.5, n)
    ).rename("latency_ms")
    return features, targets


def test_latency_model_implements_dtcomponent():
    assert isinstance(LatencyModel(), DTComponent)


def test_declares_throughput_as_orchestrator_dependency():
    assert LatencyModel.DEPENDENCIES == ("throughput",)


def test_untrained_model_reports_not_trained():
    assert LatencyModel().is_trained is False


def test_predict_before_train_raises():
    model = LatencyModel()
    features, _ = _synthetic_data(5)
    with pytest.raises(RuntimeError, match="before train"):
        model.predict(features)


def test_train_sets_is_trained_true():
    model = LatencyModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    assert model.is_trained is True


def test_predict_returns_correctly_named_and_indexed_series():
    model = LatencyModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    preds = model.predict(features)
    assert preds.name == "latency_ms_pred"
    assert list(preds.index) == list(features.index)
    assert len(preds) == len(features)


def test_evaluate_returns_rmse_and_mae():
    model = LatencyModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    metrics = model.evaluate(features, targets)
    assert set(metrics) == {"rmse", "mae"}
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0


def test_missing_raw_required_feature_raises_on_train():
    model = LatencyModel()
    features, targets = _synthetic_data()
    with pytest.raises(ValueError, match="missing required input column"):
        model.train(features.drop(columns=["sinr_db"]), targets)


def test_missing_throughput_dependency_column_raises_on_train():
    """The throughput dependency column is not in REQUIRED_FEATURES but IS required to actually
    fit/predict — dropping it must raise just as clearly as a missing raw feature."""
    model = LatencyModel()
    features, targets = _synthetic_data()
    with pytest.raises(ValueError, match="throughput_mbps_pred"):
        model.train(features.drop(columns=["throughput_mbps_pred"]), targets)


def test_missing_throughput_dependency_column_raises_on_predict():
    model = LatencyModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    with pytest.raises(ValueError, match="throughput_mbps_pred"):
        model.predict(features.drop(columns=["throughput_mbps_pred"]))


def test_nan_rows_are_dropped_before_training_not_silently_propagated():
    model = LatencyModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data(50)
    features = features.copy()
    features.loc[0, "packet_loss_pct"] = float("nan")
    model.train(features, targets)
    assert model.is_trained is True


def test_all_nan_data_raises_clear_error():
    model = LatencyModel()
    features, targets = _synthetic_data(5)
    features = features.copy()
    features.loc[:, "sinr_db"] = float("nan")
    with pytest.raises(ValueError, match="no valid"):
        model.train(features, targets)


def test_save_and_load_round_trip_preserves_predictions(tmp_path):
    model = LatencyModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    original_preds = model.predict(features)

    save_path = tmp_path / "latency_model.joblib"
    model.save(save_path)

    reloaded = LatencyModel()
    assert reloaded.is_trained is False
    reloaded.load(save_path)

    assert reloaded.is_trained is True
    pd.testing.assert_series_equal(reloaded.predict(features), original_preds)


def test_random_forest_model_type_also_works():
    model = LatencyModel(model_type="random_forest", params={"n_estimators": 10, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    preds = model.predict(features)
    assert len(preds) == len(features)


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="unknown model_type"):
        LatencyModel(model_type="not_a_real_algorithm")


def test_from_settings_uses_real_config():
    settings = load_settings()
    model = LatencyModel.from_settings(settings)
    assert model._model_type == settings.dt_models.latency.model_type
    assert model.is_trained is False
