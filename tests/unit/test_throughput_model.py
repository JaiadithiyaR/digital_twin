"""Unit tests for ThroughputModel (Module 6, src/dt_models/throughput.py).

Fast, synthetic-data tests of the DTComponent contract mechanics (train/predict/evaluate/
save/load, error paths). The real "trained on genuine bootstrap/mock data, held-out RMSE
reported" proof lives in tests/integration/test_throughput_training.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_settings
from src.dt_models.base import DTComponent
from src.dt_models.throughput import ThroughputModel


def _synthetic_data(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {
            "offered_load_mbps": rng.uniform(10, 100, n),
            "prb_utilization_pct": rng.uniform(0, 100, n),
            "sinr_db": rng.uniform(-10, 30, n),
            "rsrp_dbm": rng.uniform(-120, -60, n),
            "rsrq_db": rng.uniform(-20, -3, n),
            "ue_count": rng.integers(1, 50, n),
            "ue_speed_mps": rng.uniform(0, 30, n),
        }
    )
    # A genuine (if simple) functional relationship so a real regressor can learn something —
    # not pure noise, which would make every test below trivially satisfiable.
    targets = (
        0.6 * features["offered_load_mbps"]
        + 0.2 * (features["sinr_db"] + 10)
        + rng.normal(0, 1.0, n)
    ).rename("throughput_mbps")
    return features, targets


def test_throughput_model_implements_dtcomponent():
    assert isinstance(ThroughputModel(), DTComponent)


def test_untrained_model_reports_not_trained():
    model = ThroughputModel()
    assert model.is_trained is False


def test_predict_before_train_raises():
    model = ThroughputModel()
    features, _ = _synthetic_data(5)
    with pytest.raises(RuntimeError, match="before train"):
        model.predict(features)


def test_train_sets_is_trained_true():
    model = ThroughputModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    assert model.is_trained is True


def test_predict_returns_correctly_named_and_indexed_series():
    model = ThroughputModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    preds = model.predict(features)
    assert preds.name == "throughput_mbps_pred"
    assert list(preds.index) == list(features.index)
    assert len(preds) == len(features)


def test_evaluate_returns_rmse_and_mae():
    model = ThroughputModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    metrics = model.evaluate(features, targets)
    assert set(metrics) == {"rmse", "mae"}
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0


def test_missing_required_feature_raises_on_train():
    model = ThroughputModel()
    features, targets = _synthetic_data()
    with pytest.raises(ValueError, match="missing required feature"):
        model.train(features.drop(columns=["sinr_db"]), targets)


def test_missing_required_feature_raises_on_predict():
    model = ThroughputModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    with pytest.raises(ValueError, match="missing required feature"):
        model.predict(features.drop(columns=["ue_count"]))


def test_nan_rows_are_dropped_before_training_not_silently_propagated():
    model = ThroughputModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data(50)
    features = features.copy()
    features.loc[0, "sinr_db"] = float("nan")
    model.train(features, targets)  # must not raise, and must not crash on the NaN row
    assert model.is_trained is True


def test_all_nan_data_raises_clear_error():
    model = ThroughputModel()
    features, targets = _synthetic_data(5)
    features = features.copy()
    features.loc[:, "sinr_db"] = float("nan")
    with pytest.raises(ValueError, match="no valid"):
        model.train(features, targets)


def test_save_and_load_round_trip_preserves_predictions(tmp_path):
    model = ThroughputModel(params={"n_estimators": 20, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    original_preds = model.predict(features)

    save_path = tmp_path / "throughput_model.joblib"
    model.save(save_path)

    reloaded = ThroughputModel()  # fresh, untrained instance
    assert reloaded.is_trained is False
    reloaded.load(save_path)

    assert reloaded.is_trained is True
    pd.testing.assert_series_equal(reloaded.predict(features), original_preds)


def test_random_forest_model_type_also_works():
    model = ThroughputModel(model_type="random_forest", params={"n_estimators": 10, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    preds = model.predict(features)
    assert len(preds) == len(features)


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="unknown model_type"):
        ThroughputModel(model_type="not_a_real_algorithm")


def test_from_settings_uses_real_config():
    settings = load_settings()
    model = ThroughputModel.from_settings(settings)
    assert model._model_type == settings.dt_models.throughput.model_type
    assert model.is_trained is False
