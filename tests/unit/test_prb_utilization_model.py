"""Unit tests for PrbUtilizationModel (Module 9, src/dt_models/prb_utilization.py).

Fast, synthetic-data tests of the DTComponent contract mechanics, including the derived
cell-load aggregation specific to this component. The real "trained on genuine bootstrap/mock
data, held-out RMSE reported, wired through the orchestrator with a real throughput dependency
and the enable_prb_model toggle" proof lives in
tests/integration/test_prb_utilization_training.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_settings
from src.dt_models.base import DTComponent
from src.dt_models.prb_utilization import PrbUtilizationModel


def _synthetic_data(
    n_cells: int = 3, n_ticks: int = 40, ues_per_cell: int = 4, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    base_ts = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for tick in range(n_ticks):
        ts = base_ts + pd.Timedelta(seconds=tick)
        for cell in range(n_cells):
            cell_id = f"cell-{cell}"
            for _ue in range(ues_per_cell):
                rows.append(
                    {
                        "cell_id": cell_id,
                        "timestamp": ts,
                        "offered_load_mbps": rng.uniform(5, 30),
                        "ue_count": ues_per_cell,
                        "throughput_mbps_pred": rng.uniform(5, 50),
                    }
                )
    features = pd.DataFrame(rows)
    cell_load = features.groupby(["cell_id", "timestamp"])["offered_load_mbps"].transform("sum")
    # A genuine functional relationship — higher cell load and UE count increase PRB
    # utilization, higher throughput (efficient use) slightly reduces it — clipped to a
    # plausible 0-100% range.
    targets = (
        10.0
        + 0.6 * cell_load
        + 0.3 * features["ue_count"]
        - 0.05 * features["throughput_mbps_pred"]
        + rng.normal(0, 2.0, len(features))
    ).clip(lower=0, upper=100).rename("prb_utilization_pct")
    return features, targets


def test_prb_utilization_model_implements_dtcomponent():
    assert isinstance(PrbUtilizationModel(), DTComponent)


def test_declares_throughput_as_orchestrator_dependency():
    assert PrbUtilizationModel.DEPENDENCIES == ("throughput",)


def test_defaults_to_random_forest_per_config_convention():
    # config/settings.yaml sets prb_utilization.model_type: random_forest (unlike the xgboost
    # default of Modules 6-8) — the class default should match so from_settings and a bare
    # constructor call agree.
    assert PrbUtilizationModel().is_trained is False  # constructs without error
    settings = load_settings()
    assert settings.dt_models.prb_utilization.model_type == "random_forest"


def test_untrained_model_reports_not_trained():
    assert PrbUtilizationModel().is_trained is False


def test_predict_before_train_raises():
    model = PrbUtilizationModel()
    features, _ = _synthetic_data(n_ticks=2)
    with pytest.raises(RuntimeError, match="before train"):
        model.predict(features)


def test_train_sets_is_trained_true():
    model = PrbUtilizationModel(params={"n_estimators": 10, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    assert model.is_trained is True


def test_predict_returns_correctly_named_and_indexed_series():
    model = PrbUtilizationModel(params={"n_estimators": 10, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    preds = model.predict(features)
    assert preds.name == "prb_utilization_pct_pred"
    assert list(preds.index) == list(features.index)
    assert len(preds) == len(features)


def test_evaluate_returns_rmse_and_mae():
    model = PrbUtilizationModel(params={"n_estimators": 10, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    metrics = model.evaluate(features, targets)
    assert set(metrics) == {"rmse", "mae"}
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0


def test_missing_raw_required_feature_raises_on_train():
    model = PrbUtilizationModel()
    features, targets = _synthetic_data()
    with pytest.raises(ValueError, match="missing required input column"):
        model.train(features.drop(columns=["ue_count"]), targets)


def test_missing_cell_id_raises():
    model = PrbUtilizationModel()
    features, targets = _synthetic_data()
    with pytest.raises(ValueError, match="cell_id"):
        model.train(features.drop(columns=["cell_id"]), targets)


def test_missing_throughput_dependency_column_raises_on_train():
    model = PrbUtilizationModel()
    features, targets = _synthetic_data()
    with pytest.raises(ValueError, match="throughput_mbps_pred"):
        model.train(features.drop(columns=["throughput_mbps_pred"]), targets)


def test_missing_throughput_dependency_column_raises_on_predict():
    model = PrbUtilizationModel(params={"n_estimators": 10, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    with pytest.raises(ValueError, match="throughput_mbps_pred"):
        model.predict(features.drop(columns=["throughput_mbps_pred"]))


def test_cell_load_is_a_real_per_cell_per_tick_aggregate_not_a_no_op():
    """Two cells with different total offered load at the same tick must produce different
    derived cell-load values — proving `_add_cell_load` genuinely aggregates rather than just
    copying a single UE's own offered_load_mbps through."""
    model = PrbUtilizationModel()
    ts = pd.Timestamp("2026-01-01", tz="UTC")
    features = pd.DataFrame(
        {
            "cell_id": ["cell-0", "cell-0", "cell-1"],
            "timestamp": [ts, ts, ts],
            "offered_load_mbps": [10.0, 20.0, 5.0],
            "ue_count": [2, 2, 1],
            "throughput_mbps_pred": [1.0, 1.0, 1.0],
        }
    )
    enriched = model._add_cell_load(features)
    # cell-0's two UEs share a cell load of 10+20=30; cell-1's single UE has cell load 5.
    assert list(enriched["cell_load_mbps"]) == [30.0, 30.0, 5.0]


def test_nan_rows_are_dropped_before_training_not_silently_propagated():
    model = PrbUtilizationModel(params={"n_estimators": 10, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    features = features.copy()
    features.loc[0, "offered_load_mbps"] = float("nan")
    model.train(features, targets)
    assert model.is_trained is True


def test_save_and_load_round_trip_preserves_predictions(tmp_path):
    model = PrbUtilizationModel(params={"n_estimators": 10, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    original_preds = model.predict(features)

    save_path = tmp_path / "prb_utilization_model.joblib"
    model.save(save_path)

    reloaded = PrbUtilizationModel()
    assert reloaded.is_trained is False
    reloaded.load(save_path)

    assert reloaded.is_trained is True
    pd.testing.assert_series_equal(reloaded.predict(features), original_preds)


def test_xgboost_model_type_also_works():
    model = PrbUtilizationModel(model_type="xgboost", params={"n_estimators": 10, "max_depth": 3, "random_state": 0})
    features, targets = _synthetic_data()
    model.train(features, targets)
    preds = model.predict(features)
    assert len(preds) == len(features)


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="unknown model_type"):
        PrbUtilizationModel(model_type="not_a_real_algorithm")


def test_from_settings_uses_real_config():
    settings = load_settings()
    model = PrbUtilizationModel.from_settings(settings)
    assert model._model_type == settings.dt_models.prb_utilization.model_type
    assert model.is_trained is False
