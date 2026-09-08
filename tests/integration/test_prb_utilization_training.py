"""Integration test: Module 9's PrbUtilizationModel trained on REAL bootstrap/mock data (same
pipeline as Modules 6-8 — see tests/integration/conftest.py), evaluated on a genuine held-out
split with real RMSE/MAE reported, then run through DTOrchestrator TOGETHER with a real trained
ThroughputModel to prove the actual throughput -> prb_utilization dependency chain (explicitly
instructed this turn, unlike Modules 7/8's judgment calls). Also demonstrates the
`enable_prb_model` toggle: disabling this component excludes it from the orchestrator's plan
without affecting throughput, proving the "independent/optional" diagram note holds for real.
"""

from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import mean_squared_error

from src.common.config import load_settings
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator
from src.dt_models.prb_utilization import PrbUtilizationModel
from src.dt_models.throughput import ThroughputModel
from tests.integration.conftest import time_split


def _with_ground_truth_throughput_column(df: pd.DataFrame) -> pd.DataFrame:
    """PrbUtilizationModel expects its throughput input under `THROUGHPUT_INPUT_COLUMN`
    ("throughput_mbps_pred") — at inference time the orchestrator supplies the upstream
    Throughput component's live prediction there; when training directly, the strongest
    available signal is ground-truth historical throughput, supplied under the same column name
    (see src/dt_models/prb_utilization.py's module docstring)."""
    return df.assign(**{PrbUtilizationModel.THROUGHPUT_INPUT_COLUMN: df["throughput_mbps"]})


def test_prb_utilization_model_beats_naive_baseline_on_held_out_real_bootstrap_data(bootstrap_history):
    settings = load_settings()
    assert len(bootstrap_history) >= settings.dt_models.bootstrap_min_rows

    train_df, held_out_df = time_split(bootstrap_history)
    train_with_throughput = _with_ground_truth_throughput_column(train_df)
    held_out_with_throughput = _with_ground_truth_throughput_column(held_out_df)

    model = PrbUtilizationModel.from_settings(settings)
    model.train(train_with_throughput, train_df["prb_utilization_pct"])
    assert model.is_trained

    metrics = model.evaluate(held_out_with_throughput, held_out_df["prb_utilization_pct"])
    rmse, mae = metrics["rmse"], metrics["mae"]
    assert math.isfinite(rmse) and math.isfinite(mae)

    baseline_pred = pd.Series(train_df["prb_utilization_pct"].mean(), index=held_out_df.index)
    baseline_rmse = float(mean_squared_error(held_out_df["prb_utilization_pct"], baseline_pred) ** 0.5)

    print(
        f"\n[Module 9 PrbUtilizationModel] n_train={len(train_df)} n_held_out={len(held_out_df)} "
        f"model_type={settings.dt_models.prb_utilization.model_type} "
        f"held-out RMSE={rmse:.4f} %, MAE={mae:.4f} %, "
        f"naive-mean-baseline RMSE={baseline_rmse:.4f} %, "
        f"held-out target std={held_out_df['prb_utilization_pct'].std():.4f} %"
    )

    assert rmse < baseline_rmse, (
        f"trained model (RMSE={rmse:.4f}) did not beat the naive mean-prediction baseline "
        f"(RMSE={baseline_rmse:.4f}) — it learned nothing"
    )
    assert rmse < float(held_out_df["prb_utilization_pct"].std())


def test_prb_utilization_runs_through_orchestrator_fed_by_real_throughput_predictions(bootstrap_history):
    """The actual dependency-chain proof: throughput trained + registered first,
    prb_utilization second, and the orchestrator must execute throughput before prb_utilization
    and wire throughput's own live predictions into prb_utilization's input — not ground truth."""
    settings = load_settings()
    train_df, held_out_df = time_split(bootstrap_history)

    throughput_model = ThroughputModel.from_settings(settings)
    throughput_model.train(train_df, train_df["throughput_mbps"])

    prb_model = PrbUtilizationModel.from_settings(settings)
    prb_model.train(_with_ground_truth_throughput_column(train_df), train_df["prb_utilization_pct"])

    registry = DTModelRegistry()
    registry.register(throughput_model)
    registry.register(prb_model, enabled=settings.dt_models.enable_prb_model)
    orchestrator = DTOrchestrator(registry)

    assert settings.dt_models.enable_prb_model is True  # config default — sanity-check the premise
    order = orchestrator.build_execution_order()
    assert order.index("throughput") < order.index("prb_utilization")

    predictions = orchestrator.run_predictions(held_out_df)
    assert set(predictions.keys()) == {"throughput", "prb_utilization"}
    assert len(predictions["prb_utilization"]) == len(held_out_df)

    # Reconstruct exactly what the orchestrator should have fed prb_utilization: throughput's
    # own predictions (not ground truth) under THROUGHPUT_INPUT_COLUMN.
    expected_input = held_out_df.assign(**{PrbUtilizationModel.THROUGHPUT_INPUT_COLUMN: predictions["throughput"]})
    expected_preds = prb_model.predict(expected_input)
    pd.testing.assert_series_equal(predictions["prb_utilization"], expected_preds)

    # And confirm it's genuinely different from ground-truth-fed predictions — proving live
    # predictions, not ground truth, actually flowed through the orchestrator.
    ground_truth_fed_preds = prb_model.predict(_with_ground_truth_throughput_column(held_out_df))
    assert not predictions["prb_utilization"].equals(ground_truth_fed_preds)


def test_disabling_prb_utilization_excludes_it_without_breaking_throughput(bootstrap_history):
    """Proves the diagram's "independent/optional" note for real: disabling this component via
    the same enabled=False mechanism config.dt_models.enable_prb_model maps to must exclude it
    from the plan and the result, while the unrelated throughput component keeps working fine."""
    settings = load_settings()
    train_df, held_out_df = time_split(bootstrap_history)

    throughput_model = ThroughputModel.from_settings(settings)
    throughput_model.train(train_df, train_df["throughput_mbps"])

    prb_model = PrbUtilizationModel.from_settings(settings)
    prb_model.train(_with_ground_truth_throughput_column(train_df), train_df["prb_utilization_pct"])

    registry = DTModelRegistry()
    registry.register(throughput_model)
    registry.register(prb_model, enabled=False)  # simulates enable_prb_model: false
    orchestrator = DTOrchestrator(registry)

    order = orchestrator.build_execution_order()
    assert order == ["throughput"]

    predictions = orchestrator.run_predictions(held_out_df)
    assert set(predictions.keys()) == {"throughput"}
    assert len(predictions["throughput"]) == len(held_out_df)
