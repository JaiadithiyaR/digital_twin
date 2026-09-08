"""Integration test: Module 7's LatencyModel trained on REAL bootstrap/mock data (same pipeline
as Module 6 — see tests/integration/conftest.py), evaluated on a genuine held-out split with
real RMSE/MAE reported, then run through DTOrchestrator TOGETHER with a real trained
ThroughputModel to prove the actual throughput -> latency dependency chain (prompt.md §13):
latency's predictions come from throughput's LIVE PREDICTIONS wired in by the orchestrator, not
from hand-fed ground truth.
"""

from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import mean_squared_error

from src.common.config import load_settings
from src.dt_models.latency import LatencyModel
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator
from src.dt_models.throughput import ThroughputModel
from tests.integration.conftest import time_split


def _with_ground_truth_throughput_column(df: pd.DataFrame) -> pd.DataFrame:
    """LatencyModel expects its throughput input under `THROUGHPUT_INPUT_COLUMN`
    ("throughput_mbps_pred") — at inference time the orchestrator supplies the upstream
    Throughput component's live prediction there; when training directly, the strongest
    available signal is ground-truth historical throughput, supplied under the same column name
    (see src/dt_models/latency.py's module docstring for the rationale)."""
    return df.assign(**{LatencyModel.THROUGHPUT_INPUT_COLUMN: df["throughput_mbps"]})


def test_latency_model_beats_naive_baseline_on_held_out_real_bootstrap_data(bootstrap_history):
    settings = load_settings()
    assert len(bootstrap_history) >= settings.dt_models.bootstrap_min_rows

    train_df, held_out_df = time_split(bootstrap_history)
    train_with_throughput = _with_ground_truth_throughput_column(train_df)
    held_out_with_throughput = _with_ground_truth_throughput_column(held_out_df)

    model = LatencyModel.from_settings(settings)
    model.train(train_with_throughput, train_df["latency_ms"])
    assert model.is_trained

    metrics = model.evaluate(held_out_with_throughput, held_out_df["latency_ms"])
    rmse, mae = metrics["rmse"], metrics["mae"]
    assert math.isfinite(rmse) and math.isfinite(mae)

    baseline_pred = pd.Series(train_df["latency_ms"].mean(), index=held_out_df.index)
    baseline_rmse = float(mean_squared_error(held_out_df["latency_ms"], baseline_pred) ** 0.5)

    print(
        f"\n[Module 7 LatencyModel] n_train={len(train_df)} n_held_out={len(held_out_df)} "
        f"model_type={settings.dt_models.latency.model_type} "
        f"held-out RMSE={rmse:.4f} ms, MAE={mae:.4f} ms, "
        f"naive-mean-baseline RMSE={baseline_rmse:.4f} ms, "
        f"held-out target std={held_out_df['latency_ms'].std():.4f} ms"
    )

    assert rmse < baseline_rmse, (
        f"trained model (RMSE={rmse:.4f}) did not beat the naive mean-prediction baseline "
        f"(RMSE={baseline_rmse:.4f}) — it learned nothing"
    )
    assert rmse < float(held_out_df["latency_ms"].std())


def test_latency_runs_through_orchestrator_fed_by_real_throughput_predictions(bootstrap_history):
    """The actual dependency-chain proof: throughput trained + registered first, latency second,
    and the orchestrator must execute throughput before latency and wire throughput's own live
    predictions into latency's input — not ground truth, not a hand-fed value."""
    settings = load_settings()
    train_df, held_out_df = time_split(bootstrap_history)

    throughput_model = ThroughputModel.from_settings(settings)
    throughput_model.train(train_df, train_df["throughput_mbps"])

    latency_model = LatencyModel.from_settings(settings)
    latency_model.train(_with_ground_truth_throughput_column(train_df), train_df["latency_ms"])

    registry = DTModelRegistry()
    registry.register(throughput_model)
    registry.register(latency_model)
    orchestrator = DTOrchestrator(registry)

    order = orchestrator.build_execution_order()
    assert order.index("throughput") < order.index("latency")  # dependency order respected

    predictions = orchestrator.run_predictions(held_out_df)
    assert set(predictions.keys()) == {"throughput", "latency"}
    assert len(predictions["latency"]) == len(held_out_df)

    # Reconstruct exactly what the orchestrator should have fed latency: throughput's own
    # predictions (not ground truth) under THROUGHPUT_INPUT_COLUMN.
    expected_latency_input = held_out_df.assign(
        **{LatencyModel.THROUGHPUT_INPUT_COLUMN: predictions["throughput"]}
    )
    expected_latency_preds = latency_model.predict(expected_latency_input)
    pd.testing.assert_series_equal(predictions["latency"], expected_latency_preds)

    # And confirm it's genuinely different from what ground-truth-fed predictions would be
    # whenever the throughput model isn't perfect — proving live predictions, not ground truth,
    # are what actually flowed through (unless throughput happens to be predicted perfectly,
    # which real held-out data with injected noise will not be).
    ground_truth_fed_preds = latency_model.predict(_with_ground_truth_throughput_column(held_out_df))
    assert not predictions["latency"].equals(ground_truth_fed_preds)
