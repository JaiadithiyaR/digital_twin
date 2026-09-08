"""Integration test: Module 6's ThroughputModel trained on REAL bootstrap/mock data that has
flowed through the actual Module 2 -> Module 3 -> D1 pipeline (not hand-built arrays), evaluated
on a genuine held-out split, with the resulting RMSE/MAE asserted to be meaningfully better than
a naive baseline — proving the model actually learned something, not just that some number came
back. Also registers the trained model into a real DTModelRegistry and runs it through
DTOrchestrator, proving Module 6 -> Module 5 wiring end-to-end.

`bootstrap_history` fixture and `time_split` helper live in tests/integration/conftest.py,
shared with Module 7's training test (tests/integration/test_latency_training.py) and future
DT model modules.
"""

from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import mean_squared_error

from src.common.config import load_settings
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator
from src.dt_models.throughput import ThroughputModel
from tests.integration.conftest import time_split


def test_throughput_model_beats_naive_baseline_on_held_out_real_bootstrap_data(bootstrap_history):
    settings = load_settings()
    assert len(bootstrap_history) >= settings.dt_models.bootstrap_min_rows

    train_df, held_out_df = time_split(bootstrap_history)
    assert len(train_df) > 0 and len(held_out_df) > 0

    model = ThroughputModel.from_settings(settings)
    model.train(train_df, train_df["throughput_mbps"])
    assert model.is_trained

    metrics = model.evaluate(held_out_df, held_out_df["throughput_mbps"])
    rmse, mae = metrics["rmse"], metrics["mae"]
    assert math.isfinite(rmse) and math.isfinite(mae)

    # Naive baseline: predict the training-set mean for every held-out row. A model that hasn't
    # learned anything real would not beat this.
    baseline_pred = pd.Series(train_df["throughput_mbps"].mean(), index=held_out_df.index)
    baseline_rmse = float(mean_squared_error(held_out_df["throughput_mbps"], baseline_pred) ** 0.5)

    print(
        f"\n[Module 6 ThroughputModel] n_train={len(train_df)} n_held_out={len(held_out_df)} "
        f"model_type={settings.dt_models.throughput.model_type} "
        f"held-out RMSE={rmse:.4f} Mbps, MAE={mae:.4f} Mbps, "
        f"naive-mean-baseline RMSE={baseline_rmse:.4f} Mbps, "
        f"held-out target std={held_out_df['throughput_mbps'].std():.4f} Mbps"
    )

    assert rmse < baseline_rmse, (
        f"trained model (RMSE={rmse:.4f}) did not beat the naive mean-prediction baseline "
        f"(RMSE={baseline_rmse:.4f}) — it learned nothing"
    )
    # Sanity ceiling: RMSE should also be well within the natural spread of the target.
    assert rmse < float(held_out_df["throughput_mbps"].std())


def test_throughput_model_registered_and_run_through_real_orchestrator(bootstrap_history):
    settings = load_settings()
    train_df, held_out_df = time_split(bootstrap_history)

    model = ThroughputModel.from_settings(settings)
    model.train(train_df, train_df["throughput_mbps"])

    registry = DTModelRegistry()
    registry.register(model)
    orchestrator = DTOrchestrator(registry)

    order = orchestrator.build_execution_order()
    assert order == ["throughput"]

    predictions = orchestrator.run_predictions(held_out_df)
    assert set(predictions.keys()) == {"throughput"}
    assert len(predictions["throughput"]) == len(held_out_df)
    pd.testing.assert_series_equal(predictions["throughput"], model.predict(held_out_df))
