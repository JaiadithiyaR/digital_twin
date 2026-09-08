"""Integration test: Module 8's PacketLossModel trained on REAL bootstrap/mock data (same
pipeline as Modules 6-7 — see tests/integration/conftest.py), evaluated on a genuine held-out
split with real RMSE/MAE reported, and registered/run through a real DTOrchestrator alongside
Module 6's ThroughputModel — both root nodes, proving the orchestrator handles multiple
independent components correctly (not just a single-node or a linear chain).
"""

from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import mean_squared_error

from src.common.config import load_settings
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator
from src.dt_models.packet_loss import PacketLossModel
from src.dt_models.throughput import ThroughputModel
from tests.integration.conftest import time_split


def test_packet_loss_model_beats_naive_baseline_on_held_out_real_bootstrap_data(bootstrap_history):
    settings = load_settings()
    assert len(bootstrap_history) >= settings.dt_models.bootstrap_min_rows

    train_df, held_out_df = time_split(bootstrap_history)
    assert len(train_df) > 0 and len(held_out_df) > 0

    model = PacketLossModel.from_settings(settings)
    model.train(train_df, train_df["packet_loss_pct"])
    assert model.is_trained

    metrics = model.evaluate(held_out_df, held_out_df["packet_loss_pct"])
    rmse, mae = metrics["rmse"], metrics["mae"]
    assert math.isfinite(rmse) and math.isfinite(mae)

    baseline_pred = pd.Series(train_df["packet_loss_pct"].mean(), index=held_out_df.index)
    baseline_rmse = float(mean_squared_error(held_out_df["packet_loss_pct"], baseline_pred) ** 0.5)

    print(
        f"\n[Module 8 PacketLossModel] n_train={len(train_df)} n_held_out={len(held_out_df)} "
        f"model_type={settings.dt_models.packet_loss.model_type} "
        f"held-out RMSE={rmse:.4f} %, MAE={mae:.4f} %, "
        f"naive-mean-baseline RMSE={baseline_rmse:.4f} %, "
        f"held-out target std={held_out_df['packet_loss_pct'].std():.4f} %"
    )

    assert rmse < baseline_rmse, (
        f"trained model (RMSE={rmse:.4f}) did not beat the naive mean-prediction baseline "
        f"(RMSE={baseline_rmse:.4f}) — it learned nothing"
    )
    assert rmse < float(held_out_df["packet_loss_pct"].std())


def test_packet_loss_and_throughput_both_registered_and_run_through_orchestrator(bootstrap_history):
    """Two independent root-node components in one registry — proves the orchestrator handles
    multiple unrelated components correctly, not just a chain."""
    settings = load_settings()
    train_df, held_out_df = time_split(bootstrap_history)

    throughput_model = ThroughputModel.from_settings(settings)
    throughput_model.train(train_df, train_df["throughput_mbps"])

    packet_loss_model = PacketLossModel.from_settings(settings)
    packet_loss_model.train(train_df, train_df["packet_loss_pct"])

    registry = DTModelRegistry()
    registry.register(throughput_model)
    registry.register(packet_loss_model)
    orchestrator = DTOrchestrator(registry)

    order = orchestrator.build_execution_order()
    assert set(order) == {"throughput", "packet_loss"}  # both root nodes; order between them unconstrained

    predictions = orchestrator.run_predictions(held_out_df)
    assert set(predictions.keys()) == {"throughput", "packet_loss"}
    assert len(predictions["packet_loss"]) == len(held_out_df)
    pd.testing.assert_series_equal(predictions["packet_loss"], packet_loss_model.predict(held_out_df))
    pd.testing.assert_series_equal(predictions["throughput"], throughput_model.predict(held_out_df))
