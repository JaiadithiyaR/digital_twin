"""Integration test: Module 10's JitterModel trained on REAL bootstrap/mock data (same pipeline
as Modules 6-9 — see tests/integration/conftest.py), evaluated on a genuine held-out split with
real RMSE/MAE reported, then run through DTOrchestrator TOGETHER with real trained
ThroughputModel/LatencyModel/PacketLossModel to prove the actual three-way dependency chain
(prompt.md §13: "throughput + latency + packet loss -> jitter") — the last node in the DT
dependency graph. The full five-component chain (adding PrbUtilizationModel too) is proven
separately in tests/integration/test_full_dependency_chain.py.
"""

from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import mean_squared_error

from src.common.config import load_settings
from src.dt_models.jitter import JitterModel
from src.dt_models.latency import LatencyModel
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator
from src.dt_models.packet_loss import PacketLossModel
from src.dt_models.throughput import ThroughputModel
from tests.integration.conftest import time_split


def _with_ground_truth_dependency_columns(df: pd.DataFrame) -> pd.DataFrame:
    """JitterModel expects its three dependency inputs under THROUGHPUT_INPUT_COLUMN/
    LATENCY_INPUT_COLUMN/PACKET_LOSS_INPUT_COLUMN — at inference time the orchestrator supplies
    each upstream component's live prediction there; when training directly, the strongest
    available signal is ground-truth historical values, supplied under the same column names."""
    return df.assign(
        **{
            JitterModel.THROUGHPUT_INPUT_COLUMN: df["throughput_mbps"],
            JitterModel.LATENCY_INPUT_COLUMN: df["latency_ms"],
            JitterModel.PACKET_LOSS_INPUT_COLUMN: df["packet_loss_pct"],
        }
    )


def test_jitter_model_beats_naive_baseline_on_held_out_real_bootstrap_data(bootstrap_history):
    settings = load_settings()
    assert len(bootstrap_history) >= settings.dt_models.bootstrap_min_rows

    train_df, held_out_df = time_split(bootstrap_history)
    train_enriched = _with_ground_truth_dependency_columns(train_df)
    held_out_enriched = _with_ground_truth_dependency_columns(held_out_df)

    model = JitterModel.from_settings(settings)
    model.train(train_enriched, train_df["jitter_ms"])
    assert model.is_trained

    metrics = model.evaluate(held_out_enriched, held_out_df["jitter_ms"])
    rmse, mae = metrics["rmse"], metrics["mae"]
    assert math.isfinite(rmse) and math.isfinite(mae)

    baseline_pred = pd.Series(train_df["jitter_ms"].mean(), index=held_out_df.index)
    baseline_rmse = float(mean_squared_error(held_out_df["jitter_ms"], baseline_pred) ** 0.5)

    print(
        f"\n[Module 10 JitterModel] n_train={len(train_df)} n_held_out={len(held_out_df)} "
        f"model_type={settings.dt_models.jitter.model_type} "
        f"held-out RMSE={rmse:.4f} ms, MAE={mae:.4f} ms, "
        f"naive-mean-baseline RMSE={baseline_rmse:.4f} ms, "
        f"held-out target std={held_out_df['jitter_ms'].std():.4f} ms"
    )

    assert rmse < baseline_rmse, (
        f"trained model (RMSE={rmse:.4f}) did not beat the naive mean-prediction baseline "
        f"(RMSE={baseline_rmse:.4f}) — it learned nothing"
    )
    assert rmse < float(held_out_df["jitter_ms"].std())


def test_jitter_runs_through_orchestrator_fed_by_real_upstream_predictions(bootstrap_history):
    """The three-way dependency-chain proof: throughput, latency, and packet_loss trained +
    registered first, jitter last, and the orchestrator must execute jitter after all three and
    wire their own live predictions into jitter's input — not ground truth."""
    settings = load_settings()
    train_df, held_out_df = time_split(bootstrap_history)

    throughput_model = ThroughputModel.from_settings(settings)
    throughput_model.train(train_df, train_df["throughput_mbps"])

    packet_loss_model = PacketLossModel.from_settings(settings)
    packet_loss_model.train(train_df, train_df["packet_loss_pct"])

    latency_model = LatencyModel.from_settings(settings)
    latency_train_input = train_df.assign(**{LatencyModel.THROUGHPUT_INPUT_COLUMN: train_df["throughput_mbps"]})
    latency_model.train(latency_train_input, train_df["latency_ms"])

    jitter_model = JitterModel.from_settings(settings)
    jitter_model.train(_with_ground_truth_dependency_columns(train_df), train_df["jitter_ms"])

    registry = DTModelRegistry()
    registry.register(throughput_model)
    registry.register(packet_loss_model)
    registry.register(latency_model)
    registry.register(jitter_model)
    orchestrator = DTOrchestrator(registry)

    order = orchestrator.build_execution_order()
    assert order.index("throughput") < order.index("jitter")
    assert order.index("latency") < order.index("jitter")
    assert order.index("packet_loss") < order.index("jitter")
    assert order.index("throughput") < order.index("latency")  # latency's own dependency

    predictions = orchestrator.run_predictions(held_out_df)
    assert set(predictions.keys()) == {"throughput", "packet_loss", "latency", "jitter"}
    assert len(predictions["jitter"]) == len(held_out_df)

    # Reconstruct exactly what the orchestrator should have fed jitter: each upstream's own live
    # predictions (not ground truth) under the three dependency column names.
    expected_input = held_out_df.assign(
        **{
            JitterModel.THROUGHPUT_INPUT_COLUMN: predictions["throughput"],
            JitterModel.LATENCY_INPUT_COLUMN: predictions["latency"],
            JitterModel.PACKET_LOSS_INPUT_COLUMN: predictions["packet_loss"],
        }
    )
    expected_preds = jitter_model.predict(expected_input)
    pd.testing.assert_series_equal(predictions["jitter"], expected_preds)

    # Confirm it's genuinely different from ground-truth-fed predictions — proving live upstream
    # predictions, not ground truth, actually flowed through the orchestrator.
    ground_truth_fed_preds = jitter_model.predict(_with_ground_truth_dependency_columns(held_out_df))
    assert not predictions["jitter"].equals(ground_truth_fed_preds)
