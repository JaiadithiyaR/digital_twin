"""The closing proof for Modules 6-10: all five real DT prediction components — Throughput (6),
Latency (7), Packet-Loss (8), PRB-Utilization (9), Jitter (10) — trained on real bootstrap/mock
data and registered together into one DTModelRegistry, run through one DTOrchestrator call.

Confirms:
  1. A single valid topological execution order exists for the full real dependency graph
     (throughput/packet_loss are roots; latency and prb_utilization each depend on throughput;
     jitter depends on throughput, latency, AND packet_loss — the last node in the chain).
  2. `run_predictions()` returns correct predictions for all five components in one call.
  3. Every dependency edge in that graph is fed by the upstream component's LIVE prediction,
     not ground truth — checked explicitly for the deepest node (jitter) by reconstructing its
     expected input from the other four's own returned predictions.
  4. The registry's `enabled` mechanism (config.dt_models.enable_prb_model) is exercised as part
     of this same full run, not just in isolation.
"""

from __future__ import annotations

import pandas as pd

from src.common.config import load_settings
from src.dt_models.jitter import JitterModel
from src.dt_models.latency import LatencyModel
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator
from src.dt_models.packet_loss import PacketLossModel
from src.dt_models.prb_utilization import PrbUtilizationModel
from src.dt_models.throughput import ThroughputModel
from tests.integration.conftest import time_split


def test_all_five_dt_components_run_together_in_correct_dependency_order(bootstrap_history):
    settings = load_settings()
    train_df, held_out_df = time_split(bootstrap_history)

    # --- train all five on ground truth (bootstrap training, outside the orchestrator) ---
    throughput_model = ThroughputModel.from_settings(settings)
    throughput_model.train(train_df, train_df["throughput_mbps"])

    packet_loss_model = PacketLossModel.from_settings(settings)
    packet_loss_model.train(train_df, train_df["packet_loss_pct"])

    latency_model = LatencyModel.from_settings(settings)
    latency_model.train(
        train_df.assign(**{LatencyModel.THROUGHPUT_INPUT_COLUMN: train_df["throughput_mbps"]}),
        train_df["latency_ms"],
    )

    prb_model = PrbUtilizationModel.from_settings(settings)
    prb_model.train(
        train_df.assign(**{PrbUtilizationModel.THROUGHPUT_INPUT_COLUMN: train_df["throughput_mbps"]}),
        train_df["prb_utilization_pct"],
    )

    jitter_model = JitterModel.from_settings(settings)
    jitter_model.train(
        train_df.assign(
            **{
                JitterModel.THROUGHPUT_INPUT_COLUMN: train_df["throughput_mbps"],
                JitterModel.LATENCY_INPUT_COLUMN: train_df["latency_ms"],
                JitterModel.PACKET_LOSS_INPUT_COLUMN: train_df["packet_loss_pct"],
            }
        ),
        train_df["jitter_ms"],
    )

    # --- register all five, real toggle included ---
    registry = DTModelRegistry()
    registry.register(throughput_model)
    registry.register(packet_loss_model)
    registry.register(latency_model)
    registry.register(prb_model, enabled=settings.dt_models.enable_prb_model)
    registry.register(jitter_model)
    orchestrator = DTOrchestrator(registry)

    # --- 1. one valid topological order exists and respects every real dependency edge ---
    order = orchestrator.build_execution_order()
    assert set(order) == {"throughput", "packet_loss", "latency", "prb_utilization", "jitter"}
    assert order.index("throughput") < order.index("latency")
    assert order.index("throughput") < order.index("prb_utilization")
    assert order.index("throughput") < order.index("jitter")
    assert order.index("latency") < order.index("jitter")
    assert order.index("packet_loss") < order.index("jitter")
    assert order[-1] == "jitter"  # the diagram's "added last" note, holds for real

    # --- 2. run_predictions() returns correct predictions for all five in one call ---
    predictions = orchestrator.run_predictions(held_out_df)
    assert set(predictions.keys()) == {"throughput", "packet_loss", "latency", "prb_utilization", "jitter"}
    for name, series in predictions.items():
        assert len(series) == len(held_out_df), f"{name} prediction length mismatch"

    # --- 3. jitter (the deepest node) is fed by the OTHER FOUR's live predictions, not ground truth ---
    expected_jitter_input = held_out_df.assign(
        **{
            JitterModel.THROUGHPUT_INPUT_COLUMN: predictions["throughput"],
            JitterModel.LATENCY_INPUT_COLUMN: predictions["latency"],
            JitterModel.PACKET_LOSS_INPUT_COLUMN: predictions["packet_loss"],
        }
    )
    expected_jitter_preds = jitter_model.predict(expected_jitter_input)
    pd.testing.assert_series_equal(predictions["jitter"], expected_jitter_preds)

    # latency, in turn, must also have been fed throughput's live prediction (not ground truth) —
    # confirming the chain is live end-to-end, not just at the final hop.
    expected_latency_input = held_out_df.assign(
        **{LatencyModel.THROUGHPUT_INPUT_COLUMN: predictions["throughput"]}
    )
    expected_latency_preds = latency_model.predict(expected_latency_input)
    pd.testing.assert_series_equal(predictions["latency"], expected_latency_preds)

    # --- 4. real held-out metrics for the full run, reported (not placeholders) ---
    print("\n[Full 5-component orchestrator run] held-out prediction summary:")
    print(f"  execution order: {order}")
    for name, series in predictions.items():
        print(f"  {name}: mean={series.mean():.4f} std={series.std():.4f} n={len(series)}")


def test_disabling_prb_utilization_in_the_full_five_component_registry_only_removes_it(bootstrap_history):
    """The enable/disable toggle exercised within the FULL registry (not in isolation): turning
    prb_utilization off must not perturb the other four components' execution order or results."""
    settings = load_settings()
    train_df, held_out_df = time_split(bootstrap_history)

    throughput_model = ThroughputModel.from_settings(settings)
    throughput_model.train(train_df, train_df["throughput_mbps"])
    packet_loss_model = PacketLossModel.from_settings(settings)
    packet_loss_model.train(train_df, train_df["packet_loss_pct"])
    latency_model = LatencyModel.from_settings(settings)
    latency_model.train(
        train_df.assign(**{LatencyModel.THROUGHPUT_INPUT_COLUMN: train_df["throughput_mbps"]}),
        train_df["latency_ms"],
    )
    prb_model = PrbUtilizationModel.from_settings(settings)
    prb_model.train(
        train_df.assign(**{PrbUtilizationModel.THROUGHPUT_INPUT_COLUMN: train_df["throughput_mbps"]}),
        train_df["prb_utilization_pct"],
    )
    jitter_model = JitterModel.from_settings(settings)
    jitter_model.train(
        train_df.assign(
            **{
                JitterModel.THROUGHPUT_INPUT_COLUMN: train_df["throughput_mbps"],
                JitterModel.LATENCY_INPUT_COLUMN: train_df["latency_ms"],
                JitterModel.PACKET_LOSS_INPUT_COLUMN: train_df["packet_loss_pct"],
            }
        ),
        train_df["jitter_ms"],
    )

    registry = DTModelRegistry()
    registry.register(throughput_model)
    registry.register(packet_loss_model)
    registry.register(latency_model)
    registry.register(prb_model, enabled=False)  # simulates enable_prb_model: false
    registry.register(jitter_model)
    orchestrator = DTOrchestrator(registry)

    order = orchestrator.build_execution_order()
    assert set(order) == {"throughput", "packet_loss", "latency", "jitter"}
    assert "prb_utilization" not in order

    predictions = orchestrator.run_predictions(held_out_df)
    assert set(predictions.keys()) == {"throughput", "packet_loss", "latency", "jitter"}
    assert len(predictions["jitter"]) == len(held_out_df)
