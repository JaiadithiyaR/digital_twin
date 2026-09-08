"""Integration test: Module 12's FidelityEvaluator run against REAL predictions from all five
DT prediction components (Modules 6-10) vs. real ground truth in D1's held-out bootstrap data —
not synthetic arrays. Proves the deterministic metric engine works end-to-end against the actual
system it was built to evaluate, and that it genuinely reaches a computed composite fidelity
score (not permanently stuck at "insufficient_history") for every component.
"""

from __future__ import annotations

import math

from src.common.config import load_settings
from src.dt_models.jitter import JitterModel
from src.dt_models.latency import LatencyModel
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator
from src.dt_models.packet_loss import PacketLossModel
from src.dt_models.prb_utilization import PrbUtilizationModel
from src.dt_models.throughput import ThroughputModel
from src.fidelity.evaluator import FidelityEvaluator
from tests.integration.conftest import time_split

_GROUND_TRUTH_COLUMN = {
    "throughput": "throughput_mbps",
    "packet_loss": "packet_loss_pct",
    "latency": "latency_ms",
    "prb_utilization": "prb_utilization_pct",
    "jitter": "jitter_ms",
}


def _train_all_five_and_predict(bootstrap_history, settings):
    """Same training/registration pattern as test_full_dependency_chain.py — trains all five
    real components on the train split and runs the real orchestrator on the held-out split."""
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
    registry.register(prb_model, enabled=settings.dt_models.enable_prb_model)
    registry.register(jitter_model)
    orchestrator = DTOrchestrator(registry)

    predictions = orchestrator.run_predictions(held_out_df)
    return held_out_df, predictions


def test_fidelity_evaluator_reaches_real_composite_scores_for_all_five_components(bootstrap_history):
    settings = load_settings()
    held_out_df, predictions = _train_all_five_and_predict(bootstrap_history, settings)

    evaluator = FidelityEvaluator(settings.fidelity)
    batch_size = 10
    n_batches = len(held_out_df) // batch_size

    ok_results = {name: [] for name in _GROUND_TRUTH_COLUMN}

    for batch_index in range(n_batches):
        start, end = batch_index * batch_size, (batch_index + 1) * batch_size
        for component, gt_column in _GROUND_TRUTH_COLUMN.items():
            y_true_batch = held_out_df[gt_column].iloc[start:end].to_numpy()
            y_pred_batch = predictions[component].iloc[start:end].to_numpy()
            result = evaluator.evaluate(component, y_true_batch, y_pred_batch)

            assert math.isfinite(result.raw_metrics["rmse"])
            assert math.isfinite(result.raw_metrics["mae"])
            assert math.isfinite(result.raw_metrics["wasserstein"])
            assert math.isfinite(result.raw_metrics["mk_mmd"])

            if result.status == "ok":
                assert result.fidelity_score is not None
                assert math.isfinite(result.fidelity_score)
                ok_results[component].append(result)

    print(f"\n[Module 12 FidelityEvaluator] {n_batches} evaluation batches of {batch_size} rows "
          f"over real held-out predictions from all five DT components:")
    for component in _GROUND_TRUTH_COLUMN:
        scores = ok_results[component]
        # min_history_for_normalization (config default 10) must eventually be crossed — every
        # component must reach at least one REAL (non-None) composite score, not stay stuck at
        # "insufficient_history" forever.
        assert len(scores) > 0, f"{component} never reached a computed fidelity score"
        last = scores[-1]
        print(
            f"  {component}: {len(scores)}/{n_batches} batches scored | "
            f"last batch -> RMSE={last.raw_metrics['rmse']:.4f} MAE={last.raw_metrics['mae']:.4f} "
            f"Wasserstein={last.raw_metrics['wasserstein']:.4f} MK-MMD={last.raw_metrics['mk_mmd']:.4f} "
            f"FidelityScore={last.fidelity_score:.4f}"
        )


def test_fidelity_evaluation_is_deterministic_across_repeated_runs(bootstrap_history):
    """The same real predictions vs. ground truth, run through two independent evaluators (same
    config), must produce bit-identical results — determinism must hold end-to-end, not just for
    synthetic hand cases."""
    settings = load_settings()
    held_out_df, predictions = _train_all_five_and_predict(bootstrap_history, settings)

    def run_once():
        evaluator = FidelityEvaluator(settings.fidelity)
        results = []
        batch_size = 10
        for batch_index in range(len(held_out_df) // batch_size):
            start, end = batch_index * batch_size, (batch_index + 1) * batch_size
            for component, gt_column in _GROUND_TRUTH_COLUMN.items():
                y_true_batch = held_out_df[gt_column].iloc[start:end].to_numpy()
                y_pred_batch = predictions[component].iloc[start:end].to_numpy()
                results.append(evaluator.evaluate(component, y_true_batch, y_pred_batch).fidelity_score)
        return results

    first_run = run_once()
    second_run = run_once()
    assert first_run == second_run
