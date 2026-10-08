"""Integration test: Module 12's FidelityEvaluator run against REAL predictions from all seven
OAI KPI DT prediction components (Modules 6-10 replacement) vs. real ground truth in D1's
held-out bootstrap data — not synthetic arrays. Proves the deterministic metric engine works
end-to-end against the actual system it was built to evaluate, and that it genuinely reaches a
computed composite fidelity score (not permanently stuck at "insufficient_history") for every
component.
"""

from __future__ import annotations

import math

from src.common.config import load_settings
from src.dt_models.dl_bler import DlBlerModel
from src.dt_models.dl_prb import DlPrbModel
from src.dt_models.dl_rlc_delay import DlRlcDelayModel
from src.dt_models.dl_throughput import DlThroughputModel
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator
from src.dt_models.ul_bler import UlBlerModel
from src.dt_models.ul_prb import UlPrbModel
from src.dt_models.ul_throughput import UlThroughputModel
from src.fidelity.evaluator import FidelityEvaluator
from tests.integration.conftest import time_split

_COMPONENTS = ("dl_bler", "ul_bler", "dl_prb", "ul_prb", "dl_throughput", "ul_throughput", "dl_rlc_delay")


def _train_all_seven_and_predict(bootstrap_history, settings):
    train_df, held_out_df = time_split(bootstrap_history)
    train_with_deps = train_df.copy()

    registry = DTModelRegistry()
    classes = {
        "dl_bler": DlBlerModel, "ul_bler": UlBlerModel, "dl_prb": DlPrbModel, "ul_prb": UlPrbModel,
        "dl_throughput": DlThroughputModel, "ul_throughput": UlThroughputModel, "dl_rlc_delay": DlRlcDelayModel,
    }
    for name in _COMPONENTS:
        model = classes[name].from_settings(settings)
        model.train(train_with_deps, train_df[name])
        registry.register(model)
        train_with_deps[f"{name}_pred"] = train_df[name]

    orchestrator = DTOrchestrator(registry)
    predictions = orchestrator.run_predictions(held_out_df)
    return held_out_df, predictions


def test_fidelity_evaluator_reaches_real_composite_scores_for_all_seven_components(bootstrap_history):
    settings = load_settings()
    held_out_df, predictions = _train_all_seven_and_predict(bootstrap_history, settings)

    evaluator = FidelityEvaluator(settings.fidelity)
    batch_size = 10
    n_batches = len(held_out_df) // batch_size

    ok_results = {name: [] for name in _COMPONENTS}

    for batch_index in range(n_batches):
        start, end = batch_index * batch_size, (batch_index + 1) * batch_size
        for component in _COMPONENTS:
            y_true_batch = held_out_df[component].iloc[start:end].to_numpy()
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
          f"over real held-out predictions from all seven DT components:")
    for component in _COMPONENTS:
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
    held_out_df, predictions = _train_all_seven_and_predict(bootstrap_history, settings)

    def run_once():
        evaluator = FidelityEvaluator(settings.fidelity)
        results = []
        batch_size = 10
        for batch_index in range(len(held_out_df) // batch_size):
            start, end = batch_index * batch_size, (batch_index + 1) * batch_size
            for component in _COMPONENTS:
                y_true_batch = held_out_df[component].iloc[start:end].to_numpy()
                y_pred_batch = predictions[component].iloc[start:end].to_numpy()
                results.append(evaluator.evaluate(component, y_true_batch, y_pred_batch).fidelity_score)
        return results

    first_run = run_once()
    second_run = run_once()
    assert first_run == second_run
