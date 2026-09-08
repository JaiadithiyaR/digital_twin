"""Unit tests for FidelityEvaluator (Module 12, src/fidelity/evaluator.py).

`test_composite_score_hand_computed_case` is the key test: a fully independently-derived
expected value for the entire pipeline — RMSE/MAE/Wasserstein/MK-MMD, squaring, rolling min-max
normalization, and the final composite formula — not just one metric in isolation. The MK-MMD
component of that derivation was itself cross-checked against a brute-force reimplementation
(see test_fidelity_metrics.py) before being hardcoded here.
"""

from __future__ import annotations

import math

import pytest

from src.common.config import FidelityConfig, MkMmdConfig
from src.fidelity.evaluator import FidelityEvaluator
from src.fidelity.metrics import FidelityComputationError


def _config(**overrides) -> FidelityConfig:
    base = dict(
        epsilon=1.0e-8,
        rolling_window_length=10,
        min_history_for_normalization=2,
        metrics=["rmse", "mae", "wasserstein", "mk_mmd"],
        mk_mmd=MkMmdConfig(kernel="rbf", gamma=1.0),  # fixed (not median-heuristic) for hand-computability
    )
    base.update(overrides)
    return FidelityConfig(**base)


# --- insufficient history --------------------------------------------------------------------


def test_first_evaluation_is_insufficient_history():
    evaluator = FidelityEvaluator(_config(min_history_for_normalization=1))
    result = evaluator.evaluate("throughput", [0.0, 0.0], [1.0, 1.0])
    assert result.status == "insufficient_history"
    assert result.fidelity_score is None
    assert result.normalized_metrics is None


def test_insufficient_history_still_returns_raw_and_squared_metrics():
    evaluator = FidelityEvaluator(_config(min_history_for_normalization=5))
    result = evaluator.evaluate("throughput", [0.0, 0.0], [1.0, 1.0])
    assert result.raw_metrics["rmse"] == pytest.approx(1.0)
    assert result.squared_metrics["rmse"] == pytest.approx(1.0)


def test_becomes_sufficient_exactly_at_min_history_threshold():
    evaluator = FidelityEvaluator(_config(min_history_for_normalization=2))
    r1 = evaluator.evaluate("c", [0.0, 0.0], [1.0, 1.0])  # 0 prior -> insufficient
    r2 = evaluator.evaluate("c", [0.0, 0.0], [2.0, 2.0])  # 1 prior -> still insufficient
    r3 = evaluator.evaluate("c", [0.0, 0.0], [3.0, 3.0])  # 2 prior -> sufficient
    assert r1.status == "insufficient_history"
    assert r2.status == "insufficient_history"
    assert r3.status == "ok"
    assert r3.fidelity_score is not None


# --- THE hand-computed composite score case ---------------------------------------------------


def test_composite_score_hand_computed_case():
    """Independently derived expected values (verified against the evaluator and against a
    brute-force MK-MMD reimplementation before being hardcoded — see module docstring):

    Three calls with y_true=[0,0], y_pred=[a,a] for a=1,2,3 (gamma=1.0 fixed, so MK-MMD is
    fully hand-derivable): for point-mass pairs, RMSE=MAE=Wasserstein=a exactly, so
    D_RMSE=D_MAE=D_W1=a^2 for each call: [1, 4, 9]. MK-MMD (verified against an independent
    brute-force triple-loop reimplementation in test_fidelity_metrics.py) is
    [1.1225728234370995, 1.3774405287214764, 1.4115635724089066] for a=[1,2,3], so
    D_MMD = [1.2601697439195412, 1.8973424101645004, 1.9925117189517945].

    After 2 calls (min_history=2), call 3 normalizes against the window built from calls 1-2:
      D~_RMSE = D~_MAE = D~_W1 = (9 - 1) / (4 - 1 + eps) = 8/3 ~= 2.666666657777778
      D~_MMD  = (1.9925117189517945 - 1.2601697439195412) /
                (1.8973424101645004 - 1.2601697439195412 + eps) ~= 1.1493618642722627
      S_raw   ~= 3 * 2.666666657777778 + 1.1493618642722627 = 9.149361837605596
      FidelityScore = 1 - S_raw/4 ~= -1.287340459401399
    """
    evaluator = FidelityEvaluator(_config(min_history_for_normalization=2))
    evaluator.evaluate("component", [0.0, 0.0], [1.0, 1.0])
    evaluator.evaluate("component", [0.0, 0.0], [2.0, 2.0])
    result = evaluator.evaluate("component", [0.0, 0.0], [3.0, 3.0])

    assert result.status == "ok"
    assert result.normalized_metrics["rmse"] == pytest.approx(2.666666657777778, rel=1e-9)
    assert result.normalized_metrics["mae"] == pytest.approx(2.666666657777778, rel=1e-9)
    assert result.normalized_metrics["wasserstein"] == pytest.approx(2.666666657777778, rel=1e-9)
    assert result.normalized_metrics["mk_mmd"] == pytest.approx(1.1493618642722627, rel=1e-9)
    assert result.fidelity_score == pytest.approx(-1.287340459401399, rel=1e-9)


# --- update_window semantics / shared normalization reference (prompt.md §0.10) --------------


def test_update_window_false_does_not_grow_the_window():
    evaluator = FidelityEvaluator(_config(min_history_for_normalization=2))
    evaluator.evaluate("c", [0.0, 0.0], [1.0, 1.0])
    evaluator.evaluate("c", [0.0, 0.0], [2.0, 2.0])

    before = evaluator.evaluate("c", [0.0, 0.0], [3.0, 3.0], update_window=False)
    after_same_call_repeated = evaluator.evaluate("c", [0.0, 0.0], [3.0, 3.0], update_window=False)

    # Same window state both times (neither call grew it) -> identical results.
    assert before.window_size_used == after_same_call_repeated.window_size_used == {"rmse": 2, "mae": 2, "wasserstein": 2, "mk_mmd": 2}
    assert before.fidelity_score == pytest.approx(after_same_call_repeated.fidelity_score)


def test_production_and_candidate_evaluations_share_one_normalization_reference():
    """A candidate evaluation (update_window=False) reusing the SAME evaluator instance a
    production run already grew must be normalized against the identical window a same-valued
    production call would see — this is what makes production vs. candidate scores comparable
    per prompt.md §0.10."""
    evaluator = FidelityEvaluator(_config(min_history_for_normalization=2))
    evaluator.evaluate("c", [0.0, 0.0], [1.0, 1.0])
    evaluator.evaluate("c", [0.0, 0.0], [2.0, 2.0])

    candidate_result = evaluator.evaluate("c", [0.0, 0.0], [3.0, 3.0], update_window=False)
    production_result = evaluator.evaluate("c", [0.0, 0.0], [3.0, 3.0], update_window=True)

    assert candidate_result.normalized_metrics == production_result.normalized_metrics
    assert candidate_result.fidelity_score == pytest.approx(production_result.fidelity_score)

    # The production call (update_window=True) grew the window; a subsequent call sees more history.
    next_result = evaluator.evaluate("c", [0.0, 0.0], [4.0, 4.0], update_window=False)
    assert next_result.window_size_used["rmse"] == 3


# --- numerical stability edge cases (prompt.md §0.10) -----------------------------------------


def test_constant_window_produces_large_but_finite_value_not_nan_or_inf():
    evaluator = FidelityEvaluator(_config(min_history_for_normalization=2, epsilon=1e-8))
    # Three identical evaluations -> a perfectly constant window (max == min == 1.0).
    evaluator.evaluate("c", [0.0, 0.0], [1.0, 1.0])
    evaluator.evaluate("c", [0.0, 0.0], [1.0, 1.0])
    result = evaluator.evaluate("c", [0.0, 0.0], [1.0, 1.0])

    assert result.status == "ok"
    for value in result.normalized_metrics.values():
        assert math.isfinite(value)
    assert math.isfinite(result.fidelity_score)
    # Identical value against an identical constant window -> (D - min) = 0 exactly.
    assert result.normalized_metrics["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_epsilon_is_config_driven_not_hardcoded():
    small_eps = FidelityEvaluator(_config(min_history_for_normalization=1, epsilon=1e-8))
    large_eps = FidelityEvaluator(_config(min_history_for_normalization=1, epsilon=10.0))

    for ev in (small_eps, large_eps):
        ev.evaluate("c", [0.0, 0.0], [1.0, 1.0])  # seed one point (constant window after this)

    result_small = small_eps.evaluate("c", [0.0, 0.0], [1.0, 1.0])
    result_large = large_eps.evaluate("c", [0.0, 0.0], [2.0, 2.0])
    # Different epsilon must produce a different denominator, hence a different normalized value
    # (both hit the constant-window edge case, so the epsilon term dominates the denominator).
    assert result_small.normalized_metrics["rmse"] != result_large.normalized_metrics["rmse"]


def test_rolling_window_length_is_config_driven_and_bounds_history():
    evaluator = FidelityEvaluator(_config(rolling_window_length=3, min_history_for_normalization=1))
    for a in range(1, 8):  # 7 evaluations, window length 3
        result = evaluator.evaluate("c", [0.0, 0.0], [float(a), float(a)])
    assert result.window_size_used["rmse"] == 3  # bounded, oldest points dropped


def test_raises_on_invalid_input_not_silently_swallowed():
    evaluator = FidelityEvaluator(_config())
    with pytest.raises(FidelityComputationError):
        evaluator.evaluate("c", [], [])


def test_different_components_have_independent_rolling_windows():
    evaluator = FidelityEvaluator(_config(min_history_for_normalization=1))
    evaluator.evaluate("throughput", [0.0, 0.0], [1.0, 1.0])
    result = evaluator.evaluate("latency", [0.0, 0.0], [5.0, 5.0])
    # "latency" has zero prior history of its own, independent of "throughput"'s.
    assert result.status == "insufficient_history"
