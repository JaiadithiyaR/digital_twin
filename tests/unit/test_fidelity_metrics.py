"""Unit tests for Module 12's deterministic raw metric functions (src/fidelity/metrics.py).

RMSE/MAE/Wasserstein are verified against exact hand-computed values (simple enough to check
with pencil and paper — see comments). MK-MMD is additionally cross-checked against an
independent brute-force triple-loop reimplementation kept entirely separate from
`src/fidelity/metrics.py`'s vectorized implementation, so the two must agree only if both are
actually correct — not because one was copy-pasted from the other.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.fidelity.metrics import FidelityComputationError, mae, mk_mmd, rmse, wasserstein


# --- RMSE: hand-computed --------------------------------------------------------------------


def test_rmse_hand_computed_case():
    # errors = [0, 0, 0, -1] -> squared = [0, 0, 0, 1] -> mean = 0.25 -> sqrt = 0.5
    assert rmse([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0]) == pytest.approx(0.5)


def test_rmse_zero_for_identical_arrays():
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_rmse_is_deterministic():
    y_true, y_pred = [1.0, 5.0, 3.0, 8.0], [2.0, 4.0, 3.5, 7.0]
    assert rmse(y_true, y_pred) == rmse(y_true, y_pred)


# --- MAE: hand-computed ----------------------------------------------------------------------


def test_mae_hand_computed_case():
    # |errors| = [0, 0, 0, 1] -> mean = 0.25
    assert mae([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0]) == pytest.approx(0.25)


def test_mae_zero_for_identical_arrays():
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


# --- Wasserstein: hand-computed (point-mass distributions) ----------------------------------


def test_wasserstein_point_mass_distance_hand_computed():
    # Two degenerate distributions (all mass at a single point) have Wasserstein distance
    # exactly |a - b| — the simplest closed-form case, computable with no formula beyond
    # subtraction.
    assert wasserstein([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert wasserstein([0.0, 0.0], [2.0, 2.0]) == pytest.approx(2.0)
    assert wasserstein([0.0, 0.0], [3.0, 3.0]) == pytest.approx(3.0)


def test_wasserstein_is_order_independent_unlike_rmse():
    # Same two multisets, different order — Wasserstein compares distributions, not positions.
    a = wasserstein([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert a == pytest.approx(0.0)  # identical multisets -> distance 0
    # RMSE on the same (mis-ordered) pairing is NOT zero, proving the two metrics are genuinely
    # different kinds of comparison.
    assert rmse([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) > 0.0


# --- Shared input validation ------------------------------------------------------------------


@pytest.mark.parametrize("fn", [rmse, mae, wasserstein])
def test_metrics_raise_on_empty_input(fn):
    with pytest.raises(FidelityComputationError, match="non-empty"):
        fn([], [])


@pytest.mark.parametrize("fn", [rmse, mae, wasserstein])
def test_metrics_raise_on_mismatched_length(fn):
    with pytest.raises(FidelityComputationError, match="same length"):
        fn([1.0, 2.0, 3.0], [1.0, 2.0])


@pytest.mark.parametrize("fn", [rmse, mae, wasserstein])
def test_metrics_raise_on_nan_input(fn):
    with pytest.raises(FidelityComputationError, match="finite"):
        fn([1.0, float("nan")], [1.0, 2.0])


@pytest.mark.parametrize("fn", [rmse, mae, wasserstein])
def test_metrics_raise_on_inf_input(fn):
    with pytest.raises(FidelityComputationError, match="finite"):
        fn([1.0, float("inf")], [1.0, 2.0])


# --- MK-MMD: independent brute-force cross-check + property tests ---------------------------


def _brute_force_mmd_sq(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Deliberately a plain triple-nested-loop reimplementation, independent of
    `metrics._rbf_kernel_sum`'s vectorized version — agreement between the two is a genuine
    cross-check, not a tautology."""
    n, m = len(x), len(y)

    def k(a: float, b: float) -> float:
        return float(np.exp(-gamma * (a - b) ** 2))

    k_xx = sum(k(x[i], x[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    k_yy = sum(k(y[i], y[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    k_xy = sum(k(x[i], y[j]) for i in range(n) for j in range(m)) / (n * m)
    return k_xx + k_yy - 2.0 * k_xy


def _brute_force_mk_mmd(x, y, base_gamma: float, multipliers=(0.5, 1.0, 2.0)) -> float:
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mmd_sq = float(np.mean([_brute_force_mmd_sq(x_arr, y_arr, base_gamma * m) for m in multipliers]))
    return float(np.sqrt(max(mmd_sq, 0.0)))


@pytest.mark.parametrize("a", [1.0, 2.0, 3.0, 7.5])
def test_mk_mmd_matches_independent_bruteforce_reimplementation(a):
    x, y = [0.0, 0.0], [a, a]
    expected = _brute_force_mk_mmd(x, y, base_gamma=1.0)
    actual = mk_mmd(x, y, gamma=1.0)
    assert actual == pytest.approx(expected, rel=1e-9)


def test_mk_mmd_hand_verified_value_at_a_equals_1():
    # Independently derived (see tests/unit/test_fidelity_metrics.py commit context / module
    # docstring derivation): with base_gamma=1.0 and multipliers (0.5, 1.0, 2.0), x=[0,0],
    # y=[1,1]: K_XX=K_YY=1.0 for every bandwidth (self-distance is always 0); K_XY = exp(-g) for
    # g in (0.5, 1.0, 2.0) = 0.60653.../0.36788.../0.13534...; MMD^2_g = 2 - 2*K_XY per bandwidth
    # -> [0.78694, 1.26424, 1.72932], mean = 1.26017 (rounding), sqrt ~= 1.12257.
    assert mk_mmd([0.0, 0.0], [1.0, 1.0], gamma=1.0) == pytest.approx(1.1225728234370995, rel=1e-9)


def test_mk_mmd_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 50)
    assert mk_mmd(x, x, gamma=1.0) == pytest.approx(0.0, abs=1e-9)


def test_mk_mmd_increases_with_distributional_distance():
    # Using the same hand-verified a=1,2,3 sequence — MK-MMD must increase monotonically as the
    # two distributions move further apart.
    mmd_1 = mk_mmd([0.0, 0.0], [1.0, 1.0], gamma=1.0)
    mmd_2 = mk_mmd([0.0, 0.0], [2.0, 2.0], gamma=1.0)
    mmd_3 = mk_mmd([0.0, 0.0], [3.0, 3.0], gamma=1.0)
    assert mmd_1 < mmd_2 < mmd_3


def test_mk_mmd_is_symmetric():
    x = [0.0, 1.0, 2.0, 3.0]
    y = [5.0, 6.0, 4.0, 7.0]
    assert mk_mmd(x, y, gamma=0.5) == pytest.approx(mk_mmd(y, x, gamma=0.5), rel=1e-9)


def test_mk_mmd_is_deterministic_with_median_heuristic():
    x = [1.0, 3.0, 5.0, 2.0, 8.0]
    y = [2.0, 4.0, 6.0, 3.0, 9.0]
    assert mk_mmd(x, y) == mk_mmd(x, y)  # gamma=None -> median heuristic, still deterministic


def test_mk_mmd_requires_at_least_two_samples_per_side():
    with pytest.raises(FidelityComputationError, match="at least 2"):
        mk_mmd([1.0], [2.0])  # equal length (passes the shared length check), but only 1 sample each
