"""Module 12 — Fidelity Evaluation: deterministic raw metric functions (prompt.md §16).

Every function here is a pure, deterministic function of its `(y_true, y_pred)` inputs — the
same inputs always produce the same output, no randomness, no hidden state, no LLM involvement
anywhere (prompt.md §0.10, §43: the LLM must never compute or modify a fidelity formula). This
is the one place in the system where "deterministic" is completely non-negotiable.

RMSE/MAE are pointwise (paired, same-index) accuracy metrics — how far each individual
prediction is from its corresponding ground-truth value. Wasserstein distance and MK-MMD are
distributional metrics — they compare the two samples AS DISTRIBUTIONS (order/pairing does not
matter), capturing whether the DT's overall predicted distribution matches ground truth's
overall shape, which pointwise metrics alone cannot. All four functions return a non-negative
"distance" value in the same (non-squared) units; the fidelity evaluator (`evaluator.py`) squares
all four uniformly per the formula — no metric function here squares its own output.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance as _scipy_wasserstein_distance

# Fixed bandwidth multipliers around the base gamma for MK-MMD's "multi-kernel" mixture — using
# several kernel widths (not just one) is the actual definition of multi-kernel MMD (Gretton et
# al., 2012). This is a small, fixed implementation choice, not a user-facing tunable; the BASE
# gamma itself is configurable (config.fidelity.mk_mmd.gamma — null selects the median heuristic).
_MK_MMD_BANDWIDTH_MULTIPLIERS = (0.5, 1.0, 2.0)


class FidelityComputationError(ValueError):
    """Invalid metric inputs (empty, mismatched length, non-finite). Raised, never silently
    swallowed — a fidelity score must never be fabricated from bad data (prompt.md §0.10)."""


def _validate_inputs(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise FidelityComputationError("fidelity metrics require 1-D arrays")
    if y_true_arr.shape[0] == 0 or y_pred_arr.shape[0] == 0:
        raise FidelityComputationError("fidelity metrics require non-empty arrays (empty window)")
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise FidelityComputationError(
            f"y_true and y_pred must be the same length (missing samples): "
            f"{y_true_arr.shape[0]} vs {y_pred_arr.shape[0]}"
        )
    if not np.all(np.isfinite(y_true_arr)) or not np.all(np.isfinite(y_pred_arr)):
        raise FidelityComputationError("fidelity metrics require finite inputs (NaN/Inf detected)")
    return y_true_arr, y_pred_arr


def rmse(y_true, y_pred) -> float:
    y_true_arr, y_pred_arr = _validate_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true_arr, y_pred_arr = _validate_inputs(y_true, y_pred)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def wasserstein(y_true, y_pred) -> float:
    """1-Wasserstein (Earth Mover's) distance between the two samples treated as empirical
    distributions — pairing/order-independent, unlike RMSE/MAE. Deterministic: scipy computes it
    exactly via sorted order statistics (the closed-form solution for 1-D optimal transport), no
    randomness involved."""
    y_true_arr, y_pred_arr = _validate_inputs(y_true, y_pred)
    return float(_scipy_wasserstein_distance(y_true_arr, y_pred_arr))


def _rbf_kernel_sum(a: np.ndarray, b: np.ndarray, gamma: float) -> float:
    sq_dists = (a[:, None] - b[None, :]) ** 2
    return float(np.sum(np.exp(-gamma * sq_dists)))


def _median_heuristic_gamma(pooled: np.ndarray) -> float:
    """Standard median-heuristic RBF bandwidth: gamma = 1 / (2 * median(pairwise sq dist)),
    computed deterministically from the pooled sample (no randomness)."""
    sq_dists = (pooled[:, None] - pooled[None, :]) ** 2
    off_diag = sq_dists[~np.eye(len(pooled), dtype=bool)]  # exclude zero self-distances
    median_sq_dist = float(np.median(off_diag)) if off_diag.size else 0.0
    if median_sq_dist <= 0:
        return 1.0  # degenerate (every point identical) — any positive bandwidth is equivalent
    return 1.0 / (2.0 * median_sq_dist)


def mk_mmd(y_true, y_pred, gamma: float | None = None) -> float:
    """Multi-kernel Maximum Mean Discrepancy: the unbiased MMD^2 estimator averaged over several
    fixed RBF bandwidths around a base gamma, then sqrt'd (clamped at 0 — the unbiased MMD^2
    estimator can be slightly negative from sampling noise even when the true MMD is ~0).

    `gamma=None` (the config default) uses the median heuristic, computed deterministically from
    the pooled sample; an explicit `gamma` fixes the base bandwidth instead — both paths are
    fully deterministic, no randomness in either.
    """
    y_true_arr, y_pred_arr = _validate_inputs(y_true, y_pred)
    n, m = len(y_true_arr), len(y_pred_arr)
    if n < 2 or m < 2:
        raise FidelityComputationError("mk_mmd requires at least 2 samples per side")

    base_gamma = gamma if gamma is not None else _median_heuristic_gamma(np.concatenate([y_true_arr, y_pred_arr]))

    mmd_sq_values = []
    for multiplier in _MK_MMD_BANDWIDTH_MULTIPLIERS:
        g = base_gamma * multiplier
        # Unbiased estimator excludes the i==j (self-pair) terms, each contributing exp(0)=1.
        k_xx = (_rbf_kernel_sum(y_true_arr, y_true_arr, g) - n) / (n * (n - 1))
        k_yy = (_rbf_kernel_sum(y_pred_arr, y_pred_arr, g) - m) / (m * (m - 1))
        k_xy = _rbf_kernel_sum(y_true_arr, y_pred_arr, g) / (n * m)
        mmd_sq_values.append(k_xx + k_yy - 2.0 * k_xy)

    mmd_sq = float(np.mean(mmd_sq_values))
    return float(np.sqrt(max(mmd_sq, 0.0)))
