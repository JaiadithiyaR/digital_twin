"""Fidelity Evaluation Module — deterministic, per-component composite fidelity scoring:

    D_m = metric_m ** 2                                     (square every raw metric)
    D~_m = (D_m - min(D_m)) / (max(D_m) - min(D_m) + eps)    (rolling min-max normalize)
    S_raw,c = D~_RMSE + D~_MAE + D~_W1 + D~_MMD
    FidelityScore_c = 1 - (S_raw,c / 4)

`epsilon` and the rolling-window length come from `config.fidelity` — never hardcoded. The LLM
never computes or touches this formula; this module is the sole deterministic authority.

Normalization reference: each component keeps a rolling window of past squared-metric values.
`evaluate()` normalizes the current point against the window's contents from BEFORE this call
(never self-referential), then appends if `update_window=True` (default). Comparing a candidate
against production must call `evaluate(..., update_window=False)` on the SAME evaluator instance
production already grew, so both share one normalization reference.

Never fabricates a score: fewer than `min_history_for_normalization` prior points ->
`status="insufficient_history"`, `fidelity_score=None` (raw/squared metrics still returned).

Unified Fidelity Score: aggregates every component's `FidelityScore_c` into one system-wide
`UnifiedFidelityScore` (`compute_unified_score()`) — a health/REPORTING summary only. Adaptation
is decided per component from each `FidelityScore_c` by `src/fidelity/component_policy.py`
(which replaced the former unified-score trigger); this module only measures.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import numpy as np

from src.common.config import FidelityConfig
from src.fidelity import metrics as metric_fns

logger = logging.getLogger(__name__)

_METRIC_NAMES = ("rmse", "mae", "wasserstein", "mk_mmd")


@dataclass(frozen=True)
class UnifiedFidelityResult:
    status: Literal["ok", "insufficient_history"]
    unified_score: float | None  # None iff status == "insufficient_history" — never fabricated
    worst_component: str | None  # argmin over the currently-defined per-component scores; None iff insufficient_history
    component_scores: dict[str, float]  # only the currently well-defined FidelityScore_c values that were aggregated
    n_components_defined: int


@dataclass
class FidelityResult:
    component: str
    timestamp: datetime
    status: Literal["ok", "insufficient_history"]
    n_samples: int
    raw_metrics: dict[str, float]
    squared_metrics: dict[str, float]
    normalized_metrics: dict[str, float] | None  # None iff status == "insufficient_history"
    fidelity_score: float | None  # None iff status == "insufficient_history" — never fabricated
    window_size_used: dict[str, int] = field(default_factory=dict)  # audit: prior points available per metric


class FidelityEvaluator:
    """Owns one rolling-window normalization reference per component. Construct one instance and
    reuse it across the component's whole lifetime (or explicitly share one across a
    production/candidate comparison) — a fresh instance per call would defeat the entire point of
    a *rolling* historical reference."""

    def __init__(self, config: FidelityConfig) -> None:
        self._epsilon = config.epsilon
        self._window_length = config.rolling_window_length
        self._min_history = config.min_history_for_normalization
        self._mk_mmd_gamma = config.mk_mmd.gamma
        # windows[component][metric_name] -> bounded deque of past squared-metric values.
        self._windows: dict[str, dict[str, deque[float]]] = {}
        # The fidelity loop and adaptation worker threads (verification) share these windows;
        # normalization reads a window while another thread may append — serialize access.
        self._lock = threading.RLock()
        # Unified Fidelity Score config (prompt.md §16a) — never hardcoded, always config-driven.
        self._unified_weights = config.unified_weights
        self._unified_min_components = config.unified_min_components

    def _window_for(self, component: str, metric_name: str) -> deque[float]:
        component_windows = self._windows.setdefault(component, {})
        return component_windows.setdefault(metric_name, deque(maxlen=self._window_length))

    def history_length(self, component: str) -> int:
        """How many points this component's rolling window has accumulated so far. All four
        metrics' deques are always appended together in `evaluate()`, so any one of them (here
        "rmse") represents the component's history length. Lets a caller (e.g. `src/main.py`)
        decide whether — and how much — to backfill from already-available real history before
        `min_history_for_normalization` would otherwise require several more real evaluation
        cycles spaced out over time."""
        with self._lock:
            return len(self._windows.get(component, {}).get("rmse", ()))

    def evaluate(self, component: str, y_true, y_pred, update_window: bool = True) -> FidelityResult:
        """Compute one component's deterministic fidelity result for one `(y_true, y_pred)`
        evaluation window. Raises `metrics.FidelityComputationError` for invalid inputs (empty,
        mismatched length, NaN/Inf) — never silently computes a score from bad data.
        """
        n_samples = len(np.asarray(y_true))

        raw_metrics = {
            "rmse": metric_fns.rmse(y_true, y_pred),
            "mae": metric_fns.mae(y_true, y_pred),
            "wasserstein": metric_fns.wasserstein(y_true, y_pred),
            "mk_mmd": metric_fns.mk_mmd(y_true, y_pred, gamma=self._mk_mmd_gamma),
        }
        squared_metrics = {name: value**2 for name, value in raw_metrics.items()}

        normalized_metrics: dict[str, float] = {}
        window_size_used: dict[str, int] = {}
        insufficient = False

        with self._lock:  # read-normalize-append as one step (thread safety only; formula unchanged)
            for name in _METRIC_NAMES:
                window = self._window_for(component, name)
                window_size_used[name] = len(window)
                if len(window) < self._min_history:
                    insufficient = True
                else:
                    historical = np.array(window, dtype=float)
                    d_min, d_max = float(historical.min()), float(historical.max())
                    normalized_metrics[name] = (squared_metrics[name] - d_min) / (d_max - d_min + self._epsilon)

            if update_window:
                for name in _METRIC_NAMES:
                    self._window_for(component, name).append(squared_metrics[name])

        if insufficient:
            logger.info(
                "fidelity: insufficient rolling history, composite score withheld",
                extra={"component": component, "window_sizes": window_size_used, "min_required": self._min_history},
            )
            return FidelityResult(
                component=component,
                timestamp=datetime.now(UTC),
                status="insufficient_history",
                n_samples=n_samples,
                raw_metrics=raw_metrics,
                squared_metrics=squared_metrics,
                normalized_metrics=None,
                fidelity_score=None,
                window_size_used=window_size_used,
            )

        s_raw = sum(normalized_metrics[name] for name in _METRIC_NAMES)
        fidelity_score = 1.0 - (s_raw / 4.0)

        logger.info(
            "fidelity evaluated",
            extra={"component": component, "fidelity_score": fidelity_score, "n_samples": n_samples},
        )
        return FidelityResult(
            component=component,
            timestamp=datetime.now(UTC),
            status="ok",
            n_samples=n_samples,
            raw_metrics=raw_metrics,
            squared_metrics=squared_metrics,
            normalized_metrics=normalized_metrics,
            fidelity_score=fidelity_score,
            window_size_used=window_size_used,
        )

    # --- Unified Fidelity Score + fidelity-based trigger (prompt.md §16a) ----------------------

    def compute_unified_score(self, component_scores: dict[str, float | None]) -> UnifiedFidelityResult:
        """Aggregates every component's current FidelityScore_c into one system-wide number:

            UnifiedFidelityScore = Σ_c (w_c * FidelityScore_c) / Σ_c w_c

        over components whose score is currently defined (not `None`) — undefined components are
        excluded from both sum and weight normalization, never treated as 0."""
        defined = {c: s for c, s in component_scores.items() if s is not None}

        if len(defined) < self._unified_min_components:
            logger.info(
                "unified fidelity: insufficient defined components, score withheld",
                extra={"n_defined": len(defined), "min_required": self._unified_min_components},
            )
            return UnifiedFidelityResult(
                status="insufficient_history",
                unified_score=None,
                worst_component=None,
                component_scores=defined,
                n_components_defined=len(defined),
            )

        weights = self._unified_weights or dict.fromkeys(defined, 1.0)
        weight_sum = sum(weights[c] for c in defined if c in weights)
        weighted_sum = sum(defined[c] * weights[c] for c in defined if c in weights)
        unified_score = weighted_sum / weight_sum
        worst_component = min(defined, key=lambda c: defined[c])  # argmin over defined FidelityScore_c

        logger.info(
            "unified fidelity score computed",
            extra={"unified_score": unified_score, "worst_component": worst_component, "n_defined": len(defined)},
        )
        return UnifiedFidelityResult(
            status="ok",
            unified_score=unified_score,
            worst_component=worst_component,
            component_scores=defined,
            n_components_defined=len(defined),
        )
