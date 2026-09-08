"""Module 12 — Fidelity Evaluation Module (prompt.md §16, §0.10, §17).

Deterministic, per-DT-component composite fidelity scoring:

    D_m = metric_m ** 2                                          (square every raw metric)
    D~_m = (D_m - min(D_m)) / (max(D_m) - min(D_m) + eps)         (rolling min-max normalize)
    S_raw,c = D~_RMSE + D~_MAE + D~_W1 + D~_MMD
    FidelityScore_c = 1 - (S_raw,c / 4)

`epsilon` and the rolling-window length are read from `config.fidelity` — never hardcoded
(prompt.md §16). The LLM must never compute or modify this formula (prompt.md §0.10, §43); this
module is the sole deterministic authority for it — nothing here calls out to `src/llm/`.

**Normalization reference and "historical" (prompt.md §0.10, §17)**: each component maintains a
rolling window, per metric, of past squared-metric values. A call to `evaluate()` normalizes the
CURRENT point against the window's contents from BEFORE this call (genuinely historical — a
single evaluation can never trivially normalize to a self-referential 0), then — if
`update_window=True` (the default) — appends the current point into the window for future calls.
Production evaluations use the default; comparing a candidate against production during
verification (a later module) should call `evaluate(..., update_window=False)` using the SAME
evaluator/window instance production evaluations already grew, so both share one identical
normalization reference and remain genuinely comparable, exactly as §0.10 requires ("BOTH MUST
USE THE SAME NORMALIZATION REFERENCE").

**Never silently manufacture a score** (prompt.md §0.10): if fewer than
`config.fidelity.min_history_for_normalization` prior points exist for a component, `evaluate()`
returns `status="insufficient_history"` and `fidelity_score=None` — the raw/squared metrics are
still computed and returned (they're always well-defined), only the composite score is withheld.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import numpy as np

from src.common.config import FidelityConfig
from src.fidelity import metrics as metric_fns

logger = logging.getLogger(__name__)

_METRIC_NAMES = ("rmse", "mae", "wasserstein", "mk_mmd")


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

    def _window_for(self, component: str, metric_name: str) -> deque[float]:
        component_windows = self._windows.setdefault(component, {})
        return component_windows.setdefault(metric_name, deque(maxlen=self._window_length))

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
