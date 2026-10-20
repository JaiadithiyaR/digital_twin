"""The ablation harness's no-look-ahead view and fixed windows, and the stats pipeline's
building blocks (frozen re-scoring, Wilcoxon effect size, block bootstrap, Holm)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from baselines._lib import ORDER, RAW_METRICS, CutoffD1View, window_edges
from baselines.stats import block_bootstrap_ci, frozen_reference, holm, rescore, wilcoxon_paired


class _FakeStore:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def get_history(self, **kwargs):
        return self._df.copy()


def _history(seconds: int) -> pd.DataFrame:
    ts = pd.date_range("2026-09-28 15:00:00", periods=seconds, freq="1s", tz="UTC")
    return pd.DataFrame({"ts_1s": ts, "dl_bler": np.arange(seconds, dtype=float)})


def test_cutoff_view_hides_every_row_at_or_after_the_cutoff():
    hist = _history(180)
    view = CutoffD1View(_FakeStore(hist))
    assert len(view.get_history()) == 180  # no cutoff -> everything
    cutoff = hist["ts_1s"].iloc[60]
    view.set_cutoff(cutoff)
    seen = view.get_history()
    assert len(seen) == 60
    assert pd.to_datetime(seen["ts_1s"], utc=True).max() < cutoff


def test_window_edges_are_fixed_60s_steps_covering_all_data():
    hist = _history(125)
    edges = window_edges(hist, 60)
    assert len(edges) == 3
    assert all((end - start) == pd.Timedelta(seconds=60) for start, end in edges)
    assert edges[0][0] == hist["ts_1s"].min()
    assert edges[-1][1] > hist["ts_1s"].max()


def _timeline(scale: float, windows: int = 4) -> pd.DataFrame:
    rows = []
    for w in range(1, windows + 1):
        for c in ORDER:
            rows.append({"window": w, "component": c, **{m: scale * w for m in RAW_METRICS}})
    return pd.DataFrame(rows)


def test_frozen_rescoring_puts_every_run_on_one_scale():
    good, bad = _timeline(1.0), _timeline(3.0)
    ref = frozen_reference([{"timeline": good}, {"timeline": bad}])
    s_good, s_bad = rescore(good, ref, 1e-8), rescore(bad, ref, 1e-8)
    assert s_good["F_frozen"].between(0, 1).all() and s_bad["F_frozen"].between(0, 1).all()
    # Same window index, uniformly larger errors -> strictly lower frozen fidelity.
    assert (s_good.groupby("window")["F_frozen"].mean() > s_bad.groupby("window")["F_frozen"].mean()).all()


def test_wilcoxon_effect_size_sign_and_magnitude():
    a = np.linspace(0.6, 0.9, 20)
    out = wilcoxon_paired(a, a - 0.1)
    assert out["r_rb"] == pytest.approx(1.0)
    assert out["p"] < 0.001
    assert math.isnan(wilcoxon_paired(a[:3], a[:3] - 0.1)["p"])  # too few pairs -> no fabricated p


def test_block_bootstrap_ci_contains_the_mean_and_is_reproducible():
    x = np.random.default_rng(1).normal(0.2, 0.05, 60)
    ci1 = block_bootstrap_ci(x, 5, 2000, np.random.default_rng(0))
    ci2 = block_bootstrap_ci(x, 5, 2000, np.random.default_rng(0))
    assert ci1 == ci2
    assert ci1[0] < x.mean() < ci1[1]


def test_holm_matches_hand_computation():
    adjusted = holm([0.01, 0.04, 0.03, math.nan])
    assert adjusted[0] == pytest.approx(0.03)  # 3 * 0.01
    assert adjusted[2] == pytest.approx(0.06)  # 2 * 0.03
    assert adjusted[1] == pytest.approx(0.06)  # max(0.06, 1 * 0.04): monotone
    assert math.isnan(adjusted[3])
