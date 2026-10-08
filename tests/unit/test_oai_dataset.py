"""Unit tests for the OAI dataset pipeline (prompt2.md Step 7): deltas, resets, gaps, the
leakage rule, split disjointness, and UTC timestamps."""

from __future__ import annotations

import pandas as pd
import pytest

from src.common.config import DataSplitsConfig, load_settings
from src.data.oai_dataset import (
    LAGS,
    ROLLING_WINDOWS_SECONDS,
    _multi_lag_features,
    _rolling_window_features,
    compute_deltas,
    feature_columns,
)


def test_deltas_correct_on_hand_made_example():
    df = pd.DataFrame({
        "run_id": [1, 1, 1, 1],
        "session_id": [1, 1, 1, 1],
        "ts_1s": pd.to_datetime(
            ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:02Z", "2026-01-01T00:00:03Z"]
        ),
        "counter": [100, 150, 230, 260],
    })
    out, resets, gaps = compute_deltas(df, ["run_id", "session_id"], "ts_1s", ["counter"])
    assert out["counter_delta"].tolist() == [None, 50, 80, 30] or out["counter_delta"].iloc[1:].tolist() == [50, 80, 30]
    assert pd.isna(out["counter_delta"].iloc[0])
    assert resets["counter"] == 0
    assert gaps["counter"] == 0


def test_counter_reset_becomes_nan_not_a_negative_delta():
    df = pd.DataFrame({
        "run_id": [1, 1, 1],
        "session_id": [1, 1, 1],
        "ts_1s": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:02Z"]),
        "counter": [500, 10, 60],  # a genuine reset (500 -> 10), then normal growth (10 -> 60)
    })
    out, resets, gaps = compute_deltas(df, ["run_id", "session_id"], "ts_1s", ["counter"])
    assert pd.isna(out["counter_delta"].iloc[1])  # the reset row itself is NaN, never a negative number
    assert out["counter_delta"].iloc[2] == 50  # the row AFTER the reset resumes normal, correct delta
    assert resets["counter"] == 1
    assert gaps["counter"] == 0


def test_gap_larger_than_threshold_becomes_nan_never_spread_across_the_gap():
    df = pd.DataFrame({
        "run_id": [1, 1, 1],
        "session_id": [1, 1, 1],
        "ts_1s": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:10Z"]),  # 9s gap
        "counter": [100, 150, 500],
    })
    out, resets, gaps = compute_deltas(df, ["run_id", "session_id"], "ts_1s", ["counter"], max_gap_seconds=2.0)
    assert out["counter_delta"].iloc[1] == 50  # no gap here, real delta
    assert pd.isna(out["counter_delta"].iloc[2])  # 9s gap > 2s threshold -> NaN, not a fabricated delta
    assert gaps["counter"] == 1
    assert resets["counter"] == 0


def test_gap_exactly_at_or_under_threshold_is_not_flagged():
    df = pd.DataFrame({
        "run_id": [1, 1],
        "session_id": [1, 1],
        "ts_1s": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:02Z"]),  # exactly 2s
        "counter": [100, 150],
    })
    _, _, gaps = compute_deltas(df, ["run_id", "session_id"], "ts_1s", ["counter"], max_gap_seconds=2.0)
    assert gaps["counter"] == 0


def test_delta_computation_is_per_group_not_leaked_across_sessions():
    df = pd.DataFrame({
        "run_id": [1, 1, 1, 1],
        "session_id": [1, 1, 2, 2],
        "ts_1s": pd.to_datetime([
            "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z",
            "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z",
        ]),
        "counter": [100, 150, 9000, 9050],
    })
    out, _, _ = compute_deltas(df, ["run_id", "session_id"], "ts_1s", ["counter"])
    session1 = out[out["session_id"] == 1]
    session2 = out[out["session_id"] == 2]
    assert pd.isna(session1["counter_delta"].iloc[0])
    assert session1["counter_delta"].iloc[1] == 50
    assert pd.isna(session2["counter_delta"].iloc[0])  # must NOT compute 9000 - 150 across sessions
    assert session2["counter_delta"].iloc[1] == 50


def test_feature_columns_are_all_lagged_or_static_never_same_second_measured():
    """The leakage rule (prompt2.md Step 3 / phase13 §1.3): every declared feature column is a
    lagged (`_lag{1..5}`) or rolling-past-window (`_roll_{mean,std}_{5,10}s`) column, or a
    static/categorical identity column — never a bare, same-second measured value."""
    df = pd.DataFrame(columns=[
        "run_id", "cell_id", "session_id", "ue_id", "slice_id", "ts_1s",
        "dl_throughput", "ul_throughput", "dl_rlc_delay", "dl_prb", "ul_prb", "dl_bler", "ul_bler",
        "dl_throughput_lag1", "dl_throughput_lag5", "dl_throughput_roll_mean_5s", "dl_throughput_roll_std_10s",
        "dl_bler_avg_lag1", "dl_bler_avg_roll_mean_10s", "ue_count_cell_lag1", "requested_prb_ratio_lag1", "has_controller_lag1",
    ])
    cols = feature_columns(df)
    assert "dl_throughput" not in cols  # a raw, same-second target must never be a feature
    assert "dl_throughput_lag1" in cols
    assert "dl_throughput_lag5" in cols
    assert "dl_throughput_roll_mean_5s" in cols
    assert "dl_throughput_roll_std_10s" in cols
    assert "dl_bler_avg_lag1" in cols
    assert "dl_bler_avg_roll_mean_10s" in cols
    assert "has_controller_lag1" in cols
    lag_suffixes = tuple(f"_lag{k}" for k in LAGS)
    roll_suffixes = tuple(f"_roll_{stat}_{w}s" for w in ROLLING_WINDOWS_SECONDS for stat in ("mean", "std"))
    for c in cols:
        assert c.endswith(lag_suffixes) or c.endswith(roll_suffixes) or c in ("ue_count_cell_lag1", "ue_count_slice_lag1"), (
            f"non-lagged, non-rolling, non-static feature: {c}"
        )


def test_multi_lag_features_never_leak_the_current_second():
    """Numeric (not just naming-convention) leakage proof — phase13 §1.3: "Extend the leakage
    test so it fails if any feature uses data from second t or later." Perturbing row t's own raw
    value must NOT change row t's lag1-5 features (they must come only from t-1..t-5); perturbing
    an earlier row MUST change the corresponding lag feature, proving it's genuinely reading past
    data rather than being frozen/broken."""
    base = pd.DataFrame({
        "run_id": [1] * 10, "session_id": [1] * 10, "ts_1s": list(range(10)),
        "x": [float(i * 10) for i in range(10)],
    })
    row_t = 7
    baseline = _multi_lag_features(base, ["run_id", "session_id"], "ts_1s", ["x"])

    perturbed_same_second = base.copy()
    perturbed_same_second.loc[row_t, "x"] = 99999.0
    after_same_second = _multi_lag_features(perturbed_same_second, ["run_id", "session_id"], "ts_1s", ["x"])
    for k in LAGS:
        col = f"x_lag{k}"
        assert after_same_second.loc[row_t, col] == baseline.loc[row_t, col], (
            f"{col} at row t changed when ONLY row t's own raw value was perturbed — same-second leakage"
        )

    perturbed_past_second = base.copy()
    perturbed_past_second.loc[row_t - 1, "x"] = 99999.0
    after_past_second = _multi_lag_features(perturbed_past_second, ["run_id", "session_id"], "ts_1s", ["x"])
    assert after_past_second.loc[row_t, "x_lag1"] == 99999.0, "x_lag1 at row t did not reflect a genuine change at row t-1"


def test_rolling_window_features_never_leak_the_current_second():
    """Same numeric leakage proof as above, for the rolling mean/std features."""
    base = pd.DataFrame({
        "run_id": [1] * 15, "session_id": [1] * 15, "ts_1s": list(range(15)),
        "x": [float(i * 10) for i in range(15)],
    })
    row_t = 12
    baseline = _rolling_window_features(base, ["run_id", "session_id"], "ts_1s", ["x"])

    perturbed_same_second = base.copy()
    perturbed_same_second.loc[row_t, "x"] = 99999.0
    after_same_second = _rolling_window_features(perturbed_same_second, ["run_id", "session_id"], "ts_1s", ["x"])
    for w in ROLLING_WINDOWS_SECONDS:
        for stat in ("mean", "std"):
            col = f"x_roll_{stat}_{w}s"
            assert after_same_second.loc[row_t, col] == pytest.approx(baseline.loc[row_t, col], nan_ok=True), (
                f"{col} at row t changed when ONLY row t's own raw value was perturbed — same-second leakage"
            )

    perturbed_past_second = base.copy()
    perturbed_past_second.loc[row_t - 1, "x"] = 99999.0
    after_past_second = _rolling_window_features(perturbed_past_second, ["run_id", "session_id"], "ts_1s", ["x"])
    assert after_past_second.loc[row_t, "x_roll_mean_5s"] != pytest.approx(baseline.loc[row_t, "x_roll_mean_5s"]), (
        "x_roll_mean_5s at row t did not reflect a genuine change at row t-1 — window may be empty/broken"
    )


def test_multi_lag_and_rolling_features_never_cross_session_boundaries():
    """Both helpers must never let one session's data leak into another's lag/rolling features —
    the same per-group discipline `compute_deltas`/`_safe_shift` already established."""
    df = pd.DataFrame({
        "run_id": [1, 1, 1, 1, 1, 1],
        "session_id": [1, 1, 1, 2, 2, 2],
        "ts_1s": [0, 1, 2, 0, 1, 2],
        "x": [10.0, 20.0, 30.0, 1000.0, 2000.0, 3000.0],
    })
    lags = _multi_lag_features(df, ["run_id", "session_id"], "ts_1s", ["x"])
    assert pd.isna(lags.loc[3, "x_lag1"])  # first row of session 2 — nothing from session 1 leaks in
    assert lags.loc[4, "x_lag1"] == 1000.0  # genuinely session 2's own previous row

    rolling = _rolling_window_features(df, ["run_id", "session_id"], "ts_1s", ["x"])
    assert pd.isna(rolling.loc[3, "x_roll_mean_5s"])  # first row of session 2 — no prior data within its own group
    assert rolling.loc[4, "x_roll_mean_5s"] == 1000.0  # only session 2's own x_lag1 value, never session 1's 30.0


def test_splits_must_be_disjoint():
    with pytest.raises(ValueError, match="appear in both"):
        DataSplitsConfig(train=[3, 6, 8], test=[8], smoke_test_run=3)


def test_splits_disjoint_accepts_valid_config():
    s = DataSplitsConfig(train=[3, 6, 7], test=[8], smoke_test_run=7)
    assert s.train == [3, 6, 7]
    assert s.test == [8]
    assert s.smoke_test_run == 7


def test_smoke_test_run_must_be_a_member_of_train():
    with pytest.raises(ValueError, match="smoke_test_run"):
        DataSplitsConfig(train=[3, 6, 7], test=[8], smoke_test_run=8)


def test_real_processed_tables_have_utc_timestamps():
    """Run against the real build output (prompt2.md Step 7's own "UTC timestamps" check) — skips
    cleanly if the dataset hasn't been built yet in this environment, rather than failing on an
    unrelated missing-fixture reason."""
    settings = load_settings()
    out_dir = settings.resolve_path("data/processed/oai")
    train_path = out_dir / "train.parquet"
    if not train_path.exists():
        pytest.skip("data/processed/oai/train.parquet not built yet — run scripts/build_oai_dataset.py")
    df = pd.read_parquet(train_path)
    assert str(df["ts_1s"].dtype).endswith("UTC]") or df["ts_1s"].dt.tz is not None
