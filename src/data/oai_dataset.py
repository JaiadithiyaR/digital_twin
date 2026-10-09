"""Builds model-ready train/test tables from the exported OAI RAN KPI CSVs at
`config.data.oai_root` (read-only, outside this repo). Flat CSVs only — no database access.
Unit of analysis: one row per `(run_id, session_id, ts_1s)` at UE level.

Unit-conversion facts this module relies on:
  - `DRB.UEThpDl_avg`/`UEThpUl_avg` (v_ue_kpm_1s) are in **kbps** -> divide by 1000 for Mbps.
  - `dl_rlc_delay` is sourced entirely from `rlc_bearer_1s.tx_sdu_avg_time_to_tx_avg` (no
    `DRB.RlcSduDelayDl` column exists in this export), in **microseconds** -> divide by 1000 for ms.
  - `RRU.PrbTotDl_avg`/`PrbTotUl_avg` (v_ue_kpm_1s) are already per-second PRB-used counts, not
    cumulative — no delta needed, already match "PRBs/s".
  - `mac_ue_1s.dl_aggr_tbs`/`ul_aggr_tbs`/`dl_aggr_prb`/`ul_aggr_prb` ARE session-cumulative
    counters — used only in fallback formulas, which need `.diff()`.
  - `mac_ue_1s.dl_bler_avg`/`ul_bler_avg` are already a 0-1 ratio.
  - `v_ue_kpm_1s` has no `session_id` — joined via `dim_ue_session_run{N}.csv`'s
    `(run_id,cell_id,ue_id) -> session_id` (1:1 for every exported run).
  - `rlc_bearer_1s.rbid` is always `1` in this export — no bearer-level aggregation needed.
  - `control_actions`'s only real parameter is `dedicated_prb_ratio`
    (`scope_type="slice"`, `unit="percent"`, `target`=slice name), mapped via `dim_slice.csv`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------
# The seven targets (prompt2.md Step 3's own table), and the exact units each is reported in
# AFTER this module's conversions — never touched again downstream.
# --------------------------------------------------------------------------------------------
TARGET_COLUMNS: tuple[str, ...] = (
    "dl_throughput",
    "ul_throughput",
    "dl_rlc_delay",
    "dl_prb",
    "ul_prb",
    "dl_bler",
    "ul_bler",
)

TARGET_UNITS: dict[str, str] = {
    "dl_throughput": "Mbps",
    "ul_throughput": "Mbps",
    "dl_rlc_delay": "ms",
    "dl_prb": "PRBs/s",
    "ul_prb": "PRBs/s",
    "dl_bler": "ratio_0_1",
    "ul_bler": "ratio_0_1",
}

# The 10 MAC-layer features named by prompt2.md Step 3, each used as both its `_avg` and `_std`
# per-second statistic (20 raw columns total before lagging).
MAC_FEATURE_BASES: tuple[str, ...] = (
    "dl_curr_tbs",
    "dl_sched_rb",
    "pusch_snr",
    "pucch_snr",
    "wb_cqi",
    "dl_mcs1",
    "ul_mcs1",
    "phr",
    "dl_bler",
    "ul_bler",
)

MAX_GAP_SECONDS = 2.0

# Phase 13 (claude_code_prompt_phase13.md §1.3): "For each target and for the 10 MAC features
# ..., add: lags t-1...t-5, rolling mean and std over 5s and 10s." These two tuples are the
# single source of truth for which lags/windows are computed for EVERY target and EVERY one of
# the 20 MAC avg/std columns — never a per-column special case.
LAGS: tuple[int, ...] = (1, 2, 3, 4, 5)
ROLLING_WINDOWS_SECONDS: tuple[int, ...] = (5, 10)


def _richly_lagged_feature_names(base: str) -> tuple[str, ...]:
    """Every column name `_multi_lag_features`/`_rolling_window_features` produce for one raw
    input column `base` — the exact, enumerable feature set phase13 §1.3 specifies per target/
    MAC column (5 lags + 2 windows x 2 stats = 9 columns)."""
    return (
        *(f"{base}_lag{k}" for k in LAGS),
        *(f"{base}_roll_mean_{w}s" for w in ROLLING_WINDOWS_SECONDS),
        *(f"{base}_roll_std_{w}s" for w in ROLLING_WINDOWS_SECONDS),
    )


# The 20 raw MAC avg/std columns that get the FULL lag1-5 + rolling treatment (phase13 §1.3's
# "10 MAC features" x {avg, std}) — `bsr_avg`/`n_samples` are NOT in this list (phase13 §1.3 names
# only these 10), so they stay lag1-only, same as before.
MAC_RICH_FEATURE_BASES: tuple[str, ...] = tuple(f"{b}_{stat}" for b in MAC_FEATURE_BASES for stat in ("avg", "std"))

# The complete, canonical lagged-feature-column list this pipeline ever produces — single
# source of truth shared by `build_features()` (which produces exactly these columns) and every
# Module 5-10-replacement DT component (`src/dt_models/dl_bler.py` etc.), which all declare this
# same set as `REQUIRED_FEATURES` (prompt2.md Step 5 doesn't specify a per-component feature
# SUBSET, only the dependency graph — every component gets the full lagged feature vector, since
# that is what "declares its inputs ... as metadata" means here: the full set of legitimately
# available, non-leaking inputs).
ALL_FEATURES: tuple[str, ...] = (
    *(name for t in TARGET_COLUMNS for name in _richly_lagged_feature_names(t)),
    *(name for b in MAC_RICH_FEATURE_BASES for name in _richly_lagged_feature_names(b)),
    "bsr_avg_lag1",
    "n_samples_lag1",
    "ue_count_cell_lag1",
    "ue_count_slice_lag1",
    "requested_prb_ratio_lag1",
    "has_controller_lag1",
)


@dataclass
class DataIssue:
    """One honestly-reported finding about a target/feature column, never silently swallowed
    (prompt2.md: "Report any target that is constant, mostly NaN, or missing in a split")."""

    run_id: int
    column: str
    kind: str  # "missing" | "constant" | "mostly_nan"
    detail: str


@dataclass
class RunLoadReport:
    run_id: int
    rows_mac: int = 0
    rows_rlc: int = 0
    rows_merged: int = 0
    counter_resets: dict[str, int] = field(default_factory=dict)
    gaps_found: dict[str, int] = field(default_factory=dict)
    fallback_used: dict[str, int] = field(default_factory=dict)
    issues: list[DataIssue] = field(default_factory=list)


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required OAI export file missing: {path}")
    return pd.read_csv(path, **kwargs)


def _to_utc(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    return df


def load_run_tables(oai_root: Path, run_id: int) -> dict[str, pd.DataFrame]:
    """Loads every raw CSV this pipeline needs for one run. Raises FileNotFoundError with the
    exact missing path if anything required is absent — never silently substitutes or skips
    (prompt2.md's own "if anything is missing, STOP and report exactly which files are missing").
    """
    tables: dict[str, pd.DataFrame] = {}

    mac = _read_csv(oai_root / f"mac_ue_1s_run{run_id}.csv")
    tables["mac_ue_1s"] = _to_utc(mac, ["ts_1s", "ts_first", "ts_last"])

    rlc = _read_csv(oai_root / f"rlc_bearer_1s_run{run_id}.csv")
    tables["rlc_bearer_1s"] = _to_utc(rlc, ["ts_1s", "ts_first", "ts_last"])

    v_kpm = _read_csv(oai_root / f"v_ue_kpm_1s_run{run_id}.csv")
    tables["v_ue_kpm_1s"] = _to_utc(v_kpm, ["ts_1s", "ts_first", "ts_last"])

    dim_sess = _read_csv(oai_root / f"dim_ue_session_run{run_id}.csv")
    tables["dim_ue_session"] = _to_utc(dim_sess, ["valid_from", "valid_to"])

    dim_slice = _read_csv(oai_root / "dim_slice.csv")
    tables["dim_slice"] = dim_slice

    ca_path = oai_root / f"control_actions_run{run_id}.csv"
    if ca_path.exists():
        ca = _read_csv(ca_path)
        ca["ts"] = pd.to_datetime(ca["ts"], utc=True, format="ISO8601")
        tables["control_actions"] = ca
    else:
        # A run genuinely without a deployment (e.g. a future bare-metal baseline run) is a real,
        # legitimate case this pipeline must degrade on honestly — not a file-existence failure
        # for the files Step 3 actually depends on. `has_controller` covers this per-row.
        tables["control_actions"] = pd.DataFrame(
            columns=["ts", "param", "scope_type", "target", "slice_id", "requested_value"]
        )

    return tables


def _session_map(dim_ue_session: pd.DataFrame) -> pd.DataFrame:
    """(run_id, cell_id, ue_id) -> session_id. 1:1 for every currently-exported run (verified) —
    if a future export ever has a UE re-attach mid-run, the earliest valid_from wins and this is
    flagged, never silently producing a many-to-one join explosion."""
    m = dim_ue_session[["run_id", "cell_id", "ue_id", "session_id", "valid_from"]].copy()
    m = m.sort_values("valid_from").drop_duplicates(subset=["run_id", "cell_id", "ue_id"], keep="first")
    return m[["run_id", "cell_id", "ue_id", "session_id"]]


def _safe_shift(df: pd.DataFrame, group_cols: list[str], time_col: str, value_cols: list[str], suffix: str = "") -> pd.DataFrame:
    """Plain `groupby(group_cols).shift(1)` after sorting by `time_col` within each group — the
    literal lagging rule prompt2.md Step 3 specifies for input features. Returns a DataFrame with
    the shifted columns renamed `<col><suffix>`, aligned to the original (unsorted) index."""
    ordered = df.sort_values(group_cols + [time_col], kind="stable")
    shifted = ordered.groupby(group_cols, sort=False)[value_cols].shift(1)
    shifted.columns = [f"{c}{suffix}" for c in shifted.columns]
    return shifted.reindex(df.index) if not ordered.index.equals(df.index) else shifted


def _multi_lag_features(
    df: pd.DataFrame, group_cols: list[str], time_col: str, value_cols: list[str], lags: tuple[int, ...] = LAGS
) -> pd.DataFrame:
    """Per-group, time-sorted: `shift(k)` for every `k` in `lags` (phase13 §1.3's literal
    "lags t-1 ... t-5" rule, generalized from `_safe_shift`'s single-lag case). Returns a
    DataFrame of `<col>_lag{k}` columns aligned to `df`'s original index."""
    ordered = df.sort_values(group_cols + [time_col], kind="stable")
    grouped = ordered.groupby(group_cols, sort=False)[value_cols]
    pieces = []
    for k in lags:
        shifted = grouped.shift(k)
        shifted.columns = [f"{c}_lag{k}" for c in shifted.columns]
        pieces.append(shifted)
    out = pd.concat(pieces, axis=1)
    return out.reindex(df.index) if not ordered.index.equals(df.index) else out


def _rolling_window_features(
    df: pd.DataFrame,
    group_cols: list[str],
    time_col: str,
    value_cols: list[str],
    windows_seconds: tuple[int, ...] = ROLLING_WINDOWS_SECONDS,
) -> pd.DataFrame:
    """Per-group, time-sorted: rolling mean/std of each column's own `shift(1)` series — i.e. a
    window strictly covering `[t-w, t-1]`, never `t` itself (phase13 §1.3: "computed strictly
    from past seconds"). Rolling on the ALREADY-shifted series, rather than shifting the rolling
    result, is what makes "never includes t" a structural guarantee instead of an off-by-one risk.
    Row-count windows (this pipeline's rows are nominally 1-second ticks; a real gap larger than
    `MAX_GAP_SECONDS` is already NaN'd by `compute_deltas` elsewhere for counter-derived columns,
    and a window can legitimately span a short real gap exactly like a lag already does). Returns
    `<col>_roll_mean_{w}s`/`<col>_roll_std_{w}s` columns aligned to `df`'s original index."""
    ordered = df.sort_values(group_cols + [time_col], kind="stable")
    lag1 = ordered.groupby(group_cols, sort=False)[value_cols].shift(1)
    lag1.index = ordered.index
    keyed_lag1 = pd.concat([ordered[group_cols], lag1], axis=1)
    pieces = []
    for w in windows_seconds:
        grouped_lag1 = keyed_lag1.groupby(group_cols, sort=False)[value_cols]
        mean = grouped_lag1.rolling(window=w, min_periods=1).mean().reset_index(level=list(range(len(group_cols))), drop=True)
        std = grouped_lag1.rolling(window=w, min_periods=1).std().reset_index(level=list(range(len(group_cols))), drop=True)
        mean.columns = [f"{c}_roll_mean_{w}s" for c in mean.columns]
        std.columns = [f"{c}_roll_std_{w}s" for c in std.columns]
        pieces.append(mean)
        pieces.append(std)
    out = pd.concat(pieces, axis=1)
    return out.reindex(df.index) if not ordered.index.equals(df.index) else out


def compute_deltas(
    df: pd.DataFrame, group_cols: list[str], time_col: str, counter_cols: list[str], max_gap_seconds: float = MAX_GAP_SECONDS
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """Per `(group_cols)`, time-sorted: `diff()` of each cumulative counter column.
    - A negative delta (counter reset) -> NaN, counted in `resets`.
    - A gap in `time_col` larger than `max_gap_seconds` -> NaN (never spread across the gap),
      counted in `gaps`.
    Returns (df with `<col>_delta`/`dt_seconds` columns added, resets-per-column, gaps-per-column).
    """
    ordered = df.sort_values(group_cols + [time_col], kind="stable").copy()
    grp = ordered.groupby(group_cols, sort=False)

    dt = grp[time_col].diff().dt.total_seconds()
    ordered["dt_seconds"] = dt
    gap_mask = dt > max_gap_seconds

    resets: dict[str, int] = {}
    gaps: dict[str, int] = {}
    for col in counter_cols:
        delta = grp[col].diff()
        reset_mask = delta < 0
        resets[col] = int(reset_mask.sum())
        gaps[col] = int(gap_mask.sum())
        delta = delta.where(~reset_mask, np.nan)
        delta = delta.where(~gap_mask, np.nan)
        ordered[f"{col}_delta"] = delta

    return ordered.reindex(df.index), resets, gaps


def build_targets(tables: dict[str, pd.DataFrame], run_id: int, report: RunLoadReport) -> pd.DataFrame:
    """Builds the 7 canonical ground-truth target columns for one run, one row per
    `(run_id, session_id, ts_1s)`, applying the documented unit conversions and the fallback
    formulas (used per-row only where the primary source is NaN)."""
    mac = tables["mac_ue_1s"].copy()
    rlc = tables["rlc_bearer_1s"].copy()
    v_kpm = tables["v_ue_kpm_1s"].copy()
    sess_map = _session_map(tables["dim_ue_session"])

    report.rows_mac = len(mac)
    report.rows_rlc = len(rlc)

    # v_ue_kpm_1s has no session_id — join it in via (run_id, cell_id, ue_id).
    v_kpm = v_kpm.merge(sess_map, on=["run_id", "cell_id", "ue_id"], how="left")
    dropped = v_kpm["session_id"].isna().sum()
    if dropped:
        report.issues.append(DataIssue(run_id, "v_ue_kpm_1s", "missing", f"{dropped} rows had no session mapping — dropped"))
    v_kpm = v_kpm.dropna(subset=["session_id"])
    v_kpm["session_id"] = v_kpm["session_id"].astype(mac["session_id"].dtype)

    # Fallback deltas, computed from mac_ue_1s's genuine cumulative counters.
    mac_delta, resets, gaps = compute_deltas(
        mac, ["run_id", "session_id"], "ts_1s", ["dl_aggr_tbs", "ul_aggr_tbs", "dl_aggr_prb", "ul_aggr_prb"]
    )
    report.counter_resets.update(resets)
    report.gaps_found.update(gaps)

    base = mac_delta[["run_id", "cell_id", "session_id", "ue_id", "slice_id", "ts_1s", "dt_seconds",
                       "dl_aggr_tbs_delta", "ul_aggr_tbs_delta", "dl_aggr_prb_delta", "ul_aggr_prb_delta",
                       "dl_bler_avg", "ul_bler_avg"]]

    merged = base.merge(
        v_kpm[["run_id", "session_id", "ts_1s", "DRB.UEThpDl_avg", "DRB.UEThpUl_avg", "RRU.PrbTotDl_avg", "RRU.PrbTotUl_avg"]],
        on=["run_id", "session_id", "ts_1s"], how="left",
    )
    merged = merged.merge(
        rlc[["run_id", "session_id", "ts_1s", "tx_sdu_avg_time_to_tx_avg"]],
        on=["run_id", "session_id", "ts_1s"], how="left",
    )

    out = merged[["run_id", "cell_id", "session_id", "ue_id", "slice_id", "ts_1s"]].copy()

    # dl/ul throughput: primary = v_ue_kpm_1s (kbps -> Mbps); fallback = 8*delta_bytes/dt/1e6.
    primary_dl_thp = merged["DRB.UEThpDl_avg"] / 1000.0
    fallback_dl_thp = (8.0 * merged["dl_aggr_tbs_delta"]) / merged["dt_seconds"] / 1.0e6
    out["dl_throughput"] = primary_dl_thp.where(primary_dl_thp.notna(), fallback_dl_thp)
    report.fallback_used["dl_throughput"] = int(primary_dl_thp.isna().sum())

    primary_ul_thp = merged["DRB.UEThpUl_avg"] / 1000.0
    fallback_ul_thp = (8.0 * merged["ul_aggr_tbs_delta"]) / merged["dt_seconds"] / 1.0e6
    out["ul_throughput"] = primary_ul_thp.where(primary_ul_thp.notna(), fallback_ul_thp)
    report.fallback_used["ul_throughput"] = int(primary_ul_thp.isna().sum())

    # dl/ul PRB: primary = v_ue_kpm_1s (already PRBs, no conversion); fallback = delta/dt.
    primary_dl_prb = merged["RRU.PrbTotDl_avg"]
    fallback_dl_prb = merged["dl_aggr_prb_delta"] / merged["dt_seconds"]
    out["dl_prb"] = primary_dl_prb.where(primary_dl_prb.notna(), fallback_dl_prb)
    report.fallback_used["dl_prb"] = int(primary_dl_prb.isna().sum())

    primary_ul_prb = merged["RRU.PrbTotUl_avg"]
    fallback_ul_prb = merged["ul_aggr_prb_delta"] / merged["dt_seconds"]
    out["ul_prb"] = primary_ul_prb.where(primary_ul_prb.notna(), fallback_ul_prb)
    report.fallback_used["ul_prb"] = int(primary_ul_prb.isna().sum())

    # dl_rlc_delay: ENTIRELY from rlc_bearer_1s (no DRB.RlcSduDelayDl column exists in this
    # export's v_ue_kpm_1s — verified directly, see module docstring). us -> ms.
    out["dl_rlc_delay"] = merged["tx_sdu_avg_time_to_tx_avg"] / 1000.0
    report.fallback_used["dl_rlc_delay"] = 0  # no fallback exists for this target

    # dl/ul BLER: already a 0-1 ratio, no conversion.
    out["dl_bler"] = merged["dl_bler_avg"]
    out["ul_bler"] = merged["ul_bler_avg"]
    report.fallback_used["dl_bler"] = 0
    report.fallback_used["ul_bler"] = 0

    report.rows_merged = len(out)
    return out


def _control_prb_ratio(tables: dict[str, pd.DataFrame], run_id: int, ts_grid: pd.Series) -> pd.DataFrame:
    """Per (run_id, slice_id, ts_1s): the most recently requested `dedicated_prb_ratio` for that
    slice as of (at-or-before) this second, carried forward as a step function, plus
    `has_controller` (1 iff at least one control action for this slice has occurred by this
    second). NaN / 0 respectively where no deployment exists for this run (e.g. a bare run)."""
    ca = tables["control_actions"]
    dim_slice = tables["dim_slice"]

    ts_df = pd.DataFrame({"ts_1s": pd.Series(sorted(ts_grid.unique()))})
    slices = dim_slice[["slice_id", "name"]].rename(columns={"name": "target"})

    if ca.empty:
        grid = ts_df.merge(slices, how="cross")
        grid["run_id"] = run_id
        grid["requested_prb_ratio"] = np.nan
        grid["has_controller"] = 0
        return grid[["run_id", "slice_id", "ts_1s", "requested_prb_ratio", "has_controller"]]

    actions = ca[ca["param"] == "dedicated_prb_ratio"][["ts", "target", "requested_value"]].copy()
    actions = actions.merge(slices, on="target", how="left").dropna(subset=["slice_id"])
    actions["slice_id"] = actions["slice_id"].astype(int)

    frames = []
    for slice_id, grp in actions.groupby("slice_id"):
        grp = grp.sort_values("ts")
        merged = pd.merge_asof(
            ts_df.sort_values("ts_1s"), grp.sort_values("ts"), left_on="ts_1s", right_on="ts", direction="backward"
        )
        merged["slice_id"] = slice_id
        merged["has_controller"] = merged["requested_value"].notna().astype(int)
        frames.append(merged[["ts_1s", "slice_id", "requested_value", "has_controller"]])

    present_slice_ids = set(actions["slice_id"].unique())
    for slice_id in slices["slice_id"]:
        if slice_id not in present_slice_ids:
            frames.append(pd.DataFrame({
                "ts_1s": ts_df["ts_1s"], "slice_id": slice_id, "requested_value": np.nan, "has_controller": 0,
            }))

    out = pd.concat(frames, ignore_index=True)
    out["run_id"] = run_id
    out = out.rename(columns={"requested_value": "requested_prb_ratio"})
    return out[["run_id", "slice_id", "ts_1s", "requested_prb_ratio", "has_controller"]]


def build_features(tables: dict[str, pd.DataFrame], targets: pd.DataFrame, run_id: int) -> pd.DataFrame:
    """Builds every input feature prompt2.md Step 3 lists, ALL lagged by one second
    (`groupby(["run_id","session_id"]).shift(1)`, the literal rule specified), merged onto the
    target rows by `(run_id, session_id, ts_1s)`."""
    mac = tables["mac_ue_1s"].copy()

    # Phase 13 §1.3: the 10 MAC features (x {avg, std} = 20 raw columns) get the FULL
    # lag1-5 + rolling-5s/10s-mean/std treatment; bsr_avg/n_samples (not in that list) stay
    # lag1-only, unchanged from before.
    mac_rich_cols = [c for c in MAC_RICH_FEATURE_BASES if c in mac.columns]
    mac_lag1_only_cols = [c for c in ("bsr_avg", "n_samples") if c in mac.columns]
    mac_rich_lags = _multi_lag_features(mac, ["run_id", "session_id"], "ts_1s", mac_rich_cols)
    mac_rich_rolling = _rolling_window_features(mac, ["run_id", "session_id"], "ts_1s", mac_rich_cols)
    mac_lag1 = _safe_shift(mac, ["run_id", "session_id"], "ts_1s", mac_lag1_only_cols, suffix="_lag1")
    mac_lagged = pd.concat(
        [mac[["run_id", "cell_id", "session_id", "slice_id", "ts_1s"]], mac_rich_lags, mac_rich_rolling, mac_lag1], axis=1
    )

    # Lagged + rolling targets (own past values are legitimate predictor inputs — phase13 §1.3
    # applies the same lag1-5 + rolling-5s/10s treatment to targets as to the 10 MAC features).
    target_rich_lags = _multi_lag_features(targets, ["run_id", "session_id"], "ts_1s", list(TARGET_COLUMNS))
    target_rich_rolling = _rolling_window_features(targets, ["run_id", "session_id"], "ts_1s", list(TARGET_COLUMNS))
    targets_for_merge = pd.concat(
        [targets[["run_id", "session_id", "ts_1s"]], target_rich_lags, target_rich_rolling], axis=1
    )

    # UE count per cell and per slice, previous second — computed on the cell/slice-time grid
    # (one row per distinct (cell,ts_1s) or (slice,ts_1s)), shifted, then broadcast back to every
    # UE row sharing that (cell,ts_1s)/(slice,ts_1s) — NOT a per-UE shift, since this is a
    # cell/slice-wide quantity, not a per-UE one.
    cell_counts = mac.groupby(["run_id", "cell_id", "ts_1s"])["ue_id"].nunique().rename("ue_count_cell").reset_index()
    cell_lag = _safe_shift(cell_counts, ["run_id", "cell_id"], "ts_1s", ["ue_count_cell"], suffix="")
    cell_counts["ue_count_cell_lag1"] = cell_lag["ue_count_cell"]

    slice_counts = mac.groupby(["run_id", "slice_id", "ts_1s"])["ue_id"].nunique().rename("ue_count_slice").reset_index()
    slice_lag = _safe_shift(slice_counts, ["run_id", "slice_id"], "ts_1s", ["ue_count_slice"], suffix="")
    slice_counts["ue_count_slice_lag1"] = slice_lag["ue_count_slice"]

    # Controller PRB ratio for the UE's own slice, as-of-this-second, THEN lagged by one more
    # second like every other feature in this list (never a same-second controller value).
    prb_ratio = _control_prb_ratio(tables, run_id, mac["ts_1s"])
    prb_lag = _safe_shift(
        prb_ratio.sort_values(["slice_id", "ts_1s"]), ["run_id", "slice_id"], "ts_1s",
        ["requested_prb_ratio", "has_controller"], suffix="_lag1",
    )
    prb_ratio = pd.concat([prb_ratio[["run_id", "slice_id", "ts_1s"]], prb_lag], axis=1)
    prb_ratio["has_controller_lag1"] = prb_ratio["has_controller_lag1"].fillna(0).astype(int)

    features = targets_for_merge.merge(
        mac_lagged, on=["run_id", "session_id", "ts_1s"], how="left"
    )
    features = features.merge(cell_counts[["run_id", "cell_id", "ts_1s", "ue_count_cell_lag1"]],
                               on=["run_id", "cell_id", "ts_1s"], how="left")
    features = features.merge(slice_counts[["run_id", "slice_id", "ts_1s", "ue_count_slice_lag1"]],
                               on=["run_id", "slice_id", "ts_1s"], how="left")
    features = features.merge(prb_ratio[["run_id", "slice_id", "ts_1s", "requested_prb_ratio_lag1", "has_controller_lag1"]],
                               on=["run_id", "slice_id", "ts_1s"], how="left")
    features["has_controller_lag1"] = features["has_controller_lag1"].fillna(0).astype(int)

    return features


def build_run_table(oai_root: Path, run_id: int) -> tuple[pd.DataFrame, RunLoadReport]:
    report = RunLoadReport(run_id=run_id)
    tables = load_run_tables(oai_root, run_id)
    targets = build_targets(tables, run_id, report)
    features = build_features(tables, targets, run_id)

    # One row per (run_id, session_id, ts_1s): identity + lagged features (model INPUT) + the
    # real, un-lagged target columns (ground truth for THIS second — what fidelity compares
    # against) + slice_id as a plain categorical passthrough.
    out = targets.merge(features, on=["run_id", "session_id", "ts_1s"], how="left", suffixes=("", "_feat"))
    out = out.drop(columns=[c for c in out.columns if c.endswith("_feat")])
    out = out.sort_values(["run_id", "session_id", "ts_1s"]).reset_index(drop=True)

    for col in TARGET_COLUMNS:
        nan_rate = out[col].isna().mean()
        is_constant = out[col].dropna().nunique() <= 1
        if nan_rate > 0.5:
            report.issues.append(DataIssue(run_id, col, "mostly_nan", f"{nan_rate:.1%} NaN"))
        if is_constant and out[col].notna().any():
            report.issues.append(DataIssue(run_id, col, "constant", f"only value: {out[col].dropna().iloc[0]}"))

    return out, report


LEAKAGE_SAFE_STATIC_COLUMNS = {"run_id", "cell_id", "session_id", "ue_id", "slice_id", "ts_1s"}

_LAG_SUFFIXES = tuple(f"_lag{k}" for k in LAGS)
_ROLLING_SUFFIXES = tuple(
    f"_roll_{stat}_{w}s" for w in ROLLING_WINDOWS_SECONDS for stat in ("mean", "std")
)


def _is_leakage_safe_feature_name(c: str) -> bool:
    """True iff `c` is structurally a lagged (`_lag{1..5}`) or rolling-past-window
    (`_roll_{mean,std}_{5,10}s`) column, or one of the two cell/slice UE-count exceptions —
    never a bare, same-second measured value (prompt2.md Step 3 / phase13 §1.3's leakage rule)."""
    return c.endswith(_LAG_SUFFIXES) or c.endswith(_ROLLING_SUFFIXES) or c in ("ue_count_cell_lag1", "ue_count_slice_lag1")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Every column intended as model INPUT — lagged or rolling-past-window, or static/
    categorical identity — never a same-second measured target. Used both to build the final
    written `feature_schema.json` and by the leakage test (prompt2.md Step 3's own leakage rule)."""
    cols = []
    for c in df.columns:
        if c in TARGET_COLUMNS or c in LEAKAGE_SAFE_STATIC_COLUMNS:
            continue
        if _is_leakage_safe_feature_name(c):
            cols.append(c)
    return cols


@dataclass
class DatasetBuildResult:
    splits: dict[str, pd.DataFrame]
    reports: dict[int, RunLoadReport]
    feature_schema: dict[str, Any]


def build_dataset(oai_root: Path, splits: dict[str, list[int]]) -> DatasetBuildResult:
    """Builds every configured split's table. `splits` e.g. {"train": [3, 6], "val": [7], "test": [8]}."""
    run_tables: dict[int, pd.DataFrame] = {}
    reports: dict[int, RunLoadReport] = {}

    all_run_ids = sorted({r for runs in splits.values() for r in runs})
    seen_splits: dict[int, str] = {}
    for split_name, run_ids in splits.items():
        for r in run_ids:
            if r in seen_splits:
                raise ValueError(f"run {r} appears in both split '{seen_splits[r]}' and '{split_name}' — splits must be disjoint")
            seen_splits[r] = split_name

    for run_id in all_run_ids:
        table, report = build_run_table(oai_root, run_id)
        run_tables[run_id] = table
        reports[run_id] = report

    split_tables = {
        name: pd.concat([run_tables[r] for r in run_ids], ignore_index=True) if run_ids else pd.DataFrame()
        for name, run_ids in splits.items()
    }

    feat_cols = feature_columns(next(iter(run_tables.values())))
    schema = {
        "targets": {c: TARGET_UNITS[c] for c in TARGET_COLUMNS},
        "features": feat_cols,
        "lag_seconds": 1,
        "identity_columns": sorted(LEAKAGE_SAFE_STATIC_COLUMNS),
        "splits": splits,
        "source": "OAI_RAN_REPLAY",
    }

    return DatasetBuildResult(splits=split_tables, reports=reports, feature_schema=schema)


def write_outputs(result: DatasetBuildResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in result.splits.items():
        df.to_parquet(out_dir / f"{name}.parquet", engine="pyarrow", index=False)

    schema_path = out_dir / "feature_schema.json"
    schema_path.write_text(json.dumps(result.feature_schema, indent=2, default=str), encoding="utf-8")

    report_lines = ["# OAI Dataset Build Report\n"]
    for name, df in result.splits.items():
        report_lines.append(f"\n## Split: {name} ({len(df)} rows)\n")
        for col in TARGET_COLUMNS:
            if col not in df.columns:
                report_lines.append(f"- {col}: MISSING from this split\n")
                continue
            nan_rate = df[col].isna().mean() if len(df) else float("nan")
            report_lines.append(f"- {col}: nan_rate={nan_rate:.2%}, n={df[col].notna().sum()}\n")
    for run_id, report in result.reports.items():
        report_lines.append(f"\n## Run {run_id}\n")
        report_lines.append(f"- rows_mac={report.rows_mac}, rows_rlc={report.rows_rlc}, rows_merged={report.rows_merged}\n")
        report_lines.append(f"- counter_resets: {report.counter_resets}\n")
        report_lines.append(f"- gaps_found(>{MAX_GAP_SECONDS}s): {report.gaps_found}\n")
        report_lines.append(f"- fallback_used (primary-NaN rows): {report.fallback_used}\n")
        for issue in report.issues:
            report_lines.append(f"- ISSUE [{issue.kind}] {issue.column}: {issue.detail}\n")

    report_path = out_dir / "data_report.md"
    report_path.write_text("".join(report_lines), encoding="utf-8")
    return report_path
