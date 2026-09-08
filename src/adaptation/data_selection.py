"""Shared training-data selection helpers for the adaptation agents (Modules 14/15/16) that
retrain a `DTComponent` against a recent D1 snapshot — recency windowing, a time-ordered train/
held-out split, and populating a dependency's ground-truth column for training. Extracted here
(rather than duplicated per agent) once Module 15 needed the exact same logic Module 14 already
had — genuinely shared, not agent-specific.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd


class DataSelectionError(Exception):
    """Raised when a dependency's ground-truth column can't be resolved — never silently
    skipped, since training on a missing/wrong column would silently corrupt the candidate."""


def select_recent_window(history: pd.DataFrame, window_hours: float) -> pd.DataFrame:
    """Filter `history` (a D1 snapshot) to rows within the last `window_hours`."""
    if len(history) == 0:
        return history
    cutoff = datetime.now(UTC) - timedelta(hours=window_hours)
    timestamps = history["timestamp"]
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(UTC)
    return history[timestamps >= cutoff].reset_index(drop=True)


def time_split(df: pd.DataFrame, held_out_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered train/held-out split — avoids any leakage between the two sets."""
    ordered = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    split_at = max(1, int(len(ordered) * (1.0 - held_out_fraction)))
    return ordered.iloc[:split_at].reset_index(drop=True), ordered.iloc[split_at:].reset_index(drop=True)


def with_dependency_ground_truth(
    df: pd.DataFrame, dependencies: tuple[str, ...], dependency_output_fields: dict[str, str] | None
) -> pd.DataFrame:
    """Populate each dependency's `OUTPUT_FIELD` column from D1's ground-truth value for training
    — the "train on ground truth, serve on live predictions" convention Modules 7/9/10 already
    established, applied generically here instead of re-special-cased per component.
    `dependency_output_fields` maps each `DEPENDENCIES` entry to that dependency's `OUTPUT_FIELD`
    (e.g. `{"throughput": "throughput_mbps_pred"}`)."""
    if not dependencies:
        return df
    if dependency_output_fields is None:
        raise DataSelectionError(
            f"component has DEPENDENCIES={dependencies!r} but no dependency_output_fields mapping was supplied"
        )
    df = df.copy()
    for dep in dependencies:
        if dep not in dependency_output_fields:
            raise DataSelectionError(f"dependency_output_fields is missing an entry for dependency {dep!r}")
        output_field = dependency_output_fields[dep]
        ground_truth_col = output_field.removesuffix("_pred")
        if ground_truth_col not in df.columns:
            raise DataSelectionError(
                f"ground-truth column {ground_truth_col!r} for dependency {dep!r} not found in D1 history"
            )
        df[output_field] = df[ground_truth_col]
    return df


def iso(value: Any) -> str:
    return pd.Timestamp(value).isoformat()
