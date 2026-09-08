"""Unit tests for the shared training-data selection helpers (`src/adaptation/data_selection.py`)
used by both the Recalibration Agent (Module 14) and the Regeneration Agent (Module 15).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.adaptation.data_selection import (
    DataSelectionError,
    select_recent_window,
    time_split,
    with_dependency_ground_truth,
)


def test_select_recent_window_excludes_rows_older_than_the_window():
    now = datetime.now(UTC)
    df = pd.DataFrame(
        {
            "timestamp": [now - timedelta(hours=1), now - timedelta(hours=10), now - timedelta(hours=100)],
            "value": [1, 2, 3],
        }
    )
    windowed = select_recent_window(df, window_hours=24.0)
    assert list(windowed["value"]) == [1, 2]


def test_select_recent_window_on_empty_history_returns_empty():
    df = pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns, UTC]"), "value": pd.Series([], dtype=int)})
    assert len(select_recent_window(df, window_hours=24.0)) == 0


def test_time_split_is_time_ordered_and_never_leaks():
    df = pd.DataFrame({"timestamp": [3, 1, 2, 5, 4], "value": ["c", "a", "b", "e", "d"]})
    train_df, held_out_df = time_split(df, held_out_fraction=0.4)
    assert list(train_df["value"]) == ["a", "b", "c"]
    assert list(held_out_df["value"]) == ["d", "e"]


def test_with_dependency_ground_truth_populates_pred_column():
    df = pd.DataFrame({"throughput_mbps": [10.0, 20.0]})
    result = with_dependency_ground_truth(df, ("throughput",), {"throughput": "throughput_mbps_pred"})
    assert list(result["throughput_mbps_pred"]) == [10.0, 20.0]


def test_with_dependency_ground_truth_noop_for_root_component():
    df = pd.DataFrame({"a": [1]})
    result = with_dependency_ground_truth(df, (), None)
    assert result is df


def test_with_dependency_ground_truth_raises_when_mapping_missing():
    df = pd.DataFrame({"throughput_mbps": [10.0]})
    with pytest.raises(DataSelectionError):
        with_dependency_ground_truth(df, ("throughput",), None)


def test_with_dependency_ground_truth_raises_when_ground_truth_column_absent():
    df = pd.DataFrame({"unrelated": [1]})
    with pytest.raises(DataSelectionError):
        with_dependency_ground_truth(df, ("throughput",), {"throughput": "throughput_mbps_pred"})
