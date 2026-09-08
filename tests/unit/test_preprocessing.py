"""Unit tests for TelemetryPreprocessor (Module 2 core — prompt.md §8).

Proves: raw records go in, validated feature records / time-series windows (timestamp, UE ID,
cell ID) come out correctly — unit normalization, missing-value handling (carry-forward vs
quarantine), out-of-range flagging (kept, not silently dropped), timestamp synchronization, and
feature-window construction.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.common.config import PreprocessingConfig
from src.telemetry.preprocessing import TelemetryPreprocessor


def _config(**overrides) -> PreprocessingConfig:
    base = dict(
        required_fields=[
            "timestamp",
            "ue_id",
            "cell_id",
            "throughput_mbps",
            "offered_load_mbps",
            "latency_ms",
            "jitter_ms",
            "packet_loss_pct",
            "prb_utilization_pct",
            "sinr_db",
            "rsrp_dbm",
            "rsrq_db",
            "ue_count",
            "ue_speed_mps",
        ],
        valid_ranges={
            "throughput_mbps": (0.0, 10000.0),
            "offered_load_mbps": (0.0, 10000.0),
            "latency_ms": (0.0, 5000.0),
            "jitter_ms": (0.0, 2000.0),
            "packet_loss_pct": (0.0, 100.0),
            "prb_utilization_pct": (0.0, 100.0),
            "sinr_db": (-20.0, 40.0),
            "rsrp_dbm": (-140.0, -40.0),
            "rsrq_db": (-30.0, 0.0),
            "ue_speed_mps": (0.0, 100.0),
        },
        feature_window_size=5,
        max_missing_ratio=0.2,
    )
    base.update(overrides)
    return PreprocessingConfig(**base)


def _raw_record(**overrides) -> dict:
    base = dict(
        source="MOCK",
        timestamp=1_700_000_000.0,
        ue_id="ue-0",
        cell_id="cell-0",
        throughput_bps=50_000_000.0,  # 50 Mbps
        offered_load_bps=60_000_000.0,  # 60 Mbps
        latency_s=0.020,  # 20 ms
        jitter_s=0.003,  # 3 ms
        packet_loss_ratio=0.01,  # 1%
        prb_utilization_ratio=0.4,  # 40%
        sinr_db=18.0,
        rsrp_dbm=-90.0,
        rsrq_db=-9.0,
        ue_count=12,
        ue_speed_mps=3.5,
        ue_position_x=100.0,
        ue_position_y=-50.0,
    )
    base.update(overrides)
    return base


# --- unit normalization + happy path -----------------------------------------------------


def test_valid_record_normalizes_units_correctly():
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(_raw_record())

    assert quarantined is None
    assert clean is not None
    assert clean.throughput_mbps == pytest.approx(50.0)
    assert clean.offered_load_mbps == pytest.approx(60.0)
    assert clean.latency_ms == pytest.approx(20.0)
    assert clean.jitter_ms == pytest.approx(3.0)
    assert clean.packet_loss_pct == pytest.approx(1.0)
    assert clean.prb_utilization_pct == pytest.approx(40.0)
    # passthrough fields unchanged
    assert clean.sinr_db == 18.0
    assert clean.rsrp_dbm == -90.0
    assert clean.rsrq_db == -9.0
    assert clean.ue_count == 12
    assert clean.ue_speed_mps == 3.5
    assert clean.source == "MOCK"
    assert clean.quality.is_perfect


def test_identity_fields_preserved_exactly():
    pre = TelemetryPreprocessor(_config())
    clean, _ = pre.process_record(_raw_record(ue_id="ue-42", cell_id="cell-7"))
    assert clean.ue_id == "ue-42"
    assert clean.cell_id == "cell-7"


def test_timestamp_synchronized_to_utc_datetime_from_float_epoch():
    pre = TelemetryPreprocessor(_config())
    clean, _ = pre.process_record(_raw_record(timestamp=1_700_000_000.0))
    assert isinstance(clean.timestamp, datetime)
    assert clean.timestamp.tzinfo is not None
    assert clean.timestamp == datetime.fromtimestamp(1_700_000_000.0, tz=UTC)


def test_timestamp_synchronized_from_iso_string():
    pre = TelemetryPreprocessor(_config())
    clean, _ = pre.process_record(_raw_record(timestamp="2023-11-14T22:13:20+00:00"))
    assert clean is not None
    assert clean.timestamp.year == 2023


# --- schema / source validation ----------------------------------------------------------


def test_unknown_source_is_quarantined_not_silently_dropped():
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(_raw_record(source="SOME_OTHER_SOURCE"))
    assert clean is None
    assert quarantined is not None
    assert "invalid_source" in quarantined.reason
    assert quarantined.raw["ue_id"] == "ue-0"  # raw payload retained for audit, not discarded


def test_missing_identity_field_is_quarantined():
    raw = _raw_record()
    del raw["cell_id"]
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(raw)
    assert clean is None
    assert "missing_identity_field" in quarantined.reason


def test_unparseable_timestamp_is_quarantined():
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(_raw_record(timestamp="not-a-timestamp"))
    assert clean is None
    assert quarantined.reason == "unparseable_timestamp"


# --- missing value handling: carry-forward vs quarantine --------------------------------


def test_missing_required_field_without_history_is_quarantined():
    raw = _raw_record()
    del raw["sinr_db"]
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(raw)
    assert clean is None
    assert "missing_required_fields" in quarantined.reason
    assert "sinr_db" in quarantined.reason


def test_missing_required_field_with_prior_history_is_imputed_not_quarantined():
    pre = TelemetryPreprocessor(_config())
    # First record establishes history for this (ue_id, cell_id).
    first, _ = pre.process_record(_raw_record(sinr_db=18.0))
    assert first is not None

    raw_missing = _raw_record()
    del raw_missing["sinr_db"]
    clean, quarantined = pre.process_record(raw_missing)

    assert quarantined is None
    assert clean is not None
    assert clean.sinr_db == 18.0  # carried forward from prior record
    assert "sinr_db" in clean.quality.imputed_fields


def test_missing_optional_field_does_not_block_or_quarantine():
    raw = _raw_record()
    del raw["ue_position_x"]
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(raw)
    assert quarantined is None
    assert clean is not None
    assert clean.ue_position_x is None
    assert "ue_position_x" in clean.quality.missing_fields


# --- flagging (kept, not discarded) vs quarantine ----------------------------------------


def test_out_of_range_value_is_flagged_but_record_is_kept():
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(_raw_record(packet_loss_ratio=1.5))  # -> 150%, out of [0,100]
    assert quarantined is None
    assert clean is not None
    assert clean.packet_loss_pct == pytest.approx(150.0)
    assert "packet_loss_pct" in clean.quality.out_of_range_fields


def test_multiple_out_of_range_fields_all_flagged():
    pre = TelemetryPreprocessor(_config())
    clean, _ = pre.process_record(_raw_record(sinr_db=999.0, rsrp_dbm=10.0))
    assert set(clean.quality.out_of_range_fields) >= {"sinr_db", "rsrp_dbm"}


# --- batch processing + quality summary ---------------------------------------------------


def test_process_batch_never_loses_a_record():
    pre = TelemetryPreprocessor(_config())
    raw_records = [
        _raw_record(ue_id="ue-0"),
        _raw_record(ue_id="ue-1", source="INVALID"),
        _raw_record(ue_id="ue-2", timestamp="garbage"),
    ]
    result = pre.process_batch(raw_records)
    assert result.total_received == 3
    assert len(result.clean_records) == 1
    assert len(result.quarantined) == 2


def test_mock_source_never_produces_ns3_labeled_clean_records():
    pre = TelemetryPreprocessor(_config())
    result = pre.process_batch([_raw_record(source="MOCK") for _ in range(5)])
    assert all(r.source == "MOCK" for r in result.clean_records)
    assert not any(r.source == "NS3_5G_LENA" for r in result.clean_records)


# --- feature window construction ----------------------------------------------------------


def test_feature_windows_group_by_ue_and_cell_and_sort_by_time():
    pre = TelemetryPreprocessor(_config(feature_window_size=3))
    raw_records = []
    for i in range(7):
        raw_records.append(
            _raw_record(ue_id="ue-0", cell_id="cell-0", timestamp=1_700_000_000.0 + (6 - i))
        )  # descending timestamps on purpose — windowing must sort
    result = pre.process_batch(raw_records)
    windows = pre.build_feature_windows(result.clean_records)

    assert len(windows) == 3  # 3 + 3 + 1 (trailing partial window kept, not dropped)
    assert windows[0].size == 3
    assert windows[1].size == 3
    assert windows[2].size == 1
    assert windows[0].ue_id == "ue-0"
    assert windows[0].cell_id == "cell-0"

    frame = windows[0].frame
    assert list(frame.columns[:3]) == ["timestamp", "ue_id", "cell_id"]
    assert frame["timestamp"].is_monotonic_increasing


def test_feature_windows_separate_by_key():
    pre = TelemetryPreprocessor(_config(feature_window_size=10))
    raw_records = [_raw_record(ue_id="ue-0", cell_id="cell-0")] * 4 + [
        _raw_record(ue_id="ue-1", cell_id="cell-0")
    ] * 4
    result = pre.process_batch(raw_records)
    windows = pre.build_feature_windows(result.clean_records)
    keys = {(w.ue_id, w.cell_id) for w in windows}
    assert keys == {("ue-0", "cell-0"), ("ue-1", "cell-0")}
    assert all(w.size == 4 for w in windows)


def test_feature_window_missing_ratio_and_quality_flag():
    pre = TelemetryPreprocessor(_config(feature_window_size=4, max_missing_ratio=0.2))
    # Establish history so later missing fields can be imputed (imputed still counts as imperfect).
    pre.process_record(_raw_record(ue_id="ue-0", cell_id="cell-0"))

    raw_records = []
    for i in range(4):
        raw = _raw_record(ue_id="ue-0", cell_id="cell-0", timestamp=1_700_000_100.0 + i)
        if i < 2:
            del raw["sinr_db"]  # imputed -> counts as imperfect
        raw_records.append(raw)

    result = pre.process_batch(raw_records)
    windows = pre.build_feature_windows(result.clean_records)
    assert len(windows) == 1
    window = windows[0]
    assert window.missing_ratio == pytest.approx(2 / 4)
    assert window.quality_flag == "low_quality"  # 0.5 > max_missing_ratio 0.2


def test_feature_window_ok_when_below_missing_ratio_threshold():
    pre = TelemetryPreprocessor(_config(feature_window_size=4, max_missing_ratio=0.5))
    pre.process_record(_raw_record(ue_id="ue-0", cell_id="cell-0"))
    raw_records = []
    for i in range(4):
        raw = _raw_record(ue_id="ue-0", cell_id="cell-0", timestamp=1_700_000_200.0 + i)
        if i == 0:
            del raw["sinr_db"]
        raw_records.append(raw)
    result = pre.process_batch(raw_records)
    windows = pre.build_feature_windows(result.clean_records)
    assert windows[0].quality_flag == "ok"


def test_no_records_produces_no_windows():
    pre = TelemetryPreprocessor(_config())
    assert pre.build_feature_windows([]) == []


# --- NaN/Inf handling (prompt.md §0.26 "NaN/Inf propagation") ----------------------------


def test_nan_in_required_field_is_treated_as_missing_not_crashed_on():
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(_raw_record(sinr_db=float("nan")))
    assert clean is None
    assert quarantined is not None
    assert "sinr_db" in quarantined.reason


def test_inf_in_required_field_is_treated_as_missing_not_crashed_on():
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(_raw_record(rsrp_dbm=float("inf")))
    assert clean is None
    assert "rsrp_dbm" in quarantined.reason


def test_nan_never_pollutes_the_carry_forward_cache():
    pre = TelemetryPreprocessor(_config())
    good, _ = pre.process_record(_raw_record(sinr_db=18.0))
    assert good is not None

    # NaN is treated as "missing" -> carry-forward imputes the last GOOD value (18.0), not NaN
    # itself, and NaN is never written into the cache in the process.
    nan_raw = _raw_record(sinr_db=float("nan"))
    clean_nan, quarantined_nan = pre.process_record(nan_raw)
    assert quarantined_nan is None
    assert clean_nan is not None
    assert clean_nan.sinr_db == 18.0
    assert "sinr_db" in clean_nan.quality.imputed_fields

    # A subsequent record missing the field entirely must still see the pre-NaN good value.
    raw_missing = _raw_record()
    del raw_missing["sinr_db"]
    clean, _ = pre.process_record(raw_missing)
    assert clean is not None
    assert clean.sinr_db == 18.0  # carry-forward still returns the last GOOD value, not NaN


def test_int_field_receiving_nan_is_quarantined_not_a_crash():
    """Regression test: ue_count is a pydantic `int` field — a NaN reaching the schema
    constructor directly would previously raise ValidationError and crash process_batch."""
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(_raw_record(ue_count=float("nan")))
    assert clean is None
    assert quarantined is not None


def test_non_numeric_passthrough_field_is_treated_as_missing_not_a_crash():
    """Regression test: a malformed string in a passthrough numeric field (e.g. sinr_db) used
    to raise a bare TypeError from the `lo <= value <= hi` range check before reaching any
    quarantine path — untrusted telemetry input must never crash the batch (prompt.md §44)."""
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(_raw_record(sinr_db="not-a-number"))
    assert clean is None
    assert quarantined is not None
    assert "sinr_db" in quarantined.reason


def test_ue_count_survives_passthrough_coercion_as_int():
    pre = TelemetryPreprocessor(_config())
    clean, _ = pre.process_record(_raw_record(ue_count=7))
    assert clean is not None
    assert clean.ue_count == 7
    assert isinstance(clean.ue_count, int)
