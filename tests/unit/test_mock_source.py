"""Unit tests for MockTelemetrySource (Module 2 — real vs mock separation, prompt.md §0.3)."""

from __future__ import annotations

from src.common.config import MockTelemetryConfig
from src.telemetry.mock_source import MockTelemetrySource

RAW_NUMERIC_FIELDS = {
    "throughput_bps",
    "offered_load_bps",
    "latency_s",
    "jitter_s",
    "packet_loss_ratio",
    "prb_utilization_ratio",
    "sinr_db",
    "rsrp_dbm",
    "rsrq_db",
    "ue_count",
    "ue_speed_mps",
}


def _config(**overrides) -> MockTelemetryConfig:
    base = dict(
        seed=42,
        emit_interval_seconds=0.01,
        num_ues=6,
        num_cells=2,
        missing_field_rate=0.0,
        out_of_range_rate=0.0,
    )
    base.update(overrides)
    return MockTelemetryConfig(**base)


def test_records_are_stamped_mock_never_ns3():
    source = MockTelemetrySource(_config(), max_records=20, realtime=False)
    records = list(source.records())
    assert len(records) == 20
    assert all(r["source"] == "MOCK" for r in records)
    assert all(r["source"] != "NS3_5G_LENA" for r in records)


def test_all_expected_raw_fields_present_when_no_injected_imperfection():
    source = MockTelemetrySource(_config(), max_records=12, realtime=False)
    for record in source.records():
        for field_name in RAW_NUMERIC_FIELDS:
            assert field_name in record, f"missing {field_name}"
        assert "timestamp" in record
        assert "ue_id" in record
        assert "cell_id" in record
        assert "ue_position_x" in record
        assert "ue_position_y" in record


def test_ue_and_cell_ids_distributed_across_configured_topology():
    config = _config(num_ues=6, num_cells=2)
    source = MockTelemetrySource(config, max_records=6, realtime=False)
    records = list(source.records())
    assert {r["ue_id"] for r in records} == {f"ue-{i}" for i in range(6)}
    assert {r["cell_id"] for r in records} <= {"cell-0", "cell-1"}


def test_reproducible_with_same_seed():
    source_a = MockTelemetrySource(_config(seed=7), max_records=10, realtime=False)
    source_b = MockTelemetrySource(_config(seed=7), max_records=10, realtime=False)
    records_a = list(source_a.records())
    records_b = list(source_b.records())
    assert [r["throughput_bps"] for r in records_a] == [r["throughput_bps"] for r in records_b]


def test_different_seeds_diverge():
    source_a = MockTelemetrySource(_config(seed=1), max_records=10, realtime=False)
    source_b = MockTelemetrySource(_config(seed=2), max_records=10, realtime=False)
    records_a = list(source_a.records())
    records_b = list(source_b.records())
    assert [r["throughput_bps"] for r in records_a] != [r["throughput_bps"] for r in records_b]


def test_max_records_bounds_generation():
    source = MockTelemetrySource(_config(), max_records=5, realtime=False)
    assert len(list(source.records())) == 5


def test_missing_field_rate_actually_drops_fields():
    config = _config(missing_field_rate=1.0, out_of_range_rate=0.0)
    source = MockTelemetrySource(config, max_records=10, realtime=False)
    records = list(source.records())
    full_field_count = len(RAW_NUMERIC_FIELDS) + 6  # + source, timestamp, ue_id, cell_id, position x/y
    # With rate=1.0 every record has exactly one non-identity field removed.
    assert all(len(r) == full_field_count - 1 for r in records)


def test_out_of_range_rate_actually_corrupts_values():
    config = _config(missing_field_rate=0.0, out_of_range_rate=1.0)
    baseline = _config(missing_field_rate=0.0, out_of_range_rate=0.0)
    source_corrupt = MockTelemetrySource(config, max_records=10, realtime=False)
    source_clean = MockTelemetrySource(baseline, max_records=10, realtime=False)
    corrupt_records = list(source_corrupt.records())
    clean_records = list(source_clean.records())
    # Same seed -> same underlying values before corruption; corrupted run must differ somewhere.
    diffs = 0
    for c, clean in zip(corrupt_records, clean_records, strict=True):
        for field_name in RAW_NUMERIC_FIELDS & c.keys() & clean.keys():
            if c[field_name] != clean[field_name]:
                diffs += 1
    assert diffs > 0


def test_positions_move_smoothly_between_ticks():
    config = _config(num_ues=1, num_cells=1, emit_interval_seconds=1.0)
    source = MockTelemetrySource(config, max_records=3, realtime=False)
    positions = [(r["ue_position_x"], r["ue_position_y"]) for r in source.records()]
    assert len(set(positions)) > 1  # UE actually moves across ticks
