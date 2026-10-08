"""Unit tests for MockTelemetrySource (Module 2 — real vs mock separation, prompt.md §0.3) —
OAI migration: mock records now use the OAI schema (run_id/session_id/cell_id/slice_id/ts_1s,
the 7 raw targets, the 33 lagged features), never the retired NS-3-era field set.
"""

from __future__ import annotations

from src.common.config import MockTelemetryConfig
from src.data.oai_dataset import ALL_FEATURES, TARGET_COLUMNS
from src.telemetry.mock_source import MockTelemetrySource

IDENTITY_FIELDS = {"run_id", "cell_id", "session_id", "ue_id", "slice_id", "ts_1s"}


def _config(**overrides) -> MockTelemetryConfig:
    base = dict(seed=42, num_sessions=6, emit_interval_seconds=0.01, missing_feature_rate=0.0)
    base.update(overrides)
    return MockTelemetryConfig(**base)


def test_records_are_stamped_mock_never_oai_replay():
    source = MockTelemetrySource(_config(), max_records=20, realtime=False)
    records = list(source.records())
    assert len(records) == 20
    assert all(r["source"] == "MOCK" for r in records)
    assert all(r["source"] != "OAI_RAN_REPLAY" for r in records)


def test_all_expected_raw_fields_present_when_no_injected_imperfection():
    source = MockTelemetrySource(_config(), max_records=12, realtime=False)
    for record in source.records():
        for field_name in TARGET_COLUMNS:
            assert field_name in record, f"missing {field_name}"
        for field_name in ALL_FEATURES:
            assert field_name in record, f"missing {field_name}"
        for field_name in IDENTITY_FIELDS:
            assert field_name in record, f"missing {field_name}"


def test_session_ids_distributed_across_configured_topology():
    config = _config(num_sessions=6)
    source = MockTelemetrySource(config, max_records=6, realtime=False)
    records = list(source.records())
    assert {r["session_id"] for r in records} == set(range(1, 7))
    assert {r["slice_id"] for r in records} <= {1, 5}


def test_reproducible_with_same_seed():
    source_a = MockTelemetrySource(_config(seed=7), max_records=10, realtime=False)
    source_b = MockTelemetrySource(_config(seed=7), max_records=10, realtime=False)
    records_a = list(source_a.records())
    records_b = list(source_b.records())
    assert [r["dl_throughput"] for r in records_a] == [r["dl_throughput"] for r in records_b]


def test_different_seeds_diverge():
    source_a = MockTelemetrySource(_config(seed=1), max_records=10, realtime=False)
    source_b = MockTelemetrySource(_config(seed=2), max_records=10, realtime=False)
    records_a = list(source_a.records())
    records_b = list(source_b.records())
    assert [r["dl_throughput"] for r in records_a] != [r["dl_throughput"] for r in records_b]


def test_max_records_bounds_generation():
    source = MockTelemetrySource(_config(), max_records=5, realtime=False)
    assert len(list(source.records())) == 5


def test_missing_feature_rate_actually_drops_fields():
    config = _config(missing_feature_rate=1.0)
    source = MockTelemetrySource(config, max_records=10, realtime=False)
    records = list(source.records())
    full_field_count = len(TARGET_COLUMNS) + len(ALL_FEATURES) + len(IDENTITY_FIELDS) + 1  # +source
    # With rate=1.0 every record has exactly one feature column removed.
    assert all(len(r) == full_field_count - 1 for r in records)


def test_quality_trajectory_varies_across_ticks():
    """The per-session radio-quality random walk genuinely evolves — a real, non-degenerate
    signal driving the correlated targets/features, not frozen noise."""
    config = _config(num_sessions=1, emit_interval_seconds=1.0)
    source = MockTelemetrySource(config, max_records=5, realtime=False)
    throughputs = [r["dl_throughput"] for r in source.records()]
    assert len(set(throughputs)) > 1
