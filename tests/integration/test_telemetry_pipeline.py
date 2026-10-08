"""Integration tests: full Module 2 pipeline wired together using real `config/settings.yaml`
values — MockTelemetrySource / OaiReplaySource -> TelemetryPreprocessor.

Proves the diagram edge end-to-end: "raw telemetry" in, validated `CleanTelemetryRecord`s
(identity + lagged features + the 7 raw targets) out (prompt.md §8, fig-dataflow.png Module 2 ->
D1 edge).
"""

from __future__ import annotations

from src.common.config import load_settings
from src.data.oai_dataset import TARGET_COLUMNS
from src.telemetry.mock_source import MockTelemetrySource
from src.telemetry.oai_replay_source import OaiReplaySource
from src.telemetry.preprocessing import TelemetryPreprocessor


def test_mock_source_through_full_pipeline_using_real_config():
    settings = load_settings()
    source = MockTelemetrySource(settings.telemetry.mock, max_records=200, realtime=False)
    preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)

    result = preprocessor.process_batch(source.records())

    assert result.total_received == 200
    # Some imperfection is injected by config (missing_feature_rate > 0); the pipeline must still
    # turn the overwhelming majority into clean records, not discard them (a missing feature is
    # legitimate missingness, never a reason to quarantine the whole record).
    assert len(result.clean_records) > 150
    assert all(r.source == "MOCK" for r in result.clean_records)
    assert all(r.source != "OAI_RAN_REPLAY" for r in result.clean_records)

    for record in result.clean_records:
        for name in TARGET_COLUMNS:
            assert name in record.targets
        assert record.targets["dl_throughput"] is None or record.targets["dl_throughput"] >= 0


def test_real_oai_replay_through_full_pipeline_real_transport(tmp_path):
    """Drives the REAL processed OAI export through `OaiReplaySource` -> the exact same
    preprocessor used for mock data — proving the real-replay source path produces the same
    validated output shape, with the correct `SOURCE=OAI_RAN_REPLAY` provenance."""
    import pytest

    settings = load_settings()
    processed_dir = settings.resolve_path("data/processed/oai")
    if not (processed_dir / "test.parquet").exists():
        pytest.skip("data/processed/oai/test.parquet not built — run scripts/build_oai_dataset.py")

    source = OaiReplaySource(processed_dir, speed=0.0)
    source.start(run_ids=settings.data.splits.test)

    num_messages = 30
    received = []
    gen = source.records()
    for _ in range(num_messages):
        received.append(next(gen))
    source.close()

    assert len(received) == num_messages
    assert all(r["source"] == "OAI_RAN_REPLAY" for r in received)
    assert all(r["run_id"] in settings.data.splits.test for r in received)

    preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)
    result = preprocessor.process_batch(received)
    assert len(result.clean_records) > 0
    assert all(r.source == "OAI_RAN_REPLAY" for r in result.clean_records)
    for record in result.clean_records:
        for name in TARGET_COLUMNS:
            assert name in record.targets
