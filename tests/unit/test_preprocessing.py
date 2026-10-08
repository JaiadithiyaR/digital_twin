"""Unit tests for TelemetryPreprocessor (Module 2 core) — OAI migration (prompt2.md).

Proves: raw OAI replay/mock dicts go in, validated `CleanTelemetryRecord`s (identity + features
dict + targets dict) come out — identity/source validation, timestamp parsing, and honest
missing-value tracking (a NaN/absent feature or target is legitimate missingness, never
fabricated, never a reason to quarantine the whole record — only a structurally broken record is
quarantined). See `src/telemetry/preprocessing.py`'s own docstring for why this is considerably
simpler than the retired NS-3-era preprocessor (no carry-forward cache, no unit conversion, no
out-of-range flagging — Step 3's `oai_dataset.py` already did all of that).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.common.config import OaiPreprocessingConfig
from src.data.oai_dataset import TARGET_COLUMNS
from src.telemetry.preprocessing import TelemetryPreprocessor


def _config(**overrides) -> OaiPreprocessingConfig:
    base = dict(max_missing_ratio=0.2)
    base.update(overrides)
    return OaiPreprocessingConfig(**base)


def _raw_record(**overrides) -> dict:
    base = dict(
        source="MOCK",
        run_id=1, cell_id=1, session_id=1, ue_id=1, slice_id=1,
        ts_1s=1_700_000_000.0,
        dl_throughput=25.0, ul_throughput=5.0, dl_rlc_delay=2.0,
        dl_prb=30.0, ul_prb=5.0, dl_bler=0.05, ul_bler=0.02,
        pusch_snr_avg_lag1=15.0, wb_cqi_avg_lag1=10.0,
    )
    base.update(overrides)
    return base


# --- happy path --------------------------------------------------------------------------


def test_valid_record_splits_into_identity_features_and_targets():
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(_raw_record())

    assert quarantined is None
    assert clean is not None
    assert clean.run_id == 1
    assert clean.cell_id == 1
    assert clean.session_id == 1
    assert clean.ue_id == 1
    assert clean.slice_id == 1
    assert clean.source == "MOCK"
    for name in TARGET_COLUMNS:
        assert clean.targets[name] is not None
    assert clean.features["pusch_snr_avg_lag1"] == pytest.approx(15.0)


def test_quality_is_perfect_when_every_feature_and_target_is_present():
    from src.data.oai_dataset import ALL_FEATURES

    raw = _raw_record(**{name: 1.0 for name in ALL_FEATURES})
    pre = TelemetryPreprocessor(_config())
    clean, _ = pre.process_record(raw)
    assert clean.quality.is_perfect


def test_timestamp_synchronized_to_utc_datetime_from_float_epoch():
    pre = TelemetryPreprocessor(_config())
    clean, _ = pre.process_record(_raw_record(ts_1s=1_700_000_000.0))
    assert isinstance(clean.ts_1s, datetime)
    assert clean.ts_1s.tzinfo is not None
    assert clean.ts_1s == datetime.fromtimestamp(1_700_000_000.0, tz=UTC)


def test_timestamp_synchronized_from_iso_string():
    pre = TelemetryPreprocessor(_config())
    clean, _ = pre.process_record(_raw_record(ts_1s="2023-11-14T22:13:20+00:00"))
    assert clean is not None
    assert clean.ts_1s.year == 2023


# --- schema / source validation -----------------------------------------------------------


def test_unknown_source_is_quarantined_not_silently_dropped():
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(_raw_record(source="SOME_OTHER_SOURCE"))
    assert clean is None
    assert quarantined is not None
    assert "invalid_source" in quarantined.reason
    assert quarantined.raw["session_id"] == 1  # raw payload retained for audit, not discarded


def test_missing_identity_field_is_quarantined():
    raw = _raw_record()
    del raw["cell_id"]
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(raw)
    assert clean is None
    assert "missing_identity_field" in quarantined.reason


def test_non_integer_identity_field_is_quarantined():
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(_raw_record(session_id="not-an-int"))
    assert clean is None
    assert quarantined.reason == "non_integer_identity_field"


def test_unparseable_timestamp_is_quarantined():
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(_raw_record(ts_1s="not-a-timestamp"))
    assert clean is None
    assert quarantined.reason == "unparseable_timestamp"


# --- honest missing-value tracking — never fabricated, never quarantined for this alone ---


def test_missing_target_is_tracked_as_missing_not_quarantined():
    """A target with no ground truth this tick (e.g. a counter reset) is legitimate — never
    quarantine the whole record over it, and never fabricate a value."""
    raw = _raw_record()
    del raw["dl_bler"]
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(raw)
    assert quarantined is None
    assert clean is not None
    assert clean.targets["dl_bler"] is None
    assert "dl_bler" in clean.quality.missing_fields


def test_missing_feature_is_tracked_as_missing_not_quarantined():
    """A missing lagged feature (e.g. a session's very first tick) is expected, not an error."""
    raw = _raw_record()
    del raw["pusch_snr_avg_lag1"]
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(raw)
    assert quarantined is None
    assert clean is not None
    assert clean.features["pusch_snr_avg_lag1"] is None
    assert "pusch_snr_avg_lag1" in clean.quality.missing_fields


def test_nan_target_treated_as_missing_not_crashed_on():
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(_raw_record(dl_throughput=float("nan")))
    assert quarantined is None
    assert clean is not None
    assert clean.targets["dl_throughput"] is None


def test_inf_feature_treated_as_missing_not_crashed_on():
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(_raw_record(pusch_snr_avg_lag1=float("inf")))
    assert quarantined is None
    assert clean is not None
    assert clean.features["pusch_snr_avg_lag1"] is None


def test_non_numeric_feature_treated_as_missing_not_a_crash():
    pre = TelemetryPreprocessor(_config())
    clean, quarantined = pre.process_record(_raw_record(pusch_snr_avg_lag1="not-a-number"))
    assert quarantined is None
    assert clean is not None
    assert clean.features["pusch_snr_avg_lag1"] is None


# --- batch processing ----------------------------------------------------------------------


def test_process_batch_never_loses_a_record():
    pre = TelemetryPreprocessor(_config())
    raw_records = [
        _raw_record(session_id=1),
        _raw_record(session_id=2, source="INVALID"),
        _raw_record(session_id=3, ts_1s="garbage"),
    ]
    result = pre.process_batch(raw_records)
    assert result.total_received == 3
    assert len(result.clean_records) == 1
    assert len(result.quarantined) == 2


def test_mock_source_never_produces_oai_replay_labeled_clean_records():
    pre = TelemetryPreprocessor(_config())
    result = pre.process_batch([_raw_record(source="MOCK") for _ in range(5)])
    assert all(r.source == "MOCK" for r in result.clean_records)
    assert not any(r.source == "OAI_RAN_REPLAY" for r in result.clean_records)
