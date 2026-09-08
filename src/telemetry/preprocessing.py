"""Telemetry preprocessing pipeline (Module 2, prompt.md §8).

Turns raw wire-format dicts (from either `zmq_source.py` or `mock_source.py`) into DT-ready
`CleanTelemetryRecord`s plus `FeatureWindow`s, covering every responsibility listed in §8:

- receive telemetry           -> `TelemetryPreprocessor.process_batch` consumes an iterable of raw dicts
- validate schema              -> `_validate_shape`
- normalize units               -> `RAW_TO_CANONICAL` conversions applied in `_normalize`
- handle missing values         -> per-(ue_id, cell_id) carry-forward imputation, else quarantine
- remove/flag invalid records   -> quarantine (removed from clean stream, kept for audit) vs.
                                    range-flag (kept, marked `out_of_range_fields`)
- synchronize timestamps        -> `_parse_timestamp` normalizes float-epoch or ISO-8601 to UTC
- construct feature windows     -> `build_feature_windows`
- preserve UE/cell identifiers  -> `ue_id`/`cell_id` are identity fields, never normalized/dropped
- maintain data quality metadata -> `RecordQuality` per record, `missing_ratio`/`quality_flag` per window

Nothing here is silently lossy: every raw record ends up in either `clean_records` or
`quarantined`, and the reason for quarantine is always recorded.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from pydantic import ValidationError

from src.common.config import PreprocessingConfig
from src.telemetry.schema import (
    ALLOWED_SOURCES,
    IDENTITY_FIELDS,
    OPTIONAL_PASSTHROUGH_FIELDS,
    PASSTHROUGH_FIELDS,
    RAW_TO_CANONICAL,
    CleanTelemetryRecord,
    FeatureWindow,
    PreprocessingResult,
    QuarantinedRecord,
    RecordQuality,
)

logger = logging.getLogger(__name__)


def _parse_timestamp(value: Any) -> datetime | None:
    """Synchronize heterogeneous timestamp representations to a canonical UTC datetime."""
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=UTC)
        if isinstance(value, str):
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize(UTC)
            else:
                ts = ts.tz_convert(UTC)
            return ts.to_pydatetime()
    except (ValueError, TypeError, OverflowError):
        return None
    return None


class TelemetryPreprocessor:
    """Stateful pipeline: keeps the last known-good value per (ue_id, cell_id, field) so a
    single missing required field can be carried forward rather than immediately discarding an
    otherwise-good record (prompt.md §8 — "do not silently discard important data")."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self._config = config
        self._required = set(config.required_fields)
        self._valid_ranges = config.valid_ranges
        self._last_good: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)

    def _validate_shape(self, raw: dict[str, Any]) -> str | None:
        """Structural/schema validation. Returns a quarantine reason, or None if OK."""
        if not isinstance(raw, dict):
            return "not_a_record"
        source = raw.get("source")
        if source not in ALLOWED_SOURCES:
            return f"invalid_source:{source!r}"
        for field_name in IDENTITY_FIELDS:
            if raw.get(field_name) in (None, ""):
                return f"missing_identity_field:{field_name}"
        return None

    def _normalize(self, raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Apply unit conversion + passthrough. Returns (canonical_partial, missing_fields).

        NaN/Inf inputs are treated as missing, not as valid measurements: they must flow through
        the same carry-forward-imputation-or-quarantine path as an absent field, and must never
        be written into `_last_good` (a cached NaN would otherwise silently propagate into every
        later record via imputation without a `valid_ranges` entry to catch it — prompt.md §0.26
        "NaN/Inf propagation").
        """
        canonical: dict[str, Any] = {}
        missing: list[str] = []

        for raw_field, (canonical_field, convert) in RAW_TO_CANONICAL.items():
            if raw_field in raw and raw[raw_field] is not None:
                try:
                    converted = convert(float(raw[raw_field]))
                except (TypeError, ValueError):
                    missing.append(canonical_field)
                    continue
                if not math.isfinite(converted):
                    missing.append(canonical_field)
                else:
                    canonical[canonical_field] = converted
            else:
                missing.append(canonical_field)

        for field_name in PASSTHROUGH_FIELDS:
            value = raw.get(field_name)
            if value is None:
                missing.append(field_name)
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                # Non-numeric junk (e.g. a malformed string) in a numeric field is untrusted
                # input, not a value — treat exactly like "missing" rather than letting it reach
                # range-checking (`lo <= value <= hi`) or the pydantic constructor as a raw str.
                missing.append(field_name)
                continue
            if not math.isfinite(numeric_value):
                missing.append(field_name)
                continue
            canonical[field_name] = numeric_value

        for field_name in OPTIONAL_PASSTHROUGH_FIELDS:
            value = raw.get(field_name)
            if value is None:
                missing.append(field_name)
                continue
            if isinstance(value, (int, float)) and not math.isfinite(value):
                missing.append(field_name)
                continue
            canonical[field_name] = value

        return canonical, missing

    def _impute_or_quarantine_reason(
        self, key: tuple[str, str], canonical: dict[str, Any], missing: list[str]
    ) -> tuple[list[str], str | None]:
        """Try carry-forward imputation for missing required fields. Returns
        (imputed_field_names, quarantine_reason_or_None)."""
        imputed: list[str] = []
        cache = self._last_good[key]
        still_missing_required: list[str] = []

        for field_name in list(missing):
            if field_name in self._required and field_name in cache:
                canonical[field_name] = cache[field_name]
                imputed.append(field_name)
            elif field_name in self._required:
                still_missing_required.append(field_name)
            # missing optional (non-required) fields stay missing — recorded in quality, no impute

        if still_missing_required:
            return imputed, f"missing_required_fields:{','.join(sorted(still_missing_required))}"
        return imputed, None

    def _flag_out_of_range(self, canonical: dict[str, Any]) -> list[str]:
        flagged: list[str] = []
        for field_name, bounds in self._valid_ranges.items():
            value = canonical.get(field_name)
            if value is None:
                continue
            lo, hi = bounds
            if not (lo <= value <= hi):
                flagged.append(field_name)
        return flagged

    def process_record(
        self, raw: dict[str, Any]
    ) -> tuple[CleanTelemetryRecord | None, QuarantinedRecord | None]:
        received_at = datetime.now(UTC)
        source = raw.get("source") if isinstance(raw, dict) else None

        shape_reason = self._validate_shape(raw)
        if shape_reason is not None:
            logger.warning(
                "telemetry record quarantined",
                extra={"component": "telemetry", "reason": shape_reason, "source": source},
            )
            return None, QuarantinedRecord(
                raw=raw if isinstance(raw, dict) else {"_unparseable": str(raw)},
                reason=shape_reason,
                source=source,
                received_at=received_at,
            )

        timestamp = _parse_timestamp(raw.get("timestamp"))
        if timestamp is None:
            return None, QuarantinedRecord(
                raw=raw, reason="unparseable_timestamp", source=source, received_at=received_at
            )

        ue_id = str(raw["ue_id"])
        cell_id = str(raw["cell_id"])
        key = (ue_id, cell_id)

        canonical, missing = self._normalize(raw)
        imputed, quarantine_reason = self._impute_or_quarantine_reason(key, canonical, missing)
        if quarantine_reason is not None:
            logger.warning(
                "telemetry record quarantined",
                extra={
                    "component": "telemetry",
                    "reason": quarantine_reason,
                    "source": source,
                    "ue_id": ue_id,
                    "cell_id": cell_id,
                },
            )
            return None, QuarantinedRecord(
                raw=raw, reason=quarantine_reason, source=source, received_at=received_at
            )

        still_missing_optional = [f for f in missing if f not in imputed]
        out_of_range = self._flag_out_of_range(canonical)
        if out_of_range:
            logger.info(
                "telemetry record has out-of-range fields (kept, flagged)",
                extra={
                    "component": "telemetry",
                    "fields": out_of_range,
                    "ue_id": ue_id,
                    "cell_id": cell_id,
                },
            )

        # Update carry-forward cache with the final resolved values (only required fields need
        # to survive for imputation; caching everything numeric is harmless and simpler).
        for field_name, value in canonical.items():
            if isinstance(value, (int, float)) and value is not None:
                self._last_good[key][field_name] = value

        quality = RecordQuality(
            missing_fields=sorted(still_missing_optional),
            imputed_fields=sorted(imputed),
            out_of_range_fields=sorted(out_of_range),
        )

        try:
            clean = CleanTelemetryRecord(
                timestamp=timestamp,
                ue_id=ue_id,
                cell_id=cell_id,
                source=source,
                quality=quality,
                **canonical,
            )
        except ValidationError as exc:
            # Defense in depth: any field value that made it past the checks above but is still
            # rejected by the canonical schema (unexpected type, etc.) must quarantine the record
            # rather than crash the batch (prompt.md §44 — telemetry is untrusted input; §61 —
            # fail safely).
            logger.warning(
                "telemetry record quarantined: canonical schema rejected it",
                extra={"component": "telemetry", "error": str(exc), "ue_id": ue_id, "cell_id": cell_id},
            )
            return None, QuarantinedRecord(
                raw=raw, reason=f"schema_validation_error:{exc}", source=source, received_at=received_at
            )
        return clean, None

    def process_batch(self, raw_records: Iterable[dict[str, Any]]) -> PreprocessingResult:
        result = PreprocessingResult()
        for raw in raw_records:
            clean, quarantined = self.process_record(raw)
            if clean is not None:
                result.clean_records.append(clean)
            if quarantined is not None:
                result.quarantined.append(quarantined)
        logger.info(
            "telemetry batch processed",
            extra={
                "component": "telemetry",
                "total_received": result.total_received,
                "clean": len(result.clean_records),
                "quarantined": len(result.quarantined),
            },
        )
        return result

    def build_feature_windows(
        self, clean_records: list[CleanTelemetryRecord]
    ) -> list[FeatureWindow]:
        """Group by (ue_id, cell_id), sort by timestamp, slice into tumbling windows of
        `feature_window_size`. A trailing partial window is still emitted (smaller `size`) so no
        data is dropped — callers decide whether a partial window is usable."""
        window_size = self._config.feature_window_size
        max_missing_ratio = self._config.max_missing_ratio

        by_key: dict[tuple[str, str], list[CleanTelemetryRecord]] = defaultdict(list)
        for record in clean_records:
            by_key[(record.ue_id, record.cell_id)].append(record)

        windows: list[FeatureWindow] = []
        for (ue_id, cell_id), records in by_key.items():
            records = sorted(records, key=lambda r: r.timestamp)
            for start in range(0, len(records), window_size):
                chunk = records[start : start + window_size]
                if not chunk:
                    continue
                frame = pd.DataFrame(
                    [
                        {
                            "timestamp": r.timestamp,
                            "ue_id": r.ue_id,
                            "cell_id": r.cell_id,
                            "throughput_mbps": r.throughput_mbps,
                            "offered_load_mbps": r.offered_load_mbps,
                            "latency_ms": r.latency_ms,
                            "jitter_ms": r.jitter_ms,
                            "packet_loss_pct": r.packet_loss_pct,
                            "prb_utilization_pct": r.prb_utilization_pct,
                            "sinr_db": r.sinr_db,
                            "rsrp_dbm": r.rsrp_dbm,
                            "rsrq_db": r.rsrq_db,
                            "ue_count": r.ue_count,
                            "ue_speed_mps": r.ue_speed_mps,
                            "ue_position_x": r.ue_position_x,
                            "ue_position_y": r.ue_position_y,
                            "source": r.source,
                        }
                        for r in chunk
                    ]
                )
                imperfect_rows = sum(1 for r in chunk if not r.quality.is_perfect)
                missing_ratio = imperfect_rows / len(chunk)
                windows.append(
                    FeatureWindow(
                        ue_id=ue_id,
                        cell_id=cell_id,
                        start_timestamp=chunk[0].timestamp,
                        end_timestamp=chunk[-1].timestamp,
                        size=len(chunk),
                        frame=frame,
                        missing_ratio=missing_ratio,
                        quality_flag="low_quality" if missing_ratio > max_missing_ratio else "ok",
                    )
                )
        logger.info(
            "feature windows constructed",
            extra={"component": "telemetry", "num_windows": len(windows), "window_size": window_size},
        )
        return windows
