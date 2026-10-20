"""Telemetry preprocessing pipeline (Module 2). Turns raw wire-format dicts (one row of the OAI
replay/mock source's already unit-converted, already-lagged output — see
`src/data/oai_dataset.py`) into DT-ready `CleanTelemetryRecord`s.

Job here is just validation, the identity/feature/target split, and quality tracking — unit
conversion/lagging already happened upstream. A NaN feature/target is legitimate here (a
session's first tick has no t-1; a sensor gap), never fabricated, never a reason to quarantine
the record. Only a structurally broken record (bad identity, bad timestamp, wrong source label)
is quarantined.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from src.common.config import OaiPreprocessingConfig
from src.data.oai_dataset import ALL_FEATURES, OBSERVABLE_KPI_COLUMNS
from src.telemetry.schema import (
    ALLOWED_SOURCES,
    TARGET_COLUMNS,
    CleanTelemetryRecord,
    PreprocessingResult,
    QuarantinedRecord,
    RecordQuality,
)

logger = logging.getLogger(__name__)


def _parse_timestamp(value: Any) -> datetime | None:
    """Synchronize heterogeneous timestamp representations (already-UTC `pd.Timestamp`, ISO-8601
    string, or epoch float) to a canonical UTC `datetime`."""
    try:
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=UTC)
        if isinstance(value, str):
            import pandas as pd

            ts = pd.Timestamp(value)
            ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
            return ts.to_pydatetime()
    except (ValueError, TypeError, OverflowError):
        return None
    return None


def _as_finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


class TelemetryPreprocessor:
    """Validates raw OAI (or mock) telemetry dicts into `CleanTelemetryRecord`s. Stateless across
    records by design — unlike the retired NS-3 preprocessor, there is no carry-forward cache:
    a missing feature/target here is legitimate missingness Step 3 already determined, not a
    transient live-stream dropout to paper over."""

    def __init__(self, config: OaiPreprocessingConfig) -> None:
        self._config = config

    def _validate_shape(self, raw: dict[str, Any]) -> str | None:
        if not isinstance(raw, dict):
            return "not_a_record"
        source = raw.get("source")
        if source not in ALLOWED_SOURCES:
            return f"invalid_source:{source!r}"
        for field_name in ("run_id", "cell_id", "session_id", "ue_id", "slice_id"):
            if raw.get(field_name) in (None, ""):
                return f"missing_identity_field:{field_name}"
        if raw.get("ts_1s") in (None, ""):
            return "missing_identity_field:ts_1s"
        return None

    def process_record(self, raw: dict[str, Any]) -> tuple[CleanTelemetryRecord | None, QuarantinedRecord | None]:
        received_at = datetime.now(UTC)
        source = raw.get("source") if isinstance(raw, dict) else None

        shape_reason = self._validate_shape(raw)
        if shape_reason is not None:
            logger.warning(
                "telemetry record quarantined", extra={"component": "telemetry", "reason": shape_reason, "source": source}
            )
            return None, QuarantinedRecord(
                raw=raw if isinstance(raw, dict) else {"_unparseable": str(raw)},
                reason=shape_reason, source=source, received_at=received_at,
            )

        ts_1s = _parse_timestamp(raw.get("ts_1s"))
        if ts_1s is None:
            return None, QuarantinedRecord(raw=raw, reason="unparseable_timestamp", source=source, received_at=received_at)

        try:
            run_id, cell_id, session_id, ue_id, slice_id = (
                int(raw["run_id"]), int(raw["cell_id"]), int(raw["session_id"]), int(raw["ue_id"]), int(raw["slice_id"]),
            )
        except (TypeError, ValueError):
            return None, QuarantinedRecord(raw=raw, reason="non_integer_identity_field", source=source, received_at=received_at)

        targets: dict[str, float | None] = {}
        missing: list[str] = []
        for name in TARGET_COLUMNS:
            value = _as_finite_float_or_none(raw.get(name))
            targets[name] = value
            if value is None:
                missing.append(name)

        # Checked against the canonical feature list (`ALL_FEATURES`), exactly like targets are
        # checked against `TARGET_COLUMNS` — a KEY ABSENT from `raw` entirely (not just `None`)
        # must still show up as a tracked-missing feature, never silently vanish from both
        # `features` and `quality.missing_fields` the way iterating only over `raw.items()`
        # would (a real gap found and fixed while testing: a session's first tick, which
        # genuinely lacks every `_lag1` key, produced a `features` dict simply missing those
        # keys rather than recording them as `None`/missing).
        features: dict[str, float | None] = {}
        for name in ALL_FEATURES:
            numeric = _as_finite_float_or_none(raw.get(name))
            features[name] = numeric
            if numeric is None:
                missing.append(name)

        quality = RecordQuality(missing_fields=sorted(missing), imputed_fields=[], out_of_range_fields=[])
        observables = {name: _as_finite_float_or_none(raw.get(name)) for name in OBSERVABLE_KPI_COLUMNS}

        try:
            clean = CleanTelemetryRecord(
                run_id=run_id, cell_id=cell_id, session_id=session_id, ue_id=ue_id, slice_id=slice_id,
                ts_1s=ts_1s, features=features, targets=targets, source=source, quality=quality,
                observables=observables,
            )
        except ValidationError as exc:
            logger.warning(
                "telemetry record quarantined: canonical schema rejected it",
                extra={"component": "telemetry", "error": str(exc), "session_id": session_id},
            )
            return None, QuarantinedRecord(raw=raw, reason=f"schema_validation_error:{exc}", source=source, received_at=received_at)
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
            extra={"component": "telemetry", "total_received": result.total_received,
                   "clean": len(result.clean_records), "quarantined": len(result.quarantined)},
        )
        return result
