"""Telemetry schema (Module 2) — OAI migration (prompt2.md).

Replaces the NS-3-era fixed 14-field schema entirely. An OAI telemetry record is one row of
`src/data/oai_dataset.py`'s already-unit-converted, already-lagged processed table (one
`(run_id, session_id, ts_1s)` UE-level row): fixed identity fields, a dynamic dict of lagged
INPUT features (keys come from `feature_schema.json`, not hardcoded here — the feature set is a
data-pipeline concern, not a schema concern), and the 7 raw TARGET ground-truth values for this
exact second. Two record shapes exist on purpose, mirroring the original design's "raw vs
canonical" split:

- The replay source's wire-format dict IS already canonical (Step 3 did all unit conversion) —
  there is no separate raw-units transformation step anymore.
- `CleanOaiRecord` is the validated, DT-ready output every downstream module (D1, the orchestrator,
  fidelity) consumes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from src.data.oai_dataset import TARGET_COLUMNS

# Sources this pipeline will accept. Anything else is quarantined (prompt.md §0.3 — a record must
# never be allowed to claim a source it didn't actually come through).
ALLOWED_SOURCES = ("OAI_RAN_REPLAY", "MOCK")

IDENTITY_FIELDS = ("run_id", "cell_id", "session_id", "ue_id", "slice_id", "ts_1s")


class RecordQuality(BaseModel):
    """Per-record data quality metadata (prompt.md §8 — "maintain data quality metadata")."""

    missing_fields: list[str] = []
    imputed_fields: list[str] = []
    out_of_range_fields: list[str] = []

    @property
    def is_perfect(self) -> bool:
        return not (self.missing_fields or self.imputed_fields or self.out_of_range_fields)


class CleanTelemetryRecord(BaseModel):
    """A single validated OAI telemetry record — identity + lagged input features (model INPUT,
    all `_lag1`/static, per the leakage rule) + the 7 raw targets (ground truth FOR this second,
    what the fidelity engine compares predictions against)."""

    model_config = ConfigDict(frozen=True)

    run_id: int
    cell_id: int
    session_id: int
    ue_id: int
    slice_id: int
    ts_1s: datetime
    features: dict[str, float | None]
    targets: dict[str, float | None]
    source: Literal["OAI_RAN_REPLAY", "MOCK"]
    quality: RecordQuality
    # Same-second MAC KPIs (`oai_dataset.OBSERVABLE_KPI_COLUMNS`) — ground truth an expand-scope
    # component may predict, never a model input. Optional: absent values are not data-quality issues.
    observables: dict[str, float | None] = {}

    def target(self, name: str) -> float | None:
        return self.targets.get(name)

    def feature(self, name: str) -> float | None:
        return self.features.get(name)


class QuarantinedRecord(BaseModel):
    """A raw record that failed validation and was excluded from the clean output.

    Quarantining (not deletion) is the "do not silently discard important data" mechanism
    required by prompt.md §8 — the record and the reason it failed are retained for audit.
    """

    raw: dict[str, Any]
    reason: str
    source: str | None
    received_at: datetime


class PreprocessingResult(BaseModel):
    """Output of processing one batch of raw replay/mock records."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    clean_records: list[CleanTelemetryRecord] = []
    quarantined: list[QuarantinedRecord] = []

    @property
    def total_received(self) -> int:
        return len(self.clean_records) + len(self.quarantined)


__all__ = [
    "ALLOWED_SOURCES",
    "IDENTITY_FIELDS",
    "TARGET_COLUMNS",
    "RecordQuality",
    "CleanTelemetryRecord",
    "QuarantinedRecord",
    "PreprocessingResult",
]
