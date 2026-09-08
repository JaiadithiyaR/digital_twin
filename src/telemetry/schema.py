"""Telemetry schema (Module 2, prompt.md §7-§8).

Defines the contract between telemetry sources (Module 1's real NS-3/5G-LENA path and the mock
development source) and the preprocessing pipeline, plus the canonical DT-ready output types.

Two representations exist on purpose:

- **Raw wire fields** — what a source emits. Several are deliberately in "natural exporter"
  units (bits/second, seconds, 0-1 ratios), matching what an ns-3 FlowMonitor-style exporter
  produces, so unit normalization (prompt.md §8) is a real, testable transformation rather than
  a no-op passthrough.
- **Canonical fields** — SI/percentage units used everywhere downstream (DT models, fidelity,
  config `valid_ranges`), e.g. Mbps/ms/percent.

`RAW_TO_CANONICAL` and `PASSTHROUGH_FIELDS` together define every numeric field this pipeline
understands; `IDENTITY_FIELDS` are carried through untouched (never normalized, never dropped
for being "out of range").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

# Sources this pipeline will accept. Anything else is quarantined (prompt.md §0.3 — a record
# must never be allowed to claim a source it didn't actually come through).
ALLOWED_SOURCES = ("NS3_5G_LENA", "MOCK")

IDENTITY_FIELDS = ("timestamp", "ue_id", "cell_id")

# raw_field_name -> (canonical_field_name, conversion_fn)
RAW_TO_CANONICAL: dict[str, tuple[str, Callable[[float], float]]] = {
    "throughput_bps": ("throughput_mbps", lambda v: v / 1.0e6),
    "offered_load_bps": ("offered_load_mbps", lambda v: v / 1.0e6),
    "latency_s": ("latency_ms", lambda v: v * 1000.0),
    "jitter_s": ("jitter_ms", lambda v: v * 1000.0),
    "packet_loss_ratio": ("packet_loss_pct", lambda v: v * 100.0),
    "prb_utilization_ratio": ("prb_utilization_pct", lambda v: v * 100.0),
}

# Fields already in canonical units at the source — copied through unchanged.
PASSTHROUGH_FIELDS = (
    "sinr_db",
    "rsrp_dbm",
    "rsrq_db",
    "ue_count",
    "ue_speed_mps",
)

# Present "where available" (prompt.md §7) — never required, but preserved and quality-tracked
# when present.
OPTIONAL_PASSTHROUGH_FIELDS = ("ue_position_x", "ue_position_y")


class RecordQuality(BaseModel):
    """Per-record data quality metadata (prompt.md §8 — "maintain data quality metadata")."""

    missing_fields: list[str] = []
    imputed_fields: list[str] = []
    out_of_range_fields: list[str] = []

    @property
    def is_perfect(self) -> bool:
        return not (self.missing_fields or self.imputed_fields or self.out_of_range_fields)


class CleanTelemetryRecord(BaseModel):
    """A single validated, unit-normalized, DT-ready telemetry record."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    ue_id: str
    cell_id: str
    throughput_mbps: float
    offered_load_mbps: float
    latency_ms: float
    jitter_ms: float
    packet_loss_pct: float
    prb_utilization_pct: float
    sinr_db: float
    rsrp_dbm: float
    rsrq_db: float
    ue_count: int
    ue_speed_mps: float
    ue_position_x: float | None = None
    ue_position_y: float | None = None
    source: Literal["NS3_5G_LENA", "MOCK"]
    quality: RecordQuality


class QuarantinedRecord(BaseModel):
    """A raw record that failed validation and was excluded from the clean/windowed output.

    Quarantining (not deletion) is the "do not silently discard important data" mechanism
    required by prompt.md §8 — the record and the reason it failed are retained for audit.
    """

    raw: dict[str, Any]
    reason: str
    source: str | None
    received_at: datetime


@dataclass
class FeatureWindow:
    """A fixed-size, time-ordered slice of clean records for one (ue_id, cell_id) key.

    This is the "time-series windows (timestamp, UE ID, cell ID)" artifact named on the
    Module 2 -> Module 3 edge of the data-flow diagram.
    """

    ue_id: str
    cell_id: str
    start_timestamp: datetime
    end_timestamp: datetime
    size: int
    frame: pd.DataFrame
    missing_ratio: float
    quality_flag: Literal["ok", "low_quality"]


@dataclass
class PreprocessingResult:
    """Output of processing one batch of raw records — both required output shapes at once:
    flat validated feature records, and (via `windows`, built separately) time-series windows.
    """

    clean_records: list[CleanTelemetryRecord] = field(default_factory=list)
    quarantined: list[QuarantinedRecord] = field(default_factory=list)

    @property
    def total_received(self) -> int:
        return len(self.clean_records) + len(self.quarantined)
