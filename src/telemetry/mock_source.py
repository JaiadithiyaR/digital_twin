"""Mock telemetry source — development/testing only (prompt.md §0.3, §48, §49).

Generates telemetry matching the exact raw wire schema a real NS-3/5G-LENA exporter would emit
(same field names, same units), with plausible physical correlations (better SINR -> higher
throughput/lower loss, higher offered load -> higher latency, etc.) so it exercises the same
preprocessing/DT/fidelity pipeline a live run would (prompt.md §48 — demo mode must not be a
separate fake architecture). Every record is unconditionally stamped `source="MOCK"` — this
class must NEVER be wired to anything that reports itself as `NS3_5G_LENA`.

Deliberately injects a small, configurable rate of missing fields and out-of-range values so the
preprocessing pipeline's imputation/flagging logic is exercised even without a live network.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import numpy as np

from src.common.config import MockTelemetryConfig
from src.telemetry.base import TelemetrySource


class MockTelemetrySource(TelemetrySource):
    SOURCE_LABEL = "MOCK"

    def __init__(
        self,
        config: MockTelemetryConfig,
        max_records: int | None = None,
        realtime: bool = True,
    ) -> None:
        """
        Args:
            config: mock.* section of settings.yaml.
            max_records: stop after this many records; None = stream forever.
            realtime: sleep `emit_interval_seconds` between ticks (True for live demo mode);
                set False for fast, sleep-free generation (bootstrap data, tests).
        """
        self._config = config
        self._max_records = max_records
        self._realtime = realtime
        self._rng = np.random.default_rng(config.seed)

        self._ue_ids = [f"ue-{i}" for i in range(config.num_ues)]
        self._cell_ids = [f"cell-{c}" for c in range(config.num_cells)]
        # Fixed round-robin UE -> cell assignment (handover modeling is out of scope here).
        self._ue_cell = {
            ue_id: self._cell_ids[i % config.num_cells] for i, ue_id in enumerate(self._ue_ids)
        }
        self._cell_ue_count = {
            cell_id: sum(1 for c in self._ue_cell.values() if c == cell_id)
            for cell_id in self._cell_ids
        }

        # Per-UE persistent mobility state for smooth trajectories across ticks.
        self._position: dict[str, tuple[float, float]] = {
            ue_id: (self._rng.uniform(-500, 500), self._rng.uniform(-500, 500))
            for ue_id in self._ue_ids
        }
        self._heading: dict[str, float] = {
            ue_id: self._rng.uniform(0, 2 * math.pi) for ue_id in self._ue_ids
        }
        self._speed: dict[str, float] = {
            ue_id: self._rng.uniform(0.5, 25.0) for ue_id in self._ue_ids
        }
        self._closed = False

    def _next_position(self, ue_id: str, dt: float) -> tuple[float, float]:
        x, y = self._position[ue_id]
        heading = self._heading[ue_id] + self._rng.normal(0, 0.15)  # gentle random walk
        speed = self._speed[ue_id]
        x += speed * math.cos(heading) * dt
        y += speed * math.sin(heading) * dt
        self._heading[ue_id] = heading
        self._position[ue_id] = (x, y)
        return x, y

    def _generate_record(self, ue_id: str, tick_timestamp: float) -> dict[str, Any]:
        cell_id = self._ue_cell[ue_id]
        rng = self._rng

        sinr_db = float(np.clip(rng.normal(15.0, 8.0), -10.0, 35.0))
        rsrp_dbm = float(np.clip(rng.normal(-95.0, 15.0), -140.0, -44.0))
        rsrq_db = float(np.clip(rng.normal(-10.0, 4.0), -20.0, -3.0))

        # Channel quality (0-1) drives capacity, loss, and latency correlations below.
        channel_quality = float(np.clip((sinr_db + 10.0) / 45.0, 0.0, 1.0))

        offered_load_bps = float(max(0.0, rng.normal(30e6, 15e6)))
        # PRB utilization is driven by cell-wide demand — more offered traffic and more UEs
        # sharing the cell's fixed pool of resource blocks push utilization up. Previously drawn
        # as pure independent noise (Beta(2,3), uncorrelated with any demand signal) even though
        # it was already used to influence throughput/loss/latency below — that made it
        # structurally unlearnable from offered_load/UE-count/throughput (Module 9's actual
        # inputs), discovered when PrbUtilizationModel's held-out RMSE failed to beat the naive
        # mean baseline. Grounding it in demand is the physically correct fix, not curve-fitting.
        demand_pressure = (offered_load_bps / 40e6) + 0.02 * self._cell_ue_count[cell_id]
        prb_utilization_ratio = float(np.clip(0.15 + 0.5 * demand_pressure + rng.normal(0, 0.08), 0.0, 1.0))
        capacity_bps = offered_load_bps * (0.4 + 0.6 * channel_quality) * (1.1 - 0.3 * prb_utilization_ratio)
        throughput_bps = float(np.clip(min(offered_load_bps, capacity_bps) + rng.normal(0, 1e6), 0.0, None))

        packet_loss_ratio = float(
            np.clip(0.05 * (1.0 - channel_quality) + 0.03 * prb_utilization_ratio + rng.normal(0, 0.003), 0.0, 1.0)
        )
        latency_s = float(
            max(0.001, 0.005 + 0.02 * prb_utilization_ratio + 0.01 * (1.0 - channel_quality) + rng.normal(0, 0.002))
        )
        jitter_s = float(max(0.0001, latency_s * rng.uniform(0.05, 0.25)))

        ue_speed_mps = self._speed[ue_id]
        x, y = self._next_position(ue_id, self._config.emit_interval_seconds)

        record: dict[str, Any] = {
            "source": self.SOURCE_LABEL,
            "timestamp": tick_timestamp,
            "ue_id": ue_id,
            "cell_id": cell_id,
            "throughput_bps": throughput_bps,
            "offered_load_bps": offered_load_bps,
            "latency_s": latency_s,
            "jitter_s": jitter_s,
            "packet_loss_ratio": packet_loss_ratio,
            "prb_utilization_ratio": prb_utilization_ratio,
            "sinr_db": sinr_db,
            "rsrp_dbm": rsrp_dbm,
            "rsrq_db": rsrq_db,
            "ue_count": self._cell_ue_count[cell_id],
            "ue_speed_mps": ue_speed_mps,
            "ue_position_x": x,
            "ue_position_y": y,
        }

        # Inject realistic imperfections so the preprocessing pipeline's missing-value and
        # out-of-range handling paths are genuinely exercised in demo mode (prompt.md §48).
        non_identity_fields = [k for k in record if k not in ("source", "timestamp", "ue_id", "cell_id")]
        if rng.random() < self._config.missing_field_rate:
            del record[rng.choice(non_identity_fields)]
        if rng.random() < self._config.out_of_range_rate:
            numeric_fields = [
                k
                for k in record
                if k in non_identity_fields and isinstance(record.get(k), (int, float))
            ]
            if numeric_fields:
                field_name = rng.choice(numeric_fields)
                record[field_name] = float(record[field_name]) * rng.choice([-1.0, 50.0])

        return record

    def records(self) -> Iterator[dict[str, Any]]:
        """Yields until `max_records` is reached, or indefinitely (`max_records=None`) until
        `close()` is called — matching `ZmqTelemetrySource`'s stop semantics so a consumer (e.g.
        Module 3's `ContinuousSynchronizer`) can drive either source identically."""
        emitted = 0
        while not self._closed and (self._max_records is None or emitted < self._max_records):
            tick_timestamp = datetime.now(UTC).timestamp()
            for ue_id in self._ue_ids:
                if self._closed or (self._max_records is not None and emitted >= self._max_records):
                    return
                yield self._generate_record(ue_id, tick_timestamp)
                emitted += 1
            if self._realtime:
                time.sleep(self._config.emit_interval_seconds)

    def close(self) -> None:
        self._closed = True
