"""Mock telemetry source — unit tests only (prompt2.md: "The mock source stays for unit tests
only, tagged SOURCE=MOCK"). Never wired into `--mode live`/replay runs.

Generates OAI-schema-shaped records (the same identity/feature/target field names
`src/data/oai_dataset.py`'s real processed tables produce) with plausible physical correlations
(better radio quality -> higher throughput/lower BLER/lower delay, higher load -> more PRBs used)
so unit tests exercise the same preprocessing/DT/fidelity pipeline a real replay run would
(prompt.md §48's "demo mode must not be a separate fake architecture" principle, carried over).
Every record is unconditionally stamped `source="MOCK"` — this class must NEVER be wired to
anything that reports itself as `OAI_RAN_REPLAY`.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from src.common.config import MockTelemetryConfig
from src.data.oai_dataset import LAGS, MAC_FEATURE_BASES, ROLLING_WINDOWS_SECONDS, TARGET_COLUMNS
from src.telemetry.base import TelemetrySource

_MOCK_RUN_ID = 0
_MOCK_CELL_ID = 1
_MOCK_SLICE_IDS = (1, 5)
_HISTORY_MAXLEN = max(max(LAGS), max(ROLLING_WINDOWS_SECONDS))


class MockTelemetrySource(TelemetrySource):
    SOURCE_LABEL = "MOCK"

    def __init__(self, config: MockTelemetryConfig, max_records: int | None = None, realtime: bool = True) -> None:
        self._config = config
        self._max_records = max_records
        self._realtime = realtime
        self._rng = np.random.default_rng(config.seed)

        self._session_ids = list(range(1, config.num_sessions + 1))
        self._slice_of = {sid: _MOCK_SLICE_IDS[i % len(_MOCK_SLICE_IDS)] for i, sid in enumerate(self._session_ids)}
        # Persistent per-session radio-quality state for smooth, autocorrelated trajectories.
        self._quality: dict[int, float] = {sid: self._rng.uniform(0.3, 0.9) for sid in self._session_ids}
        # Persistent per-session, per-raw-series history (the 20 MAC avg/std series + 7 target
        # series) so this generator's own `_lag{1..5}`/`_roll_{mean,std}_{5,10}s` features are
        # genuinely computed from past ticks — the same real past-only quantities
        # `src/data/oai_dataset.py`'s `_multi_lag_features`/`_rolling_window_features` compute
        # from a real replay, not independent per-tick noise (prompt.md §48: "demo mode must not
        # be a separate fake architecture").
        rich_series_names = [f"{b}_{stat}" for b in MAC_FEATURE_BASES for stat in ("avg", "std")] + list(TARGET_COLUMNS)
        self._history: dict[int, dict[str, deque]] = {
            sid: {name: deque(maxlen=_HISTORY_MAXLEN) for name in rich_series_names} for sid in self._session_ids
        }
        self._closed = False
        self._tick = 0

    @staticmethod
    def _lag_and_rolling_features(history: deque, raw_value: float, name: str) -> dict[str, float]:
        """Computes `{name}_lag{1..5}`/`{name}_roll_{mean,std}_{5,10}s` from `history` (this
        series' past values, oldest first, NOT yet including `raw_value`) — mirrors
        `src/data/oai_dataset.py`'s own past-only lag/rolling helpers exactly, just computed
        incrementally per tick instead of over a batch DataFrame. `raw_value` is appended to
        `history` AFTER this call, by the caller."""
        past = list(history)  # oldest..newest, strictly BEFORE this tick — never includes raw_value
        out: dict[str, float] = {}
        for k in LAGS:
            out[f"{name}_lag{k}"] = past[-k] if len(past) >= k else float("nan")
        for w in ROLLING_WINDOWS_SECONDS:
            window = past[-w:]
            out[f"{name}_roll_mean_{w}s"] = float(np.mean(window)) if window else float("nan")
            out[f"{name}_roll_std_{w}s"] = float(np.std(window, ddof=1)) if len(window) >= 2 else float("nan")
        return out

    def _next_quality(self, session_id: int) -> float:
        q = self._quality[session_id] + self._rng.normal(0, 0.03)
        q = float(np.clip(q, 0.05, 0.98))
        self._quality[session_id] = q
        return q

    def _generate_record(self, session_id: int, ts_1s: datetime) -> dict[str, Any]:
        rng = self._rng
        quality = self._next_quality(session_id)  # 0-1, higher = better radio conditions

        wb_cqi = float(np.clip(quality * 15 + rng.normal(0, 1), 1, 15))
        pusch_snr = float(np.clip(quality * 30 - 5 + rng.normal(0, 2), -10, 30))
        pucch_snr = pusch_snr + float(rng.normal(0, 1))
        dl_mcs1 = float(np.clip(quality * 27 + rng.normal(0, 1.5), 0, 27))
        ul_mcs1 = dl_mcs1 + float(rng.normal(0, 1))
        phr = float(np.clip((1.0 - quality) * 20 + rng.normal(0, 2), -23, 40))
        bsr = float(max(0.0, rng.normal(5000, 3000)))
        dl_sched_rb = float(np.clip(20 + 40 * (1.0 - quality) + rng.normal(0, 3), 0, 106))
        dl_curr_tbs = float(max(0.0, quality * 8000 + rng.normal(0, 500)))

        dl_bler = float(np.clip((1.0 - quality) * 0.3 + rng.normal(0, 0.01), 0.0, 1.0))
        ul_bler = float(np.clip((1.0 - quality) * 0.2 + rng.normal(0, 0.01), 0.0, 1.0))
        dl_prb = float(max(0.0, 15 + 30 * (1.0 - quality) + rng.normal(0, 3)))
        ul_prb = float(max(0.0, 2 + 5 * (1.0 - quality) + rng.normal(0, 1)))
        dl_throughput = float(max(0.0, quality * 50 * (1.0 - dl_bler) + rng.normal(0, 2)))
        ul_throughput = float(max(0.0, quality * 10 * (1.0 - ul_bler) + rng.normal(0, 0.5)))
        dl_rlc_delay = float(max(0.01, (1.0 - quality) * 15 + rng.normal(0, 1)))

        session_history = self._history[session_id]
        targets_this_tick = {
            "dl_throughput": dl_throughput, "ul_throughput": ul_throughput, "dl_rlc_delay": dl_rlc_delay,
            "dl_prb": dl_prb, "ul_prb": ul_prb, "dl_bler": dl_bler, "ul_bler": ul_bler,
        }

        features: dict[str, float] = {}
        for base in MAC_FEATURE_BASES:
            avg_value = {
                "dl_curr_tbs": dl_curr_tbs, "dl_sched_rb": dl_sched_rb, "pusch_snr": pusch_snr,
                "pucch_snr": pucch_snr, "wb_cqi": wb_cqi, "dl_mcs1": dl_mcs1, "ul_mcs1": ul_mcs1,
                "phr": phr, "dl_bler": dl_bler, "ul_bler": ul_bler,
            }[base]
            std_value = float(abs(rng.normal(0, 0.1 * (abs(avg_value) + 1e-6))))
            for raw_name, raw_value in ((f"{base}_avg", avg_value), (f"{base}_std", std_value)):
                features.update(self._lag_and_rolling_features(session_history[raw_name], raw_value, raw_name))
                session_history[raw_name].append(raw_value)
        features["bsr_avg_lag1"] = bsr
        features["n_samples_lag1"] = 1.0
        features["ue_count_cell_lag1"] = float(len(self._session_ids))
        features["ue_count_slice_lag1"] = float(sum(1 for s in self._slice_of.values() if s == self._slice_of[session_id]))
        features["requested_prb_ratio_lag1"] = float(np.clip(50.0 + rng.normal(0, 5), 0, 100))
        features["has_controller_lag1"] = 1.0
        for name, raw_value in targets_this_tick.items():
            features.update(self._lag_and_rolling_features(session_history[name], raw_value, name))
            session_history[name].append(raw_value)

        record: dict[str, Any] = {
            "source": self.SOURCE_LABEL,
            "run_id": _MOCK_RUN_ID,
            "cell_id": _MOCK_CELL_ID,
            "session_id": session_id,
            "ue_id": session_id,
            "slice_id": self._slice_of[session_id],
            "ts_1s": ts_1s,
            "dl_throughput": dl_throughput,
            "ul_throughput": ul_throughput,
            "dl_rlc_delay": dl_rlc_delay,
            "dl_prb": dl_prb,
            "ul_prb": ul_prb,
            "dl_bler": dl_bler,
            "ul_bler": ul_bler,
            **features,
        }

        feature_keys = list(features.keys())
        if rng.random() < self._config.missing_feature_rate and feature_keys:
            del record[rng.choice(feature_keys)]

        return record

    def records(self) -> Iterator[dict[str, Any]]:
        emitted = 0
        while not self._closed and (self._max_records is None or emitted < self._max_records):
            ts_1s = datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=self._tick)
            self._tick += 1
            for session_id in self._session_ids:
                if self._closed or (self._max_records is not None and emitted >= self._max_records):
                    return
                yield self._generate_record(session_id, ts_1s)
                emitted += 1
            if self._realtime:
                time.sleep(self._config.emit_interval_seconds)

    def close(self) -> None:
        self._closed = True
