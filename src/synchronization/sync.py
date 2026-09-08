"""Continuous Synchronization (Module 3, prompt.md §9, §0.6, §0.7).

Drains a `TelemetrySource` (Module 2's real ZeroMQ path or mock path) continuously, runs each
accumulated batch through Module 2's `TelemetryPreprocessor`, and pushes the results into a
`D1StateSink` (Module 4's future write interface — see `d1_interface.py`).

There is NO manual "sync now" entry point. `start()` launches a background thread that keeps
synchronizing for as long as the source keeps producing telemetry; `run()` is the blocking form
of the same loop. This is the mechanism that keeps the Digital Twin dynamically synchronized
(prompt.md §0.6) — every valid batch of incoming telemetry updates D1's current state and
extends its history automatically, with no external trigger required.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from src.common.config import SynchronizationConfig
from src.synchronization.d1_interface import D1StateSink
from src.telemetry.base import TelemetrySource
from src.telemetry.preprocessing import TelemetryPreprocessor

logger = logging.getLogger(__name__)


class ContinuousSynchronizer:
    def __init__(
        self,
        source: TelemetrySource,
        preprocessor: TelemetryPreprocessor,
        d1_sink: D1StateSink,
        config: SynchronizationConfig,
    ) -> None:
        self._source = source
        self._preprocessor = preprocessor
        self._d1_sink = d1_sink
        self._batch_size = config.batch_size
        self._batch_timeout_seconds = config.batch_timeout_seconds
        self._stop_event = threading.Event()
        self._batches_synced = 0
        self._records_synced = 0

    @property
    def batches_synced(self) -> int:
        return self._batches_synced

    @property
    def records_synced(self) -> int:
        return self._records_synced

    def run(self) -> None:
        """Blocking continuous loop — synchronizes telemetry into D1 with no manual trigger.

        Exits when `stop()` is called, or when the underlying source's `records()` generator is
        exhausted on its own (e.g. a bounded mock source in tests; a live ZeroMQ/unbounded mock
        source in demo/live mode runs until explicitly stopped).
        """
        buffer: list[dict[str, Any]] = []
        last_flush = time.monotonic()
        logger.info(
            "continuous synchronization started",
            extra={
                "component": "synchronization",
                "batch_size": self._batch_size,
                "batch_timeout_seconds": self._batch_timeout_seconds,
            },
        )

        for raw in self._source.records():
            if self._stop_event.is_set():
                break
            buffer.append(raw)
            elapsed = time.monotonic() - last_flush
            if len(buffer) >= self._batch_size or elapsed >= self._batch_timeout_seconds:
                self._sync_batch(buffer)
                buffer = []
                last_flush = time.monotonic()

        if buffer:
            self._sync_batch(buffer)

        logger.info(
            "continuous synchronization stopped",
            extra={
                "component": "synchronization",
                "batches_synced": self._batches_synced,
                "records_synced": self._records_synced,
            },
        )

    def start(self) -> threading.Thread:
        """Launch `run()` in a background daemon thread and return the thread handle. This is
        the normal way to use the synchronizer: call `start()` once at startup and it keeps
        running — there is nothing else to call to keep the Digital Twin synchronized."""
        thread = threading.Thread(target=self.run, name="dt-continuous-sync", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        """Signal the loop to exit and close the source so a generator blocked waiting for the
        next record actually returns (both `MockTelemetrySource` and `ZmqTelemetrySource` check
        their closed state within one poll/timeout cycle)."""
        self._stop_event.set()
        self._source.close()

    def _sync_batch(self, raw_batch: list[dict[str, Any]]) -> None:
        result = self._preprocessor.process_batch(raw_batch)

        if result.clean_records:
            self._d1_sink.update_current_state(result.clean_records)
            self._d1_sink.append_history(result.clean_records)
        if result.quarantined:
            self._d1_sink.record_quarantine(result.quarantined)

        self._batches_synced += 1
        self._records_synced += result.total_received
        logger.info(
            "telemetry batch synchronized into D1",
            extra={
                "component": "synchronization",
                "batch_size": len(raw_batch),
                "clean": len(result.clean_records),
                "quarantined": len(result.quarantined),
                "batches_synced": self._batches_synced,
                "records_synced": self._records_synced,
            },
        )
