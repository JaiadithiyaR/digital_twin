"""Real telemetry transport: ZeroMQ SUB client (prompt.md §0.2 step 10, §7, §49).

Consumes the structured streaming telemetry produced by the NS-3/5G-LENA ZeroMQ exporter
(Module 1's `ns3_sim` scenario — see IMPLEMENTATION_STATUS.md for that component's build status).
Every record received through this class is unconditionally stamped `source="NS3_5G_LENA"`,
regardless of what the JSON payload itself claims — this class IS the real transport, so whatever
arrives through it is authoritatively real by construction (prompt.md §0.3, §44: untrusted
payload content must never be able to override its own provenance).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import zmq

from src.common.config import ZmqTelemetryConfig
from src.telemetry.base import TelemetrySource

logger = logging.getLogger(__name__)


class ZmqTelemetrySource(TelemetrySource):
    SOURCE_LABEL = "NS3_5G_LENA"

    def __init__(self, config: ZmqTelemetryConfig, context: "zmq.Context | None" = None) -> None:
        self._config = config
        self._owns_context = context is None
        self._context = context or zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, config.high_water_mark)
        self._socket.setsockopt(zmq.RCVTIMEO, config.recv_timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(config.endpoint)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, config.topic)
        self._closed = False
        logger.info(
            "zmq telemetry source connected",
            extra={"component": "telemetry", "source": self.SOURCE_LABEL, "endpoint": config.endpoint},
        )

    def records(self) -> Iterator[dict[str, Any]]:
        """Yield decoded telemetry messages indefinitely until `close()` is called.

        A receive timeout is expected steady-state behaviour (no telemetry arrived within the
        configured window) and is logged at debug level, not treated as an error — the caller
        keeps waiting. Malformed payloads are logged and skipped, never allowed to crash the
        continuous synchronization loop (Module 3) that consumes this generator.

        libzmq forbids using a socket from more than one thread (undefined behaviour, observed
        here as an intermittent native SIGABRT when a concurrent `close()` raced an in-flight
        `recv_multipart()` from a different thread — prompt.md §0.26 "race conditions"). So the
        socket/context are torn down HERE, in whichever thread is actually running this
        generator, once `close()` (callable from any thread) has merely flagged `_closed` — never
        from `close()` itself.
        """
        try:
            while not self._closed:
                try:
                    _topic, payload = self._socket.recv_multipart()
                except zmq.Again:
                    logger.debug(
                        "no telemetry within recv timeout, still waiting",
                        extra={"component": "telemetry", "source": self.SOURCE_LABEL},
                    )
                    continue
                except zmq.ZMQError as exc:
                    if self._closed:
                        return
                    logger.warning(
                        "zmq recv error", extra={"component": "telemetry", "error": str(exc)}
                    )
                    continue

                try:
                    record = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "dropping malformed telemetry payload",
                        extra={"component": "telemetry", "source": self.SOURCE_LABEL, "error": str(exc)},
                    )
                    continue

                if not isinstance(record, dict):
                    logger.warning(
                        "dropping non-object telemetry payload",
                        extra={"component": "telemetry", "source": self.SOURCE_LABEL},
                    )
                    continue

                # Authoritative stamp — see module docstring. Overrides any 'source' the payload
                # itself contains.
                record["source"] = self.SOURCE_LABEL
                yield record
        finally:
            # Runs whether the loop exited via `_closed` or the generator was abandoned/GC'd —
            # always in this generator's own thread, so socket/context teardown is thread-safe.
            self._socket.close()
            if self._owns_context:
                self._context.term()
            logger.info(
                "zmq telemetry source closed",
                extra={"component": "telemetry", "source": self.SOURCE_LABEL},
            )

    def close(self) -> None:
        """Signal the consuming thread's `records()` loop to stop and tear down its own socket.

        Safe to call from any thread — this only sets a flag. Actual socket/context cleanup
        happens inside `records()`, in the thread that owns the socket (see its docstring)."""
        self._closed = True
