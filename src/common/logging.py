"""Structured logging setup (prompt.md §50).

Every stage of the pipeline (telemetry -> sync -> DT -> fidelity -> drift -> PPO -> agent ->
candidate -> sandbox -> verification -> lifecycle) should log through `get_logger(__name__)` with
structured `extra=` fields (e.g. `component=`, `event=`, `version=`) so an adaptation can be
followed end-to-end from the logs. Never pass secrets as fields — `_RedactFilter` scrubs common
secret-like key names defensively, but callers must not rely on it as the only safeguard.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

_CONFIGURED = False
_REDACTED_KEYS = {"api_key", "anthropic_api_key", "secret", "token", "password", "authorization"}
_REDACTED_VALUE = "<redacted>"

# Attributes that already exist on a standard LogRecord — anything else passed via `extra=`
# is application-supplied structured context and gets folded into the JSON output.
_STANDARD_RECORD_KEYS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
}


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key in list(record.__dict__.keys()):
            if key.lower() in _REDACTED_KEYS:
                record.__dict__[key] = _REDACTED_VALUE
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_KEYS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s%(extra_suffix)s")

    def format(self, record: logging.LogRecord) -> str:
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_RECORD_KEYS and not k.startswith("_")
        }
        record.extra_suffix = (
            " | " + " ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
        )
        return super().format(record)


def setup_logging(level: str = "INFO", fmt: str = "json", log_dir: str | Path | None = None) -> None:
    """Configure the root logger once. Safe to call multiple times (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    formatter: logging.Formatter = _JsonFormatter() if fmt == "json" else _TextFormatter()
    root = logging.getLogger()
    root.setLevel(level.upper())

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_RedactFilter())
    root.addHandler(console_handler)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "digital_twin.log", maxBytes=10_000_000, backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_RedactFilter())
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
