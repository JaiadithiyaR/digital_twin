"""Unit tests for structured logging (src/common/logging.py)."""

from __future__ import annotations

import json
import logging

from src.common.logging import _JsonFormatter, _RedactFilter, get_logger


def _make_record(msg: str, **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1, msg=msg, args=(), exc_info=None
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_includes_extra_fields():
    formatter = _JsonFormatter()
    record = _make_record("drift detected", component="latency", severity=0.8)
    payload = json.loads(formatter.format(record))
    assert payload["message"] == "drift detected"
    assert payload["component"] == "latency"
    assert payload["severity"] == 0.8
    assert payload["level"] == "INFO"


def test_redact_filter_scrubs_secret_like_fields():
    record = _make_record("calling anthropic", api_key="sk-ant-super-secret")
    _RedactFilter().filter(record)
    assert record.api_key == "<redacted>"


def test_get_logger_returns_named_logger():
    log = get_logger("module.submodule")
    assert log.name == "module.submodule"
