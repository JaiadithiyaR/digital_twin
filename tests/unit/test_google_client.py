"""Tests for the centralized Anthropic API client (prompt.md §24, §42).

No real ANTHROPIC_API_KEY is configured in this development environment (`.env` holds only the
placeholder from `.env.example`) — genuinely calling the live API here would either fail loudly
or silently do nothing useful, and prompt.md §0.24/§20 forbid faking a real API response. Every
test below therefore drives `AnthropicClient` against a MOCKED `messages.create` (the one
correct way to test retry/backoff/failure-handling logic deterministically and without cost or
flakiness — this is standard practice, not a workaround), using real `httpx.Request`/
`httpx.Response` objects and real `anthropic.*Error` exception types so the client's actual
exception-classification logic is exercised, not a stand-in.

`test_live_api_smoke_if_key_configured` is the one exception: it is skipped unless a real key is
present, and would genuinely call the API if one ever is — see its docstring.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

import anthropic
from src.common.config import load_settings
from src.llm.anthropic_client import (
    AnthropicClient,
    LLMStructuredOutputError,
    LLMTransportError,
)

SETTINGS = load_settings()


class Echo(BaseModel):
    """Trivial schema used only by these tests — not a real agent output shape."""

    message: str
    count: int


def _client(**config_overrides) -> AnthropicClient:
    config = SETTINGS.llm.model_copy(update=config_overrides) if config_overrides else SETTINGS.llm
    return AnthropicClient(config, api_key="sk-ant-test-key-not-real")


def _fake_message(text: str, *, model: str = "claude-sonnet-5", stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        model=model,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=12, output_tokens=7),
    )


def _rate_limit_error(retry_after: str | None = None) -> anthropic.RateLimitError:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    resp = httpx.Response(429, request=req, headers=headers)
    return anthropic.RateLimitError("rate limited", response=resp, body=None)


def _auth_error() -> anthropic.AuthenticationError:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(401, request=req)
    return anthropic.AuthenticationError("invalid api key", response=resp, body=None)


class _FakeMessagesEndpoint:
    """Stands in for `client.messages` — `create(**kwargs)` is what `AnthropicClient` calls."""

    def __init__(self, side_effect: list[Any]) -> None:
        self._side_effect = list(side_effect)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self._side_effect.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _patch_transport(client: AnthropicClient, side_effect: list[Any]) -> _FakeMessagesEndpoint:
    fake = _FakeMessagesEndpoint(side_effect)
    client._client.messages = fake  # only the `.messages.create` surface AnthropicClient touches
    return fake


# --- successful completion --------------------------------------------------------------------


def test_complete_against_a_trivial_prompt_returns_parsed_response(monkeypatch):
    client = _client()
    fake = _patch_transport(client, [_fake_message("hello back")])

    response = client.complete("say hello")

    assert response.text == "hello back"
    assert response.model == "claude-sonnet-5"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 7
    assert response.attempts == 1
    assert len(fake.calls) == 1


def test_complete_uses_configured_model_never_hardcoded():
    client = _client(model="claude-haiku-4-5")
    fake = _patch_transport(client, [_fake_message("ok")])

    client.complete("say hello")

    assert fake.calls[0]["model"] == "claude-haiku-4-5"


def test_complete_sends_configured_effort_when_set():
    client = _client(effort="high")
    fake = _patch_transport(client, [_fake_message("ok")])

    client.complete("say hello")

    assert fake.calls[0]["output_config"]["effort"] == "high"


def test_complete_call_effort_override_takes_precedence_over_config():
    client = _client(effort="low")
    fake = _patch_transport(client, [_fake_message("ok")])

    client.complete("say hello", effort="max")

    assert fake.calls[0]["output_config"]["effort"] == "max"


# --- transport retry / backoff / failure handling ----------------------------------------------


def test_complete_retries_on_retryable_error_then_succeeds(monkeypatch):
    client = _client(max_retries=3)
    fake = _patch_transport(client, [_rate_limit_error(), _rate_limit_error(), _fake_message("finally")])
    sleeps: list[float] = []
    monkeypatch.setattr(client, "_sleep", lambda s: sleeps.append(s))

    response = client.complete("say hello")

    assert response.text == "finally"
    assert response.attempts == 3
    assert len(fake.calls) == 3
    assert len(sleeps) == 2  # slept before the 2nd and 3rd attempts, not after success


def test_complete_does_not_retry_non_retryable_error(monkeypatch):
    client = _client(max_retries=3)
    fake = _patch_transport(client, [_auth_error()])
    sleeps: list[float] = []
    monkeypatch.setattr(client, "_sleep", lambda s: sleeps.append(s))

    with pytest.raises(LLMTransportError) as exc_info:
        client.complete("say hello")

    assert exc_info.value.retryable is False
    assert len(fake.calls) == 1  # fails fast, no retry budget spent
    assert sleeps == []


def test_complete_raises_transport_error_after_exhausting_retries(monkeypatch):
    client = _client(max_retries=2)
    fake = _patch_transport(client, [_rate_limit_error(), _rate_limit_error(), _rate_limit_error()])
    monkeypatch.setattr(client, "_sleep", lambda s: None)

    with pytest.raises(LLMTransportError) as exc_info:
        client.complete("say hello")

    assert exc_info.value.retryable is True
    assert len(fake.calls) == 3  # max_retries=2 -> 3 total attempts


def test_backoff_uses_retry_after_header_when_present(monkeypatch):
    client = _client(max_retries=1, retry_backoff_seconds=1.0)
    _patch_transport(client, [_rate_limit_error(retry_after="2.5"), _fake_message("ok")])
    sleeps: list[float] = []
    monkeypatch.setattr(client, "_sleep", lambda s: sleeps.append(s))

    client.complete("say hello")

    assert sleeps == [2.5]


def test_backoff_is_exponential_without_retry_after_header(monkeypatch):
    client = _client(max_retries=3, retry_backoff_seconds=1.0)
    _patch_transport(
        client, [_rate_limit_error(), _rate_limit_error(), _rate_limit_error(), _fake_message("ok")]
    )
    sleeps: list[float] = []
    monkeypatch.setattr(client, "_sleep", lambda s: sleeps.append(s))

    client.complete("say hello")

    assert sleeps == [1.0, 2.0, 4.0]


# --- structured output ---------------------------------------------------------------------


def test_complete_structured_returns_validated_pydantic_instance():
    client = _client()
    payload = json.dumps({"message": "hi", "count": 3})
    fake = _patch_transport(client, [_fake_message(payload)])

    result = client.complete_structured("echo hi three times", Echo)

    assert isinstance(result, Echo)
    assert result.message == "hi"
    assert result.count == 3
    assert len(fake.calls) == 1


def test_complete_structured_sends_json_schema_output_format():
    client = _client()
    payload = json.dumps({"message": "hi", "count": 1})
    fake = _patch_transport(client, [_fake_message(payload)])

    client.complete_structured("echo", Echo)

    output_format = fake.calls[0]["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["schema"] == Echo.model_json_schema()


def test_complete_structured_retries_on_validation_failure_then_succeeds(monkeypatch):
    client = _client(max_retries=2)
    valid_payload = json.dumps({"message": "hi", "count": 2})
    fake = _patch_transport(client, [_fake_message("not json at all"), _fake_message(valid_payload)])
    monkeypatch.setattr(client, "_sleep", lambda s: None)

    result = client.complete_structured("echo", Echo)

    assert result.count == 2
    assert len(fake.calls) == 2


def test_complete_structured_raises_after_exhausting_validation_retries(monkeypatch):
    client = _client(max_retries=1)
    fake = _patch_transport(client, [_fake_message("still not json"), _fake_message("also not json")])
    monkeypatch.setattr(client, "_sleep", lambda s: None)

    with pytest.raises(LLMStructuredOutputError) as exc_info:
        client.complete_structured("echo", Echo)

    assert exc_info.value.raw_text == "also not json"
    assert len(fake.calls) == 2  # max_retries=1 -> 2 total attempts


def test_complete_structured_rejects_json_missing_required_field(monkeypatch):
    client = _client(max_retries=0)
    incomplete_payload = json.dumps({"message": "hi"})  # missing required "count"
    _patch_transport(client, [_fake_message(incomplete_payload)])

    with pytest.raises(LLMStructuredOutputError):
        client.complete_structured("echo", Echo)


# --- graceful degradation (complete_safe / complete_structured_safe) ---------------------------


def test_complete_safe_returns_response_on_success():
    client = _client()
    _patch_transport(client, [_fake_message("ok")])

    result = client.complete_safe("say hello")

    assert result is not None
    assert result.text == "ok"


def test_complete_safe_degrades_to_none_and_logs_warning(caplog):
    client = _client(max_retries=0)
    _patch_transport(client, [_auth_error()])

    with caplog.at_level(logging.WARNING):
        result = client.complete_safe("say hello")

    assert result is None
    assert any("degrading gracefully" in r.message for r in caplog.records)


def test_complete_structured_safe_degrades_to_none_on_failure(caplog, monkeypatch):
    client = _client(max_retries=0)
    _patch_transport(client, [_fake_message("not json")])
    monkeypatch.setattr(client, "_sleep", lambda s: None)

    with caplog.at_level(logging.WARNING):
        result = client.complete_structured_safe("echo", Echo)

    assert result is None
    assert any("degrading gracefully" in r.message for r in caplog.records)


def test_complete_structured_safe_returns_instance_on_success():
    client = _client()
    payload = json.dumps({"message": "hi", "count": 5})
    _patch_transport(client, [_fake_message(payload)])

    result = client.complete_structured_safe("echo", Echo)

    assert result is not None
    assert result.count == 5


# --- secret handling -----------------------------------------------------------------------------


def test_api_key_never_appears_in_logs(caplog):
    real_looking_key = "sk-ant-super-secret-value-should-never-appear-in-logs"
    client = AnthropicClient(SETTINGS.llm.model_copy(update={"max_retries": 0}), api_key=real_looking_key)
    _patch_transport(client, [_auth_error()])

    with caplog.at_level(logging.WARNING):
        client.complete_safe("say hello")

    for record in caplog.records:
        assert real_looking_key not in record.getMessage()
        for value in record.__dict__.values():
            assert real_looking_key not in str(value)


def test_from_settings_raises_clearly_when_no_real_key_configured():
    from src.common.config import Secrets

    placeholder_secrets = Secrets(anthropic_api_key=None)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicClient.from_settings(SETTINGS, placeholder_secrets)


# --- optional live smoke test ---------------------------------------------------------------


def _real_api_key_configured() -> bool:
    from src.common.config import load_secrets

    key = load_secrets().anthropic_api_key
    return bool(key) and key != "sk-ant-your-key-here"


@pytest.mark.skipif(not _real_api_key_configured(), reason="no real ANTHROPIC_API_KEY configured in this environment")
def test_live_api_smoke_if_key_configured():
    """Only runs if a real key is ever configured in `.env` — genuinely calls the live API with
    a trivial prompt and a trivial structured schema. Skipped (not faked) otherwise, per
    prompt.md §0.24/§20 — this repo never claims a real API call happened when it didn't."""
    from src.common.config import load_secrets

    client = AnthropicClient.from_settings(SETTINGS, load_secrets())
    response = client.complete("Reply with exactly the word: pong")
    assert "pong" in response.text.lower()

    structured = client.complete_structured("Echo back the word 'ping' and the count 1.", Echo)
    assert structured.message
    assert structured.count == 1
