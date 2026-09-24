"""Tests for the centralized Google AI (Gemini) API client (prompt.md §24, §42).

No real GOOGLE_API_KEY/GEMINI_API_KEY is configured in this development environment (`.env` holds
only the placeholder from `.env.example`) — genuinely calling the live API here would either fail
loudly or silently do nothing useful, and prompt.md §0.24/§20 forbid faking a real API response.
Every test below therefore drives `GoogleClient` against a MOCKED `models.generate_content` (the
one correct way to test retry/backoff/failure-handling logic deterministically and without cost
or flakiness — this is standard practice, not a workaround), using real `httpx.Request`/
`httpx.Response` objects and real `google.genai.errors.*` exception types so the client's actual
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

from google.genai import errors as genai_errors
from google.genai import types as genai_types
from src.common.config import load_settings
from src.llm.google_client import (
    GoogleClient,
    LLMStructuredOutputError,
    LLMTransportError,
)

SETTINGS = load_settings()
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/x:generateContent"


class Echo(BaseModel):
    """Trivial schema used only by these tests — not a real agent output shape."""

    message: str
    count: int


def _client(**config_overrides) -> GoogleClient:
    config = SETTINGS.llm.model_copy(update=config_overrides) if config_overrides else SETTINGS.llm
    return GoogleClient(config, api_key="fake-google-test-key-not-real")


def _fake_response(text: str, *, model: str = "gemini-2.5-flash", finish_reason: str = "STOP") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        model_version=model,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(value=finish_reason))],
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=7),
    )


def _client_error(code: int, message: str = "client error", *, retry_after: str | None = None) -> genai_errors.ClientError:
    req = httpx.Request("POST", _ENDPOINT)
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    resp = httpx.Response(code, request=req, headers=headers)
    return genai_errors.ClientError(code, {"message": message, "status": "ERROR"}, resp)


def _server_error(code: int = 503, message: str = "server unavailable") -> genai_errors.ServerError:
    req = httpx.Request("POST", _ENDPOINT)
    resp = httpx.Response(code, request=req)
    return genai_errors.ServerError(code, {"message": message, "status": "UNAVAILABLE"}, resp)


class _FakeModels:
    """Stands in for the real `Models` object behind `client.models` — `generate_content(**kwargs)`
    is what `GoogleClient` calls."""

    def __init__(self, side_effect: list[Any]) -> None:
        self._side_effect = list(side_effect)
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self._side_effect.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _patch_transport(client: GoogleClient, side_effect: list[Any]) -> _FakeModels:
    fake = _FakeModels(side_effect)
    # `Client.models` is a read-only property returning the stored `_models` instance attribute
    # (verified against google/genai/client.py) — patch the underlying attribute directly, since
    # the property itself has no setter.
    client._client._models = fake
    return fake


# --- successful completion --------------------------------------------------------------------


def test_complete_against_a_trivial_prompt_returns_parsed_response():
    client = _client()
    fake = _patch_transport(client, [_fake_response("hello back")])

    response = client.complete("say hello")

    assert response.text == "hello back"
    assert response.model == "gemini-2.5-flash"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 7
    assert response.attempts == 1
    assert response.stop_reason == "STOP"
    assert len(fake.calls) == 1


def test_complete_uses_configured_model_never_hardcoded():
    client = _client(model="gemini-2.0-flash")
    fake = _patch_transport(client, [_fake_response("ok")])

    client.complete("say hello")

    assert fake.calls[0]["model"] == "gemini-2.0-flash"


def test_complete_sends_configured_effort_as_thinking_level_when_set():
    client = _client(effort="high")
    fake = _patch_transport(client, [_fake_response("ok")])

    client.complete("say hello")

    assert fake.calls[0]["config"].thinking_config.thinking_level == genai_types.ThinkingLevel.HIGH


def test_complete_clamps_xhigh_and_max_effort_to_thinking_level_high():
    for effort in ("xhigh", "max"):
        client = _client(effort=effort)
        fake = _patch_transport(client, [_fake_response("ok")])
        client.complete("say hello")
        assert fake.calls[0]["config"].thinking_config.thinking_level == genai_types.ThinkingLevel.HIGH


def test_complete_call_effort_override_takes_precedence_over_config():
    client = _client(effort="low")
    fake = _patch_transport(client, [_fake_response("ok")])

    client.complete("say hello", effort="high")

    assert fake.calls[0]["config"].thinking_config.thinking_level == genai_types.ThinkingLevel.HIGH


def test_complete_sends_configured_temperature_when_set():
    # Unlike the old Anthropic client, temperature is a REAL determinism knob on this SDK
    # (verified against types.GenerateContentConfig) — genuinely sent, not a fiction.
    client = _client(temperature=0.0)
    fake = _patch_transport(client, [_fake_response("ok")])

    client.complete("say hello")

    assert fake.calls[0]["config"].temperature == 0.0


def test_complete_omits_temperature_when_none():
    client = _client(temperature=None)
    fake = _patch_transport(client, [_fake_response("ok")])

    client.complete("say hello")

    assert fake.calls[0]["config"].temperature is None


def test_complete_sends_system_instruction_when_provided():
    client = _client()
    fake = _patch_transport(client, [_fake_response("ok")])

    client.complete("say hello", system="You are terse.")

    assert fake.calls[0]["config"].system_instruction == "You are terse."


# --- transport retry / backoff / failure handling ----------------------------------------------


def test_complete_retries_on_retryable_server_error_then_succeeds(monkeypatch):
    client = _client(max_retries=3)
    fake = _patch_transport(client, [_server_error(), _server_error(), _fake_response("finally")])
    sleeps: list[float] = []
    monkeypatch.setattr(client, "_sleep", lambda s: sleeps.append(s))

    response = client.complete("say hello")

    assert response.text == "finally"
    assert response.attempts == 3
    assert len(fake.calls) == 3
    assert len(sleeps) == 2  # slept before the 2nd and 3rd attempts, not after success


def test_complete_retries_on_rate_limited_client_error_429(monkeypatch):
    client = _client(max_retries=1)
    fake = _patch_transport(client, [_client_error(429, "rate limited"), _fake_response("ok")])
    monkeypatch.setattr(client, "_sleep", lambda s: None)

    response = client.complete("say hello")

    assert response.text == "ok"
    assert len(fake.calls) == 2


def test_complete_retries_on_transient_transport_exception(monkeypatch):
    # google-genai's own transport can raise raw httpx exceptions uncaught by the SDK's error
    # hierarchy (verified against google.genai._api_client._HTTPX_TRANSIENT_EXC) — real, not
    # hypothetical, so this client must classify these as retryable too.
    client = _client(max_retries=1)
    fake = _patch_transport(client, [httpx.ConnectError("connection refused"), _fake_response("ok")])
    monkeypatch.setattr(client, "_sleep", lambda s: None)

    response = client.complete("say hello")

    assert response.text == "ok"
    assert len(fake.calls) == 2


def test_complete_does_not_retry_non_retryable_client_error(monkeypatch):
    client = _client(max_retries=3)
    fake = _patch_transport(client, [_client_error(401, "invalid api key")])
    sleeps: list[float] = []
    monkeypatch.setattr(client, "_sleep", lambda s: sleeps.append(s))

    with pytest.raises(LLMTransportError) as exc_info:
        client.complete("say hello")

    assert exc_info.value.retryable is False
    assert len(fake.calls) == 1  # fails fast, no retry budget spent
    assert sleeps == []


def test_complete_raises_transport_error_after_exhausting_retries(monkeypatch):
    client = _client(max_retries=2)
    fake = _patch_transport(client, [_server_error(), _server_error(), _server_error()])
    monkeypatch.setattr(client, "_sleep", lambda s: None)

    with pytest.raises(LLMTransportError) as exc_info:
        client.complete("say hello")

    assert exc_info.value.retryable is True
    assert len(fake.calls) == 3  # max_retries=2 -> 3 total attempts


def test_unclassified_api_error_is_treated_as_non_retryable(monkeypatch):
    # A bare APIError (neither ClientError nor ServerError) is a real possible shape per
    # errors.py's own raise_error (any status code outside 400-599) — fail safe, not guessed.
    client = _client(max_retries=3)
    unclassified = genai_errors.APIError(299, {"message": "weird status", "status": "UNKNOWN"}, None)
    fake = _patch_transport(client, [unclassified])
    sleeps: list[float] = []
    monkeypatch.setattr(client, "_sleep", lambda s: sleeps.append(s))

    with pytest.raises(LLMTransportError) as exc_info:
        client.complete("say hello")

    assert exc_info.value.retryable is False
    assert len(fake.calls) == 1
    assert sleeps == []


def test_complete_raises_on_empty_text_response(monkeypatch):
    # e.g. a safety-blocked response with no text content — must never be silently treated as ""
    client = _client(max_retries=0)
    _patch_transport(client, [_fake_response("")])

    with pytest.raises(LLMTransportError):
        client.complete("say hello")


def test_backoff_uses_retry_after_header_when_present(monkeypatch):
    client = _client(max_retries=1, retry_backoff_seconds=1.0)
    _patch_transport(client, [_client_error(429, "rate limited", retry_after="2.5"), _fake_response("ok")])
    sleeps: list[float] = []
    monkeypatch.setattr(client, "_sleep", lambda s: sleeps.append(s))

    client.complete("say hello")

    assert sleeps == [2.5]


def test_backoff_is_exponential_without_retry_after_header(monkeypatch):
    client = _client(max_retries=3, retry_backoff_seconds=1.0)
    _patch_transport(client, [_server_error(), _server_error(), _server_error(), _fake_response("ok")])
    sleeps: list[float] = []
    monkeypatch.setattr(client, "_sleep", lambda s: sleeps.append(s))

    client.complete("say hello")

    assert sleeps == [1.0, 2.0, 4.0]


# --- structured output ---------------------------------------------------------------------


def test_complete_structured_returns_validated_pydantic_instance():
    client = _client()
    payload = json.dumps({"message": "hi", "count": 3})
    fake = _patch_transport(client, [_fake_response(payload)])

    result = client.complete_structured("echo hi three times", Echo)

    assert isinstance(result, Echo)
    assert result.message == "hi"
    assert result.count == 3
    assert len(fake.calls) == 1


def test_complete_structured_sends_native_json_schema_output_format():
    client = _client()
    payload = json.dumps({"message": "hi", "count": 1})
    fake = _patch_transport(client, [_fake_response(payload)])

    client.complete_structured("echo", Echo)

    config = fake.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == Echo.model_json_schema()


def test_complete_structured_retries_on_validation_failure_then_succeeds(monkeypatch):
    client = _client(max_retries=2)
    valid_payload = json.dumps({"message": "hi", "count": 2})
    fake = _patch_transport(client, [_fake_response("not json at all"), _fake_response(valid_payload)])
    monkeypatch.setattr(client, "_sleep", lambda s: None)

    result = client.complete_structured("echo", Echo)

    assert result.count == 2
    assert len(fake.calls) == 2


def test_complete_structured_raises_after_exhausting_validation_retries(monkeypatch):
    client = _client(max_retries=1)
    fake = _patch_transport(client, [_fake_response("still not json"), _fake_response("also not json")])
    monkeypatch.setattr(client, "_sleep", lambda s: None)

    with pytest.raises(LLMStructuredOutputError) as exc_info:
        client.complete_structured("echo", Echo)

    assert exc_info.value.raw_text == "also not json"
    assert len(fake.calls) == 2  # max_retries=1 -> 2 total attempts


def test_complete_structured_rejects_json_missing_required_field(monkeypatch):
    client = _client(max_retries=0)
    incomplete_payload = json.dumps({"message": "hi"})  # missing required "count"
    _patch_transport(client, [_fake_response(incomplete_payload)])

    with pytest.raises(LLMStructuredOutputError):
        client.complete_structured("echo", Echo)


# --- graceful degradation (complete_safe / complete_structured_safe) ---------------------------


def test_complete_safe_returns_response_on_success():
    client = _client()
    _patch_transport(client, [_fake_response("ok")])

    result = client.complete_safe("say hello")

    assert result is not None
    assert result.text == "ok"


def test_complete_safe_degrades_to_none_and_logs_warning(caplog):
    client = _client(max_retries=0)
    _patch_transport(client, [_client_error(401, "invalid api key")])

    with caplog.at_level(logging.WARNING):
        result = client.complete_safe("say hello")

    assert result is None
    assert any("degrading gracefully" in r.message for r in caplog.records)


def test_complete_structured_safe_degrades_to_none_on_failure(caplog, monkeypatch):
    client = _client(max_retries=0)
    _patch_transport(client, [_fake_response("not json")])
    monkeypatch.setattr(client, "_sleep", lambda s: None)

    with caplog.at_level(logging.WARNING):
        result = client.complete_structured_safe("echo", Echo)

    assert result is None
    assert any("degrading gracefully" in r.message for r in caplog.records)


def test_complete_structured_safe_returns_instance_on_success():
    client = _client()
    payload = json.dumps({"message": "hi", "count": 5})
    _patch_transport(client, [_fake_response(payload)])

    result = client.complete_structured_safe("echo", Echo)

    assert result is not None
    assert result.count == 5


# --- secret handling -----------------------------------------------------------------------------


def test_api_key_never_appears_in_logs(caplog):
    real_looking_key = "AIzaSySuperSecretGoogleApiKeyShouldNeverAppearInLogs"
    client = GoogleClient(SETTINGS.llm.model_copy(update={"max_retries": 0}), api_key=real_looking_key)
    _patch_transport(client, [_client_error(401, "invalid api key")])

    with caplog.at_level(logging.WARNING):
        client.complete_safe("say hello")

    for record in caplog.records:
        assert real_looking_key not in record.getMessage()
        for value in record.__dict__.values():
            assert real_looking_key not in str(value)


def test_from_settings_raises_clearly_when_no_real_key_configured():
    from src.common.config import Secrets

    placeholder_secrets = Secrets(google_api_key=None)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        GoogleClient.from_settings(SETTINGS, placeholder_secrets)


# --- optional live smoke test ---------------------------------------------------------------


def _real_api_key_configured() -> bool:
    from src.common.config import load_secrets

    key = load_secrets().google_api_key
    return bool(key) and key != "your-google-ai-api-key-here"


@pytest.mark.skipif(
    not _real_api_key_configured(), reason="no real GOOGLE_API_KEY/GEMINI_API_KEY configured in this environment"
)
def test_live_api_smoke_if_key_configured():
    """Only runs if a real key is ever configured in `.env` — genuinely calls the live API with
    a trivial prompt and a trivial structured schema. Skipped (not faked) otherwise, per
    prompt.md §0.24/§20 — this repo never claims a real API call happened when it didn't."""
    from src.common.config import load_secrets

    client = GoogleClient.from_settings(SETTINGS, load_secrets())
    response = client.complete("Reply with exactly the word: pong")
    assert "pong" in response.text.lower()

    structured = client.complete_structured("Echo back the word 'ping' and the count 1.", Echo)
    assert structured.message
    assert structured.count == 1
