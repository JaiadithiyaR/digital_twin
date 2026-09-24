"""Centralized Google AI (Gemini) API client (prompt.md §42, CLAUDE.md §7) — infrastructure only.

This is the ONE place in the codebase that talks to the Google AI (Gemini) API, via the official
`google-genai` SDK. Module 13 (Decision & Root-Cause Analysis), Module 15 (Regeneration), Module
16 (Expand-Scope), and Module 17 (Agentic Verification) — plus Module 14 (Recalibration)'s
optional training-window reasoning and Module 19 (Lifecycle)'s human-readable explanations — all
call through `GoogleClient`, never construct their own `google.genai.Client()` instance. No agent
logic lives here: no prompt templates, no per-agent output schemas, no business rules about when
to call the LLM. Those belong to the agents themselves.

**Design pivot note**: this file replaces `src/llm/anthropic_client.py`/`AnthropicClient` (see
CLAUDE.md §12's design-pivot notice) — the whole project moved LLM providers from Anthropic to
Google AI. The public surface below (`LLMResponse`, `LLMUsage`, `LLMClientError`,
`LLMTransportError`, `LLMStructuredOutputError`, `complete`/`complete_structured`/`complete_safe`/
`complete_structured_safe`) is kept **intentionally identical** to the old `AnthropicClient`'s, so
every consumer agent's own code barely changes — only the import and the concrete class name.

**A real, load-bearing discovery made while building this file** (verified directly against the
installed `google-genai` SDK's actual `types.GenerateContentConfig`/`errors` modules, not assumed
from documentation — the same discipline `anthropic_client.py`'s own docstring established):

1. **Unlike the Anthropic Messages API, `google-genai`'s `GenerateContentConfig` DOES expose real
   sampling-determinism controls** (`temperature`/`top_p`/`top_k`) — the opposite conclusion from
   the Anthropic client's own documented finding. `config.llm.temperature` is therefore a genuine
   determinism knob here, not a fictional one; `0.0` genuinely minimizes sampling randomness.
2. **The SDK's own transport retry defaults to "never retry"** unless `http_options.retry_options`
   is explicitly set (verified in `google.genai._api_client.retry_args`: "If None, the 'never
   retry' stop strategy will be used"). This client deliberately never sets `retry_options`, so —
   exactly like `AnthropicClient`'s explicit `max_retries=0` — there is exactly one retry/backoff
   schedule in this system: the explicit, testable code in `_call_with_retry` below.
3. **Exception shape is structurally different from Anthropic's**: `google.genai.errors` has no
   named exception per failure type (no `RateLimitError`/`AuthenticationError`/etc.) — only
   `APIError` (base), `ClientError` (any 4xx, with the real HTTP status on `.code`), and
   `ServerError` (any 5xx). Retry classification here is therefore done on `.code`, not on
   exception identity: every `ServerError` and a `ClientError` with `.code == 429` (rate limit)
   are retryable; every other `ClientError` (400 bad request, 401/403 auth, 404 not found, 422
   unprocessable) fails fast. Raw transport-level failures (`httpx.TimeoutException`,
   `httpx.ConnectError`) can also propagate uncaught from this SDK (verified in
   `google.genai._api_client`'s own `_HTTPX_TRANSIENT_EXC` tuple) and are treated as retryable
   too, alongside `ServerError`.
4. **`effort` maps onto `types.ThinkingConfig(thinking_level=...)`, clamped, not 1:1** — Gemini's
   `ThinkingLevel` enum only has `MINIMAL`/`LOW`/`MEDIUM`/`HIGH` (no `xhigh`/`max`), so
   `config.llm.effort` values `"xhigh"`/`"max"` both clamp down to `HIGH` — documented here so a
   future reader isn't surprised the two don't produce different behavior on this provider.
5. **`HttpOptions.timeout` is in MILLISECONDS**, not seconds — `config.llm.timeout_seconds` is
   converted (`* 1000`) when constructing the client, never passed through directly.

**Structured output uses the API's own native mechanism, not prompt-engineered JSON extraction**,
exactly like the Anthropic client did: `complete_structured()` sets
`response_mime_type="application/json"` + `response_json_schema=schema.model_json_schema()` (a
real field on `GenerateContentConfig` accepting a plain JSON Schema dict — verified against
`types.py`), which constrains the model's response server-side. The response is still
independently re-validated against the caller's pydantic schema afterward (CLAUDE.md §7: "LLM
output is always untrusted") — never trusted just because the API is expected to have enforced it.

**No real `GOOGLE_API_KEY`/`GEMINI_API_KEY` is configured in this development environment** (only
a placeholder — see `.env.example`). Every test in `tests/unit/test_google_client.py` therefore
drives this client against a mocked transport (real `google.genai.errors.APIError` subclasses,
real response object shapes) — never a live call. Live-API validation is explicitly still
pending; see CLAUDE.md §12 for the honest status note.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from src.common.config import LlmConfig, Secrets, Settings

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Raw transport-level exceptions that can propagate uncaught from google-genai's own httpx-based
# transport (verified against google.genai._api_client._HTTPX_TRANSIENT_EXC) — network hiccups,
# never a signal that the request itself was invalid, so always retryable.
_TRANSIENT_TRANSPORT_EXCEPTIONS: tuple[type[Exception], ...] = (httpx.TimeoutException, httpx.ConnectError)

# Effort -> Gemini ThinkingLevel, clamped (see module docstring point 4). `None` omits
# thinking_config entirely, letting the model's own default apply.
_EFFORT_TO_THINKING_LEVEL: dict[str, genai_types.ThinkingLevel] = {
    "low": genai_types.ThinkingLevel.LOW,
    "medium": genai_types.ThinkingLevel.MEDIUM,
    "high": genai_types.ThinkingLevel.HIGH,
    "xhigh": genai_types.ThinkingLevel.HIGH,  # clamped — Gemini has no level above HIGH
    "max": genai_types.ThinkingLevel.HIGH,  # clamped — Gemini has no level above HIGH
}

# Safety ceiling on our own exponential backoff — a fixed implementation constant, not a
# per-deployment tunable; `config.llm.retry_backoff_seconds` is the actual configurable base.
_BACKOFF_CAP_SECONDS = 30.0


class LLMClientError(Exception):
    """Base class for every error this client raises. Always the result of exhausting the
    configured retry budget, or a fail-fast non-retryable failure — never raised speculatively.
    Catch this (not a bare `Exception`) for graceful degradation; see `complete_safe`/
    `complete_structured_safe`, which do exactly that."""


class LLMTransportError(LLMClientError):
    """The underlying Google AI API call failed — after this client's own retry budget was
    exhausted, or immediately for a non-retryable failure (bad credentials/request)."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        original: Exception | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.original = original
        self.retry_after_seconds = retry_after_seconds


class LLMStructuredOutputError(LLMClientError):
    """The model's response could not be validated against the requested pydantic schema, even
    after retrying the call up to `config.llm.max_retries` times."""

    def __init__(self, message: str, *, raw_text: str, original: Exception | None = None) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.original = original


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: LLMUsage
    stop_reason: str | None
    attempts: int  # 1 = succeeded on the first try; >1 = succeeded after that many retries


def _extract_text(response: genai_types.GenerateContentResponse) -> str:
    text = response.text  # a real property on GenerateContentResponse — aggregates all text parts
    if not text:
        raise LLMTransportError("Google AI response contained no text content", retryable=False)
    return text


def _extract_stop_reason(response: genai_types.GenerateContentResponse) -> str | None:
    if not response.candidates:
        return None
    finish_reason = response.candidates[0].finish_reason
    return finish_reason.value if finish_reason is not None else None


def _extract_usage(response: genai_types.GenerateContentResponse) -> LLMUsage:
    usage = response.usage_metadata
    if usage is None:
        return LLMUsage(input_tokens=0, output_tokens=0)
    return LLMUsage(
        input_tokens=usage.prompt_token_count or 0,
        output_tokens=usage.candidates_token_count or 0,
    )


def _retry_after_from_response(response: Any) -> float | None:
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    header = headers.get("retry-after")
    if header is None:
        return None
    try:
        return float(header)
    except ValueError:
        return None


class GoogleClient:
    """The centralized client. Construct once (e.g. at process startup via `from_settings`) and
    reuse — it holds no per-call mutable state."""

    def __init__(self, config: "LlmConfig", api_key: str) -> None:
        self._config = config
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=int(config.timeout_seconds * 1000)),  # ms, not seconds
        )

    @classmethod
    def from_settings(cls, settings: "Settings", secrets: "Secrets") -> "GoogleClient":
        # `require_google_key()` raises a clear RuntimeError itself if unset — never a hardcoded
        # key, never a silent empty-string fallback (prompt.md §24/§16).
        return cls(config=settings.llm, api_key=secrets.require_google_key())

    # --- request construction ------------------------------------------------------------------

    def _build_config(
        self,
        system: str | None,
        max_tokens: int | None,
        effort: str | None,
        output_format: dict[str, Any] | None,
    ) -> genai_types.GenerateContentConfig:
        kwargs: dict[str, Any] = {
            "max_output_tokens": max_tokens if max_tokens is not None else self._config.max_tokens,
        }
        if system is not None:
            kwargs["system_instruction"] = system
        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature

        resolved_effort = effort if effort is not None else self._config.effort
        if resolved_effort is not None:
            thinking_level = _EFFORT_TO_THINKING_LEVEL[resolved_effort]
            kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_level=thinking_level)

        if output_format is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = output_format

        return genai_types.GenerateContentConfig(**kwargs)

    # --- transport + retry/backoff --------------------------------------------------------------

    def _sleep(self, seconds: float) -> None:  # pragma: no cover - trivial, overridden/mocked in tests
        time.sleep(seconds)

    def _backoff_seconds(self, attempt_index: int, retry_after_seconds: float | None) -> float:
        if retry_after_seconds is not None:
            return max(0.0, retry_after_seconds)
        return min(self._config.retry_backoff_seconds * (2**attempt_index), _BACKOFF_CAP_SECONDS)

    def _call_once(self, prompt: str, config: genai_types.GenerateContentConfig) -> genai_types.GenerateContentResponse:
        try:
            return self._client.models.generate_content(model=self._config.model, contents=prompt, config=config)
        except _TRANSIENT_TRANSPORT_EXCEPTIONS as exc:
            raise LLMTransportError(f"transient transport error calling Google AI: {exc}", retryable=True, original=exc) from exc
        except genai_errors.ServerError as exc:
            raise LLMTransportError(
                f"retryable Google AI server error ({exc.code}): {exc}",
                retryable=True,
                original=exc,
                retry_after_seconds=_retry_after_from_response(exc.response),
            ) from exc
        except genai_errors.ClientError as exc:
            if exc.code == 429:  # rate limit — the one 4xx worth retrying
                raise LLMTransportError(
                    f"rate-limited by Google AI ({exc.code}): {exc}",
                    retryable=True,
                    original=exc,
                    retry_after_seconds=_retry_after_from_response(exc.response),
                ) from exc
            raise LLMTransportError(f"non-retryable Google AI client error ({exc.code}): {exc}", retryable=False, original=exc) from exc
        except genai_errors.APIError as exc:
            # Any SDK error we haven't explicitly classified above — fail safe as non-retryable
            # rather than guessing it might be transient (prompt.md §61 "fail safely").
            raise LLMTransportError(f"unclassified Google AI API error: {exc}", retryable=False, original=exc) from exc

    def _call_with_retry(self, prompt: str, config: genai_types.GenerateContentConfig) -> tuple[genai_types.GenerateContentResponse, int]:
        max_attempts = self._config.max_retries + 1
        last_error: LLMTransportError | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._call_once(prompt, config)
                if attempt > 1:
                    logger.info("Google AI API call succeeded after retry", extra={"component": "llm", "attempt": attempt})
                return response, attempt
            except LLMTransportError as exc:
                last_error = exc
                is_final_attempt = attempt == max_attempts
                if not exc.retryable or is_final_attempt:
                    logger.warning(
                        "Google AI API call failed" + (" (final attempt)" if is_final_attempt else " (non-retryable)"),
                        extra={"component": "llm", "attempt": attempt, "retryable": exc.retryable, "error": str(exc)},
                    )
                    raise
                delay = self._backoff_seconds(attempt - 1, exc.retry_after_seconds)
                logger.warning(
                    "Google AI API call failed, retrying",
                    extra={
                        "component": "llm",
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "delay_seconds": delay,
                        "error": str(exc),
                    },
                )
                self._sleep(delay)
        raise last_error  # pragma: no cover - loop above always returns or raises

    # --- public API ------------------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> LLMResponse:
        """Free-form text completion. Raises `LLMTransportError` on failure — see `complete_safe`
        for a graceful-degradation wrapper."""
        config = self._build_config(system, max_tokens, effort, output_format=None)
        response, attempts = self._call_with_retry(prompt, config)
        text = _extract_text(response)
        usage = _extract_usage(response)
        model = response.model_version or self._config.model
        logger.info(
            "Google AI completion received",
            extra={
                "component": "llm",
                "model": model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "attempts": attempts,
            },
        )
        return LLMResponse(text=text, model=model, usage=usage, stop_reason=_extract_stop_reason(response), attempts=attempts)

    def complete_structured(
        self,
        prompt: str,
        schema: type[SchemaT],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> SchemaT:
        """Structured completion, validated against `schema` (a pydantic `BaseModel` subclass).

        Uses the API's own native `response_json_schema` JSON-schema constraint (see module
        docstring) AND independently validates the result on our side — the latter is never
        skipped just because the former is expected to have already enforced it (CLAUDE.md §7:
        "LLM output is always untrusted"). If validation still fails, retries the WHOLE call
        (not just re-parsing) up to `config.llm.max_retries` times, since asking again is the
        only thing that can actually produce a different, valid response. Raises
        `LLMStructuredOutputError` if every attempt fails validation.
        """
        output_format = schema.model_json_schema()
        max_attempts = self._config.max_retries + 1
        last_error: LLMStructuredOutputError | None = None

        for attempt in range(1, max_attempts + 1):
            config = self._build_config(system, max_tokens, effort, output_format)
            response, _ = self._call_with_retry(prompt, config)
            text = _extract_text(response)
            try:
                parsed = schema.model_validate_json(text)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = LLMStructuredOutputError(
                    f"LLM response failed schema validation: {exc}", raw_text=text, original=exc
                )
                is_final_attempt = attempt == max_attempts
                logger.warning(
                    "LLM structured output failed schema validation" + (" (final attempt)" if is_final_attempt else ", retrying"),
                    extra={
                        "component": "llm",
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "schema": schema.__name__,
                        "error": str(exc),
                    },
                )
                continue
            logger.info(
                "LLM structured output validated",
                extra={"component": "llm", "schema": schema.__name__, "attempt": attempt},
            )
            return parsed

        raise last_error  # noqa: RSE102 - last_error is always set by the loop above on this path

    def complete_safe(self, *args: Any, **kwargs: Any) -> LLMResponse | None:
        """`complete()`, degrading gracefully to `None` on any `LLMClientError` instead of
        raising — for callers where an LLM failure should skip optional reasoning rather than
        halt an autonomous workflow (prompt.md §42 "LLM failures must not corrupt production").
        Every use of the fallback is logged at WARNING."""
        try:
            return self.complete(*args, **kwargs)
        except LLMClientError as exc:
            logger.warning(
                "LLM completion failed — degrading gracefully, caller receives None",
                extra={"component": "llm", "error": str(exc)},
            )
            return None

    def complete_structured_safe(self, *args: Any, **kwargs: Any) -> SchemaT | None:
        """`complete_structured()`, degrading gracefully to `None` on any `LLMClientError`
        instead of raising. See `complete_safe`'s docstring — same rule applies."""
        try:
            return self.complete_structured(*args, **kwargs)
        except LLMClientError as exc:
            logger.warning(
                "LLM structured completion failed — degrading gracefully, caller receives None",
                extra={"component": "llm", "error": str(exc)},
            )
            return None
