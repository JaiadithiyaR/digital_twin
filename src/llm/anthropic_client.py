"""Centralized Anthropic API client (prompt.md §24, §42) — infrastructure only.

This is the ONE place in the codebase that talks to the Anthropic API. Modules 15 (Regeneration),
16 (Expand-Scope), and 17 (Agentic Verification) — plus Module 14 (Recalibration)'s optional
training-window reasoning and Module 19 (Lifecycle)'s human-readable explanations, per CLAUDE.md
§7 — will all call through `AnthropicClient`, never construct their own `anthropic.Anthropic()`
instance. No agent logic lives here: no prompt templates, no per-agent output schemas, no
business rules about when to call the LLM. Those belong to Modules 14-19 themselves, when built.

**Structure note** (prompt.md §42: "the exact structure is yours to optimize"): the suggested
`src/llm/{anthropic_client.py, prompts.py, schemas.py}` layout is collapsed into this one file
for now. `prompts.py` (agent-specific prompt templates) and per-agent output schemas naturally
belong with the agents that use them — Modules 14-19 — not this infrastructure layer, and
creating them now with no consumer would be speculative content this project's conventions
explicitly avoid. `LLMResponse`/`LLMUsage` below are the only "schema" this layer owns: generic
response metadata, not agent output shapes.

**A real, load-bearing discovery made while building this file**: the installed Anthropic API
version (verified directly against `anthropic-sdk-python`'s actual `MessageCreateParams` and
`OutputConfigParam` types, not assumed from older documentation) has NO `temperature`/`top_p`/
`top_k` sampling-randomness parameter anywhere in the Messages API — a real, deliberate change in
this SDK generation, not an oversight here. prompt.md §42's "deterministic/non-deterministic
settings where appropriate" requirement is therefore satisfied honestly rather than by inventing
a parameter that doesn't exist: `config.llm.effort` maps to `output_config.effort` (reasoning
depth: low/medium/high/xhigh/max), which is the closest real, configurable lever this API
exposes — it is NOT a determinism/sampling control and must never be documented as one to a
future reader of this file.

**Structured output uses the API's own native mechanism, not prompt-engineered JSON extraction**:
`complete_structured()` passes `output_config.format = {"type": "json_schema", "schema": ...}`
(a real feature of this SDK's `OutputConfigParam`/`JSONOutputFormatParam`), which constrains the
model's response to the given JSON schema server-side. The response is still independently
validated against the caller's pydantic schema on our side afterward (prompt.md §26 / CLAUDE.md
§7: "LLM output is always untrusted") — never trusted just because the API is expected to have
enforced it.

**Retry/backoff is entirely ours, not the SDK's**: the underlying `anthropic.Anthropic` client is
constructed with `max_retries=0`, deliberately disabling its built-in transport retry layer, so
that ALL retry/backoff behavior in this system is the explicit, testable code in
`_call_with_retry`/`complete_structured` below — not a second, invisible retry schedule
compounding with ours inside the SDK.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from src.common.config import LlmConfig, Secrets, Settings

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Anthropic exception types that represent a transient condition worth retrying (network hiccups,
# rate limiting, transient server-side overload/errors). Auth/permission/bad-request/not-found
# errors are NOT in this set — retrying an invalid request or bad credentials can never succeed,
# so those fail fast instead of burning through the retry budget.
_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
    anthropic.ServiceUnavailableError,
)

_NON_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.BadRequestError,
    anthropic.NotFoundError,
    anthropic.UnprocessableEntityError,
)

# Safety ceiling on our own exponential backoff — a fixed implementation constant (like Module
# 12's MK-MMD bandwidth multipliers), not a per-deployment tunable; `config.llm.
# retry_backoff_seconds` is the actual configurable base.
_BACKOFF_CAP_SECONDS = 30.0


class LLMClientError(Exception):
    """Base class for every error this client raises. Always the result of exhausting the
    configured retry budget, or a fail-fast non-retryable failure — never raised speculatively.
    Catch this (not a bare `Exception`) for graceful degradation; see `complete_safe`/
    `complete_structured_safe`, which do exactly that."""


class LLMTransportError(LLMClientError):
    """The underlying Anthropic API call failed — after this client's own retry budget was
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


def _extract_text(message: Any) -> str:
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise LLMTransportError("Anthropic response contained no text content block", retryable=False)


class AnthropicClient:
    """The centralized client. Construct once (e.g. at process startup via `from_settings`) and
    reuse — it holds no per-call mutable state."""

    def __init__(self, config: "LlmConfig", api_key: str) -> None:
        self._config = config
        self._client = anthropic.Anthropic(api_key=api_key, timeout=config.timeout_seconds, max_retries=0)

    @classmethod
    def from_settings(cls, settings: "Settings", secrets: "Secrets") -> "AnthropicClient":
        # `require_anthropic_key()` raises a clear RuntimeError itself if unset — never a
        # hardcoded key, never a silent empty-string fallback (prompt.md §24/§16).
        return cls(config=settings.llm, api_key=secrets.require_anthropic_key())

    # --- request construction ------------------------------------------------------------------

    def _build_kwargs(
        self,
        prompt: str,
        system: str | None,
        max_tokens: int | None,
        effort: str | None,
        output_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._config.model,  # never hardcoded here — always the configured model (prompt.md §24)
            "max_tokens": max_tokens if max_tokens is not None else self._config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            kwargs["system"] = system

        resolved_effort = effort if effort is not None else self._config.effort
        output_config: dict[str, Any] = {}
        if resolved_effort is not None:
            output_config["effort"] = resolved_effort
        if output_format is not None:
            output_config["format"] = output_format
        if output_config:
            kwargs["output_config"] = output_config
        return kwargs

    # --- transport + retry/backoff --------------------------------------------------------------

    def _sleep(self, seconds: float) -> None:  # pragma: no cover - trivial, overridden/mocked in tests
        time.sleep(seconds)

    def _backoff_seconds(self, attempt_index: int, retry_after_seconds: float | None) -> float:
        if retry_after_seconds is not None:
            return max(0.0, retry_after_seconds)
        return min(self._config.retry_backoff_seconds * (2**attempt_index), _BACKOFF_CAP_SECONDS)

    def _call_once(self, kwargs: dict[str, Any]) -> Any:
        try:
            return self._client.messages.create(**kwargs)
        except _NON_RETRYABLE_EXCEPTIONS as exc:
            raise LLMTransportError(f"non-retryable Anthropic API error: {exc}", retryable=False, original=exc) from exc
        except _RETRYABLE_EXCEPTIONS as exc:
            retry_after_seconds = None
            response = getattr(exc, "response", None)
            if response is not None:
                header = response.headers.get("retry-after")
                if header is not None:
                    try:
                        retry_after_seconds = float(header)
                    except ValueError:
                        pass
            raise LLMTransportError(
                f"retryable Anthropic API error: {exc}",
                retryable=True,
                original=exc,
                retry_after_seconds=retry_after_seconds,
            ) from exc
        except anthropic.AnthropicError as exc:
            # Any SDK error we haven't explicitly classified above — fail safe as non-retryable
            # rather than guessing it might be transient (prompt.md §61 "fail safely").
            raise LLMTransportError(f"unclassified Anthropic API error: {exc}", retryable=False, original=exc) from exc

    def _call_with_retry(self, kwargs: dict[str, Any]) -> tuple[Any, int]:
        max_attempts = self._config.max_retries + 1
        last_error: LLMTransportError | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                message = self._call_once(kwargs)
                if attempt > 1:
                    logger.info(
                        "Anthropic API call succeeded after retry",
                        extra={"component": "llm", "attempt": attempt},
                    )
                return message, attempt
            except LLMTransportError as exc:
                last_error = exc
                is_final_attempt = attempt == max_attempts
                if not exc.retryable or is_final_attempt:
                    logger.warning(
                        "Anthropic API call failed"
                        + (" (final attempt)" if is_final_attempt else " (non-retryable)"),
                        extra={"component": "llm", "attempt": attempt, "retryable": exc.retryable, "error": str(exc)},
                    )
                    raise
                delay = self._backoff_seconds(attempt - 1, exc.retry_after_seconds)
                logger.warning(
                    "Anthropic API call failed, retrying",
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
        kwargs = self._build_kwargs(prompt, system, max_tokens, effort, output_format=None)
        message, attempts = self._call_with_retry(kwargs)
        text = _extract_text(message)
        usage = LLMUsage(input_tokens=message.usage.input_tokens, output_tokens=message.usage.output_tokens)
        logger.info(
            "Anthropic completion received",
            extra={
                "component": "llm",
                "model": message.model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "attempts": attempts,
            },
        )
        return LLMResponse(text=text, model=message.model, usage=usage, stop_reason=message.stop_reason, attempts=attempts)

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

        Uses the API's own native `output_config.format` JSON-schema constraint (see module
        docstring) AND independently validates the result on our side — the latter is never
        skipped just because the former is expected to have already enforced it (CLAUDE.md §7:
        "LLM output is always untrusted"). If validation still fails, retries the WHOLE call
        (not just re-parsing) up to `config.llm.max_retries` times, since asking again is the
        only thing that can actually produce a different, valid response. Raises
        `LLMStructuredOutputError` if every attempt fails validation.
        """
        output_format = {"type": "json_schema", "schema": schema.model_json_schema()}
        max_attempts = self._config.max_retries + 1
        last_error: LLMStructuredOutputError | None = None

        for attempt in range(1, max_attempts + 1):
            kwargs = self._build_kwargs(prompt, system, max_tokens, effort, output_format)
            message, _ = self._call_with_retry(kwargs)
            text = _extract_text(message)
            try:
                parsed = schema.model_validate_json(text)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = LLMStructuredOutputError(
                    f"LLM response failed schema validation: {exc}", raw_text=text, original=exc
                )
                is_final_attempt = attempt == max_attempts
                logger.warning(
                    "LLM structured output failed schema validation"
                    + (" (final attempt)" if is_final_attempt else ", retrying"),
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
        halt an autonomous workflow (prompt.md §42 "LLM failures must not corrupt production";
        CLAUDE.md §7's example: "recalibration training-window reasoning (optional)"). Every use
        of the fallback is logged at WARNING, mirroring Module 13's PPO fallback discipline."""
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
