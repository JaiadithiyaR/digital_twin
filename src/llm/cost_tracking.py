"""LLM cost tracking + budget guard, plus `MeteredGoogleClient` — a thin metering wrapper around
`GoogleClient` used only by experiment-running scripts, never by the agents (which stay on the
plain client; `GoogleClient` itself has no cost/budget logic — this module is where that lives).

Never calls the LLM itself — purely bookkeeping over the real `LLMResponse.usage`/
`.latency_seconds`/`.used_cache` a call already returns.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.llm.google_client import LLMClientError

if TYPE_CHECKING:
    from src.common.config import LlmRatesConfig
    from src.llm.google_client import GoogleClient, LLMResponse

logger = logging.getLogger(__name__)


def estimate_cost_usd(input_tokens: int, output_tokens: int, cached_tokens: int, rates: "LlmRatesConfig") -> float:
    """Cost from real post-call token counts + config rates. `cached_tokens` is a subset of
    `input_tokens` (billed at the cheaper rate instead of the full input rate) — never
    double-counted."""
    uncached_input = max(0, input_tokens - cached_tokens)
    return (
        uncached_input * rates.input_per_million
        + cached_tokens * rates.cached_input_per_million
        + output_tokens * rates.output_per_million
    ) / 1_000_000.0


def rough_token_estimate(text: str) -> int:
    """Crude pre-call estimate (~4 chars/token) used only by `BudgetGuard.check_and_estimate`
    before the real token count is known. Every recorded/reported cost uses the real post-call
    usage via `estimate_cost_usd`, never this estimate."""
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class LLMCallRecord:
    """One row of `llm_calls.jsonl` — phase13 §3.5's exact field list."""

    mode: str
    run: str
    role: str
    model: str
    service_tier: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    thinking_tokens: int
    latency_seconds: float
    cost_usd: float
    used_cache: bool
    timestamp: str


def append_llm_call_record(path: Path, record: LLMCallRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")


class BudgetExceededError(Exception):
    """Raised when even the estimated cost of the next call would exceed budget. Deliberately
    NOT a subclass of `LLMClientError` — the agents' `*_safe` wrappers only catch
    `LLMClientError`, so this propagates uncaught to the experiment-run loop, never silently
    absorbed into a fallback strategy."""


class BudgetGuard:
    """Tracks cumulative real spend across one experiment run. Thread-safe."""

    def __init__(self, budget_usd: float) -> None:
        self._budget_usd = budget_usd
        self._spent_usd = 0.0
        self._lock = threading.Lock()

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent_usd

    @property
    def budget_usd(self) -> float:
        return self._budget_usd

    def check_and_estimate(self, estimated_cost_usd: float) -> None:
        with self._lock:
            projected = self._spent_usd + estimated_cost_usd
            if projected > self._budget_usd:
                raise BudgetExceededError(
                    f"estimated cost ${estimated_cost_usd:.4f} would bring total spend to "
                    f"${projected:.4f}, exceeding the configured budget of ${self._budget_usd:.2f} "
                    f"(already spent ${self._spent_usd:.4f})"
                )

    def record_actual_spend(self, actual_cost_usd: float) -> None:
        with self._lock:
            self._spent_usd += actual_cost_usd


class MeteredGoogleClient:
    """Wraps `GoogleClient`: (1) checks `BudgetGuard` before the call with a rough pre-call
    estimate, (2) computes real cost from real post-call usage, (3) records it to both
    `BudgetGuard` and `llm_calls.jsonl`. Cache methods pass through unmetered. Used only by
    experiment-running scripts, never by an agent directly."""

    def __init__(
        self, client: "GoogleClient", *, mode: str, run: str, budget_guard: BudgetGuard, calls_log_path: Path, rates: "LlmRatesConfig"
    ) -> None:
        self._client = client
        self._mode = mode
        self._run = run
        self._guard = budget_guard
        self._calls_log_path = calls_log_path
        self._rates = rates

    def __getattr__(self, name: str) -> Any:
        # Pass through anything not explicitly metered below (start_cache/end_cache/
        # end_all_caches, and any future GoogleClient method) unchanged.
        return getattr(self._client, name)

    def _meter(self, role: str, prompt_for_estimate: str, call: Any) -> Any:
        estimated = estimate_cost_usd(rough_token_estimate(prompt_for_estimate), 0, 0, self._rates)
        self._guard.check_and_estimate(estimated)
        response: "LLMResponse" = call()
        actual_cost = estimate_cost_usd(
            response.usage.input_tokens, response.usage.output_tokens, response.usage.cached_tokens, self._rates
        )
        self._guard.record_actual_spend(actual_cost)
        append_llm_call_record(
            self._calls_log_path,
            LLMCallRecord(
                mode=self._mode, run=self._run, role=role, model=response.model,
                service_tier=self._client._config.service_tier,  # noqa: SLF001 - read-only, this wrapper's own concern
                input_tokens=response.usage.input_tokens, cached_tokens=response.usage.cached_tokens,
                output_tokens=response.usage.output_tokens, thinking_tokens=response.usage.thinking_tokens,
                latency_seconds=response.latency_seconds, cost_usd=actual_cost, used_cache=response.used_cache,
                timestamp=datetime.now(UTC).isoformat(),
            ),
        )
        return response

    def complete(self, prompt: str, *, role: str = "unspecified", **kwargs: Any) -> "LLMResponse":
        return self._meter(role, prompt, lambda: self._client.complete(prompt, **kwargs))

    def complete_structured(self, prompt: str, schema: Any, *, role: str = "unspecified", **kwargs: Any) -> Any:
        """Meters using REAL post-call usage via `GoogleClient.complete_structured_with_usage()`
        — exactly like `complete()` above, never the rough pre-call estimate for the RECORDED
        cost (that estimate is only ever used for the pre-call budget check)."""
        estimated = estimate_cost_usd(rough_token_estimate(prompt), 0, 0, self._rates)
        self._guard.check_and_estimate(estimated)
        parsed, response = self._client.complete_structured_with_usage(prompt, schema, **kwargs)
        actual_cost = estimate_cost_usd(
            response.usage.input_tokens, response.usage.output_tokens, response.usage.cached_tokens, self._rates
        )
        self._guard.record_actual_spend(actual_cost)
        append_llm_call_record(
            self._calls_log_path,
            LLMCallRecord(
                mode=self._mode, run=self._run, role=role, model=response.model,
                service_tier=self._client._config.service_tier,  # noqa: SLF001 - read-only, this wrapper's own concern
                input_tokens=response.usage.input_tokens, cached_tokens=response.usage.cached_tokens,
                output_tokens=response.usage.output_tokens, thinking_tokens=response.usage.thinking_tokens,
                latency_seconds=response.latency_seconds, cost_usd=actual_cost, used_cache=response.used_cache,
                timestamp=datetime.now(UTC).isoformat(),
            ),
        )
        return parsed


class RoleTaggedClient:
    """Wraps a `MeteredGoogleClient` with ONE fixed `role` baked in, so an agent that has no idea
    metering/roles exist (it only ever calls `complete`/`complete_structured` with `GoogleClient`'s
    own real keyword arguments — never a `role=` kwarg) still gets correctly role-tagged entries
    in `llm_calls.jsonl` (phase13 §3.5 — "log every LLM call ... mode/run/role"). Used by
    `baselines/run_all.py`: one `MeteredGoogleClient` per experiment run, wrapped three times
    (decision/codegen/explanation) around the SAME underlying budget guard/calls log/cache, and
    handed to each role's own agent constructor in place of a plain `GoogleClient`."""

    def __init__(self, metered_client: "MeteredGoogleClient", role: str) -> None:
        self._metered_client = metered_client
        self._role = role

    def complete(self, prompt: str, **kwargs: Any) -> "LLMResponse":
        return self._metered_client.complete(prompt, role=self._role, **kwargs)

    def complete_structured(self, prompt: str, schema: Any, **kwargs: Any) -> Any:
        return self._metered_client.complete_structured(prompt, schema, role=self._role, **kwargs)

    def complete_safe(self, *args: Any, **kwargs: Any) -> "LLMResponse | None":
        try:
            return self.complete(*args, **kwargs)
        except LLMClientError:
            logger.warning("LLM completion failed — degrading gracefully, caller receives None", extra={"component": "llm", "role": self._role})
            return None

    def complete_structured_safe(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self.complete_structured(*args, **kwargs)
        except LLMClientError:
            logger.warning("LLM structured completion failed — degrading gracefully, caller receives None", extra={"component": "llm", "role": self._role})
            return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._metered_client, name)
