"""Unit tests for phase13 §3.5/§3.6: LLM cost estimation, the budget guard, and
`MeteredGoogleClient`'s metering (never the real network — a hand-built fake `GoogleClient`
stands in, matching this project's own established "mock the transport, run everything
downstream for real" discipline)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.common.config import load_settings
from src.llm.cost_tracking import (
    BudgetExceededError,
    BudgetGuard,
    MeteredGoogleClient,
    append_llm_call_record,
    estimate_cost_usd,
    rough_token_estimate,
)
from src.llm.google_client import LLMClientError, LLMResponse, LLMUsage

SETTINGS = load_settings()
RATES = SETTINGS.llm.rates


def test_estimate_cost_usd_hand_computed():
    # 1,000,000 uncached input tokens @ $0.10/M + 1,000,000 output tokens @ $0.40/M = $0.50
    cost = estimate_cost_usd(1_000_000, 1_000_000, 0, RATES)
    assert cost == pytest.approx(0.10 + 0.40)


def test_estimate_cost_usd_cached_tokens_billed_at_the_cheaper_rate_not_double_counted():
    # all 1,000,000 input tokens are cached -> billed at cached_input_per_million only
    cost = estimate_cost_usd(1_000_000, 0, 1_000_000, RATES)
    assert cost == pytest.approx(RATES.cached_input_per_million / 1_000_000 * 1_000_000)


def test_rough_token_estimate_is_a_crude_but_nonzero_positive_heuristic():
    assert rough_token_estimate("") >= 1
    assert rough_token_estimate("x" * 400) == 100


# --- BudgetGuard -----------------------------------------------------------------------------


def test_budget_guard_allows_a_call_within_budget():
    guard = BudgetGuard(budget_usd=10.0)
    guard.check_and_estimate(5.0)  # must not raise
    guard.record_actual_spend(5.0)
    assert guard.spent_usd == pytest.approx(5.0)


def test_budget_guard_raises_before_a_call_that_would_exceed_budget():
    guard = BudgetGuard(budget_usd=10.0)
    guard.record_actual_spend(9.0)
    with pytest.raises(BudgetExceededError, match="exceeding the configured budget"):
        guard.check_and_estimate(2.0)
    assert guard.spent_usd == pytest.approx(9.0)  # the rejected call's cost was never added


def test_budget_exceeded_error_is_not_an_llm_client_error():
    """Phase13 §3.6: "never silently fall back to the default strategy because of the budget."
    This only holds if BudgetExceededError can propagate past DecisionAgent.decide_safe()'s own
    `except LLMClientError` — i.e. it must NOT be an LLMClientError itself."""
    assert not issubclass(BudgetExceededError, LLMClientError)


# --- llm_calls.jsonl ---------------------------------------------------------------------------


def test_append_llm_call_record_writes_one_json_line(tmp_path):
    from src.llm.cost_tracking import LLMCallRecord

    path = tmp_path / "llm_calls.jsonl"
    record = LLMCallRecord(
        mode="full_system", run="run8-seed1", role="decision", model="gemini-3.6-flash", service_tier="standard",
        input_tokens=100, cached_tokens=20, output_tokens=50, thinking_tokens=10,
        latency_seconds=1.23, cost_usd=0.0001, used_cache=True, timestamp="2026-10-07T00:00:00+00:00",
    )
    append_llm_call_record(path, record)
    append_llm_call_record(path, record)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["role"] == "decision"
    assert parsed["used_cache"] is True


# --- MeteredGoogleClient ------------------------------------------------------------------------


@dataclass
class _FakeSchema:
    value: int

    @classmethod
    def model_validate_json(cls, text: str) -> "_FakeSchema":
        return cls(**json.loads(text))


class _FakeInnerClient:
    """Stands in for a real GoogleClient — never touches the network."""

    def __init__(self) -> None:
        self._config = SETTINGS.llm
        self.complete_calls: list[str] = []
        self.complete_structured_calls: list[str] = []

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        self.complete_calls.append(prompt)
        return LLMResponse(
            text="ok", model="gemini-3.6-flash", usage=LLMUsage(input_tokens=1000, output_tokens=200, cached_tokens=100, thinking_tokens=50),
            stop_reason="STOP", attempts=1, latency_seconds=0.5, used_cache=False,
        )

    def complete_structured_with_usage(self, prompt: str, schema, **kwargs):
        self.complete_structured_calls.append(prompt)
        parsed = schema(value=42) if callable(schema) else _FakeSchema(value=42)
        response = LLMResponse(
            text='{"value": 42}', model="gemini-3.6-flash",
            usage=LLMUsage(input_tokens=2000, output_tokens=100, cached_tokens=500, thinking_tokens=20),
            stop_reason="STOP", attempts=1, latency_seconds=0.8, used_cache=True,
        )
        return parsed, response


def _metered(tmp_path: Path, budget_usd: float = 10.0) -> tuple[MeteredGoogleClient, BudgetGuard, _FakeInnerClient]:
    inner = _FakeInnerClient()
    guard = BudgetGuard(budget_usd=budget_usd)
    metered = MeteredGoogleClient(
        inner, mode="full_system", run="run8-seed1", budget_guard=guard,
        calls_log_path=tmp_path / "llm_calls.jsonl", rates=RATES,
    )
    return metered, guard, inner


def test_metered_complete_records_real_usage_and_updates_budget(tmp_path):
    metered, guard, inner = _metered(tmp_path)
    response = metered.complete("hello", role="explanation")

    assert inner.complete_calls == ["hello"]
    assert response.usage.input_tokens == 1000
    expected_cost = estimate_cost_usd(1000, 200, 100, RATES)
    assert guard.spent_usd == pytest.approx(expected_cost)

    lines = (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["role"] == "explanation"
    assert row["input_tokens"] == 1000
    assert row["cached_tokens"] == 100


def test_metered_complete_structured_uses_real_usage_not_the_rough_estimate(tmp_path):
    metered, guard, inner = _metered(tmp_path)
    parsed = metered.complete_structured("a prompt", _FakeSchema, role="decision")

    assert parsed.value == 42
    assert inner.complete_structured_calls == ["a prompt"]
    expected_cost = estimate_cost_usd(2000, 100, 500, RATES)
    assert guard.spent_usd == pytest.approx(expected_cost)  # the REAL usage-based cost, not the rough pre-call estimate

    row = json.loads((tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["role"] == "decision"
    assert row["used_cache"] is True


def test_metered_client_raises_budget_exceeded_before_calling_the_real_client(tmp_path):
    metered, guard, inner = _metered(tmp_path, budget_usd=0.0)  # zero budget — even a tiny estimate exceeds it
    with pytest.raises(BudgetExceededError):
        metered.complete("a prompt long enough to have a nonzero rough token estimate" * 100, role="decision")
    assert inner.complete_calls == []  # the real call never happened


def test_metered_client_passes_through_unmetered_methods():
    inner = _FakeInnerClient()
    inner.start_cache = lambda *a, **k: True  # type: ignore[attr-defined]
    guard = BudgetGuard(budget_usd=10.0)
    metered = MeteredGoogleClient(inner, mode="m", run="r", budget_guard=guard, calls_log_path=Path("/tmp/x.jsonl"), rates=RATES)
    assert metered.start_cache("k", "prefix") is True
