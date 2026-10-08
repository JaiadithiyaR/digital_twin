"""Unit tests for the Decision & Root-Cause Analysis Agent (Module 13, prompt.md §18-21,
src/adaptation/decision_agent.py) — design-pivot replacement for the deleted PPO `rl_agent.py`.

No real GOOGLE_API_KEY is configured in this environment (same situation as every other
LLM-driven agent's tests in this repo) — `_FakeLLMClient` stands in for `GoogleClient`,
implementing only `complete_structured` (the one method this agent calls), exactly the pattern
`tests/unit/test_verification_agent.py`/`test_regeneration_agent.py` already established.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from src.adaptation.decision_agent import _SYSTEM_PROMPT, DecisionAgent, DecisionAgentError, DecisionOutput, _build_prompt
from src.adaptation.decision_context import DecisionContext, PreviousOutcome
from src.common.config import load_settings
from src.fidelity.trigger import AdaptationTrigger
from src.llm.google_client import LLMTransportError

SETTINGS = load_settings()


def _context(**overrides) -> DecisionContext:
    base = dict(
        component_fidelity={"dl_throughput": 0.8, "dl_bler": 0.2},
        unified_fidelity_score=0.5,
        affected_component="dl_bler",
        trigger_type="fidelity_degradation",
        severity=0.6,
        previous_outcome=PreviousOutcome(),
        rag_context=[],
        network_state_summary={"throughput_mbps": 12.0},
    )
    base.update(overrides)
    return DecisionContext(**base)


class _FakeLLMClient:
    """Stands in for `GoogleClient` — only `complete_structured` is exercised by this agent."""

    def __init__(self, output: DecisionOutput | None = None, raise_on_call: Exception | None = None):
        self._output = output
        self._raise_on_call = raise_on_call
        self.calls: list[tuple[str, str | None]] = []
        self.last_cache_key: str | None = None

    def complete_structured(self, prompt, schema, *, system=None, cache_key=None, **kwargs):
        self.calls.append((prompt, system))
        self.last_cache_key = cache_key
        if self._raise_on_call is not None:
            raise self._raise_on_call
        assert schema is DecisionOutput
        return self._output


def _output(**overrides) -> DecisionOutput:
    base = dict(
        strategy="regenerate",
        root_cause_analysis="Structural drift detected in dl_bler predictions.",
        confidence=0.85,
        rationale="Recalibration alone would not fix a structural pipeline mismatch.",
        knowledge_refs=["policies/adaptation_policies.md"],
    )
    base.update(overrides)
    return DecisionOutput(**base)


# --- prompt construction -------------------------------------------------------------------


def test_prompt_includes_every_required_context_element():
    context = _context(
        previous_outcome=PreviousOutcome(action="recalibrate", verification_result="REJECT", fidelity_before=0.9, fidelity_after=0.4),
        rag_context=["[policies/x.md] some retrieved text"],
    )
    prompt = _build_prompt(context)

    assert "dl_bler" in prompt  # affected_component
    assert "fidelity_degradation" in prompt  # trigger_type
    assert "0.6000" in prompt  # severity
    assert "0.8000" in prompt  # a per-component fidelity value
    assert "0.5000" in prompt  # unified fidelity score
    assert "recalibrate" in prompt  # previous action
    assert "REJECT" in prompt  # previous verification result
    assert "12.0000" in prompt  # network state summary value
    assert "some retrieved text" in prompt  # RAG context
    assert "BEGIN RETRIEVED KNOWLEDGE" in prompt and "END RETRIEVED KNOWLEDGE" in prompt  # explicit delimiters (prompt.md §44)


def test_prompt_handles_no_previous_attempt_and_no_rag_context_honestly():
    prompt = _build_prompt(_context(previous_outcome=PreviousOutcome(), rag_context=[]))
    assert "no previous adaptation attempt" in prompt
    assert "no relevant knowledge retrieved" in prompt


def test_prompt_never_fabricates_insufficient_history_scores():
    context = _context(component_fidelity={"dl_throughput": None}, unified_fidelity_score=None)
    prompt = _build_prompt(context)
    assert "insufficient_history" in prompt


# --- decide() — genuine LLM call, never cached/hardcoded ------------------------------------


def test_decide_calls_the_llm_and_returns_its_genuine_structured_output():
    fake_client = _FakeLLMClient(output=_output(strategy="expand_scope"))
    agent = DecisionAgent(fake_client, SETTINGS)

    result = agent.decide(_context())

    assert result.strategy == "expand_scope"
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0][1] is not None  # a system prompt was genuinely sent


def test_decide_sends_a_fresh_call_per_trigger_never_caches():
    fake_client = _FakeLLMClient(output=_output())
    agent = DecisionAgent(fake_client, SETTINGS)

    agent.decide(_context(affected_component="dl_throughput"))
    agent.decide(_context(affected_component="dl_rlc_delay"))

    assert len(fake_client.calls) == 2
    assert "dl_throughput" in fake_client.calls[0][0]
    assert "dl_rlc_delay" in fake_client.calls[1][0]


def test_decide_raises_on_llm_failure_without_fallback():
    fake_client = _FakeLLMClient(raise_on_call=LLMTransportError("boom", retryable=False))
    agent = DecisionAgent(fake_client, SETTINGS)

    with pytest.raises(LLMTransportError):
        agent.decide(_context())


# --- decide_safe() — the ONE permitted deterministic fallback --------------------------------


def test_decide_safe_returns_genuine_result_on_success():
    fake_client = _FakeLLMClient(output=_output(strategy="recalibrate"))
    agent = DecisionAgent(fake_client, SETTINGS)

    result = agent.decide_safe(_context())

    assert result.strategy == "recalibrate"
    assert result.confidence == 0.85  # the genuine LLM-produced value, not the fallback's 0.0


def test_decide_safe_degrades_to_configured_default_strategy_on_llm_failure(caplog):
    fake_client = _FakeLLMClient(raise_on_call=LLMTransportError("boom", retryable=False))
    agent = DecisionAgent(fake_client, SETTINGS)

    with caplog.at_level(logging.WARNING):
        result = agent.decide_safe(_context())

    assert result.strategy == SETTINGS.decision_agent.fallback.default_strategy
    assert result.confidence == 0.0  # honestly reflects "no genuine analysis was performed"
    assert any("infrastructure failure" in r.message for r in caplog.records)


def test_decide_safe_raises_decision_agent_error_when_fallback_disabled():
    settings_no_fallback = SETTINGS.model_copy(
        update={"decision_agent": SETTINGS.decision_agent.model_copy(update={"fallback": SETTINGS.decision_agent.fallback.model_copy(update={"enabled": False})})}
    )
    fake_client = _FakeLLMClient(raise_on_call=LLMTransportError("boom", retryable=False))
    agent = DecisionAgent(fake_client, settings_no_fallback)

    with pytest.raises(DecisionAgentError):
        agent.decide_safe(_context())


def test_decide_raises_immediately_when_no_client_configured():
    agent = DecisionAgent(None, SETTINGS)
    with pytest.raises(LLMTransportError):
        agent.decide(_context())


def test_decide_safe_falls_back_when_no_client_configured(caplog):
    agent = DecisionAgent(None, SETTINGS)
    with caplog.at_level(logging.WARNING):
        result = agent.decide_safe(_context())
    assert result.strategy == SETTINGS.decision_agent.fallback.default_strategy
    assert result.confidence == 0.0


def test_fallback_strategy_is_within_the_fixed_enum():
    # The fallback must itself never produce a strategy outside prompt.md §19's fixed set —
    # enforced here by construction (DecisionAgentFallbackConfig.default_strategy is itself a
    # Literal["recalibrate","regenerate","expand_scope"] in src/common/config.py).
    assert SETTINGS.decision_agent.fallback.default_strategy in ("recalibrate", "regenerate", "expand_scope")


# --- Phase 13 §3.3: shared context-cache prefix support -----------------------------------------


def test_static_prefix_none_leaves_prompt_and_system_unchanged():
    """Default (no static_prefix, no cache_key) path must be byte-for-byte what every existing
    caller (e.g. src/main.py) already relies on — unchanged by this feature's addition."""
    fake_client = _FakeLLMClient(output=_output())
    agent = DecisionAgent(fake_client, SETTINGS)
    context = _context()
    agent.decide(context)
    (sent_prompt, sent_system), = fake_client.calls
    assert sent_prompt == _build_prompt(context)
    assert sent_system == _SYSTEM_PROMPT
    assert fake_client.last_cache_key is None


def test_static_prefix_without_cache_key_prepends_prefix_and_drops_system_kwarg():
    fake_client = _FakeLLMClient(output=_output())
    agent = DecisionAgent(fake_client, SETTINGS, static_prefix="STATIC PREFIX TEXT")
    context = _context()
    agent.decide(context)
    (sent_prompt, sent_system), = fake_client.calls
    assert sent_system is None  # never duplicated via the SDK's own system_instruction field
    assert sent_prompt.startswith("STATIC PREFIX TEXT\n\n")
    assert _build_prompt(context) in sent_prompt


def test_static_prefix_with_active_cache_key_sends_dynamic_part_only():
    fake_client = _FakeLLMClient(output=_output())
    agent = DecisionAgent(fake_client, SETTINGS, static_prefix="STATIC PREFIX TEXT")
    context = _context()
    agent.decide(context, cache_key="run-1")
    (sent_prompt, sent_system), = fake_client.calls
    assert sent_system is None
    assert "STATIC PREFIX TEXT" not in sent_prompt  # already server-side via the cache
    assert _build_prompt(context) in sent_prompt
    assert fake_client.last_cache_key == "run-1"
