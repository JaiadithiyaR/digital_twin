"""Unit tests for the Decision & Root-Cause Analysis Agent's context builder (prompt.md §18,
src/adaptation/decision_context.py) — design-pivot replacement for the deleted rl_env.py."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.adaptation.decision_context import (
    NETWORK_STATE_FEATURES,
    PreviousOutcome,
    build_decision_context,
    retrieve_rag_context,
    summarize_network_state,
)
from src.fidelity.trigger import AdaptationTrigger
from src.rag.rag_kb import RagUnavailableError, RetrievedChunk


def _trigger(**overrides) -> AdaptationTrigger:
    base = dict(component="dl_bler", severity=0.42, timestamp=datetime.now(UTC), trigger_type="fidelity_degradation")
    base.update(overrides)
    return AdaptationTrigger(**base)


def _history_row(**overrides) -> dict:
    base = dict(
        ts_1s=datetime.now(UTC),
        dl_throughput=10.0,
        ul_throughput=2.0,
        dl_rlc_delay=5.0,
        dl_prb=30.0,
        ul_prb=5.0,
        dl_bler=0.05,
        ul_bler=0.02,
        requested_prb_ratio_lag1=50.0,
        ue_count_cell_lag1=6.0,
        unrelated_column="ignored",
    )
    base.update(overrides)
    return base


# --- summarize_network_state ------------------------------------------------------------------


def test_summarize_network_state_empty_history_returns_empty_dict():
    assert summarize_network_state(pd.DataFrame()) == {}


def test_summarize_network_state_computes_mean_of_known_features():
    df = pd.DataFrame([_history_row(dl_throughput=10.0), _history_row(dl_throughput=20.0)])
    result = summarize_network_state(df)
    assert result["dl_throughput"] == pytest.approx(15.0)
    assert "unrelated_column" not in result  # only the fixed, bounded feature set


def test_summarize_network_state_bounded_to_recent_rows():
    rows = [_history_row(dl_throughput=float(i)) for i in range(300)]
    for i, r in enumerate(rows):
        r["ts_1s"] = pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=i)
    df = pd.DataFrame(rows)
    result = summarize_network_state(df, recent_rows=10)
    # mean of the LAST 10 throughput values (290..299), not all 300
    assert result["dl_throughput"] == pytest.approx(sum(range(290, 300)) / 10)


def test_summarize_network_state_missing_columns_are_simply_excluded():
    df = pd.DataFrame([{"ts_1s": datetime.now(UTC), "dl_throughput": 5.0}])
    result = summarize_network_state(df)
    assert result == {"dl_throughput": 5.0}
    assert set(result).issubset(set(NETWORK_STATE_FEATURES))


# --- retrieve_rag_context --------------------------------------------------------------------


class _FakeRagKb:
    def __init__(self, chunks):
        self.is_available = True
        self._chunks = chunks
        self.queries = []

    def retrieve(self, query, *, category=None, top_k=None):
        self.queries.append(query)
        return self._chunks


class _UnavailableRagKb:
    is_available = False

    def retrieve(self, *a, **k):
        raise RagUnavailableError("store unavailable")


def test_retrieve_rag_context_none_knowledge_base_returns_empty_list():
    assert retrieve_rag_context(None, "dl_bler", "fidelity_degradation") == []


def test_retrieve_rag_context_unavailable_returns_empty_list_never_raises():
    assert retrieve_rag_context(_UnavailableRagKb(), "dl_bler", "fidelity_degradation") == []


def test_retrieve_rag_context_formats_real_chunks():
    chunk = RetrievedChunk(
        text="Relevant policy text.", category="policies", source="adaptation_policies.md",
        document_id="doc1", chunk_index=0, version="v1", distance=0.1,
    )
    kb = _FakeRagKb([chunk])
    result = retrieve_rag_context(kb, "dl_bler", "fidelity_degradation")
    assert len(result) == 1
    assert "policies/adaptation_policies.md" in result[0]
    assert "Relevant policy text." in result[0]
    assert "dl_bler" in kb.queries[0]  # the component is genuinely embedded in the retrieval query


# --- build_decision_context -------------------------------------------------------------------


def test_build_decision_context_assembles_every_required_field():
    trigger = _trigger(component="dl_prb", severity=0.7, trigger_type="external_drift")
    fidelity = {"dl_throughput": 0.8, "dl_prb": 0.2, "dl_bler": None}
    prev = PreviousOutcome(action="recalibrate", verification_result="REJECT", fidelity_before=0.5, fidelity_after=0.3)
    history = pd.DataFrame([_history_row()])

    context = build_decision_context(trigger, fidelity, unified_fidelity_score=0.45, previous_outcome=prev, history=history)

    assert context.affected_component == "dl_prb"
    assert context.trigger_type == "external_drift"
    assert context.severity == 0.7
    assert context.component_fidelity == fidelity
    assert context.unified_fidelity_score == 0.45
    assert context.previous_outcome == prev
    assert context.network_state_summary["dl_throughput"] == 10.0
    assert context.rag_context == []  # no knowledge base supplied


def test_build_decision_context_is_source_agnostic_across_trigger_types():
    # Same function, same shape of result, regardless of which canonical trigger source fired —
    # prompt.md §16a's "so Module 13 can consume either source uniformly."
    empty_prev = PreviousOutcome()
    history = pd.DataFrame()
    drift_ctx = build_decision_context(_trigger(trigger_type="external_drift"), {}, None, empty_prev, history)
    fidelity_ctx = build_decision_context(_trigger(trigger_type="fidelity_degradation"), {}, None, empty_prev, history)
    assert type(drift_ctx) is type(fidelity_ctx)
    assert drift_ctx.trigger_type != fidelity_ctx.trigger_type


def test_previous_outcome_defaults_indicate_no_prior_attempt():
    prev = PreviousOutcome()
    assert prev.action is None
    assert prev.verification_result is None
    assert prev.fidelity_before is None
    assert prev.fidelity_after is None
