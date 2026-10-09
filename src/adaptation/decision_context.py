"""Decision Agent's context assembly — "what informs the decision", kept separate from "what the
decision is" (`decision_agent.py`). Builds a bounded, structured snapshot for the LLM: per-
component fidelity + unified score, affected component, trigger type/severity, previous
action+outcome for this component, read-only RAG retrieval, and a bounded network-state summary
(never unbounded raw telemetry). Builds this from already-computed inputs only — never computes
fidelity itself, never queries D1/RAG beyond what's asked here, never decides anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from src.fidelity.trigger import AdaptationTrigger
from src.rag.rag_kb import RagUnavailableError

if TYPE_CHECKING:
    from src.rag.rag_kb import RagKnowledgeBase

logger = logging.getLogger(__name__)

# Bounded, fixed set of real D1 columns summarized into the network-state context for the LLM
# prompt — the 7 raw KPI targets plus the controller's requested PRB ratio and UE/slice load.
NETWORK_STATE_FEATURES: tuple[str, ...] = (
    "dl_throughput",
    "ul_throughput",
    "dl_rlc_delay",
    "dl_prb",
    "ul_prb",
    "dl_bler",
    "ul_bler",
    "requested_prb_ratio_lag1",
    "ue_count_cell_lag1",
)

_NETWORK_STATE_RECENT_ROWS = 200  # bounded window — never dump unbounded raw telemetry into the prompt (prompt.md §18)
_RAG_TOP_K = 5


@dataclass(frozen=True)
class PreviousOutcome:
    """What was previously tried for this component/incident, and how it went — prompt.md §18's
    "previous action taken for this component/incident and its outcome, where available".
    `None` fields mean genuinely no prior attempt is on record (never fabricated)."""

    action: str | None = None
    verification_result: str | None = None  # "ACCEPT" | "REJECT" | None
    fidelity_before: float | None = None
    fidelity_after: float | None = None


@dataclass(frozen=True)
class DecisionContext:
    """The full, bounded snapshot handed to `decision_agent.py`'s LLM call. Frozen — a context is
    a point-in-time snapshot, never mutated after construction."""

    component_fidelity: dict[str, float | None]
    unified_fidelity_score: float | None
    affected_component: str
    trigger_type: str
    severity: float
    previous_outcome: PreviousOutcome
    rag_context: list[str]  # already-retrieved, human-readable knowledge snippets — may be empty
    network_state_summary: dict[str, float] = field(default_factory=dict)


def summarize_network_state(history: pd.DataFrame, recent_rows: int = _NETWORK_STATE_RECENT_ROWS) -> dict[str, float]:
    """Bounded, real D1-derived network-state summary — the mean of `NETWORK_STATE_FEATURES` over
    the most recent `recent_rows` rows of real D1 history. Empty/missing history yields an empty
    dict (never a fabricated summary) — the caller's prompt building must handle that explicitly.
    """
    if len(history) == 0:
        return {}
    present = [c for c in NETWORK_STATE_FEATURES if c in history.columns]
    if not present:
        return {}
    recent = history.sort_values("ts_1s", kind="stable").tail(recent_rows)
    means = recent[present].mean(numeric_only=True)
    return {col: float(means[col]) for col in present if pd.notna(means[col])}


def retrieve_rag_context(
    rag_knowledge_base: "RagKnowledgeBase | None", component: str, trigger_type: str
) -> list[str]:
    """Read-only RAG retrieval, gracefully degrading to an empty list — never fabricated content
    — exactly the pattern already established by `verification_agent.py`'s own RAG consultation
    (CLAUDE.md §7 / prompt.md §61: RAG unavailable -> continue without it, never invent a plausible
    answer)."""
    if rag_knowledge_base is None or not rag_knowledge_base.is_available:
        return []
    query = f"root cause analysis and adaptation strategy guidance for {trigger_type} affecting component {component}"
    try:
        chunks = rag_knowledge_base.retrieve(query, top_k=_RAG_TOP_K)
    except RagUnavailableError as exc:
        logger.warning("RAG retrieval unavailable for decision context", extra={"component": component, "error": str(exc)})
        return []
    return [f"[{c.category}/{c.source}] {c.text[:400]}" for c in chunks]


def build_decision_context(
    trigger: AdaptationTrigger,
    component_fidelity: dict[str, float | None],
    unified_fidelity_score: float | None,
    previous_outcome: PreviousOutcome,
    history: pd.DataFrame,
    rag_knowledge_base: "RagKnowledgeBase | None" = None,
) -> DecisionContext:
    """Assembles one `DecisionContext` from already-computed inputs. `trigger` may come from
    either canonical source (`trigger_type="external_drift"` or `"fidelity_degradation"` — see
    `src/fidelity/trigger.py`) — this function is source-agnostic by design, exactly per
    prompt.md §16a's "so Module 13 can consume either source uniformly"."""
    return DecisionContext(
        component_fidelity=dict(component_fidelity),
        unified_fidelity_score=unified_fidelity_score,
        affected_component=trigger.component,
        trigger_type=trigger.trigger_type,
        severity=trigger.severity,
        previous_outcome=previous_outcome,
        rag_context=retrieve_rag_context(rag_knowledge_base, trigger.component, trigger.trigger_type),
        network_state_summary=summarize_network_state(history),
    )
