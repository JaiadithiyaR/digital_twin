"""Module 19 — Lifecycle Management Agent (Agent 6; prompt.md §37-38).

The last of the six agents (CLAUDE.md §3): it does not decide anything — it RECORDS and EXPLAINS
what every other agent already decided. Every adaptation event (one drift notification -> one PPO
strategy choice -> one Module 14/15/16 agent execution -> one Module 17 verification decision)
must produce exactly one structured, auditable `LifecycleRecord`, plus an automatically-generated,
human-readable maintenance report.

**`LifecycleRecord` carries every field prompt.md §37 lists, no more and no fewer**: event ID,
timestamp, drift event, affected component/scope, drift severity, RL observation/context, RL
action, agent action, production version before, candidate version, fidelity before, fidelity
after, verification result, verification explanation, training window, evaluation window, model
metadata, LLM metadata if used, final status. Nothing here computes or re-derives any of these —
every field is read verbatim from the real object each upstream module already produced:

- `drift_event`/`affected_component`/`drift_severity` <- Module 11's `DriftEvent`.
- `rl_observation`/`rl_action` <- whatever observation vector and action name Module 13's
  `decide_adaptation_strategy()` was actually called with/returned.
- `agent_action` <- a small, generic summary of whichever of Module 14/15/16's result objects the
  caller passed in (`RecalibrationResult`/`RegenerationResult`/`ExpandScopeResult` — see
  `_summarize_agent_action()`; these three types deliberately share no common base class, so this
  function duck-types on the fields each one actually has rather than forcing an artificial
  shared interface onto agents that don't otherwise need one).
- `production_version_before` <- `candidate_version.parent_version_id`, i.e. exactly what Module
  14/15's candidate was recalibrated/regenerated FROM (or `None` for Module 16's genuinely new
  components) — this is already the correct value; no separate registry lookup is needed or
  performed (a fresh lookup taken AFTER verification would be wrong on ACCEPT, since the registry
  has already been updated by then).
- `candidate_version`/`fidelity_before`/`fidelity_after`/`verification_result`/
  `verification_explanation`/`final_status` <- Module 17's `VerificationResult` — the
  INDEPENDENTLY RECOMPUTED fidelity values, never an agent's own self-reported ones (Module 17 is
  the trusted authority for these; see its own module docstring).
- `training_window`/`evaluation_window`/`model_metadata`/`llm_metadata` <- the agent result's own
  windows, plus `VerificationResult.updated_version` (the ACTUAL post-promote/reject registry
  record — component, model_class, dependencies, feature_schema, output_field, adaptation_type,
  status, llm_metadata).

**Persistence is a genuinely append-only, auditable log** (prompt.md §37 "must be auditable"):
`config.lifecycle.records_path` is one JSON-Lines file — every `record_adaptation_event()` call
appends exactly one line, under a lock, never rewrites or deletes a prior line. This is the
"versioned adaptation record" requirement: each record is permanently identified by its own
`event_id` and `timestamp`, and — because every field embeds the exact component/candidate
VERSION IDs Modules 14-17 already produced — the full version lineage of any adaptation is
reconstructable from the log alone, without needing to separately version the log file itself.

**The maintenance report is generated automatically** (prompt.md §38) from the SAME already-
recorded `LifecycleRecord` — never from a re-query of live state, which could have moved on by
report-generation time. A deterministic template (`_deterministic_report()`) is ALWAYS produced
first and used as the report whenever no LLM client is supplied or the LLM call fails
(`complete_safe`'s graceful-degradation convention, same as every other optional-LLM path in this
codebase) — a maintenance report must exist for every event regardless of LLM availability. When
an `AnthropicClient` is supplied, it is asked to write better prose FROM the same deterministic
facts (never invited to invent new ones), with D2's `RagKnowledgeBase` consulted for "relevant
contextual knowledge" (prompt.md §38's own explicit requirement — "Use RAG retrieval where
useful"), gracefully degrading to "not available" exactly like Module 17's own RAG consultation
when no knowledge base is supplied or it's unavailable.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from src.llm.anthropic_client import LLMClientError
from src.rag.rag_kb import RagUnavailableError

if TYPE_CHECKING:
    import numpy as np

    from src.adaptation.expand_scope_agent import ExpandScopeResult
    from src.adaptation.recalibration_agent import RecalibrationResult
    from src.adaptation.regeneration_agent import RegenerationResult
    from src.adaptation.verification_agent import VerificationResult
    from src.common.config import Settings
    from src.drift.schema import DriftEvent
    from src.llm.anthropic_client import AnthropicClient
    from src.rag.rag_kb import RagKnowledgeBase

logger = logging.getLogger(__name__)


class LifecycleError(Exception):
    """Raised only for a usage/contract violation (e.g. requesting a record that doesn't exist)
    — never raised for anything about the adaptation event itself, which this agent only records."""


class LifecycleRecord(BaseModel):
    """Every field prompt.md §37 requires, verbatim from the real upstream objects — see this
    module's docstring for exactly where each one comes from. Immutable once created."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    timestamp: datetime
    drift_event: dict[str, Any]
    affected_component: str
    drift_severity: float
    rl_observation: list[float]
    rl_action: str
    agent_action: dict[str, Any]
    production_version_before: str | None
    candidate_version: str
    fidelity_before: float | None
    fidelity_after: float | None
    verification_result: Literal["ACCEPT", "REJECT"]
    verification_explanation: str
    training_window: dict[str, Any]
    evaluation_window: dict[str, Any]
    model_metadata: dict[str, Any]
    llm_metadata: dict[str, Any] | None
    final_status: Literal["promoted", "rejected"]


def _summarize_agent_action(agent_result: Any) -> dict[str, Any]:
    """A generic summary of whichever Module 14/15/16 result object was produced — duck-typed on
    the fields each one actually has (they deliberately share no common base class; see module
    docstring). Never touches anything but plain, already-computed data already on the object."""
    version = agent_result.version
    summary: dict[str, Any] = {
        "adaptation_type": version.adaptation_type,
        "component": version.component,
        "candidate_version_id": version.version_id,
        "model_class": version.model_class,
    }
    if hasattr(agent_result, "evaluation_metrics"):  # RecalibrationResult
        summary["evaluation_metrics"] = agent_result.evaluation_metrics
    if hasattr(agent_result, "sandbox_result"):  # RegenerationResult / ExpandScopeResult
        sandbox = agent_result.sandbox_result
        summary["sandbox_stage"] = sandbox.stage
        summary["sandbox_accepted"] = sandbox.accepted
        summary["sandbox_metrics"] = sandbox.metrics
    if hasattr(agent_result, "attempts"):  # RegenerationResult
        summary["attempts"] = agent_result.attempts
    if hasattr(agent_result, "design_attempts"):  # ExpandScopeResult
        summary["design_attempts"] = agent_result.design_attempts
        summary["implementation_attempts"] = agent_result.implementation_attempts
    if hasattr(agent_result, "design"):  # ExpandScopeResult
        design = agent_result.design
        summary["proposed_design"] = {
            "component_name": design.component_name,
            "target_column": design.target_column,
            "dependencies": list(design.dependencies),
            "purpose": design.purpose,
        }
    return summary


@dataclass(frozen=True)
class MaintenanceReport:
    event_id: str
    text: str
    llm_used: bool
    path: Path


class LifecycleAgent:
    def __init__(
        self,
        records_path: Path,
        reports_dir: Path,
        llm_client: "AnthropicClient | None" = None,
    ) -> None:
        self._records_path = records_path
        self._reports_dir = reports_dir
        self._llm_client = llm_client
        self._lock = threading.Lock()
        self._records_path.parent.mkdir(parents=True, exist_ok=True)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(cls, settings: "Settings", llm_client: "AnthropicClient | None" = None) -> "LifecycleAgent":
        cfg = settings.lifecycle
        return cls(
            records_path=settings.resolve_path(cfg.records_path),
            reports_dir=settings.resolve_path(cfg.reports_dir),
            llm_client=llm_client,
        )

    # --- recording -----------------------------------------------------------------------------

    def record_adaptation_event(
        self,
        *,
        drift_event: "DriftEvent",
        rl_observation: "np.ndarray | list[float]",
        rl_action: str,
        agent_result: "RecalibrationResult | RegenerationResult | ExpandScopeResult",
        verification_result: "VerificationResult",
    ) -> LifecycleRecord:
        """Build and durably append exactly one `LifecycleRecord` for one full adaptation event.
        Every argument is a real object a real upstream module (11/13/14-16/17) already produced —
        this method computes nothing beyond the field-by-field mapping documented in the module
        docstring above."""
        candidate_version_meta = agent_result.version
        updated_version = verification_result.updated_version

        observation_list = [float(v) for v in rl_observation]

        record = LifecycleRecord(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC),
            drift_event=json.loads(drift_event.model_dump_json()),
            affected_component=drift_event.component,
            drift_severity=drift_event.severity,
            rl_observation=observation_list,
            rl_action=rl_action,
            agent_action=_summarize_agent_action(agent_result),
            production_version_before=candidate_version_meta.parent_version_id,
            candidate_version=verification_result.candidate_version_id,
            fidelity_before=verification_result.fidelity_before,
            fidelity_after=verification_result.fidelity_after,
            verification_result=verification_result.decision,
            verification_explanation=verification_result.explanation,
            training_window=agent_result.training_window,
            evaluation_window=agent_result.evaluation_window,
            model_metadata={
                "component": updated_version.component,
                "model_class": updated_version.model_class,
                "dependencies": list(updated_version.dependencies),
                "feature_schema": list(updated_version.feature_schema),
                "output_field": updated_version.output_field,
                "adaptation_type": updated_version.adaptation_type,
                "status": updated_version.status,
            },
            llm_metadata=updated_version.llm_metadata,
            final_status="promoted" if verification_result.decision == "ACCEPT" else "rejected",
        )
        self._append_record(record)
        logger.info(
            "lifecycle event recorded",
            extra={
                "component": "lifecycle_agent",
                "event_id": record.event_id,
                "affected_component": record.affected_component,
                "rl_action": record.rl_action,
                "final_status": record.final_status,
            },
        )
        return record

    def _append_record(self, record: LifecycleRecord) -> None:
        line = record.model_dump_json() + "\n"
        with self._lock:
            with self._records_path.open("a", encoding="utf-8") as f:
                f.write(line)

    # --- read path (auditability) ---------------------------------------------------------------

    def list_records(self) -> list[LifecycleRecord]:
        if not self._records_path.exists():
            return []
        with self._lock:
            lines = self._records_path.read_text(encoding="utf-8").splitlines()
        return [LifecycleRecord.model_validate_json(line) for line in lines if line.strip()]

    def get_record(self, event_id: str) -> LifecycleRecord:
        for record in self.list_records():
            if record.event_id == event_id:
                return record
        raise LifecycleError(f"no lifecycle record found with event_id={event_id!r}")

    # --- maintenance report (prompt.md §38) --------------------------------------------------

    def generate_maintenance_report(
        self, record: LifecycleRecord, *, rag_knowledge_base: "RagKnowledgeBase | None" = None
    ) -> MaintenanceReport:
        """Automatically produce a human-readable vendor maintenance report for `record` (prompt.md
        §38's exact list: what drifted, affected scope, why adaptation was triggered, what PPO
        selected, what the adaptation agent did, what changed, fidelity before/after,
        accepted/rejected + why, relevant contextual knowledge). Always saves the report to
        `config.lifecycle.reports_dir/<event_id>.md` and returns it."""
        deterministic_text = self._deterministic_report(record)
        rag_context = self._retrieve_rag_context(record, rag_knowledge_base)

        report_text = deterministic_text
        llm_used = False
        if self._llm_client is not None:
            prompt = self._report_prompt(record, deterministic_text, rag_context)
            response = self._llm_client.complete_safe(prompt)
            if response is not None:
                report_text = response.text
                llm_used = True
            else:
                logger.warning(
                    "maintenance report LLM generation failed — using the deterministic template",
                    extra={"component": "lifecycle_agent", "event_id": record.event_id},
                )

        report_path = self._reports_dir / f"{record.event_id}.md"
        report_path.write_text(report_text, encoding="utf-8")
        logger.info(
            "maintenance report generated",
            extra={"component": "lifecycle_agent", "event_id": record.event_id, "llm_used": llm_used, "path": str(report_path)},
        )
        return MaintenanceReport(event_id=record.event_id, text=report_text, llm_used=llm_used, path=report_path)

    def _deterministic_report(self, record: LifecycleRecord) -> str:
        """A template-only report — ALWAYS available, regardless of LLM configuration/failure.
        Covers every fact prompt.md §38 lists; nothing here is invented, only formatted."""
        outcome = "ACCEPTED and promoted to production" if record.final_status == "promoted" else "REJECTED — production unchanged"
        lines = [
            f"# Adaptation Maintenance Report — {record.event_id}",
            "",
            f"**Timestamp:** {record.timestamp.isoformat()}",
            "",
            "## What drifted",
            f"Component `{record.affected_component}` reported drift severity "
            f"{record.drift_severity:.4f} (drift event metadata: {record.drift_event.get('metadata', {})}).",
            "",
            "## Affected scope",
            f"`{record.affected_component}` (production version before this event: "
            f"`{record.production_version_before or 'none — no prior production version'}`).",
            "",
            "## Why adaptation was triggered",
            f"An external/mock drift detector (source={record.drift_event.get('source')}) flagged "
            f"`{record.affected_component}` at severity {record.drift_severity:.4f}, which the RL "
            "decision agent (PPO) observed and acted on.",
            "",
            "## What PPO selected",
            f"`{record.rl_action}`",
            "",
            "## What the adaptation agent did",
            json.dumps(record.agent_action, indent=2, default=str),
            "",
            "## What changed",
            f"Candidate version `{record.candidate_version}` "
            f"({record.model_metadata.get('model_class')}, adaptation_type="
            f"{record.model_metadata.get('adaptation_type')}) was produced from production version "
            f"`{record.production_version_before or 'none'}`.",
            "",
            "## Fidelity",
            f"before = {record.fidelity_before}, after = {record.fidelity_after}",
            "",
            "## Verification result",
            f"**{record.verification_result}** — {outcome}",
            "",
            "## Why",
            record.verification_explanation,
            "",
            "## Training / evaluation windows",
            f"training_window = {json.dumps(record.training_window, default=str)}",
            f"evaluation_window = {json.dumps(record.evaluation_window, default=str)}",
            "",
            "## LLM metadata",
            json.dumps(record.llm_metadata, indent=2, default=str) if record.llm_metadata else "not used for this adaptation",
        ]
        return "\n".join(lines)

    def _report_prompt(self, record: LifecycleRecord, deterministic_text: str, rag_context: str) -> str:
        return (
            "You are the Lifecycle Management agent of an autonomous Digital Twin adaptation "
            "system, writing a human-readable vendor maintenance report (prompt.md §38). Every "
            "fact below is ALREADY DETERMINED and RECORDED — you may rephrase, structure, and "
            "explain it more clearly for a human reader, but you MUST NOT invent, omit, or "
            "contradict any fact, and you must NOT change the verification result "
            f"({record.verification_result}), which is final and already acted on.\n\n"
            f"Recorded facts (deterministic template):\n{deterministic_text}\n\n"
            f"Relevant contextual knowledge (D2 RAG retrieval): {rag_context}\n\n"
            "Write the final maintenance report in clear prose/markdown covering: what drifted, "
            "the affected scope, why adaptation was triggered, what PPO selected, what the "
            "adaptation agent did, what changed, fidelity before/after, whether accepted or "
            "rejected and why, and any relevant contextual knowledge."
        )

    def _retrieve_rag_context(self, record: LifecycleRecord, rag_knowledge_base: "RagKnowledgeBase | None") -> str:
        if rag_knowledge_base is None or not rag_knowledge_base.is_available:
            return "not available"
        try:
            chunks = rag_knowledge_base.retrieve(
                f"maintenance and adaptation context for component {record.affected_component} "
                f"adaptation_type {record.model_metadata.get('adaptation_type')}",
                top_k=3,
            )
        except RagUnavailableError as exc:
            return f"unavailable: {exc}"
        if not chunks:
            return "no relevant knowledge retrieved"
        return " | ".join(f"[{c.category}/{c.source}] {c.text[:300]}" for c in chunks)
