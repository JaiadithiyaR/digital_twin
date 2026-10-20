"""Lifecycle Management Agent (Agent 6) — decides nothing; records and explains what every other
agent already decided. Every adaptation event produces exactly one auditable `LifecycleRecord`
plus an automatically-generated maintenance report.

`LifecycleRecord` fields are read verbatim from upstream objects, never recomputed:
- `trigger`/`affected_component`/`trigger_severity`/`trigger_type` <- the canonical
  `AdaptationTrigger`.
- `decision_strategy`/`root_cause_analysis`/`decision_confidence`/`decision_rationale`/
  `knowledge_refs` <- the Decision Agent's `DecisionOutput`, verbatim.
- `agent_action` <- a duck-typed summary of whichever Recalibration/Regeneration/ExpandScope
  result was produced (`_summarize_agent_action()` — the three share no base class).
- `production_version_before` <- `candidate_version.parent_version_id` (not a fresh registry
  lookup, which would be wrong on ACCEPT since the registry has already been updated).
- `candidate_version`/`fidelity_before`/`fidelity_after`/`verification_result`/
  `verification_explanation`/`final_status` <- Verification's independently recomputed
  `VerificationResult`, never an agent's self-reported numbers.

Persistence: one append-only JSON-Lines file (`config.lifecycle.records_path`) — one line per
event, under a lock, never rewritten.

Report generation: a deterministic template (`_deterministic_report()`) is always produced first
and used whenever no LLM client is supplied or the call fails. When a client is supplied, it
writes better prose FROM those same facts (never invited to invent new ones), with RAG consulted
for context — degrading to "not available" when unavailable.
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

from pydantic import BaseModel, ConfigDict, ValidationError

from src.rag.rag_kb import RagUnavailableError

if TYPE_CHECKING:
    from src.adaptation.decision_agent import DecisionOutput
    from src.adaptation.expand_scope_agent import ExpandScopeResult
    from src.adaptation.recalibration_agent import RecalibrationResult
    from src.adaptation.regeneration_agent import RegenerationResult
    from src.adaptation.verification_agent import VerificationResult
    from src.common.config import Settings
    from src.fidelity.trigger import AdaptationTrigger
    from src.llm.google_client import GoogleClient
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
    trigger: dict[str, Any]
    affected_component: str
    trigger_severity: float
    trigger_type: str
    decision_strategy: str
    root_cause_analysis: str
    decision_confidence: float
    decision_rationale: str
    knowledge_refs: list[str]
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
    # Real values from whichever telemetry source drove this event ("OAI_RAN_REPLAY" | "MOCK").
    telemetry_source: str
    replayed_run_ids: list[int] | None
    replay_speed: float | None
    # Per-component adaptation policy outcome (supplied by the orchestrator; defaults keep records
    # written before this policy existed readable). `post_adaptation_fidelity` is the component's
    # FidelityScore_c re-measured on the latest window with whatever model is serving after the
    # event; `dependency_revalidation` lists downstream components re-measured after a hot swap.
    component_threshold: float | None = None
    fidelity_at_trigger: float | None = None
    production_version_after: str | None = None
    post_adaptation_fidelity: float | None = None
    remaining_fidelity_gap: float | None = None
    still_below_threshold: bool | None = None
    dependency_revalidation: list[dict[str, Any]] = []
    # Adaptation loop: which attempt (1-based) of how many allowed this event was.
    adaptation_attempt: int | None = None
    max_adaptation_attempts: int | None = None


def _summarize_agent_action(agent_result: Any) -> dict[str, Any]:
    """Duck-typed summary of whichever adaptation result object was produced — the three result
    types share no common base class."""
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
        llm_client: "GoogleClient | None" = None,
        llm_effort: str | None = None,
        static_prefix: str | None = None,
    ) -> None:
        self._records_path = records_path
        self._reports_dir = reports_dir
        self._llm_client = llm_client
        # Phase 13 §3.4: "explanation" role — human-readable prose FROM already-final,
        # already-recorded facts (see this class's own docstring), never deep reasoning.
        self._llm_effort = llm_effort
        # Phase 13 §3.3 — see RegenerationAgent's own entry for the full rationale: `None` for
        # every non-baseline caller, which keeps this agent's prompt content byte-for-byte
        # unchanged from before this feature existed.
        self._static_prefix = static_prefix
        self._lock = threading.Lock()
        self._records_path.parent.mkdir(parents=True, exist_ok=True)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(
        cls, settings: "Settings", llm_client: "GoogleClient | None" = None, static_prefix: str | None = None
    ) -> "LifecycleAgent":
        cfg = settings.lifecycle
        return cls(
            records_path=settings.resolve_path(cfg.records_path),
            reports_dir=settings.resolve_path(cfg.reports_dir),
            llm_client=llm_client,
            llm_effort=settings.llm.effort_by_role.explanation,
            static_prefix=static_prefix,
        )

    # --- recording -----------------------------------------------------------------------------

    def record_adaptation_event(
        self,
        *,
        trigger: "AdaptationTrigger",
        decision: "DecisionOutput",
        agent_result: "RecalibrationResult | RegenerationResult | ExpandScopeResult",
        verification_result: "VerificationResult",
        telemetry_source: str = "MOCK",
        replayed_run_ids: list[int] | None = None,
        replay_speed: float | None = None,
        component_policy_outcome: dict[str, Any] | None = None,
    ) -> LifecycleRecord:
        """Build and durably append exactly one `LifecycleRecord` for one full adaptation event.
        Every argument is a real object a real upstream module (11/12/13/14-16/17) already
        produced — this method computes nothing beyond the field-by-field mapping documented in
        the module docstring above."""
        candidate_version_meta = agent_result.version
        updated_version = verification_result.updated_version

        record = LifecycleRecord(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC),
            trigger=json.loads(trigger.model_dump_json()),
            affected_component=trigger.component,
            trigger_severity=trigger.severity,
            trigger_type=trigger.trigger_type,
            decision_strategy=decision.strategy,
            root_cause_analysis=decision.root_cause_analysis,
            decision_confidence=decision.confidence,
            decision_rationale=decision.rationale,
            knowledge_refs=list(decision.knowledge_refs),
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
            telemetry_source=telemetry_source,
            replayed_run_ids=replayed_run_ids,
            replay_speed=replay_speed,
            **(component_policy_outcome or {}),
        )
        self._append_record(record)
        logger.info(
            "lifecycle event recorded",
            extra={
                "component": "lifecycle_agent",
                "event_id": record.event_id,
                "affected_component": record.affected_component,
                "decision_strategy": record.decision_strategy,
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
        """Every line that parses against the CURRENT `LifecycleRecord` schema — a line written
        by a prior, now-superseded schema (e.g. this project's own pre-design-pivot PPO-era
        records, which used `drift_event`/`rl_observation`/`rl_action` instead of `trigger`/
        `decision_strategy`/...) is logged and skipped, never allowed to crash the entire read.
        This mirrors the same "one malformed record must never take down the whole stream"
        discipline Module 2's telemetry quarantine and Module 11's drift-event quarantine already
        establish — an append-only audit log must stay readable even across a schema migration,
        not become a single point of failure the moment one old-format line is encountered."""
        if not self._records_path.exists():
            return []
        with self._lock:
            lines = self._records_path.read_text(encoding="utf-8").splitlines()
        records: list[LifecycleRecord] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                records.append(LifecycleRecord.model_validate_json(line))
            except ValidationError as exc:
                logger.warning(
                    "skipping a lifecycle record line that doesn't match the current schema "
                    "(likely written by a superseded schema version) — never crashes the read",
                    extra={"component": "lifecycle_agent", "error": str(exc)},
                )
        return records

    def get_record(self, event_id: str) -> LifecycleRecord:
        for record in self.list_records():
            if record.event_id == event_id:
                return record
        raise LifecycleError(f"no lifecycle record found with event_id={event_id!r}")

    # --- maintenance report (prompt.md §38) --------------------------------------------------

    def generate_maintenance_report(
        self,
        record: LifecycleRecord,
        *,
        rag_knowledge_base: "RagKnowledgeBase | None" = None,
        cache_key: str | None = None,
    ) -> MaintenanceReport:
        """Automatically produce a human-readable vendor maintenance report for `record` (prompt.md
        §38's exact list: what drifted, affected scope, why adaptation was triggered, what the
        Decision & Root-Cause Analysis Agent selected, what the adaptation agent did, what changed,
        fidelity before/after, accepted/rejected + why, relevant contextual knowledge). Always
        saves the report to `config.lifecycle.reports_dir/<event_id>.md` and returns it."""
        deterministic_text = self._deterministic_report(record)
        rag_context = self._retrieve_rag_context(record, rag_knowledge_base)

        report_text = deterministic_text
        llm_used = False
        if self._llm_client is not None:
            prompt = self._report_prompt(record, deterministic_text, rag_context)
            sent_prompt = prompt if cache_key is not None else (f"{self._static_prefix}\n\n{prompt}" if self._static_prefix else prompt)
            response = self._llm_client.complete_safe(sent_prompt, effort=self._llm_effort, cache_key=cache_key)
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
            f"Component `{record.affected_component}` triggered adaptation via "
            f"`{record.trigger_type}` at severity {record.trigger_severity:.4f} "
            f"(trigger metadata: {record.trigger.get('metadata', {})}).",
            "",
            "## Affected scope",
            f"`{record.affected_component}` (production version before this event: "
            f"`{record.production_version_before or 'none — no prior production version'}`).",
            "",
            "## Why adaptation was triggered",
            f"A `{record.trigger_type}` trigger flagged `{record.affected_component}` at severity "
            f"{record.trigger_severity:.4f}, which the Decision & Root-Cause Analysis Agent "
            "analyzed and acted on.",
            "",
            "## Root-cause analysis",
            record.root_cause_analysis,
            "",
            "## Decision strategy selected",
            f"`{record.decision_strategy}` (confidence={record.decision_confidence:.2f})\n\n"
            f"Rationale: {record.decision_rationale}",
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
            "## Component fidelity policy",
            f"threshold = {record.component_threshold}, fidelity at trigger = {record.fidelity_at_trigger}, "
            f"post-adaptation fidelity = {record.post_adaptation_fidelity}, remaining gap = "
            f"{record.remaining_fidelity_gap}, still below threshold = {record.still_below_threshold}, "
            f"production version after = {record.production_version_after}, adaptation attempt = "
            f"{record.adaptation_attempt} of {record.max_adaptation_attempts}",
            "",
            "## Dependency revalidation",
            "\n".join(
                f"- `{d.get('component')}`: F_c = {d.get('fidelity_score')}, threshold = {d.get('threshold')} — {d.get('status')}"
                for d in record.dependency_revalidation
            ) or "none required (production unchanged, or no downstream components)",
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
            "the affected scope, why adaptation was triggered, what the Decision & Root-Cause "
            "Analysis Agent selected and why, what the adaptation agent did, what changed, "
            "fidelity before/after, whether accepted or rejected and why, and any relevant "
            "contextual knowledge."
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
