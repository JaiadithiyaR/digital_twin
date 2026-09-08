"""Unit tests for Module 19 — Lifecycle Management Agent (Agent 6; prompt.md §37-38).

Fast, fully-controlled tests using directly-constructed (not registry-produced) instances of the
real `DriftEvent`/`ModelVersionMetadata`/`RecalibrationResult`/`RegenerationResult`/
`ExpandScopeResult`/`VerificationResult` types — every one of these is a plain pydantic model or
frozen dataclass with no hidden validation against live state, so constructing them directly here
is a legitimate, fast way to exercise every field-mapping path in `record_adaptation_event()`
without needing a full trained model/sandboxed subprocess for every test. The REQUIRED real,
end-to-end "run one full cycle" proof lives in `tests/integration/test_lifecycle_agent.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.adaptation.expand_scope_agent import ExpandScopeResult, _ProposedComponentDesign
from src.adaptation.lifecycle_agent import LifecycleAgent, LifecycleError
from src.adaptation.recalibration_agent import RecalibrationResult
from src.adaptation.regeneration_agent import RegenerationResult
from src.adaptation.verification_agent import DeterministicCheckResult, VerificationResult
from src.common.config import load_settings
from src.drift.schema import DriftEvent
from src.llm.anthropic_client import LLMResponse, LLMUsage
from src.rag.rag_kb import RetrievedChunk
from src.registry.model_registry import ModelVersionMetadata
from src.sandbox.executor import SandboxResult

SETTINGS = load_settings()


def _drift_event(component: str = "throughput", severity: float = 0.62) -> DriftEvent:
    now = datetime.now(UTC)
    return DriftEvent(
        component=component,
        severity=severity,
        timestamp=now,
        metadata={"reason": "unit-test synthetic drift"},
        source="MOCK",
        received_at=now,
    )


def _version(
    *,
    component: str = "throughput",
    version_id: str = "throughput-v2",
    parent_version_id: str | None = "throughput-v1",
    adaptation_type: str = "recalibrate",
    status: str = "candidate",
    llm_metadata: dict | None = None,
) -> ModelVersionMetadata:
    return ModelVersionMetadata(
        component=component,
        version_id=version_id,
        model_class="ThroughputModel",
        artifact_path=f"{component}/{version_id}.joblib",
        dependencies=(),
        feature_schema=("offered_load_mbps",),
        output_field="throughput_mbps_pred",
        adaptation_type=adaptation_type,
        parent_version_id=parent_version_id,
        created_at=datetime.now(UTC),
        training_window={"n_rows": 500},
        evaluation_window={"n_rows": 100},
        evaluation_metrics={"rmse": 1.2, "mae": 0.9},
        status=status,
        llm_metadata=llm_metadata,
    )


def _recalibration_result(version: ModelVersionMetadata, fidelity_before=0.7, fidelity_after=0.9) -> RecalibrationResult:
    return RecalibrationResult(
        version=version,
        candidate_component=None,  # never read by the lifecycle agent
        training_window={"n_rows": 500, "window_hours": 24},
        evaluation_window={"n_rows": 100},
        evaluation_metrics={"rmse": 1.2, "mae": 0.9},
        fidelity_before=fidelity_before,
        fidelity_after=fidelity_after,
    )


def _regeneration_result(version: ModelVersionMetadata) -> RegenerationResult:
    sandbox_result = SandboxResult(
        accepted=True, stage="ok", error=None, stdout="", stderr="", exit_code=0,
        metrics={"rmse": 1.0}, dependencies=(), required_features=("offered_load_mbps",),
    )
    return RegenerationResult(
        version=version,
        sandbox_result=sandbox_result,
        attempts=2,
        training_window={"n_rows": 800, "window_hours": 48},
        evaluation_window={"n_rows": 150},
        fidelity_before=0.6,
        fidelity_after=0.85,
    )


def _expand_scope_result(version: ModelVersionMetadata) -> ExpandScopeResult:
    design = _ProposedComponentDesign(
        component_name="sinr_quality", target_column="sinr_db", dependencies=(),
        required_features=("offered_load_mbps",), purpose="predict SINR proactively",
        feature_extraction_notes="none",
    )
    sandbox_result = SandboxResult(
        accepted=True, stage="ok", error=None, stdout="", stderr="", exit_code=0,
        metrics={"rmse": 2.0}, dependencies=(), required_features=("offered_load_mbps",),
    )
    return ExpandScopeResult(
        version=version, design=design, sandbox_result=sandbox_result,
        design_attempts=1, implementation_attempts=1,
        training_window={"n_rows": 800, "window_hours": 48}, evaluation_window={"n_rows": 150},
        fidelity_after=0.7,
    )


def _verification_result(version: ModelVersionMetadata, decision="ACCEPT", explanation="clear improvement") -> VerificationResult:
    return VerificationResult(
        component=version.component,
        candidate_version_id=version.version_id,
        decision=decision,
        updated_version=version.model_copy(update={"status": "production" if decision == "ACCEPT" else "rejected"}),
        fidelity_before=0.7,
        fidelity_after=0.9,
        fidelity_delta=0.2,
        verification_delta=SETTINGS.adaptation.verification_delta,
        fidelity_before_result=None,
        fidelity_after_result=None,
        checks=(DeterministicCheckResult("fidelity_improved", decision == "ACCEPT", "test"),),
        explanation=explanation,
        llm_used=False,
    )


def _agent(tmp_path: Path, llm_client=None) -> LifecycleAgent:
    return LifecycleAgent(
        records_path=tmp_path / "records.jsonl", reports_dir=tmp_path / "reports", llm_client=llm_client
    )


class _FakeLLMClient:
    def __init__(self, text: str | None = "LLM-authored report text"):
        self._text = text
        self.calls: list[str] = []

    def complete_safe(self, prompt, **kwargs):
        self.calls.append(prompt)
        if self._text is None:
            return None
        return LLMResponse(text=self._text, model="claude-sonnet-5", usage=LLMUsage(10, 10), stop_reason="end_turn", attempts=1)


class _FakeRagKb:
    def __init__(self, chunks):
        self.is_available = True
        self._chunks = chunks
        self.queries: list[str] = []

    def retrieve(self, query, *, category=None, top_k=None):
        self.queries.append(query)
        return self._chunks


# --- record_adaptation_event: field-by-field completeness ------------------------------------------


def test_record_captures_every_prompt_md_37_field(tmp_path):
    agent = _agent(tmp_path)
    drift_event = _drift_event()
    version = _version()
    agent_result = _recalibration_result(version)
    verification_result = _verification_result(version, decision="ACCEPT")

    record = agent.record_adaptation_event(
        drift_event=drift_event, rl_observation=[0.1, 0.2, 0.3], rl_action="recalibrate",
        agent_result=agent_result, verification_result=verification_result,
    )

    assert record.event_id
    assert record.timestamp is not None
    assert record.drift_event["component"] == "throughput"
    assert record.affected_component == "throughput"
    assert record.drift_severity == pytest.approx(0.62)
    assert record.rl_observation == [0.1, 0.2, 0.3]
    assert record.rl_action == "recalibrate"
    assert record.agent_action["adaptation_type"] == "recalibrate"
    assert record.production_version_before == "throughput-v1"  # == candidate's parent_version_id
    assert record.candidate_version == "throughput-v2"
    assert record.fidelity_before == pytest.approx(0.7)
    assert record.fidelity_after == pytest.approx(0.9)
    assert record.verification_result == "ACCEPT"
    assert record.verification_explanation == "clear improvement"
    assert record.training_window["n_rows"] == 500
    assert record.evaluation_window["n_rows"] == 100
    assert record.model_metadata["component"] == "throughput"
    assert record.model_metadata["adaptation_type"] == "recalibrate"
    assert record.llm_metadata is None
    assert record.final_status == "promoted"


def test_final_status_rejected_when_verification_rejects(tmp_path):
    agent = _agent(tmp_path)
    version = _version(parent_version_id="throughput-v3")
    agent_result = _recalibration_result(version)
    verification_result = _verification_result(version, decision="REJECT", explanation="no improvement")

    record = agent.record_adaptation_event(
        drift_event=_drift_event(), rl_observation=[0.0], rl_action="recalibrate",
        agent_result=agent_result, verification_result=verification_result,
    )
    assert record.verification_result == "REJECT"
    assert record.final_status == "rejected"
    assert record.production_version_before == "throughput-v3"  # unchanged production, correctly reported


def test_no_parent_version_reported_as_none_for_expand_scope(tmp_path):
    agent = _agent(tmp_path)
    version = _version(component="sinr_quality", version_id="sinr_quality-v1", parent_version_id=None, adaptation_type="expand_scope")
    agent_result = _expand_scope_result(version)
    verification_result = _verification_result(version, decision="ACCEPT")

    record = agent.record_adaptation_event(
        drift_event=_drift_event(component="throughput"), rl_observation=[0.0], rl_action="expand_scope",
        agent_result=agent_result, verification_result=verification_result,
    )
    assert record.production_version_before is None
    assert record.agent_action["proposed_design"]["component_name"] == "sinr_quality"
    assert record.agent_action["design_attempts"] == 1


def test_agent_action_summary_is_generic_across_all_three_agent_types(tmp_path):
    agent = _agent(tmp_path)

    recal_version = _version(version_id="v-recal", adaptation_type="recalibrate")
    recal_record = agent.record_adaptation_event(
        drift_event=_drift_event(), rl_observation=[0.0], rl_action="recalibrate",
        agent_result=_recalibration_result(recal_version), verification_result=_verification_result(recal_version),
    )
    assert "evaluation_metrics" in recal_record.agent_action
    assert "sandbox_stage" not in recal_record.agent_action

    regen_version = _version(version_id="v-regen", adaptation_type="regenerate")
    regen_record = agent.record_adaptation_event(
        drift_event=_drift_event(), rl_observation=[0.0], rl_action="regenerate",
        agent_result=_regeneration_result(regen_version), verification_result=_verification_result(regen_version),
    )
    assert regen_record.agent_action["sandbox_stage"] == "ok"
    assert regen_record.agent_action["attempts"] == 2
    assert "proposed_design" not in regen_record.agent_action

    expand_version = _version(component="sinr_quality", version_id="v-expand", parent_version_id=None, adaptation_type="expand_scope")
    expand_record = agent.record_adaptation_event(
        drift_event=_drift_event(), rl_observation=[0.0], rl_action="expand_scope",
        agent_result=_expand_scope_result(expand_version), verification_result=_verification_result(expand_version),
    )
    assert expand_record.agent_action["design_attempts"] == 1
    assert expand_record.agent_action["proposed_design"]["target_column"] == "sinr_db"


def test_llm_metadata_carried_from_the_updated_registry_version(tmp_path):
    agent = _agent(tmp_path)
    version = _version(llm_metadata={"model": "claude-sonnet-5", "reasoning": "rebuilt pipeline"})
    verification_result = _verification_result(version, decision="ACCEPT")
    record = agent.record_adaptation_event(
        drift_event=_drift_event(), rl_observation=[0.0], rl_action="recalibrate",
        agent_result=_recalibration_result(version), verification_result=verification_result,
    )
    assert record.llm_metadata == {"model": "claude-sonnet-5", "reasoning": "rebuilt pipeline"}


# --- persistence / auditability -------------------------------------------------------------------


def test_records_persist_and_are_readable_by_a_fresh_agent_instance(tmp_path):
    agent = _agent(tmp_path)
    version = _version()
    record = agent.record_adaptation_event(
        drift_event=_drift_event(), rl_observation=[0.1], rl_action="recalibrate",
        agent_result=_recalibration_result(version), verification_result=_verification_result(version),
    )

    fresh_agent = LifecycleAgent(records_path=tmp_path / "records.jsonl", reports_dir=tmp_path / "reports")
    reloaded = fresh_agent.get_record(record.event_id)
    assert reloaded == record


def test_multiple_events_append_without_overwriting(tmp_path):
    agent = _agent(tmp_path)
    version_a = _version(version_id="v-a")
    version_b = _version(version_id="v-b")
    record_a = agent.record_adaptation_event(
        drift_event=_drift_event(), rl_observation=[0.0], rl_action="recalibrate",
        agent_result=_recalibration_result(version_a), verification_result=_verification_result(version_a),
    )
    record_b = agent.record_adaptation_event(
        drift_event=_drift_event(), rl_observation=[0.0], rl_action="recalibrate",
        agent_result=_recalibration_result(version_b), verification_result=_verification_result(version_b),
    )
    records = agent.list_records()
    assert [r.event_id for r in records] == [record_a.event_id, record_b.event_id]


def test_get_record_raises_for_unknown_event_id(tmp_path):
    agent = _agent(tmp_path)
    with pytest.raises(LifecycleError):
        agent.get_record("does-not-exist")


def test_list_records_on_a_fresh_path_is_empty(tmp_path):
    agent = _agent(tmp_path)
    assert agent.list_records() == []


# --- maintenance report (prompt.md §38) -------------------------------------------------------------


def test_deterministic_report_generated_without_llm_covers_every_required_topic(tmp_path):
    agent = _agent(tmp_path)
    version = _version()
    record = agent.record_adaptation_event(
        drift_event=_drift_event(), rl_observation=[0.0], rl_action="recalibrate",
        agent_result=_recalibration_result(version), verification_result=_verification_result(version, decision="ACCEPT"),
    )
    report = agent.generate_maintenance_report(record)

    assert report.llm_used is False
    assert report.path.exists()
    assert report.path.read_text(encoding="utf-8") == report.text
    for expected in ("throughput", "recalibrate", "ACCEPT", "0.7", "0.9", "clear improvement"):
        assert expected in report.text


def test_llm_report_is_used_verbatim_when_available(tmp_path):
    llm = _FakeLLMClient(text="A concise, LLM-written maintenance summary.")
    agent = _agent(tmp_path, llm_client=llm)
    version = _version()
    record = agent.record_adaptation_event(
        drift_event=_drift_event(), rl_observation=[0.0], rl_action="recalibrate",
        agent_result=_recalibration_result(version), verification_result=_verification_result(version),
    )
    report = agent.generate_maintenance_report(record)
    assert report.llm_used is True
    assert report.text == "A concise, LLM-written maintenance summary."
    assert report.path.read_text(encoding="utf-8") == "A concise, LLM-written maintenance summary."
    assert llm.calls  # the LLM was genuinely called


def test_llm_failure_falls_back_to_deterministic_report(tmp_path):
    llm = _FakeLLMClient(text=None)  # simulates complete_safe() already degrading to None
    agent = _agent(tmp_path, llm_client=llm)
    version = _version()
    record = agent.record_adaptation_event(
        drift_event=_drift_event(), rl_observation=[0.0], rl_action="recalibrate",
        agent_result=_recalibration_result(version), verification_result=_verification_result(version),
    )
    report = agent.generate_maintenance_report(record)
    assert report.llm_used is False
    assert "throughput" in report.text


def test_rag_context_is_retrieved_and_reaches_the_llm_prompt(tmp_path):
    chunk = RetrievedChunk(
        text="Adaptation policy: verification_delta is 0.01.", category="policies",
        source="internal:config/settings.yaml", document_id="adaptation_policies.md",
        chunk_index=0, version="abc", distance=0.05,
    )
    rag_kb = _FakeRagKb([chunk])
    llm = _FakeLLMClient(text="report referencing policy")
    agent = _agent(tmp_path, llm_client=llm)
    version = _version()
    record = agent.record_adaptation_event(
        drift_event=_drift_event(), rl_observation=[0.0], rl_action="recalibrate",
        agent_result=_recalibration_result(version), verification_result=_verification_result(version),
    )
    agent.generate_maintenance_report(record, rag_knowledge_base=rag_kb)
    assert rag_kb.queries
    assert "verification_delta is 0.01" in llm.calls[0]


def test_no_rag_knowledge_base_reports_not_available_and_never_crashes(tmp_path):
    agent = _agent(tmp_path)
    version = _version()
    record = agent.record_adaptation_event(
        drift_event=_drift_event(), rl_observation=[0.0], rl_action="recalibrate",
        agent_result=_recalibration_result(version), verification_result=_verification_result(version),
    )
    report = agent.generate_maintenance_report(record, rag_knowledge_base=None)
    assert report.path.exists()
