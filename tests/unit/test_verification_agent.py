"""Unit tests for Module 17 — Agentic Verification Agent (prompt.md §31-34, rule 9).

No real ANTHROPIC_API_KEY is configured in this environment (same situation as every other
LLM-driven agent's tests in this repo) — `_FakeLLMClient` stands in for `AnthropicClient`,
implementing only `complete_structured` (the one method this agent calls), exactly the pattern
`tests/unit/test_regeneration_agent.py`/`test_expand_scope_agent.py` already established.
Everything else — the deterministic gate, the real `FidelityEvaluator`/Module 12 recomputation,
the real `ModelRegistry` promote/reject writes — is REAL, unmocked code.

**The centerpiece requirement (rule 9, prompt.md §70: "LLMs cannot override deterministic
acceptance criteria")** is proven by `test_llm_disagreement_never_changes_the_deterministic_
decision_*` below, in BOTH directions: a fake LLM explicitly arguing for the OPPOSITE of what the
deterministic gate computes, for a scenario that deterministically ACCEPTs and one that
deterministically REJECTs — in both cases the final `decision` AND the actual `ModelRegistry`
promotion/rejection are shown to follow the deterministic result, never the LLM's stated opinion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adaptation.verification_agent import (
    VerificationAgent,
    VerificationError,
    _LLMVerificationReasoning,
)
from src.common.config import load_settings
from src.dt_models.base import DTComponent
from src.dt_models.throughput import ThroughputModel
from src.fidelity.evaluator import FidelityEvaluator
from src.llm.anthropic_client import LLMClientError
from src.rag.rag_kb import RetrievedChunk
from src.registry.model_registry import ModelRegistry

SETTINGS = load_settings()
VERIFICATION_DELTA = SETTINGS.adaptation.verification_delta

# A real spread of prediction-error magnitudes used to pre-warm a fresh `FidelityEvaluator`'s
# rolling-window normalization reference past `config.fidelity.min_history_for_normalization`
# (10) — mirrors Modules 14/15/16's own tests, which pre-warm a fresh evaluator before expecting a
# real, non-`None` fidelity_score. Deliberately NOT a degenerate point-mass construction
# (`y_true=[0]*k, y_pred=[a]*k`, as Module 12's own hand-computed unit test uses for a handful of
# exact-value cases): MK-MMD is scale/shift-invariant over such a degenerate pair, so every
# point-mass prewarm sample yields the SAME mk_mmd value regardless of `a` — collapsing that
# metric's rolling-window (max-min) range to ~0 and exploding the eps-stabilized normalization for
# the very first genuinely different mk_mmd value evaluated afterward (the same numerical-
# stability failure mode Module 13's own writeup documents for its PPO environment). Using
# genuinely-distributed `(y_true, y_pred=y_true+noise)` pairs at increasing noise levels instead
# gives all four metrics — including MK-MMD — real, non-degenerate variation across the window.
_SPREAD = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0]


def _prewarm(evaluator: FidelityEvaluator, component: str, noise_stds: list[float] = _SPREAD, n: int = 40) -> None:
    rng = np.random.default_rng(999)
    base = rng.normal(20.0, 8.0, n)
    for noise_std in noise_stds:
        y_pred = base + rng.normal(0.0, noise_std, n)
        evaluator.evaluate(component, base, y_pred, update_window=True)


def _synthetic_features(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "offered_load_mbps": rng.uniform(5, 60, n),
            "prb_utilization_pct": rng.uniform(10, 90, n),
            "sinr_db": rng.uniform(0, 30, n),
            "rsrp_dbm": rng.uniform(-120, -60, n),
            "rsrq_db": rng.uniform(-15, -5, n),
            "ue_count": rng.integers(1, 10, n).astype(float),
            "ue_speed_mps": rng.uniform(0, 15, n),
        }
    )


def _synthetic_target(features: pd.DataFrame, noise_std: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    base = 0.6 * features["offered_load_mbps"] + 0.4 * features["sinr_db"] - 0.1 * features["prb_utilization_pct"]
    return base + rng.normal(0.0, noise_std, len(features))


def _good_throughput_model(seed: int) -> ThroughputModel:
    """Trained on real signal, ample clean data — genuinely fits the real generative function."""
    model = ThroughputModel.from_settings(SETTINGS)
    features = _synthetic_features(400, seed)
    target = _synthetic_target(features, noise_std=0.3, seed=seed + 1)
    model.train(features, target)
    return model


def _bad_throughput_model(seed: int) -> ThroughputModel:
    """Trained on a CONSTANT target, unrelated to the real generative function — systematically,
    not just noisily, bad on held-out data drawn from the real function, regardless of RNG."""
    model = ThroughputModel.from_settings(SETTINGS)
    features = _synthetic_features(200, seed)
    target = pd.Series(np.full(len(features), 5.0))
    model.train(features, target)
    return model


def _registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")


def _register(registry: ModelRegistry, model, *, status, adaptation_type="bootstrap", parent_version_id=None):
    return registry.register_version(
        component_instance=model,
        adaptation_type=adaptation_type,
        parent_version_id=parent_version_id,
        training_window={"n_rows": 0},
        evaluation_window={"n_rows": 0},
        evaluation_metrics={},
        status=status,
    )


class _TinyComponent(DTComponent):
    """A trivial component used ONLY for the "no production baseline exists" (expand-scope-shaped)
    test scenarios below — never a real model, mirrors `tests/dummy_dt_components.py`'s style."""

    COMPONENT_NAME = "verification_test_component"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES = ("x",)
    OUTPUT_FIELD = "verification_test_component_pred"

    def __init__(self) -> None:
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        self._trained = True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return features["x"].rename(self.OUTPUT_FIELD)

    def evaluate(self, features: pd.DataFrame, targets: pd.Series) -> dict[str, float]:
        return {"mae": 0.0}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("trained" if self._trained else "untrained")

    def load(self, path: Path) -> None:
        self._trained = path.read_text() == "trained"


class _FakeLLMClient:
    """Stands in for `AnthropicClient` — only `complete_structured` is exercised by this agent."""

    def __init__(self, explanation: str = "", key_observations: str = "", raise_on_call: Exception | None = None):
        self._explanation = explanation
        self._key_observations = key_observations
        self._raise_on_call = raise_on_call
        self.calls: list[str] = []

    def complete_structured(self, prompt, schema, **kwargs):
        self.calls.append(prompt)
        if self._raise_on_call is not None:
            raise self._raise_on_call
        return schema(explanation=self._explanation, key_observations=self._key_observations)


class _FakeRagKb:
    def __init__(self, chunks: list[RetrievedChunk]):
        self.is_available = True
        self._chunks = chunks
        self.queries: list[str] = []

    def retrieve(self, query, *, category=None, top_k=None):
        self.queries.append(query)
        return self._chunks


class _UnavailableRagKb:
    is_available = False

    def retrieve(self, *args, **kwargs):
        raise AssertionError("retrieve() must never be called when is_available is False")


# --- basic contract / usage errors ---------------------------------------------------------------


def test_verifying_a_non_candidate_version_raises(tmp_path):
    registry = _registry(tmp_path)
    model = _good_throughput_model(seed=1)
    version = _register(registry, model, status="rejected")
    agent = VerificationAgent(SETTINGS, registry, FidelityEvaluator(SETTINGS.fidelity))
    features = _synthetic_features(10, seed=2)
    target = _synthetic_target(features, 0.3, seed=3)
    with pytest.raises(VerificationError):
        agent.verify(version, features, target, model.predict(features))


# --- the deterministic gate itself -----------------------------------------------------------------


def test_clearly_better_candidate_is_accepted_and_promoted(tmp_path):
    registry = _registry(tmp_path)
    production_model = _bad_throughput_model(seed=10)
    prod_version = _register(registry, production_model, status="production")
    candidate_model = _good_throughput_model(seed=20)
    candidate_version = _register(
        registry, candidate_model, status="candidate", adaptation_type="recalibrate", parent_version_id=prod_version.version_id
    )

    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    _prewarm(evaluator, "throughput")
    agent = VerificationAgent(SETTINGS, registry, evaluator)

    eval_features = _synthetic_features(150, seed=30)
    eval_target = _synthetic_target(eval_features, 0.3, seed=31)
    candidate_preds = candidate_model.predict(eval_features)

    result = agent.verify(
        candidate_version,
        eval_features,
        eval_target,
        candidate_preds,
        production_component_factory=lambda: ThroughputModel.from_settings(SETTINGS),
    )

    assert result.decision == "ACCEPT"
    assert all(c.passed for c in result.checks)
    assert result.fidelity_before is not None and result.fidelity_after is not None
    assert result.fidelity_after > result.fidelity_before + VERIFICATION_DELTA
    assert registry.get_current_version("throughput").version_id == candidate_version.version_id
    assert registry.get_version("throughput", prod_version.version_id).status == "superseded"


def test_clearly_worse_candidate_is_rejected_and_production_unchanged(tmp_path):
    registry = _registry(tmp_path)
    production_model = _good_throughput_model(seed=40)
    prod_version = _register(registry, production_model, status="production")
    candidate_model = _bad_throughput_model(seed=50)
    candidate_version = _register(
        registry, candidate_model, status="candidate", adaptation_type="recalibrate", parent_version_id=prod_version.version_id
    )

    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    _prewarm(evaluator, "throughput")
    agent = VerificationAgent(SETTINGS, registry, evaluator)

    eval_features = _synthetic_features(150, seed=60)
    eval_target = _synthetic_target(eval_features, 0.3, seed=61)
    candidate_preds = candidate_model.predict(eval_features)

    result = agent.verify(
        candidate_version,
        eval_features,
        eval_target,
        candidate_preds,
        production_component_factory=lambda: ThroughputModel.from_settings(SETTINGS),
    )

    assert result.decision == "REJECT"
    assert registry.get_current_version("throughput").version_id == prod_version.version_id
    assert registry.get_version("throughput", candidate_version.version_id).status == "rejected"


def test_nan_candidate_predictions_are_rejected_not_crashed(tmp_path):
    registry = _registry(tmp_path)
    production_model = _good_throughput_model(seed=70)
    prod_version = _register(registry, production_model, status="production")
    candidate_model = _good_throughput_model(seed=71)
    candidate_version = _register(
        registry, candidate_model, status="candidate", adaptation_type="recalibrate", parent_version_id=prod_version.version_id
    )
    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    _prewarm(evaluator, "throughput")
    agent = VerificationAgent(SETTINGS, registry, evaluator)

    eval_features = _synthetic_features(20, seed=72)
    eval_target = _synthetic_target(eval_features, 0.3, seed=73)
    nan_preds = np.full(len(eval_features), np.nan)

    result = agent.verify(
        candidate_version,
        eval_features,
        eval_target,
        nan_preds,
        production_component_factory=lambda: ThroughputModel.from_settings(SETTINGS),
    )
    assert result.decision == "REJECT"
    assert any(c.name == "no_invalid_output" and not c.passed for c in result.checks)


def test_length_mismatched_predictions_are_rejected_not_crashed(tmp_path):
    registry = _registry(tmp_path)
    production_model = _good_throughput_model(seed=74)
    prod_version = _register(registry, production_model, status="production")
    candidate_model = _good_throughput_model(seed=75)
    candidate_version = _register(
        registry, candidate_model, status="candidate", adaptation_type="recalibrate", parent_version_id=prod_version.version_id
    )
    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    _prewarm(evaluator, "throughput")
    agent = VerificationAgent(SETTINGS, registry, evaluator)

    eval_features = _synthetic_features(20, seed=76)
    eval_target = _synthetic_target(eval_features, 0.3, seed=77)
    mismatched_preds = np.array([1.0, 2.0, 3.0])

    result = agent.verify(
        candidate_version,
        eval_features,
        eval_target,
        mismatched_preds,
        production_component_factory=lambda: ThroughputModel.from_settings(SETTINGS),
    )
    assert result.decision == "REJECT"
    assert any(c.name == "evaluation_ran" and not c.passed for c in result.checks)


def test_missing_candidate_artifact_is_rejected(tmp_path):
    registry = _registry(tmp_path)
    production_model = _good_throughput_model(seed=80)
    prod_version = _register(registry, production_model, status="production")
    candidate_model = _good_throughput_model(seed=81)
    candidate_version = _register(
        registry, candidate_model, status="candidate", adaptation_type="recalibrate", parent_version_id=prod_version.version_id
    )
    registry.artifact_path(candidate_version).unlink()  # simulate a corrupt/missing artifact on disk

    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    _prewarm(evaluator, "throughput")
    agent = VerificationAgent(SETTINGS, registry, evaluator)

    eval_features = _synthetic_features(20, seed=82)
    eval_target = _synthetic_target(eval_features, 0.3, seed=83)
    preds = candidate_model.predict(eval_features)

    result = agent.verify(
        candidate_version,
        eval_features,
        eval_target,
        preds,
        production_component_factory=lambda: ThroughputModel.from_settings(SETTINGS),
    )
    assert result.decision == "REJECT"
    assert any(c.name == "candidate_artifact_present" and not c.passed for c in result.checks)


def test_missing_production_factory_with_existing_baseline_is_rejected_not_skipped(tmp_path):
    """Fail-safe: a production version genuinely exists, but the caller didn't supply a way to
    evaluate it under the same protocol — this must REJECT, never silently skip the primary
    regression criterion (which would let an unverifiable candidate slip through)."""
    registry = _registry(tmp_path)
    production_model = _good_throughput_model(seed=90)
    prod_version = _register(registry, production_model, status="production")
    candidate_model = _good_throughput_model(seed=91)
    candidate_version = _register(
        registry, candidate_model, status="candidate", adaptation_type="recalibrate", parent_version_id=prod_version.version_id
    )
    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    _prewarm(evaluator, "throughput")
    agent = VerificationAgent(SETTINGS, registry, evaluator)

    eval_features = _synthetic_features(20, seed=92)
    eval_target = _synthetic_target(eval_features, 0.3, seed=93)
    preds = candidate_model.predict(eval_features)

    result = agent.verify(candidate_version, eval_features, eval_target, preds)  # no factory supplied
    assert result.decision == "REJECT"
    assert any(c.name == "production_baseline_evaluable" and not c.passed for c in result.checks)
    assert registry.get_current_version("throughput").version_id == prod_version.version_id


# --- "no production baseline exists" (expand-scope-shaped) ----------------------------------------


def _register_tiny_candidate(registry: ModelRegistry) -> tuple["_TinyComponent", "ModelVersionMetadata"]:
    model = _TinyComponent()
    model.train(pd.DataFrame({"x": [1.0]}), pd.Series([1.0]))
    version = registry.register_version(
        component_instance=model,
        adaptation_type="expand_scope",
        parent_version_id=None,
        training_window={"n_rows": 0},
        evaluation_window={"n_rows": 0},
        evaluation_metrics={},
        status="candidate",
    )
    return model, version


def test_no_production_baseline_accepts_when_candidate_fidelity_is_computable(tmp_path):
    registry = _registry(tmp_path)
    model, version = _register_tiny_candidate(registry)
    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    _prewarm(evaluator, _TinyComponent.COMPONENT_NAME)
    agent = VerificationAgent(SETTINGS, registry, evaluator)

    eval_target = np.array([1.0, 2.0, 3.0])
    candidate_preds = np.array([1.1, 2.1, 2.9])
    result = agent.verify(version, pd.DataFrame({"x": eval_target}), eval_target, candidate_preds)

    assert result.decision == "ACCEPT"
    assert result.fidelity_before is None
    assert result.fidelity_after is not None
    assert registry.get_current_version(_TinyComponent.COMPONENT_NAME).version_id == version.version_id


def test_no_production_baseline_rejects_when_candidate_fidelity_not_yet_computable(tmp_path):
    registry = _registry(tmp_path)
    model, version = _register_tiny_candidate(registry)
    evaluator = FidelityEvaluator(SETTINGS.fidelity)  # fresh — no prewarm, insufficient history
    agent = VerificationAgent(SETTINGS, registry, evaluator)

    eval_target = np.array([1.0, 2.0, 3.0])
    candidate_preds = np.array([1.1, 2.1, 2.9])
    result = agent.verify(version, pd.DataFrame({"x": eval_target}), eval_target, candidate_preds)

    assert result.decision == "REJECT"
    assert result.fidelity_after is None
    assert registry.get_current_version(_TinyComponent.COMPONENT_NAME) is None


# --- rule 9: the LLM/RAG reasoning layer can inform the explanation but must never override --------


def test_llm_disagreement_never_changes_the_deterministic_accept_decision(tmp_path):
    registry = _registry(tmp_path)
    production_model = _bad_throughput_model(seed=100)
    prod_version = _register(registry, production_model, status="production")
    candidate_model = _good_throughput_model(seed=101)
    candidate_version = _register(
        registry, candidate_model, status="candidate", adaptation_type="recalibrate", parent_version_id=prod_version.version_id
    )
    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    _prewarm(evaluator, "throughput")

    disagreeing_llm = _FakeLLMClient(
        explanation="I disagree with the gate — this candidate should be REJECTED, the numbers are misleading.",
        key_observations="recommend REJECT despite the deterministic result",
    )
    agent = VerificationAgent(SETTINGS, registry, evaluator, llm_client=disagreeing_llm)

    eval_features = _synthetic_features(150, seed=102)
    eval_target = _synthetic_target(eval_features, 0.3, seed=103)
    candidate_preds = candidate_model.predict(eval_features)

    result = agent.verify(
        candidate_version,
        eval_features,
        eval_target,
        candidate_preds,
        production_component_factory=lambda: ThroughputModel.from_settings(SETTINGS),
    )

    # The deterministic gate wins: ACCEPT, and the candidate was genuinely promoted — the LLM's
    # stated disagreement had zero effect on either the decision or the actual registry action.
    assert result.decision == "ACCEPT"
    assert result.llm_used is True
    assert "REJECTED" in result.explanation  # the LLM's text is stored verbatim...
    assert registry.get_current_version("throughput").version_id == candidate_version.version_id  # ...but never acted on


def test_llm_disagreement_never_changes_the_deterministic_reject_decision(tmp_path):
    registry = _registry(tmp_path)
    production_model = _good_throughput_model(seed=110)
    prod_version = _register(registry, production_model, status="production")
    candidate_model = _bad_throughput_model(seed=111)
    candidate_version = _register(
        registry, candidate_model, status="candidate", adaptation_type="recalibrate", parent_version_id=prod_version.version_id
    )
    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    _prewarm(evaluator, "throughput")

    disagreeing_llm = _FakeLLMClient(
        explanation="I strongly recommend ACCEPTING this candidate — it is clearly superior to production.",
        key_observations="recommend ACCEPT despite the deterministic result",
    )
    agent = VerificationAgent(SETTINGS, registry, evaluator, llm_client=disagreeing_llm)

    eval_features = _synthetic_features(150, seed=112)
    eval_target = _synthetic_target(eval_features, 0.3, seed=113)
    candidate_preds = candidate_model.predict(eval_features)

    result = agent.verify(
        candidate_version,
        eval_features,
        eval_target,
        candidate_preds,
        production_component_factory=lambda: ThroughputModel.from_settings(SETTINGS),
    )

    assert result.decision == "REJECT"
    assert result.llm_used is True
    assert "ACCEPTING" in result.explanation  # the LLM's text is stored verbatim...
    assert registry.get_current_version("throughput").version_id == prod_version.version_id  # ...but never acted on
    assert registry.get_version("throughput", candidate_version.version_id).status == "rejected"


def test_llm_reasoning_schema_has_no_field_that_could_express_a_verdict():
    """Structural enforcement of rule 9: even if the LLM tried to express a contrary decision,
    there is nowhere in this schema to put one — `explanation`/`key_observations` are free text,
    never parsed for a verdict anywhere in `verification_agent.py`."""
    forbidden_field_names = {
        "decision", "verdict", "accept", "accepted", "reject", "rejected", "approve", "approved",
        "override", "should_accept", "should_promote", "should_reject", "recommendation", "result",
    }
    fields = set(_LLMVerificationReasoning.model_fields.keys())
    assert fields == {"explanation", "key_observations"}
    assert not (fields & forbidden_field_names)


def test_llm_failure_degrades_to_deterministic_explanation_decision_still_stands(tmp_path):
    registry = _registry(tmp_path)
    production_model = _bad_throughput_model(seed=120)
    prod_version = _register(registry, production_model, status="production")
    candidate_model = _good_throughput_model(seed=121)
    candidate_version = _register(
        registry, candidate_model, status="candidate", adaptation_type="recalibrate", parent_version_id=prod_version.version_id
    )
    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    _prewarm(evaluator, "throughput")

    failing_llm = _FakeLLMClient(raise_on_call=LLMClientError("simulated transport failure"))
    agent = VerificationAgent(SETTINGS, registry, evaluator, llm_client=failing_llm)

    eval_features = _synthetic_features(150, seed=122)
    eval_target = _synthetic_target(eval_features, 0.3, seed=123)
    candidate_preds = candidate_model.predict(eval_features)

    result = agent.verify(
        candidate_version,
        eval_features,
        eval_target,
        candidate_preds,
        production_component_factory=lambda: ThroughputModel.from_settings(SETTINGS),
    )
    assert result.decision == "ACCEPT"  # the deterministic decision is entirely unaffected
    assert result.llm_used is False
    assert "Deterministic gate decision: ACCEPT" in result.explanation
    assert registry.get_current_version("throughput").version_id == candidate_version.version_id


# --- RAG consultation (D2) -------------------------------------------------------------------------


def test_rag_context_is_retrieved_and_reaches_the_llm_prompt_when_available(tmp_path):
    registry = _registry(tmp_path)
    production_model = _bad_throughput_model(seed=130)
    prod_version = _register(registry, production_model, status="production")
    candidate_model = _good_throughput_model(seed=131)
    candidate_version = _register(
        registry, candidate_model, status="candidate", adaptation_type="recalibrate", parent_version_id=prod_version.version_id
    )
    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    _prewarm(evaluator, "throughput")

    chunk = RetrievedChunk(
        text="verification_delta is 0.01 per the enforced adaptation policy.",
        category="policies",
        source="internal:config/settings.yaml",
        document_id="adaptation_policies.md",
        chunk_index=0,
        version="abc123",
        distance=0.1,
    )
    rag_kb = _FakeRagKb([chunk])
    llm = _FakeLLMClient(explanation="Approved: fidelity clearly improved beyond policy.", key_observations="see policy")
    agent = VerificationAgent(SETTINGS, registry, evaluator, llm_client=llm)

    eval_features = _synthetic_features(150, seed=132)
    eval_target = _synthetic_target(eval_features, 0.3, seed=133)
    candidate_preds = candidate_model.predict(eval_features)

    result = agent.verify(
        candidate_version,
        eval_features,
        eval_target,
        candidate_preds,
        production_component_factory=lambda: ThroughputModel.from_settings(SETTINGS),
        rag_knowledge_base=rag_kb,
    )
    assert result.decision == "ACCEPT"
    assert rag_kb.queries, "RAG retrieve() was never called even though an available KB was supplied"
    assert llm.calls, "the LLM was never called"
    assert "verification_delta is 0.01" in llm.calls[0]  # retrieved chunk text reached the prompt


def test_rag_unavailable_degrades_gracefully_and_is_never_fabricated(tmp_path):
    registry = _registry(tmp_path)
    agent = VerificationAgent(SETTINGS, registry, FidelityEvaluator(SETTINGS.fidelity))
    context = agent._retrieve_rag_context("throughput", _UnavailableRagKb())  # noqa: SLF001 - focused unit test
    assert context == "not available"


def test_rag_none_is_reported_as_not_available(tmp_path):
    registry = _registry(tmp_path)
    agent = VerificationAgent(SETTINGS, registry, FidelityEvaluator(SETTINGS.fidelity))
    context = agent._retrieve_rag_context("throughput", None)  # noqa: SLF001 - focused unit test
    assert context == "not available"
