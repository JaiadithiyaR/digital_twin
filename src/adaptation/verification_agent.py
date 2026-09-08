"""Module 17 — Agentic Verification Agent (prompt.md §31-34).

This is the ACCEPT/REJECT gate every candidate produced by Modules 14 (Recalibration), 15
(Regeneration), or 16 (Expand-Scope) must pass before it can ever become production. Unlike those
three agents — which all stop at "produce an evaluated candidate" and explicitly never
self-promote — this agent's entire purpose is to decide, and then ACT on that decision:
`ModelRegistry.promote()` on ACCEPT, `ModelRegistry.reject()` on REJECT. This is the "trusted
deterministic application logic" CLAUDE.md §8 refers to ("Promotion is performed only by trusted
deterministic application logic after verification" — prompt.md line 506).

**Two layers, per prompt.md §31, kept structurally separate:**

1. **Deterministic layer** (`verify()`'s main body): recomputes RMSE/MAE/Wasserstein/MK-MMD and
   the composite FidelityScore *independently*, via the exact same `src.fidelity.evaluator.
   FidelityEvaluator` Module 12 exposes — never trusting an agent's own self-reported
   `fidelity_before`/`fidelity_after` for the actual decision (those exist on `ModelVersionMetadata`
   only as audit metadata from whichever agent produced the candidate). The primary criterion is
   exactly prompt.md §32: `FidelityScore_new > FidelityScore_old + delta`
   (`config.adaptation.verification_delta`, never hardcoded), plus a handful of other mandatory
   conditions (artifact present, interface identity consistent, evaluation actually ran, output
   finite) — ALL must pass, mirroring CLAUDE.md §8 "any failed mandatory condition -> REJECT".
2. **Agentic reasoning layer** (`_explain()`): an OPTIONAL LLM call producing a human-readable
   explanation — error patterns, drift context, RAG knowledge, general commentary — run strictly
   AFTER the deterministic `decision` variable is already final and has already been acted on
   (promote/reject already happened). Rule 9 (prompt.md §70: "LLMs cannot override deterministic
   acceptance criteria") is enforced STRUCTURALLY, not just by prompt wording: the LLM's
   structured-output schema, `_LLMVerificationReasoning`, has exactly two fields —
   `explanation`/`key_observations` — and genuinely no field anywhere that could express an
   accept/reject/verdict/override. Even if a malicious or confused LLM response's free-text
   `explanation` says "this should actually be REJECTED", no code anywhere reads that text for a
   decision signal — it is stored verbatim as `VerificationResult.explanation`, nothing more.
   `tests/unit/test_verification_agent.py::test_llm_disagreement_never_changes_the_deterministic_decision`
   proves this concretely: a fake LLM client is made to explicitly argue for the OPPOSITE outcome
   of what the deterministic gate computed, in both directions (ACCEPT-vs-LLM-says-reject and
   REJECT-vs-LLM-says-accept), and the final `decision` plus the actual registry promotion/
   rejection are shown to match the deterministic result every time.

**Candidate predictions are supplied as plain data, never as an untrusted live instance** — the
same "never import/execute LLM-generated code in this process" boundary Modules 15/16 already
established for the sandbox. `verify()` takes `candidate_predictions` (already-computed numeric
values, e.g. `RecalibrationResult.candidate_component.predict(...)` for a recalibration candidate,
or `RegenerationResult.sandbox_result.eval_predictions`/`ExpandScopeResult.sandbox_result.
eval_predictions` — both already plain data the sandbox subprocess handed back — for a regenerate/
expand-scope candidate) rather than a component instance to call `.predict()` on itself. This
keeps `VerificationAgent` uniformly agent-type-agnostic: it never needs to know or care whether
the candidate came from an in-process retrain or a sandboxed subprocess.

**The production ("before") baseline, in contrast, IS re-evaluated by this agent itself** — via an
optional `production_component_factory` that constructs a fresh instance of the CURRENT production
class, which this agent then loads the current production artifact into (a genuinely trusted,
already-production class — nothing untrusted is imported here) and predicts with, on the exact
same `eval_features`/`eval_target` the candidate was evaluated on (prompt.md §33: "before" and
"after" must use "the same comparable evaluation protocol"). `production_component_factory` is
optional because a genuinely new (expand-scope) component has no prior production version to
compare against at all — see the "no baseline" branch below.

**"No baseline" (expand-scope) is handled honestly, not by inventing a fake old score.** When
`ModelRegistry.get_current_version(component)` returns `None` (a brand-new component), the primary
`FidelityScore_new > FidelityScore_old + delta` criterion is structurally inapplicable — there is
no `FidelityScore_old`. The gate falls back to requiring only that the candidate's own
`FidelityScore` is well-defined (i.e. NOT `insufficient_history`, per Module 12's own "never
fabricate a score" rule) — a candidate whose fidelity can't yet be computed at all is REJECTed,
fail-safe, rather than blindly accepted for lack of a comparison. This mirrors Module 12's own
`insufficient_history` handling and Module 15/16's already-established `fidelity_before=None` for
genuinely-new candidates.

**Normalization reference sharing (prompt.md §0.10, Module 12's own documented hook)**: production
is evaluated with `update_window=True` (it genuinely joins the rolling history — this IS a real
evaluation of the currently-deployed model, not a throwaway probe) immediately before the candidate
is evaluated with `update_window=False` on the SAME `FidelityEvaluator` instance/window — the
established Module 14/15/16 pattern, now finally consumed by the module Module 12's own docstring
said it was built for.

**Concurrency lock (prompt.md §30) is explicitly OUT OF SCOPE for this turn** — this agent assumes
it is called with exclusive access to `candidate_version`'s component (whatever locking discipline
a future orchestration loop enforces); `ModelRegistry.promote()`/`.reject()` are individually
thread-safe (each holds the registry's own lock), but a full component-scoped adaptation lock
across an entire `verify()` call is a follow-up for whenever the main continuous loop (prompt.md
§39, not built yet) exists to actually run concurrent adaptations in the first place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.fidelity.evaluator import FidelityEvaluator, FidelityResult
from src.llm.anthropic_client import LLMClientError
from src.rag.rag_kb import RagUnavailableError
from src.registry.model_registry import ModelRegistry, ModelVersionMetadata

if TYPE_CHECKING:
    from typing import Callable

    from src.common.config import Settings
    from src.dt_models.base import DTComponent
    from src.llm.anthropic_client import AnthropicClient
    from src.rag.rag_kb import RagKnowledgeBase

logger = logging.getLogger(__name__)


class VerificationError(Exception):
    """Raised only when verification cannot meaningfully run at all (e.g. the given version is
    not actually a `"candidate"` awaiting a decision) — never raised just because a candidate
    looks bad. A bad candidate produces a REJECT `VerificationResult`, not an exception."""


class _LLMVerificationReasoning(BaseModel):
    """Structured output for the OPTIONAL agentic reasoning layer (prompt.md §31). See this
    module's docstring: deliberately has NO field anywhere that could express an accept/reject/
    verdict/override — the deterministic decision is already final by the time this is called."""

    explanation: str
    key_observations: str


@dataclass(frozen=True)
class DeterministicCheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class VerificationResult:
    component: str
    candidate_version_id: str
    decision: Literal["ACCEPT", "REJECT"]
    updated_version: ModelVersionMetadata  # candidate_version's registry record AFTER promote()/reject()
    fidelity_before: float | None
    fidelity_after: float | None
    fidelity_delta: float | None
    verification_delta: float
    fidelity_before_result: FidelityResult | None
    fidelity_after_result: FidelityResult | None
    checks: tuple[DeterministicCheckResult, ...]
    explanation: str
    llm_used: bool


class VerificationAgent:
    def __init__(
        self,
        settings: "Settings",
        model_registry: ModelRegistry,
        fidelity_evaluator: FidelityEvaluator | None = None,
        llm_client: "AnthropicClient | None" = None,
    ) -> None:
        self._settings = settings
        self._model_registry = model_registry
        self._fidelity_evaluator = fidelity_evaluator or FidelityEvaluator(settings.fidelity)
        self._llm_client = llm_client

    @classmethod
    def from_settings(
        cls, settings: "Settings", model_registry: ModelRegistry, llm_client: "AnthropicClient | None" = None
    ) -> "VerificationAgent":
        return cls(settings, model_registry, fidelity_evaluator=FidelityEvaluator(settings.fidelity), llm_client=llm_client)

    # --- main entrypoint ---------------------------------------------------------------------

    def verify(
        self,
        candidate_version: ModelVersionMetadata,
        eval_features: pd.DataFrame,
        eval_target: "pd.Series | np.ndarray",
        candidate_predictions: "pd.Series | np.ndarray",
        *,
        production_component_factory: "Callable[[], DTComponent] | None" = None,
        drift_context: str | None = None,
        rag_knowledge_base: "RagKnowledgeBase | None" = None,
    ) -> VerificationResult:
        """Verify `candidate_version` (must currently have `status == "candidate"`) and act on
        the result: promotes it to production on ACCEPT, marks it rejected on REJECT. Always
        returns a `VerificationResult` recording exactly why."""
        if candidate_version.status != "candidate":
            raise VerificationError(
                f"version {candidate_version.version_id!r} for component "
                f"{candidate_version.component!r} has status {candidate_version.status!r}, not "
                "'candidate' — verification only applies to a version awaiting a decision; it "
                "cannot be re-run on one that has already been decided."
            )
        component = candidate_version.component
        eval_target_arr = np.asarray(eval_target, dtype=float)
        candidate_preds_arr = np.asarray(candidate_predictions, dtype=float)

        checks: list[DeterministicCheckResult] = []

        # --- "candidate successfully loads" (prompt.md §32) — checked as data on disk, never by
        # importing/deserializing the candidate's class (see module docstring).
        artifact_path = self._model_registry.artifact_path(candidate_version)
        artifact_ok = artifact_path.exists() and artifact_path.stat().st_size > 0
        checks.append(
            DeterministicCheckResult(
                "candidate_artifact_present",
                artifact_ok,
                f"artifact at {artifact_path} " + ("exists and is non-empty" if artifact_ok else "is missing or empty"),
            )
        )

        # --- "required interface works" ---------------------------------------------------------
        interface_ok = True
        interface_detail = "no production_component_factory supplied to cross-check interface identity"
        if production_component_factory is not None:
            try:
                probe = production_component_factory()
                interface_ok = probe.COMPONENT_NAME == component and probe.OUTPUT_FIELD == candidate_version.output_field
                interface_detail = (
                    f"factory COMPONENT_NAME={probe.COMPONENT_NAME!r}/OUTPUT_FIELD={probe.OUTPUT_FIELD!r} vs "
                    f"candidate component={component!r}/output_field={candidate_version.output_field!r}"
                )
            except Exception as exc:  # a broken factory is a real failure, never a crash of verify()
                interface_ok = False
                interface_detail = f"production_component_factory() raised: {exc}"
        checks.append(DeterministicCheckResult("interface_identity_consistent", interface_ok, interface_detail))

        # --- "tests pass" / "evaluation succeeded" / "no invalid output" -------------------------
        length_ok = len(eval_target_arr) > 0 and len(eval_target_arr) == len(candidate_preds_arr)
        checks.append(
            DeterministicCheckResult(
                "evaluation_ran",
                length_ok,
                f"eval_target has {len(eval_target_arr)} row(s), candidate_predictions has {len(candidate_preds_arr)} row(s)",
            )
        )
        output_finite = length_ok and bool(np.all(np.isfinite(candidate_preds_arr)))
        checks.append(
            DeterministicCheckResult(
                "no_invalid_output",
                output_finite,
                "all candidate predictions are finite"
                if output_finite
                else "candidate predictions are empty, length-mismatched, or contain NaN/Inf",
            )
        )

        # --- recompute RMSE/MAE/Wasserstein/MK-MMD/FidelityScore — the ONLY deterministic
        # authority for this formula (Module 12); never trusts the candidate's own self-reported
        # evaluation_metrics/fidelity_before/fidelity_after for this decision.
        fidelity_before: float | None = None
        fidelity_after: float | None = None
        fidelity_before_result: FidelityResult | None = None
        fidelity_after_result: FidelityResult | None = None
        baseline_version = self._model_registry.get_current_version(component)

        if baseline_version is not None and production_component_factory is None:
            checks.append(
                DeterministicCheckResult(
                    "production_baseline_evaluable",
                    False,
                    f"a production version ({baseline_version.version_id}) exists for {component!r} but "
                    "no production_component_factory was supplied to evaluate it under the same "
                    "protocol — cannot verify improvement without fabricating a baseline score",
                )
            )
        elif baseline_version is not None and length_ok:
            try:
                production_instance = production_component_factory()  # type: ignore[misc]
                self._model_registry.load_artifact_into(production_instance, baseline_version)
                production_preds = np.asarray(production_instance.predict(eval_features), dtype=float)
                fidelity_before_result = self._fidelity_evaluator.evaluate(
                    component, eval_target_arr, production_preds, update_window=True
                )
                fidelity_before = fidelity_before_result.fidelity_score
                checks.append(
                    DeterministicCheckResult(
                        "production_baseline_evaluable",
                        True,
                        f"production version {baseline_version.version_id} evaluated on the same held-out set",
                    )
                )
            except Exception as exc:
                checks.append(
                    DeterministicCheckResult(
                        "production_baseline_evaluable",
                        False,
                        f"failed to evaluate current production version {baseline_version.version_id}: {exc}",
                    )
                )

        if length_ok and output_finite:
            fidelity_after_result = self._fidelity_evaluator.evaluate(
                component, eval_target_arr, candidate_preds_arr, update_window=False
            )
            fidelity_after = fidelity_after_result.fidelity_score

        verification_delta = self._settings.adaptation.verification_delta
        fidelity_delta: float | None = None

        if baseline_version is None:
            # No prior production version exists for this component at all (the expand-scope
            # case, or a component whose production version has since been rolled back to
            # nothing) — the primary regression criterion is structurally inapplicable. Accepting
            # requires only that the candidate's own FidelityScore be well-defined.
            if fidelity_after is not None:
                checks.append(
                    DeterministicCheckResult(
                        "fidelity_improved",
                        True,
                        f"no production baseline exists for {component!r} (adaptation_type="
                        f"{candidate_version.adaptation_type!r}) — accepting a genuinely new capability "
                        f"requires only a well-defined candidate FidelityScore, which is {fidelity_after:.4f}",
                    )
                )
            else:
                checks.append(
                    DeterministicCheckResult(
                        "fidelity_improved",
                        False,
                        "no production baseline exists AND the candidate's FidelityScore is not yet "
                        "computable (insufficient rolling history) — cannot accept without fabricating a score",
                    )
                )
        else:
            if fidelity_before is not None and fidelity_after is not None:
                fidelity_delta = fidelity_after - fidelity_before
                passed = fidelity_after > fidelity_before + verification_delta
                checks.append(
                    DeterministicCheckResult(
                        "fidelity_improved",
                        passed,
                        f"FidelityScore_new ({fidelity_after:.4f}) "
                        f"{'>' if passed else 'is NOT >'} FidelityScore_old ({fidelity_before:.4f}) + "
                        f"verification_delta ({verification_delta})",
                    )
                )
            else:
                checks.append(
                    DeterministicCheckResult(
                        "fidelity_improved",
                        False,
                        "FidelityScore before and/or after is not computable (insufficient rolling "
                        "history for one or both) — cannot verify improvement, fail-safe REJECT",
                    )
                )

        # --- THE deterministic gate: every mandatory condition must pass (CLAUDE.md §8: "any
        # failed mandatory condition -> REJECT"). This line is the entire ACCEPT/REJECT decision —
        # nothing below it can change `decision`.
        decision: Literal["ACCEPT", "REJECT"] = "ACCEPT" if all(c.passed for c in checks) else "REJECT"

        if decision == "ACCEPT":
            updated_version = self._model_registry.promote(component, candidate_version.version_id)
        else:
            updated_version = self._model_registry.reject(component, candidate_version.version_id)

        # --- Agentic reasoning layer — informational only, runs strictly AFTER `decision` is
        # final and already acted on. See module docstring / rule 9.
        explanation, llm_used = self._explain(
            candidate_version, decision, checks, fidelity_before, fidelity_after, drift_context, rag_knowledge_base
        )

        logger.info(
            "verification decision",
            extra={
                "component": "verification_agent",
                "component_name": component,
                "version_id": candidate_version.version_id,
                "decision": decision,
                "fidelity_before": fidelity_before,
                "fidelity_after": fidelity_after,
            },
        )
        return VerificationResult(
            component=component,
            candidate_version_id=candidate_version.version_id,
            decision=decision,
            updated_version=updated_version,
            fidelity_before=fidelity_before,
            fidelity_after=fidelity_after,
            fidelity_delta=fidelity_delta,
            verification_delta=verification_delta,
            fidelity_before_result=fidelity_before_result,
            fidelity_after_result=fidelity_after_result,
            checks=tuple(checks),
            explanation=explanation,
            llm_used=llm_used,
        )

    # --- agentic reasoning layer (informational only — see module docstring / rule 9) ------------

    def _explain(
        self,
        candidate_version: ModelVersionMetadata,
        decision: Literal["ACCEPT", "REJECT"],
        checks: list[DeterministicCheckResult],
        fidelity_before: float | None,
        fidelity_after: float | None,
        drift_context: str | None,
        rag_knowledge_base: "RagKnowledgeBase | None",
    ) -> tuple[str, bool]:
        deterministic_summary = "; ".join(f"{c.name}={'PASS' if c.passed else 'FAIL'} ({c.detail})" for c in checks)
        fallback_explanation = (
            f"Deterministic gate decision: {decision} for {candidate_version.component!r} version "
            f"{candidate_version.version_id!r} (adaptation_type={candidate_version.adaptation_type!r}). "
            f"fidelity_before={fidelity_before}, fidelity_after={fidelity_after}. Checks: {deterministic_summary}"
        )
        if self._llm_client is None:
            return fallback_explanation, False

        rag_context = self._retrieve_rag_context(candidate_version.component, rag_knowledge_base)
        prompt = (
            "You are the explanation-writing layer of an autonomous Digital Twin verification "
            "system (prompt.md §31). THE DETERMINISTIC ACCEPTANCE GATE HAS ALREADY MADE A FINAL "
            f"DECISION: {decision}. This decision is already recorded and acted on (the candidate "
            "has already been promoted or rejected) — you have no way to change it, no field to "
            "express a different verdict, and your response text will never be parsed for one. "
            "Your only job is to explain, for a human maintenance report, why this decision makes "
            f"sense given: component={candidate_version.component!r}, adaptation_type="
            f"{candidate_version.adaptation_type!r}, fidelity_before={fidelity_before}, "
            f"fidelity_after={fidelity_after}, verification_delta="
            f"{self._settings.adaptation.verification_delta}, deterministic checks="
            f"{deterministic_summary}, drift_context={drift_context or 'not provided'}, relevant "
            f"knowledge-base context={rag_context}. You MAY reason about why the candidate improved "
            "or failed, error distributions, drift context, model behaviour, the retrieved "
            "knowledge, or any additional contextual concerns."
        )
        try:
            result = self._llm_client.complete_structured(prompt, _LLMVerificationReasoning)
        except LLMClientError as exc:
            logger.warning(
                "verification LLM reasoning failed — falling back to the deterministic explanation",
                extra={"component": "verification_agent", "error": str(exc)},
            )
            return fallback_explanation, False
        return f"{result.explanation} ({result.key_observations})", True

    def _retrieve_rag_context(self, component: str, rag_knowledge_base: "RagKnowledgeBase | None") -> str:
        if rag_knowledge_base is None or not rag_knowledge_base.is_available:
            return "not available"
        try:
            chunks = rag_knowledge_base.retrieve(
                f"acceptance criteria and adaptation policy context for component {component}", top_k=3
            )
        except RagUnavailableError as exc:
            return f"unavailable: {exc}"
        if not chunks:
            return "no relevant knowledge retrieved"
        return " | ".join(f"[{c.category}/{c.source}] {c.text[:300]}" for c in chunks)
