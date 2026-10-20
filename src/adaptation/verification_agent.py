"""Agentic Verification Agent (Agent 5) — the ACCEPT/REJECT gate every candidate from
Recalibration/Regeneration/Expand-Scope must pass before becoming production. Unlike those three
(which stop at "produce an evaluated candidate"), this agent acts: `ModelRegistry.promote()` on
ACCEPT, `.reject()` on REJECT — the one trusted deterministic promoter in the system.

Two layers, kept structurally separate:

1. **Deterministic** (`verify()`'s main body): independently recomputes RMSE/MAE/Wasserstein/
   MK-MMD -> composite FidelityScore via the real `FidelityEvaluator` — never trusts an agent's
   self-reported before/after. Primary criterion:

       FidelityScore_new > FidelityScore_old + delta      (delta = config.adaptation.verification_delta)

   plus mandatory checks (artifact present, interface consistent, evaluation ran, output finite)
   — any failure -> REJECT.
2. **Agentic reasoning** (`_explain()`): an optional LLM call, run strictly AFTER the
   deterministic decision is already final and acted on. Its schema (`_LLMVerificationReasoning`)
   has exactly two fields — `explanation`/`key_observations` — structurally no field that could
   express a verdict, so the LLM can never override the gate.

Candidate predictions are supplied as plain data (never a live instance to `.predict()` on) —
the sandbox boundary Regeneration/Expand-Scope already established. The production ("before")
baseline IS re-evaluated here, via an optional `production_component_factory`, on the exact same
held-out data as the candidate. A brand-new (expand-scope) component has no production model: its
"old" side is a naive reference predictor supplied as data (persistence `y_{t-1}`), and the same
`new > old + delta` gate applies, both scored against one shared, isolated normalization
reference (`_score_against_reference`). Without a reference baseline, the gate requires only a
well-defined candidate FidelityScore — REJECT fail-safe otherwise, never a fabricated old score.

Normalization sharing: production is evaluated with `update_window=True` immediately before the
candidate with `update_window=False`, on the same `FidelityEvaluator` window — production and
candidate must be comparable.

Concurrency: this agent assumes exclusive access to the candidate's component; the
component-scoped lock lives in the orchestrator, not here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.fidelity.evaluator import FidelityEvaluator, FidelityResult
from src.llm.google_client import LLMClientError
from src.rag.rag_kb import RagUnavailableError
from src.registry.model_registry import ModelRegistry, ModelVersionMetadata

if TYPE_CHECKING:
    from typing import Callable

    from src.common.config import Settings
    from src.dt_models.base import DTComponent
    from src.llm.google_client import GoogleClient
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
        llm_client: "GoogleClient | None" = None,
        static_prefix: str | None = None,
    ) -> None:
        self._settings = settings
        self._model_registry = model_registry
        self._fidelity_evaluator = fidelity_evaluator or FidelityEvaluator(settings.fidelity)
        self._llm_client = llm_client
        # Phase 13 §3.4: "explanation" role — this call runs strictly AFTER the deterministic
        # decision is already final/acted-on (see this module's own docstring), so it needs no
        # deep reasoning, only clear prose.
        self._effort = settings.llm.effort_by_role.explanation
        # Phase 13 §3.3 — see RegenerationAgent's own entry for the full rationale: `None` for
        # every non-baseline caller, which keeps this agent's prompt content byte-for-byte
        # unchanged from before this feature existed.
        self._static_prefix = static_prefix

    @classmethod
    def from_settings(
        cls,
        settings: "Settings",
        model_registry: ModelRegistry,
        llm_client: "GoogleClient | None" = None,
        static_prefix: str | None = None,
    ) -> "VerificationAgent":
        return cls(
            settings,
            model_registry,
            fidelity_evaluator=FidelityEvaluator(settings.fidelity),
            llm_client=llm_client,
            static_prefix=static_prefix,
        )

    # --- main entrypoint ---------------------------------------------------------------------

    def verify(
        self,
        candidate_version: ModelVersionMetadata,
        eval_features: pd.DataFrame,
        eval_target: "pd.Series | np.ndarray",
        candidate_predictions: "pd.Series | np.ndarray",
        *,
        production_component_factory: "Callable[[], DTComponent] | None" = None,
        baseline_predictions: "pd.Series | np.ndarray | None" = None,
        drift_context: str | None = None,
        rag_knowledge_base: "RagKnowledgeBase | None" = None,
        cache_key: str | None = None,
    ) -> VerificationResult:
        """Verify `candidate_version` (must currently have `status == "candidate"`) and act on
        the result: promotes it to production on ACCEPT, marks it rejected on REJECT. Always
        returns a `VerificationResult` recording exactly why.

        `baseline_predictions` (plain data, used only when the component has NO production version
        — a brand-new expand-scope component): a naive reference predictor on the same rows,
        normally persistence `y_{t-1}`. The candidate must then beat it by `verification_delta`
        under the unchanged fidelity formula — see `_score_against_reference`."""
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

        length_ok = len(eval_target_arr) > 0 and len(eval_target_arr) == len(candidate_preds_arr)
        baseline_version = self._model_registry.get_current_version(component)

        # --- production ("before") predictions on the SAME held-out rows ------------------------
        production_preds: np.ndarray | None = None
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
                if len(production_preds) != len(eval_target_arr):
                    raise ValueError(f"returned {len(production_preds)} predictions for {len(eval_target_arr)} rows")
            except Exception as exc:
                production_preds = None
                checks.append(
                    DeterministicCheckResult(
                        "production_baseline_evaluable",
                        False,
                        f"failed to evaluate current production version {baseline_version.version_id}: {exc}",
                    )
                )

        # --- which held-out rows can be scored at all ----------------------------------------------
        # A row with no ground truth can't be scored, and a row the production model itself can't
        # predict (a residual component's legitimately-NaN first tick of a session, no y_{t-1}
        # anchor) isn't comparable — both are excluded for BOTH models alike, the same masking the
        # live fidelity cycle applies. The candidate must still be finite on every scored row.
        reference_preds: np.ndarray | None = None
        if baseline_version is None and baseline_predictions is not None:
            reference = np.asarray(baseline_predictions, dtype=float)
            if len(reference) == len(eval_target_arr):
                reference_preds = reference
            else:
                checks.append(
                    DeterministicCheckResult(
                        "reference_baseline_evaluable", False,
                        f"baseline_predictions has {len(reference)} row(s) for {len(eval_target_arr)} target row(s)",
                    )
                )

        scored_mask = np.isfinite(eval_target_arr) if length_ok else np.zeros(0, dtype=bool)
        if production_preds is not None:
            scored_mask = scored_mask & np.isfinite(production_preds)
        if reference_preds is not None and length_ok:
            scored_mask = scored_mask & np.isfinite(reference_preds)
        n_scored = int(scored_mask.sum())

        # --- "tests pass" / "evaluation succeeded" / "no invalid output" -------------------------
        evaluation_ok = length_ok and n_scored >= 2  # MK-MMD needs >= 2 samples per side
        checks.append(
            DeterministicCheckResult(
                "evaluation_ran",
                evaluation_ok,
                f"eval_target has {len(eval_target_arr)} row(s), candidate_predictions has "
                f"{len(candidate_preds_arr)} row(s), {n_scored} scorable row(s) (finite ground truth"
                + (" and production prediction" if production_preds is not None else "")
                + ")",
            )
        )
        output_finite = evaluation_ok and bool(np.all(np.isfinite(candidate_preds_arr[scored_mask])))
        checks.append(
            DeterministicCheckResult(
                "no_invalid_output",
                output_finite,
                "all candidate predictions on scorable rows are finite"
                if output_finite
                else "candidate predictions are empty, length-mismatched, or contain NaN/Inf on scorable rows",
            )
        )

        # --- recompute RMSE/MAE/Wasserstein/MK-MMD/FidelityScore — the ONLY deterministic
        # authority for this formula (Module 12); never trusts the candidate's own self-reported
        # evaluation_metrics/fidelity_before/fidelity_after for this decision.
        fidelity_before: float | None = None
        fidelity_after: float | None = None
        fidelity_before_result: FidelityResult | None = None
        fidelity_after_result: FidelityResult | None = None
        scored_target = eval_target_arr[scored_mask] if evaluation_ok else None

        if production_preds is not None and evaluation_ok:
            try:
                fidelity_before_result = self._fidelity_evaluator.evaluate(
                    component, scored_target, production_preds[scored_mask], update_window=True
                )
                fidelity_before = fidelity_before_result.fidelity_score
                checks.append(
                    DeterministicCheckResult(
                        "production_baseline_evaluable",
                        True,
                        f"production version {baseline_version.version_id} evaluated on the same "  # type: ignore[union-attr]
                        f"{n_scored} held-out row(s)",
                    )
                )
            except Exception as exc:
                checks.append(
                    DeterministicCheckResult(
                        "production_baseline_evaluable",
                        False,
                        f"failed to evaluate current production version {baseline_version.version_id}: {exc}",  # type: ignore[union-attr]
                    )
                )

        if output_finite:
            try:
                fidelity_after_result = self._fidelity_evaluator.evaluate(
                    component, scored_target, candidate_preds_arr[scored_mask], update_window=False
                )
                fidelity_after = fidelity_after_result.fidelity_score
            except Exception as exc:
                checks.append(DeterministicCheckResult("candidate_evaluable", False, f"fidelity evaluation failed: {exc}"))

        verification_delta = self._settings.adaptation.verification_delta
        fidelity_delta: float | None = None

        if baseline_version is None and reference_preds is not None:
            # A brand-new component: there is no production model, so the "old" side of the gate
            # is the naive reference predictor (persistence). Both are scored on the same rows
            # against ONE shared, isolated normalization reference — the live evaluator has no
            # history for a never-before-seen component, which would make the gate unpassable.
            scored = (
                self._score_against_reference(component, scored_target, reference_preds[scored_mask], candidate_preds_arr[scored_mask])
                if output_finite else None
            )
            if scored is not None:
                fidelity_before_result, fidelity_after_result = scored
                fidelity_before = fidelity_before_result.fidelity_score
                fidelity_after = fidelity_after_result.fidelity_score
            checks.append(
                DeterministicCheckResult(
                    "reference_baseline_evaluable", fidelity_before is not None,
                    f"no production version exists for {component!r}; reference (persistence) baseline scored on "
                    f"{n_scored} row(s)" if fidelity_before is not None else
                    "reference (persistence) baseline could not be scored (too few rows for a normalization reference)",
                )
            )
            if fidelity_before is not None and fidelity_after is not None:
                fidelity_delta = fidelity_after - fidelity_before
                passed = fidelity_after > fidelity_before + verification_delta
                checks.append(
                    DeterministicCheckResult(
                        "fidelity_improved", passed,
                        f"new component FidelityScore ({fidelity_after:.4f}) {'>' if passed else 'is NOT >'} "
                        f"persistence baseline y_(t-1) ({fidelity_before:.4f}) + verification_delta ({verification_delta})",
                    )
                )
            else:
                checks.append(
                    DeterministicCheckResult(
                        "fidelity_improved", False,
                        "FidelityScore of the new component and/or its reference baseline is not computable — fail-safe REJECT",
                    )
                )
        elif baseline_version is None:
            # No prior production version exists for this component at all (the expand-scope
            # case, or a component whose production version has since been rolled back to
            # nothing) and no reference baseline was supplied — the primary regression criterion
            # is structurally inapplicable. Accepting requires only that the candidate's own
            # FidelityScore be well-defined.
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
            candidate_version, decision, checks, fidelity_before, fidelity_after, drift_context, rag_knowledge_base, cache_key=cache_key
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

    def _score_against_reference(
        self, component: str, y_true: np.ndarray, reference: np.ndarray, candidate: np.ndarray
    ) -> tuple[FidelityResult, FidelityResult] | None:
        """FidelityScore of a reference predictor and of a candidate on the same rows, normalized
        against ONE shared rolling window built from both models' errors on consecutive chunks of
        those rows (alternately, so neither side defines the scale alone). Uses a fresh, isolated
        `FidelityEvaluator` — the formula is Module 12's, unchanged; only the reference window is
        built here because a new component has no live history. None if there are too few rows
        for `min_history_for_normalization` chunks of >= 2 rows each (MK-MMD needs 2 per side)."""
        n_chunks = self._settings.fidelity.min_history_for_normalization
        if len(y_true) < 2 * n_chunks:
            return None
        evaluator = FidelityEvaluator(self._settings.fidelity)
        for idx in np.array_split(np.arange(len(y_true)), n_chunks):
            evaluator.evaluate(component, y_true[idx], reference[idx], update_window=True)
            evaluator.evaluate(component, y_true[idx], candidate[idx], update_window=True)
        before = evaluator.evaluate(component, y_true, reference, update_window=False)
        after = evaluator.evaluate(component, y_true, candidate, update_window=False)
        if before.fidelity_score is None or after.fidelity_score is None:
            return None
        return before, after

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
        *,
        cache_key: str | None = None,
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
        sent_prompt = prompt if cache_key is not None else (f"{self._static_prefix}\n\n{prompt}" if self._static_prefix else prompt)
        try:
            result = self._llm_client.complete_structured(sent_prompt, _LLMVerificationReasoning, effort=self._effort, cache_key=cache_key)
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
