"""Module 16 — Expand-Scope Agent (prompt.md §27).

Expand Scope is used when "network behaviour reveals a phenomenon/capability that the current DT
does not represent" — i.e. PPO (Module 13) selected action 2 for a phenomenon NO existing
component models at all, as opposed to Recalibrate/Regenerate (Modules 14/15), which both act on
an ALREADY-EXISTING component. This agent does not decide *whether* to expand scope — PPO already
decided; the caller-supplied `expand_scope_context` explains what triggered it, exactly like
Module 15's `drift_context`.

**Sixteen-step mapping (prompt.md §27), mirroring Modules 14/15's docstring convention**:
    1-5. inspect the component registry / interface / existing components / telemetry+features /
         RAG context -> `_build_context()` (identical spirit to Module 15's `_build_context`,
         reusing `DTModelRegistry` — Module 5 — this time, not just `ModelRegistry`)
    6-9. determine a suitable new component, derive its inputs/output, define feature-extraction
         requirements -> `_propose_design()`, a DESIGN LLM call producing a structured
         `_ProposedComponentDesign` (component_name, target_column, dependencies,
         required_features, purpose, feature_extraction_notes) — validated deterministically
         (`_validate_design`) before any code is generated: the proposed name must be genuinely
         NEW (no existing version), the target column must be real D1 telemetry, dependencies
         must be existing registered components. `output_field` is NOT LLM-proposed — it is
         DERIVED deterministically as `f"{target_column}_pred"`, the same convention every
         existing component already follows, removing one whole class of possible LLM mistakes.
    10-12. generate the implementation + "unit tests" (the sandbox's deterministic conformance
         check, exactly like Module 15 — not LLM-authored tests) + training logic (the generic
         `DTComponent.train()` interface, unchanged) -> `_generate_implementation()` +
         `SandboxExecutor.run_candidate()` — THE SAME sandbox Module 15 uses, unmodified.
    13. train the candidate -> inside the sandbox, exactly like Module 15.
    14. register the candidate -> `ModelRegistry.register_version_from_artifact()` (Module 14's
        registry, unmodified) with a component name NEVER SEEN BEFORE — this genuinely proves
        "the new component must be added dynamically through the registry" (prompt.md §27): no
        code anywhere in `ModelRegistry` enumerates or limits which component names may exist.
    15. evaluate it -> component-local metrics (from the sandbox) + real Module 12
        `fidelity_after` (there is deliberately no `fidelity_before` — a genuinely new capability
        has no prior version to compare against; `None` here is correct, not a bug, mirroring
        Module 12's own "never fabricate a score" principle applied to a case that structurally
        cannot have one).
    16. "submit it to verification" -> Module 17 doesn't exist yet; this agent's contract ends at
        returning a versioned, evaluated `ExpandScopeResult`, exactly like Modules 14/15.

**"Run any generated pipeline through the same sandbox as Module 15"**: `SandboxExecutor` and
`_sandbox_driver.py` are imported and used completely UNCHANGED from `src/sandbox/executor.py` —
this file adds zero sandbox code. The driver's conformance check (COMPONENT_NAME/OUTPUT_FIELD
must exactly match what's "expected") works identically whether "expected" came from an EXISTING
production version (Module 15) or a freshly-validated, not-yet-used design proposal (this
module) — the driver has no idea which agent called it, by design.

**Why this agent does NOT wire its candidate into the live `DTModelRegistry`/`DTOrchestrator`
(Module 5), even though prompt.md §27 says "register the candidate" as one of its own steps —
a deliberate, documented safety boundary, not an oversight**: doing so would require this
process to actually IMPORT (or `joblib.load()` the pickled artifact of) the LLM-generated class,
which is exactly what prompt.md §28/§70 rule 10 forbid before verification — and Module 17
(verification) doesn't exist yet. "Register... dynamically through the registry" is satisfied by
`ModelRegistry` (Concept C, Module 14's versioned artifact store — genuinely dynamic, no
hardcoded component list, proven directly by this module registering a component name that has
never existed before). Wiring a verified candidate into the LIVE orchestrator so it actually
SERVES predictions is a follow-up concern for whenever Module 17 and a vetted dynamic-loading
path exist — the exact same boundary Module 15 already documented for regenerated candidates.
`tests/integration/test_expand_scope_agent.py` separately proves `DTModelRegistry`/
`DTOrchestrator`'s OWN registration mechanism has no hardcoded component-count/name limit (using
a trusted, test-authored component — mirroring Module 5's own original test pattern — never the
raw LLM/sandbox output), so the CAPABILITY this module's output would eventually be promoted into
is itself demonstrated to be genuinely dynamic.
"""

from __future__ import annotations

import inspect
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
from pydantic import BaseModel, Field

from src.adaptation.data_selection import (
    DataSelectionError,
    iso,
    select_recent_window,
    time_split,
    with_dependency_ground_truth,
)
from src.fidelity.evaluator import FidelityEvaluator
from src.llm.anthropic_client import LLMClientError
from src.registry.model_registry import ModelRegistry, ModelVersionMetadata
from src.sandbox.executor import SandboxExecutor, SandboxResult

if TYPE_CHECKING:
    from src.common.config import Settings
    from src.dt_models.base import DTComponent
    from src.dt_models.d1_model_store import D1Store
    from src.dt_models.model_registry import DTModelRegistry
    from src.llm.anthropic_client import AnthropicClient

logger = logging.getLogger(__name__)

_NON_FEATURE_COLUMNS = frozenset(
    {"ue_id", "cell_id", "timestamp", "source", "quality_missing_fields", "quality_imputed_fields", "quality_out_of_range_fields"}
)


class ExpandScopeError(Exception):
    """Raised when scope-expansion cannot proceed, or fails to produce an accepted candidate
    after exhausting `config.adaptation.expand_scope.max_llm_iterations` attempts at either the
    design or implementation step — never silently skipped."""


class _ProposedComponentDesign(BaseModel):
    """Structured output for steps 6-9 (determine component, derive inputs/output, define
    feature-extraction requirements). `output_field` is deliberately NOT part of this schema —
    see module docstring: it is derived deterministically, never LLM-proposed."""

    component_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    target_column: str
    dependencies: tuple[str, ...] = ()
    required_features: tuple[str, ...]
    purpose: str
    feature_extraction_notes: str


class _GeneratedComponentCode(BaseModel):
    class_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    source_code: str
    reasoning: str


@dataclass(frozen=True)
class ExpandScopeResult:
    version: ModelVersionMetadata
    design: _ProposedComponentDesign
    sandbox_result: SandboxResult
    design_attempts: int
    implementation_attempts: int
    training_window: dict[str, Any]
    evaluation_window: dict[str, Any]
    fidelity_after: float | None


class ExpandScopeAgent:
    def __init__(
        self,
        settings: "Settings",
        d1_store: "D1Store",
        model_registry: ModelRegistry,
        dt_model_registry: "DTModelRegistry",
        llm_client: "AnthropicClient",
        sandbox_executor: SandboxExecutor,
        fidelity_evaluator: FidelityEvaluator | None = None,
    ) -> None:
        self._settings = settings
        self._d1_store = d1_store
        self._model_registry = model_registry
        self._dt_model_registry = dt_model_registry
        self._llm_client = llm_client
        self._sandbox = sandbox_executor
        self._fidelity_evaluator = fidelity_evaluator

    @classmethod
    def from_settings(
        cls,
        settings: "Settings",
        d1_store: "D1Store",
        model_registry: ModelRegistry,
        dt_model_registry: "DTModelRegistry",
        llm_client: "AnthropicClient",
    ) -> "ExpandScopeAgent":
        return cls(
            settings,
            d1_store,
            model_registry,
            dt_model_registry,
            llm_client,
            sandbox_executor=SandboxExecutor.from_settings(settings),
            fidelity_evaluator=FidelityEvaluator(settings.fidelity),
        )

    # --- main entrypoint ---------------------------------------------------------------------

    def expand_scope(
        self,
        example_component_factory: Callable[[], "DTComponent"],
        *,
        ue_id: str | None = None,
        cell_id: str | None = None,
        dependency_output_fields: dict[str, str] | None = None,
        window_hours: float | None = None,
        held_out_fraction: float = 0.2,
        expand_scope_context: str | None = None,
    ) -> ExpandScopeResult:
        """Design, implement, sandbox, and register a genuinely NEW DT component.

        `example_component_factory` constructs one EXISTING, trusted component — used only to
        show the LLM a concrete implementation pattern to follow (prompt.md §27 step 3, "inspect
        existing components"); it is never the candidate. `dependency_output_fields` maps any
        EXISTING component name the new one may declare as a dependency to that component's
        `OUTPUT_FIELD` (same convention as Modules 14/15) — only consulted for whichever
        dependencies the LLM's design actually proposes.
        """
        # 1. inspect the component registry
        existing_components = self._dt_model_registry.list_components()
        existing_component_names = {c.COMPONENT_NAME for c in existing_components}

        # 3 (partial). one concrete existing implementation, for pattern reference
        example_instance = example_component_factory()

        # Snapshot only — D1 is never touched again for the rest of this call (prompt.md §0.6/§0.8).
        history = self._d1_store.get_history(ue_id=ue_id, cell_id=cell_id)
        resolved_window_hours = (
            window_hours if window_hours is not None else self._settings.adaptation.expand_scope.training_window_hours
        )
        window_df = select_recent_window(history, resolved_window_hours)
        min_rows = self._settings.adaptation.expand_scope.min_training_rows
        if len(window_df) < min_rows:
            raise ExpandScopeError(
                f"only {len(window_df)} rows available in the last {resolved_window_hours:.2f}h, "
                f"need >= {min_rows} (config.adaptation.expand_scope.min_training_rows)"
            )

        # 2, 4, 5. interface + telemetry/feature context + RAG (unavailable, reported honestly)
        context = self._build_context(existing_components, example_instance, history, expand_scope_context)

        # 6-9. propose + validate a NEW component design (self-correcting on validation failure)
        design, design_attempts = self._propose_and_validate_design(context, existing_component_names, window_df)

        try:
            window_df = with_dependency_ground_truth(window_df, design.dependencies, dependency_output_fields)
        except DataSelectionError as exc:
            raise ExpandScopeError(str(exc)) from exc
        train_df, held_out_df = time_split(window_df, held_out_fraction)
        train_target = train_df[design.target_column]
        held_out_target = held_out_df[design.target_column]
        output_field = f"{design.target_column}_pred"

        # 10-13. generate implementation -> sandbox -> self-correct on rejection
        max_attempts = self._settings.adaptation.expand_scope.max_llm_iterations
        sandbox_result: SandboxResult | None = None
        generated: _GeneratedComponentCode | None = None
        previous_attempt: dict[str, Any] | None = None
        implementation_attempt = 0
        for implementation_attempt in range(1, max_attempts + 1):
            generated = self._generate_implementation(design, output_field, context, previous_attempt)
            sandbox_result = self._sandbox.run_candidate(
                source_code=generated.source_code,
                class_name=generated.class_name,
                expected_component_name=design.component_name,
                expected_output_field=output_field,
                target_column=design.target_column,
                train_features=train_df,
                train_target=train_target,
                eval_features=held_out_df,
                eval_target=held_out_target,
            )
            if sandbox_result.accepted:
                break
            logger.warning(
                "expand-scope candidate rejected by sandbox",
                extra={
                    "component": "expand_scope_agent",
                    "proposed_component_name": design.component_name,
                    "attempt": implementation_attempt,
                    "max_attempts": max_attempts,
                    "stage": sandbox_result.stage,
                    "error": sandbox_result.error,
                },
            )
            previous_attempt = {
                "class_name": generated.class_name,
                "source_code": generated.source_code,
                "stage": sandbox_result.stage,
                "error": sandbox_result.error,
                "stderr": sandbox_result.stderr[-2000:],
            }

        assert sandbox_result is not None and generated is not None
        if not sandbox_result.accepted:
            raise ExpandScopeError(
                f"expand-scope for proposed component {design.component_name!r} produced no accepted "
                f"candidate after {max_attempts} attempt(s): stage={sandbox_result.stage} error={sandbox_result.error}"
            )

        # 15 (partial). real Module 12 fidelity_after — no fidelity_before: there is no prior
        # version of a genuinely new component to compare against.
        fidelity_after = None
        if self._fidelity_evaluator is not None and len(held_out_target) > 0:
            try:
                after = self._fidelity_evaluator.evaluate(
                    design.component_name, held_out_target, sandbox_result.eval_predictions, update_window=True
                )
                fidelity_after = after.fidelity_score
            except Exception as exc:
                logger.warning(
                    "fidelity_after computation failed, leaving it None",
                    extra={"component": "expand_scope_agent", "component_name": design.component_name, "error": str(exc)},
                )

        # 14. register the candidate — a component name that has NEVER existed before, proving
        # ModelRegistry's registration is genuinely dynamic (prompt.md §27).
        training_window = {
            "start": iso(train_df["timestamp"].min()) if len(train_df) else None,
            "end": iso(train_df["timestamp"].max()) if len(train_df) else None,
            "n_rows": len(train_df),
            "window_hours": resolved_window_hours,
            "provenance": "d1_history",
            "ue_id": ue_id,
            "cell_id": cell_id,
        }
        evaluation_window = {
            "start": iso(held_out_df["timestamp"].min()) if len(held_out_df) else None,
            "end": iso(held_out_df["timestamp"].max()) if len(held_out_df) else None,
            "n_rows": len(held_out_df),
            "provenance": "d1_history",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_tmp = Path(tmp_dir) / "artifact.joblib"
            artifact_tmp.write_bytes(sandbox_result.artifact_bytes)
            source_tmp = Path(tmp_dir) / "source.py"
            source_tmp.write_text(generated.source_code)

            version = self._model_registry.register_version_from_artifact(
                component=design.component_name,
                model_class=generated.class_name,
                artifact_source_path=artifact_tmp,
                source_code_path=source_tmp,
                dependencies=sandbox_result.dependencies or design.dependencies,
                feature_schema=sandbox_result.required_features or design.required_features,
                output_field=output_field,
                adaptation_type="expand_scope",
                parent_version_id=None,  # genuinely new — no parent version exists
                training_window=training_window,
                evaluation_window=evaluation_window,
                evaluation_metrics=sandbox_result.metrics or {},
                fidelity_before=None,
                fidelity_after=fidelity_after,
                status="candidate",
                llm_metadata={
                    "model": self._settings.llm.model,
                    "design_purpose": design.purpose,
                    "design_feature_extraction_notes": design.feature_extraction_notes,
                    "implementation_reasoning": generated.reasoning,
                    "design_attempts": design_attempts,
                    "implementation_attempts": implementation_attempt,
                },
            )

        logger.info(
            "expand-scope candidate produced",
            extra={
                "component": "expand_scope_agent",
                "component_name": design.component_name,
                "version_id": version.version_id,
                "design_attempts": design_attempts,
                "implementation_attempts": implementation_attempt,
                "fidelity_after": fidelity_after,
            },
        )
        return ExpandScopeResult(
            version=version,
            design=design,
            sandbox_result=sandbox_result,
            design_attempts=design_attempts,
            implementation_attempts=implementation_attempt,
            training_window=training_window,
            evaluation_window=evaluation_window,
            fidelity_after=fidelity_after,
        )

    # --- context gathering (steps 1-5) --------------------------------------------------------

    def _build_context(
        self,
        existing_components: list["DTComponent"],
        example_instance: "DTComponent",
        history: pd.DataFrame,
        expand_scope_context: str | None,
    ) -> dict[str, Any]:
        try:
            example_source = inspect.getsource(type(example_instance))
        except (OSError, TypeError):
            example_source = "<source unavailable>"

        available_columns = [c for c in history.columns if c not in _NON_FEATURE_COLUMNS]
        feature_statistics = history[available_columns].describe().to_dict()

        return {
            "existing_components": [
                {
                    "component_name": c.COMPONENT_NAME,
                    "dependencies": list(c.DEPENDENCIES),
                    "required_features": list(c.REQUIRED_FEATURES),
                    "output_field": c.OUTPUT_FIELD,
                }
                for c in existing_components
            ],
            "example_component_source": example_source,
            "feature_statistics": feature_statistics,
            "available_columns": available_columns,
            "expand_scope_context": expand_scope_context or "not provided",
            "rag_context": "not available — Module 18 (RAG Knowledge Base) is not built yet",
        }

    # --- steps 6-9: design proposal + deterministic validation --------------------------------

    def _propose_and_validate_design(
        self, context: dict[str, Any], existing_component_names: set[str], window_df: pd.DataFrame
    ) -> tuple[_ProposedComponentDesign, int]:
        max_attempts = self._settings.adaptation.expand_scope.max_llm_iterations
        available_columns = set(window_df.columns) - _NON_FEATURE_COLUMNS
        previous_attempt: dict[str, Any] | None = None
        last_error: str | None = None

        for attempt in range(1, max_attempts + 1):
            design = self._propose_design(context, existing_component_names, previous_attempt)
            error = self._validate_design(design, existing_component_names, available_columns)
            if error is None:
                return design, attempt
            last_error = error
            logger.warning(
                "expand-scope design proposal rejected by deterministic validation",
                extra={"component": "expand_scope_agent", "attempt": attempt, "error": error},
            )
            previous_attempt = {"design": design, "error": error}

        raise ExpandScopeError(f"no valid component design produced after {max_attempts} attempt(s): {last_error}")

    def _validate_design(
        self, design: _ProposedComponentDesign, existing_component_names: set[str], available_columns: set[str]
    ) -> str | None:
        if design.component_name in existing_component_names:
            return f"component_name {design.component_name!r} already exists — expand-scope must propose a NEW component"
        if self._model_registry.list_versions(design.component_name):
            return f"component_name {design.component_name!r} already has registered versions — not genuinely new"
        if design.target_column not in available_columns:
            return f"target_column {design.target_column!r} is not a real D1 telemetry column"
        unknown_deps = set(design.dependencies) - existing_component_names
        if unknown_deps:
            return f"dependencies {sorted(unknown_deps)} are not existing registered components"
        if not design.required_features:
            return "required_features must not be empty"
        unknown_features = set(design.required_features) - available_columns
        if unknown_features:
            return f"required_features {sorted(unknown_features)} are not available D1 columns"
        return None

    def _propose_design(
        self, context: dict[str, Any], existing_component_names: set[str], previous_attempt: dict[str, Any] | None
    ) -> _ProposedComponentDesign:
        prompt = (
            "You are proposing a genuinely NEW prediction component for a 5G network digital "
            "twin (prompt.md §27 — 'network behaviour reveals a phenomenon/capability the "
            "current DT does not represent'). Context for why this was triggered: "
            f"{context['expand_scope_context']}\n\n"
            f"Existing components (do NOT propose one of these — propose something new): "
            f"{json.dumps(context['existing_components'])}\n"
            f"Available raw/dependency columns you may use as target_column or required_features: "
            f"{context['available_columns']}\n"
            f"Recent feature statistics (pandas describe()): "
            f"{json.dumps(context['feature_statistics'], default=str)[:4000]}\n"
            f"Relevant RAG context: {context['rag_context']}\n\n"
            "Propose: a NEW component_name (lowercase snake_case, not already used), a "
            "target_column (an existing raw telemetry column not already predicted by an "
            "existing component's output), optional dependencies (existing component names "
            "whose predictions this new component may use as input), required_features (raw/"
            "dependency columns this component reads directly), a one-sentence purpose, and "
            "feature_extraction_notes describing any derived features it should compute "
            "internally (e.g. aggregates) — mirroring how existing components already do this "
            "internally, not as a separate pipeline stage."
        )
        if previous_attempt is not None:
            prompt += (
                f"\n\nYour previous proposal ({previous_attempt['design'].component_name!r}) was "
                f"REJECTED: {previous_attempt['error']}\nPropose a corrected design."
            )
        try:
            return self._llm_client.complete_structured(prompt, _ProposedComponentDesign)
        except LLMClientError as exc:
            raise ExpandScopeError(f"LLM component-design proposal failed: {exc}") from exc

    # --- steps 10-12: implementation generation ------------------------------------------------

    def _generate_implementation(
        self,
        design: _ProposedComponentDesign,
        output_field: str,
        context: dict[str, Any],
        previous_attempt: dict[str, Any] | None,
    ) -> _GeneratedComponentCode:
        interface_contract = (
            "The class MUST subclass `src.dt_models.base.DTComponent` and implement: "
            f"`COMPONENT_NAME` (class attribute, MUST equal exactly {design.component_name!r}), "
            f"`DEPENDENCIES` (MUST equal exactly {list(design.dependencies)!r}), "
            f"`REQUIRED_FEATURES` (MUST include exactly {list(design.required_features)!r} — the "
            f"columns already agreed in the design), `OUTPUT_FIELD` (class attribute, MUST equal "
            f"exactly {output_field!r}), `is_trained` (property), "
            "`train(self, features: pd.DataFrame, targets: pd.Series) -> None`, "
            "`predict(self, features: pd.DataFrame) -> pd.Series`, "
            "`evaluate(self, features, targets) -> dict[str, float]`, `save(self, path)`, "
            "`load(self, path)`. The constructor must accept no required arguments."
        )
        prompt = (
            f"Implement the new component design agreed below as a COMPLETE, self-contained "
            f"Python module (all needed imports included).\n\n"
            f"Design: component_name={design.component_name!r}, target_column="
            f"{design.target_column!r}, dependencies={list(design.dependencies)!r}, "
            f"required_features={list(design.required_features)!r}, purpose={design.purpose!r}, "
            f"feature_extraction_notes={design.feature_extraction_notes!r}\n\n"
            f"Example of an existing component's implementation pattern to follow:\n"
            f"```python\n{context['example_component_source']}\n```\n\n"
            f"Interface contract: {interface_contract}"
        )
        if previous_attempt is not None:
            prompt += (
                f"\n\nYour previous attempt (class {previous_attempt['class_name']!r}) was "
                f"REJECTED by the sandbox at stage {previous_attempt['stage']!r} with error: "
                f"{previous_attempt['error']}\n"
                f"stderr (truncated): {previous_attempt['stderr']}\n\n"
                f"Previous source code:\n```python\n{previous_attempt['source_code']}\n```\n\n"
                f"Fix the issue and produce a corrected, complete module."
            )
        try:
            return self._llm_client.complete_structured(prompt, _GeneratedComponentCode)
        except LLMClientError as exc:
            raise ExpandScopeError(f"LLM implementation generation failed: {exc}") from exc
