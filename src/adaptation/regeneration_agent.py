"""Module 15 — Regeneration Agent (prompt.md §24-26).

Regeneration is used when "the current model architecture/pipeline is no longer capable of
representing the changed behaviour" — i.e. PPO (Module 13) selected action 1 for an
already-production component whose STRUCTURE, not just its learned parameters, needs to change.
This is the primary LLM code-generation component (prompt.md §24). Like Module 14, this agent
does not decide *whether* to regenerate — PPO already decided that; it only executes the strategy.

**Nine-step mapping, mirroring Module 14's docstring convention**:
    1. receive the affected component          -> `component_factory` param
    2. inspect current model + its source code -> `ModelRegistry.get_current_version()` +
       `inspect.getsource()` on the currently-registered production class (prompt.md §25)
    3. gather regeneration context              -> `_build_context()`: current metadata, recent
       feature statistics, error patterns (residuals of the CURRENT production model on a held-out
       window), fidelity metrics, drift context (caller-supplied), RAG context (Module 18 not
       built yet — explicitly reported as unavailable, never faked)
    4. generate candidate source via the LLM     -> `_generate_candidate()`, using the centralized
       `AnthropicClient` built for this project ("Use the Anthropic client from [the LLM
       infrastructure turn] where LLM-assisted pipeline generation/reasoning is needed")
    5. sandbox the candidate                     -> `SandboxExecutor.run_candidate()`
       (`src/sandbox/executor.py`) — syntax/import/conformance/train/evaluate, prompt.md §26's
       exact pipeline, ALL deterministic, non-LLM code judging LLM-generated code
    6. self-correct on rejection                 -> up to `config.adaptation.regeneration.
       max_llm_iterations` attempts, feeding the sandbox's rejection stage/error back to the LLM
    7. register the accepted candidate as a new version -> `ModelRegistry.
       register_version_from_artifact()` (Module 14's registry, extended for sandboxed candidates
       whose class this process never imports)
    8. evaluate it (already done inside the sandbox, plus real Module 12 fidelity before/after
       computed here in the parent from data, never from untrusted code)
    9. "pass it to verification" — Module 17 doesn't exist yet; this agent's contract ends at
       returning a versioned, evaluated `RegenerationResult`, exactly like Module 14.

**Rule 10 (prompt.md §70): "LLM-generated code is never production code until sandboxed and
verified."** Enforced structurally, not by convention: this process NEVER imports, `exec()`s, or
otherwise executes the LLM's generated source directly. `SandboxExecutor.run_candidate()` is the
ONLY thing that ever runs it, as a separate OS process with a minimal environment (see
`src/sandbox/executor.py`'s module docstring for the full isolation model and its honestly-
documented limits). This agent only ever receives back: acceptance/rejection + a structured
reason, component-local metrics, the candidate's OWN declared dependencies/features, raw
prediction VALUES on the held-out set (data, not code — safe to read into this process), and the
trained artifact's raw BYTES (also just data — this process reconstructs nothing executable from
it; it hands the bytes straight to the registry, which writes them to a file). A rejected
candidate never reaches `ModelRegistry` at all — production is provably untouched by definition,
not merely by discipline.

**Continuous operation during adaptation (prompt.md §0.6/§0.8) — the same structural guarantee as
Module 14, now proven under a heavier real workload**: this agent takes exactly one `D1Store.
get_history()` snapshot per `regenerate()` call and never calls a D1 write method; every LLM call
and every sandboxed subprocess run operates entirely on that already-isolated in-memory/file
snapshot. `tests/integration/test_regeneration_agent.py` proves telemetry keeps synchronizing
into D1 for the ENTIRE duration of a real `regenerate()` call — LLM call(s) plus real sandboxed
subprocess training — not just during a quick in-memory retrain as Module 14's proof covered.
"""

from __future__ import annotations

import inspect
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
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
    from src.llm.anthropic_client import AnthropicClient

logger = logging.getLogger(__name__)

# Columns present on every D1 telemetry row that are never legitimate model inputs (identity,
# provenance, and quality-audit metadata) — excluded from the feature-statistics summary and the
# "available columns" list handed to the LLM, so it isn't tempted to declare them as features.
_NON_FEATURE_COLUMNS = frozenset(
    {"ue_id", "cell_id", "timestamp", "source", "quality_missing_fields", "quality_imputed_fields", "quality_out_of_range_fields"}
)


class RegenerationError(Exception):
    """Raised when regeneration cannot proceed at all, or fails to produce an accepted candidate
    after exhausting `config.adaptation.regeneration.max_llm_iterations` attempts — never
    silently skipped or downgraded to a partial result."""


class _GeneratedComponentCode(BaseModel):
    """Structured output schema for the LLM code-generation call — genuinely structured output
    (prompt.md §26 "use strict structured output where possible"), not free-form text parsed with
    regex. `class_name` is validated as a legal Python identifier at the schema level (defense in
    depth — the sandbox's own conformance check would catch a bad one too)."""

    class_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    source_code: str
    reasoning: str


@dataclass(frozen=True)
class RegenerationResult:
    version: ModelVersionMetadata
    sandbox_result: SandboxResult
    attempts: int
    training_window: dict[str, Any]
    evaluation_window: dict[str, Any]
    fidelity_before: float | None
    fidelity_after: float | None


class RegenerationAgent:
    def __init__(
        self,
        settings: "Settings",
        d1_store: "D1Store",
        model_registry: ModelRegistry,
        llm_client: "AnthropicClient",
        sandbox_executor: SandboxExecutor,
        fidelity_evaluator: FidelityEvaluator | None = None,
    ) -> None:
        self._settings = settings
        self._d1_store = d1_store
        self._model_registry = model_registry
        self._llm_client = llm_client
        self._sandbox = sandbox_executor
        self._fidelity_evaluator = fidelity_evaluator

    @classmethod
    def from_settings(
        cls, settings: "Settings", d1_store: "D1Store", model_registry: ModelRegistry, llm_client: "AnthropicClient"
    ) -> "RegenerationAgent":
        return cls(
            settings,
            d1_store,
            model_registry,
            llm_client,
            sandbox_executor=SandboxExecutor.from_settings(settings),
            fidelity_evaluator=FidelityEvaluator(settings.fidelity),
        )

    # --- main entrypoint ---------------------------------------------------------------------

    def regenerate(
        self,
        component_factory: Callable[[], "DTComponent"],
        target_column: str,
        *,
        ue_id: str | None = None,
        cell_id: str | None = None,
        dependency_output_fields: dict[str, str] | None = None,
        window_hours: float | None = None,
        held_out_fraction: float = 0.2,
        drift_context: str | None = None,
    ) -> RegenerationResult:
        """Rebuild `component_factory()`'s component's pipeline from scratch via LLM-generated,
        sandboxed code, and register the accepted result as a new candidate version.

        `component_factory` constructs a fresh instance of the CURRENT (still-trusted) production
        class — used only to read its source code for LLM context, load the current production
        artifact into it (for error-pattern/fidelity comparison), and read its declared metadata.
        It is never the candidate; the candidate is entirely LLM-authored and sandboxed.
        """
        current_instance = component_factory()
        component_name = current_instance.COMPONENT_NAME
        expected_output_field = current_instance.OUTPUT_FIELD

        # 2. inspect current model metadata (+ source, for LLM context)
        current_version = self._model_registry.get_current_version(component_name)
        if current_version is None:
            raise RegenerationError(
                f"no production version registered for {component_name!r} — regeneration rebuilds "
                "an EXISTING production component's pipeline (prompt.md §24); bootstrap-train and "
                "register it first."
            )
        self._model_registry.load_artifact_into(current_instance, current_version)

        # Snapshot only — D1 is never touched again for the rest of this call (prompt.md §0.6/§0.8).
        history = self._d1_store.get_history(ue_id=ue_id, cell_id=cell_id)
        resolved_window_hours = (
            window_hours if window_hours is not None else self._settings.adaptation.regeneration.training_window_hours
        )
        window_df = select_recent_window(history, resolved_window_hours)
        min_rows = self._settings.adaptation.regeneration.min_training_rows
        if len(window_df) < min_rows:
            raise RegenerationError(
                f"only {len(window_df)} rows available in the last {resolved_window_hours:.2f}h for "
                f"{component_name!r}, need >= {min_rows} (config.adaptation.regeneration.min_training_rows)"
            )
        try:
            window_df = with_dependency_ground_truth(window_df, current_instance.DEPENDENCIES, dependency_output_fields)
        except DataSelectionError as exc:
            raise RegenerationError(str(exc)) from exc
        train_df, held_out_df = time_split(window_df, held_out_fraction)
        train_target = train_df[target_column]
        held_out_target = held_out_df[target_column]

        # The FULL window (raw D1 columns + populated dependency ground-truth columns) is handed
        # to both the current instance and the sandboxed candidate — neither pre-subsets it. Each
        # component (current or candidate) selects whatever it needs via its own declared
        # REQUIRED_FEATURES, exactly like every existing DTComponent already does internally. This
        # is what lets a candidate legitimately redeclare REQUIRED_FEATURES/DEPENDENCIES: the
        # caller never has to know in advance what the rebuilt pipeline will actually use.
        context = self._build_context(current_instance, current_version, history, held_out_df, held_out_target, drift_context)

        # 3-6: generate -> sandbox -> self-correct on rejection, up to max_llm_iterations.
        max_attempts = self._settings.adaptation.regeneration.max_llm_iterations
        sandbox_result: SandboxResult | None = None
        generated: _GeneratedComponentCode | None = None
        previous_attempt: dict[str, Any] | None = None
        attempt = 0
        for attempt in range(1, max_attempts + 1):
            generated = self._generate_candidate(component_name, expected_output_field, context, previous_attempt)
            sandbox_result = self._sandbox.run_candidate(
                source_code=generated.source_code,
                class_name=generated.class_name,
                expected_component_name=component_name,
                expected_output_field=expected_output_field,
                target_column=target_column,
                train_features=train_df,
                train_target=train_target,
                eval_features=held_out_df,
                eval_target=held_out_target,
            )
            if sandbox_result.accepted:
                break
            logger.warning(
                "regeneration candidate rejected by sandbox",
                extra={
                    "component": "regeneration_agent",
                    "component_name": component_name,
                    "attempt": attempt,
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
            raise RegenerationError(
                f"regeneration for {component_name!r} produced no accepted candidate after "
                f"{max_attempts} attempt(s): stage={sandbox_result.stage} error={sandbox_result.error}"
            )

        # 8. real Module 12 fidelity before/after, computed here from DATA the sandbox returned
        # (never by trusting the candidate's own self-reported metrics for this comparison).
        fidelity_before = fidelity_after = None
        if self._fidelity_evaluator is not None and len(held_out_target) > 0:
            try:
                old_preds = current_instance.predict(held_out_df)
                before = self._fidelity_evaluator.evaluate(component_name, held_out_target, old_preds, update_window=True)
                after = self._fidelity_evaluator.evaluate(
                    component_name, held_out_target, sandbox_result.eval_predictions, update_window=False
                )
                fidelity_before, fidelity_after = before.fidelity_score, after.fidelity_score
            except Exception as exc:  # a fidelity-comparison failure must not discard an otherwise
                # valid, sandboxed, accepted candidate — it just means before/after stay None,
                # never fabricated (mirrors Module 12's own "never manufacture a score" rule).
                logger.warning(
                    "fidelity before/after comparison failed, leaving both None",
                    extra={"component": "regeneration_agent", "component_name": component_name, "error": str(exc)},
                )

        # 7. register the accepted candidate — never a live instance this process trusts, only
        # the bytes/metadata the sandbox already validated and handed back as data.
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
                component=component_name,
                model_class=generated.class_name,
                artifact_source_path=artifact_tmp,
                source_code_path=source_tmp,
                dependencies=sandbox_result.dependencies or (),
                feature_schema=sandbox_result.required_features or (),
                output_field=expected_output_field,
                adaptation_type="regenerate",
                parent_version_id=current_version.version_id,
                training_window=training_window,
                evaluation_window=evaluation_window,
                evaluation_metrics=sandbox_result.metrics or {},
                fidelity_before=fidelity_before,
                fidelity_after=fidelity_after,
                status="candidate",
                llm_metadata={
                    "model": self._settings.llm.model,
                    "reasoning": generated.reasoning,
                    "attempts": attempt,
                },
            )

        logger.info(
            "regeneration candidate produced",
            extra={
                "component": "regeneration_agent",
                "component_name": component_name,
                "version_id": version.version_id,
                "parent_version_id": current_version.version_id,
                "attempts": attempt,
                "fidelity_before": fidelity_before,
                "fidelity_after": fidelity_after,
            },
        )
        return RegenerationResult(
            version=version,
            sandbox_result=sandbox_result,
            attempts=attempt,
            training_window=training_window,
            evaluation_window=evaluation_window,
            fidelity_before=fidelity_before,
            fidelity_after=fidelity_after,
        )

    # --- LLM context + generation ---------------------------------------------------------------

    def _build_context(
        self,
        current_instance: "DTComponent",
        current_version: ModelVersionMetadata,
        history: pd.DataFrame,
        held_out_df: pd.DataFrame,
        held_out_target: pd.Series,
        drift_context: str | None,
    ) -> dict[str, Any]:
        """Everything prompt.md §25 lists the LLM may receive, gathered from real sources — never
        fabricated. RAG context (Module 18) is explicitly reported as unavailable rather than
        silently omitted, since a future reader should know it wasn't just forgotten."""
        try:
            current_source = inspect.getsource(type(current_instance))
        except (OSError, TypeError):
            current_source = "<source unavailable>"

        available_columns = [c for c in history.columns if c not in _NON_FEATURE_COLUMNS]
        feature_statistics = history[available_columns].describe().to_dict()

        try:
            old_preds = current_instance.predict(held_out_df)
            residuals = held_out_target.to_numpy() - np.asarray(old_preds)
            error_patterns: dict[str, Any] = {
                "residual_mean": float(np.mean(residuals)),
                "residual_std": float(np.std(residuals)),
                "n_eval_rows": int(len(held_out_target)),
            }
        except Exception as exc:
            error_patterns = {"note": f"could not compute residuals from the current production model: {exc}"}

        return {
            "current_source_code": current_source,
            "current_metadata": {
                "component_name": current_instance.COMPONENT_NAME,
                "dependencies": list(current_instance.DEPENDENCIES),
                "required_features": list(current_instance.REQUIRED_FEATURES),
                "output_field": current_instance.OUTPUT_FIELD,
                "current_version_id": current_version.version_id,
            },
            "current_evaluation_metrics": current_version.evaluation_metrics,
            "current_fidelity_before": current_version.fidelity_before,
            "current_fidelity_after": current_version.fidelity_after,
            "feature_statistics": feature_statistics,
            "error_patterns": error_patterns,
            "drift_context": drift_context or "not provided",
            "rag_context": "not available — Module 18 (RAG Knowledge Base) is not built yet",
            "available_columns": available_columns,
            "constraints": {
                "min_training_rows": self._settings.adaptation.regeneration.min_training_rows,
                "sandbox_timeout_seconds": self._settings.sandbox.timeout_seconds,
            },
        }

    def _generate_candidate(
        self,
        component_name: str,
        expected_output_field: str,
        context: dict[str, Any],
        previous_attempt: dict[str, Any] | None,
    ) -> _GeneratedComponentCode:
        interface_contract = (
            "The class MUST subclass `src.dt_models.base.DTComponent` and implement: "
            f"`COMPONENT_NAME` (class attribute, MUST equal exactly {component_name!r}), "
            "`DEPENDENCIES` (tuple of other component names whose predictions this needs — may "
            "differ from the current implementation), `REQUIRED_FEATURES` (tuple of raw/"
            "dependency column names this component reads directly — may differ from the "
            f"current implementation), `OUTPUT_FIELD` (class attribute, MUST equal exactly "
            f"{expected_output_field!r} — other components already depend on this exact name), "
            "`is_trained` (property), `train(self, features: pd.DataFrame, targets: pd.Series) "
            "-> None`, `predict(self, features: pd.DataFrame) -> pd.Series`, "
            "`evaluate(self, features, targets) -> dict[str, float]`, `save(self, path)`, "
            "`load(self, path)`. The constructor must accept no required arguments."
        )
        prompt = (
            f"You are rebuilding ONE Digital Twin prediction component's internal pipeline for a "
            f"5G network digital twin — not redesigning the project (prompt.md §25). The "
            f"component {component_name!r}'s current architecture is no longer adequate for the "
            f"observed behavior (drift context: {context['drift_context']}) and needs a fresh "
            f"implementation.\n\n"
            f"Current component source code:\n```python\n{context['current_source_code']}\n```\n\n"
            f"Current metadata: {json.dumps(context['current_metadata'])}\n"
            f"Current held-out evaluation metrics: {json.dumps(context['current_evaluation_metrics'])}\n"
            f"Current fidelity (Module 12, before/after its last adaptation): "
            f"{context['current_fidelity_before']} / {context['current_fidelity_after']}\n"
            f"Recent error pattern on held-out data: {json.dumps(context['error_patterns'])}\n"
            f"Available raw/dependency columns in the training data you will receive: "
            f"{context['available_columns']}\n"
            f"Recent feature statistics (pandas describe()): "
            f"{json.dumps(context['feature_statistics'], default=str)[:4000]}\n"
            f"Relevant RAG context: {context['rag_context']}\n"
            f"Constraints: {json.dumps(context['constraints'])}\n\n"
            f"Interface contract: {interface_contract}\n\n"
            f"Produce a COMPLETE, self-contained Python module (all needed imports included) "
            f"implementing a NEW class satisfying this contract. You may change DEPENDENCIES/"
            f"REQUIRED_FEATURES/the internal algorithm entirely, but COMPONENT_NAME and "
            f"OUTPUT_FIELD MUST remain exactly as given above."
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
            raise RegenerationError(f"LLM code generation failed for {component_name!r}: {exc}") from exc
