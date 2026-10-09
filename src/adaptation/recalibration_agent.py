"""Recalibration Agent (Agent 2) — retrains an EXISTING, already-production component on a recent
window when its learned parameters (not architecture) have gone stale. Does not decide whether to
recalibrate — the Decision Agent already did — and never runs on a schedule.

Flow: inspect current model (`ModelRegistry.get_current_version()`) -> snapshot D1 history ->
pick a training window (`_resolve_training_window()`) -> select + time-split data
(`data_selection`) -> `DTComponent.train()` -> register as a new candidate
(`ModelRegistry.register_version()`) -> evaluate (component-local metrics + real fidelity) ->
return `RecalibrationResult` — never self-promotes.

Continuous operation: `D1Store.get_history()` returns a deep-copy snapshot under a lock held only
for the copy; this agent never holds D1's lock while training and never writes to D1.

LLM reasoning is optional, off by default: when enabled, it may only suggest a training-window
length (clamped to a bounded range around the config default); any failure degrades to that
default. Training and evaluation are always deterministic code regardless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
from src.registry.model_registry import ModelRegistry, ModelVersionMetadata

if TYPE_CHECKING:
    from src.common.config import Settings
    from src.dt_models.base import DTComponent
    from src.dt_models.d1_model_store import D1Store
    from src.llm.google_client import GoogleClient

logger = logging.getLogger(__name__)


class RecalibrationError(Exception):
    """Raised when recalibration cannot proceed (no production version to recalibrate, not
    enough recent training data, etc.) — never silently skipped or guessed around."""


class _WindowSuggestion(BaseModel):
    """Structured output schema for the OPTIONAL LLM training-window reasoning step. Genuinely
    tiny and generic — not a per-agent business schema, just this one call's output shape."""

    window_hours: float = Field(gt=0)
    reasoning: str


@dataclass(frozen=True)
class RecalibrationResult:
    version: ModelVersionMetadata
    candidate_component: "DTComponent"  # trained, in-memory — NOT wired into the live orchestrator
    training_window: dict[str, Any]
    evaluation_window: dict[str, Any]
    evaluation_metrics: dict[str, float]
    fidelity_before: float | None
    fidelity_after: float | None


class RecalibrationAgent:
    def __init__(
        self,
        settings: "Settings",
        d1_store: "D1Store",
        model_registry: ModelRegistry,
        fidelity_evaluator: FidelityEvaluator | None = None,
        llm_client: "GoogleClient | None" = None,
    ) -> None:
        self._settings = settings
        self._d1_store = d1_store
        self._model_registry = model_registry
        self._fidelity_evaluator = fidelity_evaluator
        self._llm_client = llm_client

    @classmethod
    def from_settings(
        cls,
        settings: "Settings",
        d1_store: "D1Store",
        model_registry: ModelRegistry,
        llm_client: "GoogleClient | None" = None,
    ) -> "RecalibrationAgent":
        return cls(
            settings,
            d1_store,
            model_registry,
            fidelity_evaluator=FidelityEvaluator(settings.fidelity),
            llm_client=llm_client,
        )

    # --- step 4: training window --------------------------------------------------------------

    def _resolve_training_window_hours(
        self, history: pd.DataFrame, component: str, use_llm_window_reasoning: bool
    ) -> float:
        default_hours = float(self._settings.adaptation.recalibration.training_window_hours)
        if not use_llm_window_reasoning or self._llm_client is None:
            return default_hours

        n_rows = len(history)
        span_hours = 0.0
        if n_rows > 0:
            span = history["ts_1s"].max() - history["ts_1s"].min()
            span_hours = span.total_seconds() / 3600.0
        prompt = (
            f"A digital twin component named '{component}' needs recalibration (its production "
            f"model's behavior has drifted). It has {n_rows} recent telemetry history rows "
            f"spanning approximately {span_hours:.2f} hours. Suggest an appropriate recent "
            f"training window length in hours for retraining this component — long enough to "
            f"contain enough data, short enough to reflect the CURRENT behavior rather than "
            f"stale history. Reply with your suggested window_hours and a one-sentence reasoning."
        )
        suggestion = self._llm_client.complete_structured_safe(prompt, _WindowSuggestion)
        if suggestion is None:
            logger.warning(
                "LLM training-window reasoning unavailable, using deterministic default",
                extra={"component": "recalibration_agent", "component_name": component, "default_hours": default_hours},
            )
            return default_hours

        # The LLM's suggestion only ever adjusts the window WITHIN a bounded range around the
        # deterministic default — it can never pick an unbounded value; the actual clamping logic
        # (not the LLM) is what ultimately determines the window used.
        clamped = min(max(suggestion.window_hours, default_hours * 0.25), default_hours * 4.0)
        logger.info(
            "LLM training-window suggestion applied",
            extra={
                "component": "recalibration_agent",
                "component_name": component,
                "suggested_hours": suggestion.window_hours,
                "clamped_hours": clamped,
                "reasoning": suggestion.reasoning,
            },
        )
        return clamped

    def _llm_metadata_for(self, use_llm_window_reasoning: bool) -> dict[str, Any] | None:
        if not use_llm_window_reasoning or self._llm_client is None:
            return None
        return {"purpose": "training_window_reasoning", "model": self._settings.llm.model}

    # --- main entrypoint ---------------------------------------------------------------------

    def recalibrate(
        self,
        component_factory: Callable[[], "DTComponent"],
        target_column: str,
        *,
        ue_id: str | None = None,
        cell_id: str | None = None,
        dependency_output_fields: dict[str, str] | None = None,
        window_hours: float | None = None,
        held_out_fraction: float = 0.2,
        use_llm_window_reasoning: bool = False,
    ) -> RecalibrationResult:
        """Retrain `component_factory()`'s component on a recent D1 telemetry window and
        register the result as a new candidate version.

        `component_factory` constructs a FRESH, untrained instance of the target component class
        (e.g. `lambda: ThroughputModel.from_settings(settings)`) — this agent trains that new
        instance, it never mutates any already-deployed production instance in place.
        `dependency_output_fields` maps each of the component's `DEPENDENCIES` entries to that
        dependency's `OUTPUT_FIELD` name (e.g. `{"throughput": "throughput_mbps_pred"}`), so a
        dependent component can be recalibrated too — trained on the SAME "ground truth for
        upstream dependencies" convention every DT component (Modules 6-10) already uses.
        """
        candidate = component_factory()
        component_name = candidate.COMPONENT_NAME

        # 2. inspect current model metadata — recalibration retrains an EXISTING production
        # component; it is not how a component gets its first (bootstrap) version.
        current_version = self._model_registry.get_current_version(component_name)
        if current_version is None:
            raise RecalibrationError(
                f"no production version registered for {component_name!r} — recalibration retrains an "
                "EXISTING production component (prompt.md §23). Bootstrap-train and register it first "
                "(ModelRegistry.register_version(..., adaptation_type='bootstrap', status='production'))."
            )

        # 3. inspect recent DT/telemetry history — a SNAPSHOT (deep copy under a brief lock);
        # D1 is never touched again for the rest of this call, so the live synchronizer is never
        # blocked by anything below this line (prompt.md §0.6/§0.8).
        history = self._d1_store.get_history(ue_id=ue_id, cell_id=cell_id)

        # 4. determine an appropriate recent training window
        resolved_window_hours = (
            window_hours
            if window_hours is not None
            else self._resolve_training_window_hours(history, component_name, use_llm_window_reasoning)
        )

        # 5. select the relevant training data
        window_df = select_recent_window(history, resolved_window_hours)
        min_rows = self._settings.adaptation.recalibration.min_training_rows
        if len(window_df) < min_rows:
            raise RecalibrationError(
                f"only {len(window_df)} rows available in the last {resolved_window_hours:.2f}h for "
                f"{component_name!r}, need >= {min_rows} (config.adaptation.recalibration.min_training_rows)"
            )
        try:
            window_df = with_dependency_ground_truth(window_df, candidate.DEPENDENCIES, dependency_output_fields)
        except DataSelectionError as exc:
            raise RecalibrationError(str(exc)) from exc
        input_columns = list(candidate.REQUIRED_FEATURES) + [
            dependency_output_fields[dep] for dep in candidate.DEPENDENCIES  # type: ignore[index]
        ]
        train_df, held_out_df = time_split(window_df, held_out_fraction)

        # 6. call the existing generic train() interface (Module 5's DTComponent contract)
        candidate.train(train_df[input_columns], train_df[target_column])

        # 8. evaluate it — component-local metrics, plus real Module 12 fidelity before/after
        # when a shared evaluator was supplied (comparing production vs. candidate over the SAME
        # held-out window, sharing ONE normalization reference — prompt.md §0.10).
        evaluation_metrics = candidate.evaluate(held_out_df[input_columns], held_out_df[target_column])
        fidelity_before = fidelity_after = None
        if self._fidelity_evaluator is not None and len(held_out_df) > 0:
            fidelity_before, fidelity_after = self._compare_fidelity(
                component_factory, component_name, current_version, candidate, held_out_df, input_columns, target_column
            )

        # 7. produce a new candidate model version
        training_window = {
            "start": iso(train_df["ts_1s"].min()) if len(train_df) else None,
            "end": iso(train_df["ts_1s"].max()) if len(train_df) else None,
            "n_rows": len(train_df),
            "window_hours": resolved_window_hours,
            "provenance": "d1_history",
            "ue_id": ue_id,
            "cell_id": cell_id,
        }
        evaluation_window = {
            "start": iso(held_out_df["ts_1s"].min()) if len(held_out_df) else None,
            "end": iso(held_out_df["ts_1s"].max()) if len(held_out_df) else None,
            "n_rows": len(held_out_df),
            "provenance": "d1_history",
        }
        version = self._model_registry.register_version(
            component_instance=candidate,
            adaptation_type="recalibrate",
            parent_version_id=current_version.version_id,
            training_window=training_window,
            evaluation_window=evaluation_window,
            evaluation_metrics=evaluation_metrics,
            fidelity_before=fidelity_before,
            fidelity_after=fidelity_after,
            status="candidate",
            llm_metadata=self._llm_metadata_for(use_llm_window_reasoning),
        )

        # 9. "pass it to verification" — Module 17 doesn't exist yet; this agent's contract ends
        # at handing back a versioned, evaluated candidate for a future caller to verify/promote.
        logger.info(
            "recalibration candidate produced",
            extra={
                "component": "recalibration_agent",
                "component_name": component_name,
                "version_id": version.version_id,
                "parent_version_id": current_version.version_id,
                "training_rows": len(train_df),
                "held_out_rows": len(held_out_df),
                "fidelity_before": fidelity_before,
                "fidelity_after": fidelity_after,
            },
        )
        return RecalibrationResult(
            version=version,
            candidate_component=candidate,
            training_window=training_window,
            evaluation_window=evaluation_window,
            evaluation_metrics=evaluation_metrics,
            fidelity_before=fidelity_before,
            fidelity_after=fidelity_after,
        )

    def _compare_fidelity(
        self,
        component_factory: Callable[[], "DTComponent"],
        component_name: str,
        current_version: ModelVersionMetadata,
        candidate: "DTComponent",
        held_out_df: pd.DataFrame,
        input_columns: list[str],
        target_column: str,
    ) -> tuple[float | None, float | None]:
        old_component = component_factory()
        self._model_registry.load_artifact_into(old_component, current_version)
        y_true = held_out_df[target_column]
        old_preds = old_component.predict(held_out_df[input_columns])
        new_preds = candidate.predict(held_out_df[input_columns])
        assert self._fidelity_evaluator is not None
        before = self._fidelity_evaluator.evaluate(component_name, y_true, old_preds, update_window=True)
        # Candidate compared against the SAME normalization reference production just grew
        # (update_window=False) — prompt.md §0.10: "BOTH MUST USE THE SAME NORMALIZATION REFERENCE."
        after = self._fidelity_evaluator.evaluate(component_name, y_true, new_preds, update_window=False)
        return before.fidelity_score, after.fidelity_score
