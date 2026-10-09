"""Top-level continuous orchestration loop — wires every module into one running system.

Canonical loop: OAI replay/mock telemetry -> preprocess -> continuous sync -> D1 -> 7-component
DT prediction -> fidelity (+ unified score/trigger) -> drift -> Decision & Root-Cause Analysis
Agent -> recalibrate/regenerate/expand_scope -> verification -> lifecycle record -> continue.

Integration only: constructs already-tested components and calls their public methods in order;
no model/fidelity/verification/agent logic lives here.

Non-negotiable: telemetry ingestion must never stop or block for an adaptation cycle (LLM call +
possible sandboxed subprocess train). `ContinuousSynchronizer` runs on its own background thread
started once at `start()`; the trigger->decision->agent->verification->lifecycle cycle runs on the
orchestrator's own foreground loop — a slow adaptation cycle only ever delays the next trigger,
never a telemetry sync. See `scripts/run_orchestrator_demo.py` for a live proof.

Two trigger sources (external `DriftEvent` and the internal fidelity-based trigger) are
normalized into one `AdaptationTrigger` shape and share one queue/dispatch path. The Decision
Agent makes a genuine LLM call per trigger; `decide_safe()` falls back to the configured
deterministic strategy (logged WARNING) on any LLM failure. If `regenerate`/`expand_scope` is
selected with no LLM client available to execute it, that cycle is skipped (WARNING) without
affecting telemetry or the next trigger. `recalibrate` never needs an LLM to execute.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from src.adaptation.data_selection import select_recent_window, time_split, with_dependency_ground_truth
from src.adaptation.decision_agent import DecisionAgent, DecisionOutput
from src.adaptation.decision_context import PreviousOutcome, build_decision_context
from src.adaptation.expand_scope_agent import ExpandScopeAgent, ExpandScopeError
from src.adaptation.lifecycle_agent import LifecycleAgent
from src.adaptation.recalibration_agent import RecalibrationAgent, RecalibrationError
from src.adaptation.regeneration_agent import RegenerationAgent, RegenerationError
from src.adaptation.verification_agent import VerificationAgent
from src.common.config import Settings, load_secrets, load_settings
from src.common.logging import setup_logging
from src.data.oai_dataset import TARGET_COLUMNS as _OAI_TARGETS
from src.dt_models.base import DTComponent
from src.dt_models.d1_model_store import D1Store
from src.dt_models.dl_bler import DlBlerModel
from src.dt_models.dl_prb import DlPrbModel
from src.dt_models.dl_rlc_delay import DlRlcDelayModel
from src.dt_models.dl_throughput import DlThroughputModel
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator, OrchestratorError
from src.dt_models.ul_bler import UlBlerModel
from src.dt_models.ul_prb import UlPrbModel
from src.dt_models.ul_throughput import UlThroughputModel
from src.drift.drift_detector import DriftDetectorInterface
from src.drift.mock_drift_source import MockDriftSource
from src.fidelity.evaluator import FidelityEvaluator
from src.fidelity.trigger import AdaptationTrigger, trigger_from_drift_event
from src.llm.google_client import GoogleClient
from src.rag.rag_kb import RagKnowledgeBase
from src.registry.model_registry import ModelRegistry, ModelVersionMetadata
from src.sandbox.executor import SandboxExecutor
from src.synchronization.sync import ContinuousSynchronizer
from src.telemetry.mock_source import MockTelemetrySource
from src.telemetry.oai_replay_source import OaiReplaySource
from src.telemetry.preprocessing import TelemetryPreprocessor

if TYPE_CHECKING:
    from src.common.config import Secrets

logger = logging.getLogger(__name__)

# The 7 OAI KPI components — the only names a drift event may legitimately carry.
_DT_COMPONENT_CLASSES: dict[str, type[DTComponent]] = {
    "dl_bler": DlBlerModel,
    "ul_bler": UlBlerModel,
    "dl_prb": DlPrbModel,
    "ul_prb": UlPrbModel,
    "dl_throughput": DlThroughputModel,
    "ul_throughput": UlThroughputModel,
    "dl_rlc_delay": DlRlcDelayModel,
}
# Target column == component name here; kept as an explicit dict (never assumed) so a future
# component whose name and target column differ needs no structural change.
_TARGET_COLUMNS: dict[str, str] = {name: name for name in _OAI_TARGETS}
# Topological order so dependency-having components can be bootstrap-trained with their
# dependency's ground-truth column already available.
_BOOTSTRAP_ORDER: tuple[str, ...] = ("dl_bler", "ul_bler", "dl_prb", "ul_prb", "dl_throughput", "ul_throughput", "dl_rlc_delay")
_OUTPUT_FIELDS: dict[str, str] = {name: cls.OUTPUT_FIELD for name, cls in _DT_COMPONENT_CLASSES.items()}


class ContinuousOrchestrator:
    """Wires and runs the full canonical loop. Construct via `initialize()`, then `start()`
    (background telemetry/drift threads) and `run()` (the foreground orchestration loop)."""

    def __init__(
        self,
        settings: Settings,
        secrets: "Secrets",
        drift_severity_range_override: tuple[float, float] | None = None,
    ) -> None:
        self._settings = settings
        self._secrets = secrets
        # Only biases what severities the mock drift source GENERATES (demo/validation runs) —
        # never what DriftDetectorInterface accepts.
        self._drift_severity_range_override = drift_severity_range_override

        self._stop_event = threading.Event()
        # None until start(); lets stop() without a prior start() be a harmless no-op.
        self.sync_thread: threading.Thread | None = None
        self.drift_thread: threading.Thread | None = None
        # Both trigger sources funnel into one pending-trigger structure, coalesce policy: at
        # most one pending trigger per component, merged by max severity, picked up once that
        # component's lock releases. See _enqueue_trigger/_pop_next_trigger/_handle_trigger.
        self._pending_lock = threading.Lock()
        self._pending_by_component: dict[str, AdaptationTrigger] = {}
        self._pending_order: list[str] = []  # component names, FIFO by first-pending arrival
        self._locked_component: str | None = None  # component currently mid-cycle, else None
        self._latest_fidelity: dict[str, float | None] = {}
        # Per-component previous-outcome tracking for the Decision Agent's context.
        self._previous_outcome_by_component: dict[str, PreviousOutcome] = {}

        # Brackets the current adaptation cycle's wall-clock window; read by validation scripts.
        self.last_adaptation_started_at: float | None = None
        self.last_adaptation_finished_at: float | None = None

    # --- initialization: config -> storage -> RAG -> registry -> bootstrap models -> telemetry --

    def initialize(self) -> None:
        settings = self._settings
        self.d1_store = D1Store.from_settings(settings)
        self.rag_kb = RagKnowledgeBase.from_settings(settings)
        self.model_registry = ModelRegistry.from_settings(settings)
        self.sandbox_executor = SandboxExecutor.from_settings(settings)
        self.fidelity_evaluator = FidelityEvaluator(settings.fidelity)

        try:
            self.llm_client: GoogleClient | None = GoogleClient.from_settings(settings, self._secrets)
        except RuntimeError as exc:
            self.llm_client = None
            logger.warning(
                "no GOOGLE_API_KEY/GEMINI_API_KEY configured — the Decision & Root-Cause Analysis "
                "Agent will use its configured deterministic fallback strategy for every trigger "
                "until a real key is set, and regenerate/expand_scope adaptation cycles will be "
                "skipped (with a clear warning) if that strategy is ever selected; recalibrate "
                "execution is unaffected",
                extra={"component": "main", "error": str(exc)},
            )
        self.decision_agent = DecisionAgent.from_settings(settings, self.llm_client)
        # Must be constructed AFTER self.llm_client and passed it explicitly, or its optional
        # LLM-enhanced report prose is unreachable with no warning ever logged.
        self.lifecycle_agent = LifecycleAgent.from_settings(settings, llm_client=self.llm_client)

        # Per-component dispatch bookkeeping, seeded from the 7 known components but growable at
        # runtime — an ACCEPTed expand_scope candidate adds a new name (see _hot_swap_candidate);
        # _bootstrap_dt_models restores these for any such component from a prior process.
        self._target_columns: dict[str, str] = dict(_TARGET_COLUMNS)
        self._output_fields: dict[str, str] = dict(_OUTPUT_FIELDS)
        # Classes dynamically loaded from a regenerate/expand_scope candidate's saved source
        # (never the 7 originals, which stay statically imported).
        self._dynamic_component_classes: dict[str, type[DTComponent]] = {}

        self._bootstrap_dt_models()

        self.dt_model_registry = DTModelRegistry()
        for name, instance in self._dt_components.items():
            self.dt_model_registry.register(instance, enabled=True)
        self.dt_orchestrator = DTOrchestrator(self.dt_model_registry)

        self.preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)
        self.telemetry_source = self._build_telemetry_source()
        self.synchronizer = ContinuousSynchronizer(
            self.telemetry_source, self.preprocessor, self.d1_store, settings.synchronization
        )

        self.drift_detector = DriftDetectorInterface.from_settings(settings)
        severity_range = self._drift_severity_range_override or settings.drift.severity_range
        self.drift_source = MockDriftSource(
            settings.drift.mock,
            valid_components=settings.drift.valid_components,
            severity_range=severity_range,
            max_events=None,
            realtime=True,
        )

        logger.info(
            "orchestrator initialized",
            extra={
                "component": "main",
                "environment": settings.environment,
                "telemetry_source": settings.telemetry.source,
                "llm_available": self.llm_client is not None,
                "rag_available": self.rag_kb.is_available,
            },
        )

    def _build_telemetry_source(self):
        settings = self._settings
        if settings.telemetry.source == "oai_replay":
            source = OaiReplaySource(
                settings.resolve_path("data/processed/oai"), speed=settings.telemetry.replay.speed,
                default_split=settings.telemetry.replay.default_split,
            )
            default_run_ids = getattr(settings.data.splits, settings.telemetry.replay.default_split)
            source.start(run_ids=default_run_ids)
            return source
        return MockTelemetrySource(settings.telemetry.mock, max_records=None, realtime=True)

    def _bootstrap_dt_models(self) -> None:
        """Load each of the 7 components' current production artifact if one exists (resumed
        process), else train fresh on bootstrap telemetry and register as the first production
        version. Also loads any extra component an expand_scope ACCEPT created in a prior
        process's lifetime (`list_component_names()`) — always a load, never a fresh train."""
        settings = self._settings
        self._dt_components: dict[str, DTComponent] = {}
        extra_names = [n for n in self.model_registry.list_component_names() if n not in _BOOTSTRAP_ORDER]

        need_bootstrap_data = any(self.model_registry.get_current_version(name) is None for name in _BOOTSTRAP_ORDER)
        bootstrap_history: pd.DataFrame | None = None
        if need_bootstrap_data:
            bootstrap_history = self._generate_bootstrap_history()

        for name in _BOOTSTRAP_ORDER:
            current_version = self.model_registry.get_current_version(name)
            if current_version is not None and current_version.source_path is not None:
                # Regenerated: production artifact belongs to an LLM-generated class, not the
                # static import — load the actual class, same as a fresh hot swap.
                instance = self._load_and_restore(name, current_version)
            else:
                instance = _DT_COMPONENT_CLASSES[name].from_settings(settings)
                if current_version is not None:
                    self.model_registry.load_artifact_into(instance, current_version)
                    logger.info(
                        "loaded existing production DT model",
                        extra={"component": "main", "model": name, "version_id": current_version.version_id},
                    )
                else:
                    assert bootstrap_history is not None
                    self._bootstrap_train_and_register(name, instance, bootstrap_history)
            self._dt_components[name] = instance

        for name in extra_names:
            version = self.model_registry.get_current_version(name)
            if version is None:  # only candidate/rejected versions ever existed for it — nothing to resume
                continue
            self._dt_components[name] = self._load_and_restore(name, version)

    def _load_and_restore(self, name: str, version: ModelVersionMetadata) -> DTComponent:
        """Dynamically loads `version`'s LLM-generated class + trained artifact, and restores its
        target-column/output-field bookkeeping (needed the first time an expand_scope component
        is resumed; a no-op for a regenerated existing one)."""
        cls = self._load_dynamic_class(version)
        instance = cls()
        self.model_registry.load_artifact_into(instance, version)
        self._dynamic_component_classes[name] = cls
        self._output_fields[name] = instance.OUTPUT_FIELD
        if name not in self._target_columns:
            target_column = (version.llm_metadata or {}).get("target_column")
            if target_column is None:
                logger.warning(
                    "resumed component has no recorded target_column in llm_metadata — its "
                    "fidelity cannot be computed until this is fixed; predictions still work",
                    extra={"component": "main", "model": name, "version_id": version.version_id},
                )
            else:
                self._target_columns[name] = target_column
        logger.info(
            "dynamically loaded a regenerate/expand_scope-produced production DT model",
            extra={"component": "main", "model": name, "model_class": version.model_class, "version_id": version.version_id},
        )
        return instance

    def _load_dynamic_class(self, version: ModelVersionMetadata) -> type[DTComponent]:
        """Imports a regenerate/expand_scope candidate's LLM-generated class from its saved source
        — the one point this trusted process imports LLM code, and only ever for a version that
        already passed sandbox + verification (never a rejected/pending candidate)."""
        return self.model_registry.load_dynamic_class(version)

    def _component_class_for(self, name: str) -> "type[DTComponent] | None":
        """The class currently backing live serving for `name`, or `None` if unknown."""
        # Dynamic checked first — a class is always truthy, so `A.get(name) or B.get(name)` would
        # silently keep returning the static class even after a hot swap replaced it.
        return self._dynamic_component_classes.get(name) or _DT_COMPONENT_CLASSES.get(name)

    def _generate_bootstrap_history(self) -> pd.DataFrame:
        """The real OAI TRAIN split (`data/processed/oai/train.parquet`) — never the live
        D1Store (bootstrap data is a separate provenance from live telemetry) and never VAL/TEST."""
        settings = self._settings
        train_path = settings.resolve_path("data/processed/oai/train.parquet")
        if not train_path.exists():
            raise FileNotFoundError(
                f"{train_path} does not exist — run scripts/build_oai_dataset.py before starting "
                "the orchestrator for the first time (bootstrap training needs the real OAI TRAIN split)"
            )
        history = pd.read_parquet(train_path, engine="pyarrow")
        logger.info("bootstrap telemetry (OAI TRAIN split) loaded", extra={"component": "main", "rows": len(history)})
        return history

    def _bootstrap_train_and_register(self, name: str, instance: DTComponent, history: pd.DataFrame) -> None:
        target_column = _TARGET_COLUMNS[name]
        train_df = history
        if instance.DEPENDENCIES:
            train_df = train_df.copy()
            for dep in instance.DEPENDENCIES:
                train_df[_OUTPUT_FIELDS[dep]] = train_df[_TARGET_COLUMNS[dep]]
        instance.train(train_df, train_df[target_column])
        version = self.model_registry.register_version(
            component_instance=instance,
            adaptation_type="bootstrap",
            parent_version_id=None,
            training_window={"n_rows": len(train_df), "provenance": "bootstrap"},
            evaluation_window={"n_rows": 0},
            evaluation_metrics=instance.evaluate(train_df, train_df[target_column]),
            status="production",
        )
        logger.info(
            "bootstrap-trained and registered new production DT model",
            extra={"component": "main", "model": name, "version_id": version.version_id},
        )

    # --- lifecycle (start/stop) -----------------------------------------------------------------

    def start(self) -> None:
        self.sync_thread = self.synchronizer.start()
        self.drift_thread = threading.Thread(target=self._drift_consumer_loop, name="dt-drift-consumer", daemon=True)
        self.drift_thread.start()
        logger.info("orchestrator started — telemetry sync and drift consumption running in the background", extra={"component": "main"})

    def stop(self) -> None:
        """Deterministic shutdown: signal both background threads, then block (bounded by
        `shutdown_join_timeout_seconds`) until each actually exits — avoids a caller treating the
        orchestrator as stopped while a thread is still running/logging."""
        self._stop_event.set()
        self.synchronizer.stop()
        self.drift_source.close()

        timeout = self._settings.synchronization.shutdown_join_timeout_seconds
        still_alive: list[str] = []
        if self.sync_thread is not None:
            self.sync_thread.join(timeout=timeout)
            if self.sync_thread.is_alive():
                still_alive.append(self.sync_thread.name)
        if self.drift_thread is not None:
            self.drift_thread.join(timeout=timeout)
            if self.drift_thread.is_alive():
                still_alive.append(self.drift_thread.name)

        if still_alive:
            logger.warning(
                "orchestrator stop() timed out waiting for background thread(s) to exit",
                extra={"component": "main", "still_alive": still_alive, "timeout_seconds": timeout},
            )
        else:
            logger.info("orchestrator stopped — both background threads confirmed exited", extra={"component": "main"})

    def _drift_consumer_loop(self) -> None:
        for event in self.drift_detector.process_stream(self.drift_source):
            if self._stop_event.is_set():
                break
            self._enqueue_trigger(trigger_from_drift_event(event))

    def _enqueue_trigger(self, trigger: AdaptationTrigger) -> None:
        """Coalesce policy: at most one pending trigger per component. A second trigger for a
        component with one already pending merges in, keeping the higher severity."""
        with self._pending_lock:
            existing = self._pending_by_component.get(trigger.component)
            if existing is None or trigger.severity > existing.severity:
                self._pending_by_component[trigger.component] = trigger
            if trigger.component not in self._pending_order:
                self._pending_order.append(trigger.component)

    def _pop_next_trigger(self) -> "AdaptationTrigger | None":
        """FIFO by first-pending arrival. Only removed here (just before dispatch), so a trigger
        that coalesces in while this component is locked starts a fresh entry for the next cycle."""
        with self._pending_lock:
            if not self._pending_order:
                return None
            component = self._pending_order.pop(0)
            return self._pending_by_component.pop(component)

    # --- the continuous loop itself (prompt.md §39) ---------------------------------------------

    def run(self, max_drift_events: int | None = None, prediction_interval_seconds: float = 5.0) -> None:
        """Periodically runs prediction+fidelity (may enqueue a fidelity trigger), then dispatches
        every pending trigger — external drift or fidelity-based, one shape — through decision ->
        agent -> verification -> lifecycle, one component-scoped adaptation at a time. Runs until
        `stop()`, or (demo/validation only) until `max_drift_events` cycles complete; production
        passes `max_drift_events=None` and never stops on its own. Telemetry sync is unaffected."""
        handled = 0
        last_prediction = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now - last_prediction >= prediction_interval_seconds:
                self.run_prediction_and_fidelity_cycle()
                last_prediction = now
            trigger = self._pop_next_trigger()
            if trigger is None:
                self._stop_event.wait(timeout=0.5)
                continue
            self._handle_trigger(trigger)
            handled += 1
            if max_drift_events is not None and handled >= max_drift_events:
                logger.info(
                    "max_drift_events reached — orchestration loop stopping (telemetry sync is "
                    "unaffected and keeps running until stop() is called separately)",
                    extra={"component": "main", "handled": handled},
                )
                return

    def run_prediction_and_fidelity_cycle(self) -> dict[str, float | None]:
        """Dependency-aware DT prediction -> fidelity, against live-growing D1 state, using the
        same `DTOrchestrator`/`FidelityEvaluator` for this orchestrator's whole lifetime so
        fidelity windows accumulate real history. Also checks the internal fidelity-based trigger
        and enqueues it like an external drift event if it fires. Public so a demo/validation
        script can call it directly to warm fidelity history before triggering an adaptation."""
        history = self.d1_store.get_history()
        if len(history) < self._settings.dt_models.bootstrap_min_rows:
            return {}
        recent = history.sort_values("ts_1s", kind="stable").tail(self._settings.evaluation.window_size).reset_index(drop=True)
        try:
            predictions = self.dt_orchestrator.run_predictions(recent)
        except OrchestratorError as exc:
            logger.warning("DT prediction cycle failed", extra={"component": "main", "error": str(exc)})
            return {}

        for name, preds in predictions.items():
            target_column = self._target_columns.get(name)
            if target_column is None:
                # An expand_scope component whose target_column wasn't recovered — predictions
                # still serve; only fidelity tracking for it is skipped.
                continue
            # Residual components legitimately return NaN for each session's first row (no
            # y_t-1 anchor). The fidelity formula must never silently score NaN rows, so mask here.
            target_values = np.asarray(recent[target_column], dtype=float)
            pred_values = np.asarray(preds, dtype=float)
            valid_mask = np.isfinite(target_values) & np.isfinite(pred_values)
            if not valid_mask.any():
                logger.debug(
                    "fidelity cycle: no rows with a valid prediction+target this cycle, skipping",
                    extra={"component": "main", "dt_component": name},
                )
                continue
            result = self.fidelity_evaluator.evaluate(
                name, target_values[valid_mask], pred_values[valid_mask], update_window=True
            )
            self._latest_fidelity[name] = result.fidelity_score
        logger.info(
            "prediction + fidelity cycle complete",
            extra={"component": "main", "rows": len(recent), "fidelity": dict(self._latest_fidelity)},
        )

        fidelity_trigger = self.fidelity_evaluator.check_fidelity_trigger(self._latest_fidelity)
        if fidelity_trigger is not None:
            self._enqueue_trigger(fidelity_trigger)

        return dict(self._latest_fidelity)

    # --- the full adaptation cycle: Decision & Root-Cause Analysis Agent -> agent -> verification
    # -> lifecycle -------------------------------------------------------------------------------

    def _handle_trigger(self, trigger: AdaptationTrigger) -> None:
        """Holds this component's adaptation lock for the cycle; always released in `finally`
        (ACCEPT, REJECT, or an unexpected exception) so a failure can never deadlock it."""
        self._locked_component = trigger.component
        self.last_adaptation_started_at = time.monotonic()
        try:
            self._run_adaptation_cycle(trigger)
        finally:
            self.last_adaptation_finished_at = time.monotonic()
            self._locked_component = None

    def _previous_outcome_for(self, component: str) -> PreviousOutcome:
        return self._previous_outcome_by_component.get(component, PreviousOutcome())

    def _hot_swap_candidate(self, action_name: str, component: str, agent_result: Any) -> None:
        """Called only on a real ACCEPT — makes the candidate already promoted in `ModelRegistry`
        take over live prediction serving right now, via `dt_model_registry` (the in-memory,
        orchestrator-facing registry, distinct from `ModelRegistry`'s versioned artifact store)."""
        version = agent_result.version
        if action_name == "recalibrate":
            # Already a live instance of an already-trusted class — no dynamic loading.
            instance: DTComponent = agent_result.candidate_component
        else:
            # regenerate/expand_scope: import the class for the first time now that verification
            # has granted trust — never before.
            cls = self._load_dynamic_class(version)
            instance = cls()
            self.model_registry.load_artifact_into(instance, version)
            self._dynamic_component_classes[component] = cls

        if component in self.dt_model_registry:
            self.dt_model_registry.replace(instance)
        else:
            # A genuinely new expand_scope component, served live for the first time.
            self.dt_model_registry.register(instance)
            self._target_columns[component] = agent_result.design.target_column
            logger.info(
                "new DT component permanently added to live prediction serving",
                extra={"component": "main", "new_component": component, "version_id": version.version_id},
            )
        self._output_fields[component] = instance.OUTPUT_FIELD

        logger.info(
            "hot-swapped ACCEPTed candidate into live serving — no restart required",
            extra={
                "component": "main", "affected_component": component, "action": action_name,
                "version_id": version.version_id, "model_class": version.model_class,
            },
        )

    def _run_adaptation_cycle(self, trigger: AdaptationTrigger) -> None:
        settings = self._settings
        component = trigger.component
        component_cls = self._component_class_for(component)
        if component_cls is None:
            logger.warning("trigger names an unknown component, skipping", extra={"component": "main", "affected_component": component})
            return

        context = build_decision_context(
            trigger,
            self._latest_fidelity,
            self.fidelity_evaluator.compute_unified_score(self._latest_fidelity).unified_score,
            self._previous_outcome_for(component),
            self.d1_store.get_history(),
            self.rag_kb,
        )
        decision: DecisionOutput = self.decision_agent.decide_safe(context)
        action_name = decision.strategy
        logger.info(
            "decision agent selected a strategy",
            extra={
                "component": "main", "affected_component": component, "trigger_type": trigger.trigger_type,
                "severity": trigger.severity, "strategy": action_name, "confidence": decision.confidence,
            },
        )

        target_column = self._target_columns[component]
        # A dynamically-loaded (hot-swapped) class only guarantees a no-arg constructor — never
        # a from_settings() classmethod.
        if component in self._dynamic_component_classes:
            component_factory: Callable[[], DTComponent] = lambda cls=component_cls: cls()  # noqa: E731
        else:
            component_factory = lambda cls=component_cls: cls.from_settings(settings)  # noqa: E731
        drift_context = f"{trigger.trigger_type} on {component!r} at severity {trigger.severity:.3f}"

        if action_name in ("regenerate", "expand_scope") and self.llm_client is None:
            logger.warning(
                "the decision agent selected an LLM-driven strategy but no GOOGLE_API_KEY is "
                "configured — skipping this adaptation cycle (telemetry ingestion and the next "
                "trigger are entirely unaffected)",
                extra={"component": "main", "action": action_name, "affected_component": component},
            )
            return

        try:
            if action_name == "recalibrate":
                agent = RecalibrationAgent(settings, self.d1_store, self.model_registry, fidelity_evaluator=None, llm_client=self.llm_client)
                agent_result = agent.recalibrate(component_factory, target_column, dependency_output_fields=self._output_fields)
                window_hours = settings.adaptation.recalibration.training_window_hours
            elif action_name == "regenerate":
                agent = RegenerationAgent(settings, self.d1_store, self.model_registry, self.llm_client, self.sandbox_executor, fidelity_evaluator=None)
                agent_result = agent.regenerate(
                    component_factory, target_column, dependency_output_fields=self._output_fields, drift_context=drift_context,
                    rag_knowledge_base=self.rag_kb,
                )
                window_hours = settings.adaptation.regeneration.training_window_hours
            else:  # expand_scope
                agent = ExpandScopeAgent(
                    settings, self.d1_store, self.model_registry, self.dt_model_registry, self.llm_client, self.sandbox_executor,
                    fidelity_evaluator=None,
                )
                agent_result = agent.expand_scope(
                    component_factory, dependency_output_fields=self._output_fields, expand_scope_context=drift_context,
                    rag_knowledge_base=self.rag_kb,
                )
                window_hours = settings.adaptation.expand_scope.training_window_hours
        except (RecalibrationError, RegenerationError, ExpandScopeError) as exc:
            logger.warning(
                "adaptation agent failed to produce a candidate — production is untouched, "
                "telemetry ingestion is unaffected",
                extra={"component": "main", "action": action_name, "affected_component": component, "error": str(exc)},
            )
            return

        candidate_version = agent_result.version

        # Verify on the same held-out protocol the agent used; populate dependency ground-truth
        # columns if the target component has dependencies.
        if action_name == "recalibrate":
            window_df = select_recent_window(self.d1_store.get_history(), window_hours)
            _, held_out_df = time_split(window_df, 0.2)
            if component_cls.DEPENDENCIES:
                held_out_df = with_dependency_ground_truth(held_out_df, component_cls.DEPENDENCIES, self._output_fields)
        else:
            # regenerate/expand_scope: reuse the agent's own held_out_df — re-deriving "last 20%"
            # here, after a possibly long LLM+sandbox call, could select a different-sized slice
            # than the sandbox's fixed eval_predictions array (live telemetry keeps flowing).
            held_out_df = agent_result.held_out_df
        eval_target_column = target_column if action_name != "expand_scope" else agent_result.design.target_column
        eval_target = held_out_df[eval_target_column]

        if action_name == "recalibrate":
            candidate_predictions = agent_result.candidate_component.predict(held_out_df)
            production_factory: Callable[[], DTComponent] | None = component_factory
        elif action_name == "regenerate":
            candidate_predictions = agent_result.sandbox_result.eval_predictions
            production_factory = component_factory
        else:
            candidate_predictions = agent_result.sandbox_result.eval_predictions
            production_factory = None  # genuinely new component, no production baseline

        verification_agent = VerificationAgent(settings, self.model_registry, self.fidelity_evaluator, llm_client=self.llm_client)
        try:
            verification_result = verification_agent.verify(
                candidate_version, held_out_df, eval_target, candidate_predictions,
                production_component_factory=production_factory, drift_context=drift_context, rag_knowledge_base=self.rag_kb,
            )
        except Exception as exc:  # verification must never crash the orchestration loop
            logger.warning("verification failed unexpectedly — treating as REJECT-equivalent, no promotion occurred", extra={"component": "main", "error": str(exc)})
            return

        # Hot swap: an ACCEPT takes over live serving immediately, no restart required. Never on
        # REJECT — production is left exactly as verification decided.
        if verification_result.decision == "ACCEPT":
            try:
                self._hot_swap_candidate(action_name, component, agent_result)
            except Exception as exc:  # must never hide an already-recorded ACCEPT or crash the
                # loop — picked up correctly on the next restart via _bootstrap_dt_models.
                logger.warning(
                    "candidate was ACCEPTed and promoted, but hot-swapping into live serving "
                    "failed — it will still be picked up on the next process restart",
                    extra={"component": "main", "affected_component": component, "action": action_name, "error": str(exc)},
                )

        # Record which real telemetry source/run ids/replay speed drove this event.
        source_label = getattr(self.telemetry_source, "SOURCE_LABEL", "MOCK")
        replayed_run_ids = getattr(self.telemetry_source, "_run_ids", None) if source_label == "OAI_RAN_REPLAY" else None
        replay_speed = settings.telemetry.replay.speed if source_label == "OAI_RAN_REPLAY" else None
        record = self.lifecycle_agent.record_adaptation_event(
            trigger=trigger, decision=decision, agent_result=agent_result, verification_result=verification_result,
            telemetry_source=source_label, replayed_run_ids=replayed_run_ids, replay_speed=replay_speed,
        )
        self.lifecycle_agent.generate_maintenance_report(record, rag_knowledge_base=self.rag_kb)

        # Feed this cycle's outcome into the next decision context for this component.
        self._previous_outcome_by_component[component] = PreviousOutcome(
            action=action_name,
            verification_result=verification_result.decision,
            fidelity_before=verification_result.fidelity_before,
            fidelity_after=verification_result.fidelity_after,
        )

        logger.info(
            "adaptation cycle complete",
            extra={
                "component": "main", "event_id": record.event_id, "affected_component": component, "action": action_name,
                "decision": verification_result.decision, "fidelity_before": verification_result.fidelity_before,
                "fidelity_after": verification_result.fidelity_after,
            },
        )


def build_orchestrator(
    mode: str | None = None, drift_severity_range_override: tuple[float, float] | None = None
) -> ContinuousOrchestrator:
    settings = load_settings()
    if mode is not None:
        telemetry_source = "oai_replay" if mode == "live" else "mock"
        settings = settings.model_copy(
            update={"telemetry": settings.telemetry.model_copy(update={"source": telemetry_source}), "environment": mode}
        )
    secrets = load_secrets()
    orchestrator = ContinuousOrchestrator(settings, secrets, drift_severity_range_override=drift_severity_range_override)
    orchestrator.initialize()
    return orchestrator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI-Driven Self-Adaptive Network Digital Twin — continuous orchestration loop")
    parser.add_argument("--mode", choices=["demo", "live"], default=None, help="demo=mock telemetry, live=real OAI RAN KPI replay; omit to use config/settings.yaml as-is")
    parser.add_argument("--max-drift-events", type=int, default=None, help="stop the orchestration loop after this many adaptation cycles (telemetry sync keeps running until stop()); omit for real continuous operation")
    parser.add_argument("--prediction-interval-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)

    orchestrator = build_orchestrator(mode=args.mode)
    setup_logging(
        level=orchestrator._settings.logging.level, fmt=orchestrator._settings.logging.format, log_dir=orchestrator._settings.logging.log_dir
    )
    orchestrator.start()
    try:
        orchestrator.run(max_drift_events=args.max_drift_events, prediction_interval_seconds=args.prediction_interval_seconds)
    finally:
        orchestrator.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
