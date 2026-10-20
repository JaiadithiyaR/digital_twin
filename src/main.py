"""Top-level continuous orchestration loop — wires every module into one running system.

Canonical loop: OAI replay/mock telemetry -> preprocess -> continuous sync -> D1 -> 7-component
DT prediction -> per-component fidelity -> per-component adaptation policy -> Decision &
Root-Cause Analysis Agent -> recalibrate/regenerate/expand_scope -> verification -> lifecycle
record -> continue.

Integration only: constructs already-tested components and calls their public methods in order;
no model/fidelity/verification/agent logic lives here.

Non-negotiable: telemetry ingestion must never stop or block for an adaptation cycle (LLM call +
possible sandboxed subprocess train). `ContinuousSynchronizer` runs on its own background thread
started once at `start()`; the trigger->decision->agent->verification->lifecycle cycle runs on the
orchestrator's own foreground loop — a slow adaptation cycle only ever delays the next trigger,
never a telemetry sync. See `scripts/run_orchestrator_demo.py` for a live proof.

The ONLY trigger source is the internal per-component fidelity policy
(`src/fidelity/component_policy.py`). External drift events are permanently disabled in the live
system — no drift source or consumer is constructed here, and there is no flag to enable one
(Module 11's drift interface is still used by the separate `baselines/` experiment harness). The
Decision
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
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from src.adaptation.data_selection import (
    select_recent_window,
    time_split,
    with_dependency_ground_truth,
)
from src.adaptation.decision_agent import DecisionAgent, DecisionOutput
from src.adaptation.decision_context import PreviousOutcome, build_decision_context
from src.adaptation.expand_scope_agent import ExpandScopeAgent, ExpandScopeError
from src.adaptation.lifecycle_agent import LifecycleAgent
from src.adaptation.recalibration_agent import RecalibrationAgent, RecalibrationError
from src.adaptation.regeneration_agent import RegenerationAgent, RegenerationError
from src.adaptation.verification_agent import VerificationAgent
from src.common import console_report as console
from src.common.config import Settings, load_secrets, load_settings
from src.common.logging import setup_logging
from src.common.run_logs import AdaptationDecisionLog, ComponentFidelityLog
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
from src.fidelity.component_policy import ComponentAdaptationPolicy, ComponentFidelityStatus
from src.fidelity.evaluator import FidelityEvaluator
from src.fidelity.metrics import FidelityComputationError
from src.fidelity.trigger import AdaptationTrigger
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

# The 7 OAI KPI components.
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


def _fmt4(value: float | None) -> str:
    """Console formatting only — logs/records keep full precision."""
    return f"{value:.4f}" if value is not None else "n/a"


class ContinuousOrchestrator:
    """Wires and runs the full canonical loop. Construct via `initialize()`, then `start()`
    (background telemetry sync thread) and `run()` (the foreground orchestration loop).
    Adaptation is triggered only by the internal per-component fidelity policy."""

    def __init__(self, settings: Settings, secrets: "Secrets") -> None:
        self._settings = settings
        self._secrets = secrets

        self._stop_event = threading.Event()
        # None until start(); lets stop() without a prior start() be a harmless no-op.
        self.sync_thread: threading.Thread | None = None
        # Pending per-component fidelity triggers, coalesce policy: at most one pending trigger
        # per component, merged by max severity, picked up once that component is no longer
        # mid-adaptation. See _enqueue_trigger/_pop_next_trigger/_handle_trigger.
        self._pending_lock = threading.RLock()
        self._pending_by_component: dict[str, AdaptationTrigger] = {}
        self._pending_order: list[str] = []  # component names, FIFO by first-pending arrival
        # Components currently mid-adaptation (one adaptation per component at a time; different
        # components may adapt concurrently on worker threads — see run()/_dispatch_pending()).
        self._active_components: set[str] = set()
        self._workers: dict[str, threading.Thread] = {}
        adaptation_cfg = settings.adaptation
        self._max_workers = adaptation_cfg.max_concurrent_adaptations if adaptation_cfg.allow_concurrent_components else 1
        # Live serving (the DTModelRegistry the predictions run through) is read by the fidelity
        # loop and swapped by adaptation workers — a prediction pass never sees a half-swapped set.
        self._serving_lock = threading.RLock()
        # Decision-log entry of each in-flight adaptation, by component — lets an abnormal end
        # (Ctrl-C, unexpected exception, shutdown) still be recorded.
        self._current_entries: dict[str, dict[str, Any]] = {}
        self._entry_lock = threading.Lock()
        self._reference_history_cache: pd.DataFrame | None = None
        self._reference_lock = threading.Lock()
        self._latest_fidelity: dict[str, float | None] = {}
        # Latest D1 ts_1s seen by the last successful fidelity cycle — lets
        # run_prediction_and_fidelity_cycle() detect "nothing new has arrived" (a finite replay
        # source fully consumed) and skip re-evaluating an identical, degenerate window.
        self._last_evaluated_max_ts: Any = None
        self._reported_replay_exhausted = False  # print the "now idling" notice once, not every cycle
        # Per-component previous-outcome tracking for the Decision Agent's context.
        self._previous_outcome_by_component: dict[str, PreviousOutcome] = {}
        # Per-component adaptation policy: one independent debounce counter per component, fed
        # each cycle with that component's own fresh FidelityScore_c (replaces the unified-score
        # trigger). `_component_status` holds the latest policy evaluation per component.
        self.adaptation_policy = ComponentAdaptationPolicy(settings.fidelity)
        self._component_status: dict[str, ComponentFidelityStatus] = {}
        self._cycle_count = 0
        self._last_ingest_sample: tuple[float, int] | None = None  # (monotonic time, records_synced)
        self.component_fidelity_log = ComponentFidelityLog(settings.resolve_path(settings.logging.component_fidelity_log))
        self.adaptation_decision_log = AdaptationDecisionLog(settings.resolve_path(settings.logging.adaptation_decisions_log))

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
        sync_config = settings.synchronization
        if settings.telemetry.source == "oai_replay" and settings.telemetry.replay.speed == 0:
            # Real bug, reproduced: D1Store.append_history() rewrites the WHOLE accumulated
            # history file on every flush (never an incremental append) — fine at the live-
            # streaming default batch_size=20 for genuine telemetry trickling in in real time, but
            # an O(n**2) blowup for a finite, speed=0 ("as fast as possible") replay, where
            # thousands of rows arrive in a tight loop almost instantly (the full TEST split is
            # 25,865 rows -> ~1,293 full-file rewrites at batch_size=20, observed to make
            # `--mode live` take many minutes just to finish loading). Larger batches here only
            # affect how fast the replay's OWN backlog drains into D1 — never genuine live
            # telemetry pacing, which is untouched (this branch only fires for a non-real-time
            # replay). Matches the same batch_size=500 override already proven in
            # baselines/_lib.py and the smoke test for exactly this reason.
            sync_config = sync_config.model_copy(update={"batch_size": 500})
        self.synchronizer = ContinuousSynchronizer(self.telemetry_source, self.preprocessor, self.d1_store, sync_config)

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
        components = list(self._dt_components.keys())
        policies = {}
        for name in components:
            policy = self.adaptation_policy.policy_for(name)
            policies[name] = (policy.threshold, policy.min_consecutive_evaluations)
        log_paths = {
            "full JSON log:": str(settings.resolve_path(settings.logging.log_dir) / "digital_twin.log"),
            "component fidelity log:": str(self.component_fidelity_log.path),
            "adaptation decisions:": str(self.adaptation_decision_log.path),
            "lifecycle records:": str(settings.resolve_path(settings.lifecycle.records_path)),
            "maintenance reports:": str(settings.resolve_path(settings.lifecycle.reports_dir)),
        }
        console.startup_banner(
            mode=settings.environment, telemetry_source=settings.telemetry.source,
            components=components, llm_available=self.llm_client is not None,
            rag_available=self.rag_kb.is_available, policies=policies, log_paths=log_paths,
            model_versions=self._model_versions(),
        )

    def _model_versions(self) -> dict[str, str]:
        """Current production version id per live component (for reporting)."""
        versions = {}
        for name in self._target_columns:
            current = self.model_registry.get_current_version(name)
            versions[name] = current.version_id if current is not None else "unregistered"
        return versions

    def _build_telemetry_source(self):
        settings = self._settings
        if settings.telemetry.source == "oai_replay":
            source = OaiReplaySource(
                settings.resolve_path("data/processed/oai"), speed=settings.telemetry.replay.speed,
                default_split=settings.telemetry.replay.default_split,
            )
            default_run_ids = getattr(settings.data.splits, settings.telemetry.replay.default_split)
            source.start(run_ids=default_run_ids)
            self._resume_replay_after_persisted_history(source, default_run_ids)
            return source
        return MockTelemetrySource(settings.telemetry.mock, max_records=None, realtime=True)

    def _resume_replay_after_persisted_history(self, source: OaiReplaySource, run_ids: list[int]) -> None:
        """D1 is persistent, so a restarted process already holds every tick a previous run of
        this same replay synchronized. Replaying those ticks again from the start would only
        re-send rows D1 already has — and since none of them is newer than D1's latest `ts_1s`,
        the fidelity cycle would treat the whole (possibly hour-long, at speed>0) replay as
        "nothing new", i.e. a silent stall. Resume right after the last persisted tick instead."""
        history = self.d1_store.get_history()
        if history.empty or "run_id" not in history.columns:
            return
        replayed = history[history["run_id"].isin(run_ids)]
        if replayed.empty:
            return
        last_ts = replayed["ts_1s"].max()
        source.seek(last_ts + pd.Timedelta(nanoseconds=1))
        message = (
            f"D1 already holds {len(replayed)} replayed rows for run(s) {run_ids} up to ts_1s={last_ts} — "
            "resuming the replay after that tick (delete data/artifacts/d1_*.parquet to replay from the start)"
        )
        logger.info(message, extra={"component": "main", "rows_already_in_d1": len(replayed)})
        console.skipped(message)

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
        logger.info(
            "orchestrator started — telemetry sync running in the background; adaptation is "
            "triggered only by the internal per-component fidelity policy",
            extra={"component": "main"},
        )

    def stop(self) -> None:
        """Deterministic shutdown: signal the background sync thread, then block (bounded by
        `shutdown_join_timeout_seconds`) until it actually exits — avoids a caller treating the
        orchestrator as stopped while a thread is still running/logging. In-flight adaptation
        workers get `adaptation.shutdown_wait_seconds` to finish (they stop retrying once the stop
        event is set); any still running after that is recorded as `interrupted`."""
        self._stop_event.set()
        self.synchronizer.stop()

        timeout = self._settings.synchronization.shutdown_join_timeout_seconds
        still_alive: list[str] = []
        if self.sync_thread is not None:
            self.sync_thread.join(timeout=timeout)
            if self.sync_thread.is_alive():
                still_alive.append(self.sync_thread.name)
        self._wait_for_adaptations(self._settings.adaptation.shutdown_wait_seconds, record_unfinished=True)
        # D1 disk writes are rate-limited (storage.persist_min_interval_seconds) — persist the tail.
        self.d1_store.flush()

        if still_alive:
            logger.warning(
                "orchestrator stop() timed out waiting for background thread(s) to exit",
                extra={"component": "main", "still_alive": still_alive, "timeout_seconds": timeout},
            )
        else:
            logger.info("orchestrator stopped — background sync thread confirmed exited", extra={"component": "main"})
        console.shutdown_line(clean=not still_alive)

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
        """FIFO by first-pending arrival, skipping components that are mid-adaptation (their
        trigger stays pending — coalescing further triggers — until the adaptation ends). Only
        removed here, just before dispatch, so a trigger that arrives while its component is
        adapting starts a fresh entry for after that adaptation."""
        with self._pending_lock:
            for index, component in enumerate(self._pending_order):
                if component in self._active_components:
                    continue
                self._pending_order.pop(index)
                return self._pending_by_component.pop(component)
            return None

    @property
    def active_adaptations(self) -> list[str]:
        """Components with an adaptation in flight (read-only snapshot)."""
        with self._pending_lock:
            return sorted(self._active_components)

    def _dispatch_pending(self, budget: int | None) -> int:
        """Start one worker thread per dispatchable pending trigger, up to the free worker slots
        (and `budget`, if given). The component is marked active BEFORE its thread starts, so a
        trigger for it can never be dispatched twice."""
        started = 0
        while budget is None or started < budget:
            with self._pending_lock:
                if len(self._workers) >= self._max_workers:
                    break
                trigger = self._pop_next_trigger()
                if trigger is None:
                    break
                self._active_components.add(trigger.component)
                worker = threading.Thread(
                    target=self._adaptation_worker, args=(trigger,), name=f"dt-adapt-{trigger.component}", daemon=True
                )
                self._workers[trigger.component] = worker
            worker.start()
            started += 1
        return started

    def _adaptation_worker(self, trigger: AdaptationTrigger) -> None:
        try:
            self._handle_trigger(trigger)
        except Exception as exc:  # a worker must never die silently
            logger.exception(
                "adaptation failed unexpectedly — production untouched, continuing",
                extra={"component": "main", "affected_component": trigger.component},
            )
            console.warning(f"adaptation for {trigger.component!r} failed unexpectedly: {exc}")
        finally:
            with self._pending_lock:
                self._workers.pop(trigger.component, None)
                self._active_components.discard(trigger.component)

    def _wait_for_adaptations(self, timeout: float, *, record_unfinished: bool = False) -> list[str]:
        """Block until no adaptation worker is running, or `timeout` seconds. Returns components
        still running; with `record_unfinished`, logs each as `interrupted`."""
        deadline = time.monotonic() + timeout
        while True:
            with self._pending_lock:
                workers = list(self._workers.values())
            if not workers:
                return []
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            workers[0].join(timeout=min(remaining, 0.5))
        with self._pending_lock:
            unfinished = sorted(self._workers)
        if unfinished:
            logger.warning(
                "adaptations still running at shutdown", extra={"component": "main", "components": unfinished}
            )
            console.warning(f"adaptation(s) still running at shutdown, recorded as interrupted: {', '.join(unfinished)}")
            if record_unfinished:
                for component in unfinished:
                    entry = self._current_entries.get(component)
                    if entry is not None:
                        self._finish_decision_entry(entry, "interrupted", "process shutting down")
        return unfinished

    # --- the continuous loop itself (prompt.md §39) ---------------------------------------------

    def run(self, max_drift_events: int | None = None, prediction_interval_seconds: float = 5.0) -> None:
        """Periodically runs prediction+fidelity (which enqueues one trigger per component whose
        own fidelity policy fired) and dispatches pending triggers to adaptation WORKER THREADS —
        one adaptation per component at a time, up to `adaptation.max_concurrent_adaptations`
        components at once — so fidelity cycles keep running while adaptations (LLM calls,
        training, sandbox) are in progress. Runs until `stop()`, or (demo/validation only) until
        `max_drift_events` adaptations have been dispatched AND finished; production passes
        `max_drift_events=None` and never stops on its own. Telemetry sync is unaffected."""
        dispatched = 0
        last_prediction = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now - last_prediction >= prediction_interval_seconds:
                last_prediction = now
                try:
                    self.run_prediction_and_fidelity_cycle()
                except Exception as exc:  # one bad cycle must never kill the continuous loop
                    logger.exception("prediction + fidelity cycle failed — continuing", extra={"component": "main"})
                    console.warning(f"prediction + fidelity cycle failed: {exc}")
            budget = None if max_drift_events is None else max_drift_events - dispatched
            if budget is None or budget > 0:
                dispatched += self._dispatch_pending(budget)
            if max_drift_events is not None and dispatched >= max_drift_events and not self.active_adaptations:
                logger.info(
                    "max adaptations reached — orchestration loop stopping (telemetry sync is "
                    "unaffected and keeps running until stop() is called separately)",
                    extra={"component": "main", "handled": dispatched},
                )
                return
            self._stop_event.wait(timeout=0.25)

    def run_prediction_and_fidelity_cycle(self) -> dict[str, float | None]:
        """Dependency-aware DT prediction -> fidelity, against live-growing D1 state, using the
        same `DTOrchestrator`/`FidelityEvaluator` for this orchestrator's whole lifetime so
        fidelity windows accumulate real history. Also evaluates the per-component adaptation
        policy and enqueues a trigger for every component whose policy fired. Public so a demo/validation
        script can call it directly to warm fidelity history before triggering an adaptation."""
        history = self.d1_store.get_history()
        if len(history) < self._settings.dt_models.bootstrap_min_rows:
            return {}
        latest_ts = history["ts_1s"].max()
        if self._last_evaluated_max_ts is not None and latest_ts <= self._last_evaluated_max_ts:
            # No genuinely new telemetry since the last cycle — e.g. a finite OAI replay source
            # (speed=0) loads its entire split in under a second, after which D1 never grows
            # again. Re-evaluating the IDENTICAL (y_true, y_pred) pair on every timer tick would
            # collapse the fidelity rolling window to zero variance (every entry becomes the same
            # value) and falsely pin FidelityScore at exactly 1.0 — a real bug this reproduced:
            # the same degenerate-window pitfall Module 17's own docstring documents for
            # repeated point-mass test fixtures, just arising naturally from replay exhaustion
            # instead. Skip the cycle entirely rather than manufacture a meaningless measurement.
            #
            # Real UX bug this introduced, found and fixed in the same change: silently returning
            # here means the console goes completely quiet for the rest of the run once a finite
            # replay is exhausted — indistinguishable from a genuine hang. Tell the user exactly
            # once, then stay quiet (never spam "still idle" every cycle).
            if not self._reported_replay_exhausted:
                self._reported_replay_exhausted = True
                console.skipped(
                    f"no new telemetry since ts_1s={latest_ts} — the replay source has been fully "
                    "consumed (this is expected for a finite OAI replay once it finishes loading); "
                    "the system is alive and idling, waiting for a fidelity-based trigger from "
                    "already-replayed history, if one is warranted"
                )
            return dict(self._latest_fidelity)
        self._last_evaluated_max_ts = latest_ts
        history_sorted = history.sort_values("ts_1s", kind="stable").reset_index(drop=True)
        window_size = self._settings.evaluation.window_size
        recent = history_sorted.tail(window_size)

        # Normally this method evaluates just the single most-recent window — live telemetry
        # grows a window's worth every few real ticks, so a handful of real calls is enough to
        # reach config.fidelity.min_history_for_normalization and that's the common path below.
        # But a finite OAI replay at speed=0 (the config default) can load its ENTIRE split
        # before this method is ever called a second time — and the skip-branch above means it
        # may then never run again at all (D1 never grows further). Without backfilling, every
        # component's rolling window would be permanently stranded at 1 point, fidelity could
        # never become defined, and a fidelity-based trigger could never fire. Reproduced for
        # real: plain `python -m src.main --mode live` with no extra flags never produced a
        # defined fidelity score. Fix: when enough already-replayed history exists beyond the
        # most-recent window, walk back through it in window_size chunks and evaluate each one
        # too, oldest first — warming the rolling window from genuinely different real historical
        # periods in one burst, exactly as several real ticks against slowly-growing live
        # telemetry would have produced anyway. On genuinely live/slow-growing telemetry every
        # component is already warmed by the time a deficit could exist, so n_backfill is 0 and
        # behavior is unchanged from before this fix.
        deficits = [
            max(0, self._settings.fidelity.min_history_for_normalization - self.fidelity_evaluator.history_length(name))
            for name in self._target_columns
        ]
        max_available_backfill = max(0, len(history_sorted) // window_size - 1)
        n_backfill = min(max(deficits, default=0), max_available_backfill)

        windows = []
        for i in range(n_backfill, 0, -1):
            end = len(history_sorted) - i * window_size
            start = max(0, end - window_size)
            chunk = history_sorted.iloc[start:end]
            if not chunk.empty:
                windows.append(chunk)
        windows.append(recent)

        # Only the most recent window's results are FRESH measurements for the adaptation policy;
        # backfilled windows only warm the rolling normalization reference.
        fresh_scores: dict[str, float | None] = {}
        scored_rows: dict[str, int] = {}
        for window_df in windows:
            is_latest = window_df is recent
            try:
                with self._serving_lock:  # never a half-swapped model set
                    predictions = self.dt_orchestrator.run_predictions(window_df)
            except OrchestratorError as exc:
                logger.warning("DT prediction cycle failed", extra={"component": "main", "error": str(exc)})
                continue

            for name, preds in predictions.items():
                target_column = self._target_columns.get(name)
                if target_column is None:
                    # An expand_scope component whose target_column wasn't recovered — predictions
                    # still serve; only fidelity tracking for it is skipped.
                    continue
                # Residual components legitimately return NaN for each session's first row (no
                # y_t-1 anchor). The fidelity formula must never silently score NaN rows, so mask here.
                target_values = np.asarray(window_df[target_column], dtype=float)
                pred_values = np.asarray(preds, dtype=float)
                valid_mask = np.isfinite(target_values) & np.isfinite(pred_values)
                if not valid_mask.any():
                    logger.debug(
                        "fidelity cycle: no rows with a valid prediction+target this cycle, skipping",
                        extra={"component": "main", "dt_component": name},
                    )
                    continue
                try:
                    result = self.fidelity_evaluator.evaluate(
                        name, target_values[valid_mask], pred_values[valid_mask], update_window=True
                    )
                except FidelityComputationError as exc:  # e.g. a single valid row — MK-MMD needs >= 2
                    logger.debug(
                        "fidelity cycle: window not scorable for this component, skipping",
                        extra={"component": "main", "dt_component": name, "error": str(exc)},
                    )
                    continue
                if is_latest:
                    fresh_scores[name] = result.fidelity_score
                    scored_rows[name] = int(valid_mask.sum())

        # Latest view: fresh score where measured this cycle, None (no current measurement) else —
        # a stale score from an earlier cycle is never presented as current.
        components = list(self._target_columns)
        self._latest_fidelity = {name: fresh_scores.get(name) for name in components}
        self._cycle_count += 1

        # Independent per-component adaptation decisions (replaces the unified-score trigger).
        statuses = self.adaptation_policy.evaluate(components, fresh_scores)
        self._component_status = {s.component: s for s in statuses}
        triggers = self.adaptation_policy.triggers(statuses)
        for trigger in triggers:
            self._enqueue_trigger(trigger)

        # Reporting only — never an adaptation trigger.
        unified = self.fidelity_evaluator.compute_unified_score(fresh_scores)
        self._report_cycle(statuses, recent, len(history), n_backfill, unified.unified_score, triggers, scored_rows)
        logger.info(
            "prediction + fidelity cycle complete",
            extra={
                "component": "main",
                "cycle": self._cycle_count,
                "rows": len(recent),
                "windows_evaluated": len(windows),
                "backfilled_windows": n_backfill,
                "fidelity": dict(self._latest_fidelity),
                "policy_states": {s.component: s.state for s in statuses},
                "triggers": [t.component for t in triggers],
                "unified_score_report_only": unified.unified_score,
            },
        )
        return dict(self._latest_fidelity)

    def _ingest_rate(self) -> float | None:
        synchronizer = getattr(self, "synchronizer", None)
        if synchronizer is None:
            return None
        now, synced = time.monotonic(), synchronizer.records_synced
        previous, self._last_ingest_sample = self._last_ingest_sample, (now, synced)
        if previous is None or now <= previous[0]:
            return None
        return (synced - previous[1]) / (now - previous[0])

    def _report_cycle(
        self, statuses: list[ComponentFidelityStatus], recent: pd.DataFrame, d1_rows: int, n_backfill: int,
        unified_score: float | None, triggers: list[AdaptationTrigger], scored_rows: dict[str, int],
    ) -> None:
        versions = self._model_versions()
        window_start, window_end = recent["ts_1s"].min(), recent["ts_1s"].max()
        console.cycle_report(
            self._cycle_count, window_start=window_start, window_end=window_end, window_rows=len(recent),
            d1_rows=d1_rows, ingest_rate=self._ingest_rate(), statuses=statuses, model_versions=versions,
            unified_score=unified_score, triggers=[t.component for t in triggers], backfilled_windows=n_backfill,
        )
        logged_at = datetime.now(UTC).isoformat()
        self.component_fidelity_log.append([
            {
                "logged_at": logged_at, "cycle": self._cycle_count, "component": s.component,
                "model_version": versions.get(s.component), "fidelity_score": s.fidelity_score,
                "threshold": s.threshold, "fidelity_gap": s.fidelity_gap, "state": s.state,
                "consecutive_below": s.consecutive_below, "min_consecutive_evaluations": s.min_consecutive_evaluations,
                "severity": s.severity, "window_start_ts": window_start, "window_end_ts": window_end,
                "window_rows": len(recent), "scored_rows": scored_rows.get(s.component, 0),
                "unified_score_report_only": unified_score, "reason": s.reason,
            }
            for s in statuses
        ])

    # --- the full adaptation cycle: Decision & Root-Cause Analysis Agent -> agent -> verification
    # -> lifecycle -------------------------------------------------------------------------------

    def _handle_trigger(self, trigger: AdaptationTrigger) -> None:
        """Runs the adaptation LOOP for one component (on a worker thread in `run()`, or directly).
        The component is marked active for the whole loop — no second adaptation of it can start
        — and always released in `finally` (ACCEPT, REJECT, failure or exception), so a failure
        can never deadlock it. An abnormal end (Ctrl-C, unexpected exception) is still recorded."""
        component = trigger.component
        with self._pending_lock:
            self._active_components.add(component)
        self.last_adaptation_started_at = time.monotonic()
        try:
            self._run_adaptation_loop(trigger)
        except BaseException as exc:
            entry = self._current_entries.get(component)
            if entry is not None and entry.get("outcome") is None:
                self._finish_decision_entry(
                    entry, "interrupted" if isinstance(exc, KeyboardInterrupt) else "error", str(exc) or type(exc).__name__
                )
            raise
        finally:
            self.last_adaptation_finished_at = time.monotonic()
            self._current_entries.pop(component, None)
            with self._pending_lock:
                self._active_components.discard(component)

    def _run_adaptation_loop(self, trigger: AdaptationTrigger) -> None:
        """Repeat decision -> agent -> verification for one component until a candidate is
        ACCEPTed, the component recovers above its threshold, the outcome is not retryable (no
        LLM, unknown component, not enough data yet), the system is stopping, or
        `adaptation.max_adaptation_attempts` attempts have been made. Each attempt's rejection
        details (strategy, verification numbers, failed checks, sandbox/agent error) are fed back
        to the decision agent and the adaptation agents for the next attempt."""
        component = trigger.component
        max_attempts = self._settings.adaptation.max_adaptation_attempts
        if self._trigger_is_stale(trigger):
            status = self._component_status[component]
            console.skipped(
                f"{component}: trigger from {trigger.timestamp:%H:%M:%S} is stale — the component has since recovered "
                f"(F_c={_fmt4(status.fidelity_score)} >= threshold {status.threshold}); no adaptation needed"
            )
            entry = self._new_decision_entry(trigger, attempt=1, max_attempts=max_attempts)
            entry["post_adaptation_fidelity"] = status.fidelity_score
            entry["still_below_threshold"] = False
            self._finish_decision_entry(entry, "skipped_recovered")
            return

        feedback: list[str] = []
        for attempt in range(1, max_attempts + 1):
            if attempt > 1 and self._stop_event.is_set():
                console.skipped(f"{component}: shutting down — adaptation loop stopped after {attempt - 1} attempt(s)")
                return
            result = self._run_adaptation_cycle(
                trigger, attempt=attempt, max_attempts=max_attempts, previous_attempts=tuple(feedback)
            )
            if not result:
                return
            outcome = result["outcome"]
            if outcome == "accepted":
                if attempt > 1:
                    console.step("loop", f"{component}: candidate ACCEPTED on attempt {attempt}/{max_attempts}", style="green")
                return
            if not result.get("retryable", False):
                return
            if result.get("still_below_threshold") is False:
                console.step(
                    "loop", f"{component}: now meets its threshold (F_c={_fmt4(result.get('post_adaptation_fidelity'))}) — "
                    f"no further attempts needed after attempt {attempt}/{max_attempts}",
                )
                return
            feedback.append(result["summary"])
            if attempt < max_attempts:
                console.step(
                    "retry", f"{component}: attempt {attempt}/{max_attempts} {outcome} — trying again with this "
                    f"feedback: {result['summary']}", style="yellow",
                )
        console.warning(
            f"{component}: no accepted candidate after {max_attempts} attempt(s) — production kept; the component "
            "stays flagged and its policy re-triggers it while it remains below threshold"
        )

    def _trigger_is_stale(self, trigger: AdaptationTrigger) -> bool:
        """A fidelity trigger whose component has been measured at-or-above its threshold AFTER
        the trigger fired (it waited while the component was busy) — adapting it now is pointless.
        Forced triggers (validation scripts/tests) are never treated as stale."""
        if trigger.trigger_type != "fidelity_degradation" or trigger.metadata.get("forced"):
            return False
        status = self._component_status.get(trigger.component)
        return (
            status is not None and status.state == "ok" and status.measured_at is not None
            and status.measured_at > trigger.timestamp
        )

    def _new_decision_entry(self, trigger: AdaptationTrigger, *, attempt: int, max_attempts: int) -> dict[str, Any]:
        component = trigger.component
        policy = self.adaptation_policy.policy_for(component)
        status = self._component_status.get(component)
        current = self.model_registry.get_current_version(component)
        version_before = current.version_id if current is not None else None
        entry: dict[str, Any] = {
            "logged_at": None, "component": component, "attempt": attempt, "max_attempts": max_attempts,
            "trigger_type": trigger.trigger_type, "trigger_severity": trigger.severity,
            "trigger_reason": trigger.metadata.get("reason"),
            "fidelity_at_trigger": trigger.metadata.get("fidelity_score", status.fidelity_score if status else None),
            "threshold": policy.threshold, "min_consecutive_evaluations": policy.min_consecutive_evaluations,
            "version_before": version_before, "strategy": None, "decision_confidence": None, "decision_source": None,
            "adaptation_attempted": False, "outcome": None, "error": None, "candidate_version": None,
            "verification_fidelity_before": None, "verification_fidelity_after": None, "failed_checks": [],
            "version_after": version_before, "post_adaptation_fidelity": None, "remaining_fidelity_gap": None,
            "still_below_threshold": None, "dependency_revalidation": [], "lifecycle_event_id": None,
            "duration_seconds": None, "_t0": time.monotonic(),
        }
        self._current_entries[component] = entry
        return entry

    def _finish_decision_entry(self, entry: dict[str, Any], outcome: str, error: str | None = None) -> None:
        """Write one decision-log line — exactly once per entry, even if a worker and shutdown race."""
        with self._entry_lock:
            if entry.get("outcome") is not None:
                return
            entry["outcome"] = outcome
            entry["error"] = error
            entry["logged_at"] = datetime.now(UTC).isoformat()
            started = entry.pop("_t0", None)
            entry["duration_seconds"] = round(time.monotonic() - started, 3) if started is not None else None
            self.adaptation_decision_log.append(entry)

    def _reference_history(self) -> pd.DataFrame | None:
        """The original TRAIN pool (data production was bootstrapped on), loaded once — used as
        reference training data by recalibration/regeneration. None if the dataset isn't built."""
        with self._reference_lock:
            if self._reference_history_cache is None:
                path = self._settings.resolve_path("data/processed/oai/train.parquet")
                if not path.exists():
                    return None
                self._reference_history_cache = pd.read_parquet(path, engine="pyarrow")
            return self._reference_history_cache

    def _previous_outcome_for(self, component: str) -> PreviousOutcome:
        return self._previous_outcome_by_component.get(component, PreviousOutcome())

    def _hot_swap_candidate(self, action_name: str, component: str, agent_result: Any) -> None:
        """Called only on a real ACCEPT — makes the candidate already promoted in `ModelRegistry`
        take over live prediction serving right now, via `dt_model_registry` (the in-memory,
        orchestrator-facing registry, distinct from `ModelRegistry`'s versioned artifact store).
        The swap itself happens under the serving lock, so a concurrent prediction pass sees
        either the old or the new model set, never a mix."""
        version = agent_result.version
        if action_name == "recalibrate":
            # Already a live instance of an already-trusted class — no dynamic loading.
            instance: DTComponent = agent_result.candidate_component
            cls = None
        else:
            # regenerate/expand_scope: import the class for the first time now that verification
            # has granted trust — never before.
            cls = self._load_dynamic_class(version)
            instance = cls()
            self.model_registry.load_artifact_into(instance, version)

        with self._serving_lock:
            if cls is not None:
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

    def _run_adaptation_cycle(
        self, trigger: AdaptationTrigger, *, attempt: int = 1, max_attempts: int = 1,
        previous_attempts: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """ONE adaptation attempt for one component: decision -> agent -> verification -> (hot
        swap) -> post-adaptation re-evaluation + dependency revalidation -> lifecycle record.
        Every outcome — including skips and failures that never reach verification — is written
        to the adaptation decisions log. Returns a small result dict for the adaptation loop."""
        settings = self._settings
        component = trigger.component
        policy = self.adaptation_policy.policy_for(component)
        entry = self._new_decision_entry(trigger, attempt=attempt, max_attempts=max_attempts)
        version_before = entry["version_before"]
        fidelity_at_trigger = entry["fidelity_at_trigger"]
        attempt_started = time.monotonic()
        tag = f"{attempt}/{max_attempts}"

        def finish(outcome: str, error: str | None = None, *, retryable: bool = False, summary: str | None = None) -> dict[str, Any]:
            self._finish_decision_entry(entry, outcome, error)
            return {
                "outcome": outcome, "retryable": retryable,
                "still_below_threshold": entry.get("still_below_threshold"),
                "post_adaptation_fidelity": entry.get("post_adaptation_fidelity"),
                "summary": summary or f"attempt {attempt}: {entry.get('strategy')} -> {outcome}" + (f" ({error})" if error else ""),
            }

        if attempt == 1:
            with self._pending_lock:
                still_pending = len(self._pending_order)
            console.trigger_seen(
                component, trigger.trigger_type, trigger.severity, fidelity_score=fidelity_at_trigger,
                threshold=policy.threshold, version=version_before, queue_remaining=still_pending,
            )
            if trigger.metadata.get("reason"):
                console.step("policy", f"{component}: {trigger.metadata['reason']}", style="dim")
        else:
            console.step("attempt", f"{component}: attempt {tag} (production {version_before})", style="yellow")
        component_cls = self._component_class_for(component)
        if component_cls is None:
            logger.warning("trigger names an unknown component, skipping", extra={"component": "main", "affected_component": component})
            console.skipped(f"trigger names an unknown component ({component!r})")
            return finish("skipped_unknown_component")

        context = build_decision_context(
            trigger,
            self._latest_fidelity,
            self.fidelity_evaluator.compute_unified_score(self._latest_fidelity).unified_score,
            self._previous_outcome_for(component),
            self.d1_store.get_history(),
            self.rag_kb,
            component_threshold=policy.threshold,
            fidelity_at_trigger=fidelity_at_trigger,
            attempt=attempt,
            max_attempts=max_attempts,
            previous_attempts=previous_attempts,
        )
        decision: DecisionOutput = self.decision_agent.decide_safe(context)
        action_name = decision.strategy
        # Same marker baselines/run_all.py uses — DecisionAgent._fallback()'s fixed wording.
        is_fallback = "infrastructure failure" in decision.root_cause_analysis
        # A test harness may substitute a strategy-pinning decision agent (scripts/test_adaptation_agents.py);
        # it declares its own DECISION_SOURCE so the log never mislabels a pinned decision as an LLM one.
        decision_source = "fallback" if is_fallback else getattr(self.decision_agent, "DECISION_SOURCE", "llm")
        entry.update(strategy=action_name, decision_confidence=decision.confidence, decision_source=decision_source)
        logger.info(
            "decision agent selected a strategy",
            extra={
                "component": "main", "affected_component": component, "trigger_type": trigger.trigger_type,
                "severity": trigger.severity, "strategy": action_name, "confidence": decision.confidence,
                "attempt": attempt,
            },
        )
        console.decision_made(
            component, action_name, decision.confidence, decision.root_cause_analysis, fallback=is_fallback, source=decision_source,
        )

        target_column = self._target_columns[component]
        # A dynamically-loaded (hot-swapped) class only guarantees a no-arg constructor — never
        # a from_settings() classmethod.
        if component in self._dynamic_component_classes:
            component_factory: Callable[[], DTComponent] = lambda cls=component_cls: cls()  # noqa: E731
        else:
            component_factory = lambda cls=component_cls: cls.from_settings(settings)  # noqa: E731
        drift_context = f"{trigger.trigger_type} on {component!r} at severity {trigger.severity:.3f}"
        if previous_attempts:
            drift_context += (
                f" | attempt {tag} of an adaptation loop; earlier attempts were not accepted: "
                + "; ".join(previous_attempts)
            )

        if action_name in ("regenerate", "expand_scope") and self.llm_client is None:
            logger.warning(
                "the decision agent selected an LLM-driven strategy but no GOOGLE_API_KEY is "
                "configured — skipping this adaptation cycle (telemetry ingestion and the next "
                "trigger are entirely unaffected)",
                extra={"component": "main", "action": action_name, "affected_component": component},
            )
            console.skipped(f"{action_name} selected for {component!r} but no LLM client is configured")
            return finish("skipped_no_llm_client")

        adaptation_cfg = settings.adaptation
        window_hours = {
            "recalibrate": adaptation_cfg.recalibration.training_window_hours,
            "regenerate": adaptation_cfg.regeneration.training_window_hours,
            "expand_scope": adaptation_cfg.expand_scope.training_window_hours,
        }[action_name]
        console.step("agent", f"{action_name} running for {component} only (production {version_before}, training window {window_hours}h)")
        entry["adaptation_attempted"] = True
        try:
            if action_name == "recalibrate":
                agent = RecalibrationAgent(settings, self.d1_store, self.model_registry, fidelity_evaluator=None, llm_client=self.llm_client)
                agent_result = agent.recalibrate(
                    component_factory, target_column, dependency_output_fields=self._output_fields,
                    reference_history=self._reference_history() if adaptation_cfg.recalibration.include_reference_data else None,
                    recent_sample_weight=adaptation_cfg.recalibration.recent_sample_weight,
                )
            elif action_name == "regenerate":
                agent = RegenerationAgent(settings, self.d1_store, self.model_registry, self.llm_client, self.sandbox_executor, fidelity_evaluator=None)
                agent_result = agent.regenerate(
                    component_factory, target_column, dependency_output_fields=self._output_fields, drift_context=drift_context,
                    rag_knowledge_base=self.rag_kb,
                    reference_history=self._reference_history() if adaptation_cfg.regeneration.include_reference_data else None,
                )
            else:  # expand_scope
                agent = ExpandScopeAgent(
                    settings, self.d1_store, self.model_registry, self.dt_model_registry, self.llm_client, self.sandbox_executor,
                    fidelity_evaluator=None,
                )
                agent_result = agent.expand_scope(
                    component_factory, dependency_output_fields=self._output_fields, expand_scope_context=drift_context,
                    rag_knowledge_base=self.rag_kb,
                )
        except (RecalibrationError, RegenerationError, ExpandScopeError) as exc:
            # Data conditions a retry can't change right now (too little history; no new KPI to model).
            insufficient_data = any(marker in str(exc) for marker in ("rows available", "no unpredicted same-second KPI"))
            logger.warning(
                "adaptation agent failed to produce a candidate — production is untouched, "
                "telemetry ingestion is unaffected",
                extra={"component": "main", "action": action_name, "affected_component": component, "error": str(exc), "attempt": attempt},
            )
            console.warning(f"{action_name} failed for {component!r} (attempt {tag}): {exc} — production {version_before} unchanged")
            return finish(
                "agent_failed", str(exc), retryable=not insufficient_data,
                summary=f"attempt {attempt}: {action_name} produced no candidate — {str(exc)[:300]}",
            )

        candidate_version = agent_result.version
        entry["candidate_version"] = candidate_version.version_id
        reference_rows = agent_result.training_window.get("reference_rows") or 0
        console.step(
            "agent",
            f"candidate {candidate_version.version_id} ({candidate_version.model_class}) produced — "
            f"recent train rows {agent_result.training_window.get('n_rows')}"
            + (f" + {reference_rows} reference rows" if reference_rows and action_name == "recalibrate" else "")
            + f", held-out rows {agent_result.evaluation_window.get('n_rows')}"
            + (f", sandbox attempts {agent_result.attempts}" if hasattr(agent_result, "attempts") else ""),
        )

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
            production_factory = None  # genuinely new component, no production model

        # A new component's "old" side is persistence: its target's own previous-second value.
        reference_predictions = None
        if action_name == "expand_scope" and f"{eval_target_column}_lag1" in held_out_df.columns:
            reference_predictions = held_out_df[f"{eval_target_column}_lag1"]

        verification_agent = VerificationAgent(settings, self.model_registry, self.fidelity_evaluator, llm_client=self.llm_client)
        try:
            verification_result = verification_agent.verify(
                candidate_version, held_out_df, eval_target, candidate_predictions,
                production_component_factory=production_factory, baseline_predictions=reference_predictions,
                drift_context=drift_context, rag_knowledge_base=self.rag_kb,
            )
        except Exception as exc:  # verification must never crash the orchestration loop
            logger.warning("verification failed unexpectedly — treating as REJECT-equivalent, no promotion occurred", extra={"component": "main", "error": str(exc)})
            console.warning(f"verification failed unexpectedly for {component!r}: {exc} (treated as REJECT)")
            # Never leave the candidate stranded in "candidate" status forever.
            try:
                self.model_registry.reject(candidate_version.component, candidate_version.version_id)
            except Exception:
                logger.exception("could not mark the unverified candidate rejected", extra={"component": "main"})
            return finish("verification_error", str(exc), retryable=True)

        accepted = verification_result.decision == "ACCEPT"
        failed_checks = [c.name for c in verification_result.checks if not c.passed]
        entry.update(
            verification_fidelity_before=verification_result.fidelity_before,
            verification_fidelity_after=verification_result.fidelity_after, failed_checks=failed_checks,
        )
        console.step(
            "verify",
            f"{component} {candidate_version.version_id} {verification_result.decision}: fidelity "
            f"{_fmt4(verification_result.fidelity_before)} -> "
            f"{_fmt4(verification_result.fidelity_after)} "
            f"(required improvement > {verification_result.verification_delta}); checks: "
            + ", ".join(f"{c.name}={'PASS' if c.passed else 'FAIL'}" for c in verification_result.checks),
            style="green" if accepted else "red",
        )

        # Hot swap: an ACCEPT takes over live serving immediately, no restart required. Never on
        # REJECT — production is left exactly as verification decided.
        hot_swapped = False
        if accepted:
            try:
                self._hot_swap_candidate(action_name, component, agent_result)
                hot_swapped = True
                console.step("hot-swap", f"{candidate_version.version_id} now serving live predictions for {component}", style="green")
            except Exception as exc:  # must never hide an already-recorded ACCEPT or crash the
                # loop — picked up correctly on the next restart via _bootstrap_dt_models.
                logger.warning(
                    "candidate was ACCEPTed and promoted, but hot-swapping into live serving "
                    "failed — it will still be picked up on the next process restart",
                    extra={"component": "main", "affected_component": component, "action": action_name, "error": str(exc)},
                )
                console.warning(f"candidate ACCEPTed but hot-swap into live serving failed for {component!r}: {exc}")

        version_after = candidate_version.version_id if accepted else version_before
        # Re-evaluate the adapted component, and — only if live serving actually changed — the
        # components downstream of it, against the same shared normalization reference.
        validation = self._post_adaptation_validation(component, revalidate_dependents=hot_swapped)
        entry.update(
            version_after=version_after, post_adaptation_fidelity=validation["fidelity_score"],
            remaining_fidelity_gap=validation["remaining_gap"], still_below_threshold=validation["still_below_threshold"],
            dependency_revalidation=validation["dependents"],
        )
        console.step(
            "post-check",
            f"{component} F_c={_fmt4(validation['fidelity_score'])} vs threshold {policy.threshold} -> {validation['status']}",
            style="yellow" if validation["still_below_threshold"] else None,
        )
        if hot_swapped:
            if validation["dependents"]:
                for dep in validation["dependents"]:
                    console.step("dependents", f"revalidated {dep['component']}: F_c={_fmt4(dep['fidelity_score'])} threshold={dep['threshold']} -> {dep['status']}")
            else:
                console.step("dependents", f"no components depend on {component} — nothing to revalidate")
        else:
            console.step("dependents", f"{component}: production unchanged — no dependency revalidation required", style="dim")

        duration = time.monotonic() - attempt_started

        # Record which real telemetry source/run ids/replay speed drove this event.
        source_label = getattr(self.telemetry_source, "SOURCE_LABEL", "MOCK")
        replayed_run_ids = getattr(self.telemetry_source, "_run_ids", None) if source_label == "OAI_RAN_REPLAY" else None
        replay_speed = settings.telemetry.replay.speed if source_label == "OAI_RAN_REPLAY" else None
        record = self.lifecycle_agent.record_adaptation_event(
            trigger=trigger, decision=decision, agent_result=agent_result, verification_result=verification_result,
            telemetry_source=source_label, replayed_run_ids=replayed_run_ids, replay_speed=replay_speed,
            component_policy_outcome={
                "component_threshold": policy.threshold, "fidelity_at_trigger": fidelity_at_trigger,
                "production_version_after": version_after, "post_adaptation_fidelity": validation["fidelity_score"],
                "remaining_fidelity_gap": validation["remaining_gap"], "still_below_threshold": validation["still_below_threshold"],
                "dependency_revalidation": validation["dependents"],
                "adaptation_attempt": attempt, "max_adaptation_attempts": max_attempts,
            },
        )
        report = self.lifecycle_agent.generate_maintenance_report(record, rag_knowledge_base=self.rag_kb)
        entry["lifecycle_event_id"] = record.event_id
        console.step("lifecycle", f"{component}: record {record.event_id} written; maintenance report {report.path.name} (LLM prose={report.llm_used})", style="dim")

        console.adaptation_event(
            component, action_name, verification_result.decision, verification_result.fidelity_before,
            verification_result.fidelity_after, version_before, version_after, duration,
            threshold=policy.threshold, post_adaptation_fidelity=validation["fidelity_score"],
            remaining_gap=validation["remaining_gap"], still_flagged=validation["still_below_threshold"],
            dependents=validation["dependents"], event_id=record.event_id, attempt=tag,
        )

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
                "fidelity_after": verification_result.fidelity_after, "post_adaptation_fidelity": validation["fidelity_score"],
                "still_below_threshold": validation["still_below_threshold"], "attempt": attempt,
                "dependents_revalidated": [d["component"] for d in validation["dependents"]],
            },
        )
        summary = (
            f"attempt {attempt}: {action_name} candidate {candidate_version.version_id} "
            f"{verification_result.decision} — candidate fidelity {_fmt4(verification_result.fidelity_after)} vs "
            f"production {_fmt4(verification_result.fidelity_before)} (needs > production + {verification_result.verification_delta})"
            + (f"; failed checks: {', '.join(failed_checks)}" if failed_checks else "")
        )
        return finish("accepted" if accepted else "rejected", retryable=not accepted, summary=summary)

    def _downstream_components(self, component: str) -> list[str]:
        """Every live component that (transitively) consumes `component`'s predictions, in
        dependency (topological) order — derived from the registry, never hardcoded."""
        enabled = self.dt_model_registry.enabled_components()
        downstream: list[str] = []
        frontier = [component]
        while frontier:
            upstream = frontier.pop(0)
            for name, instance in enabled.items():
                if upstream in instance.DEPENDENCIES and name not in downstream and name != component:
                    downstream.append(name)
                    frontier.append(name)
        try:
            order = self.dt_orchestrator.build_execution_order()
            downstream.sort(key=order.index)
        except (OrchestratorError, ValueError):
            pass
        return downstream

    def _measure_current_fidelity(self, recent: pd.DataFrame, predictions: dict[str, pd.Series], name: str) -> float | None:
        """FidelityScore_c of the CURRENT live model on `recent`, against the shared rolling
        normalization reference WITHOUT appending to it (update_window=False) — a read-only
        re-measurement; `None` if not measurable (no target/prediction, too few scorable rows,
        or insufficient history)."""
        target_column = self._target_columns.get(name)
        if target_column is None or name not in predictions:
            return None
        target_values = np.asarray(recent[target_column], dtype=float)
        pred_values = np.asarray(predictions[name], dtype=float)
        mask = np.isfinite(target_values) & np.isfinite(pred_values)
        if mask.sum() < 2:
            return None
        try:
            return self.fidelity_evaluator.evaluate(name, target_values[mask], pred_values[mask], update_window=False).fidelity_score
        except FidelityComputationError:
            return None

    def _post_adaptation_validation(self, component: str, *, revalidate_dependents: bool) -> dict[str, Any]:
        """Post-adaptation check for `component` (its own threshold) plus, when live serving
        changed, targeted revalidation of its downstream dependents. Dependents are only
        measured and reported — never auto-adapted here; one that falls below its own threshold
        is picked up by its own per-component policy on subsequent cycles."""
        policy = self.adaptation_policy.policy_for(component)
        result: dict[str, Any] = {
            "fidelity_score": None, "threshold": policy.threshold, "remaining_gap": None,
            "still_below_threshold": None, "status": "not measurable (no current scorable window)", "dependents": [],
        }
        history = self.d1_store.get_history()
        if len(history) < 2:
            return result
        recent = history.sort_values("ts_1s", kind="stable").tail(self._settings.evaluation.window_size)
        try:
            with self._serving_lock:
                predictions = self.dt_orchestrator.run_predictions(recent)
        except OrchestratorError as exc:
            result["status"] = f"not measurable (prediction failed: {exc})"
            return result

        score = self._measure_current_fidelity(recent, predictions, component)
        if score is not None:
            gap = max(0.0, policy.threshold - score)
            result.update(
                fidelity_score=score, remaining_gap=gap, still_below_threshold=score < policy.threshold,
                status=(f"STILL BELOW threshold (gap {gap:.4f}) — remains flagged; its policy re-triggers after "
                        f"{policy.min_consecutive_evaluations} more consecutive below-threshold cycle(s)")
                if score < policy.threshold else "meets threshold",
            )

        if revalidate_dependents:
            for dep in self._downstream_components(component):
                dep_policy = self.adaptation_policy.policy_for(dep)
                dep_score = self._measure_current_fidelity(recent, predictions, dep)
                if dep_score is None:
                    dep_status = "not measurable"
                elif dep_score < dep_policy.threshold:
                    dep_status = "BELOW threshold — flagged for its own policy (not auto-adapted)"
                else:
                    dep_status = "ok — consistent with the new upstream model"
                result["dependents"].append(
                    {"component": dep, "fidelity_score": dep_score, "threshold": dep_policy.threshold, "status": dep_status}
                )
        return result


def build_orchestrator(mode: str | None = None, replay_speed_override: float | None = None) -> ContinuousOrchestrator:
    settings = load_settings()
    if mode is not None:
        telemetry_source = "oai_replay" if mode == "live" else "mock"
        settings = settings.model_copy(
            update={"telemetry": settings.telemetry.model_copy(update={"source": telemetry_source}), "environment": mode}
        )
    if replay_speed_override is not None:
        # `speed=0` ("as fast as possible") loads a whole finite OAI run in well under 5 seconds
        # at the batch_size=500 override above — genuinely correct, but a poor demonstration for
        # a human watching live: the system goes idle almost immediately, since there's nothing
        # more to replay. A real, positive speed paces ticks in (near-)real time instead, so the
        # run visibly spans the duration the caller actually wants (e.g. a CLI --replay-speed
        # chosen so the whole split takes ~10-15 minutes rather than seconds or the recorded
        # run's own full ~62-minute real duration).
        settings = settings.model_copy(
            update={"telemetry": settings.telemetry.model_copy(
                update={"replay": settings.telemetry.replay.model_copy(update={"speed": replay_speed_override})}
            )}
        )
    secrets = load_secrets()
    orchestrator = ContinuousOrchestrator(settings, secrets)
    orchestrator.initialize()
    return orchestrator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI-Driven Self-Adaptive Network Digital Twin — continuous orchestration loop")
    parser.add_argument("--mode", choices=["demo", "live"], default=None, help="demo=mock telemetry, live=real OAI RAN KPI replay; omit to use config/settings.yaml as-is")
    parser.add_argument(
        "--max-adaptations", "--max-drift-events", dest="max_adaptations", type=int, default=None,
        help="stop the orchestration loop after this many adaptation cycles (telemetry sync keeps "
        "running until stop()); omit for real continuous operation. --max-drift-events is a "
        "deprecated alias.",
    )
    parser.add_argument("--prediction-interval-seconds", type=float, default=5.0)
    parser.add_argument(
        "--replay-speed", type=float, default=None,
        help="--mode live only: pace the OAI replay instead of loading it near-instantly "
        "(config default speed=0 = 'as fast as possible', which loads a whole finite run in "
        "seconds and then correctly goes idle, since there is nothing left to replay). "
        "1.0 = real time (the recorded run's own real duration); e.g. 5.0 replays 5x faster than "
        "real time. Omit to use config/settings.yaml's speed as-is.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print full structured JSON logs to the console too (default: console shows only the "
        "clean status/adaptation-event lines + WARNING+; the log file always has full detail)",
    )
    args = parser.parse_args(argv)

    # Before build_orchestrator(): initialize() (bootstrap training, RAG/registry/D1 loading)
    # logs too, and those records must reach the configured handlers/log file rather than
    # Python's unformatted last-resort stderr handler.
    base_settings = load_settings()
    log_settings = base_settings.logging
    setup_logging(
        level=log_settings.level, fmt=log_settings.format, log_dir=log_settings.log_dir,
        console_level=log_settings.level if args.verbose else None,
    )
    if log_settings.console_transcript_dir:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        transcript = base_settings.resolve_path(log_settings.console_transcript_dir) / f"{args.mode or 'run'}_{stamp}.log"
        console.set_transcript(transcript)
        console.info(f"console transcript: {transcript}")
    orchestrator = build_orchestrator(mode=args.mode, replay_speed_override=args.replay_speed)
    orchestrator.start()
    try:
        orchestrator.run(max_drift_events=args.max_adaptations, prediction_interval_seconds=args.prediction_interval_seconds)
    except KeyboardInterrupt:  # Ctrl-C is the documented way to stop a continuous run, not an error
        console.skipped("interrupted (Ctrl-C) — shutting down cleanly")
    finally:
        orchestrator.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
