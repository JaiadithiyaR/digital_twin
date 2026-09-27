"""`src/main.py` — Phase 11: the top-level continuous orchestration loop (prompt.md §39,
CLAUDE.md §2/§10). This is the final piece of the 19-module architecture: it wires every
already-built module together into one real, continuously-running system.

**This file is integration only** — it constructs already-tested components (D1Store,
DTModelRegistry/DTOrchestrator, FidelityEvaluator, ContinuousSynchronizer, DriftDetectorInterface,
the Decision & Root-Cause Analysis Agent, Modules 14/15/16's agents, VerificationAgent,
LifecycleAgent, RagKnowledgeBase, GoogleClient) and calls their already-tested public methods in
exactly the order prompt.md §39 specifies. No new model, fidelity, verification, or agent logic is
written here.

Canonical loop (CLAUDE.md §2):
    NS-3/mock telemetry -> Module 2 (preprocess) -> Module 3 (continuous sync) -> D1 (Module 4)
    -> Modules 5-10 (dependency-aware DT prediction) -> Module 12 (fidelity + unified score/trigger)
    -> Module 11 (drift) -> Module 13 (Decision & Root-Cause Analysis Agent) -> Module 14/15/16
    (execution) -> Module 17 (verification) -> Module 19 (lifecycle record) -> continue.

**The one hard non-negotiable this file exists to prove** (prompt.md §0.6/§0.8, CLAUDE.md §8
"continuous operation"): telemetry ingestion/D1 synchronization must NEVER stop or block while an
adaptation cycle (which can include an LLM call and/or a sandboxed subprocess training run) is in
progress. This is a STRUCTURAL guarantee, not a sequencing accident: `ContinuousSynchronizer` runs
on its own background daemon thread (started once at `start()`, exactly as Module 3 already
guarantees — nothing in this file's own loop ever calls into it again), and the trigger ->
decision agent -> agent -> verification -> lifecycle cycle runs entirely in the orchestrator's own
foreground loop thread, so a slow adaptation cycle can only ever delay the NEXT trigger/prediction cycle,
never a telemetry batch sync. `scripts/run_orchestrator_demo.py` is the permanent, repo-tracked
validation script that proves this concretely with a real sampler thread — see that script and
CLAUDE.md's own Phase 11 entry for the actual observed numbers from a real run.

**Design pivot** (see CLAUDE.md §12's design-pivot notice): Module 13 is no longer a locally-run,
pre-trained PPO policy — it is the knowledge-based Decision & Root-Cause Analysis Agent
(`src.adaptation.decision_agent.DecisionAgent`), making a genuine LLM call on every single
trigger. There is no observation vector or trained policy artifact anymore; `_build_decision_
context()` (thin glue around `src.adaptation.decision_context.build_decision_context`) assembles
a bounded, structured snapshot instead — real per-component `FidelityEvaluator` scores, the real
Unified Fidelity Score (Module 12, §16a), the real trigger's component/severity/type, this
orchestrator's own tracked previous outcome PER COMPONENT (from the last REAL verified fidelity
delta for that component), real RAG-retrieved context, and a real D1-history-derived network-state
summary.

**Two trigger sources, normalized uniformly** (prompt.md §16a): an external `DriftEvent` (Module
11) and this orchestrator's own periodic fidelity-based trigger check (`FidelityEvaluator.
check_fidelity_trigger()`, Module 12) both funnel into the SAME `AdaptationTrigger` queue and the
SAME dispatch path — the Decision & Root-Cause Analysis Agent never needs to know which one fired.

**LLM availability**: no real `GOOGLE_API_KEY`/`GEMINI_API_KEY` is configured in this development
environment. This orchestrator constructs a real `GoogleClient` when a key IS configured (even a
placeholder counts as "configured" from this code's perspective — the actual live call is what
fails, gracefully, not construction). Unlike the old PPO design, the DECISION itself now requires
a genuine LLM call every time — `DecisionAgent.decide_safe()` degrades to the configured
deterministic fallback strategy on any LLM infrastructure failure (never a silent substitute,
never a fabricated response), logged at WARNING every time it happens. If the resulting strategy
is `regenerate`/`expand_scope` and no LLM client is available to actually EXECUTE it, that cycle
is separately skipped with its own clear WARNING — telemetry ingestion and the next trigger are
entirely unaffected either way. `recalibrate` never needs an LLM to execute.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import queue
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

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
from src.dt_models.base import DTComponent
from src.dt_models.d1_model_store import D1Store
from src.dt_models.jitter import JitterModel
from src.dt_models.latency import LatencyModel
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator, OrchestratorError
from src.dt_models.packet_loss import PacketLossModel
from src.dt_models.prb_utilization import PrbUtilizationModel
from src.dt_models.throughput import ThroughputModel
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
from src.telemetry.preprocessing import TelemetryPreprocessor
from src.telemetry.zmq_source import ZmqTelemetrySource

if TYPE_CHECKING:
    from src.common.config import Secrets

logger = logging.getLogger(__name__)

# The five real DT prediction components (Modules 6-10) — the only ones a drift event can
# legitimately name (config.drift.valid_components already enforces this at the Module 11 layer;
# these two dicts are this file's own generic dispatch table, never hardcoded per-branch logic).
_DT_COMPONENT_CLASSES: dict[str, type[DTComponent]] = {
    "throughput": ThroughputModel,
    "packet_loss": PacketLossModel,
    "latency": LatencyModel,
    "prb_utilization": PrbUtilizationModel,
    "jitter": JitterModel,
}
_TARGET_COLUMNS: dict[str, str] = {
    "throughput": "throughput_mbps",
    "packet_loss": "packet_loss_pct",
    "latency": "latency_ms",
    "prb_utilization": "prb_utilization_pct",
    "jitter": "jitter_ms",
}
# Fixed topological order (matches DTOrchestrator.build_execution_order()'s own result — proven
# in tests/integration/test_full_dependency_chain.py) so dependency-having components can be
# bootstrap-trained with their dependency's ground-truth column already available.
_BOOTSTRAP_ORDER: tuple[str, ...] = ("throughput", "packet_loss", "latency", "prb_utilization", "jitter")
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
        # ONLY affects what severities the (mock) drift source GENERATES — never what
        # DriftDetectorInterface accepts as valid, which always stays at the full configured
        # range. Used by the validation run to bias toward low-severity (recalibrate-eligible)
        # incidents so a full, genuinely non-fabricated cycle completes without needing a real
        # LLM client in an environment where none is configured — see this module's own
        # docstring and CLAUDE.md's Phase 11 entry for why this is honest, not scripted.
        self._drift_severity_range_override = drift_severity_range_override

        self._stop_event = threading.Event()
        # Both canonical trigger sources (external DriftEvent, normalized; and the internal
        # fidelity-based trigger) funnel into this ONE queue as AdaptationTrigger instances
        # (prompt.md §16a) — the dispatch path in run() never needs to know which one fired.
        self._trigger_queue: "queue.Queue[AdaptationTrigger]" = queue.Queue()
        self._latest_fidelity: dict[str, float | None] = {}
        # Per-component previous-outcome tracking for the Decision & Root-Cause Analysis Agent's
        # context (prompt.md §18 "previous action taken for this component/incident and its
        # outcome, where available") — replaces the old PPO design's single scalar
        # previous-action-index/previous-reward, since there is no fixed-size observation vector
        # to populate anymore, and outcomes are naturally per-component, not global.
        self._previous_outcome_by_component: dict[str, PreviousOutcome] = {}

        # Set by _handle_drift_event around the adaptation cycle's own wall-clock window — the
        # public hook the validation script reads to bracket its telemetry-growth sampler,
        # mirroring Modules 14/15/16's own concurrency-proof test pattern exactly.
        self.last_adaptation_started_at: float | None = None
        self.last_adaptation_finished_at: float | None = None

    # --- initialization (prompt.md §39: config -> storage -> RAG -> registry -> bootstrap models
    # -> telemetry source), each step delegated to an already-built module -----------------------

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
        # No training phase, unlike the PPO policy this replaces (prompt.md §21) — the decision
        # agent is a direct LLM call, always constructed the same way regardless of whether a real
        # key is configured (DecisionAgent accepts llm_client=None and degrades via decide_safe()
        # exactly as if a configured client's call had failed).
        self.decision_agent = DecisionAgent.from_settings(settings, self.llm_client)
        # Real bug found and fixed while validating against a genuine live key: this MUST be
        # constructed AFTER self.llm_client above and passed it explicitly — LifecycleAgent's own
        # optional LLM-enhanced maintenance-report prose (prompt.md §38) is unreachable without it,
        # silently falling back to the deterministic template every time regardless of whether a
        # working client exists, with no warning ever logged (generate_maintenance_report() only
        # warns on an LLM call that was actually attempted and failed, never on one that was never
        # attempted at all).
        self.lifecycle_agent = LifecycleAgent.from_settings(settings, llm_client=self.llm_client)

        # Per-component dispatch bookkeeping, seeded from the five statically-known DT components
        # but GROWABLE at runtime — an expand_scope candidate that gets ACCEPTed introduces a
        # genuinely new component name neither dict originally knew about (see
        # `_hot_swap_candidate`); `_bootstrap_dt_models` below restores both for any such
        # component that already existed in a PRIOR process's ModelRegistry, so it survives a
        # restart exactly like the five original components already did.
        self._target_columns: dict[str, str] = dict(_TARGET_COLUMNS)
        self._output_fields: dict[str, str] = dict(_OUTPUT_FIELDS)
        # Classes dynamically loaded from a regenerate/expand_scope candidate's saved LLM-
        # generated source (never the five originals, which stay statically imported) — populated
        # by both `_bootstrap_dt_models` (resuming one from a prior process) and
        # `_hot_swap_candidate` (promoting one for the first time, live).
        self._dynamic_component_classes: dict[str, type[DTComponent]] = {}

        self._bootstrap_dt_models()

        self.dt_model_registry = DTModelRegistry()
        for name, instance in self._dt_components.items():
            enabled = settings.dt_models.enable_prb_model if name == "prb_utilization" else True
            self.dt_model_registry.register(instance, enabled=enabled)
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
        if settings.telemetry.source == "zmq":
            return ZmqTelemetrySource(settings.telemetry.zmq)
        return MockTelemetrySource(settings.telemetry.mock, max_records=None, realtime=True)

    def _bootstrap_dt_models(self) -> None:
        """"Load/train bootstrap DT models" (prompt.md §39), extended for hot-swap "permanence"
        across a restart: for each of the five original components, LOAD the current production
        artifact if one is already registered (a resumed process), else TRAIN a fresh one on a
        real bootstrap telemetry batch and register it as the very first production version — the
        original prompt.md §39 behavior, unchanged. PLUS, for any OTHER component name the
        ModelRegistry has EVER tracked a version for (`list_component_names()`) — i.e. one an
        expand_scope candidate created in a prior process's lifetime and this one never heard of
        until now — LOAD it too, dynamically (see `_load_dynamic_class`): it can only exist here
        because a real ACCEPT already happened for it, so there is never a bootstrap-train-fresh
        case for one of these — only load what was already produced."""
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
                # A regenerate candidate was ACCEPTed for this component at some point — its
                # production artifact belongs to an LLM-generated class, never the original
                # statically-imported one, so loading it into a freshly-constructed static
                # instance here would silently mix an unrelated class's weights into the wrong
                # implementation. Load the ACTUAL trained class instead, same as a fresh hot swap.
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
        """Dynamically loads `version`'s LLM-generated class and its trained artifact, and
        restores this orchestrator's own target-column/output-field bookkeeping for it (needed
        for a genuinely new expand_scope component the first time it's resumed — a no-op
        overwrite for a regenerated EXISTING component, whose target column is already the
        correct one from `_TARGET_COLUMNS`)."""
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
        """Imports a regenerate/expand_scope candidate's LLM-generated class from its saved
        source file — the one point in this whole system where LLM-generated code is imported
        into the trusted, long-running process rather than only ever run inside the sandbox
        subprocess (`src/sandbox/executor.py`). Every caller of this method only ever does so for
        a version that has ALREADY passed the sandbox's syntax/import/conformance/train/evaluate
        pipeline AND Module 17's deterministic fidelity gate — either just now (`_hot_swap_
        candidate`, called only on a real ACCEPT) or in a past process's lifetime (`_bootstrap_
        dt_models`/`_load_and_restore`, which can only ever find a production version here in the
        first place because some past ACCEPT already promoted it) — never on a rejected or
        still-pending candidate."""
        source_path = self.model_registry.source_path(version)
        if source_path is None:
            raise RuntimeError(f"version {version.version_id!r} of {version.component!r} has no saved source to load")
        spec = importlib.util.spec_from_file_location(f"_dt_dynamic_{version.component}_{version.version_id}", source_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, version.model_class)

    def _component_class_for(self, name: str) -> "type[DTComponent] | None":
        """The class currently backing live serving for `name` — one of the five originals, or
        whichever class a hot swap or a restart-restore most recently loaded for it (see
        `_dynamic_component_classes`). `None` if `name` isn't a real, currently-known component at
        all (never guessed at — the caller must treat this as "not a valid adaptation target")."""
        # Dynamic MUST be checked first: a class object is always truthy, so
        # `_DT_COMPONENT_CLASSES.get(name) or ...` would silently keep returning the ORIGINAL
        # static class for one of the five originals even after a real regenerate hot swap
        # replaced it — caught by a real test, not by inspection (see tests/unit/test_main.py).
        return self._dynamic_component_classes.get(name) or _DT_COMPONENT_CLASSES.get(name)

    def _generate_bootstrap_history(self) -> pd.DataFrame:
        """A real, bounded telemetry batch through the real Module 2 pipeline — written into a
        throwaway in-memory-only preprocessing pass (NOT the live D1Store; bootstrap data is a
        distinct provenance category from live telemetry, per CLAUDE.md's Module 3 entry's scope
        boundary), used only to bootstrap-train the five DT models before live operation begins."""
        settings = self._settings
        mock_config = settings.telemetry.mock.model_copy(
            update={"num_ues": 25, "num_cells": 4, "missing_field_rate": 0.0, "out_of_range_rate": 0.0}
        )
        source = MockTelemetrySource(mock_config, max_records=1200, realtime=False)
        preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)
        result = preprocessor.process_batch(source.records())
        bootstrap_store = D1Store(
            current_state_path=settings.resolve_path(settings.storage.bootstrap_dir) / "_bootstrap_current.parquet",
            history_path=settings.resolve_path(settings.storage.bootstrap_dir) / "_bootstrap_history.parquet",
            quarantine_path=settings.resolve_path(settings.storage.bootstrap_dir) / "_bootstrap_quarantine.parquet",
            history_retention_rows=len(result.clean_records) + 1,
        )
        bootstrap_store.append_history(result.clean_records)
        history = bootstrap_store.get_history()
        logger.info("bootstrap telemetry generated", extra={"component": "main", "rows": len(history)})
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
        self._stop_event.set()
        self.synchronizer.stop()
        self.drift_source.close()
        logger.info("orchestrator stopped", extra={"component": "main"})

    def _drift_consumer_loop(self) -> None:
        for event in self.drift_detector.process_stream(self.drift_source):
            if self._stop_event.is_set():
                break
            self._trigger_queue.put(trigger_from_drift_event(event))

    # --- the continuous loop itself (prompt.md §39) ---------------------------------------------

    def run(self, max_drift_events: int | None = None, prediction_interval_seconds: float = 5.0) -> None:
        """The foreground orchestration loop: periodically run DT prediction + fidelity
        evaluation (which may itself enqueue a fidelity-based trigger), and dispatch every queued
        trigger — external drift or fidelity-based, both already normalized to the same
        `AdaptationTrigger` shape — through the full Decision & Root-Cause Analysis Agent -> agent
        -> verification -> lifecycle cycle. Runs until `stop()` is called, or (for demo/validation
        runs only) until `max_drift_events` adaptation cycles have completed — production/live
        usage passes `max_drift_events=None` and never stops on its own (prompt.md §39: "must not
        stop after one adaptation event"). Telemetry ingestion (started separately, in `start()`)
        is completely unaffected by anything this loop does — see this module's own docstring."""
        handled = 0
        last_prediction = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now - last_prediction >= prediction_interval_seconds:
                self.run_prediction_and_fidelity_cycle()
                last_prediction = now
            try:
                trigger = self._trigger_queue.get(timeout=0.5)
            except queue.Empty:
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
        """Modules 5-10 (dependency-aware prediction) -> Module 12 (fidelity), against the real,
        live-growing D1 state — the SAME `DTOrchestrator`/`FidelityEvaluator` used throughout this
        orchestrator's lifetime, so fidelity windows genuinely accumulate real history over time.
        Also checks Module 12's internal fidelity-based trigger (prompt.md §16a) after every
        update and enqueues it exactly like an external drift event if it fires — the SECOND of
        the two canonical trigger sources. Public (not `_`-prefixed): `run()` calls this
        periodically on its own, but a caller (e.g. a validation/demo script) may also call it
        directly to let real fidelity history accumulate past
        `config.fidelity.min_history_for_normalization` before triggering an adaptation cycle."""
        history = self.d1_store.get_history()
        if len(history) < self._settings.dt_models.bootstrap_min_rows:
            return {}
        recent = history.sort_values("timestamp", kind="stable").tail(self._settings.evaluation.window_size).reset_index(drop=True)
        try:
            predictions = self.dt_orchestrator.run_predictions(recent)
        except OrchestratorError as exc:
            logger.warning("DT prediction cycle failed", extra={"component": "main", "error": str(exc)})
            return {}

        for name, preds in predictions.items():
            target_column = self._target_columns.get(name)
            if target_column is None:
                # Only reachable for a hot-swapped-in (or restart-restored) expand_scope
                # component whose target_column couldn't be recorded/recovered — predictions
                # still serve correctly (see above); only fidelity tracking for THIS component
                # is skipped until that's fixed, never the whole cycle (see _load_and_restore).
                continue
            result = self.fidelity_evaluator.evaluate(name, recent[target_column], preds, update_window=True)
            self._latest_fidelity[name] = result.fidelity_score
        logger.info(
            "prediction + fidelity cycle complete",
            extra={"component": "main", "rows": len(recent), "fidelity": dict(self._latest_fidelity)},
        )

        fidelity_trigger = self.fidelity_evaluator.check_fidelity_trigger(self._latest_fidelity)
        if fidelity_trigger is not None:
            self._trigger_queue.put(fidelity_trigger)

        return dict(self._latest_fidelity)

    # --- the full adaptation cycle: Decision & Root-Cause Analysis Agent -> agent -> verification
    # -> lifecycle -------------------------------------------------------------------------------

    def _handle_trigger(self, trigger: AdaptationTrigger) -> None:
        self.last_adaptation_started_at = time.monotonic()
        try:
            self._run_adaptation_cycle(trigger)
        finally:
            self.last_adaptation_finished_at = time.monotonic()

    def _previous_outcome_for(self, component: str) -> PreviousOutcome:
        return self._previous_outcome_by_component.get(component, PreviousOutcome())

    def _hot_swap_candidate(self, action_name: str, component: str, agent_result: Any) -> None:
        """Called only on a real ACCEPT (see the one caller, `_run_adaptation_cycle`) — makes the
        candidate `VerificationAgent.verify()` already promoted in the versioned `ModelRegistry`
        take over LIVE prediction serving in THIS process right now, via `self.dt_model_registry`
        (Module 5's in-memory, orchestrator-facing registry — deliberately distinct from Module
        14's versioned artifact store, see `src/dt_models/model_registry.py`'s own docstring)."""
        version = agent_result.version
        if action_name == "recalibrate":
            # Already a live, trained instance of an already-imported, trusted class (the SAME
            # class already backing production) — no dynamic code loading involved at all.
            instance: DTComponent = agent_result.candidate_component
        else:
            # regenerate/expand_scope: until this exact moment, this candidate's class has only
            # ever existed as saved source on disk plus a sandboxed, fidelity-gate-ACCEPTed
            # artifact — load both now, for the first time in this trusted process (prompt.md
            # §28/§70 rule 10: only after verification has granted that trust, never before).
            cls = self._load_dynamic_class(version)
            instance = cls()
            self.model_registry.load_artifact_into(instance, version)
            self._dynamic_component_classes[component] = cls

        if component in self.dt_model_registry:
            self.dt_model_registry.replace(instance)
        else:
            # A genuinely new expand_scope component, served live for the very first time —
            # `_validate_design` already guarantees this name was not already registered, so
            # `agent_result.design` is only ever reached for the strategy that actually has one.
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
        # A dynamically-loaded class (a hot-swapped regenerate/expand_scope candidate) only ever
        # guarantees a no-arg constructor — the sandbox's own conformance contract for LLM-
        # generated code (src/sandbox/_sandbox_driver.py) — never a `from_settings(settings)`
        # classmethod the way the five hand-authored originals provide. component_factory must
        # therefore construct it the same way the sandbox itself always has.
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

        # Module 17: verify on the same held-out protocol the agent itself used. If the target
        # component itself has dependencies (e.g. latency/prb_utilization/jitter), populate their
        # ground-truth columns the same "train/verify on ground truth, serve on live predictions"
        # way every agent already does (data_selection.with_dependency_ground_truth) — needed both
        # for a recalibration candidate's own .predict() call below AND for the production
        # baseline's .predict() call VerificationAgent makes internally, since both are instances
        # of the SAME dependency-having class.
        if action_name == "recalibrate":
            # candidate_predictions is computed FRESH below, against WHATEVER held_out_df this
            # builds — a freshly re-derived "most recent window" slice is fine here; there is no
            # staleness to guard against, unlike the sandboxed-candidate case below.
            window_df = select_recent_window(self.d1_store.get_history(), window_hours)
            _, held_out_df = time_split(window_df, 0.2)
            if component_cls.DEPENDENCIES:
                held_out_df = with_dependency_ground_truth(held_out_df, component_cls.DEPENDENCIES, self._output_fields)
        else:
            # regenerate/expand_scope: candidate_predictions (below) is a FIXED array the sandbox
            # already computed against the agent's OWN internal held-out slice, selected BEFORE
            # its (potentially long: LLM call(s) + real sandboxed subprocess training) run. Real
            # telemetry keeps flowing for that entire duration by design (prompt.md §0.6/§0.8), so
            # independently re-deriving "most recent window, then last 20%" a second time here,
            # AFTER that call returns, can select a DIFFERENT-SIZED slice than the fixed-size
            # predictions array — verification would then REJECT on a length mismatch that has
            # nothing to do with the candidate's real quality (a genuine bug found and fixed while
            # validating the hot-swap path against fast-flowing live telemetry: even reconstructing
            # by the agent's own recorded evaluation_window timestamp range isn't exact, since a
            # single tick can span several UEs and the agent's ROW-count-based split can legally
            # cut a tick's rows between train and held-out). Reuse the agent's own held_out_df
            # directly instead — it already has dependency ground-truth columns populated too.
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

        # Hot swap: make an ACCEPT take over LIVE prediction serving immediately, not just the
        # versioned ModelRegistry on disk (which VerificationAgent.verify() itself already
        # promoted into a moment ago) — no process restart required. Never on REJECT: production
        # is left exactly as verification decided it should be.
        if verification_result.decision == "ACCEPT":
            try:
                self._hot_swap_candidate(action_name, component, agent_result)
            except Exception as exc:  # a hot-swap failure must never hide an already-recorded
                # ACCEPT or crash the loop — the promoted version is still correctly picked up
                # the next time this process restarts, via _bootstrap_dt_models/_load_and_restore.
                logger.warning(
                    "candidate was ACCEPTed and promoted in the versioned registry, but hot-"
                    "swapping it into live serving failed — it will still be picked up on the "
                    "next process restart",
                    extra={"component": "main", "affected_component": component, "action": action_name, "error": str(exc)},
                )

        # Module 19: record + report.
        record = self.lifecycle_agent.record_adaptation_event(
            trigger=trigger, decision=decision, agent_result=agent_result, verification_result=verification_result,
        )
        self.lifecycle_agent.generate_maintenance_report(record, rag_knowledge_base=self.rag_kb)

        # Feed this cycle's REAL outcome into the NEXT decision context's previous-outcome slot
        # for THIS component (prompt.md §18 "previous action taken for this component/incident and
        # its outcome, where available") — per-component now, not a single global scalar, since
        # there is no fixed-size observation vector to populate anymore.
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
        telemetry_source = "zmq" if mode == "live" else "mock"
        settings = settings.model_copy(
            update={"telemetry": settings.telemetry.model_copy(update={"source": telemetry_source}), "environment": mode}
        )
    secrets = load_secrets()
    orchestrator = ContinuousOrchestrator(settings, secrets, drift_severity_range_override=drift_severity_range_override)
    orchestrator.initialize()
    return orchestrator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI-Driven Self-Adaptive Network Digital Twin — continuous orchestration loop")
    parser.add_argument("--mode", choices=["demo", "live"], default=None, help="demo=mock telemetry, live=ZeroMQ telemetry; omit to use config/settings.yaml as-is")
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
