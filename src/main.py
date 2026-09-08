"""`src/main.py` — Phase 11: the top-level continuous orchestration loop (prompt.md §39,
CLAUDE.md §2/§10). This is the final piece of the 19-module architecture: it wires every
already-built module together into one real, continuously-running system.

**This file is integration only** — it constructs already-tested components (D1Store,
DTModelRegistry/DTOrchestrator, FidelityEvaluator, ContinuousSynchronizer, DriftDetectorInterface,
the PPO runtime, Modules 14/15/16's agents, VerificationAgent, LifecycleAgent, RagKnowledgeBase,
AnthropicClient) and calls their already-tested public methods in exactly the order prompt.md §39
specifies. No new model, fidelity, verification, or agent logic is written here.

Canonical loop (CLAUDE.md §2):
    NS-3/mock telemetry -> Module 2 (preprocess) -> Module 3 (continuous sync) -> D1 (Module 4)
    -> Modules 5-10 (dependency-aware DT prediction) -> Module 12 (fidelity)
    -> Module 11 (drift) -> Module 13 (PPO) -> Module 14/15/16 (execution)
    -> Module 17 (verification) -> Module 19 (lifecycle record) -> continue.

**The one hard non-negotiable this file exists to prove** (prompt.md §0.6/§0.8, CLAUDE.md §8
"continuous operation"): telemetry ingestion/D1 synchronization must NEVER stop or block while an
adaptation cycle (which can include an LLM call and/or a sandboxed subprocess training run) is in
progress. This is a STRUCTURAL guarantee, not a sequencing accident: `ContinuousSynchronizer` runs
on its own background daemon thread (started once at `start()`, exactly as Module 3 already
guarantees — nothing in this file's own loop ever calls into it again), and the drift ->
PPO -> agent -> verification -> lifecycle cycle runs entirely in the orchestrator's own foreground
loop thread, so a slow adaptation cycle can only ever delay the NEXT drift event/prediction cycle,
never a telemetry batch sync. `scripts/run_orchestrator_demo.py` is the permanent, repo-tracked
validation script that proves this concretely with a real sampler thread — see that script and
CLAUDE.md's own Phase 11 entry for the actual observed numbers from a real run.

**PPO's runtime observation** (`_build_runtime_observation`) uses the EXACT same vector schema
`src.adaptation.rl_env.AdaptationEnv._build_observation()` was trained against — [fidelity vector,
affected-component one-hot, drift severity, previous-action one-hot, previous reward, network-
state summary] — but populated from genuinely REAL sources instead of that module's synthetic
training-time simulation: real per-component `FidelityEvaluator` scores from Modules 5-10's own
live predictions against real D1 ground truth (Module 12), the real drift event's own
component/severity, this orchestrator's own tracked previous action/reward (from the last REAL
adaptation cycle's REAL verified fidelity delta), and a real D1-history-derived network-state
summary (normalized the same self-referential min-max way `build_network_state_pool` already
does, adapted for a live/streaming window instead of a precomputed pool). Reusing `AdaptationEnv`
itself for this would be WRONG — it always fabricates synthetic prediction fidelity internally
for training purposes, which would silently ignore the real system's actual fidelity.

**LLM availability**: no real `ANTHROPIC_API_KEY` is configured in this development environment.
This orchestrator constructs a real `AnthropicClient` when a key IS configured; when PPO selects
`regenerate`/`expand_scope` and no client is available, the cycle is skipped with a clear WARNING
(never a silent substitute, never a fabricated LLM response) — telemetry ingestion and the next
drift event are entirely unaffected. `recalibrate` never needs an LLM at all.
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from src.adaptation.data_selection import select_recent_window, time_split, with_dependency_ground_truth
from src.adaptation.expand_scope_agent import ExpandScopeAgent, ExpandScopeError
from src.adaptation.lifecycle_agent import LifecycleAgent
from src.adaptation.recalibration_agent import RecalibrationAgent, RecalibrationError
from src.adaptation.regeneration_agent import RegenerationAgent, RegenerationError
from src.adaptation.rl_agent import decide_adaptation_strategy_safe, load_ppo_agent
from src.adaptation.rl_env import NETWORK_STATE_FEATURES, _FIDELITY_SCORE_CLIP
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
from src.llm.anthropic_client import AnthropicClient
from src.rag.rag_kb import RagKnowledgeBase
from src.registry.model_registry import ModelRegistry
from src.sandbox.executor import SandboxExecutor
from src.synchronization.sync import ContinuousSynchronizer
from src.telemetry.mock_source import MockTelemetrySource
from src.telemetry.preprocessing import TelemetryPreprocessor
from src.telemetry.zmq_source import ZmqTelemetrySource

if TYPE_CHECKING:
    from src.common.config import Secrets
    from src.drift.schema import DriftEvent

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
        self._drift_queue: "queue.Queue[DriftEvent]" = queue.Queue()
        self._latest_fidelity: dict[str, float | None] = {}
        self._prev_action_idx: int | None = None
        self._prev_reward: float = 0.0

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
        self.lifecycle_agent = LifecycleAgent.from_settings(settings)

        try:
            self.llm_client: AnthropicClient | None = AnthropicClient.from_settings(settings, self._secrets)
        except RuntimeError as exc:
            self.llm_client = None
            logger.warning(
                "no ANTHROPIC_API_KEY configured — regenerate/expand_scope adaptation cycles "
                "will be skipped (with a clear warning) if PPO ever selects them; recalibrate is "
                "unaffected",
                extra={"component": "main", "error": str(exc)},
            )

        try:
            self.ppo_model = load_ppo_agent(settings)
        except FileNotFoundError as exc:
            self.ppo_model = None
            logger.warning(
                "no trained PPO policy found — decide_adaptation_strategy_safe will use the "
                "configured deterministic fallback for every drift event until one is trained",
                extra={"component": "main", "error": str(exc)},
            )

        self._bootstrap_dt_models()

        self.dt_model_registry = DTModelRegistry()
        for name in _BOOTSTRAP_ORDER:
            instance = self._dt_components[name]
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
                "ppo_available": self.ppo_model is not None,
                "rag_available": self.rag_kb.is_available,
            },
        )

    def _build_telemetry_source(self):
        settings = self._settings
        if settings.telemetry.source == "zmq":
            return ZmqTelemetrySource(settings.telemetry.zmq)
        return MockTelemetrySource(settings.telemetry.mock, max_records=None, realtime=True)

    def _bootstrap_dt_models(self) -> None:
        """"Load/train bootstrap DT models" (prompt.md §39): for each of the five real components,
        LOAD the current production artifact if one is already registered (a resumed process),
        else TRAIN a fresh one on a real bootstrap telemetry batch and register it as the very
        first production version."""
        settings = self._settings
        self._dt_components: dict[str, DTComponent] = {}

        need_bootstrap_data = any(self.model_registry.get_current_version(name) is None for name in _BOOTSTRAP_ORDER)
        bootstrap_history: pd.DataFrame | None = None
        if need_bootstrap_data:
            bootstrap_history = self._generate_bootstrap_history()

        for name in _BOOTSTRAP_ORDER:
            cls = _DT_COMPONENT_CLASSES[name]
            current_version = self.model_registry.get_current_version(name)
            instance = cls.from_settings(settings)
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
            self._drift_queue.put(event)

    # --- the continuous loop itself (prompt.md §39) ---------------------------------------------

    def run(self, max_drift_events: int | None = None, prediction_interval_seconds: float = 5.0) -> None:
        """The foreground orchestration loop: periodically run DT prediction + fidelity
        evaluation, and dispatch every drift event through the full PPO -> agent -> verification
        -> lifecycle cycle. Runs until `stop()` is called, or (for demo/validation runs only)
        until `max_drift_events` adaptation cycles have completed — production/live usage passes
        `max_drift_events=None` and never stops on its own (prompt.md §39: "must not stop after
        one adaptation event"). Telemetry ingestion (started separately, in `start()`) is
        completely unaffected by anything this loop does — see this module's own docstring."""
        handled = 0
        last_prediction = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now - last_prediction >= prediction_interval_seconds:
                self.run_prediction_and_fidelity_cycle()
                last_prediction = now
            try:
                event = self._drift_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._handle_drift_event(event)
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
        Public (not `_`-prefixed): `run()` calls this periodically on its own, but a caller (e.g.
        a validation/demo script) may also call it directly to let real fidelity history
        accumulate past `config.fidelity.min_history_for_normalization` before triggering an
        adaptation cycle — exactly the same "warm the rolling window with real evaluations before
        relying on it" discipline `AdaptationEnv._prewarm_fidelity_windows()` already established
        for training, applied here with genuinely live data instead of a synthetic simulation."""
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
            target_column = _TARGET_COLUMNS[name]
            result = self.fidelity_evaluator.evaluate(name, recent[target_column], preds, update_window=True)
            self._latest_fidelity[name] = result.fidelity_score
        logger.info(
            "prediction + fidelity cycle complete",
            extra={"component": "main", "rows": len(recent), "fidelity": dict(self._latest_fidelity)},
        )
        return dict(self._latest_fidelity)

    # --- runtime PPO observation (see module docstring for why this is NOT AdaptationEnv) -------

    def _build_runtime_observation(self, event: "DriftEvent") -> np.ndarray:
        settings = self._settings
        components = tuple(settings.drift.valid_components)
        fidelity_vec = np.array(
            [
                float(np.clip(self._latest_fidelity.get(c) or 0.0, *_FIDELITY_SCORE_CLIP))
                for c in components
            ],
            dtype=np.float32,
        )
        affected_onehot = np.zeros(len(components), dtype=np.float32)
        if event.component in components:
            affected_onehot[components.index(event.component)] = 1.0
        lo, hi = settings.drift.severity_range
        severity_norm = np.array([(event.severity - lo) / max(hi - lo, 1e-9)], dtype=np.float32)
        n_actions = len(settings.ppo.action_mapping)
        prev_action_onehot = np.zeros(n_actions, dtype=np.float32)
        if self._prev_action_idx is not None:
            prev_action_onehot[self._prev_action_idx] = 1.0
        reward_clip = settings.ppo.env.reward_clip
        prev_reward = np.array([np.clip(self._prev_reward, -reward_clip, reward_clip)], dtype=np.float32)
        network_state = self._compute_network_state_summary()
        obs = np.concatenate([fidelity_vec, affected_onehot, severity_norm, prev_action_onehot, prev_reward, network_state])
        return np.clip(obs, -2.0, 2.0).astype(np.float32)

    def _compute_network_state_summary(self) -> np.ndarray:
        history = self.d1_store.get_history()
        if len(history) == 0:
            return np.zeros(len(NETWORK_STATE_FEATURES), dtype=np.float32)
        recent = history.tail(500)
        values = recent[list(NETWORK_STATE_FEATURES)].to_numpy(dtype=np.float64)
        mean_vec = values.mean(axis=0)
        feature_min = values.min(axis=0)
        feature_max = values.max(axis=0)
        span = np.where(feature_max - feature_min > 1e-9, feature_max - feature_min, 1.0)
        normalized = (mean_vec - feature_min) / span
        return normalized.astype(np.float32)

    # --- the full adaptation cycle: PPO -> agent -> verification -> lifecycle -------------------

    def _handle_drift_event(self, event: "DriftEvent") -> None:
        self.last_adaptation_started_at = time.monotonic()
        try:
            self._run_adaptation_cycle(event)
        finally:
            self.last_adaptation_finished_at = time.monotonic()

    def _run_adaptation_cycle(self, event: "DriftEvent") -> None:
        settings = self._settings
        component = event.component
        if component not in _DT_COMPONENT_CLASSES:
            logger.warning("drift event names an unknown component, skipping", extra={"component": "main", "affected_component": component})
            return

        observation = self._build_runtime_observation(event)
        action_idx, action_name = decide_adaptation_strategy_safe(self.ppo_model, observation, settings)
        logger.info(
            "PPO decision", extra={"component": "main", "affected_component": component, "severity": event.severity, "action": action_name}
        )

        component_cls = _DT_COMPONENT_CLASSES[component]
        target_column = _TARGET_COLUMNS[component]
        component_factory: Callable[[], DTComponent] = lambda cls=component_cls: cls.from_settings(settings)  # noqa: E731
        drift_context = f"drift on {component!r} at severity {event.severity:.3f} (source={event.source})"

        if action_name in ("regenerate", "expand_scope") and self.llm_client is None:
            logger.warning(
                "PPO selected an LLM-driven strategy but no ANTHROPIC_API_KEY is configured — "
                "skipping this adaptation cycle (telemetry ingestion and the next drift event are "
                "entirely unaffected)",
                extra={"component": "main", "action": action_name, "affected_component": component},
            )
            return

        try:
            if action_name == "recalibrate":
                agent = RecalibrationAgent(settings, self.d1_store, self.model_registry, fidelity_evaluator=None, llm_client=self.llm_client)
                agent_result = agent.recalibrate(component_factory, target_column, dependency_output_fields=_OUTPUT_FIELDS)
                window_hours = settings.adaptation.recalibration.training_window_hours
            elif action_name == "regenerate":
                agent = RegenerationAgent(settings, self.d1_store, self.model_registry, self.llm_client, self.sandbox_executor, fidelity_evaluator=None)
                agent_result = agent.regenerate(
                    component_factory, target_column, dependency_output_fields=_OUTPUT_FIELDS, drift_context=drift_context
                )
                window_hours = settings.adaptation.regeneration.training_window_hours
            else:  # expand_scope
                agent = ExpandScopeAgent(
                    settings, self.d1_store, self.model_registry, self.dt_model_registry, self.llm_client, self.sandbox_executor,
                    fidelity_evaluator=None,
                )
                agent_result = agent.expand_scope(
                    component_factory, dependency_output_fields=_OUTPUT_FIELDS, expand_scope_context=drift_context
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
        window_df = select_recent_window(self.d1_store.get_history(), window_hours)
        _, held_out_df = time_split(window_df, 0.2)
        if component_cls.DEPENDENCIES:
            held_out_df = with_dependency_ground_truth(held_out_df, component_cls.DEPENDENCIES, _OUTPUT_FIELDS)
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

        # Module 19: record + report.
        record = self.lifecycle_agent.record_adaptation_event(
            drift_event=event, rl_observation=observation, rl_action=action_name, agent_result=agent_result, verification_result=verification_result,
        )
        self.lifecycle_agent.generate_maintenance_report(record, rag_knowledge_base=self.rag_kb)

        # Feed this cycle's REAL outcome into the NEXT observation's previous-action/previous-
        # reward slots — the same reward formula CLAUDE.md §6 specifies system-wide.
        fidelity_before = verification_result.fidelity_before or 0.0
        fidelity_after = verification_result.fidelity_after or 0.0
        cost_penalty = settings.ppo.reward.adaptation_cost_penalty.get(action_name, 0.0)
        self._prev_action_idx = action_idx
        self._prev_reward = (fidelity_after - fidelity_before) - cost_penalty

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
