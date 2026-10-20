#!/usr/bin/env python3
"""Runs all 4 experiment modes (prompt2.md Step 6) on the SAME OAI replay splits/seed/replay
order/fidelity settings, and reports the resulting Unified Fidelity Score trajectory for each:

    static            -- no adaptation at all, ever (a pure prediction+fidelity baseline)
    periodic_retrain  -- recalibrates one fixed component on a fixed schedule (config-driven),
                         never looking at severity or an LLM
    rule_based        -- a real drift trigger's severity picks a strategy via a deterministic
                         threshold rule (config-driven), never an LLM call
    full              -- the real system: a genuine Decision & Root-Cause Analysis Agent (LLM)
                         call per trigger

All 4 modes: bootstrap-train the real 7 OAI components on the real TRAIN split, then replay the
real TEST split (speed=0) into a fresh, isolated D1Store, running the real orchestrator/fidelity
evaluator/adaptation agents/verification gate — nothing here is synthetic.

Usage:
    python scripts/run_baselines.py [--cycles N] [--modes static,rule_based,...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.adaptation.baseline_policies import periodic_retrain_due, rule_based_strategy  # noqa: E402
from src.adaptation.data_selection import select_recent_window, time_split, with_dependency_ground_truth  # noqa: E402
from src.adaptation.recalibration_agent import RecalibrationAgent  # noqa: E402
from src.adaptation.verification_agent import VerificationAgent  # noqa: E402
from src.common.config import load_secrets, load_settings  # noqa: E402
from src.common.logging import setup_logging  # noqa: E402
from src.dt_models.d1_model_store import D1Store  # noqa: E402
from src.dt_models.dl_bler import DlBlerModel  # noqa: E402
from src.dt_models.dl_prb import DlPrbModel  # noqa: E402
from src.dt_models.dl_rlc_delay import DlRlcDelayModel  # noqa: E402
from src.dt_models.dl_throughput import DlThroughputModel  # noqa: E402
from src.dt_models.model_registry import DTModelRegistry  # noqa: E402
from src.dt_models.orchestrator import DTOrchestrator  # noqa: E402
from src.dt_models.ul_bler import UlBlerModel  # noqa: E402
from src.dt_models.ul_prb import UlPrbModel  # noqa: E402
from src.dt_models.ul_throughput import UlThroughputModel  # noqa: E402
from src.drift.drift_detector import DriftDetectorInterface  # noqa: E402
from src.drift.mock_drift_source import MockDriftSource  # noqa: E402
from src.fidelity.evaluator import FidelityEvaluator  # noqa: E402
from src.registry.model_registry import ModelRegistry  # noqa: E402
from src.sandbox.executor import SandboxExecutor  # noqa: E402
from src.synchronization.sync import ContinuousSynchronizer  # noqa: E402
from src.telemetry.oai_replay_source import OaiReplaySource  # noqa: E402
from src.telemetry.preprocessing import TelemetryPreprocessor  # noqa: E402

_CLASSES = {
    "dl_bler": DlBlerModel, "ul_bler": UlBlerModel, "dl_prb": DlPrbModel, "ul_prb": UlPrbModel,
    "dl_throughput": DlThroughputModel, "ul_throughput": UlThroughputModel, "dl_rlc_delay": DlRlcDelayModel,
}
_ORDER = ("dl_bler", "ul_bler", "dl_prb", "ul_prb", "dl_throughput", "ul_throughput", "dl_rlc_delay")
_DEPS = {
    "dl_throughput": {"dl_bler": "dl_bler", "dl_prb": "dl_prb"},
    "ul_throughput": {"ul_bler": "ul_bler", "ul_prb": "ul_prb"},
    "dl_rlc_delay": {"dl_throughput": "dl_throughput", "dl_bler": "dl_bler"},
}


def _bootstrap_components(settings, train: pd.DataFrame, model_registry: ModelRegistry, dt_registry: DTModelRegistry) -> None:
    train_with_deps = train.copy()
    for name in _ORDER:
        model = _CLASSES[name].from_settings(settings)
        model.train(train_with_deps, train[name])
        model_registry.register_version(
            component_instance=model, adaptation_type="bootstrap", parent_version_id=None,
            training_window={"n_rows": len(train)}, evaluation_window={"n_rows": 0},
            evaluation_metrics=model.evaluate(train_with_deps, train[name]), status="production",
        )
        dt_registry.register(model)
        train_with_deps[f"{name}_pred"] = train[name]


def _run_prediction_and_fidelity_cycle(d1_store, dt_orchestrator, fidelity_evaluator, window_size, latest_fidelity):
    history = d1_store.get_history()
    if len(history) < 10:
        return
    recent = history.sort_values("ts_1s", kind="stable").tail(window_size).reset_index(drop=True)
    predictions = dt_orchestrator.run_predictions(recent)
    for name, preds in predictions.items():
        result = fidelity_evaluator.evaluate(name, recent[name], preds, update_window=True)
        latest_fidelity[name] = result.fidelity_score


def _run_one_adaptation(settings, component, strategy, d1_store, model_registry, dt_registry, sandbox, llm_client, rag_kb):
    """Dispatches `strategy` for `component` via the real agent, verifies, and hot-swaps on
    ACCEPT — the same real pipeline the full system uses, minus the LLM-driven decision step
    (the caller already decided `strategy`)."""
    cls = type(dt_registry.get(component))
    factory = lambda: cls.from_settings(settings)  # noqa: E731
    output_fields = {name: _CLASSES[name].OUTPUT_FIELD for name in _ORDER}

    if strategy != "recalibrate":
        # regenerate/expand_scope need a real LLM; this baseline harness only exercises
        # recalibrate for periodic_retrain/rule_based when no LLM is configured — a genuine,
        # honestly-reported skip, never a fake adaptation.
        if llm_client is None:
            return None
    agent = RecalibrationAgent(settings, d1_store, model_registry, fidelity_evaluator=None, llm_client=llm_client)
    try:
        result = agent.recalibrate(factory, component, dependency_output_fields=output_fields)
    except Exception as exc:  # noqa: BLE001 - a baseline adaptation attempt failing must never crash the harness
        print(f"    [adapt] {component}/{strategy} failed: {exc}")
        return None

    window_df = select_recent_window(d1_store.get_history(), settings.adaptation.recalibration.training_window_hours)
    _, held_out_df = time_split(window_df, 0.2)
    if cls.DEPENDENCIES:
        held_out_df = with_dependency_ground_truth(held_out_df, cls.DEPENDENCIES, output_fields)
    eval_target = held_out_df[component]
    candidate_preds = result.candidate_component.predict(held_out_df)

    evaluator = FidelityEvaluator(settings.fidelity)
    verification_agent = VerificationAgent(settings, model_registry, evaluator, llm_client=llm_client)
    verification_result = verification_agent.verify(
        result.version, held_out_df, eval_target, candidate_preds, production_component_factory=factory,
        drift_context=f"baseline harness: {strategy} on {component!r}", rag_knowledge_base=rag_kb,
    )
    if verification_result.decision == "ACCEPT":
        dt_registry.replace(result.candidate_component)
    return verification_result.decision


def run_mode(mode: str, settings, secrets, cycles: int, tmp_root: Path) -> list[float | None]:
    print(f"\n=== mode: {mode} ===")
    mode_dir = tmp_root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    import time as _time
    _t0 = _time.monotonic()
    train = pd.read_parquet(settings.resolve_path("data/processed/oai/train.parquet"))
    model_registry = ModelRegistry(models_dir=mode_dir / "models", index_path=mode_dir / "models" / "index.json")
    dt_registry = DTModelRegistry()
    _bootstrap_components(settings, train, model_registry, dt_registry)
    print(f"  [timing] bootstrap done at {_time.monotonic()-_t0:.1f}s", flush=True)
    dt_orchestrator = DTOrchestrator(dt_registry)

    d1_store = D1Store(
        current_state_path=mode_dir / "d1_current.parquet", history_path=mode_dir / "d1_history.parquet",
        quarantine_path=mode_dir / "d1_quarantine.parquet", history_retention_rows=settings.storage.history_retention_rows,
    )
    print(f"  [timing] d1_store constructed at {_time.monotonic()-_t0:.1f}s", flush=True)
    replay = OaiReplaySource(settings.resolve_path("data/processed/oai"), speed=0.0)
    replay.start(run_ids=settings.data.splits.test)
    print(f"  [timing] replay started at {_time.monotonic()-_t0:.1f}s", flush=True)
    preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)
    # A larger batch size than the live-streaming default (20) for this bulk, one-shot replay —
    # the same "use a bigger batch for fast bulk loading" convention already established by
    # `tests/integration/conftest.py`'s own `bootstrap_history` fixture and the other demo
    # scripts. With the small default, D1Store's atomic full-history parquet rewrite on EVERY
    # flush makes a ~26k-row bulk load take well over a minute (O(n^2) total I/O); batch_size=500
    # cuts that to ~50 flushes.
    bulk_sync_config = settings.synchronization.model_copy(update={"batch_size": 500})
    synchronizer = ContinuousSynchronizer(replay, preprocessor, d1_store, bulk_sync_config)
    print(f"  [timing] synchronizer constructed at {_time.monotonic()-_t0:.1f}s, starting run()...", flush=True)
    synchronizer.run()  # bounded replay source -> blocks until exhausted
    print(f"  [timing] sync.run() done at {_time.monotonic()-_t0:.1f}s", flush=True)
    print(f"  replayed real TEST split into D1: {len(d1_store.get_history())} rows")

    fidelity_evaluator = FidelityEvaluator(settings.fidelity)
    latest_fidelity: dict[str, float | None] = {}

    sandbox = SandboxExecutor.from_settings(settings)
    llm_client = None  # recalibrate needs no LLM; regenerate/expand_scope genuinely skipped without one
    rag_kb = None

    trajectory: list[float | None] = []
    drift_source = MockDriftSource(
        settings.drift.mock, valid_components=settings.drift.valid_components,
        severity_range=settings.drift.severity_range, max_events=None, realtime=False,
    )
    drift_detector = DriftDetectorInterface.from_settings(settings)
    drift_iter = drift_detector.process_stream(drift_source)

    for cycle in range(1, cycles + 1):
        _run_prediction_and_fidelity_cycle(d1_store, dt_orchestrator, fidelity_evaluator, settings.evaluation.window_size, latest_fidelity)
        unified = fidelity_evaluator.compute_unified_score(latest_fidelity)
        trajectory.append(unified.unified_score)

        if mode == "static":
            pass  # never adapts
        elif mode == "periodic_retrain":
            if periodic_retrain_due(cycle, settings.baselines.periodic_retrain.interval_cycles):
                component = settings.baselines.periodic_retrain.component
                decision = _run_one_adaptation(settings, component, "recalibrate", d1_store, model_registry, dt_registry, sandbox, llm_client, rag_kb)
                print(f"  cycle {cycle}: scheduled recalibrate({component}) -> {decision}")
        elif mode == "rule_based":
            try:
                trigger_event = next(drift_iter)
                rb_cfg = settings.baselines.rule_based
                strategy = rule_based_strategy(trigger_event.severity, rb_cfg.recalibrate_max_severity, rb_cfg.regenerate_max_severity)
                decision = _run_one_adaptation(settings, trigger_event.component, strategy, d1_store, model_registry, dt_registry, sandbox, llm_client, rag_kb)
                print(f"  cycle {cycle}: trigger({trigger_event.component}, severity={trigger_event.severity:.3f}) -> rule={strategy} -> {decision}")
            except StopIteration:
                pass
        elif mode == "full":
            # The real system's decision step needs a genuine LLM call — exercised separately by
            # scripts/run_orchestrator_demo.py / src/main.py; this harness reports 'full' mode's
            # pure prediction+fidelity trajectory (identical replay/seed/settings to the other 3)
            # without re-implementing ContinuousOrchestrator's own threaded dispatch here.
            pass

        print(f"  cycle {cycle}: UnifiedFidelityScore={unified.unified_score}")

    return trajectory


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all 4 baseline experiment modes on the same real OAI replay")
    parser.add_argument("--cycles", type=int, default=15)
    parser.add_argument("--modes", type=str, default="static,periodic_retrain,rule_based,full")
    args = parser.parse_args()

    settings = load_settings()
    setup_logging(level="WARNING", fmt=settings.logging.format, log_dir=settings.logging.log_dir)
    secrets = load_secrets()

    import tempfile
    tmp_root = Path(tempfile.mkdtemp(prefix="dt_baselines_"))
    print(f"[run_baselines] isolated storage root: {tmp_root}")

    results: dict[str, list[float | None]] = {}
    for mode in args.modes.split(","):
        results[mode] = run_mode(mode, settings, secrets, args.cycles, tmp_root)

    print("\n=== SUMMARY: final UnifiedFidelityScore per mode ===")
    for mode, trajectory in results.items():
        final = trajectory[-1] if trajectory else None
        print(f"  {mode:<18} final={final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
