#!/usr/bin/env python3
"""baselines/run_all.py — phase13 §4: runs every baseline experiment mode over the real OAI TEST
split (run 8, speed=0), writing the real, honestly-reported phase13 §5/§6 outputs under
`baselines/results/<mode>/`. `compare.py` reads what this script writes — it never re-runs
anything itself.

Fairness (phase13 §4), enforced here:
  - identical bootstrap v2 (residual, CV-tuned) components — one shared `_lib.bootstrap_components()`
  - identical real TEST replay (run 8, speed=0, same chronological order) — `_lib.replay_test_split()`
  - identical fidelity config (`settings.fidelity`, never mode-specific)
  - identical seeds (`settings.baselines.seeds` / each mode's own config file)
  - every mode run starts from a FRESH ModelRegistry/D1Store/DTModelRegistry — nothing reused
    across modes or across seeds within one mode.

Usage:
    python baselines/run_all.py --modes static periodic_retrain rule_based full_system
    python baselines/run_all.py --modes static --cycles 5          # fast smoke run
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines import _console, _outputs  # noqa: E402
from baselines._lib import (  # noqa: E402
    CLASSES,
    MODES,
    ORDER,
    AdaptationOutcome,
    bootstrap_components,
    build_static_prefix,
    replay_test_split,
    run_one_adaptation,
    run_prediction_and_fidelity_cycle,
)
from src.adaptation.baseline_policies import periodic_retrain_due, rule_based_strategy  # noqa: E402
from src.adaptation.decision_agent import DecisionAgent  # noqa: E402
from src.adaptation.decision_context import PreviousOutcome, build_decision_context, retrieve_rag_context, summarize_network_state  # noqa: E402
from src.common.config import Settings, load_secrets, load_settings  # noqa: E402
from src.common.logging import setup_logging  # noqa: E402
from src.dt_models.model_registry import DTModelRegistry  # noqa: E402
from src.dt_models.orchestrator import DTOrchestrator  # noqa: E402
from src.drift.drift_detector import DriftDetectorInterface  # noqa: E402
from src.drift.mock_drift_source import MockDriftSource  # noqa: E402
from src.fidelity.evaluator import FidelityEvaluator  # noqa: E402
from src.fidelity.trigger import AdaptationTrigger, trigger_from_drift_event  # noqa: E402
from src.llm.cost_tracking import BudgetExceededError, BudgetGuard, MeteredGoogleClient, RoleTaggedClient  # noqa: E402
from src.llm.google_client import GoogleClient  # noqa: E402
from src.rag.rag_kb import RagKnowledgeBase  # noqa: E402
from src.registry.model_registry import ModelRegistry  # noqa: E402
from src.sandbox.executor import SandboxExecutor  # noqa: E402

logger = logging.getLogger(__name__)
BASELINES_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASELINES_DIR / "results"
CONFIGS_DIR = BASELINES_DIR / "configs"

_CONFIG_FILENAMES = {"static": "static.yaml", "periodic_retrain": "periodic.yaml", "rule_based": "rule_based.yaml", "full_system": "full_system.yaml"}


def _load_mode_config(mode: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIGS_DIR / _CONFIG_FILENAMES[mode]).read_text(encoding="utf-8"))


def _run_dir(mode: str, seed: int, n_seeds: int) -> Path:
    base = RESULTS_DIR / mode
    return base / f"seed_{seed}" if n_seeds > 1 else base


def preflight_smoke_test(settings: Settings, secrets, static_prefix: str) -> dict[str, Any]:
    """Phase13 §3.8: before any LLM-dependent baseline experiment, exercise every LLM role
    (decision/codegen/explanation) once against run 7 (NEVER run 8 — that's reserved for TEST),
    with caching ON, and print a one-call cost summary. Raises if any call fails — a genuine
    blocker (phase13 §0: "LLM provider unavailable ... STOP and ask"), never silently skipped."""
    print("=== Pre-flight LLM smoke test (run 7, never run 8) ===")
    client = GoogleClient.from_settings(settings, secrets)
    cache_key = f"preflight-{uuid.uuid4().hex[:8]}"
    cache_started = client.start_cache(cache_key, static_prefix, ttl_seconds=settings.llm.caching.ttl_seconds)
    try:
        decision_agent = DecisionAgent(client, settings, static_prefix=static_prefix)
        ctx = build_decision_context(
            trigger=AdaptationTrigger(component="dl_bler", severity=0.4, timestamp=pd.Timestamp.now(tz="UTC"), trigger_type="external_drift"),
            component_fidelity={"dl_bler": 0.5},
            unified_fidelity_score=0.5,
            previous_outcome=PreviousOutcome(),
            history=pd.DataFrame(),
            rag_knowledge_base=None,
        )
        decision = decision_agent.decide(ctx, cache_key=cache_key if cache_started else None)
        print(f"  decision role OK: strategy={decision.strategy!r} confidence={decision.confidence}")

        from pydantic import BaseModel

        class _Ack(BaseModel):
            ok: bool

        codegen_prompt = "Reply with JSON {\"ok\": true} — this is a connectivity smoke test only, write no code."
        sent = codegen_prompt if cache_started else f"{static_prefix}\n\n{codegen_prompt}"
        client.complete_structured(sent, _Ack, effort=settings.llm.effort_by_role.codegen, cache_key=cache_key if cache_started else None)
        print("  codegen role OK")

        explanation_prompt = "Reply with the single word: OK — this is a connectivity smoke test only."
        sent = explanation_prompt if cache_started else f"{static_prefix}\n\n{explanation_prompt}"
        resp = client.complete(sent, effort=settings.llm.effort_by_role.explanation, cache_key=cache_key if cache_started else None)
        print(f"  explanation role OK: {resp.text!r}")

        from src.llm.cost_tracking import estimate_cost_usd

        one_call_cost = estimate_cost_usd(resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.cached_tokens, settings.llm.rates)
        print(
            f"  one-call cost summary (explanation role): input={resp.usage.input_tokens} "
            f"output={resp.usage.output_tokens} cached={resp.usage.cached_tokens} thinking={resp.usage.thinking_tokens} "
            f"=> ${one_call_cost:.6f} (rates from config.llm.rates, best-effort estimate)"
        )

        summary = {"decision_ok": True, "codegen_ok": True, "explanation_ok": True, "model": settings.llm.model, "cache_started": cache_started}
        print(f"  model={settings.llm.model} service_tier={settings.llm.service_tier} caching_active={cache_started}")
        return summary
    finally:
        if cache_started:
            client.end_cache(cache_key)


def run_mode(
    mode: str, mode_cfg: dict[str, Any], seed: int, n_seeds: int, settings: Settings, secrets,
    train_df: pd.DataFrame, static_prefix: str, cycles: int, rag_kb: "RagKnowledgeBase | None",
    *, replay_run_ids: list[int] | None = None, fake_llm_client: Any | None = None,
) -> dict[str, Any]:
    """`replay_run_ids`/`fake_llm_client` are test-only hooks (default `None` — every real CLI
    invocation leaves both unset): `replay_run_ids` lets `tests/e2e/` replay a short slice of run
    7 instead of the real TEST split (run 8); `fake_llm_client` lets it exercise the LLM-using
    code paths (all three roles) with zero real API calls/cost, bypassing real `GoogleClient`/
    `MeteredGoogleClient`/context-cache construction entirely."""
    out_dir = _run_dir(mode, seed, n_seeds)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Running mode={mode!r} seed={seed} -> {out_dir} ===")

    model_registry = ModelRegistry(models_dir=out_dir / "models", index_path=out_dir / "models" / "registry_index.json")
    dt_registry = DTModelRegistry()
    bootstrap_components(settings, train_df, model_registry, dt_registry)
    model_versions = {name: model_registry.get_current_version(name).version_id for name in ORDER}
    dt_orchestrator = DTOrchestrator(dt_registry)

    d1_store = replay_test_split(settings, out_dir, run_ids=replay_run_ids)
    full_history = d1_store.get_history().sort_values("ts_1s", kind="stable").reset_index(drop=True)
    run_ids = sorted(int(r) for r in full_history["run_id"].unique().tolist()) if len(full_history) else []

    fidelity_evaluator = FidelityEvaluator(settings.fidelity)
    sandbox = SandboxExecutor.from_settings(settings)

    uses_llm = bool(mode_cfg.get("uses_llm", False))
    real_client: GoogleClient | None = None
    llm_client_decision: Any = None
    llm_client_codegen: Any = None
    llm_client_explanation: Any = None
    cache_key: str | None = None
    calls_log_path = out_dir / "llm_calls.jsonl"
    budget_guard: BudgetGuard | None = None
    run_label = f"{mode}-seed{seed}"
    if uses_llm and fake_llm_client is not None:
        llm_client_decision = llm_client_codegen = llm_client_explanation = fake_llm_client
    elif uses_llm:
        real_client = GoogleClient.from_settings(settings, secrets)
        budget_guard = BudgetGuard(float(mode_cfg.get("budget_usd", settings.llm.budget_usd)))
        metered = MeteredGoogleClient(
            real_client, mode=mode, run=run_label, budget_guard=budget_guard, calls_log_path=calls_log_path, rates=settings.llm.rates
        )
        # Phase13 §3.5: one shared MeteredGoogleClient (one budget guard, one calls log) per run,
        # wrapped three times with a fixed role baked in — each agent calls `complete`/
        # `complete_structured` exactly as it always has (no agent needs to know `role=` exists),
        # but every entry in llm_calls.jsonl is correctly attributed to decision/codegen/explanation.
        llm_client_decision = RoleTaggedClient(metered, "decision")
        llm_client_codegen = RoleTaggedClient(metered, "codegen")
        llm_client_explanation = RoleTaggedClient(metered, "explanation")
        candidate_key = f"{run_label}-{uuid.uuid4().hex[:8]}"
        cache_started = real_client.start_cache(candidate_key, static_prefix, ttl_seconds=settings.llm.caching.ttl_seconds)
        cache_key = candidate_key if cache_started else None

    decision_agent = DecisionAgent(llm_client_decision, settings, static_prefix=static_prefix) if mode == "full_system" else None

    drift_mock_cfg = settings.drift.mock.model_copy(update={"seed": seed})
    drift_source = MockDriftSource(drift_mock_cfg, settings.drift.valid_components, settings.drift.severity_range, max_events=cycles, realtime=False)
    drift_detector = DriftDetectorInterface(settings.drift.valid_components, settings.drift.severity_range)
    drift_events_iter = iter(drift_source.events())

    timeline_rows: list[dict[str, Any]] = []
    adaptation_events: list[dict[str, Any]] = []
    previous_outcome_by_component: dict[str, PreviousOutcome] = {}
    n_decisions = 0
    n_fallbacks = 0
    incomplete_reason: str | None = None

    chunk_size = max(1, len(full_history) // cycles) if len(full_history) else 0
    trigger_threshold = settings.fidelity.trigger_threshold

    def _dispatch(trigger, strategy: str, decision_meta: dict[str, Any]) -> AdaptationOutcome:
        outcome = run_one_adaptation(
            settings=settings, trigger=trigger, strategy=strategy, decision_meta=decision_meta,
            d1_store=d1_store, model_registry=model_registry, dt_registry=dt_registry, sandbox=sandbox,
            llm_client_codegen=llm_client_codegen, llm_client_explanation=llm_client_explanation,
            rag_kb=rag_kb, latest_fidelity=latest_fidelity_by_component,
            fidelity_evaluator=fidelity_evaluator, static_prefix=static_prefix, cache_key=cache_key,
        )
        outcome.event["cycle"] = cycle
        previous_outcome_by_component[trigger.component] = PreviousOutcome(
            action=strategy, verification_result=outcome.decision,
            fidelity_before=outcome.event.get("fidelity_before"), fidelity_after=outcome.event.get("fidelity_after"),
        )
        return outcome

    latest_fidelity_by_component: dict[str, float | None] = {}
    try:
        for cycle in range(1, cycles + 1):
            end = min(cycle * chunk_size, len(full_history)) if chunk_size else 0
            visible = full_history.iloc[:end]
            ts_label = visible["ts_1s"].iloc[-1] if len(visible) else None
            cycle_result = run_prediction_and_fidelity_cycle(
                cycle, ts_label, visible, dt_orchestrator, fidelity_evaluator, settings.evaluation.window_size, model_versions
            )
            for row in cycle_result.timeline_rows:
                timeline_rows.append(row)
            latest_fidelity_by_component = cycle_result.latest_fidelity
            unified = None
            lowest = None
            if cycle_result.timeline_rows:
                unified = cycle_result.timeline_rows[0].get("F_U")
                defined = {k: v for k, v in latest_fidelity_by_component.items() if v is not None}
                lowest = min(defined, key=lambda c: defined[c]) if defined else None
            if cycle % 10 == 0 or cycle == cycles:
                _console.live_line(run_label, cycle, cycles, unified, lowest)

            raw_event = next(drift_events_iter, None)
            external_trigger = None
            if raw_event is not None:
                validated, _quarantined = drift_detector.process_event(raw_event)
                if validated is not None:
                    external_trigger = trigger_from_drift_event(validated)

            trigger = cycle_result.fidelity_trigger or external_trigger

            if mode == "static":
                if trigger is not None:
                    adaptation_events.append(
                        {
                            "cycle": cycle, "trigger_time": trigger.timestamp.isoformat(), "component": trigger.component,
                            "trigger_type": trigger.trigger_type, "severity": trigger.severity,
                            "decision": {"strategy": None, "source": "static_mode_ignored"},
                            "gate_result": None, "skipped_reason": "static mode: triggers logged, never dispatched",
                        }
                    )
                continue

            if mode == "periodic_retrain":
                if periodic_retrain_due(cycle, mode_cfg.get("interval_cycles", settings.baselines.periodic_retrain.interval_cycles)):
                    for component in mode_cfg.get("components", list(ORDER)):
                        from src.fidelity.trigger import AdaptationTrigger

                        sched_trigger = AdaptationTrigger(
                            component=component, severity=0.0, timestamp=pd.Timestamp.now(tz="UTC"), trigger_type="external_drift"
                        )
                        outcome = _dispatch(sched_trigger, "recalibrate", {"source": "periodic_schedule", "confidence": 1.0})
                        adaptation_events.append(outcome.event)
                continue

            if trigger is None:
                continue

            if mode == "rule_based":
                strategy = rule_based_strategy(
                    trigger.severity, mode_cfg.get("recalibrate_max_severity", 0.3), mode_cfg.get("regenerate_max_severity", 0.7)
                )
                decision_meta = {
                    "source": "rule_based", "confidence": 1.0, "llm_or_fallback": "rule",
                    "root_cause_analysis": (
                        f"Deterministic severity-threshold rule: severity={trigger.severity:.4f} -> {strategy!r} "
                        f"(thresholds: recalibrate<={mode_cfg.get('recalibrate_max_severity', 0.3)}, "
                        f"regenerate<={mode_cfg.get('regenerate_max_severity', 0.7)})."
                    ),
                    "rationale": "No LLM call for the decision itself (phase13 §4: rule_based).",
                    "knowledge_refs": [],
                }
                outcome = _dispatch(trigger, strategy, decision_meta)
                adaptation_events.append(outcome.event)
                continue

            if mode == "full_system":
                network_summary = summarize_network_state(visible)
                rag_context = retrieve_rag_context(rag_kb, trigger.component, trigger.trigger_type)
                ctx = build_decision_context(
                    trigger=trigger, component_fidelity=dict(latest_fidelity_by_component), unified_fidelity_score=unified,
                    previous_outcome=previous_outcome_by_component.get(trigger.component, PreviousOutcome()),
                    history=visible, rag_knowledge_base=rag_kb,
                )
                try:
                    decision = decision_agent.decide_safe(ctx, cache_key=cache_key)
                except BudgetExceededError as exc:
                    incomplete_reason = f"budget guard tripped during decision call: {exc}"
                    break
                n_decisions += 1
                llm_used = decision.confidence > 0.0 or "fallback" not in decision.root_cause_analysis.lower()
                if "infrastructure failure" in decision.root_cause_analysis:
                    n_fallbacks += 1
                decision_meta = {
                    "source": "full_system", "confidence": decision.confidence,
                    "llm_or_fallback": "fallback" if "infrastructure failure" in decision.root_cause_analysis else "llm",
                    "root_cause_analysis": decision.root_cause_analysis, "rationale": decision.rationale,
                    "knowledge_refs": decision.knowledge_refs,
                }
                try:
                    outcome = _dispatch(trigger, decision.strategy, decision_meta)
                except BudgetExceededError as exc:
                    incomplete_reason = f"budget guard tripped during {decision.strategy} dispatch: {exc}"
                    break
                adaptation_events.append(outcome.event)
                continue
    finally:
        if real_client is not None and cache_key is not None:
            real_client.end_cache(cache_key)

    for ev in adaptation_events:
        if ev.get("gate_result") in ("ACCEPT", "REJECT"):
            _console.adaptation_event_block(run_label, ev)

    timeline_df = pd.DataFrame(timeline_rows) if timeline_rows else pd.DataFrame(columns=_outputs.TIMELINE_COLUMNS)
    _outputs.write_metrics_timeline_csv(out_dir / "metrics_timeline.csv", timeline_rows)
    for ev in adaptation_events:
        _outputs.append_jsonl(out_dir / "adaptation_events.jsonl", ev)
    _outputs.write_report_md(
        out_dir / "report.md", mode=mode, run_ids=run_ids, seeds=[seed], model_versions=model_versions,
        fidelity_config=settings.fidelity.model_dump(), timeline_df=timeline_df, events=adaptation_events,
    )

    graphs_dir = out_dir / "graphs"
    if len(timeline_df):
        for component in ORDER:
            _outputs.plot_actual_vs_predicted(component, timeline_df, graphs_dir)
            _outputs.plot_fidelity_component(component, timeline_df, adaptation_events, trigger_threshold, graphs_dir)
        _outputs.plot_unified_fidelity(timeline_df, adaptation_events, trigger_threshold, graphs_dir)

    n_accepted = sum(1 for e in adaptation_events if e.get("gate_result") == "ACCEPT")
    n_rejected = sum(1 for e in adaptation_events if e.get("gate_result") == "REJECT")
    fallback_rate = (n_fallbacks / n_decisions) if n_decisions else 0.0
    invalid = mode == "full_system" and fallback_rate > mode_cfg.get("fallback_rate_invalid_threshold", 0.1)

    summary = {
        "mode": mode, "seed": seed, "cycles": cycles, "run_ids": run_ids,
        "n_triggers": sum(1 for e in adaptation_events if e.get("component")),
        "n_adaptations": len([e for e in adaptation_events if e.get("gate_result") in ("ACCEPT", "REJECT")]),
        "n_accepted": n_accepted, "n_rejected": n_rejected,
        "n_decisions": n_decisions, "n_fallbacks": n_fallbacks, "fallback_rate": fallback_rate,
        "invalid": invalid, "incomplete_reason": incomplete_reason,
        "mean_unified_fidelity": float(timeline_df.drop_duplicates("cycle")["F_U"].dropna().mean()) if len(timeline_df) and timeline_df["F_U"].notna().any() else None,
        "total_duration_seconds": float(sum(e.get("duration_seconds", 0.0) for e in adaptation_events)),
        "budget_spent_usd": budget_guard.spent_usd if budget_guard is not None else 0.0,
        "out_dir": str(out_dir),
    }
    (out_dir / "run_summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")
    print(f"=== mode={mode!r} seed={seed} complete: {n_accepted} accepted, {n_rejected} rejected, mean F_U={summary['mean_unified_fidelity']} ===")
    if incomplete_reason:
        print(f"!!! RUN INCOMPLETE: {incomplete_reason}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--skip-preflight", action="store_true", help="skip the §3.8 LLM smoke test (never use for a real reported run)")
    args = parser.parse_args()

    settings = load_settings()
    secrets = load_secrets()
    setup_logging(level=settings.logging.level, fmt=settings.logging.format, log_dir=settings.logging.log_dir)

    train_df = pd.read_parquet(settings.resolve_path("data/processed/oai/train.parquet"))
    static_prefix = build_static_prefix(settings)
    rag_kb = RagKnowledgeBase.from_settings(settings)

    any_llm_mode = any(_load_mode_config(m).get("uses_llm", False) for m in args.modes)
    if any_llm_mode and not args.skip_preflight:
        preflight_smoke_test(settings, secrets, static_prefix)

    all_summaries: list[dict[str, Any]] = []
    for mode in args.modes:
        mode_cfg = _load_mode_config(mode)
        cycles = args.cycles if args.cycles is not None else settings.baselines.cycles
        seeds = args.seeds if args.seeds is not None else mode_cfg.get("seeds", [1])
        for seed in seeds:
            summary = run_mode(mode, mode_cfg, seed, len(seeds), settings, secrets, train_df, static_prefix, cycles, rag_kb)
            all_summaries.append(summary)
            if summary.get("incomplete_reason"):
                print(f"Stopping further seeds/modes after an incomplete run ({mode}/seed={seed}) per phase13 §3.6.")
                break

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_summaries).to_csv(RESULTS_DIR / "run_all_summary.csv", index=False)
    print(f"\nAll requested modes complete. Summary: {RESULTS_DIR / 'run_all_summary.csv'}")


if __name__ == "__main__":
    main()
