#!/usr/bin/env python3
"""baselines/run_all.py — the 4-step ablation ladder over the real OAI TEST split (run 8):

    step 0  static            bootstrap models, never updated; triggers logged only
    step 1  periodic_retrain  all 7 components recalibrated every 10 min of data, deployed blind
    step 2  rule_based        per-component fidelity trigger -> severity rule -> agents -> gate
    step 3  full_system       per-component fidelity trigger -> LLM+RAG decision agent -> agents -> gate

Every mode is scored on the SAME fixed 60 s windows of replayed data, with the same bootstrap,
the same fidelity config and per-component policy, and no look-ahead (an adaptation started after
window k only sees windows 1..k). rule_based and full_system use the live system's retry loop.
Writes per-mode outputs under `baselines/results/<mode>/` (or `.../seed_N/`); `stats.py` and
`compare.py` read them and never re-run anything.

Usage:
    python baselines/run_all.py                                   # all 4 modes, full run 8
    python baselines/run_all.py --modes static periodic_retrain --skip-preflight
    python baselines/run_all.py --max-windows 5 --skip-preflight  # fast smoke run
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines import _console, _outputs
from baselines._lib import (
    MODES,
    ORDER,
    CutoffD1View,
    bootstrap_components,
    build_static_prefix,
    replay_test_split,
    run_one_adaptation,
    score_window,
    window_edges,
)
from src.adaptation.baseline_policies import rule_based_strategy
from src.adaptation.decision_agent import DecisionAgent
from src.adaptation.decision_context import PreviousOutcome, build_decision_context
from src.common.config import Settings, load_secrets, load_settings
from src.common.logging import setup_logging
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator
from src.fidelity.component_policy import ComponentAdaptationPolicy
from src.fidelity.evaluator import FidelityEvaluator
from src.fidelity.trigger import AdaptationTrigger
from src.llm.cost_tracking import (
    BudgetExceededError,
    BudgetGuard,
    MeteredGoogleClient,
    RoleTaggedClient,
)
from src.llm.google_client import GoogleClient
from src.rag.rag_kb import RagKnowledgeBase
from src.registry.model_registry import ModelRegistry
from src.sandbox.executor import SandboxExecutor

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
    """Phase13 §3.8: exercise every LLM role (decision/codegen/explanation) once before any
    LLM-dependent run. Raises if any call fails — a genuine blocker, never silently skipped."""
    print("=== Pre-flight LLM smoke test (no run-8 data involved) ===")
    client = GoogleClient.from_settings(settings, secrets)
    cache_key = f"preflight-{uuid.uuid4().hex[:8]}"
    cache_started = client.start_cache(cache_key, static_prefix, ttl_seconds=settings.llm.caching.ttl_seconds)
    try:
        decision_agent = DecisionAgent(client, settings, static_prefix=static_prefix)
        ctx = build_decision_context(
            trigger=AdaptationTrigger(component="dl_bler", severity=0.4, timestamp=pd.Timestamp.now(tz="UTC"), trigger_type="fidelity_degradation"),
            component_fidelity={"dl_bler": 0.3}, unified_fidelity_score=0.5, previous_outcome=PreviousOutcome(),
            history=pd.DataFrame(), rag_knowledge_base=None,
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
        print(f"  model={settings.llm.model} caching_active={cache_started}")
        return {"decision_ok": True, "codegen_ok": True, "explanation_ok": True, "model": settings.llm.model}
    finally:
        if cache_started:
            client.end_cache(cache_key)


def run_mode(
    mode: str, mode_cfg: dict[str, Any], seed: int, n_seeds: int, settings: Settings, secrets,
    train_df: pd.DataFrame, static_prefix: str, max_windows: int | None, rag_kb: RagKnowledgeBase | None,
    *, replay_run_ids: list[int] | None = None, fake_llm_client: Any | None = None,
) -> dict[str, Any]:
    """`replay_run_ids`/`fake_llm_client` are test-only hooks (`tests/e2e/`): replay run 7 instead
    of run 8, and exercise the LLM-using paths with zero real API calls."""
    out_dir = _run_dir(mode, seed, n_seeds)
    if out_dir.exists():
        # This directory is this harness's own generated output. A previous run's D1/registry/
        # events must never leak into this one (the D1Store and ModelRegistry reload from disk).
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Running mode={mode!r} seed={seed} -> {out_dir} ===", flush=True)

    model_registry = ModelRegistry(models_dir=out_dir / "models", index_path=out_dir / "models" / "registry_index.json")
    dt_registry = DTModelRegistry()
    bootstrap_components(settings, train_df, model_registry, dt_registry)
    model_versions = {name: model_registry.get_current_version(name).version_id for name in ORDER}
    dt_orchestrator = DTOrchestrator(dt_registry)

    d1_store = replay_test_split(settings, out_dir, run_ids=replay_run_ids)
    full_history = d1_store.get_history().sort_values("ts_1s", kind="stable").reset_index(drop=True)
    run_ids = sorted(int(r) for r in full_history["run_id"].unique().tolist()) if len(full_history) else []
    ts = pd.to_datetime(full_history["ts_1s"], utc=True)
    window_seconds = settings.baselines.window_seconds
    edges = window_edges(full_history, window_seconds)
    if max_windows is not None:
        edges = edges[:max_windows]
    d1_view = CutoffD1View(d1_store)

    fidelity_evaluator = FidelityEvaluator(settings.fidelity)
    adaptation_policy = ComponentAdaptationPolicy(settings.fidelity)
    sandbox = SandboxExecutor.from_settings(settings)
    max_attempts = settings.adaptation.max_adaptation_attempts

    uses_llm = bool(mode_cfg.get("uses_llm", False))
    real_client: GoogleClient | None = None
    llm_client_decision = llm_client_codegen = llm_client_explanation = None
    cache_key: str | None = None
    budget_guard: BudgetGuard | None = None
    run_label = f"{mode}-seed{seed}"
    if uses_llm and fake_llm_client is not None:
        llm_client_decision = llm_client_codegen = llm_client_explanation = fake_llm_client
    elif uses_llm:
        real_client = GoogleClient.from_settings(settings, secrets)
        budget_guard = BudgetGuard(float(mode_cfg.get("budget_usd", settings.llm.budget_usd)))
        metered = MeteredGoogleClient(
            real_client, mode=mode, run=run_label, budget_guard=budget_guard, calls_log_path=out_dir / "llm_calls.jsonl", rates=settings.llm.rates
        )
        llm_client_decision = RoleTaggedClient(metered, "decision")
        llm_client_codegen = RoleTaggedClient(metered, "codegen")
        llm_client_explanation = RoleTaggedClient(metered, "explanation")
        candidate_key = f"{run_label}-{uuid.uuid4().hex[:8]}"
        cache_started = real_client.start_cache(candidate_key, static_prefix, ttl_seconds=settings.llm.caching.ttl_seconds)
        cache_key = candidate_key if cache_started else None
    decision_agent = DecisionAgent(llm_client_decision, settings, static_prefix=static_prefix) if mode == "full_system" else None

    timeline_rows: list[dict[str, Any]] = []
    adaptation_events: list[dict[str, Any]] = []
    previous_outcome_by_component: dict[str, PreviousOutcome] = {}
    latest_fidelity: dict[str, float | None] = {}
    n_decisions = n_fallbacks = 0
    incomplete_reason: str | None = None
    periodic_every = max(1, int(mode_cfg.get("interval_seconds", settings.baselines.periodic_retrain.interval_seconds)) // window_seconds)
    periodic_gate = bool(mode_cfg.get("use_verification_gate", settings.baselines.periodic_retrain.use_verification_gate))

    def attempt(trigger: AdaptationTrigger, strategy: str, meta: dict[str, Any], window: int, n: int, feedback: tuple[str, ...], use_gate: bool = True):
        outcome = run_one_adaptation(
            settings=settings, trigger=trigger, strategy=strategy, decision_meta=meta, d1_view=d1_view,
            model_registry=model_registry, dt_registry=dt_registry, sandbox=sandbox,
            llm_client_codegen=llm_client_codegen, llm_client_explanation=llm_client_explanation, rag_kb=rag_kb,
            latest_fidelity=latest_fidelity, fidelity_evaluator=fidelity_evaluator,
            reference_history=train_df, attempt=n, previous_attempts=feedback, use_gate=use_gate,
            static_prefix=static_prefix, cache_key=cache_key,
        )
        outcome.event["window"] = window
        if outcome.event.get("version_after") and outcome.decision in ("ACCEPT", "DEPLOYED_UNGATED") and trigger.component in model_versions:
            model_versions[trigger.component] = outcome.event["version_after"]
        previous_outcome_by_component[trigger.component] = PreviousOutcome(
            action=strategy, verification_result=outcome.decision if outcome.decision in ("ACCEPT", "REJECT") else None,
            fidelity_before=outcome.event.get("fidelity_before"), fidelity_after=outcome.event.get("fidelity_after"),
        )
        adaptation_events.append(outcome.event)
        return outcome

    def adaptation_loop(trigger: AdaptationTrigger, window: int, window_end: pd.Timestamp, window_df: pd.DataFrame, unified: float | None) -> None:
        """The live system's loop: retry with rejection feedback until ACCEPT, a non-retryable
        outcome, or max_attempts. Data time is frozen at `window_end` while it runs."""
        nonlocal n_decisions, n_fallbacks
        feedback: list[str] = []
        for n in range(1, max_attempts + 1):
            if mode == "rule_based":
                strategy = rule_based_strategy(
                    trigger.severity, mode_cfg.get("recalibrate_max_severity", settings.baselines.rule_based.recalibrate_max_severity),
                    mode_cfg.get("regenerate_max_severity", settings.baselines.rule_based.regenerate_max_severity),
                )
                meta = {
                    "source": "rule_based", "confidence": 1.0, "llm_or_fallback": "rule",
                    "root_cause_analysis": f"Deterministic severity rule: severity={trigger.severity:.4f} -> {strategy!r}.",
                    "rationale": "No LLM call for the decision itself (ablation step 2).", "knowledge_refs": [],
                }
            else:
                ctx = build_decision_context(
                    trigger=trigger, component_fidelity=dict(latest_fidelity), unified_fidelity_score=unified,
                    previous_outcome=previous_outcome_by_component.get(trigger.component, PreviousOutcome()),
                    history=d1_view.get_history(), rag_knowledge_base=rag_kb,
                    component_threshold=adaptation_policy.policy_for(trigger.component).threshold,
                    fidelity_at_trigger=latest_fidelity.get(trigger.component),
                    attempt=n, max_attempts=max_attempts, previous_attempts=tuple(feedback),
                )
                decision = decision_agent.decide_safe(ctx, cache_key=cache_key)
                n_decisions += 1
                is_fallback = "infrastructure failure" in decision.root_cause_analysis
                n_fallbacks += int(is_fallback)
                strategy = decision.strategy
                meta = {
                    "source": "full_system", "confidence": decision.confidence, "llm_or_fallback": "fallback" if is_fallback else "llm",
                    "root_cause_analysis": decision.root_cause_analysis, "rationale": decision.rationale,
                    "knowledge_refs": decision.knowledge_refs,
                }
            outcome = attempt(trigger, strategy, meta, window, n, tuple(feedback))
            _console.adaptation_event_block(run_label, outcome.event)
            if outcome.decision == "ACCEPT" or not outcome.retryable:
                break
            if mode == "rule_based" and strategy == "recalibrate":
                break  # the rule would pick the same deterministic recalibration again
            feedback.append(outcome.summary)
        # Read-only re-measurement on the same window and normalization reference (CLAUDE.md §5).
        post = score_window(window, window_end, window_df, dt_orchestrator, fidelity_evaluator, model_versions, None, update_window=False)
        adaptation_events[-1]["post_adaptation_fidelity"] = post.fidelity.get(trigger.component)

    try:
        for w, (start, end) in enumerate(edges, start=1):
            window_df = full_history[(ts >= start) & (ts < end)]
            d1_view.set_cutoff(end)
            res = score_window(w, end, window_df, dt_orchestrator, fidelity_evaluator, model_versions, adaptation_policy)
            timeline_rows.extend(res.rows)
            latest_fidelity.update(res.fidelity)
            if w % 10 == 0 or w == len(edges):
                defined = {k: v for k, v in res.fidelity.items() if v is not None}
                _console.live_line(run_label, w, len(edges), res.unified, min(defined, key=defined.get) if defined else None)

            if mode in ("static", "periodic_retrain"):
                for trigger in res.triggers:  # detected and logged identically, never dispatched
                    adaptation_events.append({
                        "window": w, "trigger_time": trigger.timestamp.isoformat(), "component": trigger.component,
                        "trigger_type": trigger.trigger_type, "severity": trigger.severity,
                        "fidelity_at_trigger": res.fidelity.get(trigger.component),
                        "decision": {"strategy": None, "source": f"{mode}_ignored"}, "gate_result": None,
                        "skipped_reason": f"{mode}: triggers are logged, never dispatched",
                    })
                if mode == "periodic_retrain" and w % periodic_every == 0:
                    for component in mode_cfg.get("components", list(ORDER)):
                        # TriggerType has no "scheduled" value; the event record is relabelled below.
                        sched = AdaptationTrigger(
                            component=component, severity=0.0, timestamp=end.to_pydatetime(), trigger_type="external_drift",
                            metadata={"scheduled": True},
                        )
                        outcome = attempt(sched, "recalibrate", {"source": "periodic_schedule", "confidence": 1.0}, w, 1, (), use_gate=periodic_gate)
                        outcome.event["trigger_type"] = "scheduled"
                        _console.adaptation_event_block(run_label, outcome.event)
                continue

            for trigger in res.triggers:
                adaptation_loop(trigger, w, end, window_df, res.unified)
    except BudgetExceededError as exc:
        incomplete_reason = f"budget guard tripped: {exc}"
    finally:
        if real_client is not None and cache_key is not None:
            real_client.end_cache(cache_key)

    timeline_df = pd.DataFrame(timeline_rows) if timeline_rows else pd.DataFrame(columns=_outputs.TIMELINE_COLUMNS)
    _outputs.write_metrics_timeline_csv(out_dir / "metrics_timeline.csv", timeline_rows)
    for ev in adaptation_events:
        _outputs.append_jsonl(out_dir / "adaptation_events.jsonl", ev)
    _outputs.write_report_md(
        out_dir / "report.md", mode=mode, run_ids=run_ids, seeds=[seed], model_versions=model_versions,
        fidelity_config=settings.fidelity.model_dump(), timeline_df=timeline_df, events=adaptation_events,
    )
    graphs_dir = out_dir / "graphs"
    threshold = settings.fidelity.trigger_threshold
    if len(timeline_df):
        for component in ORDER:
            _outputs.plot_actual_vs_predicted(component, timeline_df, graphs_dir)
            _outputs.plot_fidelity_component(component, timeline_df, adaptation_events, threshold, graphs_dir)
        _outputs.plot_unified_fidelity(timeline_df, adaptation_events, threshold, graphs_dir)

    gated = [e for e in adaptation_events if e.get("gate_result") in ("ACCEPT", "REJECT")]
    fallback_rate = (n_fallbacks / n_decisions) if n_decisions else 0.0
    summary = {
        "mode": mode, "seed": seed, "window_seconds": window_seconds, "n_windows": len(edges), "run_ids": run_ids,
        "n_triggers": sum(1 for e in adaptation_events if e.get("trigger_type") == "fidelity_degradation" and e.get("attempt", 1) == 1),
        "n_attempts": sum(1 for e in adaptation_events if "attempt" in e),
        "n_adaptations": len(gated), "n_accepted": sum(e["gate_result"] == "ACCEPT" for e in gated),
        "n_rejected": sum(e["gate_result"] == "REJECT" for e in gated),
        "n_deployed_ungated": sum(1 for e in adaptation_events if e.get("gate_result") == "DEPLOYED_UNGATED"),
        "n_agent_failures": sum(1 for e in adaptation_events if e.get("error")),
        "n_decisions": n_decisions, "n_fallbacks": n_fallbacks, "fallback_rate": fallback_rate,
        "invalid": mode == "full_system" and fallback_rate > mode_cfg.get("fallback_rate_invalid_threshold", 0.1),
        "incomplete_reason": incomplete_reason,
        "mean_unified_fidelity_runtime": float(timeline_df.drop_duplicates("window")["F_U"].dropna().mean()) if len(timeline_df) and timeline_df["F_U"].notna().any() else None,
        "adaptation_wall_seconds": float(sum(e.get("duration_seconds", 0.0) for e in adaptation_events)),
        "adaptation_cpu_seconds": float(sum(e.get("cpu_seconds", 0.0) for e in adaptation_events)),
        "budget_spent_usd": budget_guard.spent_usd if budget_guard is not None else 0.0,
        "out_dir": out_dir.relative_to(BASELINES_DIR.parent).as_posix() if out_dir.is_relative_to(BASELINES_DIR.parent) else str(out_dir),
    }
    (out_dir / "run_summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")
    print(
        f"=== mode={mode!r} seed={seed} complete: {summary['n_accepted']} accepted, {summary['n_rejected']} rejected, "
        f"{summary['n_deployed_ungated']} deployed ungated over {len(edges)} windows ===", flush=True,
    )
    if incomplete_reason:
        print(f"!!! RUN INCOMPLETE: {incomplete_reason}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--max-windows", type=int, default=None, help="limit the number of 60 s windows (smoke runs)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--skip-preflight", action="store_true", help="skip the §3.8 LLM smoke test (never use for a real reported run)")
    args = parser.parse_args()

    settings = load_settings()
    secrets = load_secrets()
    setup_logging(level=settings.logging.level, fmt=settings.logging.format, log_dir=settings.logging.log_dir)

    train_df = pd.read_parquet(settings.resolve_path("data/processed/oai/train.parquet"))
    static_prefix = build_static_prefix(settings)
    rag_kb = RagKnowledgeBase.from_settings(settings)

    if any(_load_mode_config(m).get("uses_llm", False) for m in args.modes) and not args.skip_preflight:
        preflight_smoke_test(settings, secrets, static_prefix)

    max_windows = args.max_windows if args.max_windows is not None else settings.baselines.max_windows
    all_summaries: list[dict[str, Any]] = []
    for mode in args.modes:
        mode_cfg = _load_mode_config(mode)
        seeds = args.seeds if args.seeds is not None else mode_cfg.get("seeds", [1])
        for seed in seeds:
            summary = run_mode(mode, mode_cfg, seed, len(seeds), settings, secrets, train_df, static_prefix, max_windows, rag_kb)
            all_summaries.append(summary)
            if summary.get("incomplete_reason"):
                print(f"Stopping further seeds/modes after an incomplete run ({mode}/seed={seed}) per phase13 §3.6.")
                break

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "run_all_summary.csv"
    new = pd.DataFrame(all_summaries)
    if summary_path.exists():  # keep other modes' rows when only some modes are re-run
        old = pd.read_csv(summary_path)
        if "window_seconds" in old.columns:
            keys = set(zip(new["mode"], new["seed"]))
            old = old[[(m, s) not in keys for m, s in zip(old["mode"], old["seed"])]]
            new = pd.concat([old, new], ignore_index=True)
    new.to_csv(summary_path, index=False)
    print(f"\nAll requested modes complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
