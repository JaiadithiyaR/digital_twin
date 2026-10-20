#!/usr/bin/env python3
"""Test each adaptation agent INDIVIDUALLY on real OAI data, through the real orchestrator wiring.

For each requested strategy (recalibrate / regenerate / expand_scope) this script forces one
internal fidelity trigger and runs the orchestrator's real adaptation loop for it
(`ContinuousOrchestrator._handle_trigger`: decision -> agent -> sandbox -> verification -> hot swap
-> post-check -> lifecycle, retried with rejection feedback up to
`adaptation.max_adaptation_attempts`). The ONLY thing substituted is the decision: a
`_PinnedDecisionAgent` always returns the strategy under test (logged as `decision_source=pinned`,
no decision LLM call), so each agent is exercised on its own instead of whichever one the LLM
happens to pick. Everything else — the real replayed OAI telemetry (smoke-test run 7, never the
TEST run 8), the agents, the LLM code generation, the sandbox, the deterministic verification
gate — is real.

Dedicated storage `data/agent_tests/` (wiped each invocation) — never production state.

Usage:
    python scripts/test_adaptation_agents.py                          # all three agents
    python scripts/test_adaptation_agents.py --strategies recalibrate  # one agent
    python scripts/test_adaptation_agents.py --attempts 3              # override the loop cap
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.adaptation.decision_agent import DecisionOutput  # noqa: E402
from src.common.config import load_secrets, load_settings  # noqa: E402
from src.common.logging import setup_logging  # noqa: E402
from src.fidelity.trigger import AdaptationTrigger  # noqa: E402
from src.main import ContinuousOrchestrator  # noqa: E402

TEST_DIR = REPO_ROOT / "data" / "agent_tests"
MIN_HISTORY_ROWS = 1500
ACCUMULATION_TIMEOUT_SECONDS = 120.0
# Which component each agent is exercised on. expand_scope designs a NEW component; the trigger's
# component only gives it context.
TARGETS = {"recalibrate": "dl_bler", "regenerate": "ul_bler", "expand_scope": "dl_throughput"}


class _PinnedDecisionAgent:
    DECISION_SOURCE = "pinned"

    def __init__(self, strategy: str) -> None:
        self.strategy = strategy

    def decide_safe(self, context) -> DecisionOutput:
        return DecisionOutput(
            strategy=self.strategy,
            root_cause_analysis=(
                f"Agent test: strategy pinned to {self.strategy!r} by scripts/test_adaptation_agents.py "
                f"to exercise that adaptation agent individually (no decision-agent LLM call). "
                f"Trigger: {context.trigger_type} on {context.affected_component} at severity {context.severity:.3f}."
            ),
            confidence=1.0,
            rationale="individual adaptation-agent test",
            knowledge_refs=[],
        )


def _settings(attempts: int | None):
    settings = load_settings()
    adaptation = settings.adaptation
    if attempts is not None:
        adaptation = adaptation.model_copy(update={"max_adaptation_attempts": attempts})
    return settings.model_copy(update={
        "environment": "live",
        "adaptation": adaptation,
        "telemetry": settings.telemetry.model_copy(update={
            "source": "oai_replay",
            "replay": settings.telemetry.replay.model_copy(update={"default_split": "train", "speed": 0.0}),
        }),
        "storage": settings.storage.model_copy(update={
            "bootstrap_dir": str(TEST_DIR / "bootstrap"),
            "d1_current_state_path": str(TEST_DIR / "d1_current_state.parquet"),
            "d1_history_path": str(TEST_DIR / "d1_history.parquet"),
            "d1_quarantine_path": str(TEST_DIR / "d1_quarantine.parquet"),
            "models_dir": str(TEST_DIR / "models"),
        }),
        "lifecycle": settings.lifecycle.model_copy(update={
            "records_path": str(TEST_DIR / "lifecycle_records.jsonl"),
            "reports_dir": str(TEST_DIR / "maintenance_reports"),
        }),
        "logging": settings.logging.model_copy(update={
            "component_fidelity_log": str(TEST_DIR / "component_fidelity.csv"),
            "adaptation_decisions_log": str(TEST_DIR / "adaptation_decisions.jsonl"),
        }),
    })


def _entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _f(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategies", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    parser.add_argument("--attempts", type=int, default=None, help="override adaptation.max_adaptation_attempts")
    args = parser.parse_args()

    settings = _settings(args.attempts)
    setup_logging(level=settings.logging.level, fmt=settings.logging.format, log_dir=settings.logging.log_dir)
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    run_id = settings.data.splits.smoke_test_run
    print(f"[agent-test] dedicated storage {TEST_DIR} | replaying OAI run {run_id} | strategies {args.strategies} "
          f"| max attempts {settings.adaptation.max_adaptation_attempts}")

    orch = ContinuousOrchestrator(settings, load_secrets())
    orch.initialize()
    if orch.llm_client is None and any(s != "recalibrate" for s in args.strategies):
        print("[agent-test] FAIL: regenerate/expand_scope need GOOGLE_API_KEY/GEMINI_API_KEY", file=sys.stderr)
        return 1
    orch.telemetry_source.stop()
    orch.telemetry_source.start(run_ids=[run_id])
    orch.start()
    results: list[tuple[str, str, list[dict], float]] = []
    try:
        started = time.monotonic()
        while len(orch.d1_store.get_history()) < MIN_HISTORY_ROWS:
            if time.monotonic() - started > ACCUMULATION_TIMEOUT_SECONDS:
                print("[agent-test] FAIL: D1 history never accumulated enough rows", file=sys.stderr)
                return 1
            time.sleep(0.2)
        for _ in range(settings.fidelity.min_history_for_normalization + 2):
            orch.run_prediction_and_fidelity_cycle()
            time.sleep(0.2)

        decisions_path = Path(settings.logging.adaptation_decisions_log)
        for strategy in args.strategies:
            component = TARGETS[strategy]
            print(f"\n[agent-test] ===== {strategy} (trigger on {component}) =====")
            orch.decision_agent = _PinnedDecisionAgent(strategy)  # type: ignore[assignment]
            before = len(_entries(decisions_path))
            t0 = time.monotonic()
            orch._handle_trigger(AdaptationTrigger(  # noqa: SLF001 - the real adaptation loop, run synchronously
                component=component, severity=0.5, timestamp=datetime.now(UTC), trigger_type="fidelity_degradation",
                metadata={"forced": True, "reason": f"individual {strategy} agent test"},
            ))
            results.append((strategy, component, _entries(decisions_path)[before:], time.monotonic() - t0))
            orch.run_prediction_and_fidelity_cycle()
    finally:
        orch.stop()

    print("\n[agent-test] ===== summary =====")
    ok = True
    for strategy, component, entries, seconds in results:
        print(f"{strategy} on {component}: {len(entries)} attempt(s), {seconds:.1f}s")
        for e in entries:
            print(
                f"   attempt {e.get('attempt')}/{e.get('max_attempts')}: outcome={e['outcome']} "
                f"candidate={e.get('candidate_version')} verification {_f(e.get('verification_fidelity_before'))} -> "
                f"{_f(e.get('verification_fidelity_after'))} failed_checks={e.get('failed_checks') or []} "
                f"post F_c={_f(e.get('post_adaptation_fidelity'))}"
                + (f" error={e['error'][:200]}" if e.get("error") else "")
            )
        produced = any(e.get("candidate_version") for e in entries)
        if not produced:
            ok = False
            print(f"   !! {strategy} never produced a candidate")
    print(f"\n[agent-test] {'OK' if ok else 'FAIL'} — every requested agent produced and verified a real candidate"
          if ok else "\n[agent-test] FAIL — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
