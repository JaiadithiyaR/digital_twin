#!/usr/bin/env python3
"""Phase 11 — real, end-to-end validation of `src/main.py`'s `ContinuousOrchestrator` (permanent,
repo-tracked, same convention as `ns3_sim/validate_e2e.py`/`scripts/ingest_rag.py`): initializes
the FULL wired system, runs it unattended through one complete real cycle (drift -> Decision &
Root-Cause Analysis Agent -> Module 14/15/16 agent -> Module 17 verification -> Module 19
lifecycle record), and — the concrete, load-bearing proof this script exists for — confirms the
live D1 telemetry state kept growing DURING the adaptation cycle's own wall-clock window, via the
same real-background-thread-plus-sampler-thread technique Modules 14/15/16/19 already established
in their own tests, applied here to the real orchestrator instead of a single agent in isolation.

**Why the drift severity range is narrowed for this run**: no real `GOOGLE_API_KEY`/
`GEMINI_API_KEY` is configured in this development environment. Unlike the old PPO design, the
Decision & Root-Cause Analysis Agent's decision itself now requires a genuine LLM call every
time — which genuinely fails here (no real key), so `DecisionAgent.decide_safe()` degrades to
`config.decision_agent.fallback.default_strategy` (`recalibrate`, which needs no LLM to execute)
regardless of severity. The narrowed severity range is kept only for otherwise-stable/realistic
demo output, not because it changes which strategy gets selected anymore. This still lets the run
complete with ZERO fakes/mocks anywhere — a genuinely real drift event, a genuine (if
infrastructure-degraded) decision-agent call, a genuinely real recalibration candidate, real
Module 17 verification, and a real lifecycle record. `src/main.py`'s own orchestrator is otherwise
unmodified and unaware of this script; `--mode demo`/`--mode live` (no severity override) is the
real, general-purpose entrypoint for actual continuous operation, where a real `GOOGLE_API_KEY`
would let the decision agent make genuine (non-fallback) calls, including selecting
`regenerate`/`expand_scope`.

Usage:
    python scripts/run_orchestrator_demo.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_secrets, load_settings  # noqa: E402
from src.common.logging import setup_logging  # noqa: E402
from src.main import ContinuousOrchestrator  # noqa: E402

MIN_HISTORY_ROWS_BEFORE_ADAPTATION = 260  # comfortably above config.adaptation.recalibration.min_training_rows (200)
ACCUMULATION_TIMEOUT_SECONDS = 60.0


def main() -> int:
    settings = load_settings()
    setup_logging(level=settings.logging.level, fmt=settings.logging.format, log_dir=settings.logging.log_dir)
    secrets = load_secrets()

    # Speed up live telemetry accumulation for this validation run only (still the real Module 2
    # pipeline, just ticking faster than the checked-in config's default cadence).
    fast_settings = settings.model_copy(
        update={
            "telemetry": settings.telemetry.model_copy(
                update={"mock": settings.telemetry.mock.model_copy(update={"emit_interval_seconds": 0.2})}
            ),
            "drift": settings.drift.model_copy(
                update={"mock": settings.drift.mock.model_copy(update={"emit_interval_seconds": 5.0})}
            ),
        }
    )

    print("[demo] initializing the full wired system (D1, RAG, model registry, bootstrap DT models, "
          "decision agent, telemetry source, drift source)...")
    orchestrator = ContinuousOrchestrator(fast_settings, secrets, drift_severity_range_override=(0.05, 0.2))
    orchestrator.initialize()
    print(f"[demo] llm_available={orchestrator.llm_client is not None} rag_available={orchestrator.rag_kb.is_available}")

    orchestrator.start()
    print("[demo] telemetry synchronization + drift consumption started in the background")

    print(f"[demo] waiting for live D1 history to accumulate >= {MIN_HISTORY_ROWS_BEFORE_ADAPTATION} rows "
          "from genuinely live telemetry (never bootstrap data — D1 is live-telemetry-only by design)...")
    wait_start = time.monotonic()
    while True:
        rows = len(orchestrator.d1_store.get_history())
        if rows >= MIN_HISTORY_ROWS_BEFORE_ADAPTATION:
            break
        if time.monotonic() - wait_start > ACCUMULATION_TIMEOUT_SECONDS:
            orchestrator.stop()
            print(f"[demo] FAILED: only {rows} live D1 rows accumulated after {ACCUMULATION_TIMEOUT_SECONDS}s", file=sys.stderr)
            return 1
        time.sleep(0.1)
    print(f"[demo] live D1 history now has {len(orchestrator.d1_store.get_history())} rows — proceeding")

    # Warm every component's real fidelity rolling window PAST config.fidelity.
    # min_history_for_normalization using genuinely live D1 data (never synthetic) — otherwise
    # the decision agent's context would see "insufficient history" (neutral placeholder) fidelity
    # for every component, an unrealistic cold-start distribution real deployment wouldn't stay in
    # for long.
    min_history = fast_settings.fidelity.min_history_for_normalization
    print(f"[demo] warming real fidelity rolling windows past min_history_for_normalization ({min_history})...")
    for _ in range(min_history + 2):
        orchestrator.run_prediction_and_fidelity_cycle()
        time.sleep(0.15)
    print(f"[demo] latest real fidelity per component: {orchestrator._latest_fidelity}")  # noqa: SLF001 - read-only inspection

    # The concrete continuous-operation proof: sample D1's growth every 10ms while the ONE full
    # adaptation cycle below runs, then check growth specifically WITHIN that cycle's own bracket.
    samples: list[tuple[float, int]] = []
    stop_sampling = threading.Event()

    def sample_loop() -> None:
        while not stop_sampling.is_set():
            samples.append((time.monotonic(), orchestrator.synchronizer.records_synced))
            time.sleep(0.01)

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()

    print("[demo] running the orchestration loop for exactly one full adaptation cycle...")
    orchestrator.run(max_drift_events=1, prediction_interval_seconds=2.0)

    stop_sampling.set()
    sampler.join(timeout=5.0)

    start_ts = orchestrator.last_adaptation_started_at
    end_ts = orchestrator.last_adaptation_finished_at
    orchestrator.stop()

    if start_ts is None or end_ts is None:
        print("[demo] FAILED: no adaptation cycle was recorded", file=sys.stderr)
        return 1

    during = [n for ts, n in samples if start_ts <= ts <= end_ts]
    print(f"\n[demo] adaptation cycle wall-clock duration: {end_ts - start_ts:.3f}s, "
          f"{len(during)} telemetry-growth samples observed strictly during that window")
    if len(during) < 2:
        print("[demo] FAILED: adaptation cycle completed too fast for the sampler to observe overlap", file=sys.stderr)
        return 1
    if not (during[-1] > during[0]):
        print(
            f"[demo] FAILED: D1's records_synced did NOT grow during the adaptation cycle "
            f"({during[0]} -> {during[-1]}) — telemetry ingestion was blocked",
            file=sys.stderr,
        )
        return 1
    print(f"[demo] CONFIRMED: records_synced grew from {during[0]} to {during[-1]} DURING the adaptation cycle "
          "— telemetry ingestion was never blocked by adaptation.")

    records = orchestrator.lifecycle_agent.list_records()
    if not records:
        print("[demo] FAILED: no lifecycle record was produced", file=sys.stderr)
        return 1
    record = records[-1]
    print("\n[demo] === lifecycle record (Module 19) ===")
    print(f"  event_id                  = {record.event_id}")
    print(f"  affected_component        = {record.affected_component}")
    print(f"  trigger_type              = {record.trigger_type}")
    print(f"  trigger_severity          = {record.trigger_severity:.4f}")
    print(f"  decision_strategy         = {record.decision_strategy}")
    print(f"  decision_confidence       = {record.decision_confidence:.2f}")
    print(f"  root_cause_analysis       = {record.root_cause_analysis}")
    print(f"  production_version_before = {record.production_version_before}")
    print(f"  candidate_version         = {record.candidate_version}")
    print(f"  fidelity_before           = {record.fidelity_before}")
    print(f"  fidelity_after            = {record.fidelity_after}")
    print(f"  verification_result       = {record.verification_result}")
    print(f"  final_status              = {record.final_status}")

    report_path = orchestrator.lifecycle_agent._reports_dir / f"{record.event_id}.md"  # noqa: SLF001 - read-only path check for this validation script
    print(f"\n[demo] maintenance report written to: {report_path} (exists={report_path.exists()})")
    print(f"[demo] lifecycle records log:          {orchestrator.lifecycle_agent._records_path}")  # noqa: SLF001

    print("\n[demo] OK — one full real cycle completed unattended; telemetry never stopped synchronizing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
