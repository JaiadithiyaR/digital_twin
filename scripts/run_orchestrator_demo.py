#!/usr/bin/env python3
"""Phase 11 — real, end-to-end validation of `src/main.py`'s `ContinuousOrchestrator` (permanent,
repo-tracked, same convention as `ns3_sim/validate_e2e.py`/`scripts/ingest_rag.py`): initializes
the FULL wired system, runs it unattended through one complete real cycle (internal fidelity
trigger -> Decision & Root-Cause Analysis Agent -> Module 14/15/16 agent -> Module 17
verification -> Module 19 lifecycle record), and — the concrete, load-bearing proof this script
exists for — confirms the live D1 telemetry state kept growing DURING the adaptation cycle's own
wall-clock window, via a sampler thread.

The only trigger source is the internal per-component fidelity policy (external drift is
permanently disabled). If no component's policy fires during warm-up, the script injects one
internal `fidelity_degradation` trigger on the lowest-scoring component, labelled
`metadata.forced=True`, so every run exercises a full adaptation cycle. Without a real
`GOOGLE_API_KEY` the decision degrades to the configured fallback (`recalibrate`), logged.

Usage:
    python scripts/run_orchestrator_demo.py
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_secrets, load_settings  # noqa: E402
from src.common.logging import setup_logging  # noqa: E402
from src.fidelity.trigger import AdaptationTrigger  # noqa: E402
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
        }
    )

    print("[demo] initializing the full wired system (D1, RAG, model registry, bootstrap DT models, "
          "decision agent, telemetry source)...")
    orchestrator = ContinuousOrchestrator(fast_settings, secrets)
    orchestrator.initialize()
    print(f"[demo] llm_available={orchestrator.llm_client is not None} rag_available={orchestrator.rag_kb.is_available}")

    orchestrator.start()
    print("[demo] telemetry synchronization started in the background")

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

    # The only trigger source is the internal per-component fidelity policy. If it hasn't fired
    # during warm-up (healthy components), force one internal fidelity trigger — clearly labelled
    # in its metadata — so this validation always exercises one full adaptation cycle.
    if not orchestrator._pending_order:  # noqa: SLF001
        scores = {c: s for c, s in orchestrator._latest_fidelity.items() if s is not None}  # noqa: SLF001
        component = min(scores, key=scores.get) if scores else "dl_bler"
        orchestrator._enqueue_trigger(AdaptationTrigger(  # noqa: SLF001
            component=component, severity=0.42, timestamp=datetime.now(timezone.utc),
            trigger_type="fidelity_degradation",
            metadata={"forced": True, "reason": "run_orchestrator_demo.py validation", "fidelity_score": scores.get(component)},
        ))
        print(f"[demo] no component's policy fired during warm-up — forced one internal fidelity trigger on {component}")

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
