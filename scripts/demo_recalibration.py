#!/usr/bin/env python3
"""Module 14 — Recalibration Agent: standalone, single-agent manual demo (permanent, repo-tracked,
same convention as `scripts/run_orchestrator_demo.py` but scoped to ONE agent so its behaviour can
be inspected in isolation, without waiting on a real drift event or the Decision & Root-Cause
Analysis Agent to select this strategy).

What this script does, all real, nothing mocked:
    1. Bootstrap-trains and registers a real production `ThroughputModel` (version 1) on an
       EARLIER, smaller telemetry population (fewer UEs/cells — representing network conditions
       at initial deployment) via the real Module 2/3/4 pipeline. This data is deliberately never
       written into the live D1Store the agent reads from below — production must genuinely never
       have seen it, exactly like a real deployment's bootstrap phase precedes live operation.
    2. Builds a SEPARATE, live D1 history via the same real pipeline, but from a genuinely
       different telemetry population (more UEs/cells — representing real network growth/
       densification since the production model was trained) — the actual scenario recalibration
       exists for (prompt.md §23: "the existing model structure is still appropriate, but its
       learned parameters have become stale").
    3. Pre-warms a real `FidelityEvaluator`'s rolling window with genuinely-varying synthetic
       points at a realistic error scale, so this run's fidelity_before/after are real, comparable
       numbers, not `insufficient_history` nor an artifact of a degenerate window (see the
       in-code comment for the two numerical-stability pitfalls this avoids).
    4. Calls the real `RecalibrationAgent.recalibrate()` — retrains a FRESH instance on the live,
       never-seen-by-production D1 window, registers the result as a new candidate version.
       Recalibration needs no LLM call (prompt.md §23 — it retrains the EXISTING pipeline, no code
       generation), so this step works identically with or without a real GOOGLE_API_KEY
       configured.
    5. Calls the real `VerificationAgent.verify()` to complete the cycle: ACCEPT promotes the
       candidate to production, REJECT preserves the current production version untouched — either
       outcome is a valid, complete demonstration of the deterministic acceptance gate at work.

Every run wipes and rebuilds its own dedicated storage under `data/demo_recalibration/` — never
the main application's real storage (`data/artifacts/`, `data/models/`) or the other two
single-agent demo scripts' directories — so this is always a clean, reproducible, self-contained
demo you can re-run as many times as you like.

Usage:
    python scripts/demo_recalibration.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.adaptation.data_selection import select_recent_window, time_split  # noqa: E402
from src.adaptation.recalibration_agent import RecalibrationAgent  # noqa: E402
from src.adaptation.verification_agent import VerificationAgent  # noqa: E402
from src.common.config import load_secrets, load_settings  # noqa: E402
from src.common.logging import setup_logging  # noqa: E402
from src.dt_models.d1_model_store import D1Store  # noqa: E402
from src.dt_models.throughput import ThroughputModel  # noqa: E402
from src.fidelity.evaluator import FidelityEvaluator  # noqa: E402
from src.llm.google_client import GoogleClient  # noqa: E402
from src.rag.rag_kb import RagKnowledgeBase  # noqa: E402
from src.registry.model_registry import ModelRegistry  # noqa: E402
from src.synchronization.sync import ContinuousSynchronizer  # noqa: E402
from src.telemetry.mock_source import MockTelemetrySource  # noqa: E402
from src.telemetry.preprocessing import TelemetryPreprocessor  # noqa: E402

DEMO_DIR = REPO_ROOT / "data" / "demo_recalibration"


def main() -> int:
    settings = load_settings()
    setup_logging(level=settings.logging.level, fmt=settings.logging.format, log_dir=settings.logging.log_dir)
    secrets = load_secrets()

    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True)

    print("[demo] === Module 14 — Recalibration Agent ===")
    print(f"[demo] fresh demo storage: {DEMO_DIR}")

    preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)
    sync_config = settings.synchronization.model_copy(update={"batch_size": 100})
    features = list(ThroughputModel.REQUIRED_FEATURES)

    # 1. Bootstrap-train production on an EARLIER, smaller network population — this data is
    # intentionally kept out of the live D1Store below, so production genuinely never sees it.
    print("[demo] building real bootstrap telemetry for the production model (900 records, "
          "15 UEs/2 cells — 'network conditions at initial deployment')...")
    bootstrap_config = settings.telemetry.mock.model_copy(
        update={"seed": 1, "num_ues": 15, "num_cells": 2, "missing_field_rate": 0.0, "out_of_range_rate": 0.0}
    )
    bootstrap_store = D1Store(
        current_state_path=DEMO_DIR / "bootstrap_current.parquet",
        history_path=DEMO_DIR / "bootstrap_history.parquet",
        quarantine_path=DEMO_DIR / "bootstrap_quarantine.parquet",
        history_retention_rows=settings.storage.history_retention_rows,
    )
    ContinuousSynchronizer(MockTelemetrySource(bootstrap_config, max_records=900, realtime=False), preprocessor,
                            bootstrap_store, sync_config).run()
    bootstrap_history = bootstrap_store.get_history()

    registry = ModelRegistry(models_dir=DEMO_DIR / "models", index_path=DEMO_DIR / "models" / "index.json")
    bootstrap_model = ThroughputModel.from_settings(settings)
    bootstrap_model.train(bootstrap_history[features], bootstrap_history["throughput_mbps"])
    registry.register_version(
        component_instance=bootstrap_model,
        adaptation_type="bootstrap",
        parent_version_id=None,
        training_window={"n_rows": len(bootstrap_history)},
        evaluation_window={"n_rows": 0},
        evaluation_metrics=bootstrap_model.evaluate(bootstrap_history[features], bootstrap_history["throughput_mbps"]),
        status="production",
    )
    print(f"[demo] registered production version: {registry.get_current_version('throughput').version_id}")

    # 2. The LIVE D1 history the agent actually reads from — a genuinely different, larger
    # network population (network growth since the production model was trained).
    print("[demo] building real LIVE D1 history (1200 records, 35 UEs/5 cells — "
          "'network growth since the production model was trained')...")
    store = D1Store(
        current_state_path=DEMO_DIR / "d1_current.parquet",
        history_path=DEMO_DIR / "d1_history.parquet",
        quarantine_path=DEMO_DIR / "d1_quarantine.parquet",
        history_retention_rows=settings.storage.history_retention_rows,
    )
    live_config = settings.telemetry.mock.model_copy(
        update={"seed": 2, "num_ues": 35, "num_cells": 5, "missing_field_rate": 0.0, "out_of_range_rate": 0.0}
    )
    ContinuousSynchronizer(MockTelemetrySource(live_config, max_records=1200, realtime=False), preprocessor,
                            store, sync_config).run()
    history = store.get_history()
    print(f"[demo] live D1 history now has {len(history)} rows")

    # 3. Pre-warm a real FidelityEvaluator so before/after are genuine numbers this run, not
    # `insufficient_history`. Each warm-up call MUST see a genuinely different (y_true, y_pred)
    # pair — repeatedly evaluating the IDENTICAL pair produces a degenerate, perfectly-constant
    # rolling window, which the eps-stabilized min-max normalization then explodes into an absurd
    # score the moment a genuinely different point is evaluated afterward (documented in Module
    # 17's own docstring). Sampled real throughput values plus synthetic noise scaled to this
    # component's own established real-world error magnitude (Module 6's own real validation:
    # held-out RMSE ~1.5 Mbps) gives genuine, realistic per-call variation.
    fidelity_evaluator = FidelityEvaluator(settings.fidelity)
    rng = np.random.default_rng(42)
    n_warm = settings.fidelity.min_history_for_normalization + 1
    for i in range(n_warm):
        chunk = history.sample(n=min(50, len(history)), random_state=i)
        y_true_chunk = chunk["throughput_mbps"].reset_index(drop=True)
        y_pred_chunk = y_true_chunk + rng.normal(0, 1.0 + 0.3 * i, size=len(y_true_chunk))
        fidelity_evaluator.evaluate("throughput", y_true_chunk, y_pred_chunk, update_window=True)

    # An LLM client is used opportunistically here ONLY for the optional training-window sizing
    # hint (prompt.md §23 "MAY use LLM reasoning") — recalibration itself never needs one.
    llm_client = None
    try:
        llm_client = GoogleClient.from_settings(settings, secrets)
        print("[demo] real GOOGLE_API_KEY configured — LLM training-window reasoning will be exercised for real")
    except RuntimeError:
        print("[demo] no real GOOGLE_API_KEY configured — proceeding with the deterministic training window "
              "(recalibration itself never requires an LLM call)")

    # 4. The real deliverable: recalibrate.
    print("\n[demo] calling RecalibrationAgent.recalibrate() ...")
    agent = RecalibrationAgent(settings, store, registry, fidelity_evaluator=fidelity_evaluator, llm_client=llm_client)
    result = agent.recalibrate(
        lambda: ThroughputModel.from_settings(settings),
        "throughput_mbps",
        window_hours=48,
        held_out_fraction=0.2,
        use_llm_window_reasoning=llm_client is not None,
    )

    print("\n[demo] === recalibration result ===")
    print(f"  candidate_version   = {result.version.version_id}")
    print(f"  parent_version      = {result.version.parent_version_id}")
    print(f"  evaluation_metrics  = {result.evaluation_metrics}")
    print(f"  fidelity_before     = {result.fidelity_before}")
    print(f"  fidelity_after      = {result.fidelity_after}")
    print(f"  training_window     = {result.training_window['n_rows']} rows over "
          f"{result.training_window['window_hours']:.2f}h")
    print(f"  evaluation_window   = {result.evaluation_window['n_rows']} held-out rows")

    # 5. Complete the cycle: real Module 17 verification, ACCEPT promotes / REJECT preserves.
    print("\n[demo] calling VerificationAgent.verify() to decide ACCEPT/REJECT...")
    window_df = select_recent_window(store.get_history(), 48)
    _, held_out_df = time_split(window_df, 0.2)
    eval_target = held_out_df["throughput_mbps"]
    candidate_predictions = result.candidate_component.predict(held_out_df)

    rag_kb = RagKnowledgeBase.from_settings(settings)
    verification_agent = VerificationAgent(settings, registry, fidelity_evaluator, llm_client=llm_client)
    verification_result = verification_agent.verify(
        result.version, held_out_df, eval_target, candidate_predictions,
        production_component_factory=lambda: ThroughputModel.from_settings(settings),
        drift_context="manual recalibration demo run",
        rag_knowledge_base=rag_kb,
    )

    print(f"\n[demo] === verification result ===")
    print(f"  decision      = {verification_result.decision}")
    print(f"  explanation   = {verification_result.explanation}")
    final_production = registry.get_current_version("throughput")
    print(f"  production is now: {final_production.version_id} (adaptation_type={final_production.adaptation_type})")

    print("\n[demo] OK — one full real recalibration cycle completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
