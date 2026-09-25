#!/usr/bin/env python3
"""Module 15 — Regeneration Agent: standalone, single-agent manual demo (permanent, repo-tracked,
same convention as `scripts/demo_recalibration.py`/`scripts/run_orchestrator_demo.py` but scoped
to ONE agent so its behaviour — including a genuine LLM code-generation call and a real sandboxed
subprocess training run — can be inspected in isolation).

What this script does, all real, nothing mocked:
    1. Bootstrap-trains and registers a real production `ThroughputModel` (version 1) on an
       EARLIER, smaller telemetry population (fewer UEs/cells), kept out of the live D1Store the
       agent reads from — exactly like `demo_recalibration.py`'s fair before/after split.
    2. Builds a SEPARATE, live D1 history from a genuinely different (larger) telemetry
       population — the real scenario Regeneration exists for (prompt.md §24: "the current model
       architecture/pipeline is no longer capable of representing the changed behaviour").
    3. Pre-warms a real `FidelityEvaluator`'s rolling window with genuinely-varying synthetic
       points at a realistic error scale (same technique as `demo_recalibration.py`, same reason).
    4. Constructs a REAL `GoogleClient` (requires a real GOOGLE_API_KEY/GEMINI_API_KEY — this
       agent's LLM call is not optional, unlike recalibration's) and a REAL `RagKnowledgeBase`
       pointed at this repo's already-ingested `rag_data/` corpus, so the regeneration prompt is
       genuinely grounded in retrieved knowledge, not a hardcoded "not available" placeholder.
    5. Calls the real `RegenerationAgent.regenerate()` — the LLM designs and writes a COMPLETE new
       component pipeline from scratch, which is then run through the REAL `SandboxExecutor` (a
       genuine subprocess: syntax -> import -> conformance -> train -> evaluate), with up to
       `config.adaptation.regeneration.max_llm_iterations` self-correcting retries on rejection.
    6. Calls the real `VerificationAgent.verify()` to complete the cycle: ACCEPT promotes the
       candidate to production, REJECT preserves the current production version untouched.

Every run wipes and rebuilds its own dedicated storage under `data/demo_regeneration/` — never the
main application's real storage or the other two single-agent demo scripts' directories.

This script genuinely calls the live Google AI API (at least once, up to
`max_llm_iterations` times if the sandbox rejects a candidate, plus one more for
VerificationAgent's optional LLM explanation) — mind the free tier's daily request quota if
re-running it many times in one day.

Usage:
    python scripts/demo_regeneration.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.adaptation.data_selection import select_recent_window, time_split  # noqa: E402
from src.adaptation.regeneration_agent import RegenerationAgent, RegenerationError  # noqa: E402
from src.adaptation.verification_agent import VerificationAgent  # noqa: E402
from src.common.config import load_secrets, load_settings  # noqa: E402
from src.common.logging import setup_logging  # noqa: E402
from src.dt_models.d1_model_store import D1Store  # noqa: E402
from src.dt_models.throughput import ThroughputModel  # noqa: E402
from src.fidelity.evaluator import FidelityEvaluator  # noqa: E402
from src.llm.google_client import GoogleClient  # noqa: E402
from src.rag.rag_kb import RagKnowledgeBase  # noqa: E402
from src.registry.model_registry import ModelRegistry  # noqa: E402
from src.sandbox.executor import SandboxExecutor  # noqa: E402
from src.synchronization.sync import ContinuousSynchronizer  # noqa: E402
from src.telemetry.mock_source import MockTelemetrySource  # noqa: E402
from src.telemetry.preprocessing import TelemetryPreprocessor  # noqa: E402

DEMO_DIR = REPO_ROOT / "data" / "demo_regeneration"


def main() -> int:
    settings = load_settings()
    setup_logging(level=settings.logging.level, fmt=settings.logging.format, log_dir=settings.logging.log_dir)
    secrets = load_secrets()

    print("[demo] === Module 15 — Regeneration Agent ===")
    try:
        llm_client = GoogleClient.from_settings(settings, secrets)
    except RuntimeError as exc:
        print(f"[demo] FAILED: {exc}\n[demo] regeneration requires a real GOOGLE_API_KEY/GEMINI_API_KEY "
              "in .env — set one and re-run.", file=sys.stderr)
        return 1
    print("[demo] real GOOGLE_API_KEY configured — this run will genuinely call the live Google AI API")

    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True)
    print(f"[demo] fresh demo storage: {DEMO_DIR}")

    preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)
    sync_config = settings.synchronization.model_copy(update={"batch_size": 100})
    features = list(ThroughputModel.REQUIRED_FEATURES)

    # 1. Bootstrap-train production on an EARLIER, smaller network population — kept out of the
    # live D1Store below, so production genuinely never sees it.
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

    # 3. Pre-warm a real FidelityEvaluator so before/after are genuine numbers this run — see
    # `demo_recalibration.py`'s in-code comment for exactly why this technique (real values plus
    # genuinely-varying synthetic noise, never the identical pair repeated) is used.
    fidelity_evaluator = FidelityEvaluator(settings.fidelity)
    rng = np.random.default_rng(42)
    n_warm = settings.fidelity.min_history_for_normalization + 1
    for i in range(n_warm):
        chunk = history.sample(n=min(50, len(history)), random_state=i)
        y_true_chunk = chunk["throughput_mbps"].reset_index(drop=True)
        y_pred_chunk = y_true_chunk + rng.normal(0, 1.0 + 0.3 * i, size=len(y_true_chunk))
        fidelity_evaluator.evaluate("throughput", y_true_chunk, y_pred_chunk, update_window=True)

    # 4. Real RAG knowledge base, pointed at this repo's already-ingested rag_data/ corpus.
    rag_kb = RagKnowledgeBase.from_settings(settings)
    print(f"[demo] RAG knowledge base available: {rag_kb.is_available} ({rag_kb.count() if rag_kb.is_available else 0} chunks)")

    sandbox = SandboxExecutor(
        workspace_root=DEMO_DIR / "sandbox", timeout_seconds=settings.sandbox.timeout_seconds,
        max_output_bytes=settings.sandbox.max_output_bytes,
    )
    agent = RegenerationAgent(settings, store, registry, llm_client, sandbox, fidelity_evaluator=fidelity_evaluator)

    print("\n[demo] calling RegenerationAgent.regenerate() — this genuinely calls the live LLM and "
          "runs a real sandboxed subprocess; may take a while...")
    try:
        result = agent.regenerate(
            lambda: ThroughputModel.from_settings(settings),
            "throughput_mbps",
            window_hours=96,
            held_out_fraction=0.2,
            drift_context="network densification (more UEs/cells) since the production model was trained — "
                           "its pipeline architecture may no longer represent current load conditions",
            rag_knowledge_base=rag_kb,
        )
    except RegenerationError as exc:
        print(f"[demo] FAILED: regeneration did not produce an accepted candidate: {exc}", file=sys.stderr)
        return 1

    print("\n[demo] === regeneration result ===")
    print(f"  candidate_version   = {result.version.version_id}")
    print(f"  model_class         = {result.version.model_class} (LLM-authored)")
    print(f"  parent_version      = {result.version.parent_version_id}")
    print(f"  attempts            = {result.attempts} (self-correction retries: {result.attempts - 1})")
    print(f"  sandbox stage       = {result.sandbox_result.stage} (accepted={result.sandbox_result.accepted})")
    print(f"  declared dependencies    = {result.sandbox_result.dependencies}")
    print(f"  declared features        = {result.sandbox_result.required_features}")
    print(f"  sandbox metrics     = {result.sandbox_result.metrics}")
    print(f"  fidelity_before     = {result.fidelity_before}")
    print(f"  fidelity_after      = {result.fidelity_after}")
    print(f"  LLM reasoning       = {result.version.llm_metadata.get('reasoning')}")

    # 5. Complete the cycle: real Module 17 verification, ACCEPT promotes / REJECT preserves.
    print("\n[demo] calling VerificationAgent.verify() to decide ACCEPT/REJECT...")
    window_df = select_recent_window(store.get_history(), 96)
    _, held_out_df = time_split(window_df, 0.2)
    eval_target = held_out_df["throughput_mbps"]
    candidate_predictions = result.sandbox_result.eval_predictions

    verification_agent = VerificationAgent(settings, registry, fidelity_evaluator, llm_client=llm_client)
    verification_result = verification_agent.verify(
        result.version, held_out_df, eval_target, candidate_predictions,
        production_component_factory=lambda: ThroughputModel.from_settings(settings),
        drift_context="manual regeneration demo run",
        rag_knowledge_base=rag_kb,
    )

    print("\n[demo] === verification result ===")
    print(f"  decision      = {verification_result.decision}")
    print(f"  explanation   = {verification_result.explanation}")
    final_production = registry.get_current_version("throughput")
    print(f"  production is now: {final_production.version_id} (adaptation_type={final_production.adaptation_type})")

    print("\n[demo] OK — one full real regeneration cycle completed (live LLM call + real sandboxed subprocess).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
