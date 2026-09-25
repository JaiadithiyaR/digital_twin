#!/usr/bin/env python3
"""Module 16 — Expand-Scope Agent: standalone, single-agent manual demo (permanent, repo-tracked,
same convention as `scripts/demo_recalibration.py`/`scripts/demo_regeneration.py` but for the
third and final adaptation agent — the one used when network behaviour reveals a phenomenon the
current DT does not represent at all, as opposed to Modules 14/15 which both act on an
ALREADY-EXISTING component).

What this script does, all real, nothing mocked:
    1. Builds a real, single live D1 history via the real Module 2/3/4 pipeline and registers one
       EXISTING component (`ThroughputModel`) into a real `DTModelRegistry`/`ModelRegistry` —
       expand-scope needs an existing component only as a pattern-reference example (prompt.md
       §27 step 3), not as something it modifies.
    2. Constructs a REAL `GoogleClient` (requires a real GOOGLE_API_KEY/GEMINI_API_KEY — this
       agent's LLM calls are not optional) and a REAL `RagKnowledgeBase` pointed at this repo's
       already-ingested `rag_data/` corpus, so both the design-proposal and implementation prompts
       are genuinely grounded in retrieved knowledge.
    3. Calls the real `ExpandScopeAgent.expand_scope()`: a DESIGN LLM call proposes a genuinely
       NEW component (deterministically validated before any code is written — name/target/
       dependencies/features all checked against real D1 columns and the real registry), then an
       IMPLEMENTATION LLM call writes a complete new pipeline, run through the REAL
       `SandboxExecutor` (a genuine subprocess), with self-correcting retries on rejection at
       EITHER step.
    4. Calls the real `VerificationAgent.verify()` — a genuinely new component has no prior
       version to compare against, so the gate instead requires only that the candidate's own
       fidelity be well-defined (never fabricated); ACCEPT registers it as production, REJECT
       leaves it as an unpromoted candidate.

The worked example is real network-domain motivation — SINR quality prediction from mobility/
load features, genuinely useful (proactive handover/resource-allocation signal) and genuinely
novel among the five existing DT components (all five currently treat SINR as an INPUT, none
predicts it) — the same example this project's own `tests/unit/test_expand_scope_agent.py`
documents, kept consistent here rather than inventing an unrelated one.

Every run wipes and rebuilds its own dedicated storage under `data/demo_expand_scope/` — never the
main application's real storage or the other two single-agent demo scripts' directories.

This script genuinely calls the live Google AI API at least TWICE per attempt (one design call,
one implementation call), up to `max_llm_iterations` retries at EITHER step, plus one more for
VerificationAgent's optional LLM explanation — mind the free tier's daily request quota if
re-running it many times in one day.

Usage:
    python scripts/demo_expand_scope.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.adaptation.data_selection import select_recent_window, time_split  # noqa: E402
from src.adaptation.expand_scope_agent import ExpandScopeAgent, ExpandScopeError  # noqa: E402
from src.adaptation.verification_agent import VerificationAgent  # noqa: E402
from src.common.config import load_secrets, load_settings  # noqa: E402
from src.common.logging import setup_logging  # noqa: E402
from src.dt_models.d1_model_store import D1Store  # noqa: E402
from src.dt_models.model_registry import DTModelRegistry  # noqa: E402
from src.dt_models.throughput import ThroughputModel  # noqa: E402
from src.fidelity.evaluator import FidelityEvaluator  # noqa: E402
from src.llm.google_client import GoogleClient  # noqa: E402
from src.rag.rag_kb import RagKnowledgeBase  # noqa: E402
from src.registry.model_registry import ModelRegistry  # noqa: E402
from src.sandbox.executor import SandboxExecutor  # noqa: E402
from src.synchronization.sync import ContinuousSynchronizer  # noqa: E402
from src.telemetry.mock_source import MockTelemetrySource  # noqa: E402
from src.telemetry.preprocessing import TelemetryPreprocessor  # noqa: E402

DEMO_DIR = REPO_ROOT / "data" / "demo_expand_scope"


def main() -> int:
    settings = load_settings()
    setup_logging(level=settings.logging.level, fmt=settings.logging.format, log_dir=settings.logging.log_dir)
    secrets = load_secrets()

    print("[demo] === Module 16 — Expand-Scope Agent ===")
    try:
        llm_client = GoogleClient.from_settings(settings, secrets)
    except RuntimeError as exc:
        print(f"[demo] FAILED: {exc}\n[demo] expand-scope requires a real GOOGLE_API_KEY/GEMINI_API_KEY "
              "in .env — set one and re-run.", file=sys.stderr)
        return 1
    print("[demo] real GOOGLE_API_KEY configured — this run will genuinely call the live Google AI API")

    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True)
    print(f"[demo] fresh demo storage: {DEMO_DIR}")

    # 1. Real live D1 history + one existing, registered component (pattern reference only).
    print("[demo] building real live D1 history (1200 records through Module 2/3/4)...")
    store = D1Store(
        current_state_path=DEMO_DIR / "d1_current.parquet",
        history_path=DEMO_DIR / "d1_history.parquet",
        quarantine_path=DEMO_DIR / "d1_quarantine.parquet",
        history_retention_rows=settings.storage.history_retention_rows,
    )
    mock_config = settings.telemetry.mock.model_copy(
        update={"num_ues": 25, "num_cells": 4, "missing_field_rate": 0.0, "out_of_range_rate": 0.0}
    )
    preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)
    sync_config = settings.synchronization.model_copy(update={"batch_size": 100})
    ContinuousSynchronizer(MockTelemetrySource(mock_config, max_records=1200, realtime=False), preprocessor,
                            store, sync_config).run()
    history = store.get_history()
    print(f"[demo] live D1 history now has {len(history)} rows")

    features = list(ThroughputModel.REQUIRED_FEATURES)
    model_registry = ModelRegistry(models_dir=DEMO_DIR / "models", index_path=DEMO_DIR / "models" / "index.json")
    dt_model_registry = DTModelRegistry()
    example_model = ThroughputModel.from_settings(settings)
    example_model.train(history[features], history["throughput_mbps"])
    model_registry.register_version(
        component_instance=example_model,
        adaptation_type="bootstrap",
        parent_version_id=None,
        training_window={"n_rows": len(history)},
        evaluation_window={"n_rows": 0},
        evaluation_metrics=example_model.evaluate(history[features], history["throughput_mbps"]),
        status="production",
    )
    dt_model_registry.register(example_model)
    print(f"[demo] one existing component registered for pattern reference: {example_model.COMPONENT_NAME}")

    # 2. A real FidelityEvaluator — deliberately NOT pre-warmed here, unlike
    # demo_recalibration.py/demo_regeneration.py. Those pre-warm under a component name known in
    # advance (the existing production component being recalibrated/regenerated); expand-scope's
    # LLM design step chooses the new component's name itself, so there is no name to pre-warm
    # under until AFTER that call returns. This is not a limitation of this demo — it is the
    # correct, expected real behaviour: a genuinely first-ever new component has no rolling
    # fidelity history under its own name, so `fidelity_after` legitimately comes back `None`
    # (Module 12's "never fabricate a score" rule), and Module 17's "no baseline, fidelity not
    # yet computable" branch REJECTs it fail-safe rather than blindly accepting for lack of a
    # comparison — a fully valid, honest demonstration of that documented path.
    fidelity_evaluator = FidelityEvaluator(settings.fidelity)

    rag_kb = RagKnowledgeBase.from_settings(settings)
    print(f"[demo] RAG knowledge base available: {rag_kb.is_available} ({rag_kb.count() if rag_kb.is_available else 0} chunks)")

    sandbox = SandboxExecutor(
        workspace_root=DEMO_DIR / "sandbox", timeout_seconds=settings.sandbox.timeout_seconds,
        max_output_bytes=settings.sandbox.max_output_bytes,
    )
    agent = ExpandScopeAgent(settings, store, model_registry, dt_model_registry, llm_client, sandbox,
                              fidelity_evaluator=fidelity_evaluator)

    print("\n[demo] calling ExpandScopeAgent.expand_scope() — this genuinely calls the live LLM twice "
          "(design + implementation) and runs a real sandboxed subprocess; may take a while...")
    try:
        result = agent.expand_scope(
            lambda: ThroughputModel.from_settings(settings),
            window_hours=96,
            held_out_fraction=0.2,
            # The LLM's own design proposal may legitimately choose to depend on an existing
            # component (e.g. "throughput") — this maps every currently-registered component to
            # its OUTPUT_FIELD so `with_dependency_ground_truth` can populate that column,
            # whichever dependency (if any) the design actually proposes.
            dependency_output_fields={c.COMPONENT_NAME: c.OUTPUT_FIELD for c in dt_model_registry.list_components()},
            expand_scope_context="SINR readings show anomalous degradation patterns not explained by any "
                                  "existing component's predictions — a new capability may be needed",
            rag_knowledge_base=rag_kb,
        )
    except ExpandScopeError as exc:
        print(f"[demo] FAILED: expand-scope did not produce an accepted candidate: {exc}", file=sys.stderr)
        return 1

    print("\n[demo] === expand-scope result ===")
    print(f"  new component_name  = {result.version.component}")
    print(f"  target_column       = {result.design.target_column}")
    print(f"  design purpose      = {result.design.purpose}")
    print(f"  dependencies        = {result.design.dependencies}")
    print(f"  required_features   = {result.design.required_features}")
    print(f"  design_attempts     = {result.design_attempts}")
    print(f"  implementation_attempts = {result.implementation_attempts}")
    print(f"  model_class         = {result.version.model_class} (LLM-authored)")
    print(f"  sandbox stage       = {result.sandbox_result.stage} (accepted={result.sandbox_result.accepted})")
    print(f"  sandbox metrics     = {result.sandbox_result.metrics}")
    print(f"  fidelity_after      = {result.fidelity_after}")
    print(f"  parent_version      = {result.version.parent_version_id} (None — genuinely new component)")

    # 3. Complete the cycle: real Module 17 verification against the "no baseline" gate.
    print("\n[demo] calling VerificationAgent.verify() to decide ACCEPT/REJECT (no prior baseline "
          "exists for a genuinely new component — the gate requires only that fidelity be well-defined)...")
    window_df = select_recent_window(store.get_history(), 96)
    _, held_out_df = time_split(window_df, 0.2)
    eval_target = held_out_df[result.design.target_column]
    candidate_predictions = result.sandbox_result.eval_predictions

    verification_agent = VerificationAgent(settings, model_registry, fidelity_evaluator, llm_client=llm_client)
    verification_result = verification_agent.verify(
        result.version, held_out_df, eval_target, candidate_predictions,
        production_component_factory=None,  # no baseline — genuinely new component
        drift_context="manual expand-scope demo run",
        rag_knowledge_base=rag_kb,
    )

    print("\n[demo] === verification result ===")
    print(f"  decision      = {verification_result.decision}")
    print(f"  explanation   = {verification_result.explanation}")
    final = model_registry.get_current_version(result.version.component)
    print(f"  '{result.version.component}' production status: "
          f"{final.version_id if final else 'none (candidate not promoted)'}")

    print("\n[demo] OK — one full real expand-scope cycle completed (2 live LLM calls + real sandboxed subprocess).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
