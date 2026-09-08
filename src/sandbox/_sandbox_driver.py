"""Fixed, project-authored sandbox driver (prompt.md §45/§26) — this file is NEVER LLM-generated
and NEVER modified by a candidate; it is what actually runs an untrusted candidate component.

Runs as a completely separate OS process (spawned by `SandboxExecutor.run_candidate`), isolated
from the parent process's memory, imported modules, and environment variables (see
`executor.py`'s minimal-env construction — secrets are never inherited here). Its only job:
import the candidate module, verify it conforms to the `DTComponent` interface contract
(prompt.md §26 "unit tests" — a fixed, deterministic conformance check, not LLM-authored), then
train/evaluate/save it against a data snapshot the parent already wrote into this workspace, and
report a structured result. Every stage is wrapped so a failure — including the candidate raising
something exotic like `SystemExit` — is always captured as data (`result.json`), never an opaque
crash with no diagnostic.

Usage: python _sandbox_driver.py <workspace_dir>
Reads:  <workspace_dir>/manifest.json, candidate.py, train_features.parquet, train_target.parquet,
        eval_features.parquet, eval_target.parquet (all written by SandboxExecutor beforehand).
Writes: <workspace_dir>/result.json always; artifact.joblib + eval_predictions.parquet on success.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def _write_result(workspace: Path, **kwargs: object) -> None:
    (workspace / "result.json").write_text(json.dumps(kwargs, default=str))


def main() -> int:
    workspace = Path(sys.argv[1]).resolve()
    manifest = json.loads((workspace / "manifest.json").read_text())

    import pandas as pd

    # Stage: import — the candidate module IS the untrusted code under test. A bare `except
    # BaseException` is deliberate here (and at every later stage): even something exotic like
    # a candidate calling `sys.exit()` at import time must be captured as a structured result,
    # never allowed to kill this driver without leaving `result.json` behind.
    sys.path.insert(0, str(workspace))
    try:
        import candidate
    except BaseException as exc:
        _write_result(
            workspace, status="error", stage="import",
            error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
        )
        return 1

    # Stage: conformance — deterministic, project-authored checks (prompt.md §26's "unit tests"
    # step). The candidate's identity contract (COMPONENT_NAME, OUTPUT_FIELD) must match exactly
    # what other components already depend on; DEPENDENCIES/REQUIRED_FEATURES may legitimately
    # differ (that's the "rebuilt pipeline"), reported back to the parent below.
    try:
        from src.dt_models.base import DTComponent

        cls = getattr(candidate, manifest["class_name"], None)
        if cls is None:
            raise ValueError(f"candidate.py defines no attribute named {manifest['class_name']!r}")
        if not (isinstance(cls, type) and issubclass(cls, DTComponent)):
            raise TypeError(f"{manifest['class_name']!r} is not a DTComponent subclass")
        instance = cls()  # must be constructible with no required arguments
        if instance.COMPONENT_NAME != manifest["expected_component_name"]:
            raise ValueError(
                f"COMPONENT_NAME mismatch: expected {manifest['expected_component_name']!r}, got "
                f"{instance.COMPONENT_NAME!r} — regeneration must not change a component's identity"
            )
        if instance.OUTPUT_FIELD != manifest["expected_output_field"]:
            raise ValueError(
                f"OUTPUT_FIELD mismatch: expected {manifest['expected_output_field']!r}, got "
                f"{instance.OUTPUT_FIELD!r} — other components depend on this exact column name"
            )
    except BaseException as exc:
        _write_result(
            workspace, status="error", stage="conformance",
            error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
        )
        return 1

    # Stage: train — the generic DTComponent.train() interface (Module 5), unchanged.
    try:
        train_features = pd.read_parquet(workspace / "train_features.parquet")
        train_target = pd.read_parquet(workspace / "train_target.parquet")[manifest["target_column"]]
        instance.train(train_features, train_target)
    except BaseException as exc:
        _write_result(
            workspace, status="error", stage="train",
            error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
        )
        return 1

    # Stage: evaluate — component-local metrics (Module 12 owns the authoritative fidelity
    # formula; the parent process computes that separately from `eval_predictions.parquet`).
    try:
        eval_features = pd.read_parquet(workspace / "eval_features.parquet")
        eval_target = pd.read_parquet(workspace / "eval_target.parquet")[manifest["target_column"]]
        metrics = instance.evaluate(eval_features, eval_target)
        if not isinstance(metrics, dict):
            raise TypeError(f"evaluate() must return a dict[str, float], got {type(metrics).__name__}")
        predictions = instance.predict(eval_features)
        pd.DataFrame({"prediction": predictions}).to_parquet(workspace / "eval_predictions.parquet")
    except BaseException as exc:
        _write_result(
            workspace, status="error", stage="evaluate",
            error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
        )
        return 1

    # Stage: save
    try:
        instance.save(workspace / "artifact.joblib")
    except BaseException as exc:
        _write_result(
            workspace, status="error", stage="save",
            error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
        )
        return 1

    _write_result(
        workspace,
        status="ok",
        stage="ok",
        metrics=metrics,
        dependencies=list(instance.DEPENDENCIES),
        required_features=list(instance.REQUIRED_FEATURES),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
