"""Sandbox execution layer (prompt.md §45) — shared infrastructure for Modules 15/16, whose
LLM-generated code must never be trusted or executed directly in the production process
(CLAUDE.md §7: "LLM output is always untrusted"; prompt.md §70 rule 10: "LLM-generated code is
never production code until sandboxed and verified").

**What "practical isolation available on the host OS" means here, stated plainly (prompt.md §45
explicitly asks for this security-assumptions documentation, not a false guarantee of perfect
isolation):**

- Candidate code runs as a genuinely SEPARATE OS PROCESS (`subprocess.run`, never `exec()`/
  `importlib` in the parent process). This is real, load-bearing isolation: the candidate cannot
  touch the parent's Python objects, imported modules, in-memory state, or crash the parent
  process — a candidate that raises, hangs, or does something exotic like `sys.exit()` at import
  time only ever affects its own subprocess, never this one.
- The subprocess is given a MINIMAL environment (`PATH` + a `PYTHONPATH` pointing only at this
  repo) — NOT `os.environ.copy()`. Concretely and testably: `ANTHROPIC_API_KEY` and any other
  secret present in the parent's environment is NEVER inherited by the sandboxed subprocess
  (prompt.md §16 "never hardcode API keys" extended here — a candidate must not even be *able* to
  read one via `os.environ`, whether that candidate is malicious or just careless).
- Execution is time-bounded (`config.sandbox.timeout_seconds`, enforced by `subprocess.run(...,
  timeout=...)`) and output-bounded (`config.sandbox.max_output_bytes`, truncated after capture)
  — the two concrete defenses against a runaway or output-flooding candidate.
- **What this does NOT prevent, stated honestly rather than silently assumed away**: the
  subprocess runs as the SAME OS user as the parent, with the SAME filesystem permissions. Nothing
  here stops a deliberately adversarial script from using an absolute path to read or write
  outside its workspace, or from making network calls. True filesystem/network sandboxing would
  need OS-level primitives (containers, seccomp, a chroot/namespace jail) that this environment
  cannot assume are available, and building/maintaining that infrastructure is out of scope for
  this project's current phase (prompt.md §10: "do not introduce unnecessary infrastructure"
  cuts both ways — a fake, if the host can't back it, would be worse than documenting the gap).
  In practice, candidates are LLM-generated `DTComponent` implementations with no reason to touch
  paths outside their own `cwd` (relative-path I/O, as any reasonable generated code produces),
  and are also subject to a deterministic conformance check (see `_sandbox_driver.py`) before any
  training code runs at all.
- **A deliberate, documented omission**: this executor does NOT set POSIX resource limits
  (`resource.setrlimit` via `subprocess`'s `preexec_fn`) for CPU/memory. Python's own
  `subprocess` documentation warns `preexec_fn` is unsafe in a multi-threaded process (this
  system runs `ContinuousSynchronizer` on a background thread continuously) — the fork-related
  deadlock risk that warning describes was judged worse than the resource-limit gap it would
  close, given `timeout_seconds` already bounds wall-clock (and therefore practically bounds
  runaway CPU use) and `max_output_bytes` already bounds an output-flooding vector.

**Data hand-off is entirely file-based and one-directional into the sandbox, in-memory back out**:
the parent writes a `candidate.py` + `manifest.json` + parquet snapshots of the training/
evaluation data into a fresh, per-call workspace; `_sandbox_driver.py` (a FIXED, project-authored
file — never LLM-generated, never modified by a candidate) is what actually imports and runs the
candidate, reporting `result.json`. `SandboxExecutor.run_candidate()` reads the outcome — trained
artifact bytes, evaluation predictions, metrics — straight into memory and ALWAYS deletes the
workspace directory before returning (prompt.md §45 "clean up temporary workspaces safely"),
regardless of success, failure, or an unexpected exception — so a caller never needs to remember
to clean up, and no workspace can leak past one `run_candidate()` call.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from src.common.config import Settings

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
_DRIVER_PATH = Path(__file__).resolve().parent / "_sandbox_driver.py"

SandboxStage = Literal["syntax", "import", "conformance", "train", "evaluate", "save", "ok"]


@dataclass(frozen=True)
class SandboxResult:
    accepted: bool
    stage: SandboxStage
    error: str | None
    stdout: str
    stderr: str
    exit_code: int | None
    metrics: dict[str, float] | None = None
    dependencies: tuple[str, ...] | None = None
    required_features: tuple[str, ...] | None = None
    artifact_bytes: bytes | None = None
    eval_predictions: pd.Series | None = None


class SandboxExecutor:
    def __init__(self, workspace_root: Path, timeout_seconds: int, max_output_bytes: int) -> None:
        self._workspace_root = workspace_root
        self._timeout_seconds = timeout_seconds
        self._max_output_bytes = max_output_bytes
        self._workspace_root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(cls, settings: "Settings") -> "SandboxExecutor":
        return cls(
            workspace_root=settings.resolve_path(settings.sandbox.workspace_root),
            timeout_seconds=settings.sandbox.timeout_seconds,
            max_output_bytes=settings.sandbox.max_output_bytes,
        )

    def run_candidate(
        self,
        *,
        source_code: str,
        class_name: str,
        expected_component_name: str,
        expected_output_field: str,
        target_column: str,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        eval_features: pd.DataFrame,
        eval_target: pd.Series,
    ) -> SandboxResult:
        # Stage: syntax — `compile(..., "exec")` only PARSES the source; it never executes it,
        # so this is safe to do directly in the parent process, no isolation needed.
        try:
            compile(source_code, "candidate.py", "exec")
        except SyntaxError as exc:
            return SandboxResult(
                accepted=False, stage="syntax", error=str(exc), stdout="", stderr="", exit_code=None
            )

        workspace = self._workspace_root / f"candidate-{uuid.uuid4().hex}"
        try:
            return self._run_in_workspace(
                workspace,
                source_code=source_code,
                class_name=class_name,
                expected_component_name=expected_component_name,
                expected_output_field=expected_output_field,
                target_column=target_column,
                train_features=train_features,
                train_target=train_target,
                eval_features=eval_features,
                eval_target=eval_target,
            )
        finally:
            # ALWAYS — success, rejection, or an exception raised above — never leak a workspace.
            shutil.rmtree(workspace, ignore_errors=True)

    def _run_in_workspace(
        self,
        workspace: Path,
        *,
        source_code: str,
        class_name: str,
        expected_component_name: str,
        expected_output_field: str,
        target_column: str,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        eval_features: pd.DataFrame,
        eval_target: pd.Series,
    ) -> SandboxResult:
        workspace.mkdir(parents=True, exist_ok=False)
        (workspace / "candidate.py").write_text(source_code)
        manifest = {
            "class_name": class_name,
            "expected_component_name": expected_component_name,
            "expected_output_field": expected_output_field,
            "target_column": target_column,
        }
        (workspace / "manifest.json").write_text(json.dumps(manifest))
        train_features.to_parquet(workspace / "train_features.parquet")
        train_target.to_frame(name=target_column).to_parquet(workspace / "train_target.parquet")
        eval_features.to_parquet(workspace / "eval_features.parquet")
        eval_target.to_frame(name=target_column).to_parquet(workspace / "eval_target.parquet")

        # Deliberately minimal — see module docstring: secrets in the parent's environment are
        # never inherited by the sandboxed subprocess.
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(REPO_ROOT)}

        try:
            proc = subprocess.run(
                [sys.executable, str(_DRIVER_PATH), str(workspace)],
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
            stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            return SandboxResult(
                accepted=False,
                stage="train",
                error=f"execution exceeded the {self._timeout_seconds}s sandbox timeout",
                stdout=self._truncate(stdout),
                stderr=self._truncate(stderr),
                exit_code=None,
            )

        result_path = workspace / "result.json"
        if not result_path.exists():
            return SandboxResult(
                accepted=False,
                stage="import",
                error=f"sandbox process produced no result.json (exit code {exit_code})",
                stdout=self._truncate(stdout),
                stderr=self._truncate(stderr),
                exit_code=exit_code,
            )

        payload = json.loads(result_path.read_text())
        if payload["status"] != "ok":
            return SandboxResult(
                accepted=False,
                stage=payload.get("stage", "import"),
                error=payload.get("error"),
                stdout=self._truncate(stdout),
                stderr=self._truncate(stderr),
                exit_code=exit_code,
            )

        artifact_path = workspace / "artifact.joblib"
        predictions_path = workspace / "eval_predictions.parquet"
        if not artifact_path.exists() or not predictions_path.exists():
            # Defense in depth: the driver claimed success but didn't produce what it should
            # have — never treat that as acceptance.
            return SandboxResult(
                accepted=False,
                stage="save",
                error="driver reported success but artifact/predictions files are missing",
                stdout=self._truncate(stdout),
                stderr=self._truncate(stderr),
                exit_code=exit_code,
            )

        return SandboxResult(
            accepted=True,
            stage="ok",
            error=None,
            stdout=self._truncate(stdout),
            stderr=self._truncate(stderr),
            exit_code=exit_code,
            metrics=payload["metrics"],
            dependencies=tuple(payload["dependencies"]),
            required_features=tuple(payload["required_features"]),
            artifact_bytes=artifact_path.read_bytes(),
            eval_predictions=pd.read_parquet(predictions_path)["prediction"],
        )

    def _truncate(self, text: str) -> str:
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) <= self._max_output_bytes:
            return text
        return encoded[: self._max_output_bytes].decode("utf-8", errors="ignore") + "...[truncated]"
