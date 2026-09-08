"""Unit tests for SandboxExecutor (prompt.md §45) — the isolation boundary Modules 15/16's
LLM-generated code must run inside. Drives the REAL subprocess/sandbox mechanism against real
(hand-written, standing in for LLM output — see test_regeneration_agent.py for the note on why no
real ANTHROPIC_API_KEY is available in this environment) candidate source strings, never mocked
out — this is exactly the deterministic, non-LLM code that judges untrusted code, so it needs to
be proven correct for real.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_settings
from src.sandbox.executor import SandboxExecutor

SETTINGS = load_settings()

VALID_SOURCE = """
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from src.dt_models.base import DTComponent

class Rebuilt(DTComponent):
    COMPONENT_NAME = "throughput"
    DEPENDENCIES = ()
    REQUIRED_FEATURES = ("a", "b")
    OUTPUT_FIELD = "throughput_mbps_pred"

    def __init__(self):
        self._model = LinearRegression()
        self._trained = False

    @property
    def is_trained(self):
        return self._trained

    def train(self, features, targets):
        self._model.fit(features[list(self.REQUIRED_FEATURES)], targets)
        self._trained = True

    def predict(self, features):
        preds = self._model.predict(features[list(self.REQUIRED_FEATURES)])
        return pd.Series(preds, index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features, targets):
        preds = self.predict(features)
        rmse = float(((preds - targets) ** 2).mean() ** 0.5)
        return {"rmse": rmse}

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)

    def load(self, path):
        self._model = joblib.load(path)
        self._trained = True
"""


def _executor(tmp_path, **overrides) -> SandboxExecutor:
    return SandboxExecutor(
        workspace_root=tmp_path / "sandbox",
        timeout_seconds=overrides.get("timeout_seconds", SETTINGS.sandbox.timeout_seconds),
        max_output_bytes=overrides.get("max_output_bytes", SETTINGS.sandbox.max_output_bytes),
    )


def _small_dataset():
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({"a": rng.uniform(0, 10, n), "b": rng.uniform(0, 10, n)})
    y = pd.Series(X["a"] * 2 + rng.normal(0, 0.1, n), name="t")
    return X.iloc[:45], y.iloc[:45], X.iloc[45:], y.iloc[45:]


def _run(executor, source, class_name="Rebuilt", **overrides):
    train_X, train_y, eval_X, eval_y = _small_dataset()
    return executor.run_candidate(
        source_code=source,
        class_name=class_name,
        expected_component_name=overrides.get("expected_component_name", "throughput"),
        expected_output_field=overrides.get("expected_output_field", "throughput_mbps_pred"),
        target_column="t",
        train_features=train_X,
        train_target=train_y,
        eval_features=eval_X,
        eval_target=eval_y,
    )


def test_valid_candidate_is_accepted_with_real_metrics_and_artifact(tmp_path):
    result = _run(_executor(tmp_path), VALID_SOURCE)

    assert result.accepted is True
    assert result.stage == "ok"
    assert result.error is None
    assert "rmse" in result.metrics
    assert result.dependencies == ()
    assert result.required_features == ("a", "b")
    assert result.artifact_bytes is not None and len(result.artifact_bytes) > 0
    assert result.eval_predictions is not None and len(result.eval_predictions) == 15


def test_syntax_error_rejected_without_spawning_a_subprocess(tmp_path):
    result = _run(_executor(tmp_path), "def broken(:\n  pass")
    assert result.accepted is False
    assert result.stage == "syntax"
    assert result.exit_code is None  # never even got to subprocess.run


def test_import_error_rejected(tmp_path):
    result = _run(_executor(tmp_path), "import totally_nonexistent_module_xyz\nclass Rebuilt: pass")
    assert result.accepted is False
    assert result.stage == "import"
    assert "totally_nonexistent_module_xyz" in result.error


def test_wrong_component_name_rejected_at_conformance(tmp_path):
    source = VALID_SOURCE.replace('COMPONENT_NAME = "throughput"', 'COMPONENT_NAME = "wrong_name"')
    result = _run(_executor(tmp_path), source)
    assert result.accepted is False
    assert result.stage == "conformance"
    assert "COMPONENT_NAME mismatch" in result.error


def test_wrong_output_field_rejected_at_conformance(tmp_path):
    source = VALID_SOURCE.replace('OUTPUT_FIELD = "throughput_mbps_pred"', 'OUTPUT_FIELD = "wrong_field"')
    result = _run(_executor(tmp_path), source)
    assert result.accepted is False
    assert result.stage == "conformance"
    assert "OUTPUT_FIELD mismatch" in result.error


def test_missing_abstract_methods_rejected_at_conformance(tmp_path):
    source = """
from src.dt_models.base import DTComponent
class Rebuilt(DTComponent):
    COMPONENT_NAME = "throughput"
    OUTPUT_FIELD = "throughput_mbps_pred"
"""
    result = _run(_executor(tmp_path), source)
    assert result.accepted is False
    assert result.stage == "conformance"


def test_train_raising_rejected_at_train_stage(tmp_path):
    source = VALID_SOURCE.replace(
        "    def train(self, features, targets):\n        self._model.fit(features[list(self.REQUIRED_FEATURES)], targets)\n        self._trained = True",
        '    def train(self, features, targets):\n        raise RuntimeError("boom")',
    )
    result = _run(_executor(tmp_path), source)
    assert result.accepted is False
    assert result.stage == "train"
    assert "boom" in result.error


def test_evaluate_wrong_return_type_rejected_at_evaluate_stage(tmp_path):
    source = VALID_SOURCE.replace(
        "    def evaluate(self, features, targets):\n        preds = self.predict(features)\n        rmse = float(((preds - targets) ** 2).mean() ** 0.5)\n        return {\"rmse\": rmse}",
        "    def evaluate(self, features, targets):\n        return \"not a dict\"",
    )
    result = _run(_executor(tmp_path), source)
    assert result.accepted is False
    assert result.stage == "evaluate"


def test_execution_timeout_is_enforced(tmp_path):
    source = VALID_SOURCE.replace(
        "    def train(self, features, targets):\n        self._model.fit(features[list(self.REQUIRED_FEATURES)], targets)\n        self._trained = True",
        "    def train(self, features, targets):\n        import time\n        time.sleep(30)\n",
    )
    executor = _executor(tmp_path, timeout_seconds=2)
    start = time.monotonic()
    result = _run(executor, source)
    elapsed = time.monotonic() - start

    assert result.accepted is False
    assert "timeout" in result.error.lower() or "timed out" in result.error.lower()
    assert elapsed < 10  # bounded by the 2s timeout, not the 30s sleep


def test_secret_env_vars_are_not_inherited_by_the_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TEST_SECRET", "super-secret-value-should-not-leak")
    source = VALID_SOURCE.replace(
        "    def __init__(self):\n        self._model = LinearRegression()\n        self._trained = False",
        '    def __init__(self):\n        import os\n        leaked = os.environ.get("MY_TEST_SECRET", "NOT_FOUND_IN_SANDBOX")\n        raise RuntimeError("leaked=" + leaked)',
    )
    result = _run(_executor(tmp_path), source)

    assert result.accepted is False
    assert "super-secret-value-should-not-leak" not in (result.error or "")
    assert "NOT_FOUND_IN_SANDBOX" in result.error


def test_workspace_is_cleaned_up_after_success(tmp_path):
    executor = _executor(tmp_path)
    _run(executor, VALID_SOURCE)
    remaining = list((tmp_path / "sandbox").glob("candidate-*"))
    assert remaining == []


def test_workspace_is_cleaned_up_after_rejection(tmp_path):
    executor = _executor(tmp_path)
    _run(executor, "def broken(:\n  pass")
    _run(executor, "import totally_nonexistent_module_xyz")
    remaining = list((tmp_path / "sandbox").glob("candidate-*"))
    assert remaining == []


def test_workspace_is_cleaned_up_after_timeout(tmp_path):
    source = VALID_SOURCE.replace(
        "    def train(self, features, targets):\n        self._model.fit(features[list(self.REQUIRED_FEATURES)], targets)\n        self._trained = True",
        "    def train(self, features, targets):\n        import time\n        time.sleep(30)\n",
    )
    executor = _executor(tmp_path, timeout_seconds=2)
    _run(executor, source)
    remaining = list((tmp_path / "sandbox").glob("candidate-*"))
    assert remaining == []


def test_stdout_is_truncated_to_max_output_bytes(tmp_path):
    source = VALID_SOURCE.replace(
        "    def __init__(self):\n        self._model = LinearRegression()\n        self._trained = False",
        '    def __init__(self):\n        print("x" * 5000)\n        self._model = LinearRegression()\n        self._trained = False',
    )
    executor = _executor(tmp_path, max_output_bytes=200)
    result = _run(executor, source)
    assert len(result.stdout.encode("utf-8")) <= 200 + len("...[truncated]")


def test_production_source_file_is_never_modified_by_a_candidate_run(tmp_path):
    production_file = SETTINGS.resolve_path("src/dt_models/throughput.py")
    before = production_file.read_bytes()

    _run(_executor(tmp_path), VALID_SOURCE)

    after = production_file.read_bytes()
    assert before == after
