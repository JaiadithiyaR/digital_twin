"""phase13 §8 item 6 — the required `tests/e2e/` smoke test: replay a short slice of run 7
(never run 8, reserved exclusively for the real `baselines/` TEST comparison) through the WHOLE
lifecycle in `rule_based` mode, with a mocked LLM client — no real API calls, no cost — and
assert the phase13 §6 outputs (`metrics_timeline.csv`, `adaptation_events.jsonl`, `report.md`)
are genuinely produced.

Exercises the real `baselines.run_all.run_mode()` harness end to end: real bootstrap, real OAI
replay (run 7), real prediction+fidelity cycles, real `rule_based_strategy()` decisions, the real
Recalibration/Regeneration agents, the real sandbox subprocess, and the real deterministic
Verification gate — only the LLM transport itself is faked (regenerate's codegen call and
verification's optional explanation call; `regenerate_max_severity` is widened to 1.0 for this
test so a high-severity trigger can never reach expand_scope, keeping the fake client's surface
to exactly the two schemas regenerate/verification actually use).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from baselines import run_all  # noqa: E402
from baselines._lib import build_static_prefix  # noqa: E402
from src.adaptation.regeneration_agent import _GeneratedComponentCode  # noqa: E402
from src.adaptation.verification_agent import _LLMVerificationReasoning  # noqa: E402
from src.common.config import load_secrets, load_settings  # noqa: E402

SETTINGS = load_settings()
SECRETS = load_secrets()

_GENERIC_CANDIDATE_TEMPLATE = '''
import pandas as pd
from src.dt_models.base import DTComponent

class {class_name}(DTComponent):
    COMPONENT_NAME = {component_name!r}
    DEPENDENCIES = ()
    REQUIRED_FEATURES = ("dl_curr_tbs_avg_lag1",)
    OUTPUT_FIELD = {output_field!r}

    def __init__(self):
        self._trained = False

    @property
    def is_trained(self):
        return self._trained

    def train(self, features, targets):
        self._trained = True

    def predict(self, features):
        return pd.Series([0.0] * len(features), index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features, targets):
        return {{"rmse": 0.0}}

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("dummy artifact")

    def load(self, path):
        self._trained = True
'''

_COMPONENT_RE = re.compile(r"MUST equal exactly '([a-z_]+)'\), ")
_OUTPUT_FIELD_RE = re.compile(r"`OUTPUT_FIELD` \(class attribute, MUST equal exactly '([a-z_]+)'")


class _FakeLLMClient:
    """Stands in for `GoogleClient` for exactly the two roles `rule_based` mode needs an LLM for
    (codegen, explanation) — never a real network call, never a real cost. Discriminates purely
    on `schema`, the same established pattern `tests/integration/test_hot_swap.py` already uses."""

    def complete_structured(self, prompt, schema, **kwargs):
        if schema is _GeneratedComponentCode:
            component_match = re.search(r"MUST equal exactly '([a-z_]+)'\),", prompt)
            output_field_match = re.search(r"OUTPUT_FIELD.{0,40}?MUST equal exactly '([a-z_]+)'", prompt)
            component_name = component_match.group(1) if component_match else "dl_bler"
            output_field = output_field_match.group(1) if output_field_match else f"{component_name}_pred"
            class_name = "FakeRebuilt" + "".join(p.capitalize() for p in component_name.split("_"))
            source = _GENERIC_CANDIDATE_TEMPLATE.format(class_name=class_name, component_name=component_name, output_field=output_field)
            return _GeneratedComponentCode(class_name=class_name, source_code=source, reasoning="e2e smoke test fake candidate")
        if schema is _LLMVerificationReasoning:
            return _LLMVerificationReasoning(explanation="e2e smoke test fake explanation", key_observations="none - fake LLM client")
        raise AssertionError(f"rule_based e2e test's fake LLM client was asked for an unexpected schema: {schema}")

    def complete_safe(self, *args, **kwargs):
        return None  # lifecycle_agent's optional report prose — not exercised by run_all.py, fine to degrade

    def complete_structured_safe(self, *args, **kwargs):
        try:
            return self.complete_structured(*args, **kwargs)
        except Exception:  # noqa: BLE001 - mirrors GoogleClient's own *_safe graceful-degradation contract
            return None

    def start_cache(self, *args, **kwargs):
        return False  # never a real cache — nothing to start against a fake client

    def end_cache(self, *args, **kwargs):
        return None


def test_rule_based_mode_over_a_short_run7_slice_produces_the_required_outputs(tmp_path):
    import pandas as pd

    import baselines.run_all as run_all_module

    train_df = pd.read_parquet(SETTINGS.resolve_path("data/processed/oai/train.parquet"))
    static_prefix = build_static_prefix(SETTINGS)

    original_results_dir = run_all_module.RESULTS_DIR
    run_all_module.RESULTS_DIR = tmp_path
    try:
        mode_cfg = {
            "mode": "rule_based", "uses_llm": True, "seeds": [1],
            "recalibrate_max_severity": 0.3, "regenerate_max_severity": 1.0,  # never reach expand_scope — see module docstring
        }
        summary = run_all.run_mode(
            "rule_based", mode_cfg, seed=1, n_seeds=1, settings=SETTINGS, secrets=SECRETS,
            train_df=train_df, static_prefix=static_prefix, cycles=5, rag_kb=None,
            replay_run_ids=[7], fake_llm_client=_FakeLLMClient(),
        )
    finally:
        run_all_module.RESULTS_DIR = original_results_dir

    out_dir = Path(summary["out_dir"])
    assert summary["run_ids"] == [7], "must replay run 7, never run 8 (reserved for the real TEST comparison)"
    assert (out_dir / "metrics_timeline.csv").exists()
    assert (out_dir / "adaptation_events.jsonl").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "run_summary.json").exists()

    timeline_text = (out_dir / "metrics_timeline.csv").read_text(encoding="utf-8")
    assert "component" in timeline_text and "F_c" in timeline_text  # a real header, not an empty stub

    report_text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "mode `rule_based`" in report_text
    assert "Adaptation Events" in report_text

    # llm_calls.jsonl is deliberately NOT asserted non-empty here — the fake client never goes
    # through MeteredGoogleClient (no real GoogleClient was ever constructed), which is exactly
    # the point: zero real API calls, zero cost, confirmed structurally rather than by mocking a
    # network layer that was never reached in the first place.
    assert not (out_dir / "llm_calls.jsonl").exists()
