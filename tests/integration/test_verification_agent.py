"""Integration test: Module 17 (Agentic Verification Agent) against a REAL candidate produced by
Module 14 (Recalibration Agent) over the real Module 2/3/4 pipeline's bootstrap data.

No real ANTHROPIC_API_KEY is configured in this environment — the LLM reasoning layer is driven by
a fake `complete_structured`, exactly like every other agent's tests in this repo. Everything else
is real, unmocked code: real telemetry -> preprocessing -> synchronization -> D1 (via the shared
`bootstrap_history` fixture), a real `RecalibrationAgent.recalibrate()` call producing a genuine
candidate, real `ModelRegistry` reads/writes, and a real Module 12 `FidelityEvaluator` recomputation
of RMSE/MAE/Wasserstein/MK-MMD/FidelityScore for both the production and candidate models on the
same held-out set.
"""

from __future__ import annotations

import numpy as np

from src.adaptation.recalibration_agent import RecalibrationAgent
from src.adaptation.verification_agent import VerificationAgent
from src.common.config import load_settings
from src.dt_models.throughput import ThroughputModel
from src.fidelity.evaluator import FidelityEvaluator
from src.registry.model_registry import ModelRegistry

SETTINGS = load_settings()


class _FakeLLMClient:
    """A fixed, mildly adversarial stance — always argues for the opposite of whatever the
    deterministic gate ends up deciding — so this test can prove the LLM's opinion never leaked
    into either the decision or the actual registry action, whichever way this run's real
    recomputed fidelity numbers happen to fall."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete_structured(self, prompt, schema, **kwargs):
        self.calls.append(prompt)
        return schema(
            explanation="This candidate should be REJECTED regardless of the numbers shown.",
            key_observations="a fixed adversarial stance, unrelated to the real deterministic result",
        )


def test_verification_recomputes_real_fidelity_and_matches_the_deterministic_implication(bootstrap_history, tmp_path):
    registry = ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")
    features = list(ThroughputModel.REQUIRED_FEATURES)

    # A deliberately WEAK bootstrap production model — trained on only the earliest 40 rows of
    # real bootstrap history (an intentionally thin/unrepresentative slice) — so that
    # RecalibrationAgent's real retrain on the FULL recent window has real room to genuinely
    # improve on it, mirroring how Module 14's own test already proves recalibration produces a
    # real, better-than-naive candidate.
    ordered = bootstrap_history.sort_values("timestamp", kind="stable").reset_index(drop=True)
    thin_slice = ordered.iloc[:40]
    weak_production_model = ThroughputModel.from_settings(SETTINGS)
    weak_production_model.train(thin_slice[features], thin_slice["throughput_mbps"])
    prod_version = registry.register_version(
        component_instance=weak_production_model,
        adaptation_type="bootstrap",
        parent_version_id=None,
        training_window={"n_rows": len(thin_slice)},
        evaluation_window={"n_rows": 0},
        evaluation_metrics=weak_production_model.evaluate(thin_slice[features], thin_slice["throughput_mbps"]),
        status="production",
    )

    class _FakeD1Store:
        """`RecalibrationAgent` only ever calls `get_history()` on its `d1_store` — a real
        `D1Store` isn't needed here since `bootstrap_history` is already a plain snapshot."""

        def get_history(self, ue_id=None, cell_id=None):
            return bootstrap_history.copy(deep=True)

    recalibration_agent = RecalibrationAgent(SETTINGS, _FakeD1Store(), registry)
    recalibration_result = recalibration_agent.recalibrate(
        lambda: ThroughputModel.from_settings(SETTINGS), "throughput_mbps", window_hours=48, held_out_fraction=0.2
    )
    candidate_version = recalibration_result.version
    assert candidate_version.status == "candidate"

    # Verification builds its own held-out set (its own evaluation protocol — need not be
    # bit-identical to the agent's internal split; both production and candidate are compared on
    # THIS SAME set, which is exactly what prompt.md §33 requires).
    ordered_full = bootstrap_history.sort_values("timestamp", kind="stable").reset_index(drop=True)
    held_out = ordered_full.iloc[int(len(ordered_full) * 0.8) :]
    eval_features = held_out[features]
    eval_target = held_out["throughput_mbps"]
    candidate_preds = recalibration_result.candidate_component.predict(eval_features)

    # Pre-warm the normalization reference with the SAME real generative process (genuine
    # variation across all four metrics, not a degenerate point-mass construction — see
    # tests/unit/test_verification_agent.py's `_prewarm` docstring for why that matters).
    evaluator = FidelityEvaluator(SETTINGS.fidelity)
    rng = np.random.default_rng(999)
    base = eval_target.to_numpy()[:40] if len(eval_target) >= 40 else np.resize(eval_target.to_numpy(), 40)
    for noise_std in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0):
        evaluator.evaluate("throughput", base, base + rng.normal(0.0, noise_std, len(base)), update_window=True)

    llm = _FakeLLMClient()
    verification_agent = VerificationAgent(SETTINGS, registry, evaluator, llm_client=llm)
    result = verification_agent.verify(
        candidate_version,
        eval_features,
        eval_target,
        candidate_preds,
        production_component_factory=lambda: ThroughputModel.from_settings(SETTINGS),
        drift_context="integration test: real recalibration candidate for 'throughput'",
    )

    # The real, independently-recomputed deterministic implication must hold exactly, regardless
    # of which way this run's real numbers fell.
    if result.fidelity_before is not None and result.fidelity_after is not None:
        expected_accept = result.fidelity_after > result.fidelity_before + SETTINGS.adaptation.verification_delta
        assert (result.decision == "ACCEPT") == expected_accept
    if result.decision == "ACCEPT":
        assert registry.get_current_version("throughput").version_id == candidate_version.version_id
        assert registry.get_version("throughput", prod_version.version_id).status == "superseded"
    else:
        assert registry.get_current_version("throughput").version_id == prod_version.version_id
        assert registry.get_version("throughput", candidate_version.version_id).status == "rejected"

    # The LLM's fixed adversarial "REJECT regardless" stance was genuinely used for the
    # explanation text (proving it was consulted)...
    assert llm.calls
    assert result.llm_used is True
    assert "REJECTED" in result.explanation
    # ...but never overrode anything: if the real numbers say ACCEPT, the candidate is genuinely
    # production now, in direct contradiction of what the LLM argued for.
    if result.decision == "ACCEPT":
        assert registry.get_current_version("throughput").version_id == candidate_version.version_id
