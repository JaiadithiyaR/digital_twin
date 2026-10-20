"""Unit tests for the per-component adaptation policy (`src/fidelity/component_policy.py`), which
replaced the unified-score trigger. The policy consumes `FidelityScore_c` values only — the
fidelity formula itself is covered (unchanged) by `test_fidelity_evaluator.py`."""

from __future__ import annotations

import pytest

from src.common.config import ComponentPolicyOverride, FidelityConfig, MkMmdConfig
from src.fidelity.component_policy import ComponentAdaptationPolicy, severity_for

COMPONENTS = ["comp_a", "comp_b", "comp_c"]


def _config(threshold: float = 0.5, min_consecutive: int = 3, overrides: dict | None = None) -> FidelityConfig:
    return FidelityConfig(
        epsilon=1.0e-8, rolling_window_length=10, min_history_for_normalization=2,
        metrics=["rmse", "mae", "wasserstein", "mk_mmd"], mk_mmd=MkMmdConfig(kernel="rbf", gamma=None),
        unified_weights=None, unified_min_components=2, trigger_threshold=threshold,
        trigger_min_consecutive_evaluations=min_consecutive, component_policies=overrides or {},
    )


def _fired(policy: ComponentAdaptationPolicy, scores: dict) -> list[str]:
    return [t.component for t in policy.triggers(policy.evaluate(COMPONENTS, scores))]


# --- the illustrative A/B/C example: only the low component is selected ------------------------


def test_only_the_component_below_its_threshold_is_selected_for_adaptation():
    policy = ComponentAdaptationPolicy(_config(threshold=0.85, min_consecutive=1))
    statuses = policy.evaluate(COMPONENTS, {"comp_a": 0.92, "comp_b": 0.68, "comp_c": 0.89})

    assert {s.component: s.state for s in statuses} == {"comp_a": "ok", "comp_b": "adapt", "comp_c": "ok"}
    triggers = policy.triggers(statuses)
    assert [t.component for t in triggers] == ["comp_b"]
    assert triggers[0].trigger_type == "fidelity_degradation"
    assert triggers[0].severity == pytest.approx((0.85 - 0.68) / 0.85)
    assert triggers[0].metadata["policy"] == "per_component"
    assert triggers[0].metadata["threshold"] == 0.85


def test_all_components_acceptable_means_no_adaptation():
    policy = ComponentAdaptationPolicy(_config(threshold=0.5, min_consecutive=1))
    assert _fired(policy, {"comp_a": 0.9, "comp_b": 0.5, "comp_c": 0.7}) == []  # 0.5 == threshold is NOT below


def test_multiple_low_components_each_get_an_independent_trigger():
    policy = ComponentAdaptationPolicy(_config(threshold=0.5, min_consecutive=1))
    triggers = policy.triggers(policy.evaluate(COMPONENTS, {"comp_a": 0.1, "comp_b": 0.9, "comp_c": 0.3}))
    assert {t.component: t.severity for t in triggers} == {
        "comp_a": pytest.approx(0.8), "comp_c": pytest.approx(0.4)
    }


def test_a_healthy_component_is_never_adapted_because_another_is_degraded():
    """The old unified policy could fire because ONE bad component dragged the mean down; now a
    healthy component stays put no matter how bad the others are."""
    policy = ComponentAdaptationPolicy(_config(threshold=0.5, min_consecutive=1))
    assert _fired(policy, {"comp_a": -5.0, "comp_b": 0.95, "comp_c": 0.95}) == ["comp_a"]


def test_a_component_can_adapt_even_while_the_unified_mean_is_healthy():
    """No global threshold has to be crossed first: mean of (0.95, 0.95, 0.4) = 0.77 > 0.5."""
    policy = ComponentAdaptationPolicy(_config(threshold=0.5, min_consecutive=1))
    assert _fired(policy, {"comp_a": 0.95, "comp_b": 0.95, "comp_c": 0.4}) == ["comp_c"]


# --- debounce semantics (identical to the former trigger, now per component) -----------------


def test_debounce_is_independent_per_component():
    policy = ComponentAdaptationPolicy(_config(threshold=0.5, min_consecutive=3))
    assert _fired(policy, {"comp_a": 0.2, "comp_b": 0.9, "comp_c": 0.9}) == []
    assert _fired(policy, {"comp_a": 0.2, "comp_b": 0.2, "comp_c": 0.9}) == []
    assert _fired(policy, {"comp_a": 0.2, "comp_b": 0.2, "comp_c": 0.9}) == ["comp_a"]  # a reaches 3, b only 2
    assert _fired(policy, {"comp_a": 0.2, "comp_b": 0.2, "comp_c": 0.9}) == ["comp_b"]  # b reaches 3; a reset after firing


def test_recovery_resets_the_counter_and_firing_resets_it_too():
    policy = ComponentAdaptationPolicy(_config(threshold=0.5, min_consecutive=2))
    assert _fired(policy, {"comp_a": 0.2}) == []
    assert _fired(policy, {"comp_a": 0.9}) == []  # recovered -> reset
    assert _fired(policy, {"comp_a": 0.2}) == []  # back to 1
    assert _fired(policy, {"comp_a": 0.2}) == ["comp_a"]  # 2 -> fires
    assert policy.consecutive_below("comp_a") == 0  # reset immediately after firing
    assert _fired(policy, {"comp_a": 0.2}) == []  # still degraded: must re-accumulate
    assert _fired(policy, {"comp_a": 0.2}) == ["comp_a"]  # remains flagged -> fires again


# --- missing / stale / insufficient data never produce a decision ------------------------------


def test_insufficient_history_never_fires_and_resets_the_counter():
    policy = ComponentAdaptationPolicy(_config(threshold=0.5, min_consecutive=2))
    assert _fired(policy, {"comp_a": 0.2}) == []
    statuses = policy.evaluate(COMPONENTS, {"comp_a": None})
    assert statuses[0].state == "insufficient_history" and statuses[0].fidelity_score is None
    assert policy.triggers(statuses) == []
    assert _fired(policy, {"comp_a": 0.2}) == []  # counter was reset, needs 2 fresh cycles again


def test_component_without_a_fresh_measurement_is_unavailable_not_good_or_bad():
    policy = ComponentAdaptationPolicy(_config(threshold=0.5, min_consecutive=1))
    statuses = {s.component: s for s in policy.evaluate(COMPONENTS, {"comp_a": 0.9})}
    assert statuses["comp_b"].state == "unavailable"
    assert statuses["comp_b"].fidelity_score is None and statuses["comp_b"].fidelity_gap is None
    assert policy.triggers(statuses.values()) == []


def test_stale_low_score_from_an_earlier_cycle_cannot_complete_a_debounce_run():
    policy = ComponentAdaptationPolicy(_config(threshold=0.5, min_consecutive=2))
    assert _fired(policy, {"comp_a": 0.2}) == []
    assert _fired(policy, {}) == []  # comp_a not measured this cycle -> reset, not "still low"
    assert _fired(policy, {"comp_a": 0.2}) == []


# --- per-component thresholds/policies -------------------------------------------------------


def test_thresholds_and_debounce_can_differ_between_components():
    overrides = {
        "comp_a": ComponentPolicyOverride(threshold=0.8),
        "comp_b": ComponentPolicyOverride(min_consecutive_evaluations=2),
    }
    policy = ComponentAdaptationPolicy(_config(threshold=0.5, min_consecutive=1, overrides=overrides))
    assert policy.policy_for("comp_a").threshold == 0.8 and policy.policy_for("comp_a").min_consecutive_evaluations == 1
    assert policy.policy_for("comp_b").threshold == 0.5 and policy.policy_for("comp_b").min_consecutive_evaluations == 2
    assert policy.policy_for("comp_c").threshold == 0.5  # unlisted -> defaults
    # 0.7 is fine under the default 0.5 but below comp_a's own 0.8; comp_b needs 2 cycles.
    scores = {"comp_a": 0.7, "comp_b": 0.1, "comp_c": 0.7}
    assert _fired(policy, scores) == ["comp_a"]
    assert _fired(policy, scores) == ["comp_a", "comp_b"]


def test_override_validation_rejects_nonsensical_values():
    with pytest.raises(ValueError):
        ComponentPolicyOverride(threshold=0.0)
    with pytest.raises(ValueError):
        ComponentPolicyOverride(min_consecutive_evaluations=0)
    with pytest.raises(ValueError):
        ComponentPolicyOverride(threshold=0.5, unknown_field=1)


def test_newly_added_components_are_covered_without_any_registration():
    policy = ComponentAdaptationPolicy(_config(threshold=0.5, min_consecutive=1))
    statuses = policy.evaluate(["expand_scope_new_component"], {"expand_scope_new_component": 0.1})
    assert statuses[0].state == "adapt"


# --- severity: same clip((T - F)/T, 0, 1) expression the former trigger used -------------------


def test_severity_formula_and_clipping_are_unchanged():
    assert severity_for(0.2, 0.5) == pytest.approx(0.6)
    assert severity_for(-3.0, 0.5) == 1.0
    assert severity_for(0.6, 0.5) == 0.0
