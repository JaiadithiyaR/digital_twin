"""Unit tests for the baseline experiment-mode deterministic policies (prompt2.md Step 6)."""

from __future__ import annotations

import pytest

from src.adaptation.baseline_policies import periodic_retrain_due, rule_based_strategy


def test_rule_based_strategy_low_severity_recalibrates():
    assert rule_based_strategy(0.1, recalibrate_max_severity=0.3, regenerate_max_severity=0.7) == "recalibrate"
    assert rule_based_strategy(0.3, recalibrate_max_severity=0.3, regenerate_max_severity=0.7) == "recalibrate"


def test_rule_based_strategy_mid_severity_regenerates():
    assert rule_based_strategy(0.31, recalibrate_max_severity=0.3, regenerate_max_severity=0.7) == "regenerate"
    assert rule_based_strategy(0.7, recalibrate_max_severity=0.3, regenerate_max_severity=0.7) == "regenerate"


def test_rule_based_strategy_high_severity_expands_scope():
    assert rule_based_strategy(0.71, recalibrate_max_severity=0.3, regenerate_max_severity=0.7) == "expand_scope"
    assert rule_based_strategy(1.0, recalibrate_max_severity=0.3, regenerate_max_severity=0.7) == "expand_scope"


def test_periodic_retrain_due_fires_exactly_on_multiples_of_the_interval():
    results = [periodic_retrain_due(i, interval_cycles=5) for i in range(1, 16)]
    assert results == [i % 5 == 0 for i in range(1, 16)]
    assert sum(results) == 3  # cycles 5, 10, 15


def test_periodic_retrain_due_rejects_invalid_interval():
    with pytest.raises(ValueError):
        periodic_retrain_due(1, interval_cycles=0)
