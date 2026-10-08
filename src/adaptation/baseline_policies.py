"""Baseline experiment-mode policies (prompt2.md Step 6): deterministic stand-ins for Module 13
(the Decision & Root-Cause Analysis Agent) used only by `scripts/run_baselines.py`'s
`rule_based`/`periodic_retrain` modes — never used by the real system (`src/main.py`), which
always makes a genuine LLM call. Kept here, not in `decision_agent.py`, so the real agent's own
code path can never accidentally be bypassed by a baseline's deterministic rule.
"""

from __future__ import annotations

from typing import Literal

Strategy = Literal["recalibrate", "regenerate", "expand_scope"]


def rule_based_strategy(severity: float, recalibrate_max_severity: float, regenerate_max_severity: float) -> Strategy:
    """Deterministic severity-threshold rule (prompt2.md: "rule_based decision (severity
    thresholds from config)") — the `rule_based` baseline's entire decision logic, never an LLM
    call. Maps onto the SAME closed 3-strategy enum Module 13 itself is constrained to."""
    if severity <= recalibrate_max_severity:
        return "recalibrate"
    if severity <= regenerate_max_severity:
        return "regenerate"
    return "expand_scope"


def periodic_retrain_due(cycle_index: int, interval_cycles: int) -> bool:
    """True exactly every `interval_cycles` cycles — the `periodic_retrain` baseline's fixed
    schedule (prompt2.md: "periodic_retrain (fixed schedule, config-driven)"), never tied to any
    real trigger/severity. `cycle_index` is 1-based; interval_cycles must be >= 1."""
    if interval_cycles < 1:
        raise ValueError(f"interval_cycles must be >= 1, got {interval_cycles}")
    return cycle_index % interval_cycles == 0
