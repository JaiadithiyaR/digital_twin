"""Unit tests for AdaptationEnv (Module 13's Gymnasium environment, prompt.md §18-§21).

Proves: proper Gym API compliance, correctly-dimensioned/shaped observation and action spaces,
reproducibility given a seed, and — most importantly — that the simulated adaptation-outcome
dynamics actually encode a non-trivial, state-dependent decision problem (low severity favors
recalibrate; high severity favors regenerate; repeated failures favor expand_scope) rather than
one action trivially dominating regardless of context, which is what makes this a genuine RL
problem for PPO to learn rather than a decorative one.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.adaptation.rl_env import (
    NETWORK_STATE_FEATURES,
    AdaptationEnv,
    build_drift_event_pool,
    build_network_state_pool,
)
from src.common.config import load_settings
from src.drift.schema import DriftEvent

SETTINGS = load_settings()


def _network_pool(pool_size=30, seed=1):
    return build_network_state_pool(
        SETTINGS, seed=seed, pool_size=pool_size, sample_rows=SETTINGS.ppo.env.network_state_sample_rows
    )


def _drift_pool(pool_size=100, seed=1):
    return build_drift_event_pool(SETTINGS, seed=seed, pool_size=pool_size)


def _fixed_events(component: str, severity: float, n: int = 50) -> list[DriftEvent]:
    now = datetime.now(UTC)
    return [
        DriftEvent(component=component, severity=severity, timestamp=now, metadata={}, source="MOCK", received_at=now)
        for _ in range(n)
    ]


@pytest.fixture(scope="module")
def network_pool():
    return _network_pool()


@pytest.fixture(scope="module")
def drift_pool():
    return _drift_pool()


def _run_episode(env, action_fn, seed=None):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = action_fn(steps, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    return total_reward, steps


def _mean_reward(env, action_fn, n_episodes=60, seed=500):
    totals = []
    for i in range(n_episodes):
        total, _ = _run_episode(env, action_fn, seed=seed + i)
        totals.append(total)
    return float(np.mean(totals))


def test_gymnasium_check_env_passes(network_pool, drift_pool):
    env = AdaptationEnv(SETTINGS, drift_events=drift_pool, network_state_pool=network_pool, seed=1)
    check_env(env, skip_render_check=True)


def test_action_space_is_discrete_3(network_pool, drift_pool):
    env = AdaptationEnv(SETTINGS, drift_events=drift_pool, network_state_pool=network_pool, seed=1)
    assert env.action_space.n == 3


def test_observation_space_dimensionality_matches_spec(network_pool, drift_pool):
    env = AdaptationEnv(SETTINGS, drift_events=drift_pool, network_state_pool=network_pool, seed=1)
    n_components = len(SETTINGS.drift.valid_components)
    expected_dim = (
        n_components  # fidelity vector
        + n_components  # affected-component one-hot
        + 1  # severity
        + 3  # previous action one-hot
        + 1  # previous reward
        + len(NETWORK_STATE_FEATURES)  # network state
    )
    assert env.observation_space.shape == (expected_dim,)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (expected_dim,)
    assert env.observation_space.contains(obs)


def test_reset_returns_affected_component_from_valid_scope(network_pool, drift_pool):
    env = AdaptationEnv(SETTINGS, drift_events=drift_pool, network_state_pool=network_pool, seed=1)
    for i in range(20):
        obs, info = env.reset(seed=i)
        assert info["affected_component"] in SETTINGS.drift.valid_components
        lo, hi = SETTINGS.drift.severity_range
        assert lo <= info["severity"] <= hi


def test_step_rejects_invalid_action(network_pool, drift_pool):
    env = AdaptationEnv(SETTINGS, drift_events=drift_pool, network_state_pool=network_pool, seed=1)
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(3)


def test_same_seed_reproducible_trajectory(network_pool, drift_pool):
    env_a = AdaptationEnv(SETTINGS, drift_events=drift_pool, network_state_pool=network_pool, seed=7)
    env_b = AdaptationEnv(SETTINGS, drift_events=drift_pool, network_state_pool=network_pool, seed=7)
    obs_a, info_a = env_a.reset(seed=42)
    obs_b, info_b = env_b.reset(seed=42)
    assert np.allclose(obs_a, obs_b)
    assert info_a == info_b
    for action in (0, 1, 2, 0):
        obs_a, r_a, t_a, tr_a, info_a = env_a.step(action)
        obs_b, r_b, t_b, tr_b, info_b = env_b.step(action)
        assert np.allclose(obs_a, obs_b)
        assert r_a == pytest.approx(r_b)
        assert (t_a, tr_a) == (t_b, tr_b)
        if t_a or tr_a:
            break


def test_episode_terminates_when_fidelity_resolved(network_pool):
    events = _fixed_events("throughput", severity=0.1)
    env = AdaptationEnv(SETTINGS, drift_events=events, network_state_pool=network_pool, seed=1)
    env.reset(seed=1)
    _, _, terminated, truncated, info = env.step(0)  # recalibrate: cheap and effective at low severity
    assert terminated
    assert info["new_fidelity"] >= SETTINGS.ppo.env.resolved_fidelity_threshold


def test_episode_truncates_after_max_attempts_when_unresolved(network_pool):
    events = _fixed_events("throughput", severity=1.0)
    env = AdaptationEnv(SETTINGS, drift_events=events, network_state_pool=network_pool, seed=1)
    env.reset(seed=1)
    steps = 0
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(0)  # recalibrate alone should not resolve max severity
        steps += 1
        done = terminated or truncated
    assert steps <= SETTINGS.ppo.env.max_attempts_per_incident


def test_low_severity_favors_recalibrate_over_regenerate(network_pool):
    events = _fixed_events("throughput", severity=0.1)
    env = AdaptationEnv(SETTINGS, drift_events=events, network_state_pool=network_pool, seed=1)
    recal_mean = _mean_reward(env, lambda step, info: 0)
    regen_mean = _mean_reward(env, lambda step, info: 1)
    assert recal_mean > regen_mean


def test_high_severity_favors_regenerate_over_recalibrate(network_pool):
    events = _fixed_events("throughput", severity=0.9)
    env = AdaptationEnv(SETTINGS, drift_events=events, network_state_pool=network_pool, seed=1)
    recal_mean = _mean_reward(env, lambda step, info: 0)
    regen_mean = _mean_reward(env, lambda step, info: 1)
    assert regen_mean > recal_mean


def test_expand_scope_only_pays_off_after_repeated_failed_attempts(network_pool):
    events = _fixed_events("throughput", severity=0.95, n=80)
    env = AdaptationEnv(SETTINGS, drift_events=events, network_state_pool=network_pool, seed=1)
    min_prior = SETTINGS.ppo.env.expand_scope_min_prior_attempts

    def premature_expand(step, info):
        return 2  # expand_scope on the very first attempt

    def escalate_then_expand(step, info):
        return 2 if step >= min_prior else 0  # recalibrate min_prior times, then expand_scope

    premature_mean = _mean_reward(env, premature_expand, n_episodes=80)
    escalate_mean = _mean_reward(env, escalate_then_expand, n_episodes=80)
    assert escalate_mean > premature_mean


def test_build_network_state_pool_shape_and_range():
    pool = _network_pool(pool_size=25, seed=3)
    assert pool.shape == (25, len(NETWORK_STATE_FEATURES))
    assert np.all(pool >= 0.0) and np.all(pool <= 1.0)


def test_build_drift_event_pool_matches_config_scope():
    events = _drift_pool(pool_size=40, seed=3)
    assert len(events) == 40
    assert all(e.component in SETTINGS.drift.valid_components for e in events)
    lo, hi = SETTINGS.drift.severity_range
    assert all(lo <= e.severity <= hi for e in events)
    assert all(e.source == "MOCK" for e in events)
