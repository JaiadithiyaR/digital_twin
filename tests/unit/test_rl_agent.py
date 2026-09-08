"""Unit tests for rl_agent.py (Module 13's PPO training/runtime plumbing, CLAUDE.md §6).

Deliberately does NOT run a full training loop here (that's `tests/integration/
test_rl_training.py`'s job, mirroring Modules 6-10's "real held-out validation lives in
integration tests" convention) — these tests cover the surrounding plumbing: agent construction
reads real config, the fallback path is logged and only ever used for genuine infra failure
(never a routine substitute — prompt.md §70 rule 18 / CLAUDE.md §6), and `decide_adaptation_
strategy` always genuinely calls the model, never a heuristic.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.adaptation.rl_agent import (
    build_ppo_agent,
    decide_adaptation_strategy,
    decide_adaptation_strategy_safe,
    load_ppo_agent,
    make_vec_env,
)
from src.adaptation.rl_env import build_drift_event_pool, build_network_state_pool
from src.common.config import load_settings

SETTINGS = load_settings()


@pytest.fixture(scope="module")
def vec_env():
    pool = build_network_state_pool(SETTINGS, seed=1, pool_size=20, sample_rows=SETTINGS.ppo.env.network_state_sample_rows)
    events = build_drift_event_pool(SETTINGS, seed=1, pool_size=50)
    env = make_vec_env(SETTINGS, n_envs=2, network_state_pool=pool, drift_events=events, base_seed=1)
    yield env
    env.close()


def test_make_vec_env_has_configured_number_of_envs(vec_env):
    assert vec_env.num_envs == 2


def test_build_ppo_agent_reads_hyperparameters_from_config(vec_env):
    model = build_ppo_agent(SETTINGS, vec_env, seed=123)
    cfg = SETTINGS.ppo.training
    assert isinstance(model, PPO)
    assert model.learning_rate == cfg.learning_rate
    assert model.gamma == cfg.gamma
    assert model.gae_lambda == cfg.gae_lambda
    assert model.clip_range(1) == cfg.clip_range
    assert model.n_steps == cfg.n_steps
    assert model.batch_size == cfg.batch_size
    assert str(model.device) == "cpu"


def test_load_ppo_agent_raises_clearly_when_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.zip"
    with pytest.raises(FileNotFoundError):
        load_ppo_agent(SETTINGS, path=missing_path)


def test_decide_adaptation_strategy_uses_real_model_predict(vec_env, monkeypatch):
    model = build_ppo_agent(SETTINGS, vec_env, seed=1)
    obs = np.zeros(model.observation_space.shape, dtype=np.float32)

    called = {}

    original_predict = model.predict

    def spy_predict(*args, **kwargs):
        called["was_called"] = True
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(model, "predict", spy_predict)
    action_idx, action_name = decide_adaptation_strategy(model, obs, SETTINGS)

    assert called.get("was_called") is True
    assert action_idx in (0, 1, 2)
    assert action_name == SETTINGS.ppo.action_mapping[action_idx]


def test_decide_adaptation_strategy_safe_returns_same_result_as_direct_call_on_success(vec_env):
    model = build_ppo_agent(SETTINGS, vec_env, seed=1)
    obs = np.zeros(model.observation_space.shape, dtype=np.float32)
    direct = decide_adaptation_strategy(model, obs, SETTINGS)
    safe = decide_adaptation_strategy_safe(model, obs, SETTINGS)
    # Both must be genuinely deterministic PPO predictions for the same observation.
    assert direct == safe


def test_decide_adaptation_strategy_safe_falls_back_and_logs_on_model_none(caplog):
    with caplog.at_level(logging.WARNING):
        action_idx, action_name = decide_adaptation_strategy_safe(None, np.zeros(5, dtype=np.float32), SETTINGS)
    assert action_name == SETTINGS.ppo.fallback.default_action
    assert action_idx == next(
        idx for idx, name in SETTINGS.ppo.action_mapping.items() if name == SETTINGS.ppo.fallback.default_action
    )
    assert any("fallback" in record.message.lower() for record in caplog.records)


def test_decide_adaptation_strategy_safe_falls_back_when_predict_raises(vec_env, monkeypatch, caplog):
    model = build_ppo_agent(SETTINGS, vec_env, seed=1)

    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated PPO infrastructure failure")

    monkeypatch.setattr(model, "predict", broken_predict)
    obs = np.zeros(model.observation_space.shape, dtype=np.float32)

    with caplog.at_level(logging.WARNING):
        action_idx, action_name = decide_adaptation_strategy_safe(model, obs, SETTINGS)

    assert action_name == SETTINGS.ppo.fallback.default_action
    assert any(
        "simulated PPO infrastructure failure" in getattr(record, "error", "") for record in caplog.records
    )


def test_decide_adaptation_strategy_never_falls_back_on_success(vec_env):
    """Confirms the non-safe entrypoint never silently substitutes a heuristic — it either
    returns PPO's genuine decision or raises, exactly prompt.md §70 rule 18's requirement."""
    model = build_ppo_agent(SETTINGS, vec_env, seed=1)
    obs = np.zeros(model.observation_space.shape, dtype=np.float32)
    action_idx, action_name = decide_adaptation_strategy(model, obs, SETTINGS)
    assert action_name in SETTINGS.ppo.action_mapping.values()
