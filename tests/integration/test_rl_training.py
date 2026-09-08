"""Integration test: a REAL (small) end-to-end PPO training run through the actual Gymnasium
environment (Module 13, prompt.md §21 "train PPO through the actual Gymnasium environment").

Mirrors the "REAL held-out validation, not placeholder numbers" discipline established for
Modules 6-10 (see CLAUDE.md's Module 6/7/8/9/10 entries) — this genuinely calls
`stable_baselines3.PPO.learn()`, genuinely saves/reloads the policy, and checks a real directional
learning signal against a real held-out scenario, rather than asserting against a hand-picked
constant. Kept deliberately small (a few thousand timesteps) so it runs in the fast test suite;
`scripts/train_ppo.py` is the full "small run given 16GB RAM" training + reporting deliverable.
"""

from __future__ import annotations

import numpy as np

from src.adaptation.rl_agent import decide_adaptation_strategy, load_ppo_agent, train_ppo
from src.adaptation.rl_env import AdaptationEnv, build_drift_event_pool, build_network_state_pool
from src.common.config import load_settings

SETTINGS = load_settings()


def test_ppo_trains_saves_reloads_and_predicts_valid_actions(tmp_path):
    save_path = tmp_path / "ppo_policy.zip"
    model, reward_logger, wall_clock_seconds = train_ppo(
        SETTINGS,
        total_timesteps=3000,
        n_envs=2,
        seed=11,
        save_path=save_path,
        network_pool_seed=1,
        drift_pool_seed=1,
    )

    assert save_path.exists()
    assert wall_clock_seconds > 0.0
    assert len(reward_logger.episode_rewards) > 10  # genuinely completed many episodes, not zero

    reloaded = load_ppo_agent(SETTINGS, path=save_path)
    obs = np.zeros(reloaded.observation_space.shape, dtype=np.float32)
    for _ in range(5):
        action_idx, action_name = decide_adaptation_strategy(reloaded, obs, SETTINGS)
        assert action_idx in (0, 1, 2)
        assert action_name in SETTINGS.ppo.action_mapping.values()


def test_trained_policy_beats_a_fixed_always_recalibrate_policy_on_a_held_out_high_severity_scenario():
    """A real directional-learning check: on a HELD-OUT high-severity scenario (a different
    drift-pool seed than training used), the trained policy's mean episode reward should beat a
    fixed always-recalibrate policy — recalibrate alone is a poor strategy at high severity (see
    `tests/unit/test_rl_env.py::test_high_severity_favors_regenerate_over_recalibrate`), so a
    policy that hasn't learned anything about severity should not reliably beat it."""
    model, _, _ = train_ppo(
        SETTINGS,
        total_timesteps=6000,
        n_envs=2,
        seed=21,
        save_path=None,
        network_pool_seed=1,
        drift_pool_seed=1,  # TRAIN pool
    )
    # HELD-OUT: a different seed than training used, never seen during learning.
    held_out_network_pool = build_network_state_pool(
        SETTINGS, seed=999, pool_size=30, sample_rows=SETTINGS.ppo.env.network_state_sample_rows
    )
    held_out_drift_events = build_drift_event_pool(SETTINGS, seed=999, pool_size=100)
    high_severity_events = [e.model_copy(update={"severity": 0.9}) for e in held_out_drift_events]

    def rollout(action_fn, n_episodes=40, seed=7777):
        env = AdaptationEnv(
            SETTINGS, drift_events=high_severity_events, network_state_pool=held_out_network_pool, seed=seed
        )
        totals = []
        for i in range(n_episodes):
            obs, info = env.reset(seed=seed + i)
            done = False
            total = 0.0
            while not done:
                action = action_fn(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                done = terminated or truncated
            totals.append(total)
        return float(np.mean(totals))

    trained_mean = rollout(lambda obs: decide_adaptation_strategy(model, obs, SETTINGS)[0])
    always_recalibrate_mean = rollout(lambda obs: 0)

    assert trained_mean > always_recalibrate_mean
