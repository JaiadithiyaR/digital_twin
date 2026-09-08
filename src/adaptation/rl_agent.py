"""Module 13 — PPO RL Decision Agent: training + runtime inference (prompt.md §18-§21, §0.21,
§70 rules 4/18, CLAUDE.md §6).

This file owns everything CLAUDE.md §6 requires of "the runtime":

- `train_ppo()` trains a genuine Stable-Baselines3 PPO policy through `rl_env.py`'s Gymnasium
  environment — never a rule-based function pretending to be PPO (prompt.md §0.21/§70 rule 18).
- `load_ppo_agent()` loads the saved/trained policy for inference — the default runtime path.
- `decide_adaptation_strategy()` calls `PPO.predict()` and ONLY that — the action index and
  strategy name it returns are always genuinely the trained policy's decision, never a
  heuristic's (prompt.md §70 rule 4: "PPO decides WHAT adaptation strategy to use").
- `decide_adaptation_strategy_safe()` adds the ONE permitted exception: a deterministic fallback
  used exclusively when PPO inference itself raises (infrastructure failure — model file
  missing/corrupt, `predict()` throwing), never as a routine substitute, and every use is logged
  at WARNING level (CLAUDE.md §6 — "every use of it must be logged, never a silent substitute").
  Do not call this from code paths that want a hard failure on PPO infra problems; call
  `decide_adaptation_strategy()` directly there instead.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.adaptation.rl_env import AdaptationEnv, build_drift_event_pool, build_network_state_pool
from src.drift.schema import DriftEvent

if TYPE_CHECKING:
    from src.common.config import Settings

logger = logging.getLogger(__name__)


class EpisodeRewardLogger(BaseCallback):
    """Collects the real per-episode reward total the moment each episode actually finishes
    (SB3's `Monitor` wrapper populates `info["episode"]` on that exact step) — this is what
    produces genuine reward-curve data, not a synthetic/smoothed proxy."""

    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_timesteps: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.episode_rewards.append(float(episode["r"]))
                self.episode_lengths.append(int(episode["l"]))
                self.episode_timesteps.append(int(self.num_timesteps))
        return True


def make_vec_env(
    settings: "Settings",
    n_envs: int,
    network_state_pool: np.ndarray,
    drift_events: list[DriftEvent],
    base_seed: int,
) -> DummyVecEnv:
    """Single-process vectorized envs (`DummyVecEnv`), each an independently-seeded
    `AdaptationEnv` wrapped in SB3's `Monitor`. `DummyVecEnv` (not `SubprocVecEnv`) is a
    deliberate choice for this env: `AdaptationEnv.step()` is pure numpy with no I/O, so
    multiprocessing overhead would dominate wall-clock rather than help it, and it avoids
    multiprocessing/CUDA-fork pitfalls entirely for what CLAUDE.md's own instructions call a
    "small run given 16GB RAM."""

    def _make(rank: int):
        def _init():
            env = AdaptationEnv(
                settings, drift_events=drift_events, network_state_pool=network_state_pool, seed=base_seed + rank
            )
            return Monitor(env)

        return _init

    return DummyVecEnv([_make(i) for i in range(n_envs)])


def build_ppo_agent(settings: "Settings", vec_env: DummyVecEnv, seed: int | None = None) -> PPO:
    """Construct a genuine Stable-Baselines3 PPO model. Every hyperparameter comes from
    `config.ppo.training` — never hardcoded (CLAUDE.md §6 / prompt.md §18)."""
    cfg = settings.ppo.training
    return PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        seed=seed if seed is not None else cfg.seed,
        verbose=1,
        # This env's observation is a small flat vector (MlpPolicy, no CNN) — SB3's own guidance
        # is that GPU transfer overhead makes such small policies SLOWER on GPU than CPU (the
        # project's torch/CUDA availability, noted in CLAUDE.md §12, is for the DT prediction
        # models' potential future use, not assumed beneficial here). Forcing CPU is a measured
        # performance choice, not a capability limitation.
        device="cpu",
    )


def train_ppo(
    settings: "Settings",
    total_timesteps: int | None = None,
    n_envs: int | None = None,
    seed: int | None = None,
    save_path: str | Path | None = None,
    network_pool_seed: int = 1,
    drift_pool_seed: int = 1,
) -> tuple[PPO, EpisodeRewardLogger, float]:
    """Train PPO through the real `AdaptationEnv` and save the resulting policy.

    `total_timesteps`/`n_envs` default to `config.ppo.training.*` but are overridable — used to
    run a deliberately reduced ("small run given 16GB RAM") training budget without editing the
    checked-in config. Returns `(model, reward_logger, wall_clock_seconds)` so a caller can
    report real, observed training numbers (prompt.md §0.24 — never fabricate a result).
    """
    training_cfg = settings.ppo.training
    total_timesteps = total_timesteps if total_timesteps is not None else training_cfg.total_timesteps
    n_envs = n_envs if n_envs is not None else training_cfg.n_envs
    seed = seed if seed is not None else training_cfg.seed

    network_pool = build_network_state_pool(
        settings,
        seed=network_pool_seed,
        pool_size=settings.ppo.env.network_state_pool_size,
        sample_rows=settings.ppo.env.network_state_sample_rows,
    )
    drift_events = build_drift_event_pool(settings, seed=drift_pool_seed, pool_size=max(500, n_envs * 100))

    vec_env = make_vec_env(settings, n_envs, network_pool, drift_events, base_seed=seed)
    model = build_ppo_agent(settings, vec_env, seed=seed)
    reward_logger = EpisodeRewardLogger()

    logger.info(
        "PPO training starting",
        extra={"component": "rl_agent", "total_timesteps": total_timesteps, "n_envs": n_envs},
    )
    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=reward_logger)
    wall_clock_seconds = time.time() - start
    logger.info(
        "PPO training finished",
        extra={
            "component": "rl_agent",
            "total_timesteps": total_timesteps,
            "wall_clock_seconds": wall_clock_seconds,
            "episodes_completed": len(reward_logger.episode_rewards),
        },
    )

    resolved_save_path = Path(save_path) if save_path is not None else settings.resolve_path(settings.ppo.policy_path)
    resolved_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(resolved_save_path))
    logger.info("PPO policy saved", extra={"component": "rl_agent", "path": str(resolved_save_path)})

    return model, reward_logger, wall_clock_seconds


def load_ppo_agent(settings: "Settings", path: str | Path | None = None) -> PPO:
    """Load the trained/saved policy for runtime inference — the default runtime path
    (CLAUDE.md §6). Raises `FileNotFoundError` if it hasn't been trained yet; callers that want
    the deterministic-fallback behavior on this should catch it (or call
    `decide_adaptation_strategy_safe`, which does)."""
    resolved_path = Path(path) if path is not None else settings.resolve_path(settings.ppo.policy_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"PPO policy not found at {resolved_path} — train it first via train_ppo()")
    return PPO.load(str(resolved_path))


def decide_adaptation_strategy(model: PPO, observation: np.ndarray, settings: "Settings") -> tuple[int, str]:
    """The ONLY function that should ever decide an adaptation strategy in this system
    (prompt.md §70 rule 4). Always genuinely calls the trained PPO policy — `deterministic=True`
    so runtime behavior is reproducible given the same observation, matching CLAUDE.md §6
    ("runtime uses the trained/saved policy for inference")."""
    action, _ = model.predict(observation, deterministic=True)
    action_idx = int(np.asarray(action).item())
    return action_idx, settings.ppo.action_mapping[action_idx]


def _fallback_strategy(settings: "Settings", error: Exception) -> tuple[int, str]:
    if not settings.ppo.fallback.enabled:
        raise RuntimeError("PPO inference failed and the deterministic fallback is disabled") from error
    name = settings.ppo.fallback.default_action
    action_idx = next(idx for idx, mapped_name in settings.ppo.action_mapping.items() if mapped_name == name)
    # CLAUDE.md §6: "every use of it must be logged — never a silent substitute."
    logger.warning(
        "PPO inference failed — using deterministic fallback strategy (infrastructure failure "
        "only; this must never become a routine substitute for the trained policy)",
        extra={"component": "rl_agent", "fallback_action": name, "error": str(error)},
    )
    return action_idx, name


def decide_adaptation_strategy_safe(
    model: PPO | None, observation: np.ndarray, settings: "Settings"
) -> tuple[int, str]:
    """`decide_adaptation_strategy`, wrapped with the ONE permitted deterministic fallback for
    genuine PPO infrastructure failure (a missing/corrupt model, or `model is None` because
    loading already failed upstream) — never for a "PPO chose something we don't like" case, and
    always logged when triggered. Prefer `decide_adaptation_strategy` directly wherever a hard
    failure on PPO infra problems is actually the correct behavior."""
    if model is None:
        return _fallback_strategy(settings, RuntimeError("no PPO model loaded"))
    try:
        return decide_adaptation_strategy(model, observation, settings)
    except Exception as exc:  # deliberately broad: any inference failure must degrade to the
        # logged fallback, never crash the adaptation loop or silently do nothing.
        return _fallback_strategy(settings, exc)
