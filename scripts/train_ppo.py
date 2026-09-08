#!/usr/bin/env python
"""Module 13 — PPO RL Decision Agent: the actual training + inference + reporting run.

Trains PPO through the real `AdaptationEnv` (`src/adaptation/rl_env.py`), saves the policy to
`config.ppo.policy_path`, runs genuine inference (`decide_adaptation_strategy`,
`deterministic=True`) on a HELD-OUT drift scenario (a different RNG seed than training used, so
these exact episodes were never trained on), and writes a reward-curve plot + a text/JSON report
with real, observed numbers (prompt.md §0.24 — never fabricate a result; §21 — train through the
actual Gymnasium environment, PPO alone decides at inference).

Deliberately runs a REDUCED training budget relative to `config.ppo.training.total_timesteps`
(200,000) — this machine's practical constraint (16GB RAM, CPU-only PPO forward/backward passes
per CLAUDE.md's own guidance for a small MlpPolicy) makes the full budget an unnecessarily long
single run for what this deliverable needs to show (a genuine, measurable reward-curve
improvement). The exact reduced step count and the real wall-clock it took are printed and saved,
never silently substituted for the config default.

Usage:
    python scripts/train_ppo.py [--total-timesteps N] [--n-envs N] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.adaptation.rl_agent import decide_adaptation_strategy, train_ppo  # noqa: E402
from src.adaptation.rl_env import AdaptationEnv, build_drift_event_pool, build_network_state_pool  # noqa: E402
from src.common.config import load_settings  # noqa: E402

TRAIN_SEED_BASE = 1  # pools/env seed used for TRAINING
HELD_OUT_SEED_BASE = 9001  # deliberately different — these episodes were never trained on
REPORT_DIR = REPO_ROOT / "data" / "artifacts" / "ppo_training"


def _rolling_mean(values: list[float], window: int) -> np.ndarray:
    if len(values) < window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(np.array(values, dtype=float), kernel, mode="valid")


def _plot_reward_curve(episode_rewards: list[float], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless — this script has no display
    import matplotlib.pyplot as plt

    window = max(10, len(episode_rewards) // 50)
    smoothed = _rolling_mean(episode_rewards, window)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(len(episode_rewards)), episode_rewards, alpha=0.25, linewidth=0.6, label="per-episode reward")
    ax.plot(
        range(window - 1, window - 1 + len(smoothed)),
        smoothed,
        linewidth=2.0,
        label=f"rolling mean (window={window})",
    )
    ax.set_xlabel("episode")
    ax.set_ylabel("episode reward (fidelity_improvement - adaptation_cost_penalty)")
    ax.set_title("Module 13 PPO training — real observed episode reward")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _held_out_inference_report(model, settings) -> dict:
    """Runs genuine PPO inference (never a heuristic) on a held-out drift scenario pool built
    from a seed the training run never used, at three fixed severities that
    `tests/unit/test_rl_env.py` already proved have different optimal strategies — a concrete,
    checkable demonstration of what the trained policy actually does, not just an aggregate
    number."""
    from datetime import UTC, datetime

    from src.drift.schema import DriftEvent

    network_pool = build_network_state_pool(
        settings,
        seed=HELD_OUT_SEED_BASE,
        pool_size=settings.ppo.env.network_state_pool_size,
        sample_rows=settings.ppo.env.network_state_sample_rows,
    )
    held_out_events = build_drift_event_pool(settings, seed=HELD_OUT_SEED_BASE, pool_size=200)

    def run_scenario(events, label, n_episodes=60, action_fn=None):
        env = AdaptationEnv(settings, drift_events=events, network_state_pool=network_pool, seed=HELD_OUT_SEED_BASE)
        totals, lengths, trajectories = [], [], []
        for i in range(n_episodes):
            obs, info = env.reset(seed=HELD_OUT_SEED_BASE + i)
            done = False
            total = 0.0
            steps = 0
            traj = []
            while not done:
                if action_fn is not None:
                    action = action_fn(obs)
                else:
                    action, action_name = decide_adaptation_strategy(model, obs, settings)
                obs, reward, terminated, truncated, info = env.step(action)
                traj.append(
                    {
                        "step": steps,
                        "action": settings.ppo.action_mapping[int(action)],
                        "reward": round(float(reward), 4),
                        "new_fidelity": round(float(info["new_fidelity"]), 4),
                    }
                )
                total += reward
                steps += 1
                done = terminated or truncated
            totals.append(total)
            lengths.append(steps)
            if i < 3:
                trajectories.append({"episode": i, "affected_component": info["affected_component"], "steps": traj})
        return {
            "label": label,
            "n_episodes": n_episodes,
            "mean_reward": float(np.mean(totals)),
            "mean_episode_length": float(np.mean(lengths)),
            "sample_trajectories": trajectories,
        }

    now = datetime.now(UTC)
    fixed_scenarios = {
        "low_severity_fresh_incident": [
            DriftEvent(component="throughput", severity=0.1, timestamp=now, metadata={}, source="MOCK", received_at=now)
        ]
        * 200,
        "high_severity_fresh_incident": [
            DriftEvent(component="latency", severity=0.9, timestamp=now, metadata={}, source="MOCK", received_at=now)
        ]
        * 200,
    }

    results = {"held_out_drift_pool": run_scenario(held_out_events, "held_out_drift_pool (mixed components/severities)")}
    for name, events in fixed_scenarios.items():
        results[name] = run_scenario(events, name)
        results[f"{name}_always_recalibrate_baseline"] = run_scenario(
            events, f"{name}_always_recalibrate_baseline", action_fn=lambda obs: 0
        )
        results[f"{name}_always_regenerate_baseline"] = run_scenario(
            events, f"{name}_always_regenerate_baseline", action_fn=lambda obs: 1
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=20000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = load_settings()
    print(
        f"Training PPO: total_timesteps={args.total_timesteps} (config default is "
        f"{settings.ppo.training.total_timesteps}, reduced here — see this script's docstring) "
        f"n_envs={args.n_envs} seed={args.seed}"
    )

    start = time.time()
    model, reward_logger, wall_clock_seconds = train_ppo(
        settings,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        network_pool_seed=TRAIN_SEED_BASE,
        drift_pool_seed=TRAIN_SEED_BASE,
    )
    print(f"Training finished in {wall_clock_seconds:.1f}s wall-clock, {len(reward_logger.episode_rewards)} episodes")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    curve_path = REPORT_DIR / "reward_curve.png"
    _plot_reward_curve(reward_logger.episode_rewards, curve_path)
    print(f"Reward curve saved to {curve_path}")

    n = len(reward_logger.episode_rewards)
    first_chunk = reward_logger.episode_rewards[: max(1, n // 10)]
    last_chunk = reward_logger.episode_rewards[-max(1, n // 10) :]
    early_mean = float(np.mean(first_chunk))
    late_mean = float(np.mean(last_chunk))
    print(f"Mean reward — first {len(first_chunk)} episodes: {early_mean:.4f}")
    print(f"Mean reward — last {len(last_chunk)} episodes: {late_mean:.4f}")

    print("\nRunning held-out inference (genuine PPO.predict(), never a heuristic)...")
    inference_report = _held_out_inference_report(model, settings)
    for key in ("low_severity_fresh_incident", "high_severity_fresh_incident"):
        trained = inference_report[key]
        recal_baseline = inference_report[f"{key}_always_recalibrate_baseline"]
        regen_baseline = inference_report[f"{key}_always_regenerate_baseline"]
        print(
            f"  {key}: trained_policy_mean_reward={trained['mean_reward']:.4f} "
            f"(always_recalibrate={recal_baseline['mean_reward']:.4f}, "
            f"always_regenerate={regen_baseline['mean_reward']:.4f})"
        )

    report = {
        "training": {
            "total_timesteps": args.total_timesteps,
            "config_default_total_timesteps": settings.ppo.training.total_timesteps,
            "n_envs": args.n_envs,
            "seed": args.seed,
            "wall_clock_seconds": wall_clock_seconds,
            "episodes_completed": n,
            "mean_reward_first_chunk": early_mean,
            "mean_reward_last_chunk": late_mean,
            "hyperparameters": settings.ppo.training.model_dump(),
        },
        "held_out_inference": inference_report,
    }
    report_path = REPORT_DIR / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nFull report saved to {report_path}")
    print(f"Policy saved to {settings.resolve_path(settings.ppo.policy_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
