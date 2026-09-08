"""Module 13 — PPO RL Decision Agent: Gymnasium environment (prompt.md §18-§21, §0.21, §70 rules
4/18, CLAUDE.md §6).

`AdaptationEnv` is what `rl_agent.py`'s PPO genuinely trains against — this file defines the
*environment* (what happens to the world when an action is taken), never the *decision* (which
action to take). That distinction matters: prompt.md §0.21/§70 rule 18 forbid a fake rule-based
function standing in for PPO ("if severity > X: regenerate()"). Nothing here does that — this
module never chooses an action; it only reacts to whatever action the caller (PPO during
training, or `rl_agent.py`'s trained-policy inference at runtime) supplies, exactly the way
CartPole's physics react to a push without ever deciding which push to apply.

**Why a simulated adaptation-outcome model exists at all**: Modules 14-16 (the agents that would
actually recalibrate/regenerate/expand-scope a component and produce a real post-adaptation
fidelity) are not built yet. Training PPO requires *some* environment to act in — so this module
provides a documented, config-driven ("world model") stand-in for "if you apply action A to a
component with prediction-error magnitude E and this is attempt N on this incident, how much does
the error shrink" (`config.ppo.env.*`, see `config/settings.yaml`'s comments for the exact
rationale of every constant). When Modules 14-17 exist, a production runtime environment can
supply real post-adaptation fidelity instead of this simulation without changing PPO's interface
at all — `rl_agent.py`'s `decide_adaptation_strategy()` only ever needs an observation vector.

**Real, not fabricated, integration with three already-built modules**, not synthetic numbers
invented in this file:
- **Module 11 (drift)**: `build_drift_event_pool()` drives the real `MockDriftSource` +
  `DriftDetectorInterface` to produce genuinely validated `DriftEvent`s — every episode's
  affected component/severity comes from this pool, not a hand-rolled random draw.
- **Module 12 (fidelity)**: every fidelity value in the observation comes from a real
  `FidelityEvaluator.evaluate()` call against synthetic-but-real (y_true, y_pred) arrays whose
  error magnitude the env's dynamics control — the deterministic RMSE/MAE/Wasserstein/MK-MMD
  formula genuinely runs every step, it is never a placeholder number.
- **Module 4 (D1)**: `build_network_state_pool()` builds a real `D1Store`, populates it via the
  real Module 2 pipeline (`MockTelemetrySource` -> `TelemetryPreprocessor`), and reads
  `get_history()` — the "network state" observation slice is real telemetry aggregate statistics,
  not invented numbers.

**Episode = one drift incident**, not a fixed-length rollout: `reset()` draws a fresh incident
(component, severity) from the drift pool; `step()` applies one adaptation attempt and can
terminate (fidelity recovered) or truncate (too many attempts, `config.ppo.env.
max_attempts_per_incident`). "Previous action"/"previous reward" in the observation are therefore
genuinely Markovian WITHIN an incident (what was tried last on THIS incident and how well it
worked) — not carried across incidents, which keeps the environment a well-defined MDP per
episode while still giving PPO a real reason to use that part of the observation (whether to
retry the same action or escalate).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

from src.dt_models.d1_model_store import D1Store
from src.drift.drift_detector import DriftDetectorInterface
from src.drift.mock_drift_source import MockDriftSource
from src.drift.schema import DriftEvent
from src.fidelity.evaluator import FidelityEvaluator
from src.telemetry.mock_source import MockTelemetrySource
from src.telemetry.preprocessing import TelemetryPreprocessor

if TYPE_CHECKING:
    from src.common.config import Settings

logger = logging.getLogger(__name__)

# Network-state summary features aggregated from real D1 history rows (Module 4). Fixed,
# well-dimensioned set (prompt.md §18 "kept well-dimensioned") — canonical D1 column names, no
# fabricated field.
NETWORK_STATE_FEATURES: tuple[str, ...] = (
    "throughput_mbps",
    "offered_load_mbps",
    "latency_ms",
    "jitter_ms",
    "packet_loss_pct",
    "prb_utilization_pct",
    "sinr_db",
    "ue_speed_mps",
)

# Clamp for the (theoretically unbounded-below) Module 12 fidelity score before it enters this
# env's observation/reward — see `_evaluate_fidelity`'s docstring comment for why this is a local
# RL-stability measure, never a change to Module 12's own deterministic formula or return value.
_FIDELITY_SCORE_CLIP: tuple[float, float] = (-3.0, 1.0)


def build_network_state_pool(settings: "Settings", seed: int, pool_size: int, sample_rows: int) -> np.ndarray:
    """Populate a real `D1Store` via the real Module 2 telemetry pipeline, then precompute
    `pool_size` normalized network-state summary vectors (each the mean of `sample_rows` randomly
    drawn history rows) for `AdaptationEnv` episodes to sample from.

    Precomputing a pool (rather than hitting D1 on every `reset()`) is a deliberate performance
    choice for RL training's high reset-rate, and — crucially — it also sidesteps a genuine
    hazard: multiple vectorized env copies constructing/writing their own `D1Store` at the same
    parquet paths concurrently would race. A shared, already-computed, read-only pool has none of
    that risk regardless of vectorization backend.

    Writes to a throwaway temp directory, never to `config.storage`'s real D1 paths — this is a
    training-time simulation aid, not production Digital Twin state (prompt.md §0.5's state
    separation applies here too: this must never be confused with real D1).
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="ppo_env_d1_"))
    store = D1Store(
        current_state_path=tmp_dir / "current_state.parquet",
        history_path=tmp_dir / "history.parquet",
        quarantine_path=tmp_dir / "quarantine.parquet",
        history_retention_rows=max(pool_size * sample_rows, 1000),
    )
    mock_config = settings.telemetry.mock.model_copy(update={"seed": seed})
    # Generous multiple of what's strictly needed so sampled rows aren't just the same handful
    # of UEs repeated verbatim across every pool entry.
    num_records = max(pool_size * sample_rows, mock_config.num_ues * 50)
    source = MockTelemetrySource(mock_config, max_records=num_records, realtime=False)
    preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)
    result = preprocessor.process_batch(source.records())
    store.append_history(result.clean_records)
    history_df = store.get_history()

    if len(history_df) == 0:
        raise RuntimeError("build_network_state_pool: D1 history is empty after telemetry generation")

    rng = np.random.default_rng(seed)
    values = history_df[list(NETWORK_STATE_FEATURES)].to_numpy(dtype=np.float64)
    n_rows = len(values)
    pool = np.empty((pool_size, len(NETWORK_STATE_FEATURES)), dtype=np.float64)
    for i in range(pool_size):
        idx = rng.integers(0, n_rows, size=min(sample_rows, n_rows))
        pool[i] = values[idx].mean(axis=0)

    feature_min = pool.min(axis=0)
    feature_max = pool.max(axis=0)
    span = np.where(feature_max - feature_min > 1e-9, feature_max - feature_min, 1.0)
    normalized = (pool - feature_min) / span
    logger.info(
        "network-state pool built from real D1 history",
        extra={"component": "rl_env", "pool_size": pool_size, "history_rows": n_rows},
    )
    return normalized.astype(np.float32)


def build_drift_event_pool(settings: "Settings", seed: int, pool_size: int) -> list[DriftEvent]:
    """Drive the real Module 11 `MockDriftSource` + `DriftDetectorInterface` to produce
    `pool_size` genuinely validated `DriftEvent`s for episodes to draw from.

    `invalid_event_rate` is forced to 0.0 here regardless of `config.drift.mock.
    invalid_event_rate` — Module 11's own quarantine path is already covered by its own tests
    (`tests/unit/test_drift_detector.py`); this pool exists to give the RL environment
    well-formed adaptation contexts to learn from, not to re-test drift-event validation.
    """
    mock_config = settings.drift.mock.model_copy(update={"seed": seed, "invalid_event_rate": 0.0})
    source = MockDriftSource(
        mock_config,
        valid_components=settings.drift.valid_components,
        severity_range=settings.drift.severity_range,
        max_events=pool_size * 2,  # headroom; process_stream only yields validated ones
        realtime=False,
    )
    detector = DriftDetectorInterface(
        valid_components=settings.drift.valid_components, severity_range=settings.drift.severity_range
    )
    events = list(detector.process_stream(source))[:pool_size]
    source.close()
    if len(events) < pool_size:
        raise RuntimeError(
            f"build_drift_event_pool: only {len(events)}/{pool_size} valid events produced"
        )
    logger.info(
        "drift-event pool built from real Module 11 pipeline",
        extra={"component": "rl_env", "pool_size": len(events), "seed": seed},
    )
    return events


class AdaptationEnv(gym.Env):
    """Gymnasium environment for Module 13's PPO decision agent.

    Action space: `Discrete(3)` — 0=Recalibrate, 1=Regenerate, 2=Expand Scope
    (`config.ppo.action_mapping`; prompt.md §19 — never a 4th/5th action).

    Observation (`Box`, all entries normalized/clipped — prompt.md §18): concatenation of
    [per-component fidelity vector, affected-component one-hot, drift severity,
    previous-action one-hot, previous reward, network-state summary].
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        settings: "Settings",
        drift_events: list[DriftEvent],
        network_state_pool: np.ndarray,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._env_cfg = settings.ppo.env
        self._components: tuple[str, ...] = tuple(settings.drift.valid_components)
        self._n_components = len(self._components)
        self._component_index = {name: i for i, name in enumerate(self._components)}
        self._severity_range = settings.drift.severity_range
        self._cost_penalty = settings.ppo.reward.adaptation_cost_penalty
        # Action index -> name, derived from config (never hardcoded order) but the mapping
        # itself is prompt.md §19's fixed 3-action contract.
        self._action_names = [settings.ppo.action_mapping[i] for i in sorted(settings.ppo.action_mapping)]
        self._n_actions = len(self._action_names)
        if self._n_actions != 3:
            raise ValueError(f"PPO action space must be exactly 3 (prompt.md §19); got {self._n_actions}")

        self._drift_events = drift_events
        self._network_state_pool = network_state_pool
        self._rng = np.random.default_rng(seed)

        # Fixed per-component ground-truth reference array. Deliberately NOT re-sampled on every
        # evaluate() call: the rolling-window normalization in Module 12 is sensitive to the
        # squared-metric distribution's own min/max, and re-randomizing y_true every call let the
        # *sampling noise between two independent random arrays* dominate the signal, occasionally
        # collapsing the window's (max-min) range near zero and producing wild, physically
        # meaningless normalized values. A fixed y_true (only y_pred's error varies, per the
        # current adaptation state) is what makes "prediction quality" the only thing driving
        # fidelity here — mirroring how a real component's ground truth doesn't change, only its
        # prediction quality does.
        self._y_true_by_component = {
            component: self._rng.normal(0.0, 1.0, size=self._env_cfg.fidelity_sample_size)
            for component in self._components
        }
        # `FidelityEvaluator` and its rolling windows are (re)created fresh inside every
        # `reset()`, not kept persistent across episodes — see `reset()`'s comment for why: a
        # single evaluator carried across episodes made this env's `reset(seed=X)` non-
        # reproducible (Module 12's normalization reference kept accumulating hidden state that
        # the seed didn't capture), which `gymnasium.utils.env_checker.check_env`'s determinism
        # check correctly caught. Placeholder assigned here only so the attribute exists before
        # the first `reset()` call.
        self._fidelity_evaluator: FidelityEvaluator | None = None

        n_network_features = network_state_pool.shape[1]
        obs_dim = (
            self._n_components  # fidelity vector
            + self._n_components  # affected-component one-hot
            + 1  # drift severity
            + self._n_actions  # previous action one-hot
            + 1  # previous reward
            + n_network_features  # network state (from D1)
        )
        self.action_space = gym.spaces.Discrete(self._n_actions)
        self.observation_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32)

        # Per-incident state, (re)initialized in reset().
        self._affected_component = self._components[0]
        self._severity = 0.0
        self._remaining_error_fraction = 1.0
        self._attempt_count = 0
        self._attempts_by_action: dict[str, int] = {}
        self._current_fidelity = np.ones(self._n_components, dtype=np.float64)
        self._prev_action_onehot = np.zeros(self._n_actions, dtype=np.float32)
        self._prev_reward = 0.0
        self._network_state = network_state_pool[0]

    # --- fidelity simulation helpers ---------------------------------------------------------

    def _prewarm_fidelity_windows(self) -> None:
        """Seed every component's rolling normalization window with `min_history_for_normalization`
        evaluations spanning the full realistic error range (healthy to fully-drifted) so composite
        fidelity scores are defined (status="ok") from the very first episode, AND so the window's
        min-max normalization reference already reflects a realistic mix of good and bad history
        (prompt.md §0.10's "never silently manufacture a score" still holds — this seeds real
        historical data through the real evaluator, it doesn't fabricate a score).

        Seeding with ONLY healthy (near-zero-error) points was tried first and rejected: it made
        every component's window nearly constant, so the FIRST severely-drifted evaluation right
        after prewarm compared a large squared-error value against a near-zero window (max-min)
        range — Module 12's `eps`-stabilized denominator (correctly, by design) then produced
        enormous normalized values and correspondingly wild fidelity scores. A window that already
        contains a realistic spread of error magnitudes avoids that cold-start artifact without
        touching Module 12's formula at all.
        """
        min_history = self._settings.fidelity.min_history_for_normalization
        for component in self._components:
            # Deliberately calls the evaluator directly (not `_evaluate_fidelity`, which asserts
            # a defined score) — the first `min_history` calls are EXPECTED to return
            # status="insufficient_history" by Module 12's own design (prompt.md §0.10); that is
            # not an error here, it's exactly how the window gets seeded.
            for _ in range(min_history + 1):
                spread = self._env_cfg.healthy_noise_std + self._env_cfg.drift_error_scale * self._rng.uniform(0.0, 1.0)
                y_true, y_pred = self._synthetic_prediction(component, error_std=spread)
                self._fidelity_evaluator.evaluate(component, y_true, y_pred, update_window=True)

    def _synthetic_prediction(self, component: str, error_std: float) -> tuple[np.ndarray, np.ndarray]:
        y_true = self._y_true_by_component[component]
        y_pred = y_true + self._rng.normal(0.0, max(error_std, 1e-6), size=len(y_true))
        return y_true, y_pred

    def _evaluate_fidelity(self, component: str, error_std: float) -> float:
        y_true, y_pred = self._synthetic_prediction(component, error_std)
        result = self._fidelity_evaluator.evaluate(component, y_true, y_pred, update_window=True)
        # Windows are prewarmed above min_history for every component before any episode runs,
        # so `fidelity_score` is always defined here — but never silently substitute if it isn't.
        if result.fidelity_score is None:
            raise RuntimeError(f"unexpected insufficient_history for prewarmed component {component!r}")
        # Module 12's formula is, by design, unbounded below (the eps-stabilized denominator can
        # make a single relatively-extreme point normalize to a very large value — this is
        # correct, documented behavior of the deterministic formula itself, see
        # src/fidelity/evaluator.py). Clamping HERE is a local RL-training stability measure over
        # the value Module 12 already computed — it never modifies Module 12's own formula,
        # window, or return value; `FidelityEvaluator.evaluate()` is called unmodified above and
        # its true `fidelity_score` is what gets clamped for use in this env's bounded
        # observation/reward space (prompt.md §18 "normalize observations appropriately").
        return float(np.clip(result.fidelity_score, _FIDELITY_SCORE_CLIP[0], _FIDELITY_SCORE_CLIP[1]))

    def _error_std_for(self, remaining_error_fraction: float) -> float:
        base = self._env_cfg.healthy_noise_std
        drift_component = self._env_cfg.drift_error_scale * self._severity * remaining_error_fraction
        return base + drift_component

    def _action_efficacy(self, action_name: str) -> float:
        """How much of the current prediction-error magnitude `action_name` removes this attempt.

        The severity-sensitivity shape (recalibrate scales down linearly with severity;
        regenerate only mildly) is what actually makes this a non-trivial decision problem:
        recalibration = "retrain the existing model on recent data" — genuinely effective for a
        modest distribution shift, structurally unable to fix a large/severe one no matter how
        many times it's retried (also enforced by the attempt-diminishing decay below).
        Regeneration = "rebuild the model from scratch" (LLM-driven) — costs more
        (`config.ppo.reward.adaptation_cost_penalty`) but stays effective at high severity. This
        is what should make PPO learn "recalibrate first when severity is low or this is a fresh
        incident; escalate to regenerate when severity is high or recalibrate already failed."
        """
        cfg = self._env_cfg
        prior_attempts_this_incident = self._attempt_count  # attempts BEFORE this one
        if action_name == "recalibrate":
            severity_factor = max(0.0, 1.0 - self._severity)
            decay = cfg.attempt_diminishing_factor ** self._attempts_by_action.get("recalibrate", 0)
            return cfg.recalibrate_efficacy * severity_factor * decay
        if action_name == "regenerate":
            severity_factor = max(0.0, 1.0 - cfg.regenerate_severity_sensitivity * self._severity)
            return cfg.regenerate_efficacy * severity_factor
        if action_name == "expand_scope":
            if prior_attempts_this_incident >= cfg.expand_scope_min_prior_attempts:
                return cfg.expand_scope_efficacy
            return cfg.expand_scope_efficacy * cfg.expand_scope_premature_factor
        raise ValueError(f"unknown action name {action_name!r}")  # unreachable given action_space

    # --- Gymnasium API -------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Fresh evaluator + rolling windows every episode, deterministically reseeded from
        # `self._rng` above — makes `reset(seed=X)` fully reproducible (required by
        # `gymnasium.utils.env_checker.check_env`'s determinism check), at the cost of not
        # letting the fidelity normalization reference evolve across episodes within one
        # training run. That trade was deliberate: a persistent evaluator made episode N's
        # observations depend on the full history of episodes 0..N-1, which is both
        # non-reproducible from a seed and non-Markovian in a way PPO isn't designed to handle
        # across resets.
        self._fidelity_evaluator = FidelityEvaluator(self._settings.fidelity)
        self._prewarm_fidelity_windows()

        event = self._drift_events[int(self._rng.integers(0, len(self._drift_events)))]
        self._affected_component = event.component
        self._severity = event.severity
        self._remaining_error_fraction = 1.0
        self._attempt_count = 0
        self._attempts_by_action = {}
        self._prev_action_onehot = np.zeros(self._n_actions, dtype=np.float32)
        self._prev_reward = 0.0
        self._network_state = self._network_state_pool[self._rng.integers(0, len(self._network_state_pool))]

        # Healthy fidelity for every component except the freshly-drifted one; the affected
        # component gets evaluated at full incident severity (no repair attempted yet).
        fidelity = np.empty(self._n_components, dtype=np.float64)
        for i, component in enumerate(self._components):
            if component == self._affected_component:
                fidelity[i] = self._evaluate_fidelity(component, self._error_std_for(1.0))
            else:
                fidelity[i] = self._evaluate_fidelity(component, self._env_cfg.healthy_noise_std)
        self._current_fidelity = fidelity

        obs = self._build_observation()
        info = {"affected_component": self._affected_component, "severity": self._severity}
        return obs, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} outside action_space {self.action_space}")
        action_name = self._action_names[int(action)]
        affected_idx = self._component_index[self._affected_component]
        old_fidelity = float(self._current_fidelity[affected_idx])

        efficacy = self._action_efficacy(action_name)
        self._remaining_error_fraction = float(np.clip(self._remaining_error_fraction * (1.0 - efficacy), 0.0, 1.0))
        new_fidelity = self._evaluate_fidelity(
            self._affected_component, self._error_std_for(self._remaining_error_fraction)
        )
        self._current_fidelity[affected_idx] = new_fidelity

        cost_penalty = self._cost_penalty.get(action_name, 0.0)
        raw_reward = (new_fidelity - old_fidelity) - cost_penalty

        self._attempt_count += 1
        self._attempts_by_action[action_name] = self._attempts_by_action.get(action_name, 0) + 1

        terminated = new_fidelity >= self._env_cfg.resolved_fidelity_threshold
        truncated = (not terminated) and (self._attempt_count >= self._env_cfg.max_attempts_per_incident)

        self._prev_action_onehot = np.zeros(self._n_actions, dtype=np.float32)
        self._prev_action_onehot[int(action)] = 1.0
        self._prev_reward = raw_reward

        obs = self._build_observation()
        info = {
            "affected_component": self._affected_component,
            "action_name": action_name,
            "severity": self._severity,
            "attempt_count": self._attempt_count,
            "old_fidelity": old_fidelity,
            "new_fidelity": new_fidelity,
            "efficacy": efficacy,
            "remaining_error_fraction": self._remaining_error_fraction,
        }
        return obs, float(raw_reward), bool(terminated), bool(truncated), info

    def _build_observation(self) -> np.ndarray:
        fidelity_vec = np.clip(self._current_fidelity, -2.0, 2.0).astype(np.float32)
        affected_onehot = np.zeros(self._n_components, dtype=np.float32)
        affected_onehot[self._component_index[self._affected_component]] = 1.0
        lo, hi = self._severity_range
        severity_norm = np.array([(self._severity - lo) / max(hi - lo, 1e-9)], dtype=np.float32)
        prev_reward_clipped = np.array(
            [np.clip(self._prev_reward, -self._env_cfg.reward_clip, self._env_cfg.reward_clip)], dtype=np.float32
        )
        obs = np.concatenate(
            [
                fidelity_vec,
                affected_onehot,
                severity_norm,
                self._prev_action_onehot,
                prev_reward_clipped,
                self._network_state.astype(np.float32),
            ]
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)
