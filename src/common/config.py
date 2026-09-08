"""Central configuration loader.

Every tunable parameter used anywhere in this codebase must be read through the `Settings`
object returned by `load_settings()`, sourced from `config/settings.yaml`. Secrets (API keys)
are loaded separately from the environment/.env via `load_secrets()` and are never merged into
`Settings` or serialized alongside it — this keeps a config dump or log line from ever being able
to leak a key (see prompt.md §41, §50, §62).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"


class MockTelemetryConfig(BaseModel):
    seed: int
    emit_interval_seconds: float
    num_ues: int
    num_cells: int
    missing_field_rate: float
    out_of_range_rate: float


class ZmqTelemetryConfig(BaseModel):
    endpoint: str
    topic: str
    recv_timeout_ms: int
    high_water_mark: int


class PreprocessingConfig(BaseModel):
    required_fields: list[str]
    valid_ranges: dict[str, tuple[float, float]]
    feature_window_size: int
    max_missing_ratio: float


class TelemetryConfig(BaseModel):
    source: Literal["mock", "zmq"]
    mock: MockTelemetryConfig
    zmq: ZmqTelemetryConfig
    preprocessing: PreprocessingConfig


class SynchronizationConfig(BaseModel):
    batch_size: int
    batch_timeout_seconds: float


class StorageConfig(BaseModel):
    root_dir: str
    bootstrap_dir: str
    telemetry_dir: str
    evaluation_dir: str
    artifacts_dir: str
    models_dir: str
    d1_current_state_path: str
    d1_history_path: str
    d1_quarantine_path: str
    history_retention_rows: int


class ModelSpec(BaseModel):
    model_type: str
    params: dict[str, float | int | str | None]


class DtModelsConfig(BaseModel):
    enable_prb_model: bool
    bootstrap_min_rows: int
    throughput: ModelSpec
    packet_loss: ModelSpec
    latency: ModelSpec
    prb_utilization: ModelSpec
    jitter: ModelSpec


class MkMmdConfig(BaseModel):
    kernel: str
    gamma: float | None


class FidelityConfig(BaseModel):
    epsilon: float
    rolling_window_length: int
    min_history_for_normalization: int
    metrics: list[str]
    mk_mmd: MkMmdConfig


class MockDriftConfig(BaseModel):
    seed: int
    emit_interval_seconds: float
    invalid_event_rate: float


class DriftConfig(BaseModel):
    source: Literal["mock", "external"]
    # Shared between the mock source and drift_detector.py's validation (not nested under
    # `mock`) since a real external detector's events must be validated against the exact same
    # bounds/scope list — never a mock-only concern.
    valid_components: list[str]
    severity_range: tuple[float, float]
    mock: MockDriftConfig


class PpoTrainingConfig(BaseModel):
    total_timesteps: int
    n_envs: int
    learning_rate: float
    n_steps: int
    batch_size: int
    gamma: float
    gae_lambda: float
    clip_range: float
    seed: int


class PpoRewardConfig(BaseModel):
    adaptation_cost_penalty: dict[str, float]


class PpoFallbackConfig(BaseModel):
    enabled: bool
    default_action: str


class PpoEnvConfig(BaseModel):
    """Tunables for `AdaptationEnv`'s SIMULATED adaptation-outcome dynamics (the training
    environment's "world model") — NOT PPO's own algorithm hyperparameters (those live in
    `PpoTrainingConfig`) and NOT real adaptation-agent behavior (Modules 14-16 don't exist yet;
    when they do, this env's dynamics are replaced/supplemented by real outcomes, not this
    config). See `src/adaptation/rl_env.py` module docstring for the full rationale."""

    max_attempts_per_incident: int
    resolved_fidelity_threshold: float
    fidelity_sample_size: int
    healthy_noise_std: float
    drift_error_scale: float
    recalibrate_efficacy: float
    regenerate_efficacy: float
    regenerate_severity_sensitivity: float
    expand_scope_efficacy: float
    expand_scope_min_prior_attempts: int
    expand_scope_premature_factor: float
    attempt_diminishing_factor: float
    network_state_pool_size: int
    network_state_sample_rows: int
    reward_clip: float


class PpoConfig(BaseModel):
    policy_path: str
    action_mapping: dict[int, str]
    training: PpoTrainingConfig
    reward: PpoRewardConfig
    env: PpoEnvConfig
    fallback: PpoFallbackConfig


class RecalibrationConfig(BaseModel):
    training_window_hours: int
    min_training_rows: int


class RegenerationConfig(BaseModel):
    max_llm_iterations: int  # max self-correction attempts: LLM generates -> sandbox rejects -> LLM retries with the rejection reason as feedback
    training_window_hours: int  # independently tunable from recalibration's — rebuilding a pipeline from scratch reasonably wants more history than a lightweight refit
    min_training_rows: int


class ExpandScopeConfig(BaseModel):
    max_llm_iterations: int  # shared retry budget for BOTH the design-proposal step and the implementation-generation step
    training_window_hours: int
    min_training_rows: int


class AdaptationConfig(BaseModel):
    lock_policy: Literal["queue", "coalesce", "defer"]
    recalibration: RecalibrationConfig
    regeneration: RegenerationConfig
    expand_scope: ExpandScopeConfig
    verification_delta: float


class SandboxConfig(BaseModel):
    workspace_root: str
    timeout_seconds: int
    max_output_bytes: int


class RagConfig(BaseModel):
    chroma_path: str
    collection_name: str
    corpus_dirs: dict[str, str]
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_model: str


class LlmConfig(BaseModel):
    model: str
    max_tokens: int
    # The installed Anthropic API version (anthropic-sdk-python's real MessageCreateParams, not
    # assumed from older docs) has no temperature/top_p/top_k sampling-randomness control at all
    # — verified directly against the SDK. `effort` (reasoning depth) is the closest available
    # knob; it is NOT a determinism control. See src/llm/anthropic_client.py's module docstring.
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None
    timeout_seconds: int
    max_retries: int  # shared budget for both this client's transport retry loop and its structured-output validation retry loop
    retry_backoff_seconds: float  # base for this client's own exponential backoff (capped internally); the SDK's own internal retry is disabled so this is the only retry/backoff path


class LoggingConfig(BaseModel):
    level: str
    format: Literal["json", "text"]
    log_dir: str


class LifecycleConfig(BaseModel):
    records_path: str
    reports_dir: str


class EvaluationConfig(BaseModel):
    window_size: int


class Settings(BaseModel):
    """Root configuration object. Immutable after load — treat as read-only."""

    model_config = {"frozen": True}

    environment: Literal["demo", "live"]
    telemetry: TelemetryConfig
    synchronization: SynchronizationConfig
    storage: StorageConfig
    dt_models: DtModelsConfig
    fidelity: FidelityConfig
    drift: DriftConfig
    ppo: PpoConfig
    adaptation: AdaptationConfig
    sandbox: SandboxConfig
    rag: RagConfig
    llm: LlmConfig
    logging: LoggingConfig
    lifecycle: LifecycleConfig
    evaluation: EvaluationConfig

    def resolve_path(self, relative: str) -> Path:
        """Resolve a config-declared path against the repository root."""
        return REPO_ROOT / relative


class Secrets(BaseModel):
    """Loaded from environment/.env only. Never logged, never merged into Settings."""

    anthropic_api_key: str | None = Field(default=None, repr=False)

    def require_anthropic_key(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in "
                "before using any LLM-driven agent (regeneration, expand-scope, verification "
                "reasoning, lifecycle explanations)."
            )
        return self.anthropic_api_key

    def __repr__(self) -> str:  # never let a stray print/log leak the key
        return "Secrets(anthropic_api_key=<redacted>)"


@lru_cache(maxsize=1)
def load_settings(path: Path | None = None) -> Settings:
    settings_path = path or DEFAULT_SETTINGS_PATH
    with settings_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def load_secrets() -> Secrets:
    load_dotenv(REPO_ROOT / ".env", override=False)
    return Secrets(anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None)
