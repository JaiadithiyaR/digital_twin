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
from pydantic import BaseModel, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"

# The 5 retired NS-3-era component ids — structurally forbidden from reappearing in live config.
_RETIRED_COMPONENT_IDS = frozenset({"throughput", "latency", "packet_loss", "prb_utilization", "jitter"})


class MockTelemetryConfig(BaseModel):
    """OAI-shaped mock telemetry — unit tests only (prompt2.md: "mock source stays for unit
    tests only, tagged SOURCE=MOCK"). Never used in `--mode live`/replay runs."""

    seed: int
    num_sessions: int
    emit_interval_seconds: float
    missing_feature_rate: float  # fraction of feature values randomly NaN'd — exercises the same quality-tracking path real gaps would


class ReplayConfig(BaseModel):
    """Module 1 replacement — OAI CSV replay pacing (prompt2.md Step 4)."""

    speed: float  # 0 = as fast as possible (tests/experiments); 1.0 = real time; >1 = accelerated
    default_split: Literal["train", "test"]  # "val" retired (phase13 §0/§1.1): no held-out val split anymore, only train/tune pool + strictly-held-out test


class OaiPreprocessingConfig(BaseModel):
    max_missing_ratio: float  # above this, a window/tick is flagged low-quality rather than used


class TelemetryConfig(BaseModel):
    source: Literal["mock", "oai_replay"]
    mock: MockTelemetryConfig
    replay: ReplayConfig
    preprocessing: OaiPreprocessingConfig


class DataSplitsConfig(BaseModel):
    """Config-driven run-id splits (prompt2.md: "so run 10 can be added as an extra test run
    later ... without code changes"). Disjointness is enforced at load time, not left to the
    caller to remember.

    Phase 13 (claude_code_prompt_phase13.md §0/§1.1) redesign: there is no longer a separate,
    strictly-held-out "val" split — `train` is now the full TRAIN/TUNE pool (leave-one-run-out CV
    folds over it, then a final retrain on all of it combined), and `test` remains strictly held
    out (never used for training, tuning, debugging, or smoke tests). `smoke_test_run` names
    which TRAIN run doubles as the designated smoke-test run, so `test` is never touched before
    genuine TEST-time evaluation."""

    train: list[int]
    test: list[int]
    smoke_test_run: int

    @model_validator(mode="after")
    def _check_disjoint(self) -> "DataSplitsConfig":
        overlap = set(self.train) & set(self.test)
        if overlap:
            raise ValueError(f"run(s) {sorted(overlap)} appear in both 'train' and 'test' splits")
        if self.smoke_test_run not in self.train:
            raise ValueError(f"smoke_test_run={self.smoke_test_run} must be a member of train={self.train}")
        return self


class DataConfig(BaseModel):
    """Module 1 replacement — the read-only OAI export this Digital Twin now replays telemetry
    from. Never written to, never copied into git (prompt2.md Fixed constraints)."""

    oai_root: str
    splits: DataSplitsConfig


class SynchronizationConfig(BaseModel):
    batch_size: int
    batch_timeout_seconds: float
    # How long stop() waits for background threads to actually exit before giving up.
    shutdown_join_timeout_seconds: float


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
    # Minimum seconds between D1 Parquet rewrites (0 = write-through). In-memory D1 state is
    # always current; this only bounds how often the whole table is re-serialized to disk.
    persist_min_interval_seconds: float = 0.0


class ModelSpec(BaseModel):
    model_type: str
    params: dict[str, float | int | str | None]


class DtModelsConfig(BaseModel):
    """The 7 OAI KPI prediction components (prompt2.md Step 5 — replaces the 5 NS-3-era
    components entirely). `extra="forbid"` makes loading a config that still names an old
    component id (e.g. a stray `jitter:` block) a hard validation failure at startup, not a
    silently-ignored extra key — the structural half of the addendum's "fail at startup if any
    old component id ... appears" requirement."""

    model_config = {"extra": "forbid"}

    bootstrap_min_rows: int
    dl_throughput: ModelSpec
    ul_throughput: ModelSpec
    dl_rlc_delay: ModelSpec
    dl_prb: ModelSpec
    ul_prb: ModelSpec
    dl_bler: ModelSpec
    ul_bler: ModelSpec


class MkMmdConfig(BaseModel):
    kernel: str
    gamma: float | None


class ComponentPolicyOverride(BaseModel):
    """Per-component override of the default adaptation policy. An unset field inherits the
    default (`trigger_threshold` / `trigger_min_consecutive_evaluations`)."""

    model_config = {"extra": "forbid"}

    threshold: float | None = Field(default=None, gt=0)
    min_consecutive_evaluations: int | None = Field(default=None, ge=1)


class FidelityConfig(BaseModel):
    epsilon: float
    rolling_window_length: int
    min_history_for_normalization: int
    metrics: list[str]
    mk_mmd: MkMmdConfig
    # Unified Fidelity Score (prompt.md §16a) — a system-health REPORTING aggregate only; it no
    # longer drives adaptation. `unified_weights` null means equal weighting across all
    # currently-defined components — never hardcode unequal weights in code.
    unified_weights: dict[str, float] | None
    unified_min_components: int
    # Per-component adaptation policy (src/fidelity/component_policy.py): component c is adapted
    # when its own FidelityScore_c < its threshold for its min-consecutive count of fresh
    # evaluation cycles. These two are the defaults for every component...
    trigger_threshold: float
    trigger_min_consecutive_evaluations: int  # debounce/hysteresis — a single bad cycle must never fire the trigger
    # ...and this optionally overrides either one per component id (unlisted -> defaults).
    component_policies: dict[str, ComponentPolicyOverride] = {}


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


class DecisionAgentFallbackConfig(BaseModel):
    # Used only on genuine decision-agent infrastructure failure, never a routine substitute.
    enabled: bool
    default_strategy: Literal["recalibrate", "regenerate", "expand_scope"]


class DecisionAgentConfig(BaseModel):
    """No training hyperparameters — this is a direct LLM call, not a trained policy."""

    action_mapping: dict[int, str]  # the fixed 3-strategy enum, never a 4th/5th
    fallback: DecisionAgentFallbackConfig


class RecalibrationConfig(BaseModel):
    training_window_hours: int
    min_training_rows: int
    # Train on the original TRAIN pool (the data production was bootstrapped on) PLUS the recent
    # window, recent rows up-weighted — instead of the recent window alone, which early in a run is
    # a few hundred rows and can never match a model trained on the full pool.
    include_reference_data: bool = True
    recent_sample_weight: float = Field(default=5.0, gt=0)


class RegenerationConfig(BaseModel):
    max_llm_iterations: int  # self-correction attempts: generate -> sandbox rejects -> retry with feedback
    training_window_hours: int
    min_training_rows: int
    include_reference_data: bool = True  # same reasoning as recalibration (no weighting — the generated code trains itself)


class ExpandScopeConfig(BaseModel):
    max_llm_iterations: int  # shared retry budget for both the design and implementation steps
    training_window_hours: int
    min_training_rows: int


class AdaptationConfig(BaseModel):
    lock_policy: Literal["queue", "coalesce", "defer"]
    # Different components adapt concurrently on worker threads (never two adaptations of the
    # same component). false = one adaptation at a time, still off the fidelity loop's thread.
    allow_concurrent_components: bool
    max_concurrent_adaptations: int = Field(default=2, ge=1)
    # Adaptation loop: a REJECTed (or failed) attempt is retried — with the rejection details fed
    # back to the decision agent and the agents — until a candidate is ACCEPTed, the component
    # recovers above its threshold, or this many attempts have been made.
    max_adaptation_attempts: int = Field(default=4, ge=1)
    # stop(): how long to wait for in-flight adaptations before recording them as interrupted.
    shutdown_wait_seconds: float = Field(default=20.0, ge=0)
    recalibration: RecalibrationConfig
    regeneration: RegenerationConfig
    expand_scope: ExpandScopeConfig
    verification_delta: float


class SandboxConfig(BaseModel):
    workspace_root: str
    timeout_seconds: int
    max_output_bytes: int


class PeriodicRetrainBaselineConfig(BaseModel):
    # phase13 §4: "all components retrained every N seconds of replayed data" — every one of the
    # 7 real OAI components is recalibrated on this fixed schedule, never a single hardcoded one.
    interval_seconds: int = Field(gt=0)
    # Blind retraining deploys every retrained model unconditionally (ablation step 1); set true
    # to put the scheduled candidates through the same verification gate as the adaptive modes.
    use_verification_gate: bool = False


class RuleBasedBaselineConfig(BaseModel):
    recalibrate_max_severity: float
    regenerate_max_severity: float


class BaselinesConfig(BaseModel):
    """Baseline experiment modes (prompt2.md Step 6, extended by claude_code_prompt_phase13.md
    §4) — static/periodic_retrain/rule_based/full_system, run on the same OAI replay splits/
    seed/fidelity settings for a fair comparison. 'static' needs no config of its own (never
    adapts at all); 'full_system' is just the already-configured real system, run once per
    `seeds` entry."""

    window_seconds: int = Field(default=60, gt=0)  # one evaluation/decision window of replayed DATA time
    max_windows: int | None = Field(default=None, ge=1)  # None = the whole TEST run
    seeds: list[int]
    periodic_retrain: PeriodicRetrainBaselineConfig
    rule_based: RuleBasedBaselineConfig
    fallback_rate_invalid_threshold: float


class RagConfig(BaseModel):
    chroma_path: str
    collection_name: str
    corpus_dirs: dict[str, str]
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_model: str


class LlmEffortByRoleConfig(BaseModel):
    """Phase 13 §3.4: thinking level/budget configurable PER ROLE, never one fixed value for
    every call. `GoogleClient.complete*(..., effort=...)` already accepts a per-call override;
    this is just where each role's own value is read from (never hardcoded in agent code)."""

    decision: Literal["low", "medium", "high", "xhigh", "max"]
    codegen: Literal["low", "medium", "high", "xhigh", "max"]
    explanation: Literal["low", "medium", "high", "xhigh", "max"]


class LlmCachingConfig(BaseModel):
    """Phase 13 §3.3: explicit context caching of the static prompt prefix (system instructions +
    adaptation policies + OAI/DT doc summary + feature_schema.json, in that fixed order) via the
    real `client.caches` API (verified present on the installed google-genai SDK — see
    src/llm/google_client.py). One cache per experiment run; TTL = the expected run length."""

    enabled: bool
    ttl_seconds: float  # "TTL = the expected run length" (phase13 §3.3) — set per experiment, this is the config default
    # A conservative, CHARACTER-based proxy for "the model's minimum cacheable size" (the real
    # minimum Gemini enforces is token-based and model-specific, not documented in a way this
    # client can look up generically) — below this, caching is skipped (logged, never attempted
    # and left to fail) rather than guessed at.
    min_cacheable_chars: int


class LlmRatesConfig(BaseModel):
    """Phase 13 §3.5: USD cost-per-token rates, config-driven so a price change (the spec's own
    "prices change on 1 January 2027" note) never requires a code change. These are a
    best-effort estimate for this project's configured model, not a verified live rate card —
    update this block when real, current pricing is available."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float
    cache_storage_per_million_per_hour: float


class LlmConfig(BaseModel):
    model: str
    max_tokens: int
    # Unlike the old Anthropic Messages API, the installed google-genai SDK's GenerateContentConfig
    # DOES expose a real sampling-determinism control (verified directly against types.py, not
    # assumed) — `temperature` is genuine here, not a fiction. `None` omits it (API default).
    temperature: float | None
    # `effort` maps onto Gemini's ThinkingLevel (MINIMAL/LOW/MEDIUM/HIGH — no xhigh/max; those two
    # clamp down to HIGH). See src/llm/google_client.py's module docstring for the full mapping.
    # The shared/default value — `effort_by_role` below overrides it per call site (phase13 §3.4).
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None
    effort_by_role: LlmEffortByRoleConfig
    timeout_seconds: int
    max_retries: int  # shared budget for both this client's transport retry loop and its structured-output validation retry loop
    retry_backoff_seconds: float  # base for this client's own exponential backoff (capped internally); the SDK's own internal retry defaults to "never retry" so this is the only retry/backoff path
    caching: LlmCachingConfig
    budget_usd: float  # phase13 §3.6 — the budget guard's ceiling for one experiment run
    service_tier: Literal["standard", "flex"]  # phase13 §3.7 — "flex" only for speed=0 experiments, never live demos
    rates: LlmRatesConfig


class LoggingConfig(BaseModel):
    level: str
    format: Literal["json", "text"]
    log_dir: str
    # Dedicated, human-auditable logs written by the live orchestrator (paths relative to repo root):
    component_fidelity_log: str = "logs/component_fidelity.csv"  # one row per component per fidelity cycle
    adaptation_decisions_log: str = "logs/adaptation_decisions.jsonl"  # one line per handled trigger, incl. failures
    console_transcript_dir: str | None = "logs/console"  # plain-text copy of each run's console output; null disables


class LifecycleConfig(BaseModel):
    records_path: str
    reports_dir: str


class EvaluationConfig(BaseModel):
    window_size: int


class Settings(BaseModel):
    """Root configuration object. Immutable after load — treat as read-only."""

    model_config = {"frozen": True}

    environment: Literal["demo", "live"]
    data: DataConfig
    telemetry: TelemetryConfig
    synchronization: SynchronizationConfig
    storage: StorageConfig
    dt_models: DtModelsConfig
    fidelity: FidelityConfig
    drift: DriftConfig
    decision_agent: DecisionAgentConfig
    adaptation: AdaptationConfig
    sandbox: SandboxConfig
    baselines: BaselinesConfig
    rag: RagConfig
    llm: LlmConfig
    logging: LoggingConfig
    lifecycle: LifecycleConfig
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def _forbid_retired_ns3_component_ids(self) -> "Settings":
        """Config-level fail-fast guard (OAI migration addendum item 6). `DtModelsConfig`'s own
        `extra="forbid"` + explicit new field names already make an old component id there a
        structural validation error; this additionally covers the one place an old id is still a
        free-form string rather than a typed field: `drift.valid_components`."""
        bad = _RETIRED_COMPONENT_IDS.intersection(self.drift.valid_components)
        if bad:
            raise ValueError(
                f"config/settings.yaml drift.valid_components still names retired NS-3-era "
                f"component id(s) {sorted(bad)} — the OAI migration replaced the entire "
                f"component set (see CLAUDE.md); update this list to the 7 OAI component ids."
            )
        return self

    def resolve_path(self, relative: str) -> Path:
        """Resolve a config-declared path against the repository root."""
        return REPO_ROOT / relative


class Secrets(BaseModel):
    """Loaded from environment/.env only. Never logged, never merged into Settings."""

    google_api_key: str | None = Field(default=None, repr=False)

    def require_google_key(self) -> str:
        if not self.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. Copy .env.example to .env and "
                "fill it in before using any LLM-driven agent (decision & root-cause analysis, "
                "regeneration, expand-scope, verification reasoning, lifecycle explanations)."
            )
        return self.google_api_key

    def __repr__(self) -> str:  # never let a stray print/log leak the key
        return "Secrets(google_api_key=<redacted>)"


@lru_cache(maxsize=1)
def load_settings(path: Path | None = None) -> Settings:
    settings_path = path or DEFAULT_SETTINGS_PATH
    with settings_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # The raw OAI export lives outside the repo at a machine-specific path, so it is set per
    # machine via OAI_ROOT (environment or .env) rather than committed in settings.yaml.
    load_dotenv(REPO_ROOT / ".env", override=False)
    if os.environ.get("OAI_ROOT"):
        raw.setdefault("data", {})["oai_root"] = os.environ["OAI_ROOT"]
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def load_secrets() -> Secrets:
    load_dotenv(REPO_ROOT / ".env", override=False)
    # GOOGLE_API_KEY takes precedence over GEMINI_API_KEY, matching google-genai's own SDK
    # precedence (verified in google.genai._api_client.get_env_api_key).
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or None
    return Secrets(google_api_key=key)
