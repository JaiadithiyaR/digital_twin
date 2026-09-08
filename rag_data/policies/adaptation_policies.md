<!-- source: internal:config/settings.yaml (this repository's own enforced configuration) -->
# Adaptation Policies (Enforced Configuration Values)

These are the actual, currently-enforced policy values from `config/settings.yaml`, not
illustrative examples — every value below is read live by the corresponding module; changing the
config file changes the enforced behaviour directly, with no hardcoded value anywhere overriding
it.

## Acceptance Gate / Verification Policy

- `adaptation.verification_delta = 0.01` — a candidate is only accepted if
  `FidelityScore_new > FidelityScore_old + 0.01`. A candidate that is merely equal to or only
  marginally better than production is rejected; the margin exists so noise-level fluctuation
  can never flip a promotion decision.
- Fidelity comparisons between production and a candidate always share the same rolling
  normalization window/statistics (`fidelity.rolling_window_length = 200` points,
  `fidelity.epsilon = 1.0e-8` as the denominator stabilizer, `fidelity.
  min_history_for_normalization = 10` points minimum before any composite score is computed at
  all) — comparing scores computed against different reference windows is never valid.

## Concurrency Policy

- `adaptation.lock_policy = "queue"` — when a second drift event arrives for a component that
  already has an adaptation in progress, it is queued rather than coalesced or deferred/dropped.
  Telemetry ingestion itself is never blocked by this lock regardless of policy.

## Per-Strategy Adaptation Cost Penalty

Reward for the PPO decision agent is `fidelity_improvement - adaptation_cost_penalty`, with the
penalty scaled to reflect each strategy's real relative cost:

- Recalibrate: `0.02` (cheapest — retrains an existing pipeline on recent data).
- Regenerate: `0.08` (costlier — LLM-driven full pipeline rebuild, sandboxed).
- Expand Scope: `0.12` (costliest — LLM-driven design AND implementation of a genuinely new
  component, sandboxed).

## Per-Agent Training-Data Policy

- Recalibration: a `24` hour recent window, minimum `200` training rows.
- Regeneration: a `48` hour window (wider than recalibration's — rebuilding a pipeline from
  scratch reasonably wants more history than a lightweight refit), minimum `200` rows.
- Expand Scope: a `48` hour window (a brand-new component has no prior deployment to lean on, the
  same reasoning as regeneration), minimum `200` rows.
- Regeneration and Expand Scope each get up to `2` self-correction attempts
  (`max_llm_iterations`): the LLM generates candidate code, the sandbox may reject it, and the
  rejection reason is fed back to the LLM as context for a corrected attempt.

## Sandbox Execution Policy

- `sandbox.timeout_seconds = 120` — any candidate execution exceeding this is killed and
  rejected, never allowed to hang the adaptation workflow.
- `sandbox.max_output_bytes = 1000000` — captured stdout/stderr from a candidate is truncated
  past this size, bounding an output-flooding vector.
- The sandboxed subprocess environment is built from scratch (never inherited from the parent
  process) — secrets such as `ANTHROPIC_API_KEY` are never visible inside it.

## Drift Detection Interface Policy

- `drift.valid_components = [throughput, latency, packet_loss, prb_utilization, jitter]` — an
  incoming drift event naming any other component is rejected (quarantined), never routed to a
  strategy, since it cannot be mapped to a real, adaptable DT scope.
- `drift.severity_range = [0.0, 1.0]` — a severity outside this inclusive range is likewise
  rejected rather than clamped, since a malformed trigger must fail safely rather than be guessed
  at.
