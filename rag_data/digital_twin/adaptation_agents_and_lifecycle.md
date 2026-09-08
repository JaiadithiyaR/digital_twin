<!-- source: internal:CLAUDE.md (this repository's own architecture/state document) -->
# Adaptation Agents and Candidate Lifecycle

## Exactly Six Agents

This system has exactly six agents — everything else (telemetry, preprocessing,
synchronization, DT models, orchestrator, fidelity engine, drift interface, RAG, registry,
sandbox, LLM client, storage, config) is an ordinary software component, never renamed into an
"agent":

1. **PPO RL Decision Agent** (Module 13) — decides WHAT adaptation strategy to apply. Action
   space is `Discrete(3)`: 0 = Recalibrate, 1 = Regenerate, 2 = Expand Scope. Observation: five
   per-component fidelity values, an affected-component one-hot flag, drift severity, the
   previous action (one-hot), the previous reward, plus relevant network state. Reward is
   `fidelity_improvement - adaptation_cost_penalty`. PPO never generates code, never sets
   fidelity formulas, never does LLM reasoning — it only picks the strategy index, and the
   trained/saved policy is what runs at inference time (a deterministic fallback exists only for
   genuine PPO infrastructure failure, always logged, never a silent substitute).
2. **Recalibration Agent** (Module 14) — decides HOW to recalibrate: retrains an EXISTING
   production component on a recent telemetry window when its structure is still valid but its
   learned behaviour has drifted. Never a blind periodic retrain — recalibration happens only
   because PPO selected it.
3. **Regeneration Agent** (Module 15, LLM-driven) — decides HOW to regenerate: rebuilds an
   existing component's pipeline from scratch via LLM-generated, sandboxed code when the current
   architecture can no longer represent the changed behaviour.
4. **Expand-Scope Agent** (Module 16, LLM-driven) — decides HOW to add capability: designs and
   builds a genuinely NEW DT component, via the same LLM-generation-plus-sandbox pipeline as
   Regeneration, when network behaviour reveals a phenomenon the DT does not yet represent at
   all.
5. **Agentic Verification Agent** (Module 17) — decides ACCEPT/REJECT. Verification is
   autonomous and never simply trusts the LLM: it has a deterministic layer (recomputing RMSE,
   MAE, Wasserstein, MK-MMD, and the composite FidelityScore using the exact fidelity engine) and
   an agentic reasoning layer (the LLM may explain WHY a candidate improved or failed, but can
   never override the deterministic acceptance gate).
6. **Lifecycle Management Agent** (Module 19) — records and explains: every adaptation event
   produces a structured, auditable lifecycle record and a human-readable maintenance report.

## Adaptation Safety Rules

- **Candidate isolation**: no candidate has unrestricted write access to production source,
  model artifacts, registry, configuration, secrets, or unrelated filesystem locations. A
  candidate can never promote itself — only trusted deterministic logic promotes.
- **Acceptance gate**: `FidelityScore_new > FidelityScore_old + delta` (delta is config-driven,
  never hardcoded) AND the candidate loads, its interface is valid, its tests pass, its
  evaluation succeeds, there is no critical regression, its outputs are valid, and its safety
  checks pass. Any failed mandatory condition means REJECT.
- **Production version safety**: promotion is atomic from the registry's perspective; previous
  known-good versions are always retained for rollback. A failed candidate has zero production
  effect.
- **Continuous operation**: telemetry ingestion, D1 synchronization, and the current production
  model keep serving/updating throughout recalibration, regeneration, expand-scope, sandbox
  execution, and verification. Only the candidate and its evaluation context are isolated.
- **LLM-generated code is never production code until sandboxed and verified.** Generated code
  goes to a candidate workspace, then syntax -> static -> import validation -> unit tests ->
  training -> evaluation -> deterministic verification -> promotion. The LLM can never override
  the deterministic acceptance gate.
- **RAG is read-only contextual knowledge** — this exact knowledge base. Agents may retrieve
  from it; nothing in this system uses it as a write path back into DT state, models, or the
  registry.
