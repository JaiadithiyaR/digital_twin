<!-- source: ns3_legacy -->
# Development Validation Records — NS-3-ERA, RETIRED (NOT comparable to current OAI history)

**OAI migration notice (read this before treating anything below as relevant):** every record in
this file was produced against the NOW-RETIRED NS-3/5G-LENA simulated telemetry source and the 5
NS-3-era components (`throughput`/`latency`/`packet_loss`/`prb_utilization`/`jitter`), which this
project has fully replaced with real OAI RAN KPI replay and 7 new components (`dl_bler`/`ul_bler`/
`dl_prb`/`ul_prb`/`dl_throughput`/`ul_throughput`/`dl_rlc_delay` — see `docs/oai_twin.md`). This
document's own `source` metadata is deliberately set to `ns3_legacy` (not a real URL/path) so
retrieval can label it as historical/not-comparable rather than silently surfacing it as current
guidance. **Kept here for historical reference, never deleted, and never cited as current system
behavior or as a comparison baseline for post-migration fidelity/adaptation outcomes.**

**Read this note before treating anything below as operational history (original note, still
true independent of the migration).** This corpus category is intended (prompt.md §35) for
historical *adaptation* records — the accumulated output of Module 19 (Lifecycle Management
Agent), which is not built yet, produced by a system that has run continuously in production and
genuinely experienced drift-triggered adaptations. Neither exists yet: this system has not been
run continuously against live/production traffic, and Module 19 does not exist to produce
structured lifecycle records even if it had. **This category is therefore placeholder-thin** —
what follows are real (not fabricated) events that genuinely occurred during THIS project's own
development/testing of Modules 14-16 (in the NS-3 era), kept here as the closest
currently-available real substitute, clearly distinguished from what a real operational history
would look like once Module 19 exists and the system has run for real.

## Record: Recalibration Agent (Module 14) development validation

- **Type**: development integration test, not a production drift-triggered event.
- **Component**: throughput.
- **What happened**: `RecalibrationAgent` retrained the `throughput` component on a recent D1
  window (real Module 2/3/4 telemetry pipeline output, not synthetic), produced candidate version
  `throughput-v2`, and registered it with `adaptation_type="recalibrate"`,
  `parent_version_id="throughput-v1"`, `status="candidate"`.
- **Concurrency proof**: while this call was in progress, a real background telemetry
  synchronizer kept writing new records into D1 the entire time (re-verified across 3 consecutive
  runs, stable ~14s duration each, not flaky).
- **Outcome**: candidate registered; never auto-promoted (Module 17, verification, does not exist
  yet — promotion remains a manual/future capability).

## Record: Regeneration Agent (Module 15) development validation

- **Type**: development integration test, not a production drift-triggered event.
- **Component**: throughput.
- **What happened**: with a mocked LLM transport (no real `ANTHROPIC_API_KEY` is configured in
  this development environment) standing in for the Anthropic API, `RegenerationAgent` generated
  a replacement `throughput` pipeline, sandboxed it (a real, separate OS subprocess actually
  trained and evaluated the candidate), and — in one specific validation run — a deliberately
  broken first candidate was REJECTED by the sandbox at the `import`/`conformance` stage, then a
  corrected second candidate was ACCEPTED, demonstrating the self-correction retry loop
  genuinely working, not just in principle.
- **Concurrency proof**: telemetry synchronization continued throughout (re-verified across 3
  consecutive runs, stable ~8s duration each).
- **Outcome**: candidate registered as `adaptation_type="regenerate"`; production untouched.

## Record: Expand-Scope Agent (Module 16) development validation

- **Type**: development integration test, not a production drift-triggered event.
- **New component**: `sinr_quality`, predicting `sinr_db` from mobility/load features — a
  genuinely new capability (none of the five existing DT components predict SINR; all treat it
  as an input) motivated by a realistic scenario: proactive handover / resource-allocation signal
  quality prediction.
- **What happened**: a design proposal (component name, target column, dependencies, required
  features) was generated and deterministically validated (novel name, real target column, known
  dependencies) before any code was written; an implementation was then generated and run through
  the identical sandbox Module 15 uses; the accepted candidate was registered with
  `adaptation_type="expand_scope"`, `parent_version_id=None` (genuinely new — no prior version to
  have a parent relationship with), `fidelity_before=None` (a new capability has no prior version
  to compare against — this is correct, not a missing value).
- **Concurrency proof**: telemetry synchronization continued throughout (re-verified across 3
  consecutive runs, stable ~11-13s duration each).
- **Separately verified**: `DTModelRegistry`/`DTOrchestrator` (Module 5, unmodified) correctly
  scheduled and ran a sixth component alongside real trained instances of all five Modules 6-10
  components, proving the dynamic-registration mechanism has no hardcoded component-count or
  name limit — using a trusted test-authored stand-in, not the raw LLM/sandbox output, per this
  project's documented safety boundary (a verified candidate is not wired into the live
  orchestrator before Module 17 exists to grant that trust).

## What a real future record will look like

Once Module 19 exists and the system runs continuously against live telemetry, each real record
will additionally include: the actual drift event that triggered it (component, severity,
timestamp), the RL observation/context PPO acted on, PPO's chosen action, real (not
config-frozen-window) fidelity-before/after computed against genuinely evolving production
telemetry, an actual verification result and explanation from Module 17, and a final
promoted/rejected status reflecting a real deterministic acceptance-gate decision — none of which
exist in the records above, since PPO, real drift, and verification were not part of these
specific development validation runs.
