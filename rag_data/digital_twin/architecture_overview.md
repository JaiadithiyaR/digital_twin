<!-- source: internal:CLAUDE.md (this repository's own architecture/state document) -->
# AI-Driven Self-Adaptive Network Digital Twin — Architecture Overview

This is a production-quality, modular AI-Driven Self-Adaptive Network Digital Twin (DT) platform
that replays telemetry from a real exported OAI (OpenAirInterface) RAN KPI dataset
(`SOURCE=OAI_RAN_REPLAY` — real-network-measured O-RAN/OAI RAN-layer KPIs, not a simulation),
keeps a continuously synchronized dynamic DT state (never a static periodically-replaced
dataset), runs dependency-aware DT prediction components, computes deterministic per-component
fidelity (plus a system-wide Unified Fidelity Score and an internal fidelity-based adaptation
trigger), reacts to two canonical adaptation triggers (an externally-sourced drift event, or the
internal fidelity-based trigger) via a knowledge-based (RAG-grounded) Decision & Root-Cause
Analysis Agent that analyzes the likely root cause and selects an adaptation strategy, executes
that strategy through an LLM-capable adaptation agent producing an isolated sandboxed candidate,
verifies the candidate with a deterministic acceptance gate, and promotes or rejects it while
preserving rollback safety.

(This project previously replayed a simulated NS-3/5G-LENA network before migrating to real OAI
RAN KPI replay — see `docs/oai_twin.md` for the migration and the full OAI component set.)

## Canonical Data Flow

```
OAI RAN KPI export -> replay (real exported CSVs) -> validation -> preprocessing
-> continuous synchronization -> dynamic D1 state -> dependency-aware DT prediction
-> fidelity evaluation (+ unified score/trigger) -> adaptation trigger (external drift OR
fidelity-based) -> Decision & Root-Cause Analysis Agent -> selected adaptation agent
-> candidate -> sandbox -> deterministic evaluation -> agentic contextual verification
-> deterministic acceptance gate -> promotion/rejection -> lifecycle management
-> continued operation.
```

## Three Distinct State Concepts (never conflated)

- **Concept A — Live network state**: the current OAI RAN KPI replay output.
- **Concept B — Dynamic DT state (D1)**: the continuously synchronized current + historical
  representation, updated by every valid telemetry record, never frozen wholesale during
  adaptation.
- **Concept C — Versioned DT prediction models**: independently versioned learned components.
  Updating a model version never replaces or freezes D1.

## The Seven DT Prediction Components and Their Dependency Graph

```
lagged RAN features
   |-> dl_bler --,                |-> ul_bler --,
   `-> dl_prb  --+-> dl_throughput +-> ul_prb   -+-> ul_throughput
dl_throughput + dl_bler -> dl_rlc_delay
```

`dl_bler`, `ul_bler`, `dl_prb`, and `ul_prb` are root nodes (lagged MAC/RLC features only, no
upstream DT predictions needed). `dl_throughput` depends on `dl_bler` + `dl_prb`; `ul_throughput`
depends on `ul_bler` + `ul_prb`. `dl_rlc_delay` depends on `dl_throughput` + `dl_bler` and is
therefore always scheduled last. These are real O-RAN/OAI RAN-layer KPIs — BLER is MAC-layer
block-error rate (not end-to-end packet loss), and RLC delay is RAN queueing/transmission delay
(not end-to-end application latency).

## Fidelity Formula (deterministic, never touched by an LLM)

For each component and each of RMSE, MAE, Wasserstein/EMD, and MK-MMD:

```
D_m = metric_m ** 2                                          (square every raw metric)
D~_m = (D_m - min(D_m)) / (max(D_m) - min(D_m) + eps)         (rolling-window min-max normalize)
S_raw,c = D~_RMSE + D~_MAE + D~_W1 + D~_MMD
FidelityScore_c = 1 - (S_raw,c / 4)
```

`eps` and the rolling-window length are config-driven, never hardcoded. Insufficient history,
NaN, Inf, empty windows, and missing samples are handled explicitly — a composite score is never
silently manufactured. Production-vs-candidate comparisons must use the same normalization
reference (the same rolling window/statistics) or the scores are not comparable.
