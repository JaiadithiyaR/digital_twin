<!-- source: internal:CLAUDE.md (this repository's own architecture/state document) -->
# AI-Driven Self-Adaptive Network Digital Twin — Architecture Overview

This is a production-quality, modular AI-Driven Self-Adaptive Network Digital Twin (DT) platform
that ingests telemetry from an NS-3/5G-LENA simulated 5G network, keeps a continuously
synchronized dynamic DT state (never a static periodically-replaced dataset), runs
dependency-aware DT prediction components, computes deterministic per-component fidelity, reacts
to externally-sourced drift events via a trained PPO policy that selects an adaptation strategy,
executes that strategy through an LLM-capable adaptation agent producing an isolated sandboxed
candidate, verifies the candidate with a deterministic acceptance gate, and promotes or rejects
it while preserving rollback safety.

## Canonical Data Flow

```
NS-3/5G-LENA -> telemetry extraction -> transport -> validation -> preprocessing
-> continuous synchronization -> dynamic D1 state -> dependency-aware DT prediction
-> fidelity evaluation -> external drift event -> PPO -> selected adaptation agent
-> candidate -> sandbox -> deterministic evaluation -> agentic contextual verification
-> deterministic acceptance gate -> promotion/rejection -> lifecycle management
-> continued operation.
```

## Three Distinct State Concepts (never conflated)

- **Concept A — Live network state**: the current NS-3/5G-LENA output.
- **Concept B — Dynamic DT state (D1)**: the continuously synchronized current + historical
  representation, updated by every valid telemetry record, never frozen wholesale during
  adaptation.
- **Concept C — Versioned DT prediction models**: independently versioned learned components.
  Updating a model version never replaces or freezes D1.

## The Five DT Prediction Components and Their Dependency Graph

```
raw telemetry
   |-> throughput --,
   `-> packet loss -+-> latency --,
                                   +-> jitter
throughput + latency + packet loss -'
PRB utilization: independent/optional
```

Throughput and packet loss are root nodes (no upstream DT predictions needed). Latency depends on
throughput (and, in the full spec, packet loss). Jitter depends on throughput, latency, and
packet loss, and is therefore always scheduled last. PRB utilization depends on throughput but is
independently toggleable without affecting any other component.

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
