# CLAUDE.md — AI-Driven Self-Adaptive Network Digital Twin

This is the living architecture/state document for this repository. It is updated after every
module is implemented so that any future Claude Code session (or human) can resume work without
rediscovering decisions already made. `prompt.md` is the authoritative specification; where this
file and `prompt.md` conflict, `prompt.md` wins — update this file to match, not the reverse.

Companion file: `IMPLEMENTATION_STATUS.md` tracks per-module build/test status and next tasks.
Read both before making changes.

---

## 1. Project Purpose

A production-quality, modular AI-Driven Self-Adaptive Network Digital Twin (DT) platform that:

- Ingests telemetry from an NS-3/5G-LENA simulated 5G network.
- Keeps a continuously synchronized, dynamic DT state (never a static periodically-replaced
  dataset).
- Runs dependency-aware DT prediction components (throughput, latency, packet loss, PRB
  utilization, jitter).
- Computes deterministic per-component fidelity (RMSE, MAE, Wasserstein, MK-MMD → composite
  score).
- Reacts to externally-sourced drift events via a trained PPO policy that selects an adaptation
  *strategy* (Recalibrate / Regenerate / Expand Scope).
- Executes that strategy through one of three LLM-capable adaptation agents, which produce an
  isolated, sandboxed candidate.
- Verifies the candidate with a deterministic acceptance gate (LLM may explain, never override).
- Promotes or rejects the candidate, preserving rollback safety, and records every step in an
  auditable lifecycle history.
- Operates continuously and autonomously — no human-in-the-loop for normal adaptation.

The system must be genuinely executable end-to-end (mock/demo telemetry source is acceptable when
NS-3/5G-LENA cannot be built in the current environment, but it must never be reported as real
NS-3 output — see Constraints below).

## 2. Architecture — 19 Modules + D1 + D2 (canonical diagram: `fig-dataflow.png`)

The module numbers below are the permanent identifiers for this project — use them in code
comments, commit messages, and status tracking, not ad hoc names.

| # | Module | Responsibility | Package (planned) |
|---|--------|-----------------|--------------------|
| 1 | NS-3 / OAI Network | External environment; ground-truth network behaviour, telemetry source | `ns3_sim/` |
| 2 | Telemetry Collection & Preprocessing | Validate, clean, normalize raw telemetry into DT-ready records | `src/telemetry/` |
| 3 | Continuous Synchronization | Continuously push validated telemetry into D1 (no manual sync step) | `src/synchronization/` |
| D1 (4) | DT Basic Model — state & history store | Current + historical DT state; component registry | `src/dt_models/d1_model_store.py`, `component_registry.py` |
| 5 | DT Functional Model | Common `train/predict/evaluate/save/load` interface + dependency-aware orchestrator | `src/dt_models/base.py`, `model_registry.py`, `orchestrator.py` |
| 6 | Throughput Model | Predict throughput (Mbps) — **implemented** | `src/dt_models/throughput.py`, `regressors.py` |
| 7 | Latency Model | Predict latency (ms); depends on throughput (+ packet loss once Module 8 exists) — **implemented** | `src/dt_models/latency.py` |
| 8 | Packet-Loss Model | Predict packet loss (%) — **implemented** | `src/dt_models/packet_loss.py` |
| 9 | PRB-Utilization Model | Predict PRB utilization (%); independently toggleable — **implemented** | `src/dt_models/prb_utilization.py` |
| 10 | Jitter Model | Predict jitter (ms); depends on latency + throughput + packet loss — **implemented, all 5 DT models complete** | `src/dt_models/jitter.py` |
| 11 | Drift Detection Interface | External/partner drift event contract, validation, mock source | `src/drift/` |
| 12 | Fidelity Evaluation Module | Deterministic RMSE/MAE/Wasserstein/MK-MMD → composite fidelity score — **implemented** | `src/fidelity/{metrics.py,evaluator.py}` |
| 13 | RL Decision Agent (Agent 1) | PPO chooses WHAT adaptation strategy to apply | `src/adaptation/rl_env.py`, `rl_agent.py` |
| 14 | Recalibration Agent (Agent 2) | Retrain existing component on a recent window | `src/adaptation/recalibration_agent.py` |
| 15 | Regeneration Agent (Agent 3) | LLM-driven rebuild of an inadequate component | `src/adaptation/regeneration_agent.py` |
| 16 | Expand-Scope Agent (Agent 4) | LLM-driven creation of a new DT component | `src/adaptation/expand_scope_agent.py` |
| 17 | Agentic Verification Agent (Agent 5) | Deterministic acceptance gate + LLM contextual reasoning | `src/adaptation/verification_agent.py` |
| D2 (18) | RAG Knowledge Base | Read-only vector store (O-RAN specs, DT docs, policies, adaptation history) | `src/rag/rag_kb.py` |
| 19 | Lifecycle Management Agent (Agent 6) | Auditable adaptation records + vendor maintenance report | `src/adaptation/lifecycle_agent.py` |

Canonical flow (do not reorder or bypass stages):

```
NS-3/5G-LENA → telemetry extraction → transport → validation → preprocessing
→ continuous synchronization → dynamic D1 state → dependency-aware DT prediction
→ fidelity evaluation → external drift event → PPO → selected adaptation agent
→ candidate → sandbox → deterministic evaluation → agentic contextual verification
→ deterministic acceptance gate → promotion/rejection → lifecycle management
→ continued operation.
```

Model dependency graph (module 5 orchestrator must derive this from the registry, not hardcode
call order):

```
raw telemetry
   ├→ throughput ──┐
   └→ packet loss ─┼→ latency ──┐
                                 ├→ jitter
throughput + latency + packet loss ──┘
PRB utilization: independent/optional (enable_prb_model)
```

## 3. Exactly Six Agents

1. PPO RL Decision Agent (module 13) — decides WHAT.
2. Recalibration Agent (module 14) — decides HOW to recalibrate.
3. Regeneration Agent (module 15) — decides HOW to regenerate (LLM).
4. Expand-Scope Agent (module 16) — decides HOW to add capability (LLM).
5. Agentic Verification Agent (module 17) — decides ACCEPT/REJECT.
6. Lifecycle Management Agent (module 19) — records/explains.

Everything else (telemetry, preprocessing, synchronization, DT models, orchestrator, fidelity
engine, drift interface, RAG, registry, sandbox, Anthropic client, storage, config) is an ordinary
software component, not an agent. Do not rename components into "agents".

## 4. Three Distinct State Concepts (never conflate)

- **A. Live network state** — current NS-3/5G-LENA output.
- **B. Dynamic DT state (D1)** — continuously synchronized current + historical representation.
  Updated by every valid telemetry record. Never frozen wholesale during adaptation.
- **C. Versioned DT prediction models** — independently versioned learned components. Updating a
  model version must not replace or freeze D1.

## 5. Fidelity Formula (module 12) — deterministic, LLM must never touch this

For each component and each of RMSE, MAE, Wasserstein/EMD, MK-MMD:

```
D_m = metric_m ** 2                     # square every raw metric
D~_m = (D_m - min(D_m)) / (max(D_m) - min(D_m) + eps)   # rolling-window min-max normalize
S_raw,c = D~_RMSE + D~_MAE + D~_W1 + D~_MMD
FidelityScore_c = 1 - (S_raw,c / 4)
```

- `eps` and rolling-window length are config-driven (`config/settings.yaml`), never hardcoded.
- Handle constant windows, insufficient history, NaN, Inf, empty windows, missing samples
  explicitly — never silently manufacture a score.
- Production vs candidate comparisons MUST use the same normalization reference (same rolling
  window/statistics) or scores are not comparable.

## 6. PPO (module 13)

- Stable-Baselines3 PPO + Gymnasium. Genuine trained policy — never a disguised rule-based
  function.
- Action space: `Discrete(3)` — `0 = Recalibrate`, `1 = Regenerate`, `2 = Expand Scope`. Never add
  a 4th/5th action.
- Observation: 5 per-component fidelity values + affected-component one-hot + drift severity +
  previous action (one-hot) + previous reward (plus relevant network state, kept
  well-dimensioned).
- Reward: `fidelity_improvement - adaptation_cost_penalty` (penalty configurable, may be 0).
- Runtime uses the trained/saved policy for inference. A deterministic fallback is permitted only
  on PPO infrastructure failure, and every use of it must be logged — never a silent substitute.
- PPO never generates code, never sets fidelity formulas, never does LLM reasoning. It only picks
  the strategy index.

## 7. LLM Rules (Anthropic direct API — no gateway/LiteLLM)

- Centralized client in `src/llm/anthropic_client.py`; model name from config, never a hardcoded
  deprecated model ID; API key from environment (`ANTHROPIC_API_KEY`) only, never committed.
- Used for: recalibration training-window reasoning (optional), regeneration code generation,
  expand-scope component generation, verification contextual reasoning, lifecycle human-readable
  explanations.
- Never used for: RMSE/MAE/Wasserstein/MK-MMD, the fidelity formula, PPO action selection,
  deterministic version comparison, basic telemetry validation, deterministic model training.
- LLM output is always untrusted: generated code goes to a candidate workspace, then syntax →
  static → import validation → unit tests → training → evaluation → deterministic verification →
  promotion. LLM-generated code never becomes production code directly, and the LLM can never
  override the deterministic acceptance gate.

## 8. Adaptation Safety Rules

- Candidate isolation: no candidate has unrestricted write access to production source, model
  artifacts, registry, configuration, secrets, or unrelated filesystem locations. A candidate can
  never promote itself — only trusted deterministic logic promotes.
- Acceptance gate (module 17): `FidelityScore_new > FidelityScore_old + delta` (delta config,
  never hardcoded) AND candidate loads, interface valid, tests pass, evaluation succeeds, no
  critical regression, outputs valid, safety checks pass. Any failed mandatory condition → REJECT.
- Production version safety: promotion is atomic from the registry's perspective; previous
  known-good versions are always retained for rollback. A failed candidate has zero production
  effect.
- Concurrency: one adaptation per component at a time via a component-scoped adaptation lock;
  competing drift events for a locked component are queued/coalesced/deferred (policy documented
  and tested where implemented) — telemetry ingestion itself is never blocked.
- Continuous operation: telemetry ingestion, D1 synchronization, and the current production model
  keep serving/updating throughout recalibration/regeneration/expand-scope/sandbox/verification.
  Only the candidate and its evaluation context are isolated.
- Drift detection algorithm itself is out of scope — only the external event contract, schema
  validation, normalization adapter, and mock source (module 11) are implemented here.
- Bootstrap models are trained once at init; normal telemetry processing never auto-retrains —
  only an explicit adaptation workflow retrains/replaces a model.

## 9. Repository Conventions

- Structure follows `prompt.md` §53 (`config/`, `ns3_sim/`, `src/{telemetry,synchronization,
  dt_models,fidelity,drift,adaptation,llm,rag,sandbox,registry,storage,common}/`, `data/`,
  `rag_data/`, `tests/{unit,integration,e2e}/`, `scripts/`).
- All tunable parameters live in `config/settings.yaml`; secrets only via `.env`
  (`.env.example` documents required keys, never real values).
- Real vs mock sources are physically separate modules (`zmq_source.py` vs `mock_source.py`,
  `drift_detector.py` vs `mock_drift_source.py`) and every record/log line identifies
  `SOURCE=NS3_5G_LENA` or `SOURCE=MOCK` — never conflated.
- Type hints + dataclasses/Pydantic throughout; no giant files, no circular imports, no global
  mutable state, no magic constants outside config.
- Comments explain WHY (non-obvious architecture, safety boundaries, dependency rationale), never
  WHAT.
- Update this file and `IMPLEMENTATION_STATUS.md` after completing each module — do not let them
  drift from actual repository state.

## 10. Commands to Run

```bash
# Python environment (already created — .venv/, Python 3.14.4)
source .venv/bin/activate
pip install -r requirements.txt          # idempotent
python scripts/setup.py                  # verifies dirs/.env/deps/config, non-destructive

# NS-3 / 5G-LENA (vendored, gitignored — see ns3_sim/README.md)
./scripts/setup_ns3.sh                   # idempotent clone (ns-3.48 + nr v5.1) + configure + build
cd ns3_sim/ns-3-dev && ./ns3 run <example>   # sanity-check the toolchain

# Main application (Phase 11 — implemented; see §12's Phase 11 entry for a real validated run)
python -m src.main --mode demo                   # mock telemetry, real continuous loop, runs forever (Ctrl-C to stop)
python -m src.main --mode live                    # ZeroMQ telemetry (Module 1's NS-3 exporter), real continuous loop
python -m src.main --mode demo --max-drift-events 1   # bounded run — stop after one adaptation cycle (demo/validation only)
python scripts/run_orchestrator_demo.py           # permanent, real, unattended one-cycle validation script — see §12
```

## 11. Testing Commands

```bash
.venv/bin/python -m pytest tests/unit -v          # 443 passed, 1 skipped as of this writing
.venv/bin/python -m pytest tests/integration -v   # 44 passed (Module 2 pipeline + Module 3 continuous sync + D1 wiring + orchestrator<-D1 + all five DT models' training+orchestrator + fidelity engine vs. real predictions, mock + real zmq + Module 11 drift pipeline vs. real config + Module 13 real PPO training run + Module 14 real candidate + concurrent-telemetry proof + Module 15 real sandboxed candidate + concurrent-telemetry proof + Module 16 real sandboxed new component + concurrent-telemetry proof + six-component dynamic orchestrator proof + D2 real rag_data/ corpus ingestion + genuine semantic embeddings + Module 17 real candidate verified end-to-end + Module 19 real full drift->PPO->agent->verification->lifecycle cycle + Phase 11 real orchestrator cycle w/ continuous-telemetry proof)
.venv/bin/python -m pytest tests/e2e -v           # empty so far
.venv/bin/python -m pytest tests -q               # 487 total tests, 0 collection errors — see IMPLEMENTATION_STATUS.md's Phase 11 entry: 478 confirmed passing/1 skipped in a clean full run + this turn's 9 new tests confirmed passing separately; a single combined 487-test run has been repeatedly killed by this environment's memory manager, not a code issue
```

## 12. Current State

**Phase 1 (prompt.md §67) in progress.** Done: full directory scaffold, git init (branch `main`,
no commits — commits only on explicit user request, per this project's operating rules), `.venv`
with all core dependencies installed and import-verified (`torch` reports `cuda_available=True` —
an NVIDIA GPU is present and will be used for PPO training where beneficial; CPU path remains the
required fallback per prompt.md §58), `config/settings.yaml` (all §40 tunables), `.env.example` /
`.env` (secrets never in YAML), `src/common/config.py` (pydantic-validated frozen `Settings` +
separate redacted `Secrets`), `src/common/logging.py` (structured JSON logging with secret
redaction). Both have passing unit tests.

NS-3/5G-LENA (Module 1): environment inspected — this machine's toolchain (g++ 15.2.0, cmake
4.2.3, ninja 1.13.2, python 3.14.4) satisfies plain ns-3.48 with **no root required**. Compatible
pair selected: **ns-3.48 + 5G-LENA `nr` v5.1** (newest matched row in the official compatibility
table). `nr` module prerequisites and our own planned ZeroMQ telemetry exporter's C++ dependencies
(sqlite3, libeigen3-dev, libzmq3-dev, libzmq5, cppzmq-dev) needed sudo, which this session could
not run non-interactively (`sudo: interactive authentication is required`) — the user ran the
install command directly. `ns3_sim/ns-3-dev` (ns-3.48) + `ns3_sim/ns-3-dev/contrib/nr` (v5.1) are
cloned (shallow, gitignored — see §2 table and `ns3_sim/README.md` for why they aren't committed),
configured, and **built clean (2216/2216 targets, 0 failures)**.

**REAL NS-3 VALIDATION** (not mock — prompt.md §0.4/§0.25 distinction): the stock `nr` example
`cttc-nr-demo` was executed directly
(`build/contrib/nr/examples/ns3.48-cttc-nr-demo --simTime=1`) and produced genuine measured 5G NR
KPIs from actual 3GPP channel-modeled gNB/UE flows — e.g. Flow 1 offered 10.24 Mbps → measured
throughput 10.237 Mbps, mean delay 0.276ms, mean jitter 0.030ms, 5998/6000 packets received. This
is real simulated network behaviour, not fabricated output — confirms the toolchain genuinely
builds and runs.

**Module 1 is now complete: `ns3_sim/nr_5g_telemetry_sim.cc` is a real, compiled, executed
5G-LENA scenario emitting genuine `SOURCE=NS3_5G_LENA` telemetry over ZeroMQ, consumed by the
real `src/telemetry/zmq_source.py`.** Topology: 1 gNB + 6 UEs (`RandomWalk2dMobilityModel`
within a 100m disc around the gNB, so `ue_position_x/y`/`ue_speed_mps` genuinely vary), UMa/
`ThreeGpp` channel model with shadow fading enabled, CBR UDP traffic per UE, `n=1`, 20MHz
bandwidth at 3.5GHz. Every telemetry field is real measured/derived simulator output, not
fabricated — provenance:

- `sinr_db` ← `NrUePhy::"DlDataSinr"` trace (linear SINR → dB via `10*log10`).
- `rsrp_dbm`/`rsrq_db` ← `NrUePhy::"ReportUeMeasurements"` trace, serving-cell entries only.
- `prb_utilization_pct` ← two gNB PHY traces combined: `"SlotDataStats"` (available RBs +
  data-symbol count per slot) and `"RBDataStats"` (literal per-symbol used-RB bitmap size) —
  `ratio = usedRbSymbols / (availRb * dataSymCount)`. Two separate traces are needed because
  `SlotDataStats`'s own `dataReg` field is an RBG×symbol product, not directly comparable to
  `availRb` — see bug #1 below.
- `throughput_mbps`/`offered_load_mbps`/`latency_ms`/`jitter_ms`/`packet_loss_pct` ← real
  `FlowMonitor` per-flow cumulative stats, polled every `telemetryInterval` (200ms) and
  converted to per-interval deltas (a snapshot-diff against the previous poll), not cumulative
  end-of-run stats.
- `ue_count` ← live UE count in the scenario; `ue_position_x/y`/`ue_speed_mps` ← the mobility
  model's real current position/velocity per tick.

Build wiring: `ns3_sim/ns-3-dev/scratch/nr_5g_telemetry_sim/` (a symlink to the tracked `.cc`
file + a generated `CMakeLists.txt` using `build_exec(... LIBRARIES_TO_LINK ... zmq ...)` to
link libzmq, which the default top-level `scratch/CMakeLists.txt` glob doesn't support) is
regenerated idempotently by `scripts/setup_ns3.sh` on every run — reproducible from a fresh
clone, nothing manual. Built via `ninja -j2 scratch_nr_5g_telemetry_sim` from the ns-3 build's
`cmake-cache` dir (the ninja target name differs from what `./ns3 build <name>` expects — a
build-system quirk, documented in `ns3_sim/README.md`).

**Real end-to-end validation** (`ns3_sim/validate_e2e.py`, a permanent repo-tracked script —
run it after any change to the scenario): launches the compiled binary as a subprocess, connects
the real `ZmqTelemetrySource` (no mocking on either side), consumes exactly 162 records (27
publish ticks × 6 UEs over a 6s run), confirms `source == "NS3_5G_LENA"` on every record, then
runs them through the real `TelemetryPreprocessor` — **result: 162/162 clean, 0 quarantined, 0
out-of-range flags, 6 feature windows built (one per UE)**. Re-confirmed working as of
2026-09-07. `schema.py`, `zmq_source.py`, and `preprocessing.py` needed **zero code changes** —
they were already correctly designed to consume exactly this shape of input; the only
Python-side change required was widening `config/settings.yaml`'s `sinr_db` valid_range (see
below).

**Two genuine bugs were found and fixed while validating this module** (matching the project's
established "always investigate surprising results" discipline — see the D1 dtype bug, mock PRB
realism bug, and jitter overfitting bug in Modules 6/9/10 below):

1. **PRB-utilization units mismatch (bug in this module's own new C++ code).** Comparing
   `SlotDataStats`'s `dataReg` (an RBG×symbol product) directly against `availRb` (a pure
   frequency-axis count) made the ratio structurally always ≥ 1, saturating PRB utilization at
   exactly 1.000000 regardless of real load. Fixed by adding a second trace (`"RBDataStats"`,
   giving a literal per-symbol RB bitmap in the same frequency-only unit as `availRb`) and
   recomputing the ratio from genuinely comparable quantities — post-fix PRB utilization shows a
   physically sensible 0.37-0.57 range depending on load, not pinned at an extreme.
2. **`sinr_db` valid_range too narrow for real producer output (Module 2 config, not code).** A
   UE very close to the gNB in this interference-free single-cell scenario genuinely reaches
   ~63dB SINR (real `DlDataSinr` trace values) — the pre-existing `[-20.0, 40.0]` range in
   `config/settings.yaml` (an untested guess made before any real producer existed, calibrated
   only against the mock generator which never exceeded ~35dB) would have flagged these
   legitimate real records as `out_of_range` (never dropped, per Module 2's design — just
   flagged). Fixed by widening to `[-20.0, 65.0]` with an explanatory comment in
   `config/settings.yaml`. Confirmed via the real 162-record end-to-end run (zero out-of-range
   flags) and the full `pytest tests -q` suite (**254 passed, zero regressions**).

Tuning notes (documented in-code in `nr_5g_telemetry_sim.cc`, kept here for anyone re-tuning):
default UDP load (~2.4 Mbps/UE, 6 UEs) and `totalTxPower=23.0dBm` (a realistic macro-sector
per-branch power, down from an initial 40.0dBm that produced a link budget so strong nothing
created measurable loss/throughput variation) plus `ShadowingEnabled=true` (real 3GPP TR 38.901
shadow fading) together produce genuine per-tick variation across every field: throughput
2.41-2.51Mbps, latency 2.2-3.55ms, jitter 0-0.66ms, PRB utilization 0.37-0.54, SINR 3.8-63dB,
RSRP -125.6 to -60.5dBm, RSRQ -14.55 to -10.79dB, UE speed 1.3-11.9m/s.

**Module 2 — Telemetry Collection & Preprocessing: implemented and tested.**
`src/telemetry/{base.py,schema.py,mock_source.py,zmq_source.py,preprocessing.py}`. Key design
decisions (needed context for anyone touching this module):

- **Two representations, on purpose.** Raw wire fields use "natural exporter" units — bits/sec,
  seconds, 0-1 ratios (`throughput_bps`, `latency_s`, `packet_loss_ratio`, ...) — matching what an
  ns-3 FlowMonitor-style exporter actually emits, so unit normalization
  (`schema.RAW_TO_CANONICAL`) is a real transformation, not a no-op. Canonical output fields are
  Mbps/ms/percent, matching `config.telemetry.preprocessing.valid_ranges` and everything
  downstream (DT models, fidelity). `sinr_db`/`rsrp_dbm`/`rsrq_db`/`ue_count`/`ue_speed_mps` are
  passthrough (already canonical at the source); `ue_position_x/y` are optional passthrough
  ("where available" per prompt.md §7 — never required, never fabricated when absent).
- **Source provenance is transport-authoritative, not payload-trusted.** `MockTelemetrySource`
  and `ZmqTelemetrySource` each unconditionally overwrite `record["source"]` with their own
  `SOURCE_LABEL` ("MOCK" / "NS3_5G_LENA") on every record they yield, regardless of what a
  payload itself claims. This is what makes source-masquerading (prompt.md §0.3) structurally
  impossible rather than merely policy — verified by
  `test_source_field_cannot_be_spoofed_through_real_channel`.
  `src/telemetry/preprocessing.py` additionally quarantines (never crashes on) any record whose
  `source` isn't in `schema.ALLOWED_SOURCES`.
- **Missing values: carry-forward imputation, else quarantine — never fabricate, never silently
  drop.** `TelemetryPreprocessor` keeps a per-`(ue_id, cell_id)` cache of the last known-good
  value per field. A missing *required* field is imputed from that cache if available (and
  recorded in `quality.imputed_fields`); if no history exists, the whole record is quarantined
  (kept in `PreprocessingResult.quarantined` with a reason — never deleted, per prompt.md §8 "do
  not silently discard important data"). A missing *optional* field (currently only
  `ue_position_x/y`) never blocks or quarantines — it's just recorded in `quality.missing_fields`.
- **NaN/Inf are treated as missing, not as values**, before they ever reach a range check or the
  pydantic constructor — and specifically before they can be written into the carry-forward
  cache, which would otherwise let a single bad reading silently poison every future imputed
  record for that key. Any input that still fails schema construction for an unanticipated reason
  is caught and quarantined rather than crashing `process_batch` (prompt.md §0.26 "NaN/Inf
  propagation"; §44 telemetry is untrusted input; §61 fail safely). See
  `test_nan_never_pollutes_the_carry_forward_cache`,
  `test_non_numeric_passthrough_field_is_treated_as_missing_not_a_crash`.
- **Out-of-range is a flag, not a drop.** Values outside `config.valid_ranges` are kept in the
  clean record with the field name added to `quality.out_of_range_fields` — downstream consumers
  (e.g. DT model training) decide what to do with flagged values; preprocessing itself never
  discards them.
- **Feature windows are tumbling, per-`(ue_id, cell_id)`, and never drop a trailing partial
  window** — `build_feature_windows` groups clean records by identity, sorts by timestamp, and
  slices into `config.feature_window_size`-row windows; a shorter final window is still returned
  (`FeatureWindow.size < feature_window_size`) rather than discarded. Each window carries
  `missing_ratio` (fraction of rows with any quality issue) and a `quality_flag`
  (`"ok"`/`"low_quality"` against `config.max_missing_ratio`) — this is the "time-series windows
  (timestamp, UE ID, cell ID)" artifact named on the Module 2 -> D1 edge of the data-flow diagram.
- **`ZmqTelemetrySource`** uses a real per-instance `zmq.Context`/SUB socket (`RCVTIMEO`,
  `RCVHWM` from config); a receive timeout is normal steady-state behaviour (logged at debug,
  loop continues waiting), not an error. `close()` sets a flag the loop checks after its next
  timeout, so a blocked consumer thread unblocks within one `recv_timeout_ms` window rather than
  hanging — verified by `test_close_stops_records_generator_without_hanging` using a real
  background thread, not a mock.
- **Tests exercise real transport, not mocked-out zmq**: `tests/unit/test_zmq_source.py` spins up
  an actual `zmq.PUB` socket and drives real bytes through `ZmqTelemetrySource`.
  `tests/integration/test_telemetry_pipeline.py` proves both the mock path and the real-ZeroMQ
  path end-to-end through `TelemetryPreprocessor` + `build_feature_windows` using the real
  `config/settings.yaml` values (not hardcoded test config).
- `config.telemetry.mock` gained `missing_field_rate` / `out_of_range_rate` so demo mode
  genuinely exercises the same imputation/flagging code paths a live run would (prompt.md §48 —
  demo mode must not be a separate fake architecture).

**Module 3 — Continuous Synchronization: implemented and tested.**
`src/synchronization/{d1_interface.py,sync.py}`. Key design decisions:

- **No manual sync step, by construction.** `ContinuousSynchronizer.start()` launches `run()` in
  a background daemon thread; `run()` is a `for raw in source.records(): ...` loop with no
  external "sync now" entry point anywhere in the class. As long as the source keeps producing
  telemetry, D1 keeps getting updated — proven directly (not just asserted) by
  `tests/integration/test_continuous_sync.py`: the test starts the synchronizer once, sleeps, and
  observes `D1StateSink` state grow purely from background activity, with zero manual batch/sync
  calls from the test itself, over both the mock source and a real ZeroMQ transport.
- **D1 doesn't exist yet (Module 4) — `d1_interface.py` defines the write contract Module 3
  depends on**: `D1StateSink` (ABC: `update_current_state`, `append_history`,
  `record_quarantine`) plus `InMemoryD1Stub`, a minimal thread-safe in-memory implementation used
  to build/test Module 3 now. When Module 4 is implemented, its state/history store must
  implement `D1StateSink` — nothing in `sync.py` should need to change.
- **Batching**: raw records are buffered and flushed into D1 when `config.synchronization.
  batch_size` is reached OR `batch_timeout_seconds` elapses, whichever first (both configurable,
  not hardcoded) — this is purely a batching/efficiency detail of *how* Module 3 pushes into D1,
  not a manual trigger; nothing external controls when a flush happens.
- **Current state = latest by telemetry timestamp, not arrival order.** `InMemoryD1Stub.
  update_current_state` refuses to let an out-of-order-arriving older record regress current
  state for a `(ue_id, cell_id)` key — network delivery can reorder records, and prompt.md §0.7
  requires live telemetry to be authoritative for *current* state, which means latest-by-event-
  time. `append_history` is unconditional append-only regardless of ordering (history keeps
  everything; only the "current" view is order-sensitive).
- **Quarantined records are pushed to D1 too** (`record_quarantine`), not just logged — a
  persistent data-quality audit trail alongside the history it was excluded from, not merely a
  transient log line (prompt.md §8 "do not silently discard important data").
- **Feature-window construction is deliberately NOT called by the synchronizer** on each small
  batch — `TelemetryPreprocessor.build_feature_windows` (Module 2) only produces meaningful
  windows once enough same-key history has accumulated, which is what `append_history` is for.
  Building windows over accumulated history is a query-time job for D1/its consumers (Module 4+),
  not something Module 3 should invoke repeatedly on tiny incoming batches — doing so would have
  produced a stream of degenerate near-empty "windows" instead of real ones.
- **Scope boundary** (prompt.md §0.7): Module 3 handles LIVE telemetry synchronization only.
  Bootstrap data (DT model init) and evaluation/candidate-training data (selected later by
  recalibration/verification) are separate provenance categories this module never touches.
- **A real concurrency bug was found and fixed while building this module's tests**: driving
  `ZmqTelemetrySource` continuously from a background thread and calling `close()` from the
  controlling thread caused an intermittent (~1-in-5) native SIGABRT inside libzmq — closing a
  socket / terminating a context from a different thread than the one blocked in
  `recv_multipart()` on it is undefined behaviour in libzmq. Fixed in `zmq_source.py`:
  `close()` now only sets a flag; the actual `socket.close()`/`context.term()` happens inside
  `records()`'s own `finally` block, i.e. always in the thread that owns the socket. Verified
  clean across 14+ repeated full-suite and targeted stress runs after the fix (was reproducible
  within a handful of runs before it). See `IMPLEMENTATION_STATUS.md` Module 3 entry for the
  full writeup — read this before touching `zmq_source.py`'s close/threading behaviour again.
- `MockTelemetrySource` gained the same `close()`/stop semantics as `ZmqTelemetrySource` (a
  `_closed` flag checked in its loop) so the synchronizer can drive either source identically and
  both are equally stoppable in tests and at shutdown.

**D1 (Module 4) — DT Basic Model: implemented and tested.**
`src/dt_models/{component_registry.py,d1_model_store.py}`. `D1Store` implements Module 3's
`D1StateSink` for real — `InMemoryD1Stub` is no longer the only option; `ContinuousSynchronizer`
can be pointed at either with zero changes to `sync.py`. Key design decisions:

- **Scope is exactly telemetry-derived DT state (prompt.md §0.5's three-state separation).** D1
  holds only concept B — dynamic DT state (current + historical), continuously updated by live
  telemetry (concept A). It does NOT hold DT prediction model versions/artifacts (concept C —
  that's Module 5's future `src/registry/model_registry.py`, unrelated to this file). D1 also
  does **not** fabricate a "network configuration" table: no module in this system currently
  produces network-configuration data, so inventing one would be exactly the kind of
  fake/placeholder component prompt.md prohibits. UE state and cell state
  (`get_ue_state`/`get_cell_state`) are filtered *views* over the single current-state table, not
  separate physical stores — telemetry already carries `ue_id`/`cell_id`/mobility fields, so a
  parallel table for the same data would be unnecessary infrastructure (prompt.md §10).
- **Genuinely persistent, Parquet-backed, not just in-memory.** Every write
  (`update_current_state`/`append_history`/`record_quarantine`) does an atomic
  write-to-temp-then-rename to its configured `.parquet` path (`config.storage.
  d1_current_state_path`/`d1_history_path`/`d1_quarantine_path`), and the constructor reloads
  existing files if present — a fresh `D1Store` pointed at the same paths sees the same data, i.e.
  it survives a process restart. Known limitation, documented rather than silently ignored: the
  three files are independently atomic, not transactionally coupled, so a crash between two
  writes of the same batch could in principle leave current-state and history briefly out of
  sync relative to each other — acceptable for a Pandas/Parquet-based store per prompt.md §10
  ("do not introduce unnecessary infrastructure"; a full transactional DB would be exactly that).
- **Current state = latest by telemetry timestamp**, implemented via a single pandas idiom:
  concat old+new, sort by timestamp descending, `drop_duplicates(subset=[ue_id, cell_id],
  keep="first")` — this correctly picks the max-timestamp row per key regardless of arrival
  order, including when two records for the same key arrive in the *same* batch (Module 3's
  `InMemoryD1Stub` only handled cross-batch ordering; D1Store's test suite additionally covers
  the within-batch case).
- **History is append-only** and pruned only by `config.storage.history_retention_rows` (oldest
  rows dropped once exceeded, by timestamp, not by insertion order) — it is never replaced or
  truncated by a current-state update (prompt.md §0.6: the DT is not a periodically-replaced
  dataset).
- **Quarantined records get real persistence too**, not just a stub in memory — `raw` (an
  arbitrary dict from Module 2, shape not guaranteed uniform across different failure reasons) is
  stored as a JSON string column (`raw_json`) to keep the Parquet schema stable regardless of
  what a given quarantined payload contained.
- **The "modular component registry"** (`ComponentRegistry`/`ComponentDescriptor`) is D1's own
  self-describing catalog of the state tables it tracks — name, kind (`current_state` / `history`
  / `audit`), schema, storage path. `D1Store.__init__` registers its three tables
  (`telemetry_current`, `telemetry_history`, `telemetry_quarantine`) into it. This is what keeps
  D1 modular per the diagram's own wording: a later module adding a new state category (DT
  predictions, adaptation events, evaluation windows) registers a new component instead of D1
  being redesigned — registering only records a catalog entry, it never touches a file.
- **Wired to Module 3 for real**, not just type-compatible in theory:
  `tests/integration/test_d1_synchronization.py` runs `ContinuousSynchronizer` against a real
  `D1Store` (not the stub) and asserts BOTH `get_current_state()` and `get_history()` grew from
  the same run — with an explicit check that they're structurally different (30 raw records ->
  30 history rows but only 5 deduped current-state rows) so the test can't pass by only one of
  the two actually working. A second test proves this holds under the same continuous
  background-thread, no-manual-trigger operation Module 3's own tests established, now against
  the real store.

**Module 5 — DT Functional Model: implemented and tested.** `src/dt_models/{base.py,
model_registry.py,orchestrator.py}`. **No real DT model (Modules 6-10) is implemented yet** —
this module is deliberately model-agnostic and is validated entirely by trivial dummy components
(`tests/dummy_dt_components.py`). Key design decisions:

- **`DTComponent` (base.py)**: the common interface every future throughput/latency/packet_loss/
  prb_utilization/jitter model must implement — `train`/`predict`/`evaluate`/`save`/`load` plus
  class-level metadata (`COMPONENT_NAME`, `DEPENDENCIES`, `REQUIRED_FEATURES`, `OUTPUT_FIELD`)
  the orchestrator schedules and wires purely from declared data, with zero component-specific
  code anywhere in `orchestrator.py` (prompt.md §11). `is_trained` is an **abstract property**,
  not a default — every component must track its own fit state explicitly, because the
  orchestrator refuses to predict through an untrained component (see below) rather than
  producing a silently-wrong prediction.
- **`DTModelRegistry` (model_registry.py) is a THIRD, distinct "registry" in this codebase — do
  not confuse the three**:
  1. `src/dt_models/component_registry.py`'s `ComponentRegistry` (D1/Module 4) — catalogs D1's
     own state *tables* (current/history/quarantine). Nothing to do with prediction models.
  2. `DTModelRegistry` (this module) — catalogs *prediction components* and their dependency
     metadata so the orchestrator can compute an execution order. No versioning, no artifacts.
  3. The future `src/registry/model_registry.py` (prompt.md §46, not built) — will track MODEL
     *VERSIONS* for adaptation/promotion/rollback (current/previous version, artifact path,
     training/eval metadata, status). A completely different concern despite the similar name.
  Registering a component in `DTModelRegistry` never trains it, saves it, or touches a file —
  purely visibility for the orchestrator. Components register with `enabled=True` by default;
  `enable_prb_model: false`-style toggles (prompt.md §12.4) map to `enabled=False` — a disabled
  component is excluded from scheduling entirely (not run with fake/empty input), and since
  nothing in the current spec depends on PRB utilization's output, disabling it cannot break any
  other component — verified generically (any component, not just PRB) by
  `test_disabled_component_excluded_from_execution_order` /
  `test_execution_order_detects_disabled_dependency`.
- **`DTOrchestrator` (orchestrator.py)**: `build_execution_order()` derives a topological order
  from `DTModelRegistry.enabled_components()` via `networkx.topological_sort` — never a
  hardcoded call sequence. An enabled component depending on an unregistered or disabled
  component, or a cyclic graph, raises `OrchestratorError` at plan-construction time (prompt.md
  §0.17 — the dependency graph is executable metadata that must fail explicitly, not silently
  produce incorrect predictions from a broken graph). `run_predictions(features)` then executes
  each component's `predict()` in that order: `REQUIRED_FEATURES` columns come straight from the
  caller-supplied `features` DataFrame (in practice `D1Store.get_current_state()` or
  `.get_history()`), and each declared dependency's column is populated from that dependency's
  own prediction — computed earlier in the same run, guaranteed available by construction because
  topological order puts dependencies first. A component reporting `is_trained=False` or a
  missing required feature column raises `OrchestratorError` immediately rather than predicting
  through incomplete state.
- **Deliberately scoped to prediction orchestration only, no training orchestration yet.** The
  user-facing ask for this module was specifically "runs models in dependency order reading from
  D1" (i.e. the runtime `predict` path from the canonical flow), and building a `run_training()`
  companion now would require deciding how each future component's ground-truth target column
  maps to D1 history — a design better made once the real components (Modules 6-10) and the
  boot-time training flow (prompt.md §14, future `src/main.py`) exist, not speculatively here.
  `DTComponent.train()` is still part of the interface (§11 requires it); it's just not something
  the orchestrator itself calls yet.
- **Tests prove the full "register, run, retrieve" chain with real multi-level dependencies, not
  a single trivial case**: `tests/dummy_dt_components.py` defines `DummyDoubler` (no deps) ->
  `DummyAdder` (depends on the doubler) -> `DummySummer` (depends on BOTH) — mirroring the shape
  of throughput/packet_loss -> latency -> jitter without implementing any real model.
  `test_run_predictions_produces_correct_chained_results` asserts the actual chained arithmetic
  is correct (not just that *some* dict came back), and
  `tests/integration/test_orchestrator_with_d1.py` re-runs the same proof with `features` read
  directly from a real `D1Store.get_current_state()` (fed via `update_current_state`, bypassing
  the full telemetry pipeline which is already covered elsewhere) — confirming the "reading from
  D1" wiring genuinely works, not just that a hand-built DataFrame works.

**Module 6 — Throughput Model: implemented and tested — the first real `DTComponent`.**
`src/dt_models/{regressors.py,throughput.py}`. Key design decisions:

- **`regressors.py`** is a tiny shared factory (`build_regressor(model_type, params)` ->
  `XGBRegressor` or `RandomForestRegressor`) so Modules 7-10 select their algorithm purely from
  `config.dt_models.<component>.model_type`/`params` (already defined in `config/settings.yaml`
  since Phase 1) without duplicating construction logic five times.
- **`ThroughputModel`**: `COMPONENT_NAME="throughput"`, `DEPENDENCIES=()` (a root node — no
  upstream DT predictions needed, matching prompt.md §13's dependency diagram),
  `REQUIRED_FEATURES=("offered_load_mbps","prb_utilization_pct","sinr_db","rsrp_dbm","rsrq_db",
  "ue_count","ue_speed_mps")` — these names match `d1_model_store.TELEMETRY_COLUMNS` exactly, so
  a `D1Store.get_current_state()`/`.get_history()` DataFrame can be passed straight in with zero
  reshaping. `OUTPUT_FIELD="throughput_mbps_pred"` — deliberately distinct from D1's ground-truth
  `"throughput_mbps"` column so a future downstream component (latency, jitter) can never confuse
  a prediction for the raw telemetry value it was trained against.
- **Defensive training**: NaN rows are dropped (logged, not silently kept) before fitting;
  all-NaN input raises a clear `ValueError` rather than handing a regressor empty/garbage data.
  Missing required feature columns raise on both `train()` and `predict()`.
  `evaluate()` returns RMSE/MAE — a component-local sanity metric, explicitly documented as NOT
  the authoritative fidelity score (Module 12 owns that deterministic formula).
- **Persistence via `joblib`** (already a transitive scikit-learn dependency) — works uniformly
  for both XGBoost and RandomForest estimators, avoiding a branchy per-algorithm save/load path.
- **REAL held-out validation, not placeholder numbers** — `tests/integration/
  test_throughput_training.py` runs the actual Module 2 -> Module 3 -> D1 pipeline (real
  `MockTelemetrySource` -> `TelemetryPreprocessor` -> `ContinuousSynchronizer` -> `D1Store`,
  1200 SOURCE=MOCK records with realistic SINR/PRB/offered-load -> throughput correlations),
  takes a time-ordered 80/20 train/held-out split of `D1Store.get_history()`, trains
  `ThroughputModel.from_settings(settings)` on the train split, and evaluates on the untouched
  held-out split. Actual observed result on this machine: **held-out RMSE = 1.5392 Mbps, MAE =
  1.1905 Mbps**, versus a naive mean-prediction baseline of **RMSE = 10.8198 Mbps** (held-out
  target std 10.8368 Mbps) — the model explains the overwhelming majority of variance, not
  noise-fitting. The test asserts the trained model beats the naive baseline (proving genuine
  learning, not a lucky pass) rather than an arbitrary hardcoded threshold. A second test
  registers the trained model into a real `DTModelRegistry` and runs it through
  `DTOrchestrator.run_predictions()`, confirming Module 6 -> Module 5 wiring end-to-end.
- **A real D1Store bug was found and fixed while building this test** (only surfaced because
  this was the first time D1 output was fed to something dtype-sensitive): `pd.concat`ing
  `_load_or_empty`'s columns-only placeholder (returned when no Parquet file exists yet, on
  every store's very first write) against a properly-typed incoming batch silently downcast
  *every* column — including `throughput_mbps`, `sinr_db`, etc. — to `object` dtype, and that
  corruption then persisted forever after (every later concat inherits the already-corrupted
  accumulated frame). Scalar-equality tests never caught it (`50.0 == 50.0` holds regardless of
  column dtype); XGBoost's `fit()` does, refusing to train on `object` columns. Fixed with
  `_concat_preserving_dtypes()`: skip the concat entirely when the existing frame has zero rows
  (use `new_rows`' own correctly-inferred dtypes directly) — concatenating two already-well-typed
  frames does not corrupt dtypes, so this one change is sufficient for the whole store's
  lifetime. Regression tests added directly to `tests/unit/test_d1_model_store.py` (dtype
  checked after first write, after multiple batches, and after a disk reload) so this can never
  silently regress again. This is exactly why prompt.md §0.24 requires *actually running* things
  end-to-end rather than trusting that unit tests passing implies correctness.

**Module 7 — Latency Model: implemented and tested — the second real `DTComponent`, first real
dependency wiring.** `src/dt_models/latency.py`. Key design decisions:

- **`LatencyModel`**: `COMPONENT_NAME="latency"`, `DEPENDENCIES=("throughput",)` — the first
  component with a genuine upstream dependency, executed by `DTOrchestrator` strictly after
  `"throughput"`. `REQUIRED_FEATURES=("offered_load_mbps","prb_utilization_pct","ue_count",
  "packet_loss_pct","sinr_db")`, `OUTPUT_FIELD="latency_ms_pred"`.
- **Scope decision, explicitly made this turn**: prompt.md §13 says latency depends on BOTH
  throughput and packet loss, but Module 8 (Packet-Loss) doesn't exist yet. `DEPENDENCIES`
  therefore wires only `"throughput"`; `packet_loss_pct` is consumed as a RAW ground-truth
  telemetry feature straight from D1 (in `REQUIRED_FEATURES`) instead. This keeps the dependency
  graph always valid and executable (prompt.md §0.17 — never declare a dependency on something
  unregistered) rather than blocking Module 7 on Module 8's existence. **Follow-up flagged for
  whenever Module 8 is built**: add `"packet_loss"` to `DEPENDENCIES` and drop `packet_loss_pct`
  from `REQUIRED_FEATURES` in favour of the packet-loss model's own `OUTPUT_FIELD`.
- **The throughput-dependency input column is separate from `REQUIRED_FEATURES` by design.**
  `LatencyModel._input_columns` = `REQUIRED_FEATURES + (THROUGHPUT_INPUT_COLUMN,)` where
  `THROUGHPUT_INPUT_COLUMN = "throughput_mbps_pred"` — matching `ThroughputModel.OUTPUT_FIELD`
  exactly. The orchestrator only validates `REQUIRED_FEATURES` against the raw `features` frame
  it's given (the dependency column is supplied separately, by wiring — see `DTOrchestrator.
  run_predictions`), so `REQUIRED_FEATURES` deliberately excludes it; `LatencyModel` itself
  validates the full `_input_columns` set (raising a clear error naming
  `"throughput_mbps_pred"` specifically if it's missing) since that's what `train()`/`predict()`
  actually consume.
- **Train on ground truth, serve on live predictions** — the standard pattern for a
  dependency-chained/stacked model. Training (bootstrap, outside the orchestrator) supplies
  `THROUGHPUT_INPUT_COLUMN` populated from D1's ground-truth `throughput_mbps` column
  (`history.assign(throughput_mbps_pred=history["throughput_mbps"])` — the strongest available
  training signal). At inference time, the orchestrator instead supplies the upstream
  `ThroughputModel`'s own live prediction under that same column name. `LatencyModel` itself is
  agnostic to which — it only ever consumes the column by name — so no special-casing exists
  anywhere in the component for "trained" vs. "served" throughput.
- **REAL held-out validation** (`tests/integration/test_latency_training.py`, reusing the shared
  `bootstrap_history` fixture — see `tests/integration/conftest.py`, extracted this turn from
  Module 6's test so both share one real Module 2->3->4 pipeline run instead of duplicating it).
  Actual observed result on this machine: **held-out RMSE = 2.2217 ms, MAE = 1.7600 ms**, versus
  a naive mean-prediction baseline of **RMSE = 4.4239 ms** (held-out target std 4.3707 ms) — the
  model roughly halves baseline error. Same beats-the-baseline assertion pattern as Module 6, not
  an arbitrary threshold.
- **The actual dependency chain is proven end-to-end, not just declared.**
  `test_latency_runs_through_orchestrator_fed_by_real_throughput_predictions` registers a real
  trained `ThroughputModel` AND `LatencyModel` into one `DTModelRegistry`, confirms
  `build_execution_order()` places throughput before latency, then asserts latency's predictions
  from `orchestrator.run_predictions()` exactly match what `LatencyModel.predict()` produces when
  manually fed throughput's *own predictions* — and explicitly asserts this differs from what
  ground-truth-fed predictions would be, proving the orchestrator genuinely wired in throughput's
  live output rather than the test accidentally validating against ground truth by coincidence.

**Module 8 — Packet-Loss Model: implemented and tested — the third real `DTComponent`, second
root node.** `src/dt_models/packet_loss.py`. Same shape as Module 6 (`ThroughputModel`) — a root
node, `DEPENDENCIES=()`:

- **`PacketLossModel`**: `COMPONENT_NAME="packet_loss"`,
  `REQUIRED_FEATURES=("sinr_db","rsrp_dbm","rsrq_db","prb_utilization_pct","ue_count",
  "offered_load_mbps")` — matches D1 column names exactly, `OUTPUT_FIELD="packet_loss_pct_pred"`
  (distinct from ground-truth `"packet_loss_pct"`, same reasoning as Modules 6-7's `_pred`
  suffix convention). Algorithm via `config.dt_models.packet_loss.model_type`/`params` +
  `regressors.build_regressor` — identical pattern to Modules 6-7 (NaN-row dropping, joblib
  persistence, RMSE/MAE `evaluate()`).
- **REAL held-out validation** (`tests/integration/test_packet_loss_training.py`, reusing the
  shared `bootstrap_history` fixture). Actual observed result on this machine: **held-out
  RMSE = 0.3352 %, MAE = 0.2638 %**, versus a naive mean-prediction baseline of **RMSE =
  1.0224 %** (held-out target std 1.0117 %) — roughly a 3x error reduction. Same
  beats-the-baseline assertion pattern as Modules 6-7.
- **Orchestrator test proves multiple independent root nodes, not just a chain or a single
  node**: `test_packet_loss_and_throughput_both_registered_and_run_through_orchestrator`
  registers both `ThroughputModel` and `PacketLossModel` (unrelated to each other) into one
  `DTModelRegistry` and confirms `DTOrchestrator` runs and returns correct predictions for both —
  covering a registry shape Modules 6/7's tests hadn't exercised (two roots with no edge between
  them, as opposed to one root or a linear two-node chain).
- **Deliberately NOT touched this turn**: `LatencyModel.DEPENDENCIES` (still `("throughput",)`
  only, with `packet_loss_pct` still a raw D1 feature) — Module 7 flagged wiring a real
  `"packet_loss"` dependency as a follow-up once this component existed, but that is a Module 7
  change and this turn's instruction scoped docs/work to "Module 8 only". Still an open,
  documented follow-up — see Module 7's entry and `IMPLEMENTATION_STATUS.md`.

**Module 9 — PRB-Utilization Model: implemented and tested — the fourth real `DTComponent`,
first REAL cross-module dependency (Module 6 already existed when this was built, unlike
Modules 7/8's judgment calls) — and a genuine mock-telemetry realism bug found and fixed along
the way.** `src/dt_models/prb_utilization.py`. Key design decisions:

- **`PrbUtilizationModel`**: `COMPONENT_NAME="prb_utilization"`,
  `DEPENDENCIES=("throughput",)` — explicitly instructed this turn, no judgment call needed.
  `OUTPUT_FIELD="prb_utilization_pct_pred"`. Defaults to `model_type="random_forest"` (matching
  `config.dt_models.prb_utilization`'s existing default — the first of Modules 6-9 to genuinely
  exercise the RandomForest path in a real held-out validation; Modules 6-8 all defaulted to
  xgboost).
- **"Cell load" is not a raw D1 column and is deliberately NOT approximated by
  `prb_utilization_pct` itself** — prompt.md's own input list for this component
  (offered load, UE count, throughput, cell load) excludes PRB entirely, unlike every other
  component's input list, which does include it; using PRB-derived data as this model's own
  input would be leakage/tautology (handing it a proxy for its own target). Instead, "cell load"
  is a genuine derived aggregate: `_add_cell_load` sums `offered_load_mbps` across every row
  sharing the same `(cell_id, timestamp)` — the cell's aggregate traffic demand at that instant,
  distinct from any single UE's own offered load (already a separate input).
  `REQUIRED_FEATURES` therefore includes `cell_id`/`timestamp` as grouping keys (validated by the
  orchestrator against raw D1 output, which always has them) even though they are dropped before
  reaching the regressor — only the derived `cell_load_mbps` plus `offered_load_mbps`/`ue_count`/
  the throughput dependency column are the actual model inputs. Documented limitation: a
  `features` slice missing part of a cell's cohort at a given timestamp degrades to a partial
  (not incorrect) aggregate — acceptable given `MockTelemetrySource` (and any real per-tick
  exporter) emits a full cohort per shared timestamp in normal use.
- **A genuine mock-telemetry realism gap was found and fixed while validating this module.**
  Initial held-out result: RMSE 19.7699% vs. naive-baseline RMSE 19.9659% — the model learned
  essentially nothing. Root cause: `MockTelemetrySource._generate_record` drew
  `prb_utilization_ratio` as pure independent noise (`Beta(2,3)`, uncorrelated with any demand
  signal) even though that value was *already* used to influence throughput/packet-loss/latency
  downstream — i.e. PRB utilization affected other fields but had no cause of its own, making it
  structurally unlearnable from offered load/UE count/throughput (this component's actual
  inputs). This was a genuine oversimplification in Module 2's mock generator (every other field
  — SINR→channel_quality→throughput/loss/latency — already had a real causal chain; PRB
  utilization was the one exception), not a Module 9 defect, and not something to paper over:
  in a real cellular network PRB utilization is directly driven by cell-wide demand, so fixing
  this makes the mock MORE realistic, not "tuned to pass a test." Fixed in
  `src/telemetry/mock_source.py`: `prb_utilization_ratio` is now derived from `offered_load_bps`
  and the cell's UE count (`demand_pressure = offered_load_bps/40e6 + 0.02*cell_ue_count`,
  `prb_utilization_ratio = clip(0.15 + 0.5*demand_pressure + noise, 0, 1)`), computed *before*
  `capacity_bps`/`throughput_bps` (simple reorder — the existing downstream formulas that already
  consumed `prb_utilization_ratio` as an input needed no changes). Verified via 4 consecutive
  full-suite runs (188/188 passing each time, including every earlier module's mock-dependent
  tests) that this didn't disturb anything else, and via a fresh mock-distribution sanity check
  (mean≈0.59, std≈0.19, min/max within [0,1] — a healthy, non-degenerate spread). Post-fix result:
  **held-out RMSE = 8.6235%, MAE = 6.8494%, vs. naive-mean-baseline RMSE = 19.4206%** (held-out
  target std 19.4459%) — genuine learning, more than 2x error reduction.
- **The actual dependency chain AND the enable/disable toggle are both proven end-to-end.**
  `test_prb_utilization_runs_through_orchestrator_fed_by_real_throughput_predictions` mirrors
  Module 7's dependency-chain proof pattern (real trained `ThroughputModel` + `PrbUtilizationModel`
  registered together, execution order checked, predictions confirmed to come from throughput's
  *live* predictions, not ground truth). `test_disabling_prb_utilization_excludes_it_without_
  breaking_throughput` registers with `enabled=False` (simulating `config.dt_models.
  enable_prb_model: false`) and confirms it's excluded from both the plan and the result while
  throughput keeps working — directly validating the diagram's "independent/optional" note and
  Module 5's `enabled` mechanism together, for real, not just in the abstract.

**Module 10 — Jitter Model: implemented and tested — the fifth and final `DTComponent`, the
last node in the DT dependency graph. All five DT prediction models (Modules 6-10) are now
complete.** `src/dt_models/jitter.py`. Key design decisions:

- **`JitterModel`**: `COMPONENT_NAME="jitter"`, `DEPENDENCIES=("throughput","latency",
  "packet_loss")` — all THREE real, matching prompt.md §13's diagram exactly ("throughput +
  latency + packet loss -> jitter"), no fallback judgment call (unlike Modules 7/8, every
  upstream model this one needs already existed). `REQUIRED_FEATURES=("offered_load_mbps",
  "ue_count")` — the only two genuinely raw D1 inputs from the 5-item input list; latency/
  throughput/packet-loss are consumed as upstream-model prediction columns via
  `THROUGHPUT_INPUT_COLUMN`/`LATENCY_INPUT_COLUMN`/`PACKET_LOSS_INPUT_COLUMN` (matching
  `ThroughputModel`/`LatencyModel`/`PacketLossModel`'s `OUTPUT_FIELD`s exactly), same
  train-on-ground-truth/serve-on-live-predictions pattern as Modules 7/9. `OUTPUT_FIELD=
  "jitter_ms_pred"`.
- **A genuine overfitting bug (not a mock-realism gap this time) was found and fixed.** Initial
  held-out result: RMSE 1.2982ms vs. naive-baseline RMSE 1.3164ms — essentially no better than
  guessing the mean, despite `MockTelemetrySource`'s `jitter_s = latency_s * uniform(0.05,0.25)`
  giving jitter a real, direct causal link to latency (unlike Module 9's PRB bug, this was NOT a
  data-generation problem — confirmed by checking real correlations in the bootstrap data:
  jitter_ms correlates 0.48 with latency_ms, 0.31-0.39 with throughput/offered_load/packet_loss,
  genuine signal). Diagnosis: a plain `sklearn.LinearRegression` on the exact same ground-truth
  columns achieved held-out RMSE 1.187 — clearly better than my trained XGBoost model, which
  should never happen if the model were fit well. Root cause, confirmed by comparing train vs.
  held-out RMSE directly: the shared default hyperparameters
  (`n_estimators=300, max_depth=6, learning_rate=0.05` — tuned implicitly by being "the default"
  for Modules 6-9, which all have strong, high-magnitude relationships and didn't expose this)
  massively overfit jitter's comparatively subtle, noisy relationship on a bootstrap-sized
  dataset (~960 training rows): train RMSE 0.47 vs. held-out RMSE 1.29, a textbook overfitting
  signature. Fixed in `config/settings.yaml`'s `dt_models.jitter.params` only (Modules 6-9's
  params untouched, since they weren't the problem): shallower/fewer trees with subsampling
  (`n_estimators=80, max_depth=2, learning_rate=0.1, subsample=0.9`), the best of several
  regularized configurations tested directly against the real held-out bootstrap split. Post-fix
  result: **held-out RMSE = 1.1743 ms, MAE = 0.9685 ms, vs. naive-mean-baseline RMSE = 1.3164 ms**
  (held-out target std 1.3143 ms) — genuine learning, now also beating the plain-linear-
  regression sanity check (1.187ms). This is the third real bug found purely by insisting on
  actually running things end-to-end with real reported metrics — see Module 6's D1 dtype bug
  and Module 9's mock-telemetry realism bug for the other two; unlike those, this one was in
  neither the DT component's own code nor the mock generator, but in a shared config default
  that happened to fit four other components well and this one badly — a reminder that "shared
  defaults" still need per-component validation, not just a first successful case.
- **The full five-component dependency chain is proven end-to-end, exactly as the diagram
  specifies** — `tests/integration/test_full_dependency_chain.py`: all five real trained
  components (`ThroughputModel`, `PacketLossModel`, `LatencyModel`, `PrbUtilizationModel`,
  `JitterModel`) registered into one `DTModelRegistry`, run through one `DTOrchestrator` call.
  Confirmed: `build_execution_order()` returns
  `['throughput', 'packet_loss', 'latency', 'prb_utilization', 'jitter']` — both roots first,
  then the two components depending on throughput, then jitter strictly last (the diagram's
  "added last" note, literally true); `run_predictions()` returns correct, correctly-sized
  predictions for all five in one call; jitter's actual input is reconstructed from the OTHER
  FOUR's own returned predictions and shown to match exactly (live wiring at the deepest node,
  not ground truth); latency's input is separately confirmed to have been fed throughput's live
  prediction too (the chain is live at every hop, not just the last one). A second test exercises
  the `enable_prb_model` toggle inside this same full five-component registry (not in isolation)
  and confirms disabling it removes only `"prb_utilization"` from the plan/result, leaving the
  other four — including jitter, which does NOT depend on PRB utilization — completely unaffected.

**Module 11 — Drift Detection Interface: implemented and tested.** `src/drift/{base.py,schema.py,
mock_drift_source.py,drift_detector.py}`. The actual drift-detection algorithm is external/
partner-owned and is deliberately NOT implemented (prompt.md §0.19 rule 7 / §15) — only the
receiving interface, schema validation/normalization, and a mock source for testing exist here.
Key design decisions:

- **Deliberately mirrors Module 2's two-stage design rather than inventing a new one**: a
  `DriftSource` (`base.py`, symmetric to `TelemetrySource`) emits raw wire-format dicts matching
  prompt.md §15's exact contract (`{"component", "severity", "timestamp", "metadata"}`);
  `drift_detector.py`'s `DriftDetectorInterface` validates and normalizes them into a canonical
  `DriftEvent` (frozen pydantic model in `schema.py`), exactly the `mock_source.py` ->
  `preprocessing.py` split already established. A real external/partner detector's own adapter
  will implement `DriftSource` later — `drift_detector.py` needs zero changes when that happens
  (prompt.md §0.19: "production architecture must allow the external detector to be connected
  later without redesigning the adaptation system").
- **The component-name validation IS "identify which DT scope needs adaptation"** (the
  fig-dataflow.png responsibility named for this module): `DriftDetectorInterface` checks the
  event's `component` field against `config.drift.valid_components` — a component this
  deployment doesn't actually run (typo, stale detector config, unrelated KPI) cannot be routed
  to any real adaptation scope, so it's quarantined rather than silently passed downstream. This
  list is shared config between the mock source and the real interface (not duplicated), and
  must match the `COMPONENT_NAME`s registered in Module 5's `DTModelRegistry` — currently
  throughput/latency/packet_loss/prb_utilization/jitter.
- **No carry-forward imputation, unlike Module 2 — a deliberate divergence, not an omission.** A
  missing/invalid telemetry field can be carried forward from the last known-good value for that
  (ue_id, cell_id) because telemetry is a continuous stream describing gradually-changing
  physical quantities. A drift notification is a discrete, one-off signal with no "last known
  good value" to substitute, and it feeds directly into a system that takes real, side-effecting
  adaptation actions (PPO -> an adaptation agent -> a candidate -> promotion/rejection) — per
  prompt.md §61 "fail safely," a malformed trigger must always be dropped, never guessed at.
  Every rejected event is still retained as a `QuarantinedDriftEvent` (raw + reason + source +
  received_at) for audit, mirroring `QuarantinedRecord`'s "never silently discard" principle
  (prompt.md §8) even though it's never repaired.
- **Config restructured, not just extended**: `valid_components` and `severity_range` moved out
  of `drift.mock.*` to top-level `drift.*` in `config/settings.yaml` — both the mock source and
  the real interface's own validation need the identical values, so nesting them under `mock`
  would have wrongly implied they're mock-only concerns. `drift.mock.*` now holds only
  genuinely mock-only knobs: `seed`, `emit_interval_seconds`, and the new
  `invalid_event_rate` (default 0.05) — the fraction of mock events deliberately corrupted
  (unknown component / out-of-range severity / missing timestamp / non-dict metadata) so the
  validation/quarantine path is genuinely exercised in testing, mirroring
  `telemetry.mock.missing_field_rate`/`out_of_range_rate`'s established rationale. This was the
  only config-shape change required; no other module's config was touched.
- **`process_stream(source)` is the interface's own consumption entry point** — a thin generator
  over any `DriftSource` that yields only validated `DriftEvent`s, logging and skipping (never
  raising on) a quarantined one, so one malformed notification can never break the stream a
  future continuous consumer (Module 13's PPO observation loop, not built yet) reads from.
  `process_batch()` mirrors `TelemetryPreprocessor.process_batch` for straightforward testing.
- **Tested end-to-end with the mock source, exactly as scoped** — 36 new tests (254 -> 290
  total, all passing, 0 regressions from the config restructure):
  `tests/unit/test_mock_drift_source.py` (9), `tests/unit/test_drift_detector.py` (25, every
  validation branch individually — including all 5 real configured components each proven
  individually routable, NaN/Inf severity, ISO-8601 timestamp parsing, mixed valid+invalid
  batches never raising), and `tests/integration/test_drift_pipeline.py` (3, run against real
  `config/settings.yaml` values — 500 mock events: the overwhelming majority accepted, some
  genuinely quarantined per the configured `invalid_event_rate`, every accepted event's
  `component` is confirmed to be a member of the live config's `valid_components` AND all 5 real
  components are confirmed individually exercised across the run; a second run with
  `invalid_event_rate` overridden to 0.0 yields exactly zero quarantine, proving quarantining is
  driven by the injected corruption and nothing else).

**Module 12 — Fidelity Evaluation Module: implemented and tested.** `src/fidelity/{metrics.py,
evaluator.py}`. This is the module prompt.md calls out as "extremely important" and where
determinism is completely non-negotiable — the LLM (Modules 15-17, not built yet) must never
compute or touch this formula. Key design decisions:

- **`metrics.py`** — four pure, deterministic raw metric functions, all sharing one input
  validator (`_validate_inputs`: rejects empty arrays, mismatched lengths, and non-finite
  values via `FidelityComputationError` — never silently computes a metric from bad data).
  `rmse`/`mae` are pointwise (paired, same-index); `wasserstein` (via `scipy.stats.
  wasserstein_distance`) and `mk_mmd` are distributional — they compare the two samples AS
  DISTRIBUTIONS, order/pairing-independent, capturing whether the DT's overall predicted shape
  matches ground truth even when RMSE/MAE alone couldn't. All four return a non-negative
  "distance" in non-squared units; `evaluator.py` squares all four uniformly, so no metric
  function squares its own output.
- **MK-MMD is a genuine multi-kernel implementation**, not a single-bandwidth stand-in: the
  unbiased MMD² estimator is computed at three fixed bandwidth multipliers
  (`_MK_MMD_BANDWIDTH_MULTIPLIERS = (0.5, 1.0, 2.0)`, a small fixed implementation constant, not
  user-configurable) around a base gamma, then averaged and clamped at 0 before `sqrt` — the
  unbiased estimator can legitimately go slightly negative from sampling noise even when the
  true MMD is near 0 (this is exactly what was observed running the engine against real DT
  predictions — see below). Base gamma defaults to the median heuristic
  (`config.fidelity.mk_mmd.gamma: null`), computed deterministically from the pooled sample; an
  explicit gamma is equally deterministic. **Cross-checked, not just self-tested**:
  `tests/unit/test_fidelity_metrics.py` reimplements MK-MMD as a plain triple-nested-loop
  (deliberately independent of `metrics.py`'s vectorized version) and asserts agreement to
  `rel=1e-9` across several cases — the two can only agree if both are actually correct.
- **`evaluator.py`'s `FidelityEvaluator`** owns one rolling window (deque, bounded by
  `config.fidelity.rolling_window_length`) per `(component, metric)`, holding past SQUARED
  metric values — the min-max normalization reference. `evaluate(component, y_true, y_pred,
  update_window=True)` normalizes the current point against the window's contents from BEFORE
  this call (genuinely "historical" per prompt.md §0.10 — a single evaluation can never
  trivially normalize to a self-referential 0), then appends afterward if `update_window=True`.
  **`update_window=False` is the hook for future candidate-vs-production comparisons** (Module
  17, not built yet): calling `evaluate(..., update_window=False)` on the SAME evaluator
  instance a production run already grew normalizes the candidate against the identical
  reference production used, satisfying §0.10's "BOTH MUST USE THE SAME NORMALIZATION
  REFERENCE" — verified directly by
  `test_production_and_candidate_evaluations_share_one_normalization_reference`.
- **Never silently manufactures a score** (§0.10): fewer than `config.fidelity.
  min_history_for_normalization` (default 10) prior points for a component ->
  `status="insufficient_history"`, `fidelity_score=None` — raw/squared metrics are still
  returned (always well-defined) but the composite score is explicitly withheld, not faked.
- **THE hand-computed case** (`test_composite_score_hand_computed_case`): three calls with
  `y_true=[0,0], y_pred=[a,a]` for a=1,2,3 (point-mass pairs make RMSE=MAE=Wasserstein=a
  exactly by hand; MK-MMD for each `a` was derived independently and cross-checked against the
  brute-force reimplementation — see `test_fidelity_metrics.py`). The full pipeline's expected
  output — every normalized metric and the final `FidelityScore = -1.287340459401399` — was
  computed independently (a separate, from-scratch application of the min-max + composite
  formula, not by calling `FidelityEvaluator` and copying its output) before being hardcoded as
  the test's expected value, and matches the evaluator's actual output to `rel=1e-9`.
- **Numerical stability edge cases explicitly tested, not just handled in passing**: a
  perfectly constant rolling window (identical repeated evaluations) produces a large-but-finite
  normalized value via the epsilon-stabilized denominator, never NaN/Inf
  (`test_constant_window_produces_large_but_finite_value_not_nan_or_inf`); different `epsilon`
  values produce different results, proving it's genuinely config-driven, not hardcoded
  (`test_epsilon_is_config_driven_not_hardcoded`); the rolling window is proven to actually
  bound history at `config.fidelity.rolling_window_length`, not just accumulate forever
  (`test_rolling_window_length_is_config_driven_and_bounds_history`).
- **Run against REAL predictions from Modules 6-10 vs. real D1 ground truth**, not synthetic
  arrays — `tests/integration/test_fidelity_evaluation.py`: trains all five real components
  (same pattern as `test_full_dependency_chain.py`), runs the real orchestrator on real held-out
  bootstrap data, then feeds 24 sequential 10-row batches per component through one shared
  `FidelityEvaluator` per component. Every one of the five components genuinely crosses the
  `min_history_for_normalization` threshold and receives real composite `FidelityScore` values
  (not stuck at `insufficient_history` forever) — e.g. last-batch results on this machine:
  throughput RMSE=1.0706 FidelityScore=0.8175, packet_loss RMSE=0.2228 FidelityScore=0.9338,
  latency RMSE=2.3617 FidelityScore=0.7006, prb_utilization RMSE=10.8567 FidelityScore=0.6759,
  jitter RMSE=0.8746 FidelityScore=0.9326. **MK-MMD came out as exactly 0.0 for every
  component's real-data batches** — investigated before accepting, not assumed benign: the
  underlying per-bandwidth unbiased MMD² estimates are genuinely negative (e.g. -0.06 to -0.11
  for throughput), confirming this is the expected, correct estimator-noise behavior documented
  above (these five models' predictions are close enough to ground truth that predicted/true
  *distributions* overlap almost completely at `n=10`), not a bug — the clip-at-0-before-sqrt
  logic is doing exactly its job. A second test confirms bit-identical results across two
  independent evaluator instances run over the same real predictions — determinism holds
  end-to-end against real model output, not just synthetic hand cases.

**Module 13 — PPO RL Decision Agent: implemented, trained, and validated on a held-out
scenario.** `src/adaptation/{rl_env.py,rl_agent.py}`. This is the module prompt.md §70 rules 4
and 18 single out — "PPO decides WHAT adaptation strategy to use" / "Do not replace PPO with
heuristics" — so the design below is organized around keeping the environment (what happens to
the world) and the decision (which action PPO picks) structurally incapable of being confused
with each other. Key design decisions:

- **`rl_env.py`'s `AdaptationEnv(gym.Env)`** — action space `Discrete(3)` (0=Recalibrate,
  1=Regenerate, 2=Expand Scope, from `config.ppo.action_mapping`, never a 4th/5th action per
  prompt.md §19). Observation (`Box`, all entries normalized/clipped per prompt.md §18):
  `[5 fidelity values, affected-component one-hot, drift severity, previous-action one-hot,
  previous reward, network-state summary]` — 23 dimensions total (5+5+1+3+1+8).
- **Genuine integration with three already-built modules, not synthetic numbers invented in this
  file**: (1) **Module 11** — `build_drift_event_pool()` drives the real `MockDriftSource` +
  `DriftDetectorInterface` to produce validated `DriftEvent`s; every episode's affected
  component/severity comes from this real pipeline. (2) **Module 12** — every fidelity value in
  the observation comes from a real `FidelityEvaluator.evaluate()` call; the deterministic
  RMSE/MAE/Wasserstein/MK-MMD formula genuinely runs every step, never a placeholder. (3)
  **Module 4 (D1)** — `build_network_state_pool()` builds a real (temp-path, non-production)
  `D1Store`, populates it via the real Module 2 pipeline (`MockTelemetrySource` ->
  `TelemetryPreprocessor`), and reads `get_history()` for the network-state observation slice.
- **Why a simulated adaptation-outcome "world model" exists at all, and why it is NOT the
  heuristic prompt.md §70 rule 18 forbids**: Modules 14-16 (the agents that would actually
  recalibrate/regenerate/expand-scope a component) don't exist yet, so training PPO requires
  *some* environment to act in. `config.ppo.env.*` defines a documented, config-driven "how much
  does action A shrink the current prediction-error magnitude" function — exactly the role
  CartPole's physics play for a classic Gym env. It NEVER decides which action to take; it only
  reacts to whatever action is supplied. The actual decision always goes through
  `rl_agent.py:decide_adaptation_strategy()`, which does nothing but call `PPO.predict()`.
- **The dynamics are deliberately severity- and attempt-history-dependent, which is what makes
  this a genuine (non-trivial) decision problem**: recalibrate's efficacy scales down as
  `(1 - severity)` and decays further on repeated attempts on the same incident (retraining the
  existing model can't fix a structural/severe shift no matter how many times it's retried);
  regenerate's efficacy only mildly depends on severity (an LLM-rebuilt model handles drift far
  better); expand_scope is only genuinely effective once
  `config.ppo.env.expand_scope_min_prior_attempts` prior attempts have already failed this
  incident (modeling "this needs a new capability, not a fix to the existing one"). Verified
  directly (`tests/unit/test_rl_env.py`): low severity -> recalibrate wins; high severity ->
  regenerate wins; 2+ failed attempts -> expand_scope beats continuing to escalate recalibrate/
  regenerate.
- **Two real numerical-stability bugs were found and fixed while validating this environment**
  (same "always run it and check the real numbers" discipline as every prior module — see
  Modules 6/9/10's bug writeups): (1) re-sampling a fresh random `y_true` on every
  `FidelityEvaluator.evaluate()` call let *sampling noise between two independent random arrays*
  dominate the signal, occasionally collapsing the rolling window's (max-min) range near zero and
  producing rewards in the hundreds of thousands via Module 12's `eps`-stabilized denominator
  (correct behavior of that formula, catastrophic for this env) — fixed with a FIXED per-
  component `y_true` reference array so only prediction quality varies, never the ground truth
  itself. (2) prewarming every component's rolling window with only healthy (near-zero-error)
  points made the FIRST severely-drifted evaluation after prewarm compare a large value against
  a near-constant window — fixed by prewarming with a realistic SPREAD of error magnitudes
  (healthy to fully-drifted) so the window already reflects a mixed operational history before
  any episode begins. A defensive clamp (`_FIDELITY_SCORE_CLIP = (-3.0, 1.0)`) was also added as
  a local RL-stability measure over Module 12's own (correctly unbounded-below) return value —
  Module 12's formula itself was never modified.
- **A genuine reproducibility bug was also found and fixed via `gymnasium.utils.env_checker.
  check_env`**: an initial design kept ONE `FidelityEvaluator` instance persistent across
  episodes (so its rolling windows evolved "realistically" over a training run) — but this made
  `reset(seed=X)` non-reproducible, since episode N's observations silently depended on the
  accumulated history of episodes 0..N-1, which the seed didn't capture. Fixed by rebuilding
  (and re-prewarming) a fresh `FidelityEvaluator` inside every `reset()`, deterministically
  reseeded from the env's own RNG — `check_env(env, skip_render_check=True)` passes cleanly.
- **`rl_agent.py`**: `build_ppo_agent()` constructs a genuine `stable_baselines3.PPO("MlpPolicy",
  ...)` reading every hyperparameter from `config.ppo.training` (never hardcoded).
  `device="cpu"` is forced deliberately — SB3's own guidance is that a small flat-vector
  `MlpPolicy` (no CNN) is slower on GPU than CPU due to transfer overhead; this is a measured
  performance choice, not a capability limitation (the project's CUDA availability, noted in §12
  below, remains for the DT prediction models' potential future use). `make_vec_env()` uses
  `DummyVecEnv` (single-process), a deliberate choice since `AdaptationEnv.step()` is pure numpy
  with no I/O — multiprocessing overhead would dominate wall-clock rather than help, and it
  avoids multiprocessing/CUDA-fork pitfalls entirely. `train_ppo()` trains, saves the policy to
  `config.ppo.policy_path`, and returns the real per-episode reward history (via
  `EpisodeRewardLogger`, an SB3 `BaseCallback` reading `Monitor`-populated `info["episode"]`) so
  a caller can report genuinely observed numbers, never fabricated ones.
- **The runtime decision path is exactly one function call to PPO, with the CLAUDE.md §6
  fallback rule enforced in code, not just documentation**: `decide_adaptation_strategy(model,
  observation, settings)` calls `model.predict(observation, deterministic=True)` and nothing
  else — it can only return PPO's genuine decision or raise. `decide_adaptation_strategy_safe()`
  adds the ONE permitted exception: on a genuine PPO inference failure (missing/corrupt model,
  `model is None`, `predict()` raising), it falls back to `config.ppo.fallback.default_action`
  and logs a WARNING every time (`tests/unit/test_rl_agent.py` verifies both the fallback value
  and that the log line fires) — never a silent or routine substitute.
- **REAL training run, not placeholder numbers** (`scripts/train_ppo.py`, run 2026-09-07):
  **20,480 timesteps actually run** (requested 20,000; SB3 rounds up to whole rollout batches of
  `n_steps=256 x n_envs=4=1024` — a deliberately reduced budget vs. `config.ppo.training.
  total_timesteps`'s full 200,000, noted explicitly per this task's "16GB RAM, small run" scope,
  never silently substituted), **n_envs=4, device=cpu, wall-clock = 412.4s (~6.9 minutes),
  15,942 episodes completed** (episodes are short — 1 to `max_attempts_per_incident=4` steps
  each, since an episode is one drift incident, not a fixed-length rollout). **Real reward
  curve**: mean episode reward over the first 1,594 episodes = 0.2536, over the last 1,594 =
  0.3091 (~22% improvement) — saved as `data/artifacts/ppo_training/reward_curve.png` (raw
  per-episode reward, high-variance by construction since episodes span the full severity range
  and all 5 components, plus a `window=318` rolling mean showing the upward trend). SB3's own
  training diagnostics corroborate genuine learning, not noise: `explained_variance` rose from
  -0.58 (iteration 2) to 0.99 (iteration 20) — the value function learned to predict returns
  accurately — while `entropy_loss` fell from -1.09 to -0.17 as the policy became more decisive.
- **Held-out inference (genuinely `PPO.predict()`, never a heuristic) on scenarios never seen
  during training** (different RNG seed, `scripts/train_ppo.py`'s `HELD_OUT_SEED_BASE=9001` vs.
  training's `TRAIN_SEED_BASE=1`): on a **held-out low-severity fresh incident**, the trained
  policy's mean reward across 60 episodes was **-0.0151 — exactly matching an always-recalibrate
  baseline (-0.0151)** and beating always-regenerate (-0.0746): PPO learned that recalibrate is
  the right (cheap, sufficient) choice at low severity. On a **held-out high-severity fresh
  incident**, the trained policy's mean reward was **0.7010 — exactly matching an
  always-regenerate baseline (0.7010)** and clearly beating always-recalibrate (0.1812): PPO
  learned to escalate to regenerate when severity is high. Full numeric report (per-scenario
  means, episode lengths, and sample step-by-step trajectories) saved to
  `data/artifacts/ppo_training/training_report.json`; trained policy saved to
  `data/models/ppo/policy.zip` (`config.ppo.policy_path`).
- **Tests**: `tests/unit/test_rl_env.py` (13 — Gym API compliance via `check_env`, observation/
  action space shapes, reproducibility, termination/truncation logic, and — the key structural
  proof — that low/high severity and repeated-failure scenarios each have a different, correctly
  learnable optimal action), `tests/unit/test_rl_agent.py` (8 — agent construction reads real
  config, `decide_adaptation_strategy` genuinely calls `model.predict()` [verified via a spy],
  the fallback path returns the configured default and logs a WARNING only on genuine failure,
  never on success), `tests/integration/test_rl_training.py` (2 — a REAL small training run
  through `train_ppo()`, save/reload round-trip, and a held-out-scenario check that the trained
  policy beats a fixed always-recalibrate baseline at high severity — mirroring Modules 6-10's
  "real held-out validation, not placeholder numbers" discipline).

**LLM Infrastructure — centralized Anthropic API client: implemented and tested.**
`src/llm/anthropic_client.py`. Not one of the 19 numbered modules itself — this is the
cross-cutting infrastructure prompt.md §24/§42 requires *before* Modules 15 (Regeneration), 16
(Expand-Scope), 17 (Agentic Verification), and 19 (Lifecycle) can be built, plus Module 14
(Recalibration)'s optional training-window reasoning per CLAUDE.md §7. **Infrastructure only —
no agent logic**: no prompt templates, no per-agent output schemas, no business rules about when
to call the LLM. Key design decisions:

- **One centralized client, never a per-agent `anthropic.Anthropic()` instance.**
  `AnthropicClient.from_settings(settings, secrets)` is how every future agent will construct it
  — `config.llm.model` is the only place a model ID is set (never hardcoded in agent code, never
  a deprecated/obsolete model), and `Secrets.require_anthropic_key()` is the only source of the
  API key (raises a clear `RuntimeError` if unset — never a hardcoded key, never a silent empty
  fallback).
- **A real, load-bearing discovery, not an assumption from older docs**: the installed Anthropic
  API version was checked directly against `anthropic-sdk-python`'s actual
  `MessageCreateParams`/`OutputConfigParam` types (not older API documentation) — this API
  generation has **no `temperature`/`top_p`/`top_k` sampling-randomness parameter anywhere** in
  the Messages API. prompt.md §42's "deterministic/non-deterministic settings where appropriate"
  requirement is satisfied honestly rather than by inventing a parameter that would silently do
  nothing: `config.llm.effort` maps to the real `output_config.effort` field (reasoning depth:
  null/low/medium/high/xhigh/max) — explicitly documented in both `config/settings.yaml` and the
  client's own module docstring as NOT a determinism/sampling control, since this API genuinely
  has none. Getting this wrong (assuming `temperature` still existed) would have produced a
  parameter that the SDK silently ignores or rejects — exactly the kind of "actually run it and
  check" discovery this project's discipline exists to catch (see Modules 6/9/10/13's own bug
  writeups for the same pattern).
- **Structured output uses the API's own native mechanism, not prompt-engineered JSON
  extraction.** `complete_structured(prompt, schema)` passes
  `output_config.format={"type":"json_schema","schema":schema.model_json_schema()}` — a real
  feature of this SDK that constrains the model's response server-side — and *independently
  re-validates* the result against the caller's pydantic schema afterward regardless (CLAUDE.md
  §7: "LLM output is always untrusted" — never skipped just because the API is expected to have
  already enforced it). If validation still fails, the WHOLE call is retried (not just
  re-parsing, since only a fresh call can produce different output) up to `config.llm.
  max_retries` times before raising `LLMStructuredOutputError`.
- **All retry/backoff is this client's own explicit, testable code — not the SDK's.** The
  underlying `anthropic.Anthropic` client is constructed with `max_retries=0`, deliberately
  disabling the SDK's built-in transport retry layer, so there is exactly ONE retry schedule in
  this system (`_call_with_retry`), not two silently compounding (e.g. the SDK retrying 2x
  internally while our own loop also retries 3x, yielding up to 9 real HTTP attempts for one
  logical call). Retryable failures (`RateLimitError`, `APIConnectionError`, `APITimeoutError`,
  `InternalServerError`, `OverloadedError`, `ServiceUnavailableError`) get exponential backoff
  (`config.llm.retry_backoff_seconds * 2^attempt`, capped at a fixed 30s ceiling) — honoring a
  `Retry-After` response header when the API provides one (genuine rate-limit-aware handling,
  not just a fixed guess). Non-retryable failures (`AuthenticationError`, `PermissionDeniedError`,
  `BadRequestError`, `NotFoundError`, `UnprocessableEntityError`) fail fast on the first attempt
  — retrying bad credentials or a malformed request can never succeed, so the retry budget isn't
  wasted on them.
- **Graceful degradation is an explicit opt-in, not a default.** `complete()`/
  `complete_structured()` raise `LLMTransportError`/`LLMStructuredOutputError` (both subclass
  `LLMClientError`) on failure — the right behavior for a caller that needs a hard failure.
  `complete_safe()`/`complete_structured_safe()` catch `LLMClientError` and return `None`
  instead, always logging a WARNING when they do — mirroring Module 13's
  `decide_adaptation_strategy_safe` fallback discipline exactly, for the same reason (prompt.md
  §42: "LLM failures must not corrupt production"; CLAUDE.md §7's own example — recalibration's
  training-window reasoning is *optional*, so a future Module 14 should call the `_safe` variant
  and fall back to its deterministic default window on `None`, never letting an LLM outage halt
  adaptation).
- **`prompts.py`/`schemas.py` (prompt.md §42's suggested layout) were deliberately NOT created
  now.** Both would hold agent-specific content (prompt templates, per-agent structured-output
  shapes) that belongs with Modules 14-19 when they're actually built — creating them now with no
  real consumer would be exactly the speculative/premature content this project's conventions
  avoid. `AnthropicClient`'s own response types (`LLMResponse`/`LLMUsage`) are the only "schema"
  this layer owns.
- **Tested against a mocked transport, not a real network call — and this is stated plainly, not
  hidden.** No real `ANTHROPIC_API_KEY` is configured in this development environment (`.env`
  holds only the placeholder from `.env.example`); genuinely calling the live API would either
  fail loudly or silently accomplish nothing useful for a test, and prompt.md §0.24/§20 forbid
  faking a real API response. `tests/unit/test_anthropic_client.py` (20 tests, all passing)
  therefore drives `AnthropicClient` against a mocked `messages.create`, using REAL
  `httpx.Request`/`httpx.Response` objects and REAL `anthropic.*Error` exception types (so the
  client's actual exception-classification logic is genuinely exercised, not a stand-in) —
  covering: a trivial-prompt completion end-to-end; the configured model/effort are what's
  actually sent (never hardcoded); retryable-error-then-success (right attempt count, right
  sleep count); non-retryable errors fail fast with zero sleeps; exhausting the retry budget
  raises with the right attempt count; `Retry-After` header honored when present, exponential
  backoff when absent; structured output returns a validated pydantic instance, sends the
  correct `output_config.format`, retries on validation failure then succeeds, raises after
  exhausting validation retries, and rejects JSON missing a required field; both `_safe` wrappers
  degrade to `None` and log a WARNING on failure while still returning the real result on
  success; and a dedicated test asserts a real-looking API key string never appears in ANY log
  record's message or `extra` fields. One additional test,
  `test_live_api_smoke_if_key_configured`, is skipped (not faked) until a real key is ever
  configured — it would genuinely call the live API with a trivial prompt and a trivial
  structured schema if one is.

**Versioned Model Registry (Concept C — prompt.md §46/§29): implemented and tested.**
`src/registry/model_registry.py`. Built now (ahead of Module 15/16) because the user's explicit
instruction for this turn was to implement it as shared infrastructure those two future modules
will also need — every future adaptation agent registers its candidates the same way. **This is
the THIRD, distinct "registry" in this codebase** — see the module's own docstring for the full
disambiguation from `ComponentRegistry` (D1's state-table catalog, Module 4) and `DTModelRegistry`
(Module 5's in-memory component-wiring catalog, no versioning). `ModelRegistry` tracks MODEL
**VERSIONS**: every artifact ever produced (bootstrap/recalibrate/regenerate/expand_scope), which
one is currently `"production"` per component, and the full history needed for rollback — Concept
C from CLAUDE.md §4, deliberately independent of D1 (Concept B): promoting a version here never
touches D1, and D1 being updated never touches this registry (prompt.md §0.5).

- **`ModelVersionMetadata`** (frozen pydantic model) carries everything prompt.md §29 requires:
  component, version_id, model_class, artifact_path, dependencies, feature_schema, output_field,
  adaptation_type, parent_version_id, created_at, training_window, evaluation_window,
  evaluation_metrics, fidelity_before/after, status, llm_metadata. Immutable once created — status
  transitions (`promote`/`reject`/rollback's re-promote) replace the record wholesale via
  `model_copy`, never mutate in place, so a `"production"` record retained after being superseded
  is bit-for-bit what was actually promoted at the time.
- **Persistence**: one JSON index file (`data/models/registry_index.json`, atomic
  write-temp-then-rename, mirroring `D1Store`'s `_atomic_write_parquet` pattern) plus one artifact
  file per version (`data/models/<component>/<version_id>.joblib`), written via the component's
  own already-tested `DTComponent.save()` — the registry never reimplements serialization.
  Verified to survive a process restart: a fresh `ModelRegistry` instance pointed at the same
  paths sees exactly what a prior instance wrote, including promotion/rollback state
  (`tests/unit/test_model_registry.py`).
- **`register_version()` never decides `status` on its own** — a candidate is `"candidate"`
  unless the caller (e.g. bootstrap training) explicitly says otherwise; this registry never
  promotes anything itself (prompt.md §46: "must never allow an invalid candidate to become
  active"). `promote()`/`reject()`/`rollback()` exist as registry primitives Module 17 (Agentic
  Verification, not built yet) will call — nothing in this codebase invokes them automatically
  yet. Exactly one version can be `"production"` per component at a time, enforced by
  `promote()` touching every record for that component atomically under one lock.
- **`load_artifact_into(component_instance, version)`** restores a version's trained state into
  an already-constructed instance of the matching class — this registry deliberately does NOT
  reconstruct `DTComponent` instances itself (that would require a class-name lookup table
  duplicating `DTModelRegistry`'s own metadata); the caller already knows which class, since they
  supplied the trained instance to `register_version` in the first place.
- **Tests**: `tests/unit/test_model_registry.py` (15 — version creation + artifact persistence,
  version-ID incrementing per component, current-vs-candidate distinction, promote/reject/
  rollback including the "exactly one production version" invariant, error cases for
  unknown component/version, `load_artifact_into` producing a genuinely working restored
  component, and persistence-across-reload for both plain registration and post-promotion state).

**Module 14 — Recalibration Agent (Agent 2): implemented and tested.**
`src/adaptation/recalibration_agent.py`. Retrains an EXISTING, already-production DT component on
a recent D1 telemetry window — prompt.md §23: "the existing model structure/pipeline is still
appropriate, but its learned parameters/behaviour have become stale." This agent does not decide
*whether* to recalibrate (PPO/Module 13 already decided that — prompt.md §70 rule 4); it only
executes the strategy once told to, and never runs on a schedule ("do not perform blind periodic
recalibration"). Key design decisions:

- **All nine steps of prompt.md §23 map directly onto `RecalibrationAgent.recalibrate()`**:
  receive the component (`component_factory` param) -> inspect current model metadata
  (`ModelRegistry.get_current_version()`, raises `RecalibrationError` if none exists — recalibration
  is not how a component gets its first version) -> inspect recent history
  (`D1Store.get_history()`, a snapshot) -> determine a training window (config-driven default,
  optional LLM assist — see below) -> select training data (`_select_window` + a time-ordered
  train/held-out split) -> call the generic `train()` interface (Module 5's `DTComponent`
  contract, unchanged) -> produce a new candidate version (`ModelRegistry.register_version(...,
  status="candidate")`) -> evaluate it (component-local RMSE/MAE plus real Module 12
  fidelity-before/after) -> "pass to verification": Module 17 doesn't exist yet, so this agent's
  contract ends at returning a versioned, evaluated `RecalibrationResult` for a future caller to
  hand to it — it never self-promotes.
- **Continuous operation during adaptation is a structural property, not a convention followed by
  discipline** (prompt.md §0.6/§0.8): the agent calls `D1Store.get_history()` exactly once per
  `recalibrate()` call, which takes a `.copy(deep=True)` snapshot under a lock held only for that
  one copy, and calls NO D1 write method anywhere. The live `ContinuousSynchronizer` can therefore
  keep ingesting telemetry for the entire duration of a recalibration run, completely unblocked —
  proven concretely, not just asserted, by
  `tests/integration/test_recalibration_agent.py::test_telemetry_keeps_synchronizing_into_d1_while_recalibration_is_in_progress`:
  a real background `ContinuousSynchronizer` thread's `records_synced` count is sampled every 20ms
  by a separate sampler thread while a real (foreground) `recalibrate()` call is in progress
  (deliberately using a slower RandomForest(400 trees) so there's a real, measurable window to
  sample within), and shown to strictly increase during that window — re-run 3 consecutive times
  with stable ~14s timing, not flaky.
- **`ModelRegistry.get_current_version()`'s snapshot is likewise never written to** — the
  candidate produced is a brand-new `DTComponent` instance from `component_factory()`, never the
  production instance in place; `tests/unit/test_recalibration_agent.py::
  test_recalibrate_never_writes_to_d1` and `..._registers_candidate_with_correct_parent_and_status`
  both confirm the production version and D1's row counts are bit-for-bit untouched by a
  recalibration call.
- **Real Module 12 fidelity-before/after, using the SAME normalization reference** — when a
  `FidelityEvaluator` is supplied, `_compare_fidelity` loads the CURRENT production artifact via
  `ModelRegistry.load_artifact_into`, generates predictions from both the old and new component
  over the identical held-out window, and calls `evaluate(..., update_window=True)` for
  production then `evaluate(..., update_window=False)` for the candidate — sharing one window,
  exactly prompt.md §0.10's "BOTH MUST USE THE SAME NORMALIZATION REFERENCE" requirement. **This
  is the first real consumer of `update_window=False`** — Module 12's own entry noted this hook
  was "tested now, just not consumed yet"; it's now genuinely exercised end-to-end. Never
  fabricated: with a freshly-constructed evaluator (no prior rolling-window history),
  fidelity_before/after both come back `None` (Module 12's own "insufficient_history" status,
  never bypassed) — verified explicitly, alongside a second test pre-warming the evaluator past
  `min_history_for_normalization` with real `evaluate()` calls and confirming real, defined scores
  come back.
- **Dependency-having components are handled generically, not hardcoded per component.**
  `dependency_output_fields: dict[str, str]` maps each `DEPENDENCIES` entry to that dependency's
  `OUTPUT_FIELD` (e.g. `{"throughput": "throughput_mbps_pred"}`); the agent populates that column
  from D1's ground-truth value (`output_field.removesuffix("_pred")`) — the same "train on ground
  truth, serve on live predictions" convention Modules 7/9/10 already established, applied
  generically here instead of re-special-cased. Verified by recalibrating `LatencyModel` (which
  depends on `throughput`), not just a root component.
- **LLM training-window reasoning is optional and off by default**
  (`use_llm_window_reasoning=False`), per prompt.md §23 ("MAY use LLM reasoning... if beneficial")
  and CLAUDE.md §7. When enabled with an `AnthropicClient` supplied, the agent asks it (via
  `complete_structured_safe` — graceful degradation built in) to suggest a window length given a
  short summary of recent history, but the suggestion is always clamped to `[default*0.25,
  default*4]` before use — the clamping code, not the LLM, ultimately bounds what's used. Any LLM
  failure (`None` from `complete_structured_safe`) falls straight back to the deterministic
  config default. The actual metric calculations and model training are ALWAYS deterministic
  code regardless of whether the LLM path ran (prompt.md §23's explicit requirement) — verified
  by three tests: disabled-by-default never calls the client; an absurd (1000x) suggestion is
  proven clamped; a `None` response is proven to fall back to the exact config default.
- **Tests**: `tests/unit/test_recalibration_agent.py` (17 — window-selection filtering on a
  hand-built history with mixed old/recent timestamps, dependency ground-truth wiring and its
  error cases, missing-production-version and insufficient-training-rows errors, candidate
  registration/parent-linkage, a working trained candidate, D1 never written to, fidelity
  none-vs-real as above, the dependency-having-component case, and all three LLM-reasoning
  behaviors) and `tests/integration/test_recalibration_agent.py` (2 — a real candidate through
  the full Module 2/3/4 pipeline that genuinely beats a naive baseline, plus the concurrent-
  telemetry proof above).

**Sandbox Execution Layer (prompt.md §45): implemented and tested.**
`src/sandbox/{executor.py,_sandbox_driver.py}`. Built this turn (ahead of Module 16) as shared
infrastructure Module 15's LLM-generated code needs and Module 16's will too — the same
"user asked for shared infra now" pattern as Module 14's versioned registry. This is the concrete
mechanism behind CLAUDE.md §7/prompt.md §70 rule 10: "LLM-generated code is never production code
until sandboxed and verified."

- **Real subprocess isolation, not `exec()`/`importlib` in-process.** `SandboxExecutor.
  run_candidate()` writes a candidate's source to an isolated per-call workspace and runs a
  FIXED, project-authored driver (`_sandbox_driver.py` — never LLM-generated, never modified by a
  candidate) via `subprocess.run`, never importing the untrusted code into this process. A
  candidate that raises, hangs, or calls `sys.exit()` at import time only ever affects its own
  subprocess.
- **Concretely, testably secret-safe**: the subprocess environment is built from scratch (`PATH`
  + a `PYTHONPATH` pointing at this repo) — never `os.environ.copy()`. Verified directly, not
  just asserted: a test sets a real-looking secret in the parent process's environment, and a
  candidate that tries to read it back gets `NOT_FOUND_IN_SANDBOX`
  (`tests/unit/test_sandbox_executor.py::test_secret_env_vars_are_not_inherited_by_the_sandbox`).
- **Security assumptions documented honestly, exactly as prompt.md §45 asks** (see the module's
  own extensive docstring): subprocess isolation is real, load-bearing protection against the
  candidate touching the PARENT process's memory/imports/secrets, and `timeout_seconds`/
  `max_output_bytes` bound runaway execution/output — but the subprocess still runs as the SAME
  OS user with the SAME filesystem permissions as the parent, so nothing here stops a
  deliberately adversarial script from using an absolute path to read/write outside its
  workspace. True filesystem/network sandboxing would need container/namespace primitives this
  environment can't assume are available — documented as a real, deliberate gap rather than
  pretended away. A second deliberate, documented omission: no POSIX `resource.setrlimit` via
  `subprocess`'s `preexec_fn` for CPU/memory — Python's own docs warn `preexec_fn` is unsafe in a
  multi-threaded process (this system runs `ContinuousSynchronizer` on a background thread
  continuously), and the fork-deadlock risk was judged worse than the resource-limit gap it would
  close, given `timeout_seconds` already bounds wall-clock in practice.
- **`_sandbox_driver.py`'s stages map directly onto prompt.md §26's pipeline**: syntax (checked
  in-process via `compile(..., "exec")`, which only parses — never executes — so no isolation is
  needed for it) -> import -> conformance (a FIXED, deterministic, project-authored check — not
  LLM-authored "unit tests" — that the candidate subclasses `DTComponent`, is constructible with
  no required args, and its `COMPONENT_NAME`/`OUTPUT_FIELD` exactly match what other components
  already depend on; `DEPENDENCIES`/`REQUIRED_FEATURES` may legitimately differ — that's the
  "rebuilt pipeline") -> train -> evaluate -> save. Every stage is wrapped so ANY failure —
  including a candidate raising something exotic — is always captured as a structured
  `result.json`, never an opaque crash.
- **Data hand-off is one-directional file-based INTO the sandbox, in-memory data-only back OUT.**
  The parent writes parquet snapshots of the training/eval data in; the driver writes back a
  trained artifact's raw BYTES and the candidate's held-out predictions as plain numeric values
  (never anything executable) — `SandboxExecutor` reads both straight into memory and ALWAYS
  deletes the workspace in a `finally` block before returning (prompt.md §45 "clean up temporary
  workspaces safely"), regardless of success, rejection, or an unexpected exception. Verified
  directly: workspace directories are asserted gone after success, every rejection stage, and a
  timeout.
- **Tests**: `tests/unit/test_sandbox_executor.py` (15 — a valid hand-written candidate accepted
  with real metrics/artifact bytes/predictions; every rejection stage individually [syntax,
  import, wrong `COMPONENT_NAME`, wrong `OUTPUT_FIELD`, missing abstract methods, `train()`
  raising, `evaluate()` returning the wrong type]; timeout enforcement bounded by the configured
  limit, not the candidate's actual runaway sleep; the secret-non-inheritance proof; workspace
  cleanup after success/rejection/timeout; stdout truncation to `max_output_bytes`; and a direct
  byte-for-byte proof that a real production source file (`src/dt_models/throughput.py`) is
  untouched by a candidate run).

**Module 15 — Regeneration Agent (Agent 3): implemented and tested.**
`src/adaptation/regeneration_agent.py`. Rebuilds an EXISTING, already-production DT component's
pipeline via LLM-generated, sandboxed code — prompt.md §24: "the current model architecture/
pipeline is no longer capable of representing the changed behaviour." Like Module 14, this agent
does not decide *whether* to regenerate (PPO already decided); it only executes the strategy.
Key design decisions:

- **All of prompt.md §24-26 maps directly onto `RegenerationAgent.regenerate()`** — see the
  module's own docstring for the full nine-step breakdown (mirroring Module 14's convention):
  inspect current version + real source code (`inspect.getsource`) -> gather context (§25's
  exact list: metadata, recent feature statistics, error-pattern residuals of the current
  production model on a held-out window, fidelity metrics, caller-supplied drift context, RAG
  context explicitly reported unavailable since Module 18 isn't built — never faked) -> generate
  candidate source via the centralized `AnthropicClient` with a structured-output schema
  (`class_name`/`source_code`/`reasoning` — prompt.md §26 "use strict structured output where
  possible," not free-form text parsed with regex) -> sandbox it -> self-correct on rejection
  (up to `config.adaptation.regeneration.max_llm_iterations`, feeding the rejection stage/error
  back to the LLM as prompt context — genuinely proven to work: a first BROKEN candidate gets
  rejected, the retry prompt is confirmed to contain "REJECTED by the sandbox," and a second
  FIXED candidate is accepted) -> register the accepted candidate
  (`ModelRegistry.register_version_from_artifact`) -> real Module 12 fidelity-before/after,
  computed in the parent from DATA the sandbox returned, never from the candidate's own
  self-reported metrics -> return for a future Module 17 to verify (not built yet).
- **A candidate may legitimately redeclare `DEPENDENCIES`/`REQUIRED_FEATURES` — that IS
  "rebuilding the pipeline" — but `COMPONENT_NAME`/`OUTPUT_FIELD` never change.** The parent
  hands the sandbox the FULL available window (every raw D1 column plus every dependency's
  ground-truth column already populated, via the same `data_selection.with_dependency_ground_
  truth` helper Module 14 uses) rather than a pre-subset DataFrame — the candidate (like every
  existing `DTComponent`) selects whatever columns it actually needs by name. This is what lets
  the LLM genuinely redesign a pipeline's inputs without the caller needing to know in advance
  what it will choose. `COMPONENT_NAME`/`OUTPUT_FIELD`, in contrast, are the identity contract
  other components already depend on — the sandbox's conformance check REJECTS any candidate
  that changes either, verified directly by dedicated tests.
- **`ModelRegistry.register_version_from_artifact()`** (extended this turn) is what makes
  registering a sandboxed candidate possible at all: unlike Module 14's `register_version()`
  (which calls `.save()` on an already-trusted live instance), this process never imports the
  LLM-generated class — it only ever has the artifact's raw bytes and the candidate's own
  self-reported metadata (dependencies/features, from the sandbox's `result.json`), so this
  method just copies the already-produced artifact (and, for audit, the source code — prompt.md
  §29 "source code/artifact location") into the registry's managed storage. Documented
  consequence: a regenerated candidate's class can never be reconstructed as a live instance in
  this process either — promoting one to actually SERVE production predictions is therefore a
  follow-up concern for whenever Module 17 and a vetted dynamic-loading path exist.
- **`src/adaptation/data_selection.py`** — extracted this turn from Module 14's agent
  (`select_recent_window`/`time_split`/`with_dependency_ground_truth`/`iso`) once Module 15
  needed the EXACT same logic; both agents now share one implementation instead of two
  copies drifting apart. `RecalibrationAgent`'s own tests were updated to import from the new
  shared module; its behavior is unchanged (18 tests still pass across both files).
- **Continuous operation during adaptation, proven under a heavier real workload than Module
  14's**: the same structural guarantee (one `D1Store.get_history()` snapshot, zero D1 writes)
  now covers an LLM call PLUS a genuinely sandboxed subprocess training run, not just an
  in-memory retrain. `tests/integration/test_regeneration_agent.py`'s concurrency test — the
  same sampling-thread-during-the-call pattern Module 14 established — confirms D1's
  `records_synced` counter keeps growing throughout, re-run 3 consecutive times with stable ~8s
  timing, not flaky.
- **No real ANTHROPIC_API_KEY is configured in this environment** (same situation as the LLM
  Infrastructure entry above) — every test drives `RegenerationAgent` against a fake LLM client
  returning hand-written source strings standing in for real model output, exactly the
  established "mock the transport, run everything downstream for real" discipline. The
  self-correction loop, the sandbox execution, the registry writes, and the fidelity comparison
  are ALL real, unmocked code, genuinely exercised end-to-end.
- **Tests**: `tests/unit/test_regeneration_agent.py` (11 — missing-production-version and
  insufficient-training-rows errors, successful candidate registration with correct parent/
  metadata, production version and D1 provably untouched, the self-correction retry loop with
  feedback verified present in the retry prompt, exhausting all attempts registers nothing in the
  registry, an LLM transport failure wrapped as `RegenerationError`, fidelity none-vs-real
  exactly mirroring Module 14's pattern, and the prompt genuinely containing the current
  production source code/drift context/RAG-unavailable note) and
  `tests/integration/test_regeneration_agent.py` (2 — a real candidate through the full pipeline
  AND the real sandbox subprocess, plus the concurrent-telemetry proof above).

**Module 16 — Expand-Scope Agent (Agent 4): implemented and tested.**
`src/adaptation/expand_scope_agent.py`. The third and final adaptation agent — used when "network
behaviour reveals a phenomenon/capability the current DT does not represent" (prompt.md §27), as
opposed to Modules 14/15 which both act on an component that already exists. Like Modules 14/15,
this agent does not decide *whether* to expand scope — PPO already decided; a caller-supplied
`expand_scope_context` explains what triggered it, exactly like Module 15's `drift_context`. Key
design decisions:

- **All sixteen of prompt.md §27's steps map onto `ExpandScopeAgent.expand_scope()`** — full
  breakdown in the module's own docstring, mirroring Modules 14/15's convention. The genuinely
  new piece relative to Module 15: steps 6-9 ("determine a suitable new DT component, derive its
  inputs/output, define feature extraction requirements") are their own DESIGN LLM call,
  producing a structured `_ProposedComponentDesign` (`component_name`, `target_column`,
  `dependencies`, `required_features`, `purpose`, `feature_extraction_notes`) —
  **deterministically validated BEFORE any code is generated**: the proposed name must be
  genuinely new (absent from both the live `DTModelRegistry` AND `ModelRegistry`'s version
  history), the target column must be a real D1 telemetry column, every dependency must be an
  existing registered component, and every required feature must be an available column. A
  design that fails validation gets a self-correcting retry (the validation error fed back as
  prompt context) up to `config.adaptation.expand_scope.max_llm_iterations` attempts — entirely
  separate from, and prior to, the SEPARATE self-correcting retry loop around the implementation/
  sandbox step Module 15 already established (reused here verbatim).
- **`output_field` is deterministically derived (`f"{target_column}_pred"`), never LLM-proposed**
  — the same `"<canonical>_pred"` convention Modules 6-10 already established, computed by code
  instead of asked of the LLM, removing an entire class of possible naming mistakes before they
  can happen.
- **"Run any generated pipeline through the same sandbox as Module 15" — literally true, not just
  in spirit.** `SandboxExecutor`/`_sandbox_driver.py` are imported from `src/sandbox/executor.py`
  completely UNCHANGED; this module adds zero sandbox code. The driver's conformance check
  (COMPONENT_NAME/OUTPUT_FIELD must exactly match what's "expected") works identically whether
  "expected" came from an existing production version (Module 15) or a freshly-validated, never-
  before-used design proposal (this module) — the driver has no idea which agent invoked it.
- **A deliberate, carefully-reasoned safety boundary: this agent does NOT wire its candidate into
  the live `DTModelRegistry`/`DTOrchestrator` (Module 5), even though prompt.md §27 lists
  "register the candidate" as one of ITS OWN steps.** "Register... dynamically through the
  registry" is satisfied by `ModelRegistry` (Concept C, Module 14's versioned artifact store) —
  proven directly by this module registering a component name that has never existed before, with
  zero code anywhere in `ModelRegistry` enumerating or limiting which names may exist. Actually
  wiring the candidate into the LIVE orchestrator so it SERVES predictions would require this
  process to import the LLM-generated class (or `joblib.load()` its pickled artifact) BEFORE
  verification exists to grant that trust — exactly what prompt.md §28/§70 rule 10 forbid. A
  genuinely important nuance surfaced while reasoning through this (not just an abstract caution):
  even `joblib.load()`-ing ONLY the trained artifact's bytes (never importing the candidate's
  source) is not unconditionally safe either — pickle deserialization is a well-known code-
  execution vector regardless of "it's just a sklearn model," since a maliciously-crafted
  `__reduce__` could execute arbitrary code on load. This is exactly why Module 15's (and this
  module's) `SandboxExecutor` never calls `joblib.load()` on a candidate's artifact in the parent
  process either — the parent only ever treats artifact bytes as an opaque blob, written straight
  to a file `ModelRegistry` manages, never deserialized here. Promoting a verified candidate into
  the live orchestrator is therefore a follow-up concern for whenever Module 17 exists AND a
  vetted, deliberately-hardened loading path is built for it — not something either Module 15 or
  16 should improvise.
- **The underlying mechanism this eventual promotion would rely on is proven anyway — using a
  TRUSTED stand-in, never the raw LLM/sandbox output.**
  `tests/integration/test_expand_scope_agent.py::test_dt_model_registry_dynamically_accepts_a_
  brand_new_component_alongside_existing_ones` registers real, individually-trained instances of
  ALL FIVE Modules 6-10 components PLUS a sixth, genuinely new `_TrustedSinrQualityModel` (a
  test-authored `DTComponent` — its name appears nowhere in Module 5's own source) into one
  `DTModelRegistry`, and confirms `DTOrchestrator.build_execution_order()`/`run_predictions()`
  correctly schedule and run all six. This is the concrete proof that Module 5's registration
  mechanism itself has no hardcoded component-count or name limit — exactly the capability a
  future, vetted promotion step would depend on, demonstrated honestly without pretending this
  module already does the unsafe part.
- **Worked example used consistently across the design and every test** (standing in for real
  LLM output — no real `ANTHROPIC_API_KEY` is configured in this environment, same situation as
  Modules 15/"LLM Infrastructure"): a new `sinr_quality` component predicting `sinr_db` (SINR)
  from mobility/load features — genuinely useful (proactive handover/resource-allocation
  signal quality prediction) and genuinely novel (none of the existing 5 components predict SINR;
  all of them treat it as an INPUT). Chosen because it's a realistic, well-motivated "phenomenon
  the current DT does not represent," not an arbitrary placeholder.
- **Continuous operation during adaptation, same structural guarantee as Modules 14/15**: one
  `D1Store.get_history()` snapshot, zero D1 writes, for the entire duration of a design LLM call
  + implementation LLM call + real sandboxed subprocess training run.
  `tests/integration/test_expand_scope_agent.py`'s concurrency test — the same sampling-thread
  pattern established by Module 14 — confirms D1's `records_synced` counter keeps growing
  throughout, re-run 3 consecutive times with stable ~11-13s timing, not flaky.
- **Every piece of shared infrastructure from Modules 14/15 is reused with ZERO code changes**:
  `AnthropicClient`, `ModelRegistry` (including `register_version_from_artifact`), `data_
  selection.py`'s helpers, and `SandboxExecutor`/`_sandbox_driver.py`. This is the strongest
  evidence yet that those earlier design decisions — shared, generic interfaces; a sandbox that
  doesn't know or care which agent calls it; a registry with no hardcoded component list — were
  the right calls, not premature abstraction.
- **Tests**: `tests/unit/test_expand_scope_agent.py` (16 — insufficient-rows error; every design-
  validation rejection individually [name collision with `DTModelRegistry`, name already
  versioned in `ModelRegistry` via a distinct check path, unreal target column, unknown
  dependency, unavailable required features]; design self-correction; successful registration
  with `parent_version_id=None`/`fidelity_before=None`; `DTModelRegistry`/D1 provably untouched;
  the implementation self-correction retry loop; exhausting attempts registers nothing; an LLM
  failure at EITHER the design or implementation step wrapped as `ExpandScopeError`; fidelity
  none-vs-real; and the design prompt genuinely containing the existing-components list and the
  RAG-unavailable note) and `tests/integration/test_expand_scope_agent.py` (3 — a real new
  component through the full pipeline and the real sandbox subprocess, the concurrent-telemetry
  proof, and the six-component dynamic `DTModelRegistry`/`DTOrchestrator` proof above).

**D2 (Module 18) — RAG Knowledge Base: implemented and tested.** `src/rag/rag_kb.py` +
`scripts/ingest_rag.py` + real content under `rag_data/{oran,digital_twin,policies,history}/`.
Implemented out of the original module order (before Module 17) per this turn's explicit
instruction. Full `documents -> chunking -> embeddings -> vector store -> retrieval -> LLM
context` pipeline (prompt.md §35), backed by a real persistent ChromaDB store. Key design
decisions:

- **Read-only is structural, not a policy comment — the requirement this turn most needed to be
  concretely verified, not just documented.** Two separate classes: `RagKnowledgeBase` (the ONLY
  class any agent ever receives) exposes exactly `retrieve()`/`count()`/`is_available` — no
  write/add/update/delete/ingest method exists anywhere on it, genuinely absent, not hidden or
  disabled. `RagIngestor` (admin/build-time-only — never constructed by or handed to an agent) is
  the ONLY class that writes to the vector store. Neither class's constructor accepts a
  `D1Store`/`ModelRegistry`/`DTModelRegistry` reference, so there is no object-graph path from
  either class to DT state — proven directly:
  `test_agent_cannot_reach_a_write_path_through_the_knowledge_base` (the concrete "write-through
  path from an agent" test this turn required) simulates an agent holding a `RagKnowledgeBase`
  and attempts ~20 plausible write-method names (`add`, `write`, `update`, `delete`, `upsert`,
  `register`, `promote`, `write_to_d1`, ...), asserting every one raises `AttributeError` and that
  the object's entire public surface is `{retrieve, count, from_settings}`.
  `test_knowledge_base_holds_no_reference_to_dt_state_objects` plus matching constructor-signature
  checks prove this structurally (no D1/registry object is reachable from an instance, and
  neither `__init__` even accepts one as a parameter — a future edit couldn't accidentally
  introduce a write path without the test suite catching it immediately).
  `test_ingestion_never_touches_d1_or_model_registry_state` proves it at the filesystem level too:
  a real `D1Store` and `ModelRegistry` are constructed alongside a real RAG ingestion run, and
  their on-disk files are confirmed byte-for-byte unchanged afterward.
- **Chunking**: a deterministic, word-based greedy chunker (`chunk_text()`). An initial
  character-offset sliding-window design (snapping chunk boundaries forward to the nearest
  whitespace) was tried first and rejected once testing showed the boundary-snapping logic could
  still let a SUBSEQUENT chunk's start land mid-word (the snap only adjusted `end`, not the next
  chunk's `start`, which was computed from a fixed step independent of where `end` actually
  landed). Rewritten to operate on whole words from the start — packing words greedily up to
  `config.rag.chunk_size` characters, carrying `config.rag.chunk_overlap` characters' worth of
  trailing whole words into the next chunk — which makes "never splits a word" a structural
  guarantee instead of a best-effort snap, and is simpler code besides.
- **Embeddings**: ChromaDB's own bundled `DefaultEmbeddingFunction`, which runs `all-MiniLM-L6-v2`
  (exactly `config.rag.embedding_model`'s configured value) via a local ONNX runtime — chosen
  over adding `sentence-transformers` (which would pull in a much larger dependency stack just to
  re-select a model ChromaDB already runs natively) — prompt.md §10's "do not introduce
  unnecessary infrastructure" applied to a dependency choice, not just an architecture one.
  Genuine, real embeddings — verified directly, not assumed: `test_embeddings_are_genuinely_
  semantic_not_a_stub` confirms an on-topic query against its own category retrieves a measurably
  closer match than the identical query against an unrelated category. Documented, tested
  limitation: `config.rag.embedding_model` set to anything other than `"all-MiniLM-L6-v2"` raises
  a clear `ValueError` at construction (a real config-validation error) rather than silently doing
  nothing — a real bug caught and fixed while building this: an early version of
  `RagKnowledgeBase`'s constructor wrapped ALL construction logic in one broad
  try/except-and-degrade-to-"unavailable" block, which silently swallowed exactly this
  `ValueError` into a misleading "RAG unavailable" state instead of the config error it actually
  was — fixed by validating the embedding model BEFORE entering the try/except that's meant only
  to catch genuine store-connection failures.
- **Idempotent ingestion** (prompt.md §36): every chunk's ChromaDB ID is deterministic
  (`f"{category}/{document_id}::{chunk_index}"`) and every write uses `upsert()` — re-running
  ingestion on unchanged files overwrites identically rather than duplicating (verified directly:
  re-running the real `scripts/ingest_rag.py` against the real corpus twice leaves the chunk
  count unchanged at 31, not 62). If a document's content changes, the same IDs are overwritten
  with new text. If a document SHRINKS (fewer chunks than a previous ingest produced), the
  now-stale trailing chunk IDs from the longer previous version are explicitly deleted, so
  retrieval can never return orphaned old content from a document that's since gotten shorter.
- **Metadata** (prompt.md §36's exact list — document ID, source, category, version, timestamp,
  chunk index) is recorded on every chunk. `source` is a real URL when a document begins with a
  `<!-- source: URL -->` comment (stripped from the indexed text before chunking/embedding), else
  the document's own path relative to its corpus directory — every chunk has genuine, real
  provenance, never left blank. `version` is a content hash, so it changes exactly when a
  document's content does.
- **RAG-unavailable graceful degradation** (prompt.md §61: "continue only where RAG is
  non-critical; never fabricate retrieved information"): a broken/corrupt store degrades
  `RagKnowledgeBase` into an explicit `is_available=False` state at construction (logged as a
  WARNING), and `retrieve()` raises `RagUnavailableError` rather than returning an empty list that
  a careless caller might mistake for "genuinely nothing relevant was found."
- **Corpus content — real, not fabricated, with honest labeling of what's thin** (exactly as this
  turn instructed): **oran** (2 documents, genuinely fetched via WebFetch from
  `docs.o-ran-sc.org` — the O-RAN Software Community's real public documentation site — covering
  the architecture overview [Near-RT RIC/Non-RT RIC/O-CU/O-DU/O-RU/xApp definitions, O1 interface]
  and the A1 interface/policy-management data model; explicitly noted as thin relative to the
  FULL O-RAN Alliance specification corpus, which requires a member-portal registration this
  session could not complete — but genuinely real, source-attributed content, never invented from
  memory). **digital_twin** (2 documents, derived faithfully from this repository's own real
  `CLAUDE.md` — the canonical data flow, the three state concepts, the fidelity formula, the
  six-agent breakdown, and the adaptation safety rules — arguably the most directly relevant
  "digital twin documentation" obtainable, since it documents the actual DT this knowledge base
  serves). **policies** (1 document, derived faithfully from this repository's own real,
  currently-enforced `config/settings.yaml` values — the actual verification delta, per-strategy
  cost penalties, per-agent training windows, sandbox timeout/output limits, and drift validation
  bounds, not illustrative examples). **history** (1 document, explicitly and prominently labeled
  placeholder-thin at the top of the file itself: real events from this project's own Module
  14/15/16 development validation runs, honestly distinguished from genuine production lifecycle
  records — which cannot exist yet, since Module 19 isn't built and the system has not run
  continuously against live traffic — with a closing section spelling out exactly what fields a
  REAL future record would additionally have).
- **Deliberately NOT done this turn, per its own stated scope**: Modules 15/16's LLM-prompt
  context building still hardcodes `"RAG context: not available — Module 18 ... is not built
  yet"` — now factually outdated, but updating it to genuinely call `RagKnowledgeBase.retrieve()`
  would be a Module 15/16 code change, and this turn's instruction scoped documentation updates to
  "D2 only." Flagged as the natural next follow-up in `IMPLEMENTATION_STATUS.md`.
- **Tests**: `tests/unit/test_rag_kb.py` (24 — chunking correctness including the never-splits-
  a-word guarantee and empty-input handling, ingest/retrieve round trip, metadata correctness,
  source-comment extraction and its path fallback, idempotent re-ingestion for unchanged/changed/
  shrinking documents, category filtering, `top_k`, the unsupported-embedding-model config-error
  fix described above, the smoke-test round trip, RAG-unavailable graceful degradation, and the
  write-through-path rejection tests described above) and `tests/integration/test_rag_kb.py` (7 —
  the real corpus ingests all four categories, real O-RAN content attributed to its actual fetched
  URL, real project-architecture/policy/history content retrieved correctly, genuine semantic
  embedding behavior, and idempotent re-ingestion of the real corpus).

**Module 17 — Agentic Verification Agent: implemented and tested.**
`src/adaptation/verification_agent.py`. This is the ACCEPT/REJECT gate every candidate produced
by Modules 14 (Recalibration), 15 (Regeneration), or 16 (Expand-Scope) must pass before it can
ever become production — and unlike those three agents (which all deliberately stop at "produce
an evaluated candidate" and never self-promote), this agent's entire purpose is to decide AND act:
`ModelRegistry.promote()` on ACCEPT, `ModelRegistry.reject()` on REJECT. It is the "trusted
deterministic application logic" CLAUDE.md §8 refers to ("only trusted deterministic logic
promotes"; prompt.md line 506: "Promotion is performed only by trusted deterministic application
logic after verification"). Key design decisions:

- **Two layers, kept structurally separate, exactly as prompt.md §31 specifies**:
  1. **Deterministic layer** (`verify()`'s main body): recomputes RMSE/MAE/Wasserstein/MK-MMD
     and the composite FidelityScore *independently*, via the SAME `src.fidelity.evaluator.
     FidelityEvaluator` Module 12 exposes — never trusting an agent's own self-reported
     `fidelity_before`/`fidelity_after` (those remain on `ModelVersionMetadata` purely as audit
     metadata from whichever agent produced the candidate). The primary criterion is exactly
     prompt.md §32: `FidelityScore_new > FidelityScore_old + delta`
     (`config.adaptation.verification_delta` — already existed since Modules 14-16's turns, no
     config change needed this turn), plus mandatory sanity conditions (candidate artifact
     present on disk, interface identity consistent, evaluation actually ran, output finite) —
     ALL must pass (CLAUDE.md §8: "any failed mandatory condition -> REJECT").
  2. **Agentic reasoning layer** (`_explain()`): an OPTIONAL LLM call producing a human-readable
     explanation — run strictly AFTER the deterministic `decision` is already final and ALREADY
     ACTED ON (promote/reject has already happened by the time this runs). **Rule 9 (prompt.md
     §70: "LLMs cannot override deterministic acceptance criteria") is enforced structurally, not
     by prompt wording alone**: the LLM's structured-output schema, `_LLMVerificationReasoning`,
     has exactly two fields — `explanation`/`key_observations` — and genuinely no field anywhere
     that could express an accept/reject/verdict/override; even a manipulative response's free
     text is stored verbatim into `VerificationResult.explanation` and never parsed for a
     decision signal.
     `tests/unit/test_verification_agent.py::test_llm_disagreement_never_changes_the_
     deterministic_accept_decision` and `..._reject_decision` prove this concretely, in BOTH
     directions: a fake LLM explicitly arguing for the OPPOSITE of what the deterministic gate
     computed (a deterministic ACCEPT scenario paired with an LLM saying "REJECT this", and a
     deterministic REJECT scenario paired with an LLM saying "ACCEPT this") — in both cases the
     final `decision` AND the actual `ModelRegistry` promotion/rejection are shown to follow the
     deterministic result, never the LLM's stated opinion.
     `test_llm_reasoning_schema_has_no_field_that_could_express_a_verdict` additionally enumerates
     a list of plausible verdict-like field names (`decision`/`verdict`/`accept`/`override`/...)
     and asserts none exist on the schema at all — the same "enumerate every plausible
     write/override surface and prove it's absent" pattern D2's own write-path test established.
- **Candidate predictions are supplied as plain data, never as an untrusted live instance** — the
  same "never import/execute LLM-generated code in this process" boundary Modules 15/16 already
  established for the sandbox. `verify()` takes `candidate_predictions` (already-computed values —
  `RecalibrationResult.candidate_component.predict(...)` for a recalibration candidate, or
  `RegenerationResult`/`ExpandScopeResult`'s `sandbox_result.eval_predictions` — already plain
  data the sandbox subprocess handed back — for a regenerate/expand-scope candidate) rather than a
  component instance. This keeps `VerificationAgent` uniformly agent-type-agnostic: it never
  needs to know or care whether a candidate came from an in-process retrain or a sandboxed
  subprocess.
- **The production ("before") baseline, in contrast, IS re-evaluated by this agent itself** — via
  an optional `production_component_factory` that constructs a fresh instance of the CURRENT
  production class (a genuinely trusted, already-production class — nothing untrusted is imported
  here), which this agent loads the current production artifact into
  (`ModelRegistry.load_artifact_into`) and predicts with, on the EXACT SAME `eval_features`/
  `eval_target` the candidate was evaluated on (prompt.md §33: "before"/"after" must use "the same
  comparable evaluation protocol"). If a production baseline exists in the registry
  (`ModelRegistry.get_current_version(component)` is not `None`) but no
  `production_component_factory` was supplied, this is treated as a fail-safe REJECT (a dedicated
  `production_baseline_evaluable` check fails) — never as a silent skip of the primary criterion,
  which would let an unverifiable candidate slip through unchallenged.
- **"No baseline" (expand-scope) is handled honestly, not by inventing a fake old score.** When
  `ModelRegistry.get_current_version(component)` returns `None` (a brand-new component), the
  primary `FidelityScore_new > FidelityScore_old + delta` criterion is structurally inapplicable —
  there is no `FidelityScore_old`. The gate falls back to requiring only that the candidate's own
  FidelityScore be well-defined (i.e. NOT Module 12's own `insufficient_history` status) — a
  candidate whose fidelity can't yet be computed at all is REJECTed, fail-safe, rather than
  blindly accepted for lack of a comparison, mirroring Module 15/16's own already-established
  `fidelity_before=None` for genuinely-new candidates.
- **Normalization reference sharing is finally consumed, not just hooked** (prompt.md §0.10,
  Module 12's own documented `update_window=False` hook, noted as "tested now, just not consumed
  yet" back when Module 12 was built, then genuinely first consumed by Module 14's own
  production-vs-candidate comparison): production is evaluated with `update_window=True`
  immediately before the candidate is evaluated with `update_window=False` on the SAME
  `FidelityEvaluator` instance/window — the established Module 14/15/16 pattern, now the actual
  ACCEPT/REJECT decision mechanism it was always meant to support.
- **A genuine numerical-stability pitfall was found and avoided while building this module's
  tests** (the same "always run it and check the real numbers" discipline as every prior module —
  see Modules 6/9/10/13's own bug writeups): an initial test helper pre-warmed a fresh
  `FidelityEvaluator`'s rolling window using Module 12's OWN hand-computed-test convention
  (`y_true=[0]*k, y_pred=[a]*k`, a degenerate point-mass pair) at increasing `a` — this produces a
  perfectly valid RMSE/MAE/Wasserstein spread, but MK-MMD is provably scale/shift-invariant over
  such a degenerate pair (verified directly: `mk_mmd(zeros(5), full(5,a))` returned the IDENTICAL
  value `0.9117` for every `a` from 0.5 to 20), collapsing that one metric's rolling-window
  (max-min) range to ~0 and exploding the eps-stabilized normalization the first time a genuinely
  different mk_mmd value was evaluated afterward — a fidelity_score in the hundreds of thousands,
  the exact same failure mode Module 13's own PPO-environment writeup documents for the identical
  underlying reason. Fixed by pre-warming with genuinely-distributed `(y_true, y_pred=y_true+
  noise)` pairs at increasing noise levels instead of degenerate point masses — real variance in
  the underlying samples gives all four metrics, including MK-MMD, genuine non-degenerate
  variation across the window. This is a general lesson for any future test/bootstrap code that
  needs to pre-warm a `FidelityEvaluator`: Module 12's own point-mass unit-test convention is
  correct for a handful of EXACT hand-computed values, but is NOT a safe general-purpose
  window-seeding technique.
- **Concurrency lock (prompt.md §30) is explicitly out of scope for this turn** — `verify()`
  assumes exclusive access to `candidate_version`'s component for its own duration.
  `ModelRegistry.promote()`/`.reject()` are individually thread-safe (each holds the registry's
  own lock), but a full component-scoped adaptation lock spanning an entire `verify()` call is a
  follow-up for whenever the main continuous loop (prompt.md §39, not built yet) exists to
  actually run concurrent adaptations in the first place.
- **`ModelRegistry` gained one small, read-only addition**: `artifact_path(version) -> Path`,
  resolving a version's artifact to an absolute filesystem path without importing or
  deserializing it — used by this module's `candidate_artifact_present` check (prompt.md §32
  "candidate successfully loads", satisfied here as "the artifact genuinely exists on disk",
  since this process never imports/unpickles a candidate's class to literally load it — same
  pickle-deserialization caution Module 16's own docstring already documented).
- **Tests**: `tests/unit/test_verification_agent.py` (16 — a non-`"candidate"` status raises;
  a clearly-better real trained candidate is ACCEPTed and genuinely promoted (production
  demoted to `"superseded"`, never deleted); a clearly-worse one is REJECTed with production
  provably unchanged; NaN candidate predictions and length-mismatched predictions are both
  REJECTed, not crashed; a missing candidate artifact on disk is REJECTed; a production baseline
  existing without a `production_component_factory` is REJECTed fail-safe, not silently skipped;
  both "no baseline" branches (candidate fidelity computable -> ACCEPT, not yet computable ->
  REJECT); the two LLM-disagreement tests described above in both directions; the LLM schema
  structural test; an LLM transport failure degrades to the deterministic explanation with the
  decision entirely unaffected; RAG context is genuinely retrieved and reaches the LLM prompt
  when a knowledge base is supplied; RAG unavailable/`None` degrades to `"not available"`, never
  fabricated) and `tests/integration/test_verification_agent.py` (1 — the key deliverable: a
  REAL `RecalibrationAgent.recalibrate()` candidate, produced from the real Module 2/3/4
  bootstrap pipeline against a deliberately weak early-slice bootstrap production model, verified
  by a real `VerificationAgent.verify()` call with a fixed adversarial fake LLM ("REJECT this
  regardless of the numbers") — actual observed result on this machine:
  **fidelity_before = 0.9800, fidelity_after = 0.9987 -> ACCEPT**, the candidate genuinely
  promoted to production over the LLM's explicit contrary opinion, with the LLM's disagreement
  text confirmed present in `result.explanation` and confirmed to have had zero effect on the
  actual registry state).

**Module 19 — Lifecycle Management Agent (Agent 6): implemented, tested, and run end-to-end for
real.** `src/adaptation/lifecycle_agent.py`. This is the last of the six agents (CLAUDE.md §3) and
the last of the 19 numbered modules — it decides nothing; it RECORDS and EXPLAINS what every other
agent already decided. Key design decisions:

- **`LifecycleRecord` carries every field prompt.md §37 lists, no more and no fewer** — event ID,
  timestamp, drift event, affected component/scope, drift severity, RL observation/context, RL
  action, agent action, production version before, candidate version, fidelity before, fidelity
  after, verification result, verification explanation, training window, evaluation window, model
  metadata, LLM metadata if used, final status. Nothing here computes or re-derives any of these
  — every field is read verbatim from the real object each upstream module already produced (see
  the module's own extensive docstring for the exact source of every field). Two field-mapping
  decisions worth calling out explicitly: (1) `production_version_before` is simply the
  candidate's own `parent_version_id` — exactly what Module 14/15 recalibrated/regenerated FROM
  (or `None` for Module 16's genuinely new components) — rather than a fresh registry lookup,
  which would be WRONG on ACCEPT (the registry has already been updated to the candidate by the
  time this agent runs); (2) `fidelity_before`/`fidelity_after`/`verification_result`/
  `verification_explanation` all come from Module 17's `VerificationResult` — the INDEPENDENTLY
  RECOMPUTED values — never an agent's own self-reported `evaluation_metrics`/`fidelity_before`/
  `fidelity_after`, since Module 17 is this system's trusted authority for those numbers.
- **`agent_action` is built by `_summarize_agent_action()`, a small duck-typed helper** — Modules
  14/15/16's `RecalibrationResult`/`RegenerationResult`/`ExpandScopeResult` deliberately share no
  common base class (each agent's own turn made that call independently, and forcing one onto them
  retroactively purely for this module would be exactly the kind of premature-abstraction
  reshaping this project's conventions avoid), so this helper reads whichever fields each result
  object actually has (`evaluation_metrics` for recalibration; `sandbox_result`/`attempts` for
  regeneration; `sandbox_result`/`design_attempts`/`implementation_attempts`/`design` for
  expand-scope) rather than requiring an artificial shared interface.
- **Persistence is a genuinely append-only, auditable log** (prompt.md §37 "must be auditable"):
  `config.lifecycle.records_path` (`data/artifacts/lifecycle_records.jsonl`) is one JSON-Lines
  file — every `record_adaptation_event()` call appends exactly one line, under a lock, never
  rewrites or deletes a prior line. This satisfies "produces a versioned adaptation record": each
  record is permanently identified by its own `event_id`/`timestamp`, and — because every field
  already embeds the exact component/candidate VERSION IDs Modules 14-17 produced — the full
  version lineage of any adaptation is reconstructable from the log alone. `list_records()`/
  `get_record(event_id)` are the read path; verified to survive a process restart exactly like
  `D1Store`/`ModelRegistry` (a fresh `LifecycleAgent` instance pointed at the same path sees prior
  writes).
- **The maintenance report is generated automatically from the ALREADY-RECORDED
  `LifecycleRecord`** (prompt.md §38), never from a fresh query of live state (which could have
  moved on by report-generation time). `_deterministic_report()` — a plain template, no LLM
  needed — is ALWAYS produced first and used whenever no LLM client is supplied or the LLM call
  fails (`AnthropicClient.complete_safe`'s existing graceful-degradation convention, reused as-is
  — this module adds no new retry/fallback logic of its own): a maintenance report must exist for
  every event regardless of LLM availability. When an `AnthropicClient` IS supplied, it's asked to
  write better prose FROM the same deterministic facts (explicitly instructed never to invent new
  ones or change the already-final, already-acted-on verification result), with D2's
  `RagKnowledgeBase` consulted for "relevant contextual knowledge" (prompt.md §38's own explicit
  "Use RAG retrieval where useful") — gracefully degrading to `"not available"` exactly like
  Module 17's own RAG consultation when no knowledge base is supplied or it's unavailable. Every
  report is saved to `config.lifecycle.reports_dir/<event_id>.md`.
- **Tests**: `tests/unit/test_lifecycle_agent.py` (14 — every `LifecycleRecord` field correctly
  populated for an ACCEPT scenario; `final_status="rejected"` and the correct unchanged
  `production_version_before` for a REJECT scenario; `production_version_before=None` correctly
  reported for an expand-scope-shaped candidate; `_summarize_agent_action`'s field selection
  proven distinct across all three real result types in one test; `llm_metadata` correctly carried
  from the registry's `updated_version`; records persist and are readable by a FRESH
  `LifecycleAgent` instance; multiple events append without overwriting, in order; an unknown
  `event_id` raises `LifecycleError`; an empty log returns `[]`; the deterministic report covers
  every required topic with no LLM configured; an LLM-authored report is used VERBATIM when
  available; an LLM failure falls back to the deterministic report; RAG context is genuinely
  retrieved and reaches the LLM prompt when a knowledge base is supplied; no knowledge base
  degrades to `"not available"` and never crashes) and, the key deliverable,
  `tests/integration/test_lifecycle_agent.py` (1 — **the REQUIRED "run one full cycle" proof**,
  described in full below).
- **The full real cycle was run end-to-end and the resulting record was inspected for
  completeness, exactly as this turn's instruction required — not simulated, not asserted in the
  abstract.** `tests/integration/test_lifecycle_agent.py` chains together, for real: (1) a real
  raw drift notification through Module 11's actual `DriftDetectorInterface.process_event()`; (2)
  a real observation from a real `AdaptationEnv.reset()` (Module 13), decided by the REAL
  already-trained PPO policy on disk (`data/models/ppo/policy.zip`, trained in Module 13's own
  turn) via `decide_adaptation_strategy()` — never a scripted/hardcoded action; (3) dispatch, in
  the test itself, to WHICHEVER of Modules 14/15/16's real agents PPO actually selected (the test
  does not assume or force a particular branch — all three are wired and handle whichever action
  comes back); (4) a real Module 17 `VerificationAgent.verify()` call recomputing fidelity
  independently and promoting/rejecting the real registry; (5) this module's
  `record_adaptation_event()` and `generate_maintenance_report()`. **Actual observed result on
  this machine (2026-09-07)**: PPO selected **`regenerate`** for a synthetic `throughput` drift
  event at severity 0.65; `RegenerationAgent` produced a real sandboxed candidate
  (`RebuiltThroughput`, sandbox `rmse=1.6435`, 1 attempt, no self-correction needed);
  `VerificationAgent` independently recomputed **fidelity_before=0.9800, fidelity_after=0.9977 ->
  ACCEPT**, genuinely promoting `throughput-v2` to production; the resulting `LifecycleRecord`
  was confirmed, field-by-field, to contain every one of prompt.md §37's required fields with
  correct, non-fabricated values (full JSON dump inspected directly — see the test file for the
  exact assertions), and the generated maintenance report was confirmed to mention the affected
  component, the selected action, and the verification decision. A second run of the same test
  file is expected to potentially select a DIFFERENT action (recalibrate/expand-scope) depending
  on the exact observation PPO is handed — the test's generic three-way dispatch means this is a
  feature (proving genuine, non-scripted integration across all three adaptation agents), not a
  source of flakiness: whichever branch runs, the same completeness assertions apply and pass.
- **Deliberately NOT done this turn, per its own scope**: no code in Modules 11/13/14-17 was
  modified to automatically CALL this agent — `src/main.py` (prompt.md §39's continuous
  orchestration loop, Phase 11, not yet built) is what will eventually wire
  `record_adaptation_event()`/`generate_maintenance_report()` into the live continuous loop after
  every real verification decision; this turn only builds and proves the recording/reporting
  agent itself, exactly as scoped ("Module 19 only").

**Phase 11 — `src/main.py`: the top-level continuous orchestration loop — implemented and run
end-to-end for real.** This is the final piece of the architecture: `ContinuousOrchestrator`
wires every already-built module (1-19, D1, D2) together into one real, continuously-running
system, exactly the canonical loop in §2 above / prompt.md §39. **Integration only** — no new
model, fidelity, verification, or agent logic; this file only constructs already-tested
components and calls their already-tested public methods in the specified order. Key design
decisions:

- **Structural continuous-operation guarantee** (prompt.md §0.6/§0.8, §8's own "continuous
  operation" rule): `ContinuousSynchronizer` runs on its own background daemon thread (started
  once in `start()`), and the ENTIRE drift -> PPO -> agent -> verification -> lifecycle cycle runs
  in the orchestrator's own separate foreground loop (`run()`), started via a second background
  thread for drift consumption plus the main loop's own thread for adaptation dispatch. A slow
  adaptation cycle (LLM call, sandboxed subprocess) can only ever delay the NEXT drift event or
  prediction cycle — it structurally cannot block a telemetry batch sync, which lives on an
  entirely separate thread this loop never touches again after `start()`.
- **Bootstrap models are genuinely loaded-or-trained, not always retrained** (CLAUDE.md §8 "normal
  telemetry processing never auto-retrains"): `_bootstrap_dt_models()` checks `ModelRegistry.
  get_current_version(name)` for each of the five real components; if a production version
  already exists (a resumed process), it's LOADED via `load_artifact_into`; only genuinely missing
  components trigger a real bootstrap telemetry batch (1200 rows, the same real Module 2 pipeline
  every other bootstrap fixture in this repo already uses) and real training. Bootstrap telemetry
  is deliberately NEVER written into the live `D1Store` — it is a distinct provenance category
  from live-synchronized telemetry, per Module 3's own already-established scope boundary; the
  live D1Store starts empty and grows ONLY from genuinely live telemetry once `start()` runs.
- **PPO's runtime observation is NOT built by reusing `AdaptationEnv`** — that class always
  fabricates synthetic prediction fidelity internally for training, which would silently ignore
  the real system's actual fidelity if reused here. `_build_runtime_observation()` instead
  constructs the EXACT SAME vector schema `AdaptationEnv._build_observation()` was trained
  against — [fidelity vector, affected-component one-hot, drift severity, previous-action
  one-hot, previous reward, network-state summary] — but populated from genuinely real sources:
  real per-component `FidelityEvaluator` scores from Modules 5-10's own live predictions against
  real D1 ground truth (`run_prediction_and_fidelity_cycle()`, public — called periodically by
  `run()` and also directly by validation tooling to pre-warm real fidelity history), the real
  drift event's own component/severity, this orchestrator's own tracked previous action/reward
  (computed from the last REAL adaptation cycle's REAL verified fidelity delta, via the exact
  reward formula CLAUDE.md §6 specifies: `fidelity_improvement - adaptation_cost_penalty`), and a
  real D1-history-derived network-state summary (self-referential min-max normalization, the same
  convention `build_network_state_pool` already uses, adapted for a live/streaming window instead
  of a precomputed training pool). A component with no fidelity evaluated yet (cold start)
  defaults to a neutral 0.0 placeholder, never a fabricated score.
- **Dependency-having target components are handled generically at verification time too** — a
  genuine bug found and fixed while validating this module (see below): if the drifted component
  itself has `DEPENDENCIES` (latency/prb_utilization/jitter), the held-out `DataFrame` this
  orchestrator builds for Module 17 verification must ALSO carry those dependencies' ground-truth
  columns (`data_selection.with_dependency_ground_truth`) — needed both for a recalibration
  candidate's own `.predict()` call and for `VerificationAgent`'s internal production-baseline
  `.predict()` call, since both are instances of the same dependency-having class. Missed on the
  first real run (a `jitter` drift event crashed with `JitterModel missing required input
  column(s)`); fixed by applying the same ground-truth-population helper every adaptation agent
  already uses internally, once more here for the orchestrator's OWN externally-built held-out set.
- **LLM availability handled honestly, never faked**: no real `ANTHROPIC_API_KEY` is configured in
  this development environment (only a placeholder in `.env`) — the orchestrator constructs a real
  `AnthropicClient` regardless (a placeholder key is non-empty, so construction succeeds), and any
  ACTUAL API call made through it (a `regenerate`/`expand_scope` agent, or `VerificationAgent`'s
  optional explanation step) genuinely fails at the network/auth layer and gracefully degrades —
  `RegenerationError`/`ExpandScopeError` are caught and logged, the cycle is skipped (telemetry
  unaffected, loop continues to the next drift event), and `VerificationAgent`'s explanation falls
  back to its deterministic template exactly as Module 17 already specifies. Nothing here invents
  a fake LLM response anywhere in `src/main.py` itself.
- **Real validation run performed exactly as this turn required** — `scripts/run_orchestrator_demo.py`
  (permanent, repo-tracked, same convention as `ns3_sim/validate_e2e.py`/`scripts/train_ppo.py`/
  `scripts/ingest_rag.py`) initializes the full real system against this repo's REAL configured
  storage paths (the very first genuine `D1Store`/`ModelRegistry`/lifecycle-records state this
  project has ever produced — `data/artifacts/d1_*.parquet`, `data/models/{throughput,latency,
  packet_loss,prb_utilization,jitter}/`, `data/models/registry_index.json`,
  `data/artifacts/lifecycle_records.jsonl`, `data/artifacts/maintenance_reports/*.md` did not
  exist before this run), narrows the mock drift severity range to bias toward `recalibrate`
  (real Module 13 held-out evidence: low severity reliably selects it — this is the ONLY
  intentional bias in the run; PPO still genuinely decides, and telemetry/D1/prediction/fidelity
  are all completely real and unmodified) so the run completes with zero fakes despite this
  environment's missing API key, and runs it unattended through one complete real cycle. **Actual
  observed result on this machine (2026-09-07)**: a real drift event on `jitter` at severity
  0.1846 -> PPO genuinely selected `recalibrate` -> a real candidate `jitter-v2` was produced and
  independently re-verified by Module 17: **fidelity_before=1.8686, fidelity_after=0.9853 ->
  REJECT** (the candidate was genuinely worse — production correctly, automatically preserved,
  `jitter-v1` untouched) -> a complete Module 19 lifecycle record and human-readable maintenance
  report were generated. **The concrete continuous-operation proof**: a real sampler thread
  observed `ContinuousSynchronizer.records_synced` grow from **560 to 680 records** strictly
  WITHIN the adaptation cycle's own 2.438-second wall-clock window (148 samples taken during that
  window, 10ms apart) — telemetry ingestion never paused for the adaptation cycle, confirmed with
  real numbers, not asserted in the abstract. A REJECT outcome is treated as an equally valid,
  equally complete demonstration of "one full cycle" as an ACCEPT would have been — Module 17
  correctly protecting production from a worse candidate is exactly the deterministic gate
  working as designed, not a failure of this validation.
- **Tests**: `tests/unit/test_main.py` (8 — the component-spec dispatch tables are internally
  consistent with each other and with `config.drift.valid_components`; the runtime observation
  vector has the exact dimension PPO was trained on and stays within its clipped bounds; the
  affected-component and previous-action one-hots are placed correctly; a cold-start component
  with no fidelity yet defaults to a finite neutral value, never NaN/crash; the network-state
  summary is zeros for empty history and a bounded, self-normalized vector for real data; bootstrap
  dispatch genuinely LOADS every component when production versions already exist, without ever
  generating bootstrap telemetry) and `tests/integration/test_main_orchestrator.py` (1 — a
  smaller/faster but still fully real re-run of the exact same proof `scripts/run_
  orchestrator_demo.py` performs at full scale, against temp storage paths, as an automated
  regression test).

See `IMPLEMENTATION_STATUS.md` for the full module-by-module table and the exact next task.
