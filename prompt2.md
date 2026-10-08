# CLAUDE CODE TASK — Migrate the Digital Twin from NS-3 to OAI KPI replay

You are working in the EXISTING repository `ai-self-adaptive-digital-twin`. This is a migration, not a rebuild. Keep every module that is not listed below as changing.

Before editing anything, read:
- `CLAUDE.md`
- `IMPLEMENTATION_STATUS.md`
- `docs/dbDetails.pdf` (the OAI schema)
- the export's `_meta/` folder (manifest, KPI catalog) and `approved_overrides.json`

## Goals

1. **Remove NS-3 completely from the runtime.** No module, config, test or demo may depend on NS-3, 5G-LENA or the NS-3 ZeroMQ bridge.
2. **Replace the network source with an OAI CSV replay source.** It streams the exported 1-second OAI KPI rows into the twin, one second at a time, in timestamp order, so continuous synchronization works exactly as it did with NS-3.
3. **Replace the 5 old prediction components** (latency, jitter, throughput, PRB utilization, packet loss) with the OAI KPI components below.
4. **Keep the full lifecycle working end-to-end** on OAI data: fidelity, triggers, decision agent, adaptation agents, sandbox, verification, registry and lifecycle records.

## Fixed constraints

- **Data location (read-only, outside the repo):** `/home/harie06/oai_db/data/rankpi_kpi/`.
  - Put the path in config as `data.oai_root`. Never copy the CSVs into git, and never modify files there.
  - There is no database access. Do not add any PostgreSQL code.
- **Splits:** train = runs 3, 6, 7; validation = run 8; test = run 10. Splits are by run only, and no run may appear in two splits.
  - First verify that every needed file exists for every run, e.g. `mac_ue_1s_run10.csv`.
  - If anything for run 10 is missing, STOP and report exactly which files are missing. Do not substitute another run without my approval.
- **Fidelity:** keep the ORIGINAL fidelity formula unchanged (RMSE, MAE, Wasserstein, MK-MMD; squared; rolling min-max; `F_c = 1 − S/4`; weighted unified score). Only the set of components changes.
- **Source label:** every record is tagged `SOURCE=OAI_RAN_REPLAY`.
  - The mock source stays for unit tests only, tagged `SOURCE=MOCK`.
  - Never present replayed data as live.
- **No fabricated results:** never invent metrics or fidelity numbers. Report only what actually ran.

## Step 1 — Inventory and NS-3 removal plan, then STOP

1. Grep the repo for `ns3`, `ns-3`, `NS3`, `lena`, `5G-LENA`, `zmq`, `zeromq` and `NS3_5G_LENA`. List every file and what depends on it.
2. List the current prediction components, their files, the dependency graph, and every place the five old component names are hard-coded: config weights, tests, RAG docs, the decision agent context, reports and the dashboard.
3. Inspect the export:
   - list the files per run and the row counts from the manifest
   - show the header of `mac_ue_1s`, `rlc_bearer_1s`, `pdcp_bearer_1s` and `v_ue_kpm_1s`, and the slice/cell files
   - check whether `ue_kpi_1s` / `slice_kpi_1s` wide tables already exist
4. Show me the plan.

**STOP and wait for my confirmation.**

## Step 2 — Remove NS-3

- Move `ns3_sim/` and the ZeroMQ telemetry source to `archive/ns3/`, with a README saying they are retired and not used.
- Exclude `archive/` from imports, tests and packaging. Add a test that fails if any module under `src/` imports from `archive/`, or imports `zmq`.
- Remove NS-3 settings from `config/settings.yaml`, NS-3 steps from the demo scripts, and NS-3 dependencies from `requirements.txt` (keep `pyzmq` only if something else needs it).
- Update `CLAUDE.md`, the README and `IMPLEMENTATION_STATUS.md`.
- After this step, `grep -ri "ns3\|ns-3\|lena\|zmq" src/ config/ scripts/ tests/` must return nothing except the archive-import guard test.

## Step 3 — Restructure the data (`src/data/oai_dataset.py`)

Build model-ready tables from the CSVs, reusing the existing wide tables if they exist and are valid.

**Loading:**
- Parse every timestamp with `pd.to_datetime(..., utc=True, format="ISO8601")`.
- Use the manifest's column types.

**Unit of analysis:** one row per `(run_id, session_id, ts_1s)` at UE level. Drop rows with empty `session_id` from UE-level tables; keep them in slice/cell totals.

**Deltas:**
- Within each `(run_id, session_id)`, sorted by time, compute `diff()` of the `_last` cumulative counters.
- A negative delta is a counter reset: set it to NaN and flag it.
- Gaps larger than 2 s are NaN, never spread across the gap.
- Δt comes from consecutive `ts_last` values.

**Targets (exactly these seven, one per component):**

| Component id | Target | Unit | Source |
|---|---|---|---|
| `dl_throughput` | DRB.UEThpDl | Mbps | `v_ue_kpm_1s` (fallback: `8·Δdl_aggr_tbs/Δt/1e6` from `mac_ue_1s`) |
| `ul_throughput` | DRB.UEThpUl | Mbps | `v_ue_kpm_1s` (fallback: `ul_aggr_tbs`) |
| `dl_rlc_delay` | DRB.RlcSduDelayDl | ms | `v_ue_kpm_1s`, cross-checked against `rlc_bearer_1s.tx_sdu_avg_time_to_tx_avg` |
| `dl_prb` | RRU.PrbTotDl per UE | PRBs/s | `v_ue_kpm_1s` (fallback: `Δdl_aggr_prb/Δt`) |
| `ul_prb` | RRU.PrbTotUl per UE | PRBs/s | `v_ue_kpm_1s` (fallback: `Δul_aggr_prb/Δt`) |
| `dl_bler` | DL block error rate | 0–1 | `mac_ue_1s.dl_bler_avg` |
| `ul_bler` | UL block error rate | 0–1 | `mac_ue_1s.ul_bler_avg` |

Before using any target:
- Read the actual `v_ue_kpm_1s` column names and map them explicitly.
- Check the delay unit: compare magnitudes with the RLC table and report whether values look like ms or µs. Convert to ms and document the conversion.
- Report any target that is constant, mostly NaN, or missing in a split.

**Input features:** all values are LAGGED by one second, using `groupby(["run_id","session_id"]).shift(1)`.
- These 10 MAC features, as `_avg` and also `_std`: `dl_curr_tbs`, `dl_sched_rb`, `pusch_snr`, `pucch_snr`, `wb_cqi`, `dl_mcs1`, `ul_mcs1`, `phr`, `dl_bler`, `ul_bler`
- `bsr_avg`
- lagged values of all seven targets
- UE count per cell and slice for the previous second
- slice id, as a categorical
- the requested PRB ratio for the UE's slice from `control_actions`, carried forward as a step function. NaN where there is no deployment (e.g. run 10), plus a `has_controller` flag.
- `n_samples` from the previous second

**Leakage rule:** no same-second measured value may be an input for any target. Add a test that checks every feature column is a shifted column or a static/categorical one.

**Output:** `data/processed/oai/{train,val,test}.parquet`, outside git or git-ignored, plus `feature_schema.json` (feature list, target list, units, lag, source columns) and a data report (rows per split, NaN rates, resets found, unit conversions).

## Step 4 — Replay source and continuous synchronization (`src/telemetry/oai_replay_source.py`)

The source implements the SAME telemetry-source interface the ZeroMQ source used, so modules M2–M4 need no redesign.

**What it emits:**
- It replays one or more runs from the processed tables, or the raw 1-s CSVs where that is cleaner.
- Rows are grouped by `ts_1s`. Each tick emits every UE/slice/cell row for that second, in strict timestamp order.

**Pacing:** set by `replay.speed` in config.
- `1.0` = real time, one tick per wall-clock second.
- `>1` = accelerated.
- `0` = as fast as possible, for experiments and tests.

**Gaps:** seconds with no data emit no tick, and a gap event is logged. The replay never invents rows.

**Interface:**
- `start(run_ids, start_ts=None)`, `pause()`, `resume()`, `stop()`, plus a deterministic seek to a timestamp.
- The same input always yields the same sequence of ticks.

**Causality:** at tick `t`, the twin predicts second `t` using state up to `t−1`. The ground truth for `t` is the target values that arrive in tick `t`, and that is what the fidelity engine compares against.
- The synchronizer writes each tick to D1 immediately, including current state, history, predictions and ground truth.
- Telemetry ingestion and production inference keep running during adaptation, as before.

## Step 5 — New prediction components and dependency graph

Each component uses the existing component interface (`train/predict/evaluate/save/load`), the default model class (XGBoost, or the repo's existing default), and declares its inputs, outputs and dependencies as metadata.

Dependency graph (it must be acyclic and validated by the orchestrator):
```
dl_bler, ul_bler, dl_prb, ul_prb      (roots — lagged features only)
dl_throughput  ← dl_bler, dl_prb      (uses the PREDICTED values for second t)
ul_throughput  ← ul_bler, ul_prb
dl_rlc_delay   ← dl_throughput, dl_bler
```

Requirements:
- Predicted upstream values are inputs; measured same-second values never are.
- Remove the jitter component entirely. Packet loss is replaced by `dl_bler`/`ul_bler`.
- Update the unified-score weights in config to the seven components. Use equal weights unless I say otherwise.
- Bootstrap: train each component on TRAIN, tune only on VAL, and never touch TEST during training.
- Save the versions to the registry as `v1 (bootstrap, OAI)`.
- Report bootstrap metrics on VAL per component, against a persistence baseline (previous-second value). Do not report TEST metrics in this step.

## Step 6 — Lifecycle components that must follow the change

- **Fidelity engine and trigger:** only the component list and weights change. The formula is unchanged.
- **Decision and RCA agent:** update the context schema and prompts to the new component names and their meaning (radio-level BLER, not end-to-end loss; RAN queueing delay, not end-to-end latency).
- **Regeneration and Expand-Scope agents:** update the LLM context to the new feature schema (`feature_schema.json`). Expand-Scope candidates to mention in the policy docs:
  - `dl_mcs`
  - HARQ failure ratio
  - slice throughput
  - achieved PRB share vs. requested (`v_slice_prb_share` + `control_actions`)
  - slice fairness (`jain_dl`)
  - PDCP volume rate

  All of these exist in the export.
- **RAG knowledge base:**
  - Remove the NS-3/5G-LENA documents from the digital-twin corpus.
  - Ingest `docs/dbDetails.pdf`, the KPI catalog and a new `docs/oai_twin.md` describing the components, the dependency graph and the replay design.
  - Mark NS-3-era adaptation history records with metadata `source=ns3_legacy`, so retrieval labels them as not comparable. Do not delete them.
  - Bump document versions and re-ingest (idempotent).
- **Baselines:** make sure these experiment modes run on OAI replay:
  - `static` (no adaptation)
  - `periodic_retrain` (fixed schedule, config-driven)
  - `rule_based` decision (severity thresholds from config)
  - the full system

  All modes use the same splits, seeds, replay order and fidelity settings.
- **Lifecycle records and reports:** record `source=OAI_RAN_REPLAY`, the run ids replayed, and the replay speed.

## Step 7 — Tests (must actually run)

- the import guard (no `archive/`, no `zmq` imports under `src/`)
- dataset:
  - deltas are correct on a hand-made example
  - resets → NaN
  - gaps > 2 s → NaN
  - the leakage test (all features lagged or static)
  - splits are disjoint
  - UTC timestamps
- replay:
  - strict timestamp order
  - one tick per second present
  - no invented rows at gaps
  - deterministic seek
  - pause/resume
  - `speed=0` finishes quickly
- orchestrator: the new graph is valid; an artificially injected cycle is rejected
- components: train, predict, save, load round-trip on a small fixture
- an end-to-end smoke run: replay part of run 8 at `speed=0` → predictions → fidelity → at least one forced trigger → decision → adaptation in the sandbox → verification → lifecycle record
- the full existing test suite, with NS-3 tests removed or replaced

## Final report

```
NS-3 REMOVAL (files archived, grep result)
DATA (rows per split, targets available, unit conversions, NaN/reset report)
COMPONENTS + GRAPH
BOOTSTRAP VAL METRICS vs PERSISTENCE (per component)
REPLAY / SYNC CHECK
LIFECYCLE SMOKE RUN RESULT
TESTS RUN (command, pass/fail counts)
OPEN ISSUES
```

ADDENDUM — project-wide consistency (mandatory):
This migration applies to the ENTIRE repository. No module, config, schema, test, script, dashboard, report template, RAG document, or registry entry may keep NS-3 dependencies or the old component set (latency, jitter, throughput, prb_utilization, packet_loss) as active components.

Additionally update:
1. Dashboard and report templates: panels, labels, units and plots for the 7 OAI components; remove jitter/packet-loss panels.
2. D1 state store schema: current/historical state, predictions and ground-truth tables use the new component ids. Provide a migration for existing local data, or start a fresh store and archive the old one; ask me which.
3. Model registry: mark all NS-3-trained versions as `retired (ns3_legacy)`. Never load them as production. Register the OAI bootstrap versions as current.
4. Drift-trigger contract and canonical trigger schema: allowed component ids = the 7 new ids; validation rejects old ids.
5. Sandbox fixtures, demo scripts, evaluation/baseline scripts and test fixtures: use the new ids and OAI replay.
6. Config validation: fail at startup if any old component id or NS-3 setting appears.

Final consistency check (report the output):
  grep -rniE "ns3|ns-3|lena|zmq|jitter|packet_loss|NS3_5G_LENA" src/ config/ scripts/ tests/ dashboard/ rag_data/ docs/
Allowed matches only: archive/, the import-guard test, and the RAG/legacy labels marking NS-3 history as not comparable. List every remaining match and explain it, or fix it.
