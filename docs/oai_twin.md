# The OAI Digital Twin — Components, Dependency Graph, and Replay Design

This document describes the post-migration Digital Twin (prompt2.md: "Migrate the Digital Twin
from NS-3 to OAI KPI replay"). It replaces the retired NS-3/5G-LENA simulated-telemetry design
with a replay of a real exported OpenAirInterface (OAI) RAN KPI dataset — genuine
testbed-measured O-RAN/OAI RAN-layer KPIs, not simulation output.

## 1. Data source

Read-only, outside this repository: `config.data.oai_root`
(`/home/harie06/oai_db/data/rankpi_kpi/` in this deployment). There is no database access
anywhere in this system — only the already-exported flat CSV files under this path. The CSVs are
never copied into git and never modified by this project.

Config-driven splits (`config.data.splits`), by run id only, disjoint by construction
(`DataSplitsConfig` fails to load if any run id appears in two splits):

| Split | Run id(s) |
|---|---|
| train/tune | 3, 6, 7 |
| test | 8 |

> **Revised in claude_code_prompt_phase13.md.** The original Phase 12 design (train=[3,6],
> val=[7], test=[8]) is superseded — there is no separate "val" split anymore. All of runs 3/6/7
> are now the TRAIN/TUNE pool (leave-one-run-out cross-validation over these 3 runs, per
> `scripts/bootstrap_oai_models.py`), and run 8 remains the TEST split, reserved exclusively for
> the `baselines/` comparison — never used for training, tuning, debugging, or smoke tests (use
> run 7, a member of the train/tune pool, for smoke tests instead). See `docs/MODELS.md` for the
> current, authoritative per-component CV results; the "Bootstrap VAL metrics" paragraph later in
> this file describes the now-superseded Phase 12 (non-residual, 33-feature, train=[3,6]/val=[7])
> design and is kept only for historical comparison.

Run 10 was the originally-intended test split, but its CSV export is genuinely incomplete (9/24
tables, all left as `.csv.partial`, export stalled with no process running and no database access
to re-export it) and was excluded entirely per explicit confirmation, never substituted silently.
The splits are config-driven specifically so run 10 (or any future run) can be added later with
zero code changes once/if it is ever re-exported successfully.

## 2. KPI catalog (ground truth, verified against the real export)

Source: this project's own direct inspection of the real OAI export's dimension tables
(`dim_kpm_meas_name.csv`, `dim_control_param.csv`, `dim_slice.csv`, `dim_cell.csv`) and the real
CSV headers of `mac_ue_1s`, `rlc_bearer_1s`, `pdcp_bearer_1s`, and `v_ue_kpm_1s` — not re-derived
from prose documentation, and more authoritative than it: these are the literal column
definitions and measured values the export actually contains.

| KPI | Unit (per `dim_kpm_meas_name`) | Level | Cumulative? | Notes |
|---|---|---|---|---|
| `DRB.UEThpDl` | **kbps** | ue | no | OAI computes it over the granularity period. Converted to Mbps (÷1000) for this project's `dl_throughput` target — confirmed empirically: raw values up to ~6390 only make sense as kbps (6.39 Mbps), not Mbps. |
| `DRB.UEThpUl` | kbps | ue | no | → `ul_throughput`, same conversion. |
| `DRB.PdcpSduVolumeDL` / `UL` | kb | ue | yes | TRUE payload, unlike `mac_ue.dl_aggr_tbs` which includes padding. Not one of the 7 core targets, but a real candidate expand-scope feature (PDCP volume rate). |
| `DRB.RlcSduDelayDl` | ms | ue | no | **Not actually present as a column in this export's `v_ue_kpm_1s` view**, despite being defined in the catalog — verified directly against the real CSV header. `dl_rlc_delay` is therefore sourced ENTIRELY from `rlc_bearer_1s.tx_sdu_avg_time_to_tx_avg`, which is itself in **microseconds** (verified by magnitude: mean 483, max 248446 — plausible only as µs, not ms) and converted (÷1000). |
| `RRU.PrbTotDl` / `Ul` | prb (raw count) | cell | yes (per the catalog's own collection semantics) | OAI reports a raw PRB count where TS 28.552 defines a percentage. The per-second `v_ue_kpm_1s` column is already a usable per-second value — no further conversion needed for `dl_prb`/`ul_prb`. |

Controller parameters (`dim_control_param`): the only parameter genuinely implemented end-to-end
in this OAI deployment is `dedicated_prb_ratio` (scope=slice, unit=percent) — reaches the
scheduler as `pct_reserved`, rescaled by `rc_ctrl_service_style_2.c` (`pct/100 *
(1 - default_slice_pct)`, so a requested 40% does not literally mean 40%). `min_prb_ratio` and
`max_prb_ratio` are decoded but never actually applied by the scheduler in this OAI build.
`qos_flow_mapping` is the only OTHER control stock OAI implements. This is why the replacement
feature set's "requested PRB ratio" input is sourced specifically from `control_actions`'s
`dedicated_prb_ratio` rows, mapped via `dim_slice` (slice_id 1 = `slice-a`, slice_id 5 =
`slice-b`).

## 3. The 7 prediction components and dependency graph

Replace the 5 retired NS-3-era components (`throughput`/`latency`/`packet_loss`/
`prb_utilization`/`jitter`) entirely:

```
dl_bler, ul_bler, dl_prb, ul_prb      (roots — lagged features only)
dl_throughput  <- dl_bler, dl_prb      (uses the PREDICTED values for second t)
ul_throughput  <- ul_bler, ul_prb
dl_rlc_delay   <- dl_throughput, dl_bler
```

| Component id | Target | Unit | Primary source | Fallback |
|---|---|---|---|---|
| `dl_throughput` | DRB.UEThpDl | Mbps | `v_ue_kpm_1s` (kbps÷1000) | `8·Δdl_aggr_tbs/Δt/1e6` from `mac_ue_1s` |
| `ul_throughput` | DRB.UEThpUl | Mbps | `v_ue_kpm_1s` (kbps÷1000) | `8·Δul_aggr_tbs/Δt/1e6` |
| `dl_rlc_delay` | DRB.RlcSduDelayDl | ms | `rlc_bearer_1s.tx_sdu_avg_time_to_tx_avg` (µs÷1000) | none — no secondary source exists in this export |
| `dl_prb` | RRU.PrbTotDl per UE | PRBs/s | `v_ue_kpm_1s` | `Δdl_aggr_prb/Δt` from `mac_ue_1s` |
| `ul_prb` | RRU.PrbTotUl per UE | PRBs/s | `v_ue_kpm_1s` | `Δul_aggr_prb/Δt` |
| `dl_bler` | DL block error rate | 0-1 | `mac_ue_1s.dl_bler_avg` | none |
| `ul_bler` | UL block error rate | 0-1 | `mac_ue_1s.ul_bler_avg` | none |

Every component's `REQUIRED_FEATURES` is the full 33-column lagged feature vector
(`src/data/oai_dataset.ALL_FEATURES`): the 10 MAC features (`dl_curr_tbs`, `dl_sched_rb`,
`pusch_snr`, `pucch_snr`, `wb_cqi`, `dl_mcs1`, `ul_mcs1`, `phr`, `dl_bler`, `ul_bler`) as both
`_avg` and `_std`, `bsr_avg`, `n_samples`, per-cell/per-slice UE counts, the controller's
requested PRB ratio + `has_controller` flag, and the lagged value of all 7 targets themselves —
all shifted by exactly one second (`groupby(["run_id","session_id"]).shift(1)`), so no same-second
measured value is ever a model input (the leakage rule, enforced by a dedicated test).

Bootstrap VAL metrics (train=[3,6], val=[7], regularized config — 150 trees/depth 4/subsample
0.8, found necessary after the original 300-tree/depth-6 default was genuinely WORSE than a
trivial previous-second persistence baseline on 5 of 7 components): `ul_bler` and `dl_rlc_delay`
beat the persistence baseline when evaluated over the FULL val set (including the ~3.5% of rows
with no controller deployment yet active, which the other 5 components' predictions are
measurably worse on) — see `IMPLEMENTATION_STATUS.md` for the exact numbers and the honest
discussion of this real, non-fabricated result.

## 4. Replay design

`src/telemetry/oai_replay_source.py`'s `OaiReplaySource` implements the SAME `TelemetrySource`
interface the retired ZeroMQ source used (`records()`/`close()`), so Modules 2-4 needed no
redesign. It replays `src/data/oai_dataset.py`'s already-processed, already-lagged, already
unit-converted per-run tables (`data/processed/oai/{train,val,test}.parquet`) — never the raw
CSVs directly.

- **Causality**: a processed-table row at `ts_1s` already carries exactly what causality
  requires — its `_lag1` features reflect only state strictly before this second, and its raw
  target columns are the measured ground truth FOR this second. Replaying a row verbatim as one
  tick automatically satisfies "predict second t using state up to t-1; compare against t's own
  ground truth."
- **Determinism**: the processed table is re-sorted by `(ts_1s, session_id)` before replay, so
  the same input always yields the same tick sequence regardless of on-disk row order.
- **Pacing**: `config.telemetry.replay.speed` — `0` = as fast as possible (demo/test default),
  `1.0` = real time, `>1` = accelerated.
- **Gaps**: a second with no data in the source table simply has no tick — the replay never
  invents a row to fill a gap; a real gap in the underlying data is logged, not repaired.
- **Control surface**: `start(run_ids, start_ts=None)`, `pause()`, `resume()`, `stop()`, plus a
  deterministic `seek(timestamp)` that lands on the first REAL tick at-or-after the target,
  never a synthetic one.

Mock telemetry (`src/telemetry/mock_source.py`) stays for unit tests only, generating the same
OAI-shaped record fields (identity + the 33 lagged features + the 7 targets, with plausible
physically-correlated values), always tagged `SOURCE=MOCK` — never presented as, or confusable
with, real replayed data (`SOURCE=OAI_RAN_REPLAY`).
