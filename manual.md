# MANUAL.md — AI-Driven Self-Adaptive Network Digital Twin

The run-and-reference manual for this repository: how to set it up, how to run the live system,
what you will see while it runs, how to test it, and what every folder/file does.

- **Why** things are built the way they are: `CLAUDE.md` (the authoritative architecture/design
  document — §5 defines the per-component adaptation policy; §12 records every phase).
- **Module-by-module status** and open items: `IMPLEMENTATION_STATUS.md`.
- **What changed and why**, newest first: `CHANGELOG.md`.

**At a glance (current behaviour, 2026-10-20):**
- The live system replays real OAI RAN KPI data (run 8) and keeps 7 component models in sync with it.
- Each component is adapted **independently**, only when its **own** fidelity score `F_c` stays
  below its threshold. The unified score `F_U` is reported, never used to trigger.
- **External drift is permanently disabled** — the internal per-component fidelity policy is the
  only trigger source.
- A rejected adaptation is **retried in a loop** — with the rejection details fed back to the
  agents — until a candidate is accepted, the component recovers, or 5 attempts were made.
- Adaptations run on **worker threads** (up to 2 components at once, never the same component
  twice); fidelity cycles and telemetry ingestion never wait for them.
- Recalibrate, regenerate and expand-scope all work on the OAI data; expand-scope can add a new
  component predicting one of 8 observable MAC KPIs (e.g. `wb_cqi_avg`).
- An accepted model is **hot-swapped** into live serving immediately (no restart) and survives
  restarts; components downstream of it are re-checked.
- `docs/MANUAL.md` and `docs/HOW_TO_RUN.md` are older, narrower manuals; where they conflict with
  this file, this file wins.

---

## 1. Environment

Python ≥ 3.12 on Linux or macOS, or on Windows inside **WSL** (Ubuntu). It was developed and
tested with Python 3.14 under WSL. On Windows, keep the virtual environment on the Linux
filesystem (e.g. `~/.venvs/dt`), not under `/mnt/c/...`: installing and importing packages from
`/mnt/c` is many times slower.

Start every session in the repository root with the environment active:

```bash
cd digital_twin-main            # wherever you cloned it (quote the path if it contains spaces)
source .venv/bin/activate       # or: source ~/.venvs/dt/bin/activate
```

All commands below assume this.

---

## 2. One-time setup

### 2.1 Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt   # exact tested versions (requirements.txt = minimums)
python scripts/setup.py        # verifies dirs/.env/dependencies/config — non-destructive, safe to re-run
```

### 2.2 Secrets

Copy `.env.example` to `.env` and set a real `GOOGLE_API_KEY` (or `GEMINI_API_KEY`;
`GOOGLE_API_KEY` wins if both are set). Never commit `.env`. Without a key the system still runs:
the Decision Agent falls back to its configured deterministic strategy (`recalibrate`, logged at
WARNING) and regenerate/expand-scope cycles are skipped.

### 2.3 The OAI dataset

The processed dataset ships with the repository in `data/processed/oai/`: train = runs 3, 6, 7;
test = run 8. Run 10's export is incomplete and is excluded on purpose. **Nothing needs
downloading.**

To rebuild it from a raw OAI CSV export, set `OAI_ROOT` in `.env`. On WSL use the `/mnt/c/...`
form, never Windows backslashes:

```
OAI_ROOT=/path/to/rankpi_kpi
```

### 2.4 Build the RAG knowledge base (and optionally rebuild the dataset)
```bash
python scripts/build_oai_dataset.py   # OPTIONAL, needs OAI_ROOT; ~10 s; rewrites data/processed/oai/{train,test}.parquet,
                                      # feature_schema.json, data_report.md
python scripts/ingest_rag.py          # first run downloads the embedding model; writes rag_data/chroma/
```

Expected output (verified 2026-10-19): `train: 60858 rows, 21 sessions`, `test: 25865 rows, 7
sessions`, 0% NaN on all 7 targets and on the 8 observable MAC KPI columns (`feature_schema.json` lists them under `observables`); RAG `chunks upserted: 47` across 4 categories. Both scripts
are safe to re-run.

**Optional** — `python scripts/bootstrap_oai_models.py` runs leave-one-run-out CV tuning and
registers tuned bootstrap models (slow; writes `reports/model_eval/`). If you skip it, the
orchestrator bootstrap-trains the 7 models itself on first start (~20 s).

> **Warning:** `bootstrap_oai_models.py` **promotes its new models over whatever is currently in
> production** — including models the live system adapted and hot-swapped in. Run it only when
> you deliberately want to reset all 7 components to freshly tuned bootstrap models. (The previous
> versions are kept as `superseded` and can be rolled back.)

---

## 3. Running the live system

```bash
python -m src.main --mode live --replay-speed 10
```

`--mode live` replays the real OAI **TEST split (run 8)** through the full pipeline:
replay → preprocessing → continuous sync into D1 → 7-component dependency-aware prediction →
per-component fidelity → **per-component adaptation policy** → Decision & Root-Cause Analysis
Agent (Gemini + RAG) → recalibrate / regenerate / expand-scope agent → sandbox → deterministic
verification gate → promote/reject + live hot-swap → post-adaptation re-check + dependency
revalidation → lifecycle record + maintenance report. It runs until you stop it with **Ctrl-C**,
which shuts down cleanly (`[shutdown] background sync thread confirmed exited, D1 flushed to disk`).

**Adaptation policy (per component).** Every component has its own fidelity score `F_c` and its
own policy. Component c is adapted when its own `F_c` is below its threshold for N consecutive
fidelity cycles (defaults: threshold 0.5, N = 3; overridable per component — §7). Each component
is decided independently: several can be adapted in the same period, a healthy component is
never touched because another degraded, and no system-wide score has to drop first. A cycle with
no usable measurement for a component (`warming` / `no data`) never triggers it. The unified
score `F_U` is still printed as a system-health summary but never triggers anything. After an
accepted model swap, every component downstream of it (e.g. `dl_bler` → `dl_throughput` →
`dl_rlc_delay`) is re-measured and reported — never auto-adapted.

**Adaptation loop (retry until fixed).** One trigger starts a loop for that component:
decision (Gemini) → agent → sandbox → verification. If the candidate is REJECTed (or the agent or
sandbox fails), the component is re-measured and, while it is still below its threshold, the loop
runs again — and the next decision and the regenerate/expand-scope agents are told exactly why
every earlier attempt failed (strategy, candidate vs. production fidelity, failed checks, error).
The loop stops when:

| Stop reason | Console |
|---|---|
| A candidate is ACCEPTed (and hot-swapped) | `[loop] X: candidate ACCEPTED on attempt k/5` |
| The component meets its threshold again | `[loop] X: now meets its threshold ... no further attempts needed` |
| Retrying can't help right now (not enough data, no LLM key, nothing new to model) | the `[warning]` line of that attempt |
| `adaptation.max_adaptation_attempts` (5) attempts made | `[warning] X: no accepted candidate after 5 attempt(s) — production kept` |
| Ctrl-C | `[skipped] X: shutting down — adaptation loop stopped` |

If production is kept after 5 attempts, the component stays flagged and its policy fires again
after N more low cycles. A trigger that waited while its component was busy is dropped if the
component has meanwhile recovered (`[skipped] X: trigger from HH:MM:SS is stale`).

**Multithreading.** Each adaptation loop runs on its own worker thread (`dt-adapt-<component>`).
Up to `adaptation.max_concurrent_adaptations` (2) components adapt at the same time; a component
is never adapted by two threads at once (a second trigger for it waits, merged with any other
pending one). Telemetry sync and the fidelity cycles keep running throughout, so cycle tables
appear between the steps of an adaptation. Live predictions and hot swaps are serialized, so a
swap never happens in the middle of a prediction.

### 3.1 Options

| Flag | Effect |
|---|---|
| `--mode live` | Real OAI replay (`SOURCE=OAI_RAN_REPLAY`). `--mode demo` uses mock telemetry instead (still bootstraps from the real TRAIN split). Omit to use `config/settings.yaml` as-is. |
| `--replay-speed N` | Pace of the replay: `1` = real time (run 8 is ~62 min), `10` = 10× (~6–8 min). `0` / omitted = config default (as fast as possible: the whole run loads in seconds and the system then idles). |
| `--max-adaptations N` | Stop the orchestration loop after N adaptation cycles (demos/validation only). `--max-drift-events` is a deprecated alias. |
| `--prediction-interval-seconds S` | Seconds between prediction+fidelity cycles (default 5). |
| `--verbose` | Print full JSON logs to the console too (they always go to `logs/digital_twin.log`). |

Recommended run:

```bash
python -m src.main --mode live --replay-speed 10
```

**Trigger source.** The live system adapts **only** on its internal per-component fidelity
policy. External drift events are permanently disabled: no drift source or consumer is built and
there is no flag to turn one on. (The Module 11 drift interface still exists for the separate
`baselines/` experiments.)

### 3.2 What you will see

At startup the banner lists every component with its production model version and its own
threshold / consecutive-cycle policy, states the trigger source (`internal per-component fidelity
only (external drift permanently disabled)`), and prints the path of every log file (§3.5). Then,
every prediction cycle (default 5 s), one table — real output from a live run:

```
[cycle 18] window 2026-09-28 15:50:53+00:00 .. 2026-09-28 15:51:21+00:00 (200 rows) | D1 history 5,480 rows | ingest 63.6 rows/s
   component        model version                  F_c  thresh      gap  below  state
   dl_throughput    dl_throughput-v3            0.9725    0.50   0.0000    0/3  ok
   ul_throughput    ul_throughput-v4            0.9662    0.50   0.0000    0/3  ok
   dl_rlc_delay     dl_rlc_delay-v6             0.3846    0.50   0.1154    1/3  BELOW
   dl_prb           dl_prb-v5                   0.8357    0.50   0.0000    0/3  ok
   ul_prb           ul_prb-v5                   0.9857    0.50   0.0000    0/3  ok
   dl_bler          dl_bler-v6                  0.5161    0.50   0.0000    0/3  ok
   ul_bler          ul_bler-v4                  0.3616    0.50   0.1384    3/3  ADAPT -> trigger severity=0.277
   F_U (report only, not an adaptation trigger) = 0.7175 | triggers this cycle: ul_bler
```

`below` is the component's own debounce counter (`n/N`); `state` is `ok`, `BELOW` (counting),
`ADAPT ->` (its policy fired this cycle), `warming` (fidelity history not yet long enough) or
`no data` (no scorable rows this cycle). Note `F_U = 0.7175` above: the system as a whole looks
healthy, yet `ul_bler` is adapted on its own score, while `dl_rlc_delay` independently counts its
first low cycle. An adaptation then prints one line per step. Real output (abridged) of an
adaptation loop from the 2026-10-20 live run — attempt 1 is rejected, the decision agent reads
why and switches strategy for attempt 2:

```
[trigger] component=dl_prb type=fidelity_degradation severity=0.134 F_c=0.4329 threshold=0.50 production=dl_prb-v10 | still pending: 2
[policy] dl_prb: F_c 0.4329 < threshold 0.5 for 3 consecutive cycle(s) (policy requires 3)
[decision] component=dl_prb strategy=recalibrate confidence=0.88 (LLM)
           root cause: The fidelity of dl_prb dropped to 0.4329, breaching the component threshold of 0.5. ...
[agent] recalibrate running for dl_prb only (production dl_prb-v10, training window 24h)
[agent] candidate dl_prb-v13 (DLPRBComponent) produced — recent train rows 1568 + 60858 reference rows, held-out rows 392
[verify] dl_prb dl_prb-v13 REJECT: fidelity 0.5512 -> 0.1226 (required improvement > 0.01); checks: ... fidelity_improved=FAIL
[post-check] dl_prb F_c=0.3766 vs threshold 0.5 -> STILL BELOW threshold (gap 0.1234) — remains flagged; ...
[retry] dl_prb: attempt 1/5 rejected — trying again with this feedback: attempt 1: recalibrate candidate dl_prb-v13 REJECT —
        candidate fidelity 0.1226 vs production 0.5512 (needs > production + 0.01); failed checks: fidelity_improved
[attempt] dl_prb: attempt 2/5 (production dl_prb-v10)
[decision] component=dl_prb strategy=regenerate confidence=0.95 (LLM)
           root cause: The previous attempt to recalibrate the dl_prb model on recent telemetry resulted in severe fidelity
           degradation (dropping from 0.5512 to 0.1226) ... adjusting parameters alone is insufficient ...
[agent] regenerate running for dl_prb only (production dl_prb-v10, training window 48h)
[agent] candidate dl_prb-v14 (DLPRBComponent) produced — recent train rows 63866, held-out rows 752, sandbox attempts 1
[verify] dl_prb dl_prb-v14 REJECT: fidelity 0.8454 -> 0.6963 ...
[post-check] dl_prb F_c=0.8935 vs threshold 0.5 -> meets threshold
──────────────────── adaptation event ────────────────────
ADAPTATION EVENT — component=dl_prb action=regenerate attempt=2/5 event_id=5e4f3637-...
  gate_result=REJECT  (verification fidelity 0.8454 -> 0.6963)
  version dl_prb-v10 -> dl_prb-v10
  post-adaptation F_c=0.8935 vs threshold 0.50 -> meets threshold
  duration_seconds=35.20
[loop] dl_prb: now meets its threshold (F_c=0.8935) — no further attempts needed after attempt 2/5
```

Meanwhile, on a second worker thread, `ul_bler` was adapted at the same time
(`[agent] recalibrate running for ul_bler only ...` → `[hot-swap] ul_bler-v15 now serving live
predictions for ul_bler`), and cycle tables kept printing between these lines. On an ACCEPT the
candidate is hot-swapped in and its downstream components are re-checked — see §3.3.

What is normal, and what is not:

| You see | Meaning |
|---|---|
| `gate_result=REJECT` followed by `[retry]` and `[attempt 2/5]` | Normal — the adaptation loop is trying again with the rejection details as feedback; production is untouched meanwhile. |
| `[skipped] X: trigger from ... is stale` | Normal — X recovered while its trigger was waiting; adapting a healthy model is pointless. |
| A `[cycle N]` table printed between the steps of an adaptation | Normal — adaptations run on worker threads; cycles never wait for them. |
| Steps of two different components interleaved | Normal — up to 2 components adapt concurrently. |
| `[warning] recalibrate failed ...: only 180 rows available ..., need >= 200` | Normal early in a run — not enough live history yet; that cycle is skipped, nothing else is affected. |
| A component shows `BELOW 1/3`, `2/3` and then `ok` | Normal — its policy needs 3 consecutive low cycles; a recovery resets the counter. |
| `F_U` is low but nothing triggers (or `F_U` is high and a component still adapts) | Normal — `F_U` is a summary only; adaptation is decided per component. |
| `[post-check] ... STILL BELOW threshold` after an adaptation | The adaptation (or the REJECT) did not bring that component up to its threshold; it stays flagged and its own policy re-triggers after N more low cycles. |
| `[dependents] revalidated X: ... BELOW` | A downstream component is below its threshold after an upstream swap. It is reported, not auto-adapted; its own policy handles it. |
| Negative `F_U`/`F_c` values | Allowed by the fidelity formula — the current error exceeds everything in the rolling window. |
| `[skipped] no new telemetry since ts_1s=... — the replay source has been fully consumed` | The replay finished. With no new data there are no new fidelity measurements, so nothing further triggers; the system idles. Stop with Ctrl-C. |
| Few or no adaptations during a run | Normal when the models are healthy — a component adapts only when its own `F_c` stays below its threshold. To make a component adapt more readily, raise its threshold in `fidelity.component_policies`. |
| `[skipped] D1 already holds N replayed rows ... resuming the replay after that tick` | A restart: D1 is persistent, so the replay resumes where the last run stopped. See §3.5. |
| `decision ... confidence=0.00` / log line "deterministic fallback" | The LLM call failed (no key, quota exhausted, network). The system continues with the fallback strategy. |
| A Python traceback, or no console output for minutes during a replay | **Not normal** — see §6. |

Reference numbers from the verified 15-minute live run with the adaptation loop and worker
threads (2026-10-20, fresh D1, `--mode live --replay-speed 10`): all 25,865 rows of run 8
ingested, 89 fidelity cycles, 7 per-component triggers → 5 adaptation attempts (all genuine LLM
decisions, 0 fallbacks): **2 ACCEPTED and hot-swapped** (`dl_bler` → `dl_bler-v13`, `ul_bler` →
`ul_bler-v15`), 3 REJECTED (production kept), plus 2 stale triggers skipped because their
component had recovered; one loop retried (`dl_prb`: recalibrate → regenerate) and stopped when
the component recovered; two pairs of components adapted concurrently (`dl_bler`+`dl_prb`,
`dl_prb`+`ul_bler`), never the same component twice; 0 warnings, 0 errors, 0 tracebacks. (The
previous run, before these fixes, ended 5 of 5 adaptations in REJECT.) A separate 4-minute run
verified Ctrl-C mid-adaptation: `[skipped] interrupted (Ctrl-C) — shutting down cleanly` →
`[shutdown] background sync thread confirmed exited, D1 flushed to disk`, and it produced a third
ACCEPT (`ul_bler-v17`).

Timing: one attempt takes ≈15–30 s for recalibrate and 30–90 s for regenerate/expand-scope
(LLM calls + sandboxed training); a full 5-attempt loop can take several minutes. Nothing else
waits for it.

### 3.3 Model updates and hot swap

When verification ACCEPTs a candidate, it replaces the production model **in the running process
immediately** — no restart:

- **recalibrate**: the retrained model (same class) is swapped into live serving.
- **regenerate / expand-scope**: the LLM-generated class is imported for the first time — only
  after the sandbox and the verification gate have both passed it — loaded with its trained
  artifact, and swapped in. Its source is kept at `data/models/<component>/<version>.py`.
- The registry marks the new version `production` and the old one `superseded` (kept, so it can be
  rolled back). A REJECTed candidate is marked `rejected` and never served.
- Right after the swap, the component and every component downstream of it are re-measured on the
  latest data; downstream components are reported, never auto-adapted.
- **After a restart** the current production versions are loaded from the registry — including
  LLM-generated classes — so a hot-swapped model stays in service.

Real output of an ACCEPT from a live run (numbers rounded as the console prints them):

```
[agent] candidate dl_prb-v6 (DlPrbModel) produced — train rows 10368, held-out rows 2592
[verify] ACCEPT: fidelity 0.7526 -> 0.7931 (required improvement > 0.01); checks: candidate_artifact_present=PASS, ...
[hot-swap] dl_prb-v6 now serving live predictions for dl_prb
[post-check] dl_prb F_c=0.7911 vs threshold 0.5 -> meets threshold
[dependents] revalidated dl_throughput: F_c=0.7610 threshold=0.5 -> ok — consistent with the new upstream model
[dependents] revalidated dl_rlc_delay: F_c=0.8190 threshold=0.5 -> ok — consistent with the new upstream model
[lifecycle] record 87c8c1f9-... written; maintenance report 87c8c1f9-....md (LLM prose=True)
──────────────────── adaptation event ────────────────────
ADAPTATION EVENT — component=dl_prb action=recalibrate event_id=87c8c1f9-...
  gate_result=ACCEPT  (verification fidelity 0.7526 -> 0.7931)
  version dl_prb-v5 -> dl_prb-v6
  post-adaptation F_c=0.7911 vs threshold 0.50 -> meets threshold
  dependent dl_throughput    F_c=0.7610 threshold=0.50 -> ok — consistent with the new upstream model
  dependent dl_rlc_delay     F_c=0.8190 threshold=0.50 -> ok — consistent with the new upstream model
```

**How to check that hot swaps worked:**

- `grep "hot-swapped ACCEPTed candidate" logs/digital_twin.log` — one line per swap;
  `grep "hot-swapping into live serving failed" logs/digital_twin.log` should find nothing.
- The next cycle tables show the new version in the `model version` column.
- `logs/adaptation_decisions.jsonl`: `outcome: "accepted"` with `version_before` → `version_after`.
- On the next start, the startup banner lists the swapped-in versions, and the JSON log contains
  `dynamically loaded a regenerate/expand_scope-produced production DT model` for an LLM-generated one.

Verified 2026-10-20 across all live runs: 9 hot swaps (recalibrated and LLM-regenerated
`dl_prb`, `dl_bler`, `dl_rlc_delay` models), 0 swap failures, every ACCEPT promoted in the
registry, no candidate left undecided, and the LLM-regenerated `dl_prb-v10` reloaded after a
restart and served valid predictions for all downstream components.

### 3.4 The three adaptation agents — and why most candidates are rejected

| Agent | What it builds | Training data | Verified against |
|---|---|---|---|
| **recalibrate** | Same model class, retrained | The original TRAIN pool (60,858 rows) **plus** the recent window (last 24 h of D1), recent rows weighted ×5; held-out = last 20% of the recent window | Current production model, same held-out rows |
| **regenerate** | A new pipeline written by Gemini, trained in the sandbox | TRAIN pool + recent window (48 h) | Current production model, same held-out rows |
| **expand-scope** | A **new component** predicting a KPI no component covers yet — one of the 8 observable MAC KPIs (`dl_curr_tbs_avg`, `dl_sched_rb_avg`, `pusch_snr_avg`, `pucch_snr_avg`, `wb_cqi_avg`, `dl_mcs1_avg`, `ul_mcs1_avg`, `phr_avg`) | Recent window (48 h) | The naive persistence predictor `y(t) = y(t−1)` — a new component must beat it |

Every candidate faces the same deterministic gate: **ACCEPT only if `F_new > F_old + 0.01`**
(`adaptation.verification_delta`) and all safety checks pass. The fidelity formula itself is
unchanged.

Why REJECT is still the common outcome — and why that is healthy:
- **Production is strong.** The 7 models are residual models (they predict the one-second change
  and add it to the last value), trained on 60k rows. On most windows of run 8 they are already
  close to the best achievable; a candidate has to be *measurably* better (by 0.01), not just as
  good.
- **Held-out data is in the future.** Candidates are scored on rows *after* their training rows;
  when traffic changes regime (e.g. a link going idle at the end of a run), a model can't have
  seen it coming. Example from the agent tests: an LLM-generated `dl_curr_tbs` component trained
  where the KPI averaged 96 was rejected on a held-out tail where it averaged 0.04 — persistence
  wins on an idle link.
- **A rejection is never wasted**: the loop retries with the numbers, and production is never put
  at risk.

What was fixed so candidates *can* win (2026-10-20, see `CHANGELOG.md`): training no longer
throws away every row with any missing lagged feature (early in run 8 that was nearly all of
them); recalibration/regeneration train on the TRAIN pool plus recent data instead of a few
hundred recent rows; stale triggers for recovered components are skipped; LLM code generation is
no longer truncated (`llm.max_tokens` 32768); and expand-scope — previously impossible on OAI
data — now has real targets and a gate it can actually pass. LLM-generated candidates never see
any value measured in the same second as their target (no label leakage).

Test any agent on its own, on real OAI data (run 7), through the real orchestrator wiring:

```bash
python scripts/test_adaptation_agents.py                          # all three
python scripts/test_adaptation_agents.py --strategies expand_scope --attempts 3
```

The script pins the strategy (logged as `decision_source: pinned`, console `(PINNED)`), uses its
own storage `data/agent_tests/` (wiped each run) and prints one line per attempt.

### 3.5 State, restarts, and starting over

Everything the live system produces is persistent:

| Path | Contents |
|---|---|
| `data/artifacts/d1_current_state.parquet`, `d1_history.parquet`, `d1_quarantine.parquet` | D1 — the DT state store (current per-UE state + full history; quarantine only if records were rejected). |
| `data/models/registry_index.json`, `data/models/<component>/` | Versioned model registry: every bootstrap/recalibrated/regenerated version, which is production, and the LLM-generated source (`.py`) of regenerated ones. |
| `data/artifacts/lifecycle_records.jsonl` | One auditable record per adaptation event (trigger, root-cause analysis, strategy, fidelity before/after, gate result, telemetry source, replayed runs). |
| `data/artifacts/maintenance_reports/<event_id>.md` | Human-readable maintenance report per event (includes the component's threshold, post-adaptation fidelity and dependency revalidation). |
| `logs/component_fidelity.csv` | One row per component per fidelity cycle: model version, `F_c`, threshold, gap, policy state, debounce counter, severity, data window, scored rows, `F_U` (report only). The per-component fidelity history — open in any spreadsheet. |
| `logs/adaptation_decisions.jsonl` | One line per adaptation **attempt** (`attempt`/`max_attempts`) — **including** skipped, failed and interrupted ones (`outcome`: `accepted`, `rejected`, `agent_failed`, `verification_error`, `skipped_recovered` (stale trigger), `skipped_no_llm_client`, `skipped_unknown_component`, `interrupted` (Ctrl-C mid-attempt), `error`), with fidelity at trigger, threshold, strategy, decision source (`llm` / `fallback` / `pinned`), versions before/after, verification fidelity, failed checks, post-adaptation fidelity, remaining gap, dependency revalidation and duration. |
| `logs/console/<mode>_<UTC timestamp>.log` | Plain-text copy of that run's console output, timestamped per line. |
| `logs/digital_twin.log` | Full structured JSON log (rotating, 10 MB × 5). |

On restart:

- **Models** are loaded from the registry (including any hot-swapped LLM-regenerated class) —
  never retrained. (Exception: running `scripts/bootstrap_oai_models.py` replaces them — §2.4.)
- **The replay resumes** after the last tick already in D1. To replay run 8 from the beginning,
  delete the D1 files first:
  ```bash
  rm -f data/artifacts/d1_current_state.parquet data/artifacts/d1_history.parquet data/artifacts/d1_quarantine.parquet
  ```
- To also reset the models (fresh bootstrap on next start), delete `data/models/`. Lifecycle
  records/reports are an append-only audit trail — remove them only if you really want a clean
  slate.

D1 is written to disk at most every `storage.persist_min_interval_seconds` (10 s); in-memory state
is always current and Ctrl-C flushes the remainder. A hard kill (`kill -9`, closing the terminal)
can lose at most the last ~10 s of D1 history.

---

## 4. Other runnable scripts

```bash
python scripts/run_oai_e2e_smoke.py        # replays run 7 into data/e2e_oai_smoke/ and forces one internal fidelity trigger
python scripts/run_orchestrator_demo.py    # one full cycle on mock telemetry — WRITES to the real data/ paths
python scripts/test_adaptation_agents.py   # each adaptation agent individually on real OAI data (§3.4) — own storage
python scripts/demo_recalibration.py       # older single-agent demos on MOCK telemetry (regeneration/expand-scope need an API key)
python scripts/demo_regeneration.py
python scripts/demo_expand_scope.py
python scripts/bootstrap_oai_models.py     # CV-tuned bootstrap models + reports/model_eval/ — REPLACES production models (§2.4)
python scripts/plot_model_eval.py          # graphs from reports/model_eval/model_eval.csv (never retrains)
python scripts/visualize_metrics.py        # plots from live D1 history
```

`run_orchestrator_demo.py` and `--mode demo` bootstrap and register models in the real
`data/models/`; don't mix them into a registry you intend to use for live results. The two
validation scripts inject one clearly-labelled internal fidelity trigger (`metadata.forced=True`)
if no component's policy fires on its own, so they always exercise a full adaptation cycle.

### 4.1 Baseline comparison experiments (`baselines/`)

A 4-step ablation over the real OAI TEST split (run 8), scored on the same 62 fixed 60-second
windows:

- `static`: no adaptation.
- `periodic_retrain`: all 7 components recalibrated every 10 min of data, deployed without the gate.
- `rule_based`: severity rule picks the strategy, then the real agents, gate and retry loop.
- `full_system`: the real LLM Decision Agent with the same agents, gate and retry loop; 3 runs.

Agents only ever see data up to the current window (no look-ahead), and no mock drift is injected.

```bash
python baselines/run_all.py                                        # all 4 modes -> baselines/results/<mode>/ (~2-3 h)
python baselines/run_all.py --modes static periodic_retrain --skip-preflight
python baselines/run_all.py --max-windows 15 --seeds 99 --skip-preflight      # smoke run
python baselines/stats.py                 # frozen common-scale scores, Wilcoxon + block bootstrap + Holm -> results/stats/
python baselines/compare.py               # operational counts (runtime scores are NOT comparable across modes)
python scripts/generate_ieee_figures.py   # 7 IEEE figures -> figures/ (PDF+PNG+CSV); --synthetic = watermarked placeholders
```

Each mode run deletes and rebuilds its own registry/D1 under `baselines/results/`, never `data/`.
LLM spend is capped by `llm.budget_usd`. Results from before 2026-10-20 are in
`baselines/results_archive_pre_2026-10-20/`. They used the leaky protocol, so do not use them.
See `baselines/README.md` for the full protocol.

**Measured result (2026-10-20, run 8).** On the frozen common scale, mean windowed fidelity was:

| Static | Rule-based | Proposed (3 runs) | Periodic |
|---|---|---|---|
| 0.8196 | 0.8174 | 0.8155 | 0.8122 |

Every pairwise difference is below 0.006 and Holm-significant. The proposed system was slightly
*worse* than static and rule-based. Run 8 has traffic only in its first ~18 minutes, and all
triggers fired at minutes 14–15; the 44 idle windows dominate the averages. Candidates accepted
during that burst scored slightly lower afterwards than the untouched bootstrap models. Run 8
alone cannot show an advantage for adaptation; see IMPLEMENTATION_STATUS item 16.

---

## 5. Testing

```bash
python -m pytest tests/unit -q                    # 576 tests, ~2-7 min
python -m pytest tests/integration tests/e2e -q   # 35 tests, ~6-13 min
```

Run unit and integration separately — a single combined run has previously been killed for
memory in this environment. Verified 2026-10-20 with the rebuilt dataset (after the adaptation
loop / multithreading / expand-scope changes): **576 + 35 passed, 0 skipped, 0 failed.**

Where the key behaviour is tested:

| Behaviour | Tests |
|---|---|
| Per-component policy decisions | `tests/unit/test_component_policy.py` |
| Policy wired into the orchestrator, logging, dependency revalidation, failures | `tests/unit/test_per_component_orchestration.py` |
| Fidelity formula unchanged (hand-computed values) | `test_composite_score_hand_computed_case` in `tests/unit/test_fidelity_evaluator.py` |
| Hot swap (ACCEPT swaps in, REJECT doesn't, survives restart) | `tests/integration/test_hot_swap.py`, `tests/unit/test_main.py` |
| Full live cycle, telemetry never blocked, no external drift wired in | `tests/integration/test_main_orchestrator.py` |
| Adaptation loop (retry with feedback, stop rules), stale-trigger skip, worker threads (two components truly concurrent, never one twice) | `tests/unit/test_adaptation_lock.py` |
| NaN-tolerant residual training, sample weights | `tests/unit/test_residual_base.py` |
| Reference-data training set | `tests/unit/test_data_selection.py` |
| Expand-scope targets (observable KPIs only, no same-second inputs) | `tests/unit/test_expand_scope_agent.py` |
| New component vs. persistence gate | `tests/unit/test_verification_agent.py` |
| Same-second columns never reach a sandboxed candidate | `tests/unit/test_sandbox_executor.py` |

- Tests that need `data/processed/oai/*.parquet` **skip** with an explicit reason if the dataset
  hasn't been built (§2.4) — build it to run the full suite.
- Some integration tests read the real `.env` and make real Gemini calls (small cost). A
  free-tier daily quota can be exhausted by repeated runs on the same day — that shows up as real
  test failures/fallbacks, not as a code bug.

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `expected escape sequence of 8 hexadecimal numbers` when anything starts | Backslash Windows path in `config/settings.yaml` — see §2.3. |
| `FileNotFoundError: .../data/processed/oai/train.parquet does not exist` | Dataset not built — run `python scripts/build_oai_dataset.py`. |
| `required OAI export file missing: ...` from the dataset build | `data.oai_root` is wrong, or a run in `data.splits` is incomplete. |
| `python: command not found` / `No module named pandas` | Not inside WSL, or the venv isn't active — `source ~/.venvs/dt/bin/activate`. |
| Replay ingests slowly / output stalls as history grows | D1 disk-write throttling disabled — check `storage.persist_min_interval_seconds` is > 0 (default 10). Ingestion should be ~55–70 rows/s at `--replay-speed 10`. |
| Nothing happens after the replay finishes | Expected (§3.2). Delete the D1 files and restart to replay again (§3.5). |
| Restarted run immediately says "replay fully consumed" | D1 already contains the whole run — delete the D1 files to replay again (§3.5). |
| A hot-swapped model is no longer in production; the registry shows a newer `bootstrap` version | `scripts/bootstrap_oai_models.py` was run — it promotes fresh bootstrap models over production (§2.4). Roll back via the registry if that wasn't intended. |
| `[warning] candidate ACCEPTed but hot-swap into live serving failed` | The registry promotion stands; the new model is picked up on the next restart. Check the error in `logs/digital_twin.log`. |
| Every decision has `confidence=0.00` | LLM failing — check `GOOGLE_API_KEY`, quota, network; details in `logs/digital_twin.log` (`"Google AI API call failed"`). |
| First adaptation pauses noticeably | First RAG query loads the embedding model; LLM calls take seconds each; regenerate trains a candidate in a sandboxed subprocess (~30–50 s). Telemetry ingestion and fidelity cycles continue throughout. |
| `expand_scope failed ...: no unpredicted same-second KPI with enough data` | The dataset predates the observable KPI columns, or the D1 history was written before them — rebuild (`python scripts/build_oai_dataset.py`) and start with fresh D1 files (§3.5). |
| `structured-output validation failed ... finish_reason=MAX_TOKENS` in the JSON log | The LLM answer was cut off — raise `llm.max_tokens`. The attempt counts as failed and the loop retries. |
| Ctrl-C takes up to ~20 s | Shutdown waits `adaptation.shutdown_wait_seconds` for running adaptation threads, then records unfinished ones as `interrupted`. |

---

## 7. Configuration reference (`config/settings.yaml`)

Every tunable lives here; nothing that belongs here is hardcoded in `src/`. The settings you are
most likely to touch:

| Key | Default | Meaning |
|---|---|---|
| `data.oai_root` | WSL path to the export | Read-only OAI RAN KPI export (§2.3). |
| `data.splits` | train `[3, 6, 7]`, test `[8]`, smoke `7` | Which runs bootstrap the models and which one `--mode live` replays. |
| `telemetry.replay.speed` | `0.0` | Default replay pace (`--replay-speed` overrides). |
| `storage.persist_min_interval_seconds` | `10.0` | Minimum seconds between D1 Parquet rewrites (0 = every batch). |
| `fidelity.trigger_threshold` / `trigger_min_consecutive_evaluations` | `0.5` / `3` | Default per-component policy: a component adapts when its own `F_c` is below the threshold for this many consecutive cycles. |
| `fidelity.component_policies` | `{}` | Per-component overrides, e.g. `dl_rlc_delay: { threshold: 0.4, min_consecutive_evaluations: 4 }`. |
| `logging.component_fidelity_log` / `adaptation_decisions_log` / `console_transcript_dir` | `logs/...` | Where the run logs of §3.5 are written (`console_transcript_dir: null` disables transcripts). |
| `adaptation.verification_delta` | `0.01` | Candidate must beat production fidelity by more than this to be ACCEPTed. |
| `adaptation.max_adaptation_attempts` | `5` | Attempts per adaptation loop before production is kept (§3). |
| `adaptation.allow_concurrent_components` / `max_concurrent_adaptations` | `true` / `2` | Worker threads: how many components may adapt at once (`false` = one at a time). |
| `adaptation.shutdown_wait_seconds` | `20` | How long Ctrl-C waits for running adaptations. |
| `adaptation.recalibration.include_reference_data` / `recent_sample_weight` | `true` / `5.0` | Train recalibration candidates on TRAIN pool + recent window, recent rows weighted ×5. |
| `adaptation.regeneration.include_reference_data` | `true` | Same reference rows for LLM-regenerated candidates. |
| `adaptation.*.min_training_rows` / `training_window_hours` | `200` / 24–48 h | Data requirements per adaptation agent. |
| `sandbox.timeout_seconds` | `120` | Wall-clock limit for an LLM-generated candidate's sandboxed train/eval. |
| `llm.model` / `max_tokens` / `timeout_seconds` / `max_retries` | `gemini-3.6-flash` / `32768` / `90` / `3` | Gemini settings. `max_tokens` includes thinking tokens — too low truncates generated code. |
| `drift.*` | — | Used only by the `baselines/` experiment harness; the live system never reads it (external drift is disabled). |

---

## 8. Repository map

### 8.1 Top-level files

| File | What it is |
|---|---|
| `CLAUDE.md` | Architecture/design document and full project history (§12). |
| `IMPLEMENTATION_STATUS.md` | Per-module status tracker and open items. |
| `manual.md` | This file. |
| `README.md` | Short project overview. |
| `AGENTS.md` | The 6 agents — what each decides and which tools/resources it may access. |
| `BASELINES.md` | The baseline comparison framework: modes, formulas, graphs. |
| `CHANGELOG.md` | Bugs found while running the system and design changes (per-component policy, external drift removal) — newest first. |
| `PHASE13_FINAL_REPORT.md` | Phase-13 results: model CV results, LLM cost/caching, baseline comparison. |
| `config/settings.yaml` | All configuration (§7). |
| `.env` / `.env.example` | Secrets / template. `.env` is never committed. |
| `requirements.txt`, `pyproject.toml` | Dependencies; project metadata + pytest config. |
| `fig-dataflow.png` | Canonical 19-module + D1/D2 data-flow diagram. |

### 8.2 `src/` — production code

| Path | Responsibility |
|---|---|
| `src/main.py` | `ContinuousOrchestrator` and the CLI: bootstrap/load models, start telemetry sync, prediction+fidelity cycles with the per-component policy (the only trigger source), trigger queue (one adaptation per component, coalesced), decision → agent → verification → lifecycle, live hot-swap, replay resume, clean shutdown. |
| `src/common/` | `config.py` (validated `Settings` + redacted `Secrets`), `logging.py` (structured JSON logging with secret redaction), `console_report.py` (the detailed console output + transcript), `run_logs.py` (component fidelity CSV + adaptation decisions JSONL). |
| `src/telemetry/` | `oai_replay_source.py` (Module 1 — deterministic, pausable/seekable replay of the real OAI rows), `mock_source.py` (mock telemetry), `preprocessing.py`/`schema.py`/`base.py` (Module 2 — validation and quarantine). |
| `src/data/oai_dataset.py` | Builds the causality-correct lagged/rolling feature set from the raw OAI export. |
| `src/synchronization/` | Module 3 — `ContinuousSynchronizer` (background thread, batched, no manual sync step) and the `D1StateSink` contract. |
| `src/dt_models/` | D1 store (`d1_model_store.py`, `component_registry.py`), Module 5 (`base.py`, `model_registry.py` — in-memory component wiring, `orchestrator.py`), the shared residual base and regressor factory, and the 7 components (`dl_bler`, `ul_bler`, `dl_prb`, `ul_prb`, `dl_throughput`, `ul_throughput`, `dl_rlc_delay`). |
| `src/fidelity/` | Module 12 — `metrics.py` (RMSE/MAE/Wasserstein/MK-MMD), `evaluator.py` (rolling normalization, per-component `F_c` + unified score for reporting), `component_policy.py` (the per-component adaptation policy), `trigger.py` (canonical `AdaptationTrigger`). |
| `src/drift/` | Module 11 — external drift-event contract, validation/quarantine, and the mock drift source. Not used by the live system (external drift is disabled); used by `baselines/`. |
| `src/adaptation/` | Agents: `decision_context.py`/`decision_agent.py` (Module 13), `recalibration_agent.py` (14), `regeneration_agent.py` (15), `expand_scope_agent.py` (16), `verification_agent.py` (17 — the deterministic gate), `lifecycle_agent.py` (19); `data_selection.py` (shared window helpers); `baseline_policies.py` (used only by `baselines/`). |
| `src/rag/rag_kb.py` | D2/Module 18 — read-only `RagKnowledgeBase` for agents; `RagIngestor` for `scripts/ingest_rag.py` only. |
| `src/registry/model_registry.py` | Versioned model registry — every version, production pointer, promote/reject/rollback, saved LLM-generated source. |
| `src/sandbox/` | Subprocess isolation for LLM-generated candidates (syntax → import → conformance → train → evaluate → save). |
| `src/llm/` | `google_client.py` (the single Gemini client: retries, structured output, caching, safe wrappers); `cost_tracking.py` (budget guard and LLM call audit log for experiments). |
| `src/storage/` | Empty placeholder. |

### 8.3 Other folders

| Path | What it is |
|---|---|
| `baselines/` | Comparison framework: `run_all.py` (CLI), `compare.py` (cross-mode report), `_lib.py` (shared harness), `_outputs.py`, `_console.py`, `configs/{static,periodic,rule_based,full_system}.yaml`, `results/` (gitignored output). |
| `data/processed/oai/` | Built dataset (`train.parquet`, `test.parquet`, `feature_schema.json`, `data_report.md`). Gitignored. |
| `data/artifacts/`, `data/models/` | Live-system state (§3.5). Gitignored. `data/artifacts/sandbox/` is the sandbox scratch area. |
| `data/e2e_oai_smoke/` | Storage used only by `scripts/run_oai_e2e_smoke.py`. |
| `docs/` | `MODELS.md` (authoritative per-model documentation), `oai_twin.md` (OAI migration design), `architecture_diagram.drawio`, older manuals, and `FORMULAS.pdf`/`IEEE_Digital_Twin_Paper.docx` (pre-OAI, see `docs/README.md`). |
| `reports/model_eval/` | CV evaluation report and graphs from `scripts/bootstrap_oai_models.py`/`plot_model_eval.py`. |
| `rag_data/` | RAG source documents (`oran/`, `digital_twin/`, `policies/`, `history/`) and the vector store (`chroma/`, gitignored). |
| `logs/` | `digital_twin.log` (full JSON log), `component_fidelity.csv`, `adaptation_decisions.jsonl`, `console/` (per-run console transcripts) — see §3.5. |
| `scripts/` | Setup, dataset/RAG build, demos and validation scripts (§2, §4). |
| `tests/` | `unit/` (one file per module), `integration/` (multi-module, real sandbox/persistence, some real LLM calls), `e2e/` (baseline `rule_based` lifecycle over run 7 with a faked LLM), `dummy_dt_components.py`. |
| `archive/ns3_sim/` | Retired NS-3/5G-LENA toolchain and data, kept for reference. Nothing in `src/` may import from it (enforced by `tests/unit/test_no_archive_imports.py`). |

### 8.4 A completed `baselines/results/<mode>/` folder

```
results/<mode>/                      (results/full_system/seed_<n>/ for full_system)
├── metrics_timeline.csv             # per cycle/component: actual, predicted, F_c, F_U, trigger_fired, model_version
├── adaptation_events.jsonl          # one line per adaptation attempt
├── report.md                        # human-readable run report
├── run_summary.json                 # triggers, accepted/rejected, mean unified fidelity, budget spent
├── llm_calls.jsonl                  # every real LLM call (absent for static/periodic_retrain)
├── d1_current.parquet / d1_history.parquet
├── models/                          # this run's isolated registry
└── graphs/                          # actual_vs_predicted_<component>, fidelity_<component>, unified_fidelity
```

`baselines/compare.py` writes `results/comparison/{comparison.md, comparison.csv}` and
`graphs/{unified_fidelity_all_modes, mean_fidelity_by_mode, recovery_time_by_mode,
adaptations_by_mode, strategy_distribution}.{png,pdf,csv}`.
