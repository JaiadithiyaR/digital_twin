# CLAUDE CODE TASK — Phase 13: Model fixes, LLM setup, baselines, graphs, readable outputs, docs

You are working in the existing repo `ai-self-adaptive-digital-twin`, after the OAI migration (Phase 12). Read `CLAUDE.md`, `IMPLEMENTATION_STATUS.md` and `data/processed/oai/data_report.md` first.

Work through the tasks **in order**. Stop and ask me ONLY if one of the blockers in §0 occurs; otherwise continue to the end and give the final report.

---

## 0. Ground rules

- **Data:** read-only, at `config.data.oai_root`. No database access exists; never add database code.
- **Splits:** TRAIN/TUNE = runs **3, 6, 7**; TEST = run **8**.
  - Run 8 must not be used for training, tuning, debugging or smoke tests. Use run 7 for smoke tests.
  - Run 10 stays excluded (incomplete export).
- **Fidelity formula:** unchanged (RMSE, MAE, Wasserstein, MK-MMD; original normalization; weighted unified score).
- **No fabricated results:** every number, table and graph must come from code that actually ran. If something fails, report it; never fill gaps with estimates.
- **Units:** throughput in kbps and RLC delay in µs, as found in Phase 12. Pick ONE display unit per KPI (Mbps for throughput, ms for delay), convert in a single place, and state units in every table, graph axis and doc.
- **Blockers — STOP and ask me if:**
  1. the LLM provider is unavailable, out of quota, or the budget guard is hit when an LLM-dependent mode needs it (see §3 and §4)
  2. any required run file is missing or `.partial`
  3. the 989 MB directory turns out NOT to be a duplicate and its contents look irreplaceable

---

## 1. Fix the bootstrap models

**1.1 Training data and tuning**
- Train on runs 3, 6 and 7.
- Tune hyperparameters with **leave-one-run-out CV** over {3, 6, 7}: three folds, each holding out one whole run. Never split a run's rows across folds.
- After tuning, retrain each component on all of 3 + 6 + 7.

**1.2 Residual targets**
- Each component learns `Δy = y_t − y_{t−1}`, and its prediction is `ŷ_t = y_{t−1} + Δŷ`.
- The public interface is unchanged: `predict()` still returns `ŷ_t`.

**1.3 Richer past-only features**
- For each target and for the 10 MAC features (`dl_curr_tbs`, `dl_sched_rb`, `pusch_snr`, `pucch_snr`, `wb_cqi`, `dl_mcs1`, `ul_mcs1`, `phr`, `dl_bler`, `ul_bler`), add:
  - lags `t−1 … t−5`
  - rolling mean and std over 5 s and 10 s
- All of these are computed strictly from past seconds, within `(run_id, session_id)`.
- Extend the leakage test so it fails if any feature uses data from second `t` or later.

**1.4 Dependency inputs**
- Downstream components receive the **predicted** upstream values for `t` AND the **measured** upstream values for `t−1`.

**1.5 Evaluation, recorded in `reports/model_eval/`**
- Per component, per CV fold, and averaged: RMSE and MAE for the model, persistence (`y_{t−1}`) and a 5-s moving average.
- Mark whether the model beats persistence. If a component still loses, report it honestly; do not keep tuning to hide it.
- Write `reports/model_eval/model_eval.md` and `model_eval.csv`.

**1.6 Register**
- Retire the current OAI v1 versions in the registry, keeping them for rollback.
- Register the new versions as `v2 (bootstrap, OAI, residual)`.
- Re-run all tests and update `data_report.md`.

---

## 2. Adaptation lock (the paper describes it, so it must exist)

Implement `config.adaptation.lock_policy = coalesce`:
- At most **one adaptation per component** at a time.
- Triggers for a locked component are **coalesced**: duplicates are merged and processed in the next cycle after the lock is released, keeping the highest severity.
- Telemetry ingestion and production inference continue during adaptation.
- Different components may adapt concurrently only if config allows it (default: no).

Tests:
- a second trigger for the same component is coalesced
- the lock is released after accept, reject and error
- ingestion continues during adaptation
- no deadlock when the sandbox fails

---

## 3. LLM configuration and cost controls (gemini-3.6-flash)

**3.1 Model**
- Use `gemini-3.6-flash` for ALL LLM roles: decision/RCA, regeneration, expand-scope and verification explanation.
- The model id lives in config (`llm.model`). Never hardcode it.
- `rule_based` and `full_system` must use the identical model for every role they share.
- Record the model id on every LLM call in `adaptation_events.jsonl`, and in `comparison.md`.

**3.2 API key**
- Use a dedicated **paid-tier** key from an environment variable.
- Never write the key to files, logs, reports or commits.

**3.3 Context caching**
- Restructure every prompt so that ALL static content comes first, in a fixed order:
  1. system instructions
  2. adaptation policies
  3. OAI/DT documentation summary
  4. `feature_schema.json`
- Dynamic content comes after it: trigger, fidelity values, RAG chunks, code and statistics.
- Cache the static prefix with explicit context caching:
  - one cache per experiment run
  - TTL = the expected run length (config)
  - delete the cache at the end of the run, including on error
- If the prefix is below the model's minimum cacheable size, or caching fails, log it and continue without caching. Never change prompt content to make caching work.
- Caching must not change model inputs. Add a test that the assembled prompt (cached prefix + dynamic part) is byte-identical to the uncached prompt.

**3.4 Thinking budget**
- Make the thinking level/budget configurable per role:
  - `decision`: low/medium
  - `codegen`: medium
  - `explanation`: low
- Log thinking tokens separately.

**3.5 Cost tracking** — log every LLM call to `llm_calls.jsonl` with:
- mode, run, role
- input tokens, cached tokens, output tokens, thinking tokens
- latency
- cost, computed from rates in config (`llm.rates`: input, output, cached input, cache storage per hour). Do not hardcode rates; prices change on 1 January 2027.

Report per-mode totals (calls, tokens, cost, LLM time) in `comparison.md` and in the final report.

**3.6 Budget guard**
- Config `llm.budget_usd` (default **20**).
- Before each call, estimate its cost. If spent + estimate would exceed the budget, stop the experiment cleanly, save all outputs so far, mark the run incomplete, and report.
- Never silently fall back to the default strategy because of the budget.

**3.7 Service tier**
- Config `llm.service_tier: standard | flex` (default `standard`).
- `flex` is allowed only for `speed=0` experiments, never for live demos.
- Record the tier used in every result.

**3.8 Pre-flight**
- Before any baseline experiment, run one smoke test on **run 7** (never run 8) that exercises every LLM role with caching ON.
- Print a one-call cost summary, and STOP if any call fails.

---

## 4. Baseline comparison (new top-level folder `baselines/`)

```
baselines/
  README.md              how to run, what each mode is, fairness rules
  configs/               static.yaml, periodic.yaml, rule_based.yaml, full_system.yaml
  run_all.py             runs every mode on TEST run 8 and writes results
  compare.py             builds tables + graphs from results only
  results/<mode>/        per-mode outputs (see §6)
  results/comparison/    comparison tables + graphs
```

**Modes:**

| Mode | Behaviour |
|---|---|
| `static` | Bootstrap v2 models, never adapted. Triggers are logged but ignored. |
| `periodic` | All components retrained every N seconds of replayed data (config, default 600 s) on a trailing window. No triggers, decision agent or LLM. |
| `rule_based` | Same triggers, adaptation agents, sandbox and gate as the full system. Strategy comes from fixed severity rules in config (e.g. `<0.3` recalibrate, `<0.7` regenerate, else expand_scope). No LLM. |
| `full_system` | The knowledge-based LLM + RAG decision agent. |

**Fairness rules (enforce in code and state in the README):**
- same bootstrap v2 models
- same TEST replay (run 8, `speed=0`, same order)
- same fidelity config
- same seeds
- every mode starts from a fresh copy of the registry and state store

**LLM check (full_system):**
- Before starting, run the pre-flight check from §3.8.
- If the LLM fails (quota, key, network) or the budget guard is hit, **STOP and tell me**. Do not run `full_system` on the fallback strategy.
- During the run, record per adaptation whether the decision came from the LLM or from the fallback. If more than 10% are fallbacks, flag the run as invalid in the comparison.
- Run `full_system` 3 times (seeds from config) and report the mean ± std.

**Comparison metrics** (written to `results/comparison/comparison.md` and `.csv`):
- mean unified fidelity and mean per-component fidelity
- % of time above threshold θ
- time to recover above θ after each trigger
- number of triggers, number of adaptations, accepted vs. rejected
- total retraining/adaptation time
- LLM calls and fallbacks (`full_system` only)
- LLM tokens, cache hit share, cost (USD) and LLM time per mode (`rule_based` and `full_system`)

---

## 5. Graphs (separate files)

Use matplotlib only: one figure per file, PNG at 300 dpi plus PDF. Titles, labelled axes with units, legends, and a consistent style.

Next to every graph, save the CSV of exactly the data it plots.

**Per mode → `baselines/results/<mode>/graphs/`:**
1. `actual_vs_predicted_<component>.png` — one per component: actual vs predicted over time for run 8.
2. `fidelity_<component>.png` — one per component: `F_c` over time, threshold line, trigger markers, adaptation markers (accepted = green, rejected = red).
3. `unified_fidelity.png` — `F_U` over time, with the same markers.

**Comparison → `baselines/results/comparison/graphs/`:**

4. `unified_fidelity_all_modes.png` — `F_U` over time for all four modes on one axis.
5. `mean_fidelity_by_mode.png` — bar chart, with error bars for full_system.
6. `recovery_time_by_mode.png` — bar chart.
7. `adaptations_by_mode.png` — stacked bars, accepted vs. rejected.
8. `strategy_distribution.png` — recalibrate / regenerate / expand_scope counts for rule_based vs full_system.

**Model evaluation → `reports/model_eval/graphs/`:**

9. `model_vs_baselines_<component>.png` — RMSE of model vs persistence vs 5-s moving average per CV fold, one per component.

---

## 6. Readable outputs for every run

Each mode run writes to `baselines/results/<mode>/`:

**`metrics_timeline.csv`** — one row per second per component:
`ts, component, actual, predicted, abs_error, F_c, F_U, trigger_fired, model_version`

**`adaptation_events.jsonl`** — one record per trigger, containing:
- trigger time, component, trigger type, severity
- fidelity at trigger time
- decision: strategy, confidence, LLM or fallback
- model id, tokens and cost of the LLM calls for this event
- RCA and rationale, and the RAG references used
- candidate version and sandbox result
- `F_c` before → after, with the gate result (accepted / rejected) and the reason
- versions before → after, and duration

**`report.md`** — a human-readable summary:
- the header: mode, run replayed, seeds, model versions, fidelity config
- a summary table: per component, mean actual, mean predicted, RMSE, MAE, mean `F_c`
- an **Adaptation Events** section with one block per event, laid out like this:

```
### Event 3 — dl_throughput — 14:32:07 UTC
Trigger:      fidelity_degradation, severity 0.42 (F_U = 0.61 < θ = 0.70)
Decision:     regenerate (confidence 0.78, source: LLM)
Root cause:   <RCA text>
Reasoning:    <rationale text>
References:   <RAG doc ids>
Candidate:    dl_throughput-v3 — sandbox: passed
Verification: F_c 0.58 → 0.74 (margin 0.05) → ACCEPTED
```

**Console output** while a run is in progress uses `rich` if it is installed, otherwise plain aligned text:
- a compact live line per 10 s: time, `F_U`, and the lowest component
- a clearly framed block whenever an adaptation event happens

The values in the example block above are only a format example. Every real value must come from the run.

---

## 7. `docs/MODELS.md` — models inside the digital twin

One section per component (7), each containing:
- **Purpose:** what network behaviour it represents.
- **Target KPI:** O-RAN name, display unit, raw unit, and the OAI source (CSV file + column, and the original database table/view), including any unit conversion.
- **Inputs:** the feature list grouped by source file, and *why* each group is relevant (e.g. CQI/SNR/MCS → link quality drives BLER and throughput).
- **ML model:** algorithm (XGBoost residual), the final hyperparameters from CV, and why this choice (tabular data, small dataset, fast retraining for recalibration, residual over persistence).
- **Dependencies:** upstream components, and why.
- **Results:** the actual CV numbers vs persistence and moving average, copied from `model_eval.csv`, not retyped.
- **Limitations:** e.g. radio-level BLER, not end-to-end loss; RLC queueing delay, not end-to-end latency; UL SNR only; training on 3 runs.

Also include:
- the dependency graph as a Mermaid diagram
- a short "KPIs available but not modelled" section, listing the Expand-Scope candidates

---

## 8. Fix the open issues from Phase 12

1. **989 MB `rankpi_kpi/` in the repo root:** do NOT delete it. Compare it with `/home/harie06/oai_db/data/rankpi_kpi` (file list, sizes, sha256 of a sample) and report the result. Keep it git-ignored. If it is a duplicate, recommend moving it out, but do not move it yourself.
2. **`docs/architecture_diagram.drawio`:** update it to the OAI architecture: OAI replay source, the 7 components and their graph. Remove the NS-3 elements.
3. **`docs/FO_Twin_Paper.docx`:** do not edit it. Add a note in `docs/README.md` that it is superseded by the LaTeX paper.
4. **Shutdown race in the `dt-continuous-sync` thread:** make shutdown deterministic, with a stop event, a join with timeout, and no logging after teardown. Add a test that starts and stops live mode 5 times without errors.
5. **Missing pre-migration `data/models/` artifacts:** state in `archive/ns3_sim/pre_oai_migration_data/README.md` that they were not preserved and cannot be recovered. No other action.
6. **`tests/e2e/` is empty:** add one end-to-end test, replay a short slice of run 7 at `speed=0` through the whole lifecycle in `rule_based` mode with a mocked LLM client (no real API calls, no cost), and assert that the outputs in §6 are produced.

---

## 9. Final report

```
MODEL FIX:     per component — model vs persistence vs MA (CV mean), beats persistence? (yes/no)
LOCK:          implemented + tests
LLM:           model, tier, caching active (yes/no), total cost per mode, budget used
BASELINES:     table of comparison metrics per mode (from comparison.csv); any invalid runs and why
GRAPHS:        list of generated files
OUTPUTS:       paths to report.md / metrics_timeline.csv / adaptation_events.jsonl per mode
DOCS:          MODELS.md, updated diagram, docs/README.md note
OPEN ISSUES:   each Phase-12 item → fixed / reported / not fixed (why)
TESTS:         commands + pass/fail counts
```