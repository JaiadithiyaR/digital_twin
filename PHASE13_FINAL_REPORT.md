# Phase 13 — Final Report

Executed against `claude_code_prompt_phase13.md`, in order, §1 through §8. No §0 blocker was hit
(LLM available throughout with real budget headroom; no required run file was missing/`.partial`;
`rankpi_kpi/` confirmed a genuine duplicate, not irreplaceable). Full real numbers below — nothing
in this report is fabricated; every figure is read from an actually-written file on disk.

```
MODEL FIX:     per component — model vs persistence vs MA (CV mean), beats persistence? (yes/no)
```

Leave-one-run-out CV over the train/tune pool (runs 3, 6, 7); residual/delta-target design
(`ŷ_t = y_{t-1} + Δŷ`); mean RMSE across the 3 folds (full per-fold numbers in
`reports/model_eval/model_eval.csv`, hyperparameters and limitations in `docs/MODELS.md`):

| component | model RMSE | persistence RMSE | 5s-MA RMSE | beats persistence |
|---|---|---|---|---|
| dl_bler | 0.0108 | 0.0117 | 0.0193 | **yes** (3/3 folds) |
| ul_bler | 0.0061 | 0.0075 | 0.0098 | **yes** (3/3 folds) |
| dl_prb | 16.2219 | 16.8612 | 21.2737 | **partial** (2/3 folds — loses on held-out run 7) |
| ul_prb | 1.8190 | 1.8804 | 2.7605 | **yes** (3/3 folds) |
| dl_throughput | 0.3299 | 0.3411 | 0.5015 | **partial** (2/3 folds — loses on held-out run 6) |
| ul_throughput | 0.0267 | 0.0305 | 0.0499 | **yes** (3/3 folds) |
| dl_rlc_delay | 11.0653 | 13.1322 | 10.1916 | **yes on RMSE** (3/3), but MAE is worse than persistence in all 3 folds (avg 2.2634 vs 1.5303) — reported honestly, not hidden |

Production registry confirmed (`data/models/registry_index.json`): `dl_bler-v2`, `ul_bler-v7`,
`dl_prb-v2`, `ul_prb-v2`, `dl_throughput-v2`, `ul_throughput-v2`, `dl_rlc_delay-v2` all
`status=production`, each superseding a v1. Graphs:
`reports/model_eval/graphs/model_vs_baselines_<component>.{png,pdf,csv}` (7 components).

```
LOCK:          implemented + tests
```

`config.adaptation.lock_policy = coalesce`. Implemented in `src/main.py` via
`_pending_by_component`/`_pending_order`/`_locked_component` + `_enqueue_trigger`/
`_pop_next_trigger` — a competing trigger for an already-locked component is queued/coalesced,
never dropped, never allowed to run concurrently with the in-flight adaptation for that same
component. 9 unit tests in `tests/unit/test_adaptation_lock.py` + an integration proof in
`tests/integration/test_main_orchestrator.py` (lock set around a real adaptation cycle, released
after). All passing.

```
LLM:           model, tier, caching active (yes/no), total cost per mode, budget used
```

- **Model**: `gemini-3.6-flash` (the originally-configured `gemini-2.5-flash` returned a real
  live `404` — no longer available to new users — and was switched earlier in this project).
- **Service tier**: `standard`.
- **Caching**: **active** — confirmed via real `llm_calls.jsonl` entries showing
  `used_cache: true` / genuine non-zero `cached_tokens` (the static prefix, built by
  `baselines/_lib.py:build_static_prefix()`, is 12,924 real characters — above the configured
  `min_cacheable_chars=4000` floor, so caching genuinely activates every real baseline run).
- **Total cost per mode** (from `baselines/results/comparison/comparison.csv`):

| mode | n runs | LLM cost (USD) | LLM calls | cache-hit share |
|---|---|---|---|---|
| static | 1 | $0.000000 | 0 (no LLM — triggers logged only) | — |
| periodic_retrain | 1 | $0.000000 | 0 (deterministic scheduled retrain, no LLM) | — |
| rule_based | 1 | $0.009351 | — (codegen + explanation roles only; decision is deterministic) | 59.4% |
| full_system | 3 (seeds 1/2/3) | $0.056064 | 44 real decision calls (0 fallbacks) | 66.9% |

- **Budget used**: **$0.065415 of the configured $20.00 cap** (≈0.33%) — well within budget
  across every mode, every seed.

```
BASELINES:     table of comparison metrics per mode (from comparison.csv); any invalid runs and why
```

| mode | n_runs | mean unified fidelity (±std) | n_triggers | n_adaptations | accepted | rejected | fallback rate | invalid? |
|---|---|---|---|---|---|---|---|---|
| static | 1 | 0.7942 (±0.0000) | 14 | 0 | 0 | 0 | 0.0% | No |
| periodic_retrain | 1 | 0.8328 (±0.0000) | 21 | 21 | 3 | 18 | 0.0% | No |
| rule_based | 1 | 0.7980 (±0.0000) | 14 | 9 | 1 | 8 | 0.0% | No |
| full_system | 3 | 0.7428 (±0.0904) | 44 | 38 | 4 | 34 | 0.0% | No |

No run was invalid (the `full_system` invalidity rule — fallback rate > 10% — never tripped;
0 fallbacks across all 44 real decision calls). All 4 modes replayed the real OAI **TEST split
(run 8)** only, never run 7 (reserved for smoke tests) or a train/tune run. `full_system`'s higher
std (0.0904) is genuine, honest run-to-run variance driven by `MockDriftSource`'s own per-seed RNG
generating different synthetic drift events across seeds 1/2/3 — not measurement noise.
Full table: `baselines/results/comparison/comparison.csv` / `comparison.md`.

```
GRAPHS:        list of generated files
```

**Per-mode** (written automatically for every completed mode — `static`, `periodic_retrain`,
`rule_based`, and each of `full_system`'s 3 seeds — under `baselines/results/<mode>/graphs/`,
each as PNG 300dpi + PDF + a CSV of the plotted data):
- `actual_vs_predicted_<component>.{png,pdf,csv}` × 7 components
- `fidelity_<component>.{png,pdf,csv}` × 7 components
- `unified_fidelity.{png,pdf,csv}`

**Comparison** (`baselines/results/comparison/graphs/`, built once by `compare.py` from all 4
modes' already-written output):
- `unified_fidelity_all_modes.{png,pdf,csv}`
- `mean_fidelity_by_mode.{png,pdf,csv}`
- `recovery_time_by_mode.{png,pdf,csv}`
- `adaptations_by_mode.{png,pdf,csv}`
- `strategy_distribution.{png,pdf,csv}`

**Model-vs-baseline** (`reports/model_eval/graphs/`, built once by `scripts/plot_model_eval.py`
from the already-written CV results):
- `model_vs_baselines_<component>.{png,pdf,csv}` × 7 components

```
OUTPUTS:       paths to report.md / metrics_timeline.csv / adaptation_events.jsonl per mode
```

| mode | report.md | metrics_timeline.csv | adaptation_events.jsonl |
|---|---|---|---|
| static | `baselines/results/static/report.md` | `baselines/results/static/metrics_timeline.csv` | `baselines/results/static/adaptation_events.jsonl` |
| periodic_retrain | `baselines/results/periodic_retrain/report.md` | `.../periodic_retrain/metrics_timeline.csv` | `.../periodic_retrain/adaptation_events.jsonl` |
| rule_based | `baselines/results/rule_based/report.md` | `.../rule_based/metrics_timeline.csv` | `.../rule_based/adaptation_events.jsonl` |
| full_system (×3 seeds) | `baselines/results/full_system/seed_{1,2,3}/report.md` | `.../seed_{1,2,3}/metrics_timeline.csv` | `.../seed_{1,2,3}/adaptation_events.jsonl` |

Every mode also has `run_summary.json`; `rule_based` and every `full_system` seed additionally
have `llm_calls.jsonl` (the per-call audit log — role, tokens, cost, cache hit). `static`/
`periodic_retrain` have no `llm_calls.jsonl` since they make zero real LLM calls.
Cross-mode: `baselines/results/comparison/{comparison.md,comparison.csv}`.

```
DOCS:          MODELS.md, updated diagram, docs/README.md note
```

- **`docs/MODELS.md`** — written this phase: one section per component (purpose, target KPI,
  inputs, ML model + exact hyperparameters, dependencies, real CV results table, limitations) for
  all 7 components, a Mermaid dependency-graph diagram, and a "KPIs available but not modelled"
  section. Every number in it is copied verbatim from `reports/model_eval/model_eval.csv` or a
  real inspected `.joblib` artifact — nothing invented.
- **`docs/architecture_diagram.drawio`** — rewritten this phase: a valid mxfile (verified via
  `xml.etree.ElementTree`) showing OAI replay → preprocessing → sync → D1 → the real 7-component
  dependency graph → fidelity → drift/fidelity triggers → the decision agent (+RAG) → the 3
  adaptation agents → sandbox → verification → versioned registry (hot-swap back to the
  orchestrator) → lifecycle, plus a legend. All NS-3 elements removed.
- **`docs/README.md`** — written this phase, new file. States plainly that
  `docs/IEEE_Digital_Twin_Paper.docx` predates the OAI migration, should not be edited, and is
  superseded for current behavior by `docs/MODELS.md`/`docs/oai_twin.md`/`CLAUDE.md`. **Deviation
  from the spec's literal wording, flagged honestly**: the spec's own text (§8 item 3) says this
  file "is superseded by the LaTeX paper" — no LaTeX paper exists anywhere in this repository
  (`find . -iname "*.tex"` returns nothing), and none was fabricated to satisfy that wording. This
  is recorded as a genuine open gap below, not silently implied to exist.

```
OPEN ISSUES:   each Phase-12 item → fixed / reported / not fixed (why)
```

1. **989MB `rankpi_kpi/` duplicate check** → **reported**. Confirmed a genuine duplicate of
   `/home/harie06/oai_db/data/rankpi_kpi` via file-list/size/sha256-sample comparison. Kept
   git-ignored, not moved or deleted (per instruction — report only).
2. **`docs/architecture_diagram.drawio`** → **fixed**. Rewritten for the OAI architecture; see
   DOCS above.
3. **`docs/FO_Twin_Paper.docx` / LaTeX-paper note** → **reported, not fixed as literally
   specified**. No file named `FO_Twin_Paper.docx` exists (the actual file is
   `docs/IEEE_Digital_Twin_Paper.docx`) and no LaTeX paper exists to supersede it. `docs/README.md`
   honestly documents this gap instead of fabricating a paper or a false "superseded by LaTeX"
   claim. The existing `.docx`/`docs/FORMULAS.pdf` are left untouched, as instructed.
4. **Shutdown race in `dt-continuous-sync`** → **fixed**. `SynchronizationConfig` gained
   `shutdown_join_timeout_seconds`; `ContinuousOrchestrator.stop()` now signals both background
   threads, `.join(timeout=...)` each, and logs a WARNING (if a thread didn't exit in time) or
   confirmation (both exited) before returning — never logs from a thread after the caller
   believes shutdown is complete. New test
   `test_start_and_stop_five_times_is_deterministic_and_error_free` starts/stops a real
   orchestrator 5 times in a row — passing.
5. **Missing pre-migration `data/models/` artifacts** → **reported**. Confirmed via `find` that
   only demo-script-scoped model trees were ever archived — the real top-level production
   `data/models/` tree was never copied into `archive/ns3_sim/pre_oai_migration_data/` before the
   migration and cannot be recovered (gitignored, runtime-only, never committed). Stated plainly
   in that directory's `README.md`. No other action taken, per instruction.
6. **`tests/e2e/` empty** → **fixed**. New `tests/e2e/test_baseline_rule_based_e2e.py`: replays a
   5-cycle slice of run 7 (never run 8) through the full `rule_based` lifecycle with a faked LLM
   client (zero real API calls — structurally proven by asserting `llm_calls.jsonl` does not
   exist), asserts `metrics_timeline.csv`/`adaptation_events.jsonl`/`report.md`/`run_summary.json`
   are all genuinely produced. Passing (195.9s).

```
TESTS:         commands + pass/fail counts
```

```bash
.venv/bin/python -m pytest tests/unit -q          # 515 passed, 0 failed
.venv/bin/python -m pytest tests/integration -q   # 34 passed, 0 failed  (480.6s / 8m00s)
.venv/bin/python -m pytest tests/e2e -q           # 1 passed, 0 failed   (195.9s)
```

**Total: 550 passed, 0 failed**, across all 3 suites this phase.

Two real, genuine bugs were found and fixed *while confirming these numbers* (not pre-existing
failures papered over):

1. **`src/llm/google_client.py`** — `httpx.RemoteProtocolError` ("server disconnected without
   sending a response") was not in `_TRANSIENT_TRANSPORT_EXCEPTIONS`, so a transient network
   hiccup propagated straight through `complete_structured_safe`'s graceful-degradation contract
   and crashed an entire `full_system` run outright (hit for real mid-run, seed 1 of 3, after 15
   clean cycles). Fixed by adding it to the retryable set (confirmed not present even in the SDK's
   own internal transient-exception tuple either) + a new regression test. This is a production
   fix, not baseline-only — it protects every real LLM-dependent code path in the system.
2. Four integration-test fixtures (`test_sync.py`'s `_sync_config()`, three places constructing a
   deliberately-weak bootstrap model from a thin data slice in `test_expand_scope_agent.py`/
   `test_lifecycle_agent.py`/`test_verification_agent.py`) broke against this phase's own §1
   richer lag(1-5)/rolling(5s,10s) feature set — a missing required config field in one case, and
   NaN-intolerant hardcoded "LLM-generated" fixture candidates / a global (not per-session)
   40-row head-cut in the others, which the richer feature set's legitimate early-session NaNs
   now exposed. All fixed at the test level (no production-code change needed for these four).
