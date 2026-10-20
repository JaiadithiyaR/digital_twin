# CHANGELOG

Real bugs found and fixed while actually running this system end-to-end — and, since
2026-10-19, design changes too — newest first. Every entry names the files, the real symptom or
motivation, the root cause, and the fix — no invented numbers, nothing papered over.

---

## 2026-10-20 — Public release preparation

- **Runnable from a fresh clone.** `data/processed/oai/` (the processed OAI testbed dataset)
  and all ablation results (`baselines/results/`, `figures/`, `reports/`) are now committed.
- **Raw-export path.** The machine-specific `data.oai_root` path is gone from `settings.yaml`.
  It now defaults to `data/raw/rankpi_kpi`, can be overridden with `OAI_ROOT` in `.env`, and
  `build_oai_dataset.py` explains clearly when the raw export is absent.
- **Removed from the published files:** personal machine paths from docs and logs (run logs
  copied to `baselines/results/run_logs/` with repo-relative paths).
- **Not published (gitignored):** the pre-audit leaky results, live-run state and logs,
  the per-run D1 replay copies, and the unpublished manuscript draft.
- **Licensing and packaging:** MIT `LICENSE`; `pyproject.toml` now has its own lint rules and an
  up-to-date description; `requirements-lock.txt` pins the exact tested versions;
  `.gitattributes` normalizes line endings.
- **CI and docs:** a GitHub Actions workflow runs ruff and the unit tests. `README.md` was
  rewritten for public users (current architecture diagram, quick start, results), and the
  `manual.md` setup sections were made machine-independent.
- **Cleanup:** 2 real lint issues fixed — an undefined name in a test and a dead variable.
  `scripts/run_baselines.py`, which was broken (it read a removed config key), moved to
  `archive/legacy_scripts/`. `BASELINES.md` and `PHASE13_FINAL_REPORT.md` are marked superseded.
- **Verified:** the exact staged file set was exported to a clean directory and, with no API
  key, ruff, setup, RAG ingest, stats (identical numbers), figures, a baseline smoke run, a live
  run and the unit suite all ran there.

---

## 2026-10-20 — Ablation harness audit, statistics pipeline, IEEE figures

**Motivation:** a publication-grade 4-step ablation (static / periodic / rule-based / proposed)
with 60 s windowing, Wilcoxon, block bootstrap and Holm correction. An audit of `baselines/`
found the existing results could not support that comparison.

**Audit findings (root causes), all fixed:**
- *Test-set leakage* (`baselines/_lib.py`). Run 8 was bulk-replayed into D1 before the loop, and
  agents read `get_history()`. An adaptation at minute 5 therefore trained on minutes 6–61, which
  were later scored. Fix: `CutoffD1View` exposes only rows before the current window's end.
- *Incomparable scores*. Each run's F_c is min-max normalized against its own rolling history, so
  mode-to-mode fidelity numbers were on different scales. Fix: `baselines/stats.py` re-scores every
  window with the same formula against ONE frozen reference (pooled min/max of each squared metric
  over all runs). `src/fidelity` is unchanged.
- *Mock external drift* was still injected into every mode (and was the only thing the
  full-system "seeds" varied). Removed: the live system's per-component policy is the only trigger.
- *Harness ≠ live system*: no retry loop, no reference TRAIN data or recent weighting, no
  persistence baseline for expand-scope verification. All mirrored from `src/main.py` now.
- *Periodic baseline* went through the gate and ran on a cycle count. Now: every 600 s of data,
  deployed unconditionally (blind), configurable (`baselines.periodic_retrain`).
- *15 coarse cycles* replaced by fixed 60 s windows of data time (`baselines.window_seconds`);
  four raw metrics are now recorded per window and component.
- *Units bug* (`baselines/_outputs.py`): throughput/delay targets are already Mbps/ms, but graphs
  divided by 1000 again.

**Live-system bug found by the full run** (`src/dt_models/orchestrator.py`): a hot-swapped
LLM-generated `dl_rlc_delay` listed `dl_throughput_pred`/`dl_bler_pred` in `REQUIRED_FEATURES` as
well as `DEPENDENCIES`. The orchestrator demanded those columns from raw telemetry, so every later
prediction cycle failed: 47 of 62 windows were lost in that run, and the live loop would be
affected the same way. Fix: dependency output fields are wired from the predictions, never
demanded from features. A regression test was added; the affected rule_based run was re-run.

**Added:** `baselines/stats.py` (frozen scoring, Wilcoxon + rank-biserial *r*, moving-block
bootstrap CI, Holm), `scripts/generate_ieee_figures.py` (7 IEEE figures, PDF+PNG+CSV; watermarked
`_SYNTHETIC` placeholders when no measured data exists), `tests/unit/test_baselines_harness.py`,
and `seaborn` in requirements. Old results were moved to `baselines/results_archive_pre_2026-10-20/`.

**Measured (run 8, 62 windows, frozen scale):** Static 0.8196, Rule-based 0.8174, Proposed
0.8155 (3 runs, SD across runs 0.0002), Periodic 0.8122. All pairwise differences are < 0.006 and
Holm-significant. Proposed beats Periodic but is slightly below Static and Rule-based. Run 8 has
traffic only in minutes ~0–18, and every trigger fired at minutes 14–15 (IMPLEMENTATION_STATUS
item 16). 583 unit tests pass.

---

## 2026-10-20 — Adaptation loop (retry until fixed), multithreaded adaptations, reject root causes fixed, expand-scope made possible

**Symptoms** (from a live run): almost every adaptation was REJECTed; an LLM decision sometimes
failed to parse; expand-scope never produced anything; a rejected attempt was simply abandoned
until the component's policy fired again; one slow adaptation held up every other component.

**Root causes, measured** (`scratchpad` experiments on real run-8 windows, production trained on
the 60,858-row TRAIN pool):
1. *Training discarded most of the data.* `ResidualDTComponent.train` dropped every row with ANY
   NaN feature. The first ~1,547 rows of run 8 have no `requested_prb_ratio_lag1` (no controller
   action yet), so early recalibration windows had 0–250 trainable rows — and training no longer
   matched serving, where those rows ARE predicted.
2. *Candidates trained on a few minutes of data cannot beat production trained on 60k rows.* Even
   with NaN rows kept, a recent-window-only candidate lost to production on the same held-out
   rows; TRAIN pool + recent window (recent rows weighted ×5) matched or beat it.
3. *Stale triggers.* A trigger waiting behind a busy component was still executed after the
   component had recovered (`dl_bler`, `dl_rlc_delay`), so an already-healthy production model
   was "adapted" and the candidate predictably rejected.
4. *Truncated LLM output.* With thinking enabled, thinking tokens count against `llm.max_tokens`
   (8000); long code generations were cut off mid-JSON → structured-output validation failure.
5. *Expand-scope was structurally impossible on OAI data* (two independent blockers): D1 held only
   the 7 already-predicted targets plus lagged features, so every design ("wb_cqi", "pusch_snr",
   …) failed validation; and even a valid new component could never pass verification, because the
   live fidelity evaluator has no history for a never-seen component (`insufficient_history` →
   fail-safe REJECT, every time).
6. *Leakage hole in the sandbox hand-off* (found while fixing 5): regenerate/expand-scope passed the
   FULL window to the candidate, including the same-second value of every target — a generated
   model could read its own label and post a falsely perfect score.

**Fixes**:
- `src/dt_models/residual_base.py` — `train(..., sample_weight=None)` drops only rows whose residual
  target is undefined (missing `y_t` or `y_{t-1}`); NaN input features are kept (XGBoost/sklearn
  handle them; serving already does). `sample_weight` is aligned to the kept rows.
- `src/adaptation/data_selection.py` — `with_reference_data()`; `recalibration_agent.py` and
  `regeneration_agent.py` take `reference_history` (the TRAIN pool) on the training side only —
  held-out evaluation stays purely recent. Recalibration weights recent rows
  `adaptation.recalibration.recent_sample_weight` (5.0) when the component's `train` accepts
  `sample_weight`. New config: `adaptation.{recalibration,regeneration}.include_reference_data`.
- `src/main.py` — **adaptation loop**: `_run_adaptation_loop` repeats decision → agent → sandbox →
  verification for one component until a candidate is ACCEPTed, the component meets its threshold
  again, the outcome is not retryable (no LLM, unknown component, not enough data, no new KPI to
  model), the system is stopping, or `adaptation.max_adaptation_attempts` (5) attempts were made.
  Every attempt's rejection summary (strategy, fidelity numbers, failed checks, sandbox/agent error)
  is fed to the next attempt's decision prompt (`DecisionContext.previous_attempts`) and to the
  regenerate/expand-scope agents. Stale triggers are skipped (`outcome: skipped_recovered`).
- `src/main.py` — **multithreaded**: each adaptation loop runs on its own worker thread
  (`dt-adapt-<component>`), at most `adaptation.max_concurrent_adaptations` (2) at once; a component
  is never adapted twice concurrently (active set + coalesced pending triggers). Prediction/hot-swap
  share a serving lock; `FidelityEvaluator` gained an internal lock (formula untouched); decision-log
  writes are idempotent; shutdown waits `adaptation.shutdown_wait_seconds` for workers and records
  any unfinished one as `interrupted`. Removed the old "concurrent components forbidden" validator.
- `config/settings.yaml` — `llm.max_tokens: 32768`; `src/llm/google_client.py` logs
  `finish_reason`/token counts on every structured-output failure (diagnosable truncation).
- **Expand-scope** — `src/data/oai_dataset.py`: `OBSERVABLE_KPI_COLUMNS` (8 same-second MAC KPIs:
  `dl_curr_tbs_avg`, `dl_sched_rb_avg`, `pusch_snr_avg`, `pucch_snr_avg`, `wb_cqi_avg`,
  `dl_mcs1_avg`, `ul_mcs1_avg`, `phr_avg`) carried through dataset → replay → preprocessing →
  `CleanTelemetryRecord.observables` → D1 as ground truth a NEW component may predict — never a
  feature (`feature_columns()` excludes them). `expand_scope_agent.py`: the target must be an
  unpredicted same-second KPI with data; inputs may never be same-second values; the prompt lists
  both explicitly and states the acceptance rule. `verification_agent.py`: a component with no
  production version is verified against persistence `y_{t-1}` (`baseline_predictions`), both scored
  against one shared, isolated normalization reference — same `new > old + delta` gate, same
  fidelity formula (unchanged).
- `src/sandbox/executor.py` — every same-second measured column (7 targets + 8 observables) is
  stripped from the features a candidate receives; its label still arrives separately.
- `scripts/test_adaptation_agents.py` (new) — runs each adaptation agent individually on real OAI
  data (run 7) through the real orchestrator loop, strategy pinned (`decision_source: pinned`).
- Dataset rebuilt (`build_oai_dataset.py`): row counts unchanged (train 60,858 / test 25,865), 8
  observable columns, 0% NaN, 0 leaked into features.

- Console: `[verify]`/`[dependents]`/`[lifecycle]` lines now name their component (two components'
  steps interleave under concurrency); a pinned decision prints `(PINNED)`, never `(LLM)`.

**Fidelity formula**: unchanged (`evaluate()`/`compute_unified_score()` untouched; the
hand-computed regression test still passes).

**Verified**:
- Tests: 576 unit + 35 integration/e2e passed, 0 failed.
- Agent tests (`scripts/test_adaptation_agents.py`, real run-7 data):
  - recalibrate `dl_bler-v2` and regenerate `ul_bler-v2` each produced and verified a real candidate (REJECT: production already better or within delta).
  - expand-scope — previously 10/10 designs rejected, 0 candidates — designed a new `dl_curr_tbs` component that passed the sandbox and was verified against persistence. REJECT, genuinely: run 7's traffic collapses in the held-out tail (KPI mean 96 → 0.04).
- Final live run (15 min, fresh D1, run 8 at 10×):
  - 25,865 rows, 89 cycles, 7 triggers, 5 attempts → 2 ACCEPT + hot swap (`dl_bler-v13`, `ul_bler-v15`), 3 REJECT, 2 stale skips.
  - 1 retry loop (`dl_prb` recalibrate → regenerate, using the feedback).
  - Concurrent pairs `dl_bler`+`dl_prb`, `dl_prb`+`ul_bler`.
  - 0 warnings/errors.
  - The previous run, before these fixes: 5/5 REJECT.
- Ctrl-C mid-adaptation shuts down cleanly. Note: a plain `timeout -s INT` launched from a non-interactive WSL script never delivers SIGINT to the child — not even to a 3-line `time.sleep` script — so use `timeout --foreground` for scripted runs. A real terminal Ctrl-C is unaffected.

---

## 2026-10-19 — External drift permanently disabled in the live system

**Change**: the live orchestrator adapts only on the internal per-component fidelity policy.
`src/main.py`: removed the mock drift source/detector construction, the `dt-drift-consumer`
thread, the `ContinuousOrchestrator(drift_severity_range_override=…, enable_external_drift=…)`
parameters and the `--enable-external-drift` CLI flag (no way remains to turn it back on);
`--max-drift-events` renamed `--max-adaptations` (old name kept as an alias). Validation scripts
and the orchestrator integration test that relied on the drift stream to get an adaptation now
inject one clearly-labelled internal `fidelity_degradation` trigger (`metadata.forced=True`) when
no component's policy fires during warm-up — the pattern `run_oai_e2e_smoke.py` already used.
Module 11 (`src/drift/`) is unchanged and still used by the `baselines/` harness.

---

## 2026-10-19 — Adaptation policy: unified fidelity score → per-component fidelity policy

**Motivation**: adaptation was decided from ONE number. `FidelityEvaluator.check_fidelity_trigger()`
kept a single debounce counter on the Unified Fidelity Score `F_U` (weighted mean of all
`FidelityScore_c`) and, after 3 cycles with `F_U < 0.5`, blamed the `argmin` component. A badly
degraded component could not be adapted while the others kept the mean above 0.5, and a
component's counter could advance on cycles where it wasn't even the worst one.

**Formula**: unchanged. `FidelityScore_c` was already defined and computed per component
(`evaluate()`, per-(component, metric) rolling windows); `evaluate()` and
`compute_unified_score()` are byte-for-byte unchanged and the hand-computed regression test
(`test_composite_score_hand_computed_case`) still passes.

**Change**:
- `src/fidelity/component_policy.py` (new) — `ComponentAdaptationPolicy`: one independent debounce
  counter per component; component c is adapted when its own FRESH `FidelityScore_c < threshold_c`
  for `min_consecutive_c` cycles. Same semantics as before (reset on recovery, reset after firing,
  `insufficient_history` never fires, severity `clip((T − F)/T, 0, 1)`), plus: no fresh
  measurement this cycle → `unavailable`, counter reset (never read as good or bad).
- `src/common/config.py`, `config/settings.yaml` — `trigger_threshold` /
  `trigger_min_consecutive_evaluations` are now the per-component defaults;
  `fidelity.component_policies` overrides them per component id. New `logging.*` log paths.
- `src/fidelity/evaluator.py` — `check_fidelity_trigger()` and its global counter removed; `F_U`
  is reporting-only.
- `src/main.py` — the fidelity cycle evaluates the policy for every component and enqueues one
  trigger per firing component (existing per-component coalesce/lock unchanged). After each
  adaptation: the component is re-measured on the latest window (read-only, shared normalization
  reference) and reported as meeting/still below its threshold; after a hot swap its transitive
  downstream dependents are revalidated (measured + reported, never auto-adapted). Every handled
  trigger — including skips and agent/verification failures — is logged.
- `src/adaptation/lifecycle_agent.py` — `LifecycleRecord` gains `component_threshold`,
  `fidelity_at_trigger`, `production_version_after`, `post_adaptation_fidelity`,
  `remaining_fidelity_gap`, `still_below_threshold`, `dependency_revalidation` (all optional, so
  older records still parse); the maintenance report has matching sections.
- `src/adaptation/decision_{context,agent}.py` — the decision prompt carries the component's own
  threshold and labels `F_U` as a summary, not a trigger.
- `src/common/console_report.py`, `src/common/run_logs.py` (new) — detailed live output (per-cycle
  per-component table; one line per adaptation step), a per-run console transcript, and two
  append-only logs: `logs/component_fidelity.csv`, `logs/adaptation_decisions.jsonl`.
- `baselines/_lib.py`, `baselines/run_all.py` — the baseline harness uses the same policy; every
  firing component is dispatched (coalesced per component with that cycle's external event). An
  unused, duplicated RAG query per `full_system` decision was removed. **Baseline results already
  under `baselines/results/` were produced with the unified-score trigger** — re-run before
  comparing against new runs.

**Verification**: new `tests/unit/test_component_policy.py` and
`tests/unit/test_per_component_orchestration.py`; unified-trigger tests replaced; integration tests
assert the new lifecycle fields and decision log. 555 unit + 35 integration/e2e passed. Real live
runs: policy-only — `ul_bler` adapted on its own `F_c` (0.36) while `F_U` was 0.72; with external
drift — 17 verified adaptations, 2 ACCEPT hot swaps each followed by dependency revalidation, all
failures/interruptions logged, 0 errors (details: CLAUDE.md §12).

**Found while validating**: a Ctrl-C during an in-flight adaptation left no decision-log entry
(`KeyboardInterrupt` bypassed the logging path). `_handle_trigger` now records `interrupted` /
`error` for any abnormal end before re-raising; covered by a unit test and confirmed in the final
live run.

---

## 2026-10-19 — Live replay stalled: every 20-row sync batch rewrote the whole D1 history file

**Symptom**: `--mode live --replay-speed 10` ingested ~16 rows/s instead of ~70, and slowed further
as history grew. **Root cause**: `D1Store` re-serialized the entire history + current-state
Parquet on every batch; on a WSL `/mnt/c` mount each rewrite took 0.6–1.0 s (in-memory work: ms),
and the replay generator only advances when the synchronizer pulls. **Fix**
(`src/dt_models/d1_model_store.py`, `storage.persist_min_interval_seconds: 10`): in-memory state is
updated immediately; disk writes are rate-limited; `flush()` on shutdown. **Verified**: ~58 rows/s,
full run 8 (25,865 rows) ingested in a 15-minute live run.

## 2026-10-19 — Restarted live replay duplicated D1 history and silently idled

**Symptom**: a second `--mode live` run re-appended every tick already in D1 and, since no tick
was newer than D1's latest `ts_1s`, reported "replay fully consumed" while still replaying.
**Fix**: `append_history` dedupes on `(run_id, session_id, ts_1s)`; `src/main.py` seeks the replay
past D1's last persisted tick for those runs (delete `data/artifacts/d1_*.parquet` to start over).

## 2026-10-19 — Console labels swallowed / MarkupError crash; continuous loop killed by one exception

**Symptoms**: rich parsed `[fidelity]`/`[trigger]`/`[warning]` prefixes as markup (silently
dropped) and raised `MarkupError` on exception text containing `[/x]`; any unexpected exception in
a prediction or adaptation cycle (e.g. MK-MMD on a 1-row window) terminated `run()`; Ctrl-C ended
in a traceback; initialization logs bypassed the log file. **Fixes** (`console_report.py`,
`src/main.py`): `markup=False`; `run()` logs and continues; `FidelityComputationError` caught per
component; `KeyboardInterrupt` is a clean shutdown; `setup_logging()` runs before `initialize()`.

## 2026-10-19 — Verification rejected or crashed on legitimately-NaN rows; expand-scope target leakage

**Symptoms**: residual components predict NaN where `y_{t−1}` is unknown, so `verify()` REJECTed
such candidates (`no_invalid_output`) or crashed on a NaN target, leaving the candidate stuck in
`candidate` status; an expand-scope design could list its own target as an input (near-perfect,
auto-accepted fidelity) or reuse an existing component's output field. **Fixes**
(`verification_agent.py`, `expand_scope_agent.py`, `src/main.py`): rows without ground truth or a
production prediction are excluded for both models; the candidate must be finite on every scored
row; a verification exception rejects the candidate; design validation rejects target leakage and
output-field collisions. Also: `MockDriftSource`'s 30 s sleep is interruptible (clean `stop()`),
and `config/settings.yaml`'s `oai_root` must be a WSL path (a `"C:\Users\..."` YAML string fails to
parse: `\U` is a YAML escape).

---

## Live system: `--mode live` with no extra flags could never produce a defined fidelity score

**Symptom**: running the documented command exactly as written — `python -m src.main --mode
live`, no extra flags — printed one `insufficient_history` line, then immediately "no new
telemetry ... idling", forever. Looked like the earlier idle-notice fix had made it *worse*, not
better.

**Root cause**: a finite OAI replay at the config default `speed=0` ("as fast as possible") loads
its entire split before the *first* 5-second prediction cycle even fires. The fidelity engine
needs `config.fidelity.min_history_for_normalization` (10) separate historical evaluation points
before it will report a real score — normally built up over several real ticks against
slowly-growing live telemetry. Here, the first (and only) cycle appended exactly one point, and
every cycle after that correctly detected "no new telemetry since the last cycle" (the earlier
fix for the false-`1.0000`-pinning bug) and skipped re-evaluating. The two correct-on-their-own
behaviors combined into a dead end: under the system's own default configuration, fidelity could
*never* become defined and a fidelity-based trigger could *never* fire — not a hang, but a
structural no-op.

**Fix**: `src/fidelity/evaluator.py` gained `FidelityEvaluator.history_length(component)` (reads
how many points a component's rolling window has already accumulated). `src/main.py`'s
`run_prediction_and_fidelity_cycle()` now checks, every cycle, whether any component is still
short of `min_history_for_normalization` — and if enough already-replayed real history exists
beyond the usual single most-recent window, it backfills by evaluating several earlier
`window_size` chunks too (oldest first), in the same burst, instead of only the final window.
This is exactly what several real ticks against slowly-arriving telemetry would have produced
anyway — just computed immediately from data that's already there, rather than waiting on ticks
that will never come once a finite replay is already fully loaded. Live/slow-growing telemetry is
unaffected: the backfill deficit is 0 once a component is already warmed, so behavior there is
unchanged. Verified with your exact command, fresh state: the very first line now reads
`F_U=0.5471 lowest=dl_bler defined=7` — a real, fully-defined unified fidelity score, not
`insufficient_history`. Full unit + integration suites re-run clean after the fix.

---

## Live system: console went silent once replay finished (looked hung)

**Symptom**: `python -m src.main --mode live` printed one status line, then nothing for the rest
of a 7-minute run — looked exactly like a hang, even though the system was healthy.

**Root cause**: a finite OAI replay source (`speed=0`) loads its entire split into D1 almost
instantly. Once no new telemetry arrives, `run_prediction_and_fidelity_cycle()` correctly skips
re-evaluating (see the next entry) — but it skipped *silently*, every single cycle, for the rest
of the run. Nothing was wrong; the console just never said so.

**Fix**: `src/main.py` now prints one clear, one-time notice ("no new telemetry... the system is
alive and idling") the first time this happens, then stays quiet — never a hang, never spam.

---

## Live system: fidelity falsely pinned at a perfect `1.0000` forever

**Symptom**: after disabling external drift (below), fidelity stopped reading
`insufficient_history` — but then sat at exactly `F_U=1.0000`, never moving, for the rest of the
run. Too perfect to be real.

**Root cause**: once a finite OAI replay (run 8, 25,865 rows) fully loads, D1 stops growing.
`run_prediction_and_fidelity_cycle()` kept re-evaluating the *identical* `(y_true, y_pred)` pair
every timer tick. The fidelity formula's own min-max normalization window collapsed to zero
variance (every entry became the same value), and the current point trivially normalized to a
perfect score — a real measurement of nothing, not a genuine fidelity reading.

**Fix**: `src/main.py` now tracks the latest D1 timestamp it has actually evaluated and skips the
cycle when nothing new has arrived, instead of manufacturing a repeated, meaningless reading.
Verified with real data: fidelity now reads `0.7955 → 0.7993 → 0.8029 → ...`, genuinely varying.

---

## Live system: `--mode live` took "forever" to produce any output

**Symptom**: starting the live system and watching it sit at the startup banner for a very long
time with no progress.

**Root cause**: `D1Store.append_history()` rewrites the *entire* accumulated history file to disk
on every flush (never an incremental append) — fine at the live-telemetry default batch size of
20 for genuine telemetry trickling in in real time, but an O(n²) blowup for a finite,
`speed=0` ("as fast as possible") replay: the default test split is 25,865 rows, so batch
size 20 meant ~1,293 full-file rewrites.

**Fix**: `src/main.py` now uses a batch size of 500 specifically when replaying a finite OAI split
at `speed=0` — the same override already proven safe in `baselines/` and the smoke test. Genuine
live/continuous telemetry sources are completely untouched. Verified: real, varying fidelity
scores now appear within 30–40 seconds instead of an unbounded wait.

---

## Live system: only external drift ever triggered adaptation, never fidelity

**Symptom**: every adaptation event was `trigger_type=external_drift`; the internal fidelity
trigger never fired, and fidelity stayed stuck at `insufficient_history`.

**Root cause**: the mock external-drift source fired roughly every 20–30 seconds, and each
adaptation attempt (recalibrate/regenerate) takes 10–35 seconds on the *same* single-threaded
loop that also runs the fidelity cycle — starving the fidelity evaluator of the 10 consecutive
clean cycles it needs to warm up.

**Fix**: external drift is now **off by default** for the live/demo system. New
`ContinuousOrchestrator(..., enable_external_drift=True)` constructor flag (default `True` so
nothing else changes) and a `--enable-external-drift` CLI flag to opt back in. `python -m
src.main` now adapts purely from its own internal fidelity trigger, as requested.

---

## `baselines/_lib.py`: `expand_scope` unconditionally broken

**Symptom**: every real selection of the `expand_scope` strategy inside a `full_system` baseline
run failed with `TypeError: expand_scope() missing 1 required positional argument:
'example_component_factory'`.

**Root cause**: the harness's `expand_scope` dispatch never passed that required argument.

**Fix**: pass the already-built `factory` callable. Verified: `full_system`'s 3 seeds all
completed with zero crashes after the fix.

---

## `baselines/_lib.py`: recalibration crashed on a previously-regenerated component

**Symptom**: a `recalibrate` attempt on a component that had already been through one real
`regenerate` ACCEPT crashed with `TypeError: HistGradientBoostingRegressor.__init__() got an
unexpected keyword argument 'n_estimators'`.

**Root cause**: the harness's `factory` lambda unconditionally called `cls.from_settings(settings)`
— but a hot-swapped, LLM-generated class only guarantees a no-arg constructor (`src/main.py`
already handles this correctly elsewhere). This specific LLM-authored candidate's own
`from_settings()` blindly forwarded config params shaped for XGBoost into a different estimator.

**Fix**: branch on whether the class is one of the 7 originals before choosing how to construct
it, mirroring `src/main.py`'s existing, correct pattern exactly.

---

## `baselines/_lib.py`: an `OrchestratorError` could crash an entire baseline run

**Symptom**: a hot-swapped `dl_throughput` regenerate candidate crashed the whole `run_all.py`
process mid-run, losing an entire seed's results.

**Root cause**: the candidate declared a dependency's `_pred` column directly in
`REQUIRED_FEATURES` instead of properly via `DEPENDENCIES` — tolerated by the sandbox's own
training harness, but not by the live orchestrator, which only auto-wires `DEPENDENCIES`.
`src/main.py`'s own prediction cycle already guards against exactly this; the baseline harness's
copy of the same function did not.

**Fix**: added the identical `try/except OrchestratorError` guard already present in
`src/main.py`. Known general risk, left as a documented finding, not something fully closeable
here: the sandbox's conformance check never verifies that a candidate's dependency wiring matches
how live serving actually works.

---

## `src/adaptation/recalibration_agent.py`: an all-NaN training window crashed the whole system

**Symptom**: `ValueError: UlBlerModel.train received no valid (non-NaN) rows` propagated all the
way out of `python -m src.main`'s continuous loop and killed the process.

**Root cause**: `recalibrate()` only checks raw row *count* against `min_training_rows` before
training — never whether those rows survive NaN-filtering. A window landing on sessions too young
for the richer lag(1-5)/rolling(5s,10s) features to be populated yet can pass the count check
and still be 100% NaN after feature engineering. `src/main.py`'s own safety net only catches
`(RecalibrationError, RegenerationError, ExpandScopeError)` — not a raw `ValueError`.

**Fix**: `candidate.train()` is now wrapped; any `ValueError` is converted to a clean
`RecalibrationError` with context, caught by the orchestrator's existing handler exactly like any
other recalibration failure — log it, skip the cycle, keep going. Added a regression test
reproducing this exact scenario.

---

## Three standalone agent demo scripts were completely broken since the OAI migration

**Symptom**: `scripts/demo_recalibration.py`, `scripts/demo_regeneration.py`, and
`scripts/demo_expand_scope.py` all failed immediately with `ModuleNotFoundError: No module named
'src.dt_models.throughput'`.

**Root cause**: all three scripts still imported `ThroughputModel`, a component retired during
the Phase-12 OAI migration and replaced by the current 7 real OAI KPI components. They were never
updated — a genuine gap in that migration's own "final consistency sweep".

**Fix**: rewrote all three to use the real `DlBlerModel` in place of `ThroughputModel` (target
column `dl_bler`, feature set `ALL_FEATURES`) — mock telemetry already emits the full OAI schema,
so this was a contained swap, not a redesign. Verified each one for real, end to end:
- **Recalibration**: real candidate trained, fidelity 0.9350→0.9396, correct REJECT (short of the
  required delta).
- **Regeneration**: a genuine LLM-authored `HistGradientBoostingRegressor` candidate, sandbox-
  accepted, fidelity 0.9350→0.9441, correct REJECT (improvement 0.0091, just short of 0.01).
- **Expand-Scope**: two real LLM calls (design + implementation), real sandboxed training, correct
  REJECT via the documented "no prior baseline" path.

---

## `scripts/run_oai_e2e_smoke.py`: stale data from an older run leaked into a fresh one

**Symptom**: the script printed `smoke_test_run=7` but the orchestrator actually replayed run 8.

**Root cause**: `D1Store`'s history is append-only and reloads whatever already exists on disk at
construction. This script's own dedicated storage subtree still had run-8 rows left over from
before the Phase-13 rule changed the smoke-test run from 8 to 7 — never cleared between runs.

**Fix**: the script now wipes its own storage subtree fresh at the start of every run (it already
documented this as "never the shared production paths" — now it's also genuinely fresh every
time). Also relaxed its pass/fail check: an adaptation agent genuinely failing to produce a
candidate at all (e.g. zero valid training rows in a real window) is now reported as the
legitimate, gracefully-handled outcome it is, not a hard failure.

---

## `src/common/console_report.py` / `baselines/_console.py`: buffered output could be lost

**Symptom**: found while diagnosing the "looks hung" issue above — not itself the cause, but a
real, separate robustness gap.

**Root cause**: neither bare `print()` nor `rich.Console.print()` forces a flush when stdout is
redirected to a file (not a terminal). A long-running process whose output is redirected and then
killed before a clean exit (Ctrl-C, `timeout`, a crash) can lose every buffered line that never
made it to disk. The structured JSON logger doesn't have this problem (`logging.StreamHandler`
flushes after every record); this module never went through `logging`.

**Fix**: every console line in both modules now flushes explicitly and immediately.

---

## Console output was dense JSON, unreadable while the system runs

**Symptom**: watching `python -m src.main` live showed a wall of structured JSON log lines —
correct for log aggregation, unreadable for a human.

**Fix**: new `src/common/console_report.py` — clean, human-readable status lines and framed
adaptation-event blocks, printed directly (never through the logger). The console now shows only
WARNING+ JSON (genuine problems are never hidden) plus this new readable channel; the log file
(`logs/digital_twin.log`) still gets everything, full detail, unchanged. `--verbose` restores the
old full-JSON-on-console behavior if ever wanted.
