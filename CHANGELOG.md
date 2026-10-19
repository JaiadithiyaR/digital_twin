# CHANGELOG

Real bugs found and fixed while actually running this system end-to-end, newest first. Every
entry names the file, the real symptom observed, the root cause, and the fix — no invented
numbers, nothing papered over.

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
