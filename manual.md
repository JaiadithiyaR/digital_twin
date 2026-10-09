# MANUAL.md — AI-Driven Self-Adaptive Network Digital Twin

This is the single run-and-reference manual for this repository: how to set it up, every command
you can run, and what every folder/file does. It supersedes the old repo-root `manual.txt` (that
file documented the pre-Phase-13 architecture and some retired NS-3 content — kept on disk,
unmodified, purely as a historical artifact; read this file instead). `docs/MANUAL.md` and
`docs/HOW_TO_RUN.md` are earlier, narrower versions written mid-project (post-OAI-migration,
pre-Phase-13) — this file is the current, complete one; where they conflict, this file wins.

The authoritative architecture/design document is `CLAUDE.md` (205KB, extremely detailed —
read it when you need the "why", not just the "what"). `IMPLEMENTATION_STATUS.md` tracks
module-by-module build status. `claude_code_prompt_phase13.md` is the specification this latest
phase of work was executed against. This manual only covers *how to run things and what's where*.

---

## 1. One-time setup

```bash
cd /home/harie06/dt_adaptation

# Python environment — already created at .venv/ (Python 3.14.4). If it ever needs rebuilding:
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Verify dirs/.env/dependencies/config are all sane (non-destructive, safe to re-run anytime)
.venv/bin/python scripts/setup.py
```

**Secrets**: copy `.env.example` to `.env` and fill in a real `GOOGLE_API_KEY` (or `GEMINI_API_KEY`
— either env var works, `GOOGLE_API_KEY` takes precedence if both are set). Never commit `.env`.
Every LLM-dependent command below needs this; everything else works without it.

**Note on activating the venv**: this environment has occasionally reset the shell's PATH/venv
activation between commands (a container-restart quirk, not a code issue). If `python`/`pytest`
suddenly can't be found or import your installed packages, use the venv's own binaries directly
— `.venv/bin/python`, `.venv/bin/pytest` — instead of relying on `source .venv/bin/activate`
having stuck. Every command in this manual already does this.

**Real OAI dataset build** (one-time, before first bootstrap/replay/baseline run — needs read
access to the real OAI export at `config.data.oai_root`; this never copies the raw CSVs into git):

```bash
.venv/bin/python scripts/build_oai_dataset.py
# writes data/processed/oai/{train,test}.parquet
```

---

## 2. Running the system

```bash
# Main continuous orchestrator — real, runs forever until Ctrl-C
.venv/bin/python -m src.main --mode demo                      # OAI-schema-shaped mock telemetry
.venv/bin/python -m src.main --mode live                      # real OAI RAN KPI replay
.venv/bin/python -m src.main --mode demo --max-drift-events 1 # bounded: stop after one adaptation cycle

# Permanent, unattended, real validation scripts (safe to re-run any time — each writes to its
# own storage subtree, never the shared production paths, except run_orchestrator_demo.py which
# genuinely touches data/{artifacts,models}/ since that IS what it's demonstrating)
.venv/bin/python scripts/run_orchestrator_demo.py       # one full real cycle against mock telemetry
.venv/bin/python scripts/run_oai_e2e_smoke.py            # real run-8 replay + a forced-trigger smoke test

# Standalone single-agent demos (recalibration needs no LLM key; regeneration/expand_scope do)
.venv/bin/python scripts/demo_recalibration.py
.venv/bin/python scripts/demo_regeneration.py
.venv/bin/python scripts/demo_expand_scope.py

# Bootstrap-train (or re-train) the 7 production OAI KPI models from scratch
.venv/bin/python scripts/bootstrap_oai_models.py
# writes reports/model_eval/{model_eval.csv,model_eval.md} + promotes v2 "bootstrap, OAI, residual"
# versions into data/models/registry_index.json

# Plot model-vs-baseline RMSE graphs from the already-written model_eval.csv (never re-trains)
.venv/bin/python scripts/plot_model_eval.py
# writes reports/model_eval/graphs/model_vs_baselines_<component>.{png,pdf,csv}

# RAG knowledge base ingestion (idempotent — re-running on unchanged files is a no-op)
.venv/bin/python scripts/ingest_rag.py

# Dashboard-style plots from live D1 history
.venv/bin/python scripts/visualize_metrics.py
```

### Baseline comparison experiments (`baselines/`)

Four experiment modes comparing adaptation strategies over the real OAI **TEST split (run 8)**
— never run 7 (reserved for smoke tests) and never any training run:

```bash
# Run everything (all 4 modes); writes under baselines/results/<mode>/
.venv/bin/python baselines/run_all.py

# Run a specific subset, override cycle count, skip the pre-flight LLM smoke test (dev only —
# never skip it for a run whose numbers you intend to actually report)
.venv/bin/python baselines/run_all.py --modes static periodic_retrain
.venv/bin/python baselines/run_all.py --modes rule_based --cycles 5 --skip-preflight
.venv/bin/python baselines/run_all.py --modes full_system      # 3 seeds, real decision-agent LLM calls

# Build the final cross-mode comparison report + 5 graphs from whichever modes have already run
# (never re-runs anything itself — reads only already-written baselines/results/<mode>/ output)
.venv/bin/python baselines/compare.py
```

The 4 modes: `static` (no adaptation, triggers logged only), `periodic_retrain` (blind scheduled
retraining of all 7 components), `rule_based` (fixed severity-threshold strategy selection, real
LLM only for codegen/explanation), `full_system` (the real knowledge-based `DecisionAgent` decides
every trigger's strategy — the actual system under test). See `baselines/README.md` for the full
fairness rules and cost-control design.

---

## 3. Testing

```bash
.venv/bin/python -m pytest tests/unit -q           # ~515 tests, fast (a few minutes)
.venv/bin/python -m pytest tests/integration -q    # ~35 tests, slower — real sandbox subprocesses,
                                                     # real file I/O, some real LLM calls if a key
                                                     # is configured; budget 10-15 minutes
.venv/bin/python -m pytest tests/e2e -q             # 1 test, ~3 minutes, zero real LLM cost
                                                     # (fakes the LLM transport only)
.venv/bin/python -m pytest tests -q                 # everything — this environment has
                                                     # intermittently been unable to sustain one
                                                     # single combined run without being killed;
                                                     # running unit/integration/e2e separately is
                                                     # the reliable path, not a sign of a bug.
```

`tests/unit/test_live_api_smoke_if_key_configured`-style tests genuinely call the live Google AI
endpoint when a real key is configured — they can fail for real if that day's free-tier quota
(20 requests/model/day) is already exhausted by other runs made the same day. That's expected,
not a regression to chase.

---

## 4. Top-level files

| File | What it is |
|---|---|
| `CLAUDE.md` | The living architecture/design document — read this for *why* things are built the way they are. Updated after every module. |
| `IMPLEMENTATION_STATUS.md` | Per-module build/test status tracker, companion to `CLAUDE.md`. |
| `claude_code_prompt_phase13.md` | The specification this latest round of work (model fixes, LLM caching/cost controls, the `baselines/` framework, graphs, docs, Phase-12 cleanup) was executed against. |
| `prompt.md` | The *original* project specification (NS-3-era). Superseded by `prompt2.md` where they conflict. Kept for historical context. |
| `prompt2.md` | The OAI-migration specification ("migrate the Digital Twin from NS-3 to OAI KPI replay"). Wins over `prompt.md` on everything it covers. Superseded in turn, for this latest round, by `claude_code_prompt_phase13.md`. |
| `README.md` | Short top-level project readme. |
| `manual.md` | **This file.** |
| `PHASE13_FINAL_REPORT.md` | The §9 final report for this phase — model-fix results, lock, LLM cost/caching, baseline comparison table, graphs list, outputs list, docs status, Phase-12 open-issues resolution, and test pass counts. |
| `AGENTS.md` | The 6 agentic AI agents (Decision & Root-Cause, Recalibration, Regeneration, Expand-Scope, Verification, Lifecycle) — what each does, how it does it, and exactly what tools/resources (LLM, RAG, sandbox, registries, D1) each one can access. |
| `manual.txt` | Superseded by this file — kept for history, documents the pre-Phase-13 state. |
| `config/settings.yaml` | Every tunable parameter in the system. Nothing is hardcoded in `src/` that should live here. |
| `.env` / `.env.example` | Secrets (`GOOGLE_API_KEY`/`GEMINI_API_KEY`). `.env` is never committed; `.env.example` documents the required keys with placeholder values. |
| `.gitignore` | Standard excludes — notably all of `data/`'s runtime output, `baselines/results/`, `.venv/`, real OAI exports, `rankpi_kpi/`. |
| `fig-dataflow.png` | The canonical 19-module + D1/D2 data-flow diagram referenced throughout `CLAUDE.md`. |
| `pyproject.toml` | Python project metadata + pytest config. |
| `requirements.txt` | Pinned Python dependencies. |
| `rankpi_kpi/` (989MB, gitignored) | A confirmed duplicate of the real OAI export already present at `/home/harie06/oai_db/data/rankpi_kpi` — verified via file-list/size/sha256 comparison, never modified or deleted by any automated process; kept here only because the project's own historical investigation found it and reported it, not moved. |

---

## 5. `src/` — production code

| Path | Responsibility |
|---|---|
| `src/main.py` | `ContinuousOrchestrator` — the top-level loop wiring every module together: bootstrap, continuous sync, prediction+fidelity cycles, trigger dispatch, decision→agent→verification→lifecycle, and live hot-swap of an ACCEPTed candidate into the running process. |
| `src/common/config.py` | Pydantic-validated, frozen `Settings` (loaded from `config/settings.yaml`) + separate redacted `Secrets` (from `.env`). Every config section (`data`, `telemetry`, `synchronization`, `dt_models`, `fidelity`, `drift`, `decision_agent`, `adaptation`, `sandbox`, `rag`, `baselines`, `llm`, `logging`, `lifecycle`, `evaluation`) is a typed model here. |
| `src/common/logging.py` | Structured JSON logging with secret redaction. |
| `src/telemetry/` | Module 1 (OAI replay) + Module 2 (preprocessing). `oai_replay_source.py` deterministically replays real exported OAI RAN KPI rows in genuine timestamp order; `mock_source.py` generates schema-matching mock telemetry for demo mode; `preprocessing.py` validates/cleans/imputes; `schema.py`/`base.py` define the shared contract. |
| `src/data/oai_dataset.py` | Builds the causality-correct lagged/rolling-window feature set (249 columns) from the real OAI export; the data pipeline behind `scripts/build_oai_dataset.py`. |
| `src/synchronization/` | Module 3 — `ContinuousSynchronizer` pushes validated telemetry into D1 on a background thread, batched, with no manual sync step anywhere. `d1_interface.py` defines the `D1StateSink` write contract. |
| `src/dt_models/` | D1 (state store) + Module 5 (orchestrator) + the 7 real OAI KPI prediction components. `d1_model_store.py`/`component_registry.py` = D1. `base.py`/`model_registry.py` (note: this is the *in-memory component-wiring* registry, distinct from `src/registry/model_registry.py`)/`orchestrator.py` = Module 5. `residual_base.py` = the shared residual/delta-target base class every component subclasses. `{dl_bler,ul_bler,dl_prb,ul_prb,dl_throughput,ul_throughput,dl_rlc_delay}.py` = the 7 real components. `regressors.py` = the shared XGBoost/RandomForest factory. |
| `src/fidelity/` | Module 12 — deterministic fidelity engine. `metrics.py` (RMSE/MAE/Wasserstein/MK-MMD, pure functions). `evaluator.py` (`FidelityEvaluator` — rolling-window normalization, per-component + unified fidelity score). `trigger.py` (the internal fidelity-based adaptation trigger). |
| `src/drift/` | Module 11 — the external drift-event contract. `base.py`/`schema.py`/`drift_detector.py` (validation/normalization) + `mock_drift_source.py` (the only source implemented; a real partner detector would plug in here with zero changes to `drift_detector.py`). |
| `src/adaptation/` | Modules 13-17, 19 (five of the six agents, plus shared helpers). `decision_context.py`/`decision_agent.py` = Module 13 (the knowledge-based LLM decision agent). `recalibration_agent.py` = Module 14. `regeneration_agent.py` = Module 15. `expand_scope_agent.py` = Module 16. `verification_agent.py` = Module 17 (the deterministic accept/reject gate). `lifecycle_agent.py` = Module 19. `data_selection.py` = shared window-selection helpers used by Modules 14-16. `baseline_policies.py` = the deterministic rule-based/periodic strategies used ONLY by `baselines/`, never by the real `decision_agent.py`. |
| `src/rag/rag_kb.py` | D2/Module 18 — `RagKnowledgeBase` (read-only; the only class any agent ever touches) + `RagIngestor` (write-only, admin/build-time use only, via `scripts/ingest_rag.py`). Backed by a real persistent ChromaDB store at `rag_data/chroma/`. |
| `src/registry/model_registry.py` | The versioned model registry (Concept C) — tracks every model VERSION ever produced (bootstrap/recalibrate/regenerate/expand_scope), current production version per component, promote/reject/rollback. Distinct from `src/dt_models/model_registry.py` (see above) and `src/dt_models/component_registry.py` (D1's state-table catalog) — three different "registries", see `CLAUDE.md` §5's module entry for the full disambiguation. |
| `src/sandbox/` | `executor.py`/`_sandbox_driver.py` — subprocess isolation for every LLM-generated candidate (Modules 15/16). Syntax→import→conformance→train→evaluate→save, each stage fully captured, workspace always cleaned up. |
| `src/llm/` | `google_client.py` — the one centralized Google AI (Gemini) client every agent uses; retry/backoff, structured output, prompt caching (`start_cache`/`end_cache`), safe (`*_safe`) graceful-degradation wrappers. `cost_tracking.py` — `BudgetGuard`, `MeteredGoogleClient`, `RoleTaggedClient`, cost estimation, and the `llm_calls.jsonl` audit log. |
| `src/storage/__init__.py` | Present but currently unused (empty placeholder from early project history). |

---

## 6. `baselines/` — the Phase-13 comparison framework

| Path | What it is |
|---|---|
| `run_all.py` | CLI entry point — runs one or more of the 4 modes over the real OAI TEST replay (run 8), writes everything under `results/<mode>/` (or `results/<mode>/seed_<n>/` for `full_system`'s 3 seeds). Includes the `preflight_smoke_test()` (exercises all 3 LLM roles against run 7 before any real run, per §3.8 of the spec). |
| `compare.py` | Reads only already-written `results/<mode>/` output and builds `results/comparison/{comparison.md,comparison.csv}` plus 5 cross-mode graphs. Never re-runs anything. |
| `_lib.py` | Shared machinery every mode calls: `bootstrap_components()` (fairness — identical bootstrap for every mode), `replay_test_split()`, `run_prediction_and_fidelity_cycle()`, `run_one_adaptation()`, `build_static_prefix()` (the cacheable LLM prompt prefix). |
| `_outputs.py` | `metrics_timeline.csv`/`adaptation_events.jsonl`/`report.md` writers, all per-mode graphs, and the `DISPLAY_UNITS` conversion (Mbps/ms at plot time only — stored data stays in raw units). |
| `_console.py` | Live console output during a run (uses `rich` if installed, falls back to plain text). |
| `configs/{static,periodic,rule_based,full_system}.yaml` | Per-mode configuration (seeds, severity thresholds, retrain interval, etc.). |
| `README.md` | The fairness rules and cost-control design in full detail. |
| `results/` (gitignored) | All real run output — see §8 below for what's inside each mode's folder. |

---

## 7. `config/`, `data/`, `docs/`, `reports/`, `rag_data/`, `scripts/`, `tests/`, `archive/`

| Path | What it is |
|---|---|
| `config/settings.yaml` | Every tunable value in the whole system (data splits, fidelity formula constants, sandbox timeouts, LLM model/rates/caching, baseline mode settings, etc.) — nothing in `src/` should hardcode a value that belongs here. |
| `data/models/` | The 7 real **production** component artifacts (`dl_bler/`, `ul_bler/`, `dl_prb/`, `ul_prb/`, `dl_throughput/`, `ul_throughput/`, `dl_rlc_delay/`, each `.joblib`-serialized) + `registry_index.json` (the versioned registry index). Gitignored, runtime-only. |
| `data/processed/oai/{train,test}.parquet` | The real, causality-correct engineered feature sets built by `scripts/build_oai_dataset.py`. Gitignored (derived from the real OAI export, which itself is never copied into git). |
| `data/artifacts/` | D1 state (`d1_current.parquet`/`d1_history.parquet`/`d1_quarantine.parquet`), `lifecycle_records.jsonl`, `maintenance_reports/*.md`, sandbox scratch space. |
| `data/bootstrap/`, `data/telemetry/`, `data/evaluation/` | Scratch/working subtrees for bootstrap training data, telemetry capture, and evaluation windows respectively. |
| `data/e2e_oai_smoke/` | Dedicated storage subtree for `scripts/run_oai_e2e_smoke.py` — never the shared production paths. |
| `docs/MODELS.md` | Full per-component documentation (purpose, inputs, hyperparameters, dependencies, real CV results, limitations) for all 7 OAI KPI models + the dependency graph + "KPIs available but not modelled" section. **The authoritative model documentation** — read this, not the stale splits table in `docs/oai_twin.md`. |
| `docs/oai_twin.md` | The original OAI-migration design document (real KPI catalog, data-splits rationale, replay design). Some sections (old splits table, old bootstrap metrics) are explicitly flagged superseded by `docs/MODELS.md`/`config/settings.yaml`. |
| `docs/architecture_diagram.drawio` | The current (OAI-era, Phase-13-rebuilt) architecture diagram — open with diagrams.net / the draw.io desktop app or VS Code's Draw.io Integration extension. |
| `docs/README.md` | Explains the status of `docs/IEEE_Digital_Twin_Paper.docx`/`docs/FORMULAS.pdf` — both predate the OAI migration; no updated paper has been produced (an honestly-flagged open gap, not hidden). |
| `docs/HOW_TO_RUN.md`, `docs/MANUAL.md` | Earlier, narrower run manuals written mid-project (post-OAI-migration, pre-Phase-13). This file (`manual.md`, repo root) is the current, complete one. |
| `reports/model_eval/{model_eval.csv,model_eval.md}` | The real leave-one-run-out CV results for all 7 components (RMSE/MAE vs. persistence vs. 5s moving-average, beats-persistence flag) — written by `scripts/bootstrap_oai_models.py`. |
| `reports/model_eval/graphs/` | `model_vs_baselines_<component>.{png,pdf,csv}` — written by `scripts/plot_model_eval.py`. |
| `rag_data/{oran,digital_twin,policies,history}/` | The real source documents ingested into the RAG knowledge base. `rag_data/chroma/` is the persistent vector store itself (gitignored). |
| `scripts/` | One-shot/permanent utility and demo scripts — see §2 above for the ones you'll actually run; `setup.py` verifies the environment, the rest are documented inline in §2. |
| `tests/unit/` | ~515 fast, isolated tests, one file per module. |
| `tests/integration/` | ~35 slower tests exercising real multi-module pipelines (real sandbox subprocesses, real D1/registry persistence, some real LLM calls). |
| `tests/e2e/test_baseline_rule_based_e2e.py` | The one required end-to-end test: replays a slice of run 7 through the full `rule_based` baseline lifecycle with a faked LLM transport — zero real API cost, asserts all required outputs exist. |
| `tests/dummy_dt_components.py`, `tests/conftest.py`, `tests/integration/conftest.py` | Shared test fixtures (dummy dependency-chained components, the shared bootstrap-history fixture, etc.). |
| `archive/ns3_sim/` | The retired NS-3/5G-LENA toolchain and its pre-migration runtime data — kept, never deleted, for historical reference. Not used by anything live; `tests/unit/test_no_archive_imports.py` structurally guarantees nothing under `src/` ever imports from here. |

---

## 8. What a completed `baselines/results/<mode>/` folder contains

```
results/<mode>/                      (or results/full_system/seed_<n>/ for that mode)
├── metrics_timeline.csv             # per-cycle, per-component: actual, predicted, F_c, F_U, trigger_fired, model_version
├── adaptation_events.jsonl          # one line per adaptation attempt: trigger/decision/gate_result/fidelity before+after
├── report.md                        # human-readable report: header + per-component summary table + one block per event
├── run_summary.json                 # n_triggers/n_accepted/n_rejected/mean_unified_fidelity/budget_spent_usd/...
├── llm_calls.jsonl                  # every real LLM call made (role, tokens, cost, cache hit) — absent for static/periodic_retrain (no LLM) and for a mode run against a faked client
├── d1_current.parquet / d1_history.parquet   # the bootstrap D1 state this run's bootstrap produced
├── models/                          # this run's own isolated model registry (never touches the shared data/models/)
└── graphs/                          # actual_vs_predicted_<component>, fidelity_<component>, unified_fidelity (PNG+PDF+CSV each)
```

`results/comparison/` (written by `baselines/compare.py`, once all 4 modes have run):
`comparison.md`, `comparison.csv`, and `graphs/{unified_fidelity_all_modes, mean_fidelity_by_mode,
recovery_time_by_mode, adaptations_by_mode, strategy_distribution}.{png,pdf,csv}`.
