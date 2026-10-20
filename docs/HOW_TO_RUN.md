# How to Run — AI-Driven Self-Adaptive Network Digital Twin

Step-by-step instructions for setting up and running this project from a clean checkout. See
`CLAUDE.md` for architecture/design and `IMPLEMENTATION_STATUS.md` for current build status —
this file is only "how do I actually run things." **OAI migration note**: this project no longer
uses NS-3/5G-LENA — telemetry is a replay of real exported OAI RAN KPI CSVs (see
`docs/oai_twin.md` and `archive/ns3_sim/README.md` for what changed and why).

---

## 1. One-time setup

```bash
cd dt_adaptation

# Python 3.14 virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (idempotent — safe to re-run)
pip install -r requirements.txt

# Secrets — never committed. Fill in a real Google AI (Gemini) API key to enable the
# Decision & Root-Cause Analysis Agent, Regeneration/Expand-Scope/Verification-explanation/
# Lifecycle-report LLM calls. Without a real key, the system still runs: the decision agent
# degrades to its configured deterministic fallback strategy (logged every time, never silent),
# recalibration execution needs no LLM at all, and every other LLM call degrades gracefully
# (logged, never faked) — see CLAUDE.md §7.
cp .env.example .env
# then edit .env and set GOOGLE_API_KEY=... (or GEMINI_API_KEY=...)

# Verify the environment: directories, .env, dependencies, config all check out (non-destructive)
python scripts/setup.py
```

All tunable parameters live in `config/settings.yaml` — nothing to edit there for a normal run,
except `data.oai_root` (the read-only OAI export path, outside this repo) if yours differs from
the checked-in default.

---

## 2. Run the automated test suite

```bash
.venv/bin/python -m pytest tests/unit -v          # fast, isolated
.venv/bin/python -m pytest tests/integration -v   # slower — real pipelines, real (small) model training
.venv/bin/python -m pytest tests -q               # everything (on a small/memory-constrained
                                                   # machine, run tests/unit and tests/integration
                                                   # as two separate invocations if a single
                                                   # combined run gets OOM-killed)
```

No test hits a real network/API — the Google AI (Gemini) client's transport is mocked in every
test; only `test_google_client.py`'s one live-smoke test would call the real API, and it
auto-skips unless a real `GOOGLE_API_KEY`/`GEMINI_API_KEY` is configured.

---

## 3. Build the real OAI dataset (one-time, before first bootstrap)

Reads the read-only OAI export at `config.data.oai_root` (default
`/path/to/oai_db/data/rankpi_kpi/`) and writes the model-ready, config-driven
train/val/test splits (`config.data.splits` — never hardcoded run ids) this system trains and
replays from. Never copies the raw CSVs into this repo.

```bash
python scripts/build_oai_dataset.py
```

Produces `data/processed/oai/{train,val,test}.parquet`, `feature_schema.json`, and
`data_report.md` (NaN/reset/gap counts, rows per split, and an honest note on any excluded
incomplete run). `--mode demo`/`--mode live` and every demo script below read these files —
run this once before the first real bootstrap/replay.

---

## 4. (Optional) Prerequisite for a full adaptation cycle

There is no training step for the Decision & Root-Cause Analysis Agent (Module 13) — it makes a
genuine LLM call on every trigger instead of using a pre-trained policy — so the only real
one-time prerequisite is the RAG knowledge base, whose output is saved to disk and reused
automatically after the first run (`rag_data/chroma/`).

```bash
# Ingest the RAG knowledge base corpus into ChromaDB (idempotent)
python scripts/ingest_rag.py
```

If skipped, the system still runs: with no RAG store it proceeds with "not available" context
(logged, never fabricated). Separately, with no real `GOOGLE_API_KEY`/`GEMINI_API_KEY` configured
(step 1), the decision agent's own LLM call genuinely fails and falls back to
`config.decision_agent.fallback.default_strategy` (logged every time, never silent).

---

## 5. Run the full system

```bash
python -m src.main --mode demo                        # OAI-schema-shaped mock telemetry (SOURCE=MOCK),
                                                        # runs forever (Ctrl-C to stop) — no OAI
                                                        # export needed, used for unit-test-style iteration
python -m src.main --mode live                         # real OAI RAN KPI replay (SOURCE=OAI_RAN_REPLAY),
                                                        # streamed from data/processed/oai/ (step 3) in
                                                        # timestamp order at config.telemetry.replay.speed
python -m src.main --mode demo --max-adaptations 1     # bounded run — stop after one adaptation cycle
```

This is the real continuous loop (`ContinuousOrchestrator`): telemetry -> preprocessing ->
continuous D1 synchronization -> dependency-aware DT prediction (the 7 OAI KPI components) ->
fidelity evaluation -> drift event/fidelity-based trigger -> Decision & Root-Cause Analysis
Agent -> recalibrate/regenerate/expand_scope -> verification -> lifecycle record -> repeat,
forever, without stopping — exactly prompt.md §39. It writes to this repo's REAL configured
storage paths (`data/artifacts/d1_*.parquet`, `data/models/`,
`data/artifacts/lifecycle_records.jsonl`, `data/artifacts/maintenance_reports/`) and logs to
`logs/digital_twin.log` (structured JSON) as well as stdout.

---

## 6. Run the permanent, repo-tracked demo/validation scripts

Each is an unattended, single-adaptation-cycle run that prints its own real observed results and
exits non-zero if anything didn't actually happen as claimed (no fabricated success).

```bash
# Mock telemetry — fast (~1-2 minutes), good for iterating, no OAI export needed
python scripts/run_orchestrator_demo.py

# REAL OAI run-8 (TEST split) replay at speed=0, with a forced adaptation trigger for
# determinism (prompt2.md Step 7's required smoke test) — writes to its OWN dedicated storage
# path (data/e2e_oai_smoke/), never the shared --mode demo/live paths
python scripts/run_oai_e2e_smoke.py

# All four baseline experiment modes (static / periodic_retrain / rule_based / full) over the
# same real OAI TEST-split replay/seed/fidelity settings, for comparison
python baselines/run_all.py   # then: python baselines/stats.py
```

Each prints: whether an LLM/RAG were available, live telemetry row accumulation progress, the
real fidelity values it warmed up with, the full lifecycle record (trigger, decision agent
strategy, candidate version, fidelity before/after, ACCEPT/REJECT decision), and (for the
OAI-replay scripts) the telemetry source/replayed run ids/replay speed actually used.

---

## 6b. Visualize telemetry and fidelity from a previous run

`scripts/visualize_metrics.py` is a standalone, read-only program that charts whatever the system
has already genuinely produced — it never runs the pipeline itself, only reads and plots. Run it
any time after step 5 or 6 above has produced some real data:

```bash
# config/settings.yaml's own production storage paths (what `python -m src.main` writes)
python scripts/visualize_metrics.py

# an explicit alternate storage path (e.g. a demo script's own dedicated subtree)
python scripts/visualize_metrics.py --history-path data/e2e_oai_smoke/d1_history.parquet \
    --models-dir data/e2e_oai_smoke/models --lifecycle-path data/e2e_oai_smoke/lifecycle_records.jsonl
```

Produces, under `--output-dir` (default `data/artifacts/plots/`):

- `telemetry_core_metrics.png` — the 7 OAI KPI targets over time (dl/ul throughput, dl_rlc_delay,
  dl/ul_prb, dl/ul_bler)
- `telemetry_radio_conditions.png` — lagged RAN-condition/control features over time (PUSCH SNR,
  wideband CQI, DL current TBS, requested PRB ratio)
- `fidelity_over_time.png` — composite FidelityScore and raw RMSE per component, **recomputed by
  replaying the real D1 history through each component's real current production model via the
  same Module 12 `FidelityEvaluator` the live system uses** (not a re-implementation or estimate)
- `fidelity_timeseries_<component>.csv` — the full unclipped numbers behind that chart (the PNG's
  y-axis is capped for readability on outliers; the CSV never is)
- `lifecycle_adaptation_outcomes.png` — fidelity before/after per Module 19 lifecycle record,
  outlined green (ACCEPT) or red (REJECT)

`--window`/`--stride` control how many rows go into each fidelity-recompute window (default 20,
non-overlapping); `--history-path`/`--models-dir`/`--lifecycle-path` override individual paths
directly if your data doesn't live under the production paths. Nothing here writes to D1, the
model registry, or the lifecycle log.

---

## 7. Where things end up

```
data/processed/oai/{train,val,test}.parquet   model-ready OAI splits (step 3) — regenerable, never committed
data/artifacts/d1_current_state.parquet       live DT current state (Module 4/D1)
data/artifacts/d1_history.parquet             live DT history
data/artifacts/d1_quarantine.parquet          rejected telemetry records (never silently dropped)
data/models/<component>/<version>.joblib      versioned DT model artifacts (the 7 OAI components + adapted)
data/models/registry_index.json               versioned model registry (Concept C)
data/artifacts/lifecycle_records.jsonl        every adaptation event ever recorded (Module 19)
data/artifacts/maintenance_reports/*.md       human-readable report per adaptation event
rag_data/chroma/                              RAG vector store (D2)
logs/digital_twin.log                         structured JSON log of every run
data/e2e_oai_smoke/                           scripts/run_oai_e2e_smoke.py's OWN dedicated D1/
                                               registry/lifecycle state — separate from the paths
                                               above so it's never confused with a production run
archive/ns3_sim/                              retired NS-3/5G-LENA code + pre-migration runtime
                                               state — historical reference only, never run
```

All of the above (except `archive/ns3_sim/`, which is tracked source + gitignored vendored
build output) are gitignored — they're real, regenerable runtime state, not source.
