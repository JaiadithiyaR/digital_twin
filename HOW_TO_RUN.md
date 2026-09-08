# How to Run — AI-Driven Self-Adaptive Network Digital Twin

Step-by-step instructions for setting up and running this project from a clean checkout. See
`CLAUDE.md` for architecture/design and `IMPLEMENTATION_STATUS.md` for current build status —
this file is only "how do I actually run things."

---

## 1. One-time setup

```bash
cd dt_adaptation

# Python 3.14 virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (idempotent — safe to re-run)
pip install -r requirements.txt

# Secrets — never committed. Fill in a real Anthropic API key to enable the
# Regeneration/Expand-Scope/Verification-explanation/Lifecycle-report LLM calls.
# Without a real key, the system still runs: recalibration needs no LLM at all, and every
# other LLM call degrades gracefully (logged, never faked) — see CLAUDE.md §7.
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# Verify the environment: directories, .env, dependencies, config all check out (non-destructive)
python scripts/setup.py
```

All tunable parameters live in `config/settings.yaml` — nothing to edit there for a normal run.

---

## 2. Run the automated test suite

```bash
.venv/bin/python -m pytest tests/unit -v          # fast, isolated
.venv/bin/python -m pytest tests/integration -v   # slower — real pipelines, real (small) model training
.venv/bin/python -m pytest tests -q               # everything (currently 487 tests; on a small/
                                                   # memory-constrained machine, run tests/unit and
                                                   # tests/integration as two separate invocations
                                                   # if a single combined run gets OOM-killed)
```

No test hits a real network/API — the Anthropic client's transport is mocked in every test; only
`test_anthropic_client.py`'s one live-smoke test would call the real API, and it auto-skips
unless a real `ANTHROPIC_API_KEY` is configured.

---

## 3. (Optional) Build the real NS-3 / 5G-LENA simulator

Only needed for `--mode live` / the real-NS-3 end-to-end demo (step 6). Everything else
(`--mode demo`, all tests, PPO training, RAG ingestion) uses mock telemetry and does not need this.

```bash
./scripts/setup_ns3.sh   # idempotent: clones ns-3.48 + 5G-LENA (nr v5.1) into ns3_sim/ns-3-dev,
                          # configures, and builds (2000+ targets — takes a while the first time)
```

Some of the `nr` module's own and our telemetry exporter's C++ dependencies
(`sqlite3 libsqlite3-dev libeigen3-dev libzmq3-dev libzmq5 cppzmq-dev`) need `sudo apt-get
install` — see `ns3_sim/README.md` for the exact command if `setup_ns3.sh` warns they're missing.

Sanity-check the build + the real telemetry producer in isolation (no orchestrator involved):

```bash
python ns3_sim/validate_e2e.py
```

---

## 4. (Optional) Prerequisites for a full adaptation cycle

These two are only needed the FIRST time — their outputs are saved to disk and reused
automatically after that (`data/models/ppo/policy.zip`, `rag_data/chroma/`).

```bash
# Train the PPO policy (real Stable-Baselines3 training against the real AdaptationEnv)
python scripts/train_ppo.py --total-timesteps 20000 --n-envs 4 --seed 42

# Ingest the RAG knowledge base corpus into ChromaDB (idempotent)
python scripts/ingest_rag.py
```

If skipped, the system still runs: with no trained PPO policy it falls back to
`config.ppo.fallback.default_action` (logged every time, never silent); with no RAG store it
proceeds with "not available" context (logged, never fabricated).

---

## 5. Run the full system

```bash
python -m src.main --mode demo                        # mock telemetry, runs forever (Ctrl-C to stop)
python -m src.main --mode live                         # real NS-3 telemetry over ZeroMQ (needs step 3
                                                         # AND a separately-running NS-3 process — see
                                                         # scripts/run_e2e_demo.py for how it's launched)
python -m src.main --mode demo --max-drift-events 1     # bounded run — stop after one adaptation cycle
```

This is the real continuous loop (`ContinuousOrchestrator`): telemetry -> preprocessing ->
continuous D1 synchronization -> dependency-aware DT prediction -> fidelity evaluation -> drift
event -> PPO decision -> recalibrate/regenerate/expand_scope -> verification -> lifecycle record
-> repeat, forever, without stopping — exactly prompt.md §39. It writes to this repo's REAL
configured storage paths (`data/artifacts/d1_*.parquet`, `data/models/`,
`data/artifacts/lifecycle_records.jsonl`, `data/artifacts/maintenance_reports/`) and logs to
`logs/digital_twin.log` (structured JSON) as well as stdout.

---

## 6. Run the two permanent, repo-tracked demo/validation scripts

Both are unattended, single-adaptation-cycle runs that print their own real observed results and
exit non-zero if anything didn't actually happen as claimed (no fabricated success).

```bash
# Mock telemetry — fast (~1-2 minutes), good for iterating
python scripts/run_orchestrator_demo.py

# REAL NS-3 telemetry — launches the compiled binary from step 3 as a subprocess, connects the
# orchestrator to it over real ZeroMQ, and runs one full real cycle (~1-3 minutes). Writes to its
# OWN dedicated storage path (data/e2e_ns3_demo/), never the shared --mode demo/live paths, so a
# leftover mock-telemetry run can never be confused with real NS-3 telemetry.
python scripts/run_e2e_demo.py 2>&1 | tee logs/e2e_demo_output.log
```

Each prints: whether an LLM/PPO/RAG were available, live telemetry row accumulation progress, the
real fidelity values it warmed up with, a concrete before/after `records_synced` proof that
telemetry never stopped during the adaptation cycle, the full lifecycle record (drift event, PPO
action, candidate version, fidelity before/after, ACCEPT/REJECT decision), and the path to the
generated maintenance report.

---

## 7. Where things end up

```
data/artifacts/d1_current_state.parquet     live DT current state (Module 4/D1)
data/artifacts/d1_history.parquet           live DT history
data/artifacts/d1_quarantine.parquet        rejected telemetry records (never silently dropped)
data/models/<component>/<version>.joblib    versioned DT model artifacts (Modules 6-10 + adapted)
data/models/registry_index.json            versioned model registry (Concept C)
data/models/ppo/policy.zip                  trained PPO policy
data/artifacts/lifecycle_records.jsonl      every adaptation event ever recorded (Module 19)
data/artifacts/maintenance_reports/*.md     human-readable report per adaptation event
rag_data/chroma/                            RAG vector store (D2)
logs/digital_twin.log                       structured JSON log of every run
logs/ns3_e2e_demo_stdout.log                real NS-3 simulator's own console output (step 6, real-NS-3 run)
data/e2e_ns3_demo/                          the real-NS-3 demo script's OWN dedicated D1/registry/
                                             lifecycle state (step 6) — separate from the paths above
                                             so it's never confused with a mock-telemetry run's data
```

All of the above are gitignored — they're real, regenerable runtime state, not source.
