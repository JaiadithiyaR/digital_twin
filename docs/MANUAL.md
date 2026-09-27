# MANUAL — What Every File In This Project Does

This is a plain-English guide to every file in this repository. `CLAUDE.md` explains the
*architecture* (the 19 numbered modules, the design rules); `HOW_TO_RUN.md` explains *how to run
things*. This file explains **what each individual file is for**, in simple terms, so that anyone
— not just someone who already knows the codebase — can open the project and understand what
each piece does and how it connects to the rest.

---

## 1. What this project actually is

Imagine a real 5G phone network. This project builds a **"digital twin"** of that network — a
computer program that watches the real network's behavior and keeps its own live copy that
predicts what the network is doing (how fast is the internet, how much delay is there, how many
packets are being dropped, etc.).

Because a real network keeps changing (more people join, weather changes signal quality, towers
get congested), the digital twin's predictions can start to become wrong over time. This project's
whole point is to make the twin **notice when it's becoming wrong and fix itself automatically**,
without a human having to step in — by:

1. Constantly listening to real (or simulated) network telemetry.
2. Predicting network behavior using small machine-learning models.
3. Constantly checking "how close are my predictions to reality?" (this is called **fidelity**).
4. When fidelity drops too much ("drift"), deciding what to do about it — by asking an AI
   language model to analyze the likely root cause and pick a fix strategy (this project calls it
   the **Decision & Root-Cause Analysis Agent**).
5. Actually fixing the broken piece — either by retraining it, or by asking that same kind of AI
   language model to rewrite it, or even to invent a brand-new prediction component.
6. Double-checking that the fix is genuinely better before trusting it (never blindly trusting the
   AI's own opinion).
7. Writing down exactly what happened, so a human can always audit every automatic decision later.

Everything below maps onto one of those seven steps.

---

## 2. The 30-second map

```
Network → telemetry → cleaning → digital twin's memory → predictions → fidelity check
                                                                              │
                                                                     "did it drift?"
                                                                              │
                                                              AI decides HOW to fix it
                                                                              │
                                                         fix is built in an isolated sandbox
                                                                              │
                                                        a strict, non-AI check: is it ACTUALLY better?
                                                                              │
                                                          yes → promote it / no → throw it away
                                                                              │
                                                              write down what happened
```

Every file in this repo exists to do one link in that chain. The sections below go folder by
folder.

---

## 3. Top-level files (the front door of the repository)

| File | What it's for, in plain terms |
|---|---|
| `README.md` | The very first thing anyone sees. A short "what is this, how do I start" page that points to the more detailed guides below. |
| `CLAUDE.md` | The architecture rulebook. Explains the 19 "modules" (building blocks) of the system, the safety rules (e.g. "the AI can never override the safety check"), and a running diary of exactly how each module was built and validated. Read this to understand *why* the code is shaped the way it is. |
| `IMPLEMENTATION_STATUS.md` | A big checklist/status board: which modules are done, what tests exist, what's left to do. Like a project tracker, but living inside the repo. |
| `prompt.md` | The original specification this whole project was built from — the "brief" that was handed over describing exactly what the finished system must do. Treated as the ultimate source of truth if anything else disagrees with it. |
| `pyproject.toml` | A small configuration file that tells Python tools (like the test runner and the code-formatter "ruff") basic facts about this project: its name, where the tests live, what Python version it needs. |
| `requirements.txt` | The shopping list of external software libraries this project depends on (e.g. `pandas` for data tables, `xgboost` for machine learning, `google-genai` for talking to the AI). Running `pip install -r requirements.txt` installs everything on this list. |
| `.env.example` | A template showing which secret values (like the Google AI API key used to talk to the AI) need to be filled in. You copy this to a real `.env` file and put your own secret key in — `.env` itself is never shared or committed to version control. |
| `.env` | Your actual secrets file (copied from `.env.example`). Currently holds a placeholder key, not a real working one. |
| `.gitignore` | Tells the version-control system (git) which files/folders to *never* track — generated data, secrets, build outputs, virtual environments, log files, etc. Keeps the repository clean of things that shouldn't be shared. |
| `fig-dataflow.png` | The original hand-drawn/diagram picture showing how data is supposed to flow through all 19 modules. This is the master architecture diagram every module was built to match. |

The step-by-step run guide and this file itself now live in `docs/` — see section 9 below.

---

## 4. `config/` — every adjustable knob in one place

| File | What it's for |
|---|---|
| `config/settings.yaml` | **The single control panel for the entire system.** Every number or setting that could reasonably need tuning lives here instead of being buried in code — how many telemetry readings make a "batch," how sensitive the fidelity formula is, how long the AI is allowed to think, sandbox time limits, and so on. If you want to change how the system behaves without touching code, this is the file to edit. |

---

## 5. `ns3_sim/` — the pretend (but realistic) 5G network

Since we don't have a real 5G tower to plug into, this project uses **ns-3**, a widely-used
network simulator, together with an add-on called **5G-LENA** that makes it behave like real 5G
equipment. This is "Module 1" in the architecture — the stand-in for the real physical network.

| File | What it's for |
|---|---|
| `ns3_sim/README.md` | Explains exactly which versions of the simulator are used, how to install its prerequisites, and the rule that real simulator output must never be confused with fake/mock data. |
| `ns3_sim/nr_5g_telemetry_sim.cc` | **The actual simulated 5G network**, written in C++ (the language ns-3 is built in). Sets up one cell tower and six moving phones, runs a realistic radio simulation (signal strength, interference, motion), and streams out live readings (speed, delay, signal quality, etc.) over the network so the rest of this Python project can consume them, exactly like a real base station would. |
| `ns3_sim/validate_e2e.py` | A stand-alone sanity-check script: starts the simulator above, listens for its output, and confirms the data it sends is well-formed and realistic — used to verify Module 1 works correctly on its own, before wiring it into the bigger system. |
| `ns3_sim/ns-3-dev/` *(not tracked in git)* | The actual ns-3 simulator software itself, downloaded and built by `scripts/setup_ns3.sh`. Not part of this project's own code — it's an external tool (like a compiler), so it isn't stored in the repository, only referenced. |

---

## 6. `src/` — the actual application

Everything the digital twin does at runtime lives here, organized into folders that match the
architecture's stages. Below is every folder, in the order data flows through them.

### 6.1 `src/common/` — shared basics every other file relies on

| File | What it's for |
|---|---|
| `src/common/config.py` | Reads `config/settings.yaml` and `.env` and turns them into a well-checked, typo-proof Python object (`Settings`/`Secrets`) that the rest of the code uses to look up any setting. If the config file has a mistake (like a missing required value), this file is what catches it immediately instead of letting a bad value slip through silently. |
| `src/common/logging.py` | Sets up how the system writes its log messages (the "diary" of everything happening while it runs) — formatted as structured data so a human or a tool could later trace exactly what happened at each step of an adaptation, from telemetry all the way to a final decision. Also makes sure secret values (like API keys) never accidentally get written into logs. |

### 6.2 `src/telemetry/` — Module 2: listening to the network and cleaning up what it hears

| File | What it's for |
|---|---|
| `src/telemetry/base.py` | The common "shape" every telemetry source must follow (mock or real) — just says "you must be able to give me a stream of readings, and you must honestly label where they came from." |
| `src/telemetry/schema.py` | Defines exactly what a single valid telemetry reading looks like (which fields it has, what units they're in) and how to convert between "raw" wire formats and the clean format the rest of the system expects. |
| `src/telemetry/mock_source.py` | A fake-but-realistic telemetry generator, used when the real simulator isn't running. It doesn't just spit out random numbers — it makes throughput, delay, signal strength etc. behave like they would on a real network (e.g. more phones → more congestion → less throughput per phone), so the machine-learning models have something real to learn from. |
| `src/telemetry/zmq_source.py` | The real telemetry listener — connects to the actual ns-3 simulator (or a real network exporter) over a messaging protocol called ZeroMQ and receives its live readings. |
| `src/telemetry/preprocessing.py` | The "cleaning station." Takes raw incoming readings and: fills in small gaps sensibly, flags (but never silently deletes) values that look physically impossible, and groups readings into small time-windows the machine-learning models can consume. Bad data is quarantined and kept for inspection, never just thrown away. |

### 6.3 `src/synchronization/` — Module 3: keeping the twin's memory always up to date

| File | What it's for |
|---|---|
| `src/synchronization/d1_interface.py` | Defines the "contract" for how cleaned telemetry gets written into the digital twin's memory (see D1 below) — this lets the writing logic and the storage logic be built/tested independently. |
| `src/synchronization/sync.py` | Runs continuously in the background, forever pulling in fresh telemetry and pushing it into the twin's memory — with no "pause" button. This is what makes the digital twin's state genuinely live instead of a snapshot that gets stale. |

### 6.4 `src/dt_models/` — D1 (the twin's memory) and Modules 5–10 (the prediction engine)

This is the biggest folder — it holds both the twin's "memory" and the actual prediction models.

| File | What it's for |
|---|---|
| `src/dt_models/component_registry.py` | A small internal catalog the twin's memory (D1) uses to keep track of what kinds of data tables it's storing (current readings vs. history vs. rejected/bad readings). |
| `src/dt_models/d1_model_store.py` | **The digital twin's actual memory.** Keeps the network's current state and its full history, saved permanently to disk, always growing as new telemetry arrives, never wiped or replaced wholesale. This is literally "D1" from the architecture diagram. |
| `src/dt_models/base.py` | The common "shape" every prediction model (throughput, latency, etc.) must follow — must be able to train itself, make predictions, evaluate how good it is, and save/load itself to disk. |
| `src/dt_models/regressors.py` | A small shared helper that builds the underlying machine-learning algorithm (either "XGBoost" or "Random Forest" — two well-known prediction techniques) for whichever prediction model asks for one. |
| `src/dt_models/model_registry.py` | A lightweight "which prediction models exist and in what order should they run" catalog used while the system is live (different from the *versioned* registry below — this one doesn't track history, just "what's currently wired up"). |
| `src/dt_models/orchestrator.py` | **The conductor.** Figures out the correct order to run the five prediction models in (since some depend on others' output — e.g. "jitter" needs "throughput" and "latency" to already be predicted), then runs them all in that order automatically. |
| `src/dt_models/throughput.py` | Predicts **throughput** (how many Mbps a phone is actually getting) from things like signal quality and how many other phones are competing for the same tower. |
| `src/dt_models/latency.py` | Predicts **latency** (the delay in milliseconds) — depends on the throughput prediction, since a busier connection tends to be slower to respond. |
| `src/dt_models/packet_loss.py` | Predicts **packet loss** (the percentage of data that gets lost in transit) from signal quality and congestion. |
| `src/dt_models/prb_utilization.py` | Predicts **PRB utilization** — basically, "how full is the cell tower's radio capacity right now," a key congestion indicator. Can be switched off independently of the other four without breaking anything. |
| `src/dt_models/jitter.py` | Predicts **jitter** (how inconsistent the delay is, tick to tick) — the most complex model, since it depends on throughput, latency, *and* packet loss all at once. |

### 6.5 `src/fidelity/` — Module 12: "how close are our predictions to reality?"

| File | What it's for |
|---|---|
| `src/fidelity/metrics.py` | The actual math formulas that measure prediction accuracy: two simple ones (average error size), and two more advanced ones that compare the overall *shape* of predicted vs. real data, not just point-by-point differences. All are plain, deterministic math — never touched by any AI. |
| `src/fidelity/evaluator.py` | Combines those four measurements into one single "**FidelityScore**" per prediction model (roughly: 1.0 = very accurate, going down/negative = increasingly inaccurate), using a rolling comparison against recent history so a single unlucky reading can't unfairly tank the score. This score is what the rest of the system uses to decide "is this model still good enough?" |

### 6.6 `src/drift/` — Module 11: noticing "something's wrong, look at component X"

| File | What it's for |
|---|---|
| `src/drift/base.py` | The common "shape" any drift-alert source must follow. |
| `src/drift/schema.py` | Defines exactly what a valid drift alert looks like (which component is affected, how severe it is, when it happened). |
| `src/drift/mock_drift_source.py` | A fake-but-structured generator of drift alerts, used for testing/demo since a real external drift-detection system isn't part of this project (that piece is intentionally left as "someone else's specialized tool" per the original spec). |
| `src/drift/drift_detector.py` | Validates every incoming drift alert (rejecting malformed or nonsensical ones, e.g. "fix a component that doesn't exist") before letting it trigger anything downstream. |

### 6.7 `src/adaptation/` — Modules 13–17 and 19: the "notice, decide, fix, verify, record" brain

This is where the actual self-healing decisions and actions happen.

| File | What it's for |
|---|---|
| `src/adaptation/decision_context.py` | Gathers everything the AI decision-maker needs to know before it decides anything — the current accuracy numbers, what triggered the alarm, what was tried last time and how that went, and any relevant background knowledge pulled from the searchable library below. Just assembles the information; makes no decision itself. |
| `src/adaptation/decision_agent.py` | Asks the AI a real question every single time something goes wrong: "here's the situation — why do you think this happened, and which of the three fix strategies (Recalibrate / Regenerate / Expand Scope) should we use?" Never invents a fourth option, and never re-uses an old answer for a new problem. |
| `src/adaptation/data_selection.py` | Small shared helpers used by the three "fixer" agents below — picking a sensible recent slice of history to train on, splitting it into train/test portions fairly, and filling in a dependency model's needed inputs correctly. |
| `src/adaptation/recalibration_agent.py` | **Fix strategy #1 — Recalibrate.** The cheapest fix: just retrain the existing prediction model on fresh recent data. No AI language model needed — this is a plain retrain. Good for small, gradual drift. |
| `src/adaptation/regeneration_agent.py` | **Fix strategy #2 — Regenerate.** For bigger problems: asks the AI (a large language model) to rewrite the prediction model's entire approach from scratch, then tests the AI's new code safely (see sandbox below) before it's ever trusted. |
| `src/adaptation/expand_scope_agent.py` | **Fix strategy #3 — Expand Scope.** The biggest fix: asks the AI to design and build a *brand-new* prediction component the twin didn't have before, for a kind of network behavior it currently can't represent at all. |
| `src/adaptation/verification_agent.py` | **The strict referee.** After any of the three fixers above produces a candidate fix, this file independently re-checks — using the same plain math as `fidelity/evaluator.py`, never trusting the AI's own claims — whether the candidate is genuinely better than what's currently running. Only if it strictly passes does it get promoted to production. The AI may be asked to *explain* the decision in plain English afterward, but it can never change the decision itself. |
| `src/adaptation/lifecycle_agent.py` | **The record-keeper.** After every single fix attempt (whether accepted or rejected), this writes a permanent, timestamped record of exactly what happened — what triggered it, what the AI decided, what was tried, what the before/after accuracy was, and the final verdict — plus a friendly, human-readable summary report. Nothing here decides anything; it only documents. |

### 6.8 `src/llm/` — talking to the AI safely

| File | What it's for |
|---|---|
| `src/llm/google_client.py` | The single, shared connector to the AI (Google's Gemini). Every part of the system that needs to ask the AI something goes through this one file — so the API key, retry logic (what to do if the request fails), and safety checks only need to exist in one place instead of being copy-pasted everywhere. |

### 6.9 `src/rag/` — D2: a searchable reference library for the AI

**RAG** stands for "Retrieval-Augmented Generation" — basically "let the AI look things up before
answering," instead of relying only on what it already knows.

| File | What it's for |
|---|---|
| `src/rag/rag_kb.py` | A read-only, searchable knowledge library (built from the files in `rag_data/`, see below) that the AI agents can consult for extra context — e.g. relevant network-standards documentation or a summary of similar past fixes — when writing an explanation or generating code. Deliberately **read-only**: nothing in the rest of the system can write into it, only search it, so it can never become a hidden way for the AI to change real system state. |

### 6.10 `src/registry/` — remembering every version of every prediction model ever made

| File | What it's for |
|---|---|
| `src/registry/model_registry.py` | The permanent, versioned history book for every prediction model. Every time a new candidate is trained (recalibrated, regenerated, or newly invented), it's registered here with its own version number. Keeps track of which version is currently "live," and keeps every past version around so the system could always be rolled back if needed. Nothing is ever silently overwritten or deleted. |

### 6.11 `src/sandbox/` — the safety cage for AI-written code

| File | What it's for |
|---|---|
| `src/sandbox/executor.py` | Runs any AI-generated code in a completely separate, isolated process — never inside the main program — so that even badly-behaved or buggy AI-written code can't crash the real system, read its secrets, or touch real production files. |
| `src/sandbox/_sandbox_driver.py` | The actual script that runs *inside* that isolated process. Checks the AI's code step by step (does it even parse? does it follow the required structure? does it train successfully? does it produce sane predictions?) and reports back a clear pass/fail result with a reason. |

### 6.12 `src/storage/`

| File | What it's for |
|---|---|
| `src/storage/__init__.py` | Currently an empty placeholder Python package. Reserved in case a more general storage helper is needed later; nothing lives here yet — the real storage logic lives in `src/dt_models/d1_model_store.py` and `src/registry/model_registry.py`. |

### 6.13 The top-level conductor

| File | What it's for |
|---|---|
| `src/main.py` | **The file that starts and runs the entire system end-to-end.** Wires every piece above together into one continuous loop: keep listening to telemetry forever in the background, periodically check prediction accuracy, and whenever a drift alert comes in, run the full "AI decides → fix is built → fix is verified → outcome is recorded" cycle — all while telemetry keeps flowing in the background, never pausing. This is what you actually run (`python -m src.main`) to bring the whole digital twin to life. |

---

## 7. `scripts/` — one-off tools you run by hand

These aren't part of the always-running system — they're utilities you run yourself for setup,
training, or demonstrations.

| File | What it's for |
|---|---|
| `scripts/setup.py` | Checks that your local computer is properly set up (right folders exist, `.env` is filled in, all required software libraries are installed) — run this once after installing dependencies. |
| `scripts/setup_ns3.sh` | Downloads and builds the ns-3/5G-LENA network simulator (a big one-time step, since it's a full C++ project). |
| `scripts/ingest_rag.py` | Reads every document in `rag_data/` and loads it into the searchable AI reference library (`src/rag/rag_kb.py`'s storage) — run this once (or whenever the documents change) so the AI agents have something to search. |
| `scripts/run_orchestrator_demo.py` | A ready-to-run demo that starts the whole system using **fake (mock) telemetry**, lets it run through one full "drift → fix → verify → record" cycle unattended, and prints exactly what happened — good for a quick, fast sanity check. |
| `scripts/run_e2e_demo.py` | The more serious demo — starts the **real** ns-3/5G-LENA network simulator as an actual running program, connects the whole system to its real live telemetry, and runs a full real adaptation cycle against genuinely simulated 5G behavior (not fake data). |
| `scripts/visualize_metrics.py` | A stand-alone charting tool. Reads whatever real data the system has already produced (telemetry history, prediction accuracy, past fix outcomes) and draws PNG graphs — network metrics over time, prediction-accuracy over time, and a before/after chart of every automatic fix that's ever been attempted. Never runs the system itself, only reads and draws pictures of what already happened. |

---

## 8. `rag_data/` — the documents the AI can search through

Plain text/Markdown reference material, loaded into the searchable library by
`scripts/ingest_rag.py`.

| File | What it's for |
|---|---|
| `rag_data/oran/architecture_overview.md` | Real public documentation explaining O-RAN (an industry-standard way of building flexible, software-driven mobile networks) — background context the AI can reference. |
| `rag_data/oran/a1_interface.md` | Real public documentation about one specific piece of the O-RAN standard (the "A1" interface, used for sending policies to network components). |
| `rag_data/digital_twin/architecture_overview.md` | A description of *this project's own* architecture (based on `CLAUDE.md`) — lets the AI explain decisions using this project's actual design, not guesses. |
| `rag_data/digital_twin/adaptation_agents_and_lifecycle.md` | A description of *this project's own* six AI agents and how they work together — same idea as above, focused specifically on the self-healing part. |
| `rag_data/policies/adaptation_policies.md` | A plain-English summary of this project's actual current settings (from `config/settings.yaml`) — e.g. how big a fidelity improvement is required before a fix is accepted — so the AI's explanations stay accurate to the real configured rules. |
| `rag_data/history/development_validation_records.md` | A record of real events from this project's own development/testing — openly labeled as a thin starting example, since genuine long-term production history doesn't exist yet. |
| `rag_data/chroma/` *(not tracked in git)* | The actual searchable database built from all the documents above — regenerated automatically by `scripts/ingest_rag.py`, not something to edit by hand. |

---

## 9. `docs/` — polished write-ups and deliverables

| File | What it's for |
|---|---|
| `docs/HOW_TO_RUN.md` | Step-by-step instructions: install dependencies, run the tests, build the network simulator, run the whole system. This is the "just tell me the commands" file. |
| `docs/MANUAL.md` | This file — plain-English explanation of every file. |
| `docs/FORMULAS.pdf` | An 11-page PDF collecting every mathematical formula actually used in this project (prediction-accuracy math, the reinforcement-learning math, telemetry-derived formulas) with the exact source-code location each one came from. |
| `docs/IEEE_Digital_Twin_Paper.docx` | A formal, publication-style research paper describing this project's design and real measured results, written in the standard two-column academic format used by IEEE journals/conferences — a ready-to-open Microsoft Word version. |

---

## 10. `data/` — everything the system creates while it runs

This whole folder is **generated output**, not something you write by hand — it's excluded from
git (see `.gitignore`) because it's just runtime state that can always be regenerated. Listed here
so you know what you're looking at if you open it:

| Path | What lives there |
|---|---|
| `data/bootstrap/` | The one-time starter dataset used to train each prediction model for the very first time, before any real telemetry has accumulated. |
| `data/artifacts/d1_current_state.parquet` / `d1_history.parquet` / `d1_quarantine.parquet` | The digital twin's actual memory (D1) — current readings, full history, and any rejected/bad readings — saved to disk. |
| `data/artifacts/lifecycle_records.jsonl` | The permanent audit log of every single automatic fix attempt ever made (Module 19's output). |
| `data/artifacts/maintenance_reports/` | Human-readable summary reports, one per fix attempt. |
| `data/artifacts/plots/` | Charts produced by `scripts/visualize_metrics.py`. |
| `data/models/` | Every version of every trained prediction model. |
| `data/e2e_ns3_demo/` | A completely separate copy of all of the above, used only by `scripts/run_e2e_demo.py` so its real-simulator run never gets mixed up with fake/mock-telemetry runs. |

---

## 11. `tests/` — proof that everything actually works

Every meaningful piece of behavior described above has an automated test proving it really works,
not just that it looks right. Tests are split into two kinds:

- **`tests/unit/`** — fast tests that check one small file/function in isolation.
- **`tests/integration/`** — slower tests that run several real pieces together (e.g. actually
  training a real model, actually running a real background thread), to prove the pieces
  genuinely work *together*, not just individually.

As a rule of thumb, **each test file checks the source file with the matching name** — e.g.
`tests/unit/test_throughput_model.py` tests `src/dt_models/throughput.py`,
`tests/integration/test_regeneration_agent.py` tests `src/adaptation/regeneration_agent.py`
running for real (including a real sandboxed run), and so on. A few helper files support the
tests themselves rather than testing anything on their own:

| File | What it's for |
|---|---|
| `tests/dummy_dt_components.py` | Three intentionally trivial, fake prediction models (a doubler, an adder, a summer) used only to test the general "run models in the right order" orchestration logic (`src/dt_models/orchestrator.py`) without needing a real, slow machine-learning model. |
| `tests/integration/conftest.py` | Shared setup code several integration tests reuse — most importantly, generating one realistic batch of bootstrap telemetry data once per test run, so every model's training test doesn't have to regenerate it separately. |

The full list of test files, grouped by what they check:

| Area | Test files |
|---|---|
| Configuration & logging | `test_config.py`, `test_logging.py` |
| Telemetry (Module 2) | `test_mock_source.py`, `test_zmq_source.py`, `test_preprocessing.py`, `test_telemetry_pipeline.py` |
| Continuous sync (Module 3) | `test_sync.py`, `test_d1_interface.py`, `test_continuous_sync.py`, `test_d1_synchronization.py` |
| Twin memory & orchestration (D1/Module 5) | `test_d1_model_store.py`, `test_component_registry.py`, `test_dt_model_registry.py`, `test_orchestrator.py`, `test_orchestrator_with_d1.py`, `test_full_dependency_chain.py` |
| Prediction models (Modules 6–10) | `test_throughput_model.py`/`test_throughput_training.py`, `test_latency_model.py`/`test_latency_training.py`, `test_packet_loss_model.py`/`test_packet_loss_training.py`, `test_prb_utilization_model.py`/`test_prb_utilization_training.py`, `test_jitter_model.py`/`test_jitter_training.py` |
| Drift alerts (Module 11) | `test_mock_drift_source.py`, `test_drift_detector.py`, `test_drift_pipeline.py` |
| Fidelity scoring (Module 12) | `test_fidelity_metrics.py`, `test_fidelity_evaluator.py`, `test_fidelity_evaluation.py` |
| AI decision-maker (Module 13) | `test_decision_context.py`, `test_decision_agent.py` |
| Fix strategies (Modules 14–16) | `test_recalibration_agent.py` (x2), `test_regeneration_agent.py` (x2), `test_expand_scope_agent.py` (x2) |
| Sandbox safety cage | `test_sandbox_executor.py` |
| Talking to the AI | `test_google_client.py` |
| Searchable AI library (D2) | `test_rag_kb.py` (x2) |
| Versioned model history | `test_model_registry.py` |
| Data-selection helpers | `test_data_selection.py` |
| Strict referee (Module 17) | `test_verification_agent.py` (x2) |
| Record-keeper (Module 19) | `test_lifecycle_agent.py` (x2) |
| Whole-system wiring (`src/main.py`) | `test_main.py`, `test_main_orchestrator.py` |

---

## 12. Quick glossary

- **Telemetry** — the raw stream of measurements coming out of the network (speed, delay, signal
  strength, etc.), the same idea as a car's dashboard sending live speed/fuel readings.
- **Digital twin** — a live computer copy of a real system that tries to mirror what the real
  thing is doing right now.
- **Fidelity** — how accurate the twin's predictions currently are, compared to reality.
- **Drift** — when a prediction model's accuracy has degraded enough that it needs fixing.
- **LLM** — "Large Language Model," a general-purpose AI that understands and writes text/code —
  in this project, that's Gemini, made by Google.
- **RAG** — "Retrieval-Augmented Generation" — letting an AI search a document library before
  answering, instead of only using what it already knows.
- **Sandbox** — an isolated, safe place to run code you don't fully trust yet (here: AI-written
  code), so it can't damage anything real even if it misbehaves.
- **Fidelity gate / verification** — the strict, non-AI final check that a proposed fix is
  genuinely better before it's allowed to go live.
- **Lifecycle record** — the permanent written history of one complete "something went wrong, here's
  what we did about it, here's whether it worked" event.
- **Mock** — fake-but-realistic stand-in data, used when the real thing (a real 5G network) isn't
  available — always clearly labeled as mock, never confused with the real thing.
