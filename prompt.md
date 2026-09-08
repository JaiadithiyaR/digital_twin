# MASTER CLAUDE CODE IMPLEMENTATION PROMPT

## AI-Driven Self-Adaptive Network Digital Twin

You are Claude Code acting as the **lead software architect, ML engineer, RL engineer, LLM engineer, network-simulation engineer, and implementation engineer** for this project.

Your task is to **CREATE A COMPLETELY NEW REPOSITORY FROM SCRATCH** and implement the entire system described below.

The development machine has 16 GB ram so the implementation must be memory conscious. 

This is not a refactoring task.

There is **NO existing application codebase to assume**.

Do not assume that any source files, classes, models, databases, agents, APIs, configuration, simulator integration, RL environment, RAG system, or infrastructure already exists.

Build the project completely from the ground up.

\---

# 0\. PRIMARY OBJECTIVE

Build a **production-quality, modular AI-Driven Self-Adaptive Network Digital Twin platform** capable of:

1. Receiving network telemetry from an NS-3/5G-LENA simulated network.
2. Cleaning and preprocessing the telemetry.
3. Continuously synchronizing the Digital Twin with the latest network state.
4. Maintaining current and historical Digital Twin state.
5. Running dependency-aware DT prediction models.
6. Comparing DT predictions against ground-truth network behaviour.
7. Computing deterministic per-component fidelity.
8. Receiving drift events from an external drift detector.
9. Using a trained PPO agent to select:

   * Recalibration
   * Regeneration
   * Expand Scope
10. Autonomously executing the selected adaptation.
11. Using Anthropic Claude where LLM reasoning/code generation is required.
12. Running all generated candidates in a sandbox before they can affect production.
13. Verifying candidates against controlled evaluation data.
14. Automatically accepting or rejecting adaptations.
15. Promoting accepted versions and preserving rejected versions safely.
16. Continuing operation with the updated Digital Twin.
17. Recording every adaptation in an auditable lifecycle history.
18. Using a read-only RAG knowledge base for contextual reasoning.
19. Operating continuously without requiring human intervention during normal adaptation.

The resulting repository must contain **actual executable code**, not pseudocode or architectural placeholders.

\---

# CRITICAL IMPLEMENTATION OVERRIDES — READ FIRST

The following rules are NON-NEGOTIABLE and take precedence over any ambiguous wording elsewhere in this specification.

These rules exist to eliminate architectural ambiguity and ensure that the implementation is a genuinely executable, production-quality system rather than a Python-only prototype.

---

# 0.1 — NEW REPOSITORY: ABSOLUTE REQUIREMENT

This project MUST be created as a completely new repository from scratch.

You own the creation of the repository.

You must:

1. create the repository directory
2. initialize Git
3. create the complete project structure
4. create the Python virtual environment
5. install and verify dependencies
6. create all source code
7. create all configuration
8. create all tests
9. create all documentation
10. create the NS-3/5G-LENA simulator implementation
11. build and execute whatever can actually be executed
12. validate the resulting system

Do NOT assume an existing application, repository, NS-3 scenario, simulator code, telemetry exporter, model implementation, database, registry, RL environment, RAG system, or adaptation framework exists.

Do NOT merely create a Python framework around a hypothetical simulator.

The repository itself must contain the simulator-side implementation required by this project.

---

# 0.2 — NS-3 / 5G-LENA IS YOUR RESPONSIBILITY

This is a CRITICAL clarification.

You are responsible for implementing the NS-3 / 5G-LENA simulator side of this project, not merely documenting an integration boundary.

The NS-3 subsystem is a first-class implementation component of this repository.

You must:

1. inspect the available environment for NS-3
2. inspect whether 5G-LENA is available
3. determine compatible versions
4. create the required NS-3 simulation scenario
5. configure the required 5G network topology
6. configure UEs/cells/traffic/network conditions
7. implement telemetry extraction from the simulator
8. expose telemetry through a structured machine-readable interface
9. implement the ZeroMQ producer if ZeroMQ is used
10. connect the simulator producer to the Python telemetry consumer
11. build the simulator
12. execute the simulator
13. verify that telemetry is actually emitted
14. verify that Python actually receives the telemetry
15. verify that the telemetry reaches the DT synchronization layer

The NS-3 implementation must not be represented by pseudocode, TODOs, comments saying "implement this later", or an interface with no real simulator implementation.

---

# 0.3 — NS-3 REALISM REQUIREMENT

The simulator must produce genuine network behaviour.

Do not create a Python process that pretends to be NS-3.

Do not generate synthetic telemetry inside the NS-3 integration layer and describe it as NS-3 output.

The intended production data path is:

NS-3 + 5G-LENA
↓
actual simulated network behaviour
↓
actual simulator telemetry extraction
↓
ZeroMQ / structured transport
↓
Python telemetry ingestion
↓
preprocessing
↓
continuous DT synchronization

A synthetic/mock telemetry source is allowed ONLY as a separate development/testing source.

The mock source must never masquerade as real NS-3 telemetry.

Logs and metadata must clearly identify:

SOURCE=NS3_5G_LENA

or

SOURCE=MOCK

as appropriate.

---

# 0.4 — NS-3 EXTERNAL DEPENDENCY FAILURE

If NS-3 or 5G-LENA cannot be installed or executed because the environment lacks a required external dependency, do NOT fabricate success.

Instead:

1. implement the complete simulator-side code that can be implemented
2. implement the Python-side integration
3. implement the telemetry contract
4. implement the ZeroMQ transport
5. implement the mock source
6. attempt the real simulator build
7. record the exact command executed
8. record the exact failure
9. identify the missing dependency/toolchain/version
10. validate everything else that can genuinely be validated

The final report must explicitly distinguish:

REAL NS-3 VALIDATION

from

MOCK/DEMO VALIDATION.

Never report a mock run as a successful NS-3 run.

---

# 0.5 — THREE DISTINCT STATE CONCEPTS

Maintain a strict separation between:

A. LIVE NETWORK STATE

The current state emitted by NS-3/5G-LENA.

B. DYNAMIC DIGITAL TWIN STATE

The continuously synchronized representation of the latest and historical network state.

C. VERSIONED DT PREDICTION MODELS

The independently versioned learned prediction implementations used by the DT.

These concepts MUST NOT be conflated.

The Digital Twin state is continuously updated.

The prediction model implementation is independently versioned.

Updating a model MUST NOT replace or freeze the live DT state.

---

# 0.6 — ABSOLUTE DYNAMIC STATE REQUIREMENT

The Digital Twin must remain dynamically synchronized for the entire lifetime of the application.

Every valid incoming telemetry record/batch must be capable of updating the appropriate current state.

The runtime architecture must NOT behave like:

collect dataset
→ periodically replace DT
→ retrain everything
→ replace state

Instead it must behave like:

NS-3 telemetry
→ validate
→ preprocess
→ synchronize
→ update current DT state
→ append historical state
→ generate predictions using currently deployed model versions
→ compare against ground truth
→ calculate fidelity
→ process drift
→ optionally create isolated candidate
→ verify candidate
→ promote/reject candidate
→ continue receiving telemetry

While adaptation is occurring:

THE LIVE DT MUST CONTINUE RECEIVING TELEMETRY.

THE LIVE DT MUST CONTINUE UPDATING.

THE CURRENT PRODUCTION MODEL MUST REMAIN AVAILABLE.

ONLY THE CANDIDATE MODEL AND ITS EVALUATION CONTEXT ARE ISOLATED.

Never freeze the entire DT merely because a model adaptation is taking place.

---

# 0.7 — TELEMETRY MUST BE THE AUTHORITATIVE RUNTIME STATE UPDATE

Do not allow stale bootstrap data, cached state, or historical snapshots to silently become the current DT state.

Bootstrap data is used for initialization/training where explicitly required.

Live telemetry is authoritative for runtime current-state synchronization.

Historical data is retained for analysis, training-window selection, fidelity evaluation, and lifecycle history.

Clearly distinguish:

bootstrap data
historical data
live data
evaluation data
candidate-training data.

Every dataset used by a model must have provenance metadata.

---

# 0.8 — CONTINUOUS OPERATION DURING ADAPTATION

Adaptation must not stop the telemetry pipeline.

During:

recalibration
regeneration
expand-scope
sandbox execution
candidate training
candidate evaluation
verification

the system must continue accepting and synchronizing live telemetry whenever technically possible.

The production model remains the active model until a candidate is deterministically accepted and promoted.

Therefore:

LIVE STATE
↓
continues updating

while simultaneously:

CURRENT PRODUCTION MODEL
↓
continues serving predictions

and independently:

CANDIDATE MODEL
↓
isolated training/evaluation
↓
verification
↓
promotion or rejection.

---

# 0.9 — BASELINE / CANDIDATE EVALUATION SEMANTICS

"Before" means:

the currently deployed production model evaluated against the controlled evaluation window selected for the adaptation event.

"After" means:

the candidate model evaluated against that same comparable evaluation protocol and evaluation data.

Do NOT compare the candidate against a different network condition merely because the live DT has continued evolving.

The evaluation context must therefore capture:

* evaluation window
* sample identifiers where available
* component
* production version
* candidate version
* relevant configuration
* dataset provenance
* timestamp
* preprocessing version
* feature schema version.

The live DT may continue changing outside this isolated evaluation context.

---

# 0.10 — FIDELITY NORMALIZATION MUST BE COMPARABLE

For each component and each fidelity metric:

1. compute the raw metric
2. square the metric as specified
3. obtain rolling historical normalization statistics
4. normalize using the defined min-max formula
5. calculate the composite fidelity score.

The normalization reference must be explicitly identified.

When comparing production versus candidate:

BOTH MUST USE THE SAME NORMALIZATION REFERENCE.

Do not independently recompute min/max values for production and candidate if that would make the scores incomparable.

Handle:

* constant windows
* insufficient history
* NaN
* infinity
* empty windows
* missing samples

explicitly.

Never silently manufacture a fidelity score.

---

# 0.11 — PPO MUST BE A GENUINE TRAINED POLICY

PPO must not be implemented as a disguised rule-based system.

The Gymnasium environment must have a meaningful adaptation-learning formulation.

The environment must explicitly define:

* observation space
* action space
* transition dynamics
* adaptation outcome
* reward calculation
* episode termination
* reset behaviour.

The PPO training environment must expose meaningful variation in:

* affected component
* fidelity degradation
* drift severity
* previous action
* previous reward
* network conditions
* adaptation outcome.

The trained PPO policy must actually be saved and subsequently loaded for runtime inference.

Runtime action selection must come from the trained PPO policy.

A deterministic fallback is allowed ONLY when the PPO infrastructure itself fails.

The fallback must be logged.

The fallback must never silently replace PPO during normal operation.

---

# 0.12 — EXACT PPO RESPONSIBILITY

PPO decides:

WHAT adaptation strategy should be selected.

PPO does NOT generate code.

PPO does NOT directly modify model parameters.

PPO does NOT determine fidelity formulas.

PPO does NOT perform LLM reasoning.

PPO selects exactly one of:

0 = Recalibrate
1 = Regenerate
2 = Expand Scope.

The selected action is then passed to the corresponding adaptation agent.

---

# 0.13 — EXACT AGENT COUNT

There must be exactly SIX agents:

1. PPO RL Decision Agent
2. Recalibration Agent
3. Regeneration Agent
4. Expand-Scope Agent
5. Agentic Verification Agent
6. Lifecycle Management Agent

Do not create additional agents merely by renaming ordinary software components.

Telemetry ingestion, synchronization, DT models, orchestrators, registries, sandboxes, RAG, storage, configuration, and API clients are NOT agents.

---

# 0.14 — LLM CODE GENERATION MUST BE REAL

Regeneration and Expand Scope must genuinely use the Anthropic API when enabled.

Do not create a hardcoded implementation template and call it "LLM generation".

The LLM must receive structured context and produce structured candidate artifacts.

However, LLM output is always untrusted.

LLM output must NEVER directly become production code.

The mandatory path is:

LLM generation
→ candidate workspace
→ syntax validation
→ static validation
→ import validation
→ unit tests
→ training
→ evaluation
→ deterministic verification
→ promotion.

---

# 0.15 — CANDIDATE ISOLATION

Candidate code and artifacts must be physically/logically separated from production artifacts.

A candidate must not have unrestricted writable access to:

* production source
* production model artifacts
* production registry
* production configuration
* secrets
* unrelated filesystem locations.

A candidate must never be able to promote itself.

Promotion is performed only by trusted deterministic application logic after verification.

---

# 0.16 — EXPAND-SCOPE SAFETY

Expand Scope may dynamically introduce a new DT component.

However, every generated component MUST:

1. implement the common DT interface
2. declare its input schema
3. declare its output schema
4. declare dependencies
5. declare its feature requirements
6. pass registry validation
7. pass dependency-graph validation
8. pass sandbox validation
9. pass tests
10. train successfully
11. evaluate successfully
12. pass deterministic verification
13. receive a version
14. only then become eligible for promotion.

Expand Scope must not be permitted to redesign unrelated components or bypass the existing architecture.

---

# 0.17 — DEPENDENCY GRAPH MUST BE EXECUTABLE

The DT dependency graph is not merely documentation.

It must be represented as executable metadata.

The orchestrator must derive a valid topological execution order from the registry.

At minimum:

raw telemetry
↓
throughput ─────┐
├→ latency
packet loss ────┘
↓
throughput + packet loss + latency
↓
jitter

PRB utilization remains independently configurable.

If a dependency is disabled or unavailable, the orchestrator must detect the resulting invalid dependency graph and fail explicitly rather than silently producing incorrect predictions.

---

# 0.18 — MODEL TRAINING RULE

Initial models are trained during bootstrap.

Normal telemetry processing must NOT automatically retrain models.

Retraining/recalibration/replacement occurs only through an explicit adaptation workflow.

Therefore:

telemetry
→ prediction

does NOT imply:

telemetry
→ retraining.

---

# 0.19 — DRIFT DETECTION REMAINS EXTERNAL

Do not implement the actual drift-detection algorithm.

Implement:

* external event contract
* schema validation
* normalization
* adapter
* mock source.

A mock drift source exists only for testing.

Never claim that the mock drift source represents a real drift detector.

---

# 0.20 — DETERMINISTIC ACCEPTANCE GATE

The LLM may explain a candidate.

The LLM may identify contextual concerns.

The LLM may provide root-cause reasoning.

The LLM MUST NOT determine the final numerical acceptance result.

Promotion requires deterministic checks.

At minimum:

FidelityScore_new > FidelityScore_old + δ

plus:

* candidate loads
* interface valid
* tests pass
* evaluation succeeds
* no critical regression
* outputs are valid
* safety checks pass.

If any mandatory condition fails:

REJECT.

The LLM cannot override this.

---

# 0.21 — PRODUCTION VERSION SAFETY

The currently deployed version must always remain recoverable.

Promotion must be atomic from the perspective of the production registry.

Never destructively overwrite the previous production artifact.

Maintain:

current version
previous known-good version(s)
candidate version
artifact metadata
training metadata
evaluation metadata
verification metadata.

A failed candidate must have zero production effect.

A later failure of a promoted version must permit rollback to a known-good version.

---

# 0.22 — ADAPTATION CONCURRENCY

Only one adaptation may modify a particular component at a time.

Implement a component-scoped adaptation lock.

If a second drift event arrives for the same component while adaptation is active:

do NOT start a competing modification.

Handle it deterministically using an explicit policy such as:

queue
coalesce
or defer.

The policy must be documented and tested.

Telemetry ingestion itself must continue.

---

# 0.23 — SOURCE-OF-TRUTH ARCHITECTURE

The authoritative architectural flow is:

NS-3 / 5G-LENA
→ telemetry extraction
→ transport
→ telemetry validation
→ preprocessing
→ continuous synchronization
→ dynamic D1 state
→ dependency-aware DT prediction
→ fidelity evaluation
→ external drift event
→ PPO
→ selected adaptation agent
→ candidate
→ sandbox
→ deterministic evaluation
→ agentic contextual verification
→ deterministic acceptance gate
→ promotion/rejection
→ lifecycle management
→ continued operation.

Do not bypass stages.

Do not silently reorder stages.

Do not introduce an alternative simplified architecture.

---

# 0.24 — REAL VALIDATION REQUIREMENT

Before declaring completion, actually execute:

1. static/lint checks
2. unit tests
3. integration tests
4. end-to-end mock/demo pipeline
5. PPO training/inference validation
6. candidate sandbox validation
7. adaptation validation
8. rejection validation
9. rollback/preservation validation.

If the NS-3/5G-LENA environment is executable, additionally execute:

NS-3 build
→ NS-3 run
→ telemetry emission
→ ZeroMQ transport
→ Python ingestion
→ preprocessing
→ synchronization
→ DT prediction
→ fidelity
→ drift
→ PPO
→ adaptation
→ candidate validation
→ verification
→ lifecycle recording.

Inspect actual outputs.

Do not infer success merely because a command exited successfully if the expected telemetry/data/artifact was not actually produced.

---

# 0.25 — NO FALSE COMPLETION

Never claim:

* NS-3 worked unless NS-3 actually ran
* telemetry was emitted unless telemetry was observed
* PPO was trained unless training actually ran
* an LLM generated a candidate unless the API call actually occurred
* candidate verification passed unless verification actually ran
* an adaptation succeeded unless promotion actually occurred
* rollback worked unless rollback was actually tested.

Distinguish clearly between:

IMPLEMENTED
VALIDATED
PARTIALLY VALIDATED
BLOCKED BY EXTERNAL DEPENDENCY.

---

# 0.26 — PRODUCTION-READINESS REVIEW BEFORE COMPLETION

Before declaring completion, perform a final architecture review specifically looking for:

* ambiguous data ownership
* stale-state usage
* race conditions
* circular dependencies
* dependency-order violations
* data leakage
* train/evaluation contamination
* candidate/production contamination
* unsafe filesystem access
* secret leakage
* silent exception handling
* missing validation
* invalid numerical states
* NaN/Inf propagation
* configuration inconsistencies
* versioning inconsistencies
* incorrect rollback semantics
* incorrect PPO action mapping
* hidden heuristic replacement of PPO
* fake LLM generation
* fake NS-3 behaviour
* fake drift detection
* mock/real source confusion
* broken disabled-component behaviour
* missing lifecycle records
* incomplete error handling
* missing tests.

Fix discovered problems before completion.

Do not merely document a logical error if it can be fixed.

---

# 0.27 — IMPLEMENTATION PRIORITY

When choosing between speed and correctness:

1. correctness
2. safety
3. dynamic state integrity
4. actual simulator integration
5. deterministic verification
6. end-to-end functionality
7. tests
8. maintainability
9. documentation
10. optimization.

Do not reduce architectural correctness merely to finish faster.

---

# 0.28 — FINAL OPERATING INSTRUCTION

Do not stop after generating files.

Implement.

Build.

Run.

Inspect.

Test.

Diagnose.

Fix.

Repeat.

Continue until the complete system is implemented and validated to the maximum extent permitted by the actual environment.

If an external dependency genuinely prevents full validation, clearly isolate that blocker and continue implementing and validating everything else.

The goal is not a convincing demonstration.

The goal is a real, executable, production-quality implementation of the specified architecture.

# 1\. SOURCE OF TRUTH

Use the following as the authoritative specification:

### A. Original project requirements

The complete functional requirements supplied for this project.

### B. Data Flow Diagram

The 19-module architecture:

1. NS-3/OAI Network
2. Telemetry Collection \& Preprocessing
3. Continuous Synchronization
4. D1 — DT Basic Model
5. DT Functional Model
6. Throughput Model
7. Latency Model
8. Packet-Loss Model
9. PRB-Utilization Model
10. Jitter Model
11. Drift Detection Interface
12. Fidelity Evaluation
13. RL Decision Agent
14. Recalibration Agent
15. Regeneration Agent
16. Expand-Scope Agent
17. Agentic Verification Agent
18. D2 — RAG Knowledge Base
19. Lifecycle Management Agent

The DFD explicitly defines telemetry → synchronization → DT prediction → fidelity → drift → PPO adaptation → verification → lifecycle recording. Preserve this architecture.

### C. Decisions explicitly defined in this prompt

Where any old requirement, example, or implementation detail conflicts with this prompt, **this prompt takes precedence**.

Do not silently invent alternative architecture.

\---

# 2\. ABSOLUTE IMPLEMENTATION RULE

## CREATE EVERYTHING FROM ZERO

Initialize a new Git repository.

Create the project structure yourself.

Do not write a "plan only".

Do not stop after generating skeleton files.

Do not leave TODOs for core functionality.

Do not replace difficult components with fake implementations while claiming completion.

Do not fabricate successful simulator execution, API calls, model training, or verification results.

Implement, execute, test, diagnose failures, and fix them.

Continue until the system is as complete and executable as the available environment permits.

If a genuinely external dependency cannot be installed or executed because the machine lacks a required system package/toolchain, clearly document that exact blocker while still implementing everything else that can be implemented locally.

\---

# 3\. AUTONOMY RULE

You have authority to make reasonable implementation-level decisions without asking me questions.

Do NOT repeatedly stop to ask:

* which Python framework
* which class name
* which database schema
* which logging library
* which test framework
* which file naming convention
* which internal API shape
* which reasonable implementation detail

Choose sensible production-quality defaults.

Only stop for a question if there is a **true external blocker that cannot be resolved autonomously**.

You have extremely limited execution/context budget.

Therefore:

* work autonomously
* avoid unnecessary explanations
* inspect efficiently
* implement rather than discuss
* do not repeatedly rediscover decisions already made
* preserve progress in repository documentation
* do not waste tokens rewriting completed work

\---

# 4\. IMPORTANT ARCHITECTURAL PRINCIPLE

The Digital Twin is **DYNAMIC**.

Do NOT design the DT as a static dataset that is periodically replaced.

The runtime DT continuously receives new telemetry and updates its current state.

However, the **deployed prediction model/component version** is independently versioned.

Therefore:

### Dynamic DT state

Continuously changes with:

* latest telemetry
* historical telemetry
* UE state
* cell state
* network configuration
* derived features
* predictions

### Model version

Represents the currently deployed learned model implementation.

During adaptation:

```text
LIVE DYNAMIC DT
      |
      | current deployed model version
      ↓
BASELINE SNAPSHOT
      |
      | adaptation
      ↓
CANDIDATE MODEL VERSION
      |
      | controlled evaluation
      ↓
VERIFICATION
      |
      ├── ACCEPT → promote candidate
      |
      └── REJECT → keep current production model
```

The entire DT does NOT need to freeze.

Only the relevant model/component version and evaluation context are isolated.

This distinction is critical.

\---

# 5\. EXACT AGENT COUNT

There are exactly **six agents**.

## Agent 1 — PPO RL Decision Agent

Chooses WHAT adaptation strategy to use.

## Agent 2 — Recalibration Agent

Executes HOW an existing component should be recalibrated.

## Agent 3 — Regeneration Agent

Uses LLM reasoning/code generation to determine HOW an inadequate component should be regenerated.

## Agent 4 — Expand-Scope Agent

Uses LLM reasoning/code generation to determine HOW a missing DT capability should be created.

## Agent 5 — Agentic Verification Agent

Determines whether the candidate should be accepted, using deterministic numerical evaluation plus LLM reasoning/context where appropriate.

## Agent 6 — Lifecycle Management Agent

Records and explains every adaptation.

\---

# 6\. THINGS THAT ARE NOT AGENTS

Do NOT turn the following into additional agents:

* telemetry ingestion
* preprocessing
* synchronization
* DT models
* DT orchestrator
* fidelity engine
* drift detector
* RAG database
* model registry
* sandbox
* Anthropic API client
* storage
* configuration system

These are ordinary software components/services.

\---

# 7\. MODULE 1 — NS-3 / 5G-LENA NETWORK

Create the simulator integration required for ground-truth telemetry.

The intended environment is:

```text
NS-3 + 5G-LENA
        ↓
network behaviour
        ↓
telemetry
        ↓
Digital Twin
```

The simulator must expose telemetry containing, where available:

* timestamp
* UE ID
* cell ID
* throughput
* offered load
* latency
* jitter
* packet loss
* PRB utilization
* SINR
* RSRP
* RSRQ
* UE count
* UE speed
* UE position
* relevant network configuration/state

Prefer a structured streaming mechanism such as ZeroMQ for continuous telemetry.

Telemetry must be machine-readable.

Implement the integration cleanly enough that a real NS-3/5G-LENA producer can be used.

If the full external NS-3/5G-LENA toolchain is unavailable in the execution environment:

1. still implement the complete integration interface
2. implement a realistic mock/synthetic producer for testing
3. never claim that real NS-3 execution succeeded unless it actually did
4. document the exact external blocker

If the environment supports NS-3/5G-LENA, actually build and execute the simulator.

Verify that telemetry is genuinely emitted.

\---

# 8\. MODULE 2 — TELEMETRY COLLECTION \& PREPROCESSING

Implement a robust telemetry ingestion layer.

Responsibilities:

* receive telemetry
* validate schema
* normalize units
* handle missing values
* remove/flag invalid records
* synchronize timestamps
* construct feature windows
* preserve UE/cell identifiers
* maintain data quality metadata

Do not silently discard important data.

Use explicit validation and logging.

The preprocessing layer must output clean DT-ready records.

\---

# 9\. MODULE 3 — CONTINUOUS SYNCHRONIZATION

The synchronization service must continuously update the Digital Twin.

There must be:

**NO manual synchronization step.**

Conceptually:

```text
Telemetry
   ↓
Preprocessing
   ↓
Synchronization
   ↓
D1 current state + historical state
```

The synchronizer should:

* ingest new validated telemetry
* update latest network state
* append historical data
* maintain timestamps
* maintain UE/cell state
* make recent training/evaluation windows available
* expose consistent snapshots to downstream components

The DT remains continuously dynamic.

\---

# 10\. MODULE 4 — D1 DT BASIC MODEL

Create a persistent state/history layer.

It must maintain:

### Current state

* latest telemetry
* current network state
* UE state
* cell state
* configuration
* latest predictions

### Historical state

* historical telemetry
* historical predictions
* ground truth
* adaptation events
* evaluation windows

Use a practical tabular/time-series representation such as:

* Pandas
* Parquet
* lightweight database where appropriate

Do not introduce unnecessary infrastructure.

Implement a modular component registry.

\---

# 11\. MODULE 5 — DT FUNCTIONAL MODEL

All DT prediction components must implement a common interface.

At minimum:

```python
train(...)
predict(...)
evaluate(...)
save(...)
load(...)
```

The exact internal class design is yours to choose, but the interface must be consistent.

The DT orchestrator must execute components according to dependencies.

Do NOT simply call all five models independently from `main.py`.

Implement a topological/dependency-aware scheduler.

\---

# 12\. REQUIRED DT COMPONENTS

Implement these five components.

\---

## 12.1 Throughput Model

Inputs include:

* offered load
* PRB
* SINR
* RSRP/RSRQ
* UE count
* UE speed

Output:

```text
throughput (Mbps)
```

Use a robust classical ML regression model such as XGBoost or Random Forest.

The implementation must expose the common DT interface.

\---

## 12.2 Packet-Loss Model

Inputs include:

* SINR
* RSRP/RSRQ
* PRB
* UE count
* offered load

Output:

```text
packet loss (%)
```

Use a suitable regression approach.

\---

## 12.3 Latency Model

Inputs include:

* offered load
* throughput
* PRB
* UE count
* packet loss
* SINR

Output:

```text
latency (ms)
```

Latency depends on throughput and packet loss.

Therefore it must execute after the required upstream models.

\---

## 12.4 PRB Utilization Model

Inputs include:

* offered load
* UE count
* throughput
* cell load

Output:

```text
PRB utilization (%)
```

This model must be toggleable through configuration:

```yaml
enable\_prb\_model: true
```

It must not break the rest of the system when disabled.

\---

## 12.5 Jitter Model

Inputs include:

* latency
* throughput
* offered load
* packet loss
* UE count

Output:

```text
jitter (ms)
```

Jitter must execute after its upstream dependencies.

\---

# 13\. REQUIRED MODEL DEPENDENCY ORDER

Implement the dependency graph explicitly.

At minimum:

```text
Raw telemetry
      |
      +--------------------+
      |                    |
      ↓                    ↓
Throughput           Packet Loss
      |                    |
      +---------+----------+
                |
                ↓
             Latency

PRB Utilization
(independent/optional)

Latency + Throughput + Packet Loss
                |
                ↓
             Jitter
```

Do not violate these dependencies.

The orchestrator should calculate a valid execution order automatically from the component registry.

\---

# 14\. BOOT-TIME MODEL TRAINING RULE

Initial DT models are trained once during system initialization using historical/bootstrap data.

Normal operation must NOT continuously retrain every model.

During normal operation:

```text
telemetry → prediction → fidelity → drift → adaptation
```

Only an adaptation action should intentionally retrain or replace a model.

This prevents accidental continuous retraining.

\---

# 15\. MODULE 11 — DRIFT DETECTION INTERFACE

The actual drift detector is an **external/partner module**.

DO NOT implement the actual drift-detection algorithm as part of this project.

Implement the integration contract.

Expected event structure:

```json
{
  "component": "...",
  "severity": 0.0,
  "timestamp": "...",
  "metadata": {}
}
```

The interface must validate and normalize incoming drift events.

Create:

```text
drift\_detector.py
mock\_drift\_source.py
```

The mock source exists solely for development/integration testing.

The production architecture must allow the external detector to be connected later without redesigning the adaptation system.

\---

# 16\. MODULE 12 — FIDELITY ENGINE

This module is extremely important.

Fidelity calculations must be **deterministic**.

The LLM must NOT invent or modify fidelity formulas.

For every DT component calculate:

1. RMSE
2. MAE
3. Wasserstein distance / EMD
4. MK-MMD

Then square every metric:

```text
D\_RMSE = RMSE²
D\_MAE  = MAE²
D\_W1   = W1²
D\_MMD  = MK-MMD²
```

Perform min-max normalization over a rolling historical window:

```text
D\~m = (Dm - min(Dm))
      -----------------------------
      (max(Dm) - min(Dm) + ε)
```

Then:

```text
S\_raw,c =
    D\~RMSE
  + D\~MAE
  + D\~W1
  + D\~MMD
```

Finally:

```text
FidelityScore\_c =
    1 - (S\_raw,c / 4)
```

The implementation must:

* preserve numerical stability
* handle constant rolling windows
* handle NaN/invalid inputs explicitly
* make epsilon configurable
* make rolling-window length configurable
* never hardcode these values

Configuration must contain the fidelity parameters.

Create unit tests against known numerical examples.

\---

# 17\. FIDELITY WINDOW POLICY

Fidelity evaluation must use clearly defined comparable data.

When evaluating:

```text
current production model
vs
candidate model
```

both must be evaluated against the same controlled/comparable evaluation window wherever valid.

Do not compare models using unrelated network conditions.

Store evaluation metadata including:

* window start
* window end
* records/sample identifiers where possible
* component
* model version
* evaluation timestamp
* configuration
* fidelity metrics

For temporal data, avoid leakage.

Training data must not contain future evaluation information.

\---

# 18\. MODULE 13 — PPO RL DECISION AGENT

PPO is responsible for choosing **WHAT adaptation strategy to apply**.

The LLM must NOT replace PPO as the adaptation decision-maker.

Use:

```text
Stable-Baselines3 PPO
+
Gymnasium
```

Create:

```text
src/adaptation/rl\_env.py
src/adaptation/rl\_agent.py
```

\---

## PPO observation

The observation must represent the adaptation context.

Include:

* per-component fidelity vector
* affected-component one-hot flag
* drift severity
* previous action one-hot
* previous reward

Also include relevant network state where necessary and where the dimensionality is kept well-defined.

The core required state representation is:

```text
5 fidelity values
+
affected component indicator
+
drift severity
+
previous action
+
previous reward
```

Normalize observations appropriately.

\---

# 19\. PPO ACTION SPACE

Use:

```text
Discrete(3)
```

Mapping:

```text
0 → Recalibrate
1 → Regenerate
2 → Expand Scope
```

Do not add arbitrary fourth/fifth actions.

\---

# 20\. PPO REWARD

The primary reward is:

```text
new\_fidelity - old\_fidelity
```

Optionally incorporate a configurable adaptation-cost penalty:

```text
reward =
    fidelity\_improvement
    -
    adaptation\_cost\_penalty
```

The implementation must make this configurable.

\---

# 21\. PPO TRAINING VS RUNTIME

Train PPO through the actual Gymnasium environment.

Do not create a fake rule-based function such as:

```python
if severity > X:
    regenerate()
```

and call it PPO.

The RL agent must genuinely use the trained policy.

At runtime:

```text
drift
 ↓
observation
 ↓
trained PPO
 ↓
action
```

Do not retrain PPO on every drift event.

If online PPO updating is ever implemented, it must be explicitly isolated/configurable and must not destabilize the production policy.

The default runtime path should use the trained policy for inference.

Provide a deterministic fallback only for infrastructure failure, not as a replacement for PPO.

Log when such a fallback is ever used.

\---

# 22\. ADAPTATION ARCHITECTURE

The central distinction is:

```text
PPO decides WHAT
LLM-enabled adaptation agents determine HOW
```

Therefore:

```text
Drift
  ↓
PPO
  ↓
Recalibrate / Regenerate / Expand
  ↓
appropriate agent
  ↓
candidate
  ↓
verification
```

Do not reverse these responsibilities.

\---

# 23\. MODULE 14 — RECALIBRATION AGENT

Recalibration is used when:

> The existing model structure/pipeline is still appropriate, but its learned parameters/behaviour have become stale.

The Recalibration Agent must operate autonomously.

It must:

1. receive the affected component
2. inspect current model metadata
3. inspect recent DT/telemetry history
4. determine an appropriate recent training window
5. select the relevant training data
6. call the existing generic `train()` interface
7. produce a new candidate model version
8. evaluate it
9. pass it to verification

The agent may use LLM reasoning to determine training context/window if beneficial, but **the actual metric calculations and model training remain deterministic code**.

Do not perform blind periodic recalibration.

Recalibration happens because PPO selected it.

\---

# 24\. MODULE 15 — REGENERATION AGENT

Regeneration is used when:

> The current model architecture/pipeline is no longer capable of representing the changed behaviour.

This is the primary LLM code-generation component.

Use the **direct Anthropic API**.

Do NOT introduce LiteLLM or another LLM gateway unless there is a compelling technical necessity.

Create a centralized Anthropic client/service so all LLM calls are managed consistently.

The model name must be configurable.

Do not hardcode an obsolete/deprecated Claude model.

Use a currently supported Anthropic Claude model according to the current Anthropic API/documentation available at implementation time.

Never hardcode API keys.

Use environment variables/secrets.

\---

# 25\. REGENERATION INPUT TO LLM

The LLM may receive:

* current component source code
* component interface
* component metadata
* recent telemetry distributions
* feature statistics
* fidelity metrics
* error patterns
* drift metadata
* relevant RAG context
* training/evaluation constraints

The LLM must understand that it is modifying **one component**, not redesigning the entire project.

\---

# 26\. REGENERATION OUTPUT

The LLM must produce valid implementation artifacts.

Do not allow free-form conversational output to become executable code directly.

Use strict structured output where possible.

If source code is generated:

1. save it only to a candidate workspace
2. syntax-check it
3. import-check it
4. run unit tests
5. run model training
6. run evaluation
7. run verification

Never overwrite the production implementation directly.

\---

# 27\. MODULE 16 — EXPAND-SCOPE AGENT

Expand Scope is used when:

> Network behaviour reveals a phenomenon/capability that the current DT does not represent.

This agent is also LLM-driven.

It must autonomously:

1. inspect the component registry
2. inspect the common DT model interface
3. inspect existing components
4. inspect relevant telemetry/features
5. inspect RAG context
6. determine a suitable new DT component
7. derive its inputs
8. derive its output
9. define feature extraction requirements
10. generate the implementation
11. generate required tests
12. create training logic
13. train the candidate
14. register the candidate
15. evaluate it
16. submit it to verification

The new component must be added dynamically through the registry.

Do not hardcode every possible future component.

\---

# 28\. GENERATED CODE SAFETY

LLM-generated code is **untrusted candidate code**.

Never execute LLM-generated code directly inside the production process.

Use an isolated candidate workspace/sandbox.

At minimum perform:

```text
Generate
 ↓
Syntax check
 ↓
Static validation
 ↓
Import validation
 ↓
Unit tests
 ↓
Training
 ↓
Evaluation
 ↓
Verification
 ↓
Promotion
```

The production DT remains untouched until verification succeeds.

\---

# 29\. CANDIDATE VERSIONING

Every adaptation must produce an identifiable candidate version.

A version should contain enough metadata to identify:

* component
* parent production version
* candidate version
* adaptation type
* timestamp
* source code/artifact location
* training dataset/window
* evaluation dataset/window
* configuration
* LLM metadata where applicable
* fidelity before
* fidelity after
* verification result

Use immutable artifacts where practical.

\---

# 30\. CONCURRENCY / ADAPTATION LOCK

Prevent two adaptations from simultaneously modifying the same production component.

Implement an adaptation lock or equivalent concurrency control.

While candidate verification is running:

```text
production model = trusted
candidate model = isolated
```

Only one successful candidate can be promoted for a given component at a time.

Never corrupt the live DT because two adaptation processes raced.

\---

# 31\. MODULE 17 — AGENTIC VERIFICATION

Verification is autonomous.

It must not simply trust the LLM.

Verification has two layers:

### Deterministic layer

Recompute:

* RMSE
* MAE
* Wasserstein
* MK-MMD
* FidelityScore

using the exact fidelity engine.

### Agentic reasoning layer

The LLM may reason about:

* why the candidate improved/failed
* error distributions
* drift context
* model behaviour
* RAG knowledge
* whether additional contextual concerns exist

However:

**The numerical acceptance gate must remain deterministic.**

The LLM cannot override the deterministic fidelity gate.

\---

# 32\. ACCEPTANCE RULE

The primary acceptance criterion is:

```text
FidelityScore\_new >
FidelityScore\_old + δ
```

where:

```text
δ
```

is configurable.

Do not hardcode δ.

The verification system must also ensure:

* candidate successfully loads
* required interface works
* tests pass
* no critical regression
* no invalid output
* no unacceptable safety/quality condition

If accepted:

```text
ACCEPT
→ promote candidate
→ update registry/version
→ preserve previous version
→ lifecycle record
```

If rejected:

```text
REJECT
→ discard candidate from production
→ retain current production model
→ record rejection
→ record reason
```

\---

# 33\. IMPORTANT: "BEFORE" VS "AFTER"

The "before" fidelity score means:

> fidelity of the currently deployed production model at the adaptation event under the controlled evaluation protocol.

The "after" fidelity score means:

> fidelity of the candidate under the same comparable evaluation protocol.

Do NOT interpret "before" as a frozen copy of the entire Digital Twin.

The DT itself continues receiving telemetry dynamically.

\---

# 34\. ROLLBACK

The current production version must remain recoverable.

Do not perform destructive replacement.

Use versioned artifacts/model registry.

If generated source code is involved, maintain safe source/version boundaries.

A rejected candidate must never become production.

If a promoted candidate later causes a detected failure, the system must have enough version information to restore the previous known-good version.

\---

# 35\. MODULE 18 — RAG KNOWLEDGE BASE

Create a read-only knowledge base.

Use a practical vector database such as ChromaDB.

The knowledge base should support:

```text
documents
 ↓
chunking
 ↓
embeddings
 ↓
vector store
 ↓
retrieval
 ↓
LLM context
```

Corpus categories:

1. O-RAN specifications
2. Digital Twin documentation
3. Adaptation policies
4. Historical adaptation records

Agents can retrieve knowledge.

Agents must **NOT use RAG as a write path into the DT**.

The RAG system must not directly modify models, state, or registry.

\---

# 36\. RAG INITIALIZATION

On first startup:

1. initialize persistent ChromaDB
2. create required collection
3. insert/test a small record if appropriate
4. verify retrieval
5. then support real corpus ingestion

Make ingestion idempotent.

Do not duplicate documents every time the application starts.

Keep metadata such as:

* document ID
* source
* category
* version
* timestamp
* chunk index

\---

# 37\. MODULE 19 — LIFECYCLE MANAGEMENT AGENT

Every adaptation event must generate a structured lifecycle record.

At minimum record:

```text
event ID
timestamp
drift event
affected component/scope
drift severity
RL observation/context
RL action
agent action
production version before
candidate version
fidelity before
fidelity after
verification result
verification explanation
training window
evaluation window
model metadata
LLM metadata if used
final status
```

The record must be auditable.

\---

# 38\. HUMAN-READABLE MAINTENANCE REPORT

Lifecycle Management should also generate a human-readable vendor maintenance report.

The report should explain:

* what drifted
* affected scope
* why adaptation was triggered
* what PPO selected
* what the adaptation agent did
* what changed
* fidelity before
* fidelity after
* whether accepted/rejected
* why
* relevant contextual knowledge

Use RAG retrieval where useful.

The report must be generated automatically.

\---

# 39\. MAIN CONTINUOUS LOOP

Implement a real orchestration loop approximately equivalent to:

```text
START
 ↓
Initialize configuration
 ↓
Initialize storage
 ↓
Initialize RAG
 ↓
Initialize model registry
 ↓
Load/train bootstrap DT models
 ↓
Start telemetry source
 ↓
Continuous loop
      ↓
Receive telemetry
      ↓
Preprocess
      ↓
Synchronize D1
      ↓
Run dependency-aware DT prediction
      ↓
Evaluate fidelity
      ↓
Receive drift event
      ↓
Construct PPO observation
      ↓
PPO chooses action
      ↓
Selected adaptation agent
      ↓
Create candidate
      ↓
Sandbox validation
      ↓
Verification
      ↓
 ACCEPT ───────→ promote version
      │
      └ REJECT ──→ preserve production
      ↓
Lifecycle record
      ↓
Continue telemetry loop
```

The system must not stop after one adaptation event.

\---

# 40\. CONFIGURATION

Create a central configuration file such as:

```text
config/settings.yaml
```

All important parameters must come from configuration.

At minimum configure:

* telemetry source
* ZMQ settings
* storage paths
* model paths
* model training parameters
* rolling windows
* fidelity epsilon
* fidelity window length
* verification delta
* adaptation settings
* PPO settings
* reward penalty
* PRB model enable/disable
* RAG paths
* ChromaDB path
* Anthropic model
* LLM temperature/settings
* sandbox limits
* logging
* retention
* evaluation settings

Do not scatter magic constants through source code.

Secrets must NOT be stored in YAML.

\---

# 41\. ENVIRONMENT VARIABLES

Use `.env` or environment variables for secrets.

At minimum:

```text
ANTHROPIC\_API\_KEY
```

Never commit secrets.

Create:

```text
.env.example
```

with placeholders only.

\---

# 42\. LLM API ARCHITECTURE

Use the direct Anthropic API.

Create a centralized abstraction such as:

```text
src/llm/
    anthropic\_client.py
    prompts.py
    schemas.py
```

The exact structure is yours to optimize.

Requirements:

* one centralized API client
* retries
* timeout
* structured output validation
* logging without leaking secrets
* configurable model
* token limits
* failure handling
* rate-limit handling
* deterministic/non-deterministic settings where appropriate

LLM failures must not corrupt production.

\---

# 43\. LLM RESPONSIBILITIES

Use LLMs where reasoning genuinely adds value.

### Recalibration

Optional reasoning for selecting training context/window and diagnosing drift.

### Regeneration

LLM-driven architecture/code generation.

### Expand Scope

LLM-driven component definition/code generation.

### Verification

LLM-driven contextual/root-cause reasoning.

### Lifecycle

LLM-assisted human-readable explanation where useful.

Do NOT use LLMs for:

* RMSE calculation
* MAE calculation
* Wasserstein calculation
* MK-MMD calculation
* fidelity formula
* deterministic version comparisons
* basic telemetry validation
* deterministic model training
* PPO action selection

\---

# 44\. PROMPT-INJECTION / UNTRUSTED DATA SAFETY

Telemetry, retrieved documents, generated code, and external text must be treated as untrusted input.

Do not blindly execute instructions found in:

* telemetry fields
* RAG documents
* generated model output
* metadata
* external drift events

Keep system instructions separate from retrieved context.

Use explicit delimiters for contextual data.

\---

# 45\. SANDBOX REQUIREMENTS

Create a sandbox execution layer.

It must:

* operate outside the live production module
* use isolated candidate directories
* enforce execution timeout
* capture stdout/stderr
* capture exit codes
* reject syntax/import/test failures
* prevent candidate from silently modifying production files
* clean up temporary workspaces safely

Do not execute generated code with unrestricted direct access to the production environment.

Choose practical isolation available on the host OS.

Document its security assumptions.

\---

# 46\. MODEL REGISTRY

Create a model/component registry.

It must support:

```text
component
current version
previous versions
artifact path
model class
dependencies
feature schema
training metadata
evaluation metadata
status
```

The registry must support dynamic registration for Expand Scope.

It must never allow an invalid candidate to become active.

\---

# 47\. TESTING

Create a comprehensive test suite.

At minimum:

### Unit tests

* telemetry validation
* preprocessing
* synchronization
* each DT model
* dependency scheduler
* component registry
* fidelity metrics
* fidelity formula
* drift interface
* RL environment
* action mapping
* reward calculation
* lifecycle records
* RAG retrieval
* candidate validation
* version promotion
* rollback

### Integration tests

Test:

```text
telemetry
→ preprocessing
→ synchronization
→ DT
→ fidelity
```

Then:

```text
drift
→ PPO
→ adaptation
→ candidate
→ verification
```

### End-to-end test

Use the mock telemetry/drift source if real NS-3 is unavailable.

Simulate at least:

1. normal operation
2. drift
3. recalibration
4. regeneration
5. expand scope
6. successful verification
7. failed verification
8. rollback/preservation of production

Do not fake test results.

Actually run the tests.

\---

# 48\. BOOTSTRAP / DEMO MODE

Because real NS-3/5G-LENA may not always be available during development, create a realistic development mode.

For example:

```text
--mode demo
--mode live
```

Demo mode must generate realistic network telemetry matching the actual schema.

It must exercise the same downstream pipeline.

It must NOT be a completely separate fake architecture.

The same:

```text
preprocessing
synchronization
DT
fidelity
drift interface
PPO
agents
verification
lifecycle
```

should be exercised.

\---

# 49\. REAL VS MOCK SEPARATION

Make the distinction obvious.

For example:

```text
src/telemetry/
    base.py
    zmq\_source.py
    mock\_source.py
```

The mock source must never masquerade as real NS-3 telemetry.

Logs should identify the active source.

\---

# 50\. LOGGING

Implement structured logging.

Logs should make it possible to follow an adaptation:

```text
DRIFT DETECTED
→ component=latency
→ severity=...
→ PPO ACTION=REGENERATE
→ candidate=v...
→ sandbox tests=PASS
→ fidelity before=...
→ fidelity after=...
→ verification=ACCEPT
→ promoted=v...
```

Do not log:

* API keys
* secrets
* unnecessary raw sensitive data

\---

# 51\. COMMENTS AND USER-FRIENDLINESS

The code must be understandable.

Add useful comments/docstrings explaining:

* WHY an architectural decision exists
* non-obvious data transformations
* dependency relationships
* agent responsibilities
* safety boundaries
* candidate/production distinction
* fidelity methodology
* adaptation flow

Do NOT add useless comments like:

```python
# increment i
i += 1
```

Prefer clear names, type hints, docstrings, and concise comments around genuinely non-obvious logic.

\---

# 52\. TYPE SAFETY / CODE QUALITY

Use:

* Python type hints
* dataclasses/Pydantic where appropriate
* clear interfaces
* modular packages
* meaningful exception types
* input validation
* deterministic components where required
* clean separation of concerns

Avoid giant files.

Avoid circular imports.

Avoid global mutable state.

Avoid hidden magic constants.

\---

# 53\. PROJECT STRUCTURE

Use a clean structure similar to:

```text
ai-self-adaptive-digital-twin/
│
├── README.md
├── CLAUDE.md
├── IMPLEMENTATION\_STATUS.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
│
├── config/
│   └── settings.yaml
│
├── ns3\_sim/
│   ├── nr\_5g\_telemetry\_sim.cc
│   ├── README.md
│   └── ...
│
├── src/
│   ├── main.py
│   │
│   ├── telemetry/
│   │   ├── base.py
│   │   ├── schema.py
│   │   ├── preprocessing.py
│   │   ├── zmq\_source.py
│   │   └── mock\_source.py
│   │
│   ├── synchronization/
│   │   └── sync.py
│   │
│   ├── dt\_models/
│   │   ├── base.py
│   │   ├── d1\_model\_store.py
│   │   ├── component\_registry.py
│   │   ├── orchestrator.py
│   │   ├── throughput.py
│   │   ├── latency.py
│   │   ├── packet\_loss.py
│   │   ├── prb\_utilization.py
│   │   └── jitter.py
│   │
│   ├── fidelity/
│   │   ├── metrics.py
│   │   └── evaluator.py
│   │
│   ├── drift/
│   │   ├── drift\_detector.py
│   │   └── mock\_drift\_source.py
│   │
│   ├── adaptation/
│   │   ├── rl\_env.py
│   │   ├── rl\_agent.py
│   │   ├── recalibration\_agent.py
│   │   ├── regeneration\_agent.py
│   │   ├── expand\_scope\_agent.py
│   │   ├── verification\_agent.py
│   │   ├── lifecycle\_agent.py
│   │   └── adaptation\_manager.py
│   │
│   ├── llm/
│   │   ├── anthropic\_client.py
│   │   ├── prompts.py
│   │   └── schemas.py
│   │
│   ├── rag/
│   │   └── rag\_kb.py
│   │
│   ├── sandbox/
│   │   └── executor.py
│   │
│   ├── registry/
│   │   └── model\_registry.py
│   │
│   ├── storage/
│   │   └── ...
│   │
│   └── common/
│       ├── config.py
│       ├── logging.py
│       └── ...
│
├── data/
│   ├── bootstrap/
│   ├── telemetry/
│   ├── evaluation/
│   └── artifacts/
│
├── rag\_data/
│   ├── oran/
│   ├── digital\_twin/
│   ├── policies/
│   └── history/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
└── scripts/
    ├── setup.py
    ├── train\_ppo.py
    ├── ingest\_rag.py
    └── run\_demo.py
```

You may improve this structure if doing so genuinely improves maintainability, but preserve the architectural separation.

\---

# 54\. CLAUDE.md

Create a `CLAUDE.md` for the repository.

It must contain:

* project purpose
* architecture
* module responsibilities
* commands to run
* testing commands
* important constraints
* adaptation safety rules
* fidelity formula
* PPO action mapping
* LLM rules
* repository conventions

This file exists so future Claude Code sessions can continue without rediscovering the architecture.

\---

# 55\. IMPLEMENTATION\_STATUS.md

Create a concise persistent status file.

Track:

```text
module
implementation status
tests
known blockers
last verified command
next task
```

Update it during implementation.

If context is compacted or the task spans multiple execution phases, use this file to recover state.

\---

# 56\. GIT

Initialize Git.

Create sensible commits at logical milestones if useful.

Never commit:

* API keys
* `.env`
* temporary sandbox code
* generated secrets
* huge datasets unnecessarily

Use `.gitignore`.

Never perform destructive Git operations that could delete user work.

Since this is a new repository, you may establish the project history yourself.

\---

# 57\. DEPENDENCIES

Use a clean Python virtual environment.

Actually create it.

Install dependencies.

Do not merely write:

```text
requirements.txt
```

and claim the environment is ready.

Check imports.

Important likely dependencies include:

* pandas
* numpy
* scikit-learn
* xgboost
* scipy
* torch
* gymnasium
* stable-baselines3
* anthropic
* chromadb
* pyzmq
* pyyaml
* pydantic
* pytest

Use appropriate versions compatible with the actual environment.

Do not blindly install unnecessary packages.

\---

# 58\. GPU / CUDA

Check whether CUDA is available.

If PyTorch supports CUDA:

```python
torch.cuda.is\_available()
```

record the result.

Use GPU where genuinely useful.

Do not make GPU availability a hard requirement for the project.

The system must still function on CPU.

\---

# 59\. NS-3 ENVIRONMENT

Check whether the environment contains:

* compiler
* CMake
* Python development support
* NS-3
* 5G-LENA
* ZeroMQ development/runtime support

If unavailable, install what can reasonably be installed without destructive/system-level assumptions.

If elevated permissions are genuinely required and unavailable:

* report the blocker
* continue with the Python system
* maintain a clean integration boundary
* use mock mode for testing
* do not fabricate real simulator validation

\---

# 60\. REAL END-TO-END VALIDATION

Do not finish merely because files exist.

Run:

```text
lint/static checks where configured
unit tests
integration tests
demo end-to-end run
```

If NS-3/5G-LENA is available:

```text
build simulator
→ run simulator
→ receive telemetry
→ DT prediction
→ fidelity
→ drift
→ PPO
→ adaptation
→ verification
→ lifecycle
```

Actually inspect the output.

A successful completion claim requires actual execution evidence.

\---

# 61\. ERROR HANDLING

The system must fail safely.

Examples:

### Anthropic API failure

Do not corrupt production.

### Generated code syntax failure

Reject candidate.

### Generated code test failure

Reject candidate.

### Model training failure

Reject candidate.

### Fidelity evaluation failure

Do not promote candidate.

### Verification failure

Reject candidate.

### RAG unavailable

Continue only where RAG is non-critical; never fabricate retrieved information.

### Drift detector unavailable

Remain in waiting state or demo mode; do not invent drift.

### Candidate timeout

Reject candidate.

### Candidate crashes

Reject candidate.

Production must remain on the last known-good version.

\---

# 62\. SECURITY BOUNDARIES

Treat:

* LLM output
* RAG content
* telemetry metadata
* external drift messages
* generated code

as untrusted.

Never allow them to:

* expose secrets
* overwrite arbitrary production files
* modify configuration secretly
* bypass verification
* change fidelity formulas
* alter PPO action mapping
* promote themselves

Promotion is controlled by deterministic system logic.

\---

# 63\. NO HUMAN INTERVENTION DURING ADAPTATION

The following must happen automatically:

```text
drift
→ PPO decision
→ adaptation agent
→ candidate generation/training
→ sandbox
→ evaluation
→ verification
→ accept/reject
→ promotion/rollback
→ lifecycle record
```

Do not add:

```text
"Ask user whether to proceed"
```

Do not require manual approval for:

* recalibration
* regeneration
* expand scope
* candidate verification
* promotion
* rejection
* lifecycle recording

The system is intended to be self-adaptive.

Human involvement is only for external operational setup/configuration or genuinely unavailable infrastructure—not normal adaptation decisions.

\---

# 64\. NO ARCHITECTURAL DRIFT

Do NOT simplify the project by removing:

* PPO
* LLM-driven regeneration
* LLM-driven expand scope
* autonomous verification
* lifecycle management
* RAG
* model versioning
* sandboxing
* fidelity evaluation
* dynamic synchronization

Do NOT replace PPO with heuristics.

Do NOT replace LLM generation with hardcoded templates.

Do NOT replace verification with "tests passed".

Do NOT replace fidelity with a single metric.

Do NOT turn the dynamic DT into a static dataset.

\---

# 65\. REQUIRED DATA FLOW

The final implementation must preserve this conceptual flow:

```text
                    ┌─────────────────────┐
                    │ NS-3 / 5G-LENA      │
                    │ Network             │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Telemetry Collection│
                    │ \& Preprocessing     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Continuous          │
                    │ Synchronization     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ D1 Dynamic DT State │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ DT Orchestrator     │
                    │ Dependency-aware   │
                    └──────────┬──────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │ DT Prediction Components     │
                │ Throughput / Loss / Latency  │
                │ PRB / Jitter                 │
                └──────────────┬───────────────┘
                               │
                    predictions + ground truth
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Fidelity Engine     │
                    │ RMSE / MAE / W1     │
                    │ MK-MMD / Score      │
                    └──────────┬──────────┘
                               │
                               │
                 ┌─────────────▼─────────────┐
                 │ External Drift Detector  │
                 └─────────────┬─────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ PPO Decision Agent   │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼──────────────────┐
             │                 │                  │
             ▼                 ▼                  ▼
       Recalibrate        Regenerate         Expand Scope
             │                 │                  │
             │                 └──────┬───────────┘
             │                        │
             └────────────┬───────────┘
                          ▼
                 Candidate DT Version
                          │
                          ▼
                   ┌──────────────┐
                   │   Sandbox    │
                   └──────┬───────┘
                          │
                          ▼
                 Agentic Verification
                          │
                    ┌─────┴─────┐
                    │           │
                  ACCEPT      REJECT
                    │           │
                    ▼           ▼
                Promote      Discard
                    │           │
                    └─────┬─────┘
                          ▼
                Lifecycle Management
                          │
                          ▼
                 Auditable Record
```

\---

# 66\. FINAL IMPLEMENTATION CHECKLIST

Before declaring completion, verify every item.

## Core

* \[ ] new Git repository created
* \[ ] clean project structure
* \[ ] configuration implemented
* \[ ] virtual environment created
* \[ ] dependencies installed
* \[ ] tests configured
* \[ ] logging configured

## Telemetry

* \[ ] telemetry schema
* \[ ] preprocessing
* \[ ] synchronization
* \[ ] mock source
* \[ ] ZMQ source
* \[ ] NS-3 integration boundary

## DT

* \[ ] D1 state store
* \[ ] history store
* \[ ] registry
* \[ ] base model interface
* \[ ] orchestrator
* \[ ] throughput
* \[ ] packet loss
* \[ ] latency
* \[ ] PRB utilization
* \[ ] jitter
* \[ ] dependency scheduling

## Fidelity

* \[ ] RMSE
* \[ ] MAE
* \[ ] Wasserstein
* \[ ] MK-MMD
* \[ ] squared metrics
* \[ ] rolling min-max normalization
* \[ ] epsilon
* \[ ] composite fidelity
* \[ ] deterministic tests

## Drift

* \[ ] external interface
* \[ ] schema validation
* \[ ] mock drift source
* \[ ] no fake detector implementation

## PPO

* \[ ] Gymnasium environment
* \[ ] PPO
* \[ ] Stable-Baselines3
* \[ ] observation
* \[ ] action mapping
* \[ ] reward
* \[ ] training
* \[ ] runtime inference

## Adaptation

* \[ ] recalibration agent
* \[ ] regeneration agent
* \[ ] expand-scope agent
* \[ ] autonomous execution
* \[ ] candidate versioning
* \[ ] sandbox

## LLM

* \[ ] direct Anthropic API
* \[ ] centralized client
* \[ ] configurable Claude model
* \[ ] API failure handling
* \[ ] structured outputs
* \[ ] secret management
* \[ ] regeneration
* \[ ] expand scope
* \[ ] verification reasoning

## Verification

* \[ ] deterministic metrics
* \[ ] deterministic acceptance gate
* \[ ] LLM contextual reasoning
* \[ ] ACCEPT
* \[ ] REJECT
* \[ ] rollback/preservation
* \[ ] regression checks

## RAG

* \[ ] ChromaDB
* \[ ] embeddings
* \[ ] chunking
* \[ ] ingestion
* \[ ] retrieval
* \[ ] read-only agent access
* \[ ] four corpus categories

## Lifecycle

* \[ ] structured adaptation records
* \[ ] version history
* \[ ] fidelity before/after
* \[ ] verification result
* \[ ] vendor maintenance report

## Validation

* \[ ] unit tests
* \[ ] integration tests
* \[ ] end-to-end demo
* \[ ] real NS-3 validation if available
* \[ ] no fabricated results
* \[ ] documented blockers

\---

# 67\. DEVELOPMENT EXECUTION STRATEGY

Do not try to write everything blindly in one pass.

Work in dependency order:

### Phase 1

Repository + environment + configuration + common infrastructure.

### Phase 2

Telemetry + preprocessing + synchronization + D1.

### Phase 3

DT models + registry + orchestrator.

### Phase 4

Fidelity engine.

### Phase 5

Drift interface.

### Phase 6

PPO environment + PPO training/inference.

### Phase 7

Adaptation agents + model registry + sandbox.

### Phase 8

Anthropic integration.

### Phase 9

RAG.

### Phase 10

Verification + lifecycle.

### Phase 11

Full orchestration.

### Phase 12

Testing and real execution.

After each phase:

1. run relevant tests
2. fix errors
3. update `IMPLEMENTATION\_STATUS.md`
4. continue

Do not proceed while knowingly leaving foundational modules broken.

\---

# 68\. TOKEN/CONTEXT EFFICIENCY

The available Claude Code budget is extremely limited.

Therefore:

* do not repeatedly explain what you are doing
* do not print huge source files unnecessarily
* do not repeatedly inspect unchanged files
* use repository state as persistent memory
* keep `CLAUDE.md` authoritative
* keep `IMPLEMENTATION\_STATUS.md` updated
* make edits directly
* run targeted tests
* avoid redundant package installation
* avoid unnecessary subagents
* use subagents only when parallel isolated work genuinely saves time
* never sacrifice correctness simply to produce less code

If context becomes constrained, prioritize:

1. core functionality
2. correctness
3. tests
4. safety
5. end-to-end integration
6. documentation

\---

# 69\. COMPLETION STANDARD

You are NOT finished when:

* directories exist
* imports merely parse
* README exists
* placeholders exist
* mocks are being passed off as real components
* tests have not been run

You ARE finished when the repository contains a coherent, executable implementation and you have actually validated as much of the complete pipeline as the environment allows.

At the end, provide a concise completion report containing only:

```text
IMPLEMENTED
- ...

VALIDATED
- ...

END-TO-END RESULT
- ...

EXTERNAL BLOCKERS
- ...

HOW TO RUN
- ...
```

Do not claim something was validated unless you actually ran it.

\---

# 70\. FINAL NON-NEGOTIABLE RULES

Remember these above everything else:

1. **This is a NEW repository. Create everything from scratch.**
2. **The Digital Twin is continuously dynamic.**
3. **Model versions are independently versioned from dynamic DT state.**
4. **PPO decides WHAT adaptation strategy to use.**
5. **Adaptation agents decide HOW to execute that strategy.**
6. **There are exactly six agents.**
7. **Drift detection itself is external; implement only its interface/adapter.**
8. **Fidelity calculations are deterministic.**
9. **LLMs cannot override deterministic acceptance criteria.**
10. **LLM-generated code is never production code until sandboxed and verified.**
11. **Rejected candidates cannot affect production.**
12. **Accepted candidates are versioned and reversible.**
13. **Recalibration, regeneration, expand-scope, verification, and lifecycle management are autonomous.**
14. **RAG is read-only contextual knowledge.**
15. **Use the direct Anthropic API.**
16. **Never hardcode API keys.**
17. **Never hardcode deprecated Claude model identifiers.**
18. **Do not replace PPO with heuristics.**
19. **Do not turn the DT into a static dataset.**
20. **Do not fake NS-3 execution, telemetry, API responses, tests, or verification.**
21. **Actually run the code and tests.**
22. **Preserve the complete 19-module architecture.**
23. **Use comments/docstrings to make non-obvious architecture understandable.**
24. **Do not ask unnecessary questions.**
25. **Build the actual project—not a demonstration pretending to be the project.**

Now begin by creating the repository and implementing the system.

