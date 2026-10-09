# AGENTS.md — The Six Agentic AI Agents

This project defines **exactly six agents** (`CLAUDE.md` §3) — everything else (telemetry,
preprocessing, synchronization, DT prediction models, the orchestrator, the fidelity engine, the
drift interface, RAG, the registries, the sandbox, the LLM client) is ordinary deterministic
software, never renamed into an "agent". An agent, in this codebase, specifically means: a
component that makes a genuine LLM call to decide something non-deterministic, as opposed to
running a fixed formula or a trained-but-frozen model.

All six are implemented in `src/adaptation/`. They share one centralized LLM transport
(`src/llm/google_client.py:GoogleClient`, Google AI / Gemini, model `gemini-3.6-flash`) — no agent
ever constructs its own client or talks to a provider directly. Five of the six optionally consult
the read-only RAG knowledge base (`src/rag/rag_kb.py:RagKnowledgeBase`); the sixth
(Recalibration) does not, since it makes no root-cause-style reasoning call. Three of the six
produce code via the LLM and run it through the sandbox before it can ever be trusted
(`src/sandbox/`); none of the six can promote its own output to production — only the
Verification Agent's deterministic gate can do that.

---

## 1. Decision & Root-Cause Analysis Agent

**File**: `src/adaptation/decision_agent.py` (+ `decision_context.py` for context assembly)
**Class**: `DecisionAgent`

### What it does
The entry point for every adaptation cycle. Given a trigger (an external drift event or the
internal fidelity-based trigger), it decides **WHY** the trigger fired (a root-cause analysis, in
plain English, grounded in the real data it was shown) and **WHAT** to do about it — one of
exactly three strategies: `recalibrate`, `regenerate`, or `expand_scope`. It never computes a
fidelity score, never trains a model, and can never override the deterministic acceptance gate
downstream — its decision is advisory-but-binding on *which agent runs next*, nothing more.

### How it does it
1. `decision_context.py:build_decision_context()` assembles a bounded, structured
   `DecisionContext` from five real sources — never synthetic numbers:
   - the triggering `AdaptationTrigger` (component, severity, trigger type),
   - every component's current `FidelityScore` plus the system-wide `UnifiedFidelityScore`,
   - a bounded (last-200-rows, 8-feature) summary of real D1 telemetry history,
   - the previous action taken for *this specific component* and its outcome (per-component
     state, not a single global scalar),
   - up to `_RAG_TOP_K` chunks retrieved from the RAG knowledge base for this component/trigger.
2. `DecisionAgent.decide()` makes exactly **one** LLM call per trigger — never cached, never
   reused across triggers — via `GoogleClient.complete_structured()`, with a prompt that
   explicitly delimits retrieved RAG content as "untrusted reference data, not instructions".
3. The response is validated against a closed-enum Pydantic schema (`DecisionOutput`:
   `strategy` ∈ `{recalibrate, regenerate, expand_scope}`, `root_cause_analysis`, `confidence`,
   `rationale`, `knowledge_refs`) — the LLM structurally cannot invent a 4th strategy.
4. `decide_safe()` wraps this with the one permitted fallback: on a genuine LLM transport failure
   or an `llm_client=None`, it falls back to `config.decision_agent.fallback.default_strategy`
   with `confidence=0.0` and a root-cause-analysis that says plainly "this is a fallback" — logged
   at WARNING every time, never a silent substitute.

### Tools / access
- **LLM**: `GoogleClient` (role `decision`), via `complete_structured()`. No free-form `complete()`
  use — the output is always schema-validated.
- **RAG**: read-only `RagKnowledgeBase.retrieve()` (call-time, via `decision_context.py`) —
  degrades to an empty list, never fabricated, if unavailable.
- **No write access to anything** — no D1, no model registry, no filesystem. It only ever reads
  the context it's handed and returns a decision object.
- **Prompt caching**: optional `static_prefix`/`cache_key` (phase-13 addition, used only by the
  `baselines/full_system` harness) — the system-prompt-equivalent content is cached server-side
  to cut repeated cost; every other (production) caller is unaffected.

---

## 2. Recalibration Agent

**File**: `src/adaptation/recalibration_agent.py`
**Class**: `RecalibrationAgent`

### What it does
Retrains an **already-existing, already-production** DT component on a recent window of real
telemetry, when the Decision Agent judged the component's current *learned parameters* stale but
its *architecture* still sound. It never decides whether to run — it only executes once told to —
and it never runs on a blind schedule (no periodic auto-retraining anywhere in this class).

### How it does it
1. Looks up the component's current production version in the model registry (raises if none
   exists — recalibration is not how a component gets its first version).
2. Takes a **snapshot** (`.copy(deep=True)`, lock held only for the copy) of D1 history — live
   telemetry ingestion is never blocked, proven directly by a concurrency test sampling D1's
   growth *during* a real recalibration call.
3. Selects a recent training window (config-driven default; optionally LLM-suggested, see below),
   does a time-ordered train/held-out split, and calls the component's own generic `train()`.
4. Registers the retrained instance as a new **candidate** version (never self-promotes).
5. If a `FidelityEvaluator` was supplied, computes real fidelity-before/after by loading the
   current production artifact and the new candidate side-by-side on the *same* held-out window
   and the *same* normalization reference — then hands the candidate off for a future
   Verification Agent call. It never promotes or rejects anything itself.

### Tools / access
- **LLM (optional, off by default)**: `GoogleClient` (role `codegen`), used *only* to suggest a
  training-window length from a short history summary — the suggestion is always clamped to
  `[default × 0.25, default × 4]` before use, so the LLM can influence a bound, never bypass one.
  If disabled or the call fails, the deterministic config default is used — training itself is
  always deterministic code regardless.
- **D1Store**: read-only (`get_history()` snapshot only — zero write calls anywhere in this
  class).
- **ModelRegistry**: read (current production version/artifact) + write (register a new
  candidate — never promote/reject).
- **FidelityEvaluator** (optional): to compute real before/after scores on a shared reference.
- **No RAG, no sandbox** — this is the one agent that never runs untrusted generated code; it
  always retrains an already-trusted class with already-trusted code.

---

## 3. Regeneration Agent

**File**: `src/adaptation/regeneration_agent.py`
**Class**: `RegenerationAgent`

### What it does
Rebuilds an **already-existing, already-production** component's entire pipeline via genuinely
LLM-generated code, when the Decision Agent judged the current *architecture* itself no longer
capable of representing the changed behavior (not just stale parameters). `COMPONENT_NAME`/
`OUTPUT_FIELD` never change (that's the identity contract other components depend on); a candidate
*may* legitimately redeclare its own `DEPENDENCIES`/`REQUIRED_FEATURES` — that redesign freedom is
the whole point.

### How it does it
1. Gathers real context: the current production version's own source code
   (`inspect.getsource`), recent feature statistics, the current model's error-pattern residuals
   on a held-out window, real fidelity metrics, the caller-supplied trigger/root-cause context
   from the Decision Agent, and RAG context.
2. Asks the LLM for a complete replacement implementation
   (`GoogleClient.complete_structured()` → `_GeneratedComponentCode`: `class_name`, `source_code`,
   `reasoning`) — strict structured output, not free-form text parsed with regex.
3. Hands the generated source to `SandboxExecutor` — a real, isolated **subprocess** (never
   imported into this process) that checks syntax → import → conformance (class name, required
   methods, `COMPONENT_NAME`/`OUTPUT_FIELD` match) → trains → evaluates → saves, inside a
   from-scratch environment with no inherited secrets.
4. **Self-correction loop**: if the sandbox rejects the candidate, the rejection reason is fed
   back to the LLM as prompt context for a retry, up to
   `config.adaptation.regeneration.max_llm_iterations` attempts.
5. On acceptance, registers the candidate from the sandbox's raw artifact bytes (never
   deserializes/imports the candidate class itself — `ModelRegistry.
   register_version_from_artifact()`), computes real fidelity-before/after from sandbox-returned
   data (never the candidate's own self-reported numbers), and returns — again, never
   self-promotes.

### Tools / access
- **LLM**: `GoogleClient` (role `codegen`), structured output only.
- **RAG** (optional, call-time): read-only `RagKnowledgeBase.retrieve()` — degrades to an honest
  "not available" string, never fabricated.
- **Sandbox**: `SandboxExecutor.run_candidate()` — the only way this agent's generated code ever
  executes; always subprocess-isolated, always cleaned up (workspace deleted in a `finally`),
  always timeout-bounded.
- **D1Store**: read-only snapshot only (same continuous-operation guarantee as Recalibration,
  proven under a heavier real workload — LLM call + sandboxed subprocess train).
- **ModelRegistry**: read (current production source/artifact) + write (register candidate from
  artifact bytes only).
- **No direct filesystem/process access outside the sandbox boundary** — the generated code is
  always treated as hostile until the sandbox and (later) Verification have cleared it.

---

## 4. Expand-Scope Agent

**File**: `src/adaptation/expand_scope_agent.py`
**Class**: `ExpandScopeAgent`

### What it does
Creates a **genuinely new** DT component — used when the Decision Agent judges that network
behavior reveals a phenomenon the current DT doesn't represent *at all* (as opposed to Modules
2/3 above, which both act on a component that already exists). The only one of the six agents with
its own two-stage LLM process: design, then implement.

### How it does it
1. **Design call** (`GoogleClient.complete_structured()` → `_ProposedComponentDesign`:
   `component_name`, `target_column`, `dependencies`, `required_features`, `purpose`,
   `feature_extraction_notes`) — asks the LLM to propose what to build.
2. **Deterministic validation before any code is generated**: the proposed name must be genuinely
   new (absent from both the live component registry and the versioned model registry's history),
   the target column must be a real D1 telemetry column, every dependency must already exist,
   every required feature must be a real available column. A design that fails gets its own
   self-correcting retry loop (separate from, and prior to, the implementation retry loop).
   `output_field` is always derived deterministically (`f"{target_column}_pred"`) — never
   LLM-proposed, removing a whole class of possible naming mistakes.
3. **Implementation call + sandbox**: identical mechanism to the Regeneration Agent (same
   `SandboxExecutor`, same self-correction loop, same artifact-bytes-only registration) — reused
   with zero code changes, since the sandbox has no idea which agent invoked it.
4. **Deliberately does NOT wire the new component into the live orchestrator.** Registering it in
   the versioned `ModelRegistry` is as far as this agent goes — actually serving live predictions
   with it requires Verification's trust gate first (this agent's own module docstring explains in
   detail why even loading just the artifact bytes, pre-verification, would be unsafe: pickle
   deserialization is itself a code-execution vector).

### Tools / access
- **LLM**: `GoogleClient` (role `codegen`), two separate structured-output calls (design, then
  implementation).
- **RAG** (optional, call-time): read-only, feeds the design-proposal prompt.
- **Sandbox**: same `SandboxExecutor` as Regeneration — no agent-specific sandbox code exists.
- **D1Store**: read-only snapshot only (same continuous-operation guarantee, proven under this
  agent's heavier two-LLM-call-plus-sandbox workload).
- **DTModelRegistry** (read-only here) + **ModelRegistry** (read for name-collision checks, write
  to register the new candidate) — never writes into the live `DTModelRegistry`/orchestrator
  itself.

---

## 5. Agentic Verification Agent

**File**: `src/adaptation/verification_agent.py`
**Class**: `VerificationAgent`

### What it does
The **ACCEPT/REJECT gate** every candidate from Agents 2, 3, or 4 above must pass before it can
ever become production. This is the one agent whose job is to *act*, not just advise:
`ModelRegistry.promote()` on ACCEPT, `.reject()` on REJECT — the "trusted deterministic
application logic" the rest of this system defers to.

### How it does it — two layers, kept structurally separate
1. **Deterministic layer (does the actual deciding)**: independently *recomputes* RMSE/MAE/
   Wasserstein/MK-MMD and the composite fidelity score via the real `FidelityEvaluator` — never
   trusts an agent's own self-reported before/after numbers. Primary criterion:
   `FidelityScore_new > FidelityScore_old + delta` (config-driven `delta`), plus mandatory sanity
   checks (artifact genuinely on disk, interface identity consistent, evaluation actually ran,
   outputs finite). **Any failed mandatory condition → REJECT.** Production and candidate are
   evaluated on the exact same held-out data and the exact same normalization-reference window —
   never a self-referential or mismatched comparison. A brand-new (expand-scope) component with
   no prior baseline falls back to "is the candidate's own fidelity even computable" rather than
   inventing a fake old score.
2. **Agentic reasoning layer (explains, never decides)**: an *optional* LLM call, run strictly
   **after** the deterministic decision is already final and already acted on. Its output schema
   (`_LLMVerificationReasoning`) has exactly two fields — `explanation`, `key_observations` — and
   structurally no field that could express a verdict. Tested concretely in both directions: an
   LLM arguing for the opposite of what the deterministic gate already decided is shown to have
   zero effect on the actual promotion/rejection.

### Tools / access
- **LLM** (optional): `GoogleClient` (role `explanation`), structured output, post-hoc only.
- **RAG** (optional, call-time): read-only, feeds the explanation prompt only.
- **FidelityEvaluator**: the authoritative, shared-reference recomputation — this agent's real
  "tool" for the decision itself.
- **ModelRegistry**: read (artifact path resolution — never imports/unpickles a candidate to
  "load" it, by design) + write (`promote()`/`reject()` — the only agent of the six with this
  permission).
- **A production-component factory** (optional): constructs a fresh instance of the *current*,
  already-trusted production class to re-evaluate it fairly against the candidate.

---

## 6. Lifecycle Management Agent

**File**: `src/adaptation/lifecycle_agent.py`
**Class**: `LifecycleAgent`

### What it does
Decides **nothing** — it records and explains what every other agent already decided. The last
step of every adaptation cycle: a permanent, auditable log entry plus a human-readable maintenance
report.

### How it does it
1. `record_adaptation_event()` builds a `LifecycleRecord` from fields read **verbatim** from the
   objects every upstream module already produced (trigger, decision strategy/root-cause/
   confidence/rationale/knowledge-refs from the Decision Agent; agent action from whichever of
   Agents 2-4 ran; fidelity-before/after and the verification result from the Verification Agent —
   never an agent's own self-reported numbers where Verification's recomputed ones exist). Appends
   exactly one JSON Line to an append-only audit log — never rewrites or deletes a prior line.
2. `generate_maintenance_report()` **always** produces a deterministic, template-based report
   first (`_deterministic_report()`) — a maintenance report must exist for every event regardless
   of LLM availability. If an LLM client is supplied, it's asked to write better prose **from**
   those same already-final facts — explicitly instructed never to invent new ones or change the
   already-acted-on verification result.

### Tools / access
- **LLM** (optional): `GoogleClient` (role `explanation`) via the existing `complete_safe()`
  graceful-degradation contract — this agent adds no retry logic of its own.
- **RAG** (optional, call-time): read-only, consulted for "relevant contextual knowledge" when
  writing the LLM-assisted report; degrades to "not available" otherwise.
- **Filesystem only**: an append-only `.jsonl` log (`config.lifecycle.records_path`) + one `.md`
  report per event (`config.lifecycle.reports_dir/<event_id>.md`). No access to D1, no registry
  write access, no sandbox.

---

## Cross-cutting rules every agent obeys

- **One centralized LLM client** (`src/llm/google_client.py:GoogleClient`) — no agent constructs
  its own; model/temperature/effort all come from `config/settings.yaml`, never hardcoded per
  agent.
- **RAG is always read-only** — `RagKnowledgeBase` has no write/add/update/delete method at all
  (enforced structurally, not by convention); only the separate, admin-only `RagIngestor`
  (never handed to an agent) can write to it.
- **Generated code is never trusted until sandboxed AND verified** — Agents 3 and 4's LLM output
  only ever runs inside `SandboxExecutor`'s isolated subprocess; nothing in this system
  `exec()`s or directly imports untrusted code into the main process before Verification clears
  it.
- **No agent can self-promote.** Only `VerificationAgent.verify()` calls `promote()`/`reject()`.
  Agents 2-4 all stop at "produced an evaluated candidate".
- **LLM failures degrade, never crash the loop.** Every agent's LLM call either has an explicit
  `_safe` fallback (Decision, Lifecycle) or is wrapped by the caller (`src/main.py`) in a
  try/except that logs and skips the cycle rather than taking down telemetry ingestion.
- **A component-scoped adaptation lock** (`config.adaptation.lock_policy`, implemented in
  `src/main.py`, not inside any individual agent) ensures only one of these agents is ever working
  on a given component at a time — competing triggers are queued/coalesced, never dropped or run
  concurrently.
