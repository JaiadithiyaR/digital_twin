# AI-Driven Self-Adaptive Network Digital Twin

A production-quality, modular Digital Twin platform for a real 5G RAN (OpenAirInterface/OAI):
continuously synchronized DT state, dependency-aware prediction components for 7 real OAI RAN
KPIs, deterministic fidelity evaluation (including a unified fidelity score and an internal
fidelity-based adaptation trigger), a knowledge-based (RAG-grounded) Decision & Root-Cause
Analysis Agent, LLM-assisted (Google AI/Gemini) regeneration/expand-scope adaptation agents,
sandboxed candidate verification, and full lifecycle auditability.

See **`CLAUDE.md`** for the full architecture (19-module map + D1/D2), agent responsibilities, the
fidelity formula, decision-strategy enum, and adaptation safety rules — read it before making
changes (including its "Phase 12 — OAI Migration" entry, the most recent architectural change).
See **`IMPLEMENTATION_STATUS.md`** for current per-module build/test status and the next task.
`prompt.md` is the original authoritative specification; `prompt2.md` ("Migrate the Digital Twin
from NS-3 to OAI KPI replay") supersedes it for Module 1's data source and the DT component set.
See **`docs/oai_twin.md`** for the real OAI KPI catalog, data-splits design, and the 7-component
dependency graph.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GOOGLE_API_KEY (or GEMINI_API_KEY)
python scripts/setup.py
```

Real OAI RAN KPI data (Module 1's data source — needed for `--mode live` and the bootstrap
models; `--mode demo`/tests/RAG ingestion use a mock generator and don't need this):

```bash
python scripts/build_oai_dataset.py   # reads the real, read-only export at config.data.oai_root
                                       # and writes data/processed/oai/{train,val,test}.parquet
```

NS-3/5G-LENA is retired — see `archive/ns3_sim/README.md` for why and what's kept there for
historical reference.

## Running the platform

```bash
python -m src.main --mode demo   # OAI-schema-shaped mock telemetry + mock drift, full pipeline exercised
python -m src.main --mode live   # real OAI RAN KPI replay (SOURCE=OAI_RAN_REPLAY)
```

See **`docs/HOW_TO_RUN.md`** for a full step-by-step guide (setup, tests, dataset build,
individual module validation scripts, and the real end-to-end demo/smoke-test runners).

## Testing

```bash
pytest tests/unit
pytest tests/integration
pytest tests/e2e
```

## Repository layout

```
config/        settings.yaml — every tunable parameter (no magic constants in source)
src/           application source — see CLAUDE.md for the module -> package map
data/          bootstrap / telemetry / evaluation / artifacts / processed (generated, gitignored)
rag_data/      RAG corpus (O-RAN specs, DT docs, policies, adaptation history)
tests/         unit / integration / e2e
scripts/       setup, OAI dataset build, RAG ingestion, demo runners, metrics visualization
archive/       retired NS-3/5G-LENA code + pre-migration runtime state (historical reference only)
```
