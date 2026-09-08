# AI-Driven Self-Adaptive Network Digital Twin

A production-quality, modular Digital Twin platform for a simulated 5G network: continuously
synchronized DT state, dependency-aware prediction components, deterministic fidelity evaluation,
a PPO-trained decision agent, LLM-assisted (Anthropic Claude) regeneration/expand-scope adaptation
agents, sandboxed candidate verification, and full lifecycle auditability.

See **`CLAUDE.md`** for the full architecture (19-module map + D1/D2), agent responsibilities, the
fidelity formula, PPO action mapping, and adaptation safety rules — read it before making changes.
See **`IMPLEMENTATION_STATUS.md`** for current per-module build/test status and the next task.
`prompt.md` is the original authoritative specification.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY
```

NS-3 / 5G-LENA (Module 1 ground-truth simulator):

```bash
./scripts/setup_ns3.sh   # clones ns-3.48 + 5G-LENA (nr v5.1) into ns3_sim/ns-3-dev and builds
```

See `ns3_sim/README.md` for pinned versions, prerequisites, and real-vs-mock telemetry source
rules.

## Running the platform

```bash
python -m src.main --mode demo   # mock telemetry + mock drift, full pipeline exercised
python -m src.main --mode live   # real NS-3/5G-LENA telemetry over ZeroMQ
```

See **`HOW_TO_RUN.md`** for a full step-by-step guide (setup, tests, individual module
validation scripts, and the two real end-to-end demo runners).

## Testing

```bash
pytest tests/unit
pytest tests/integration
pytest tests/e2e
```

## Repository layout

```
config/        settings.yaml — every tunable parameter (no magic constants in source)
ns3_sim/       our NS-3/5G-LENA scenario + telemetry exporter (vendored ns-3-dev is gitignored)
src/           application source — see CLAUDE.md for the module -> package map
data/          bootstrap / telemetry / evaluation / artifacts (generated, gitignored)
rag_data/      RAG corpus (O-RAN specs, DT docs, policies, adaptation history)
tests/         unit / integration / e2e
scripts/       setup, PPO training, RAG ingestion, demo runner
```
