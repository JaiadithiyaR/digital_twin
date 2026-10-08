# Baseline comparison (phase13 §4)

Compares 4 adaptation strategies over the same real OAI TEST replay (run 8, speed=0), using the
same bootstrap v2 (residual, CV-tuned) components:

| mode | strategy source | LLM used for |
|---|---|---|
| `static` | none — every trigger is logged but never dispatched | nothing |
| `periodic_retrain` | fixed schedule, recalibrates all 7 components every `interval_cycles` cycles | nothing |
| `rule_based` | deterministic severity thresholds (`src/adaptation/baseline_policies.py`) | codegen (regenerate/expand_scope) + verification explanation only — never the decision itself |
| `full_system` | the real knowledge-based LLM+RAG Decision & Root-Cause Analysis Agent (Module 13) | decision + codegen + verification explanation |

## Fairness rules (enforced in code, not just documented)

- **Same bootstrap v2 models** — `baselines/_lib.py:bootstrap_components()` is the one shared
  entry point every mode calls; no mode gets a different training recipe.
- **Same real TEST replay** — `baselines/_lib.py:replay_test_split()` always replays
  `config.data.splits.test` (run 8) at `speed=0`, in genuine chronological order. Run 8 is NEVER
  used for training/tuning/debugging anywhere in this repository (see `claude_code_prompt_phase13.md`
  §0) — only here, as the held-out comparison set, and in `scripts/run_baselines.py`'s own
  existing `--mode full` baselines (a Phase-12 script; this folder is the phase13 redesign of
  that same idea, built from scratch with real cost/caching/budget controls).
- **Same fidelity config** — every mode constructs its `FidelityEvaluator` from the identical
  `settings.fidelity` section; nothing here is mode-specific.
- **Same seeds** — each mode's own `configs/<mode>.yaml` declares its seed(s); `full_system` is
  the only mode run multiple times (3 seeds, mean ± std reported) per phase13 §4's own
  instruction — the other three modes are deterministic given the shared bootstrap + replay, so
  one run each is sufficient and does not spend LLM budget repeating a result with no variance.
- **Fresh state per run** — `run_mode()` constructs a brand-new `ModelRegistry`/`D1Store`/
  `DTModelRegistry` under its own `baselines/results/<mode>/` (or `.../seed_N/`) directory; no
  mode or seed ever reads another's state.
- **One shared "cycle" definition** — the fully-replayed TEST split (loaded once, in order) is
  split into `config.baselines.cycles` equal chronological chunks; cycle `i` reveals chunks
  `1..i` to the prediction+fidelity loop, simulating incremental arrival without needing a
  real-time (`speed>0`) replay for every mode run.

## Running it

```bash
# One real LLM smoke test (phase13 §3.8, run 7, never run 8) + all 4 modes, default cycles/seeds
# from config/settings.yaml (cycles=15, full_system seeds=[1,2,3]):
python baselines/run_all.py

# Only the non-LLM modes, fast iteration, no pre-flight check needed:
python baselines/run_all.py --modes static periodic_retrain --skip-preflight

# Everything, with an explicit cycle count override (useful for a fast smoke run):
python baselines/run_all.py --cycles 5 --skip-preflight

# Build the comparison tables/graphs from whatever run_all.py has already written (never re-runs):
python baselines/compare.py
```

Each mode run writes, under `baselines/results/<mode>/` (or `.../seed_N/` for `full_system`):
`metrics_timeline.csv`, `adaptation_events.jsonl`, `report.md`, `run_summary.json`,
`llm_calls.jsonl` (LLM-using modes only), and `graphs/` (phase13 §5 items 1-3 per component/run).
`compare.py` then reads every mode's `run_summary.json`/`metrics_timeline.csv`/
`adaptation_events.jsonl`/`llm_calls.jsonl` and writes `baselines/results/comparison/
{comparison.md,comparison.csv}` plus `graphs/` (phase13 §5 items 4-8).

## Cost controls (phase13 §3)

`rule_based` and `full_system` share one `GoogleClient` wrapped in `src/llm/cost_tracking.py`'s
`MeteredGoogleClient`, which enforces `config.llm.budget_usd` (default $20) via a `BudgetGuard`
BEFORE every call and logs the REAL post-call usage/cost to that run's own `llm_calls.jsonl` —
never a rough pre-call estimate. If the budget would be exceeded, the run stops immediately
(an incomplete run is reported honestly in `run_summary.json`/the console, never silently
degraded to the deterministic fallback strategy just because of cost). Every LLM call also uses
one shared, explicit Gemini context cache per run (`config.llm.caching`, phase13 §3.3) holding
the static prompt prefix (system instructions + adaptation policies + an OAI/DT documentation
summary + `feature_schema.json`) — `baselines/_lib.py:build_static_prefix()` builds it once, and
every agent call this run makes reuses the same cache key.
