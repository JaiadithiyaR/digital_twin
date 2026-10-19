# BASELINES.md — Baseline Comparison Framework

What `baselines/` compares, the exact formulas behind every number, and every graph produced.
Code lives in `baselines/{run_all.py,_lib.py,_outputs.py,compare.py}`; run with
`python baselines/run_all.py` then `python baselines/compare.py` (see `manual.md` §2).

---

## 1. What is being compared

Four adaptation strategies, run over the **same** bootstrap models and the **same** real OAI
**TEST split (run 8)**, replayed at `speed=0` into a fresh, isolated D1Store per mode/seed —
never run 7 (reserved for smoke tests) and never a train/tune run:

| mode | strategy selection | uses LLM for |
|---|---|---|
| `static` | none — triggers are logged, never dispatched | nothing |
| `periodic_retrain` | blind scheduled retrain of all 7 components every `interval_cycles` | nothing |
| `rule_based` | fixed severity-threshold rule (`src/adaptation/baseline_policies.py:rule_based_strategy`) | codegen (regenerate/expand_scope) + explanation (verification), never the strategy choice itself |
| `full_system` | the real `DecisionAgent` — a genuine LLM call per trigger | decision + codegen + explanation (all 3 roles) |

**Fairness rules**, enforced in code, not just documented:
- Identical bootstrap (`baselines/_lib.py:bootstrap_components()`) — same 7 production models every mode starts from.
- Identical replay (`replay_test_split()` — run 8, `speed=0`).
- Identical fidelity config (`settings.fidelity` is never mode-specific).
- Every mode/seed starts from a brand-new `ModelRegistry`/`D1Store`/`DTModelRegistry` — never reused across modes or seeds.
- `full_system` runs 3 seeds (`seeds: [1,2,3]` in `baselines/configs/full_system.yaml`) — seeds vary only `MockDriftSource`'s own RNG (which synthetic drift events get generated); the replay/bootstrap/fidelity config is identical across seeds.

`rule_based`'s rule (exact thresholds in `baselines/configs/rule_based.yaml`):
```
severity <= recalibrate_max_severity                           -> recalibrate
recalibrate_max_severity < severity <= regenerate_max_severity  -> regenerate
severity > regenerate_max_severity                              -> expand_scope
```

---

## 2. Formulas used

### 2.1 Per-component fidelity (`src/fidelity/evaluator.py`, `src/fidelity/metrics.py`)

For each component, each cycle, over RMSE / MAE / Wasserstein (1-D Earth Mover's distance) /
MK-MMD (multi-kernel Maximum Mean Discrepancy):

```
D_m   = metric_m ** 2                                        # square every raw metric
D~_m  = (D_m - min(D_m)) / (max(D_m) - min(D_m) + eps)        # rolling-window min-max normalize
S_raw,c = D~_RMSE + D~_MAE + D~_W1 + D~_MMD
FidelityScore_c = 1 - (S_raw,c / 4)
```

`min(D_m)`/`max(D_m)` are taken over a rolling window of past squared-metric values
(`config.fidelity.rolling_window_length`), from **before** the current point — never
self-referential. `eps` is config-driven. Fewer than `min_history_for_normalization` prior
points -> `status="insufficient_history"`, `FidelityScore_c=None` (never fabricated).

### 2.2 Unified (system-wide) fidelity

```
UnifiedFidelityScore = Σ_c (w_c * FidelityScore_c) / Σ_c w_c      # over components with a defined score
```

`w_c` from `config.fidelity.unified_weights` (equal weight by default). `None` if fewer than
`unified_min_components` components currently have a defined score.

### 2.3 Acceptance gate (`src/adaptation/verification_agent.py`)

```
FidelityScore_new > FidelityScore_old + delta        (delta = config.adaptation.verification_delta)
```

plus mandatory checks (artifact present, interface consistent, evaluation ran, output finite) —
any failure -> REJECT. A brand-new (expand-scope) component has no `FidelityScore_old`: the gate
falls back to requiring only that the candidate's own score is defined.

### 2.4 Comparison-table metrics (`baselines/compare.py`, computed ONLY from already-written per-mode output — never re-run)

| column | formula |
|---|---|
| `mean_unified_fidelity` | mean of each run's `run_summary.json["mean_unified_fidelity"]`, averaged across seeds |
| `mean_unified_fidelity_std` | std of the same across seeds (0 for single-seed modes) |
| `pct_time_above_threshold` | `% of (cycle, F_U) pairs across all runs where F_U >= trigger_threshold` (threshold = `config.fidelity.trigger_threshold`, read live) |
| `mean_recovery_time_cycles` | mean, over every below-threshold "dip" in the unified-fidelity timeline, of `(cycle it recovered above threshold) - (cycle it first dipped below)`; `None` — not 0 — if a run never dipped |
| `n_triggers` / `n_adaptations` / `n_accepted` / `n_rejected` | summed across seeds; `n_adaptations = n_accepted + n_rejected` |
| `fallback_rate` | `n_llm_fallbacks / n_llm_decisions` (0 if no decisions were made — `static`/`periodic_retrain`) |
| `run_invalid_fallback_rate_too_high` | `True` if `fallback_rate > fallback_rate_invalid_threshold` (full_system's config, default 0.1) — `static`/`periodic_retrain`/`rule_based` can never trip this (no decision-agent calls to fall back from) |
| `llm_cache_hit_share` | `total_cached_tokens / total_input_tokens` across every `llm_calls.jsonl` entry for the mode |
| `llm_cost_usd` / `llm_time_seconds` | summed `cost_usd`/`latency_seconds` across every real LLM call logged for the mode |

### 2.5 Display units (`baselines/_outputs.py:DISPLAY_UNITS`)

Applied **only at plot time** — `metrics_timeline.csv` itself always stores raw units:

| component | stored (raw) | displayed |
|---|---|---|
| `dl_throughput`, `ul_throughput` | kbps | Mbps (÷1000) |
| `dl_rlc_delay` | microseconds | ms (÷1000) |
| `dl_bler`, `ul_bler`, `dl_prb`, `ul_prb` | ratio / PRBs-per-second | unchanged |

### 2.6 Model-vs-baseline CV comparison (`scripts/bootstrap_oai_models.py`, §7 below)

Leave-one-run-out cross-validation over the train/tune pool (runs 3, 6, 7): for each held-out
run, RMSE/MAE of (a) the real trained model, (b) a **persistence** baseline (`ŷ_t = y_{t-1}`),
(c) a **5-second moving-average** baseline — never fabricated, read straight from
`reports/model_eval/model_eval.csv`.

---

## 3. Graphs produced

### 3.1 Per-mode (one set per completed mode/seed, under `baselines/results/<mode>/graphs/` — or `.../seed_<n>/graphs/`)

| file | content |
|---|---|
| `actual_vs_predicted_<component>.{png,pdf,csv}` (×7) | actual vs. predicted value per cycle, display units |
| `fidelity_<component>.{png,pdf,csv}` (×7) | `F_c` over cycles, threshold line, trigger markers (×), ACCEPT/REJECT vertical lines (green/red) |
| `unified_fidelity.{png,pdf,csv}` | `F_U` over cycles, threshold line, ACCEPT/REJECT vertical lines |

### 3.2 Cross-mode comparison (`baselines/results/comparison/graphs/`, built once by `compare.py` from already-written output)

| file | content |
|---|---|
| `unified_fidelity_all_modes.{png,pdf,csv}` | one line per mode, `F_U` vs. cycle, shared threshold line |
| `mean_fidelity_by_mode.{png,pdf,csv}` | bar chart, mean `F_U` per mode, error bars = std across seeds |
| `recovery_time_by_mode.{png,pdf,csv}` | bar chart, mean recovery-time-in-cycles per mode |
| `adaptations_by_mode.{png,pdf,csv}` | stacked bar, accepted (green) vs. rejected (red) per mode |
| `strategy_distribution.{png,pdf,csv}` | grouped bar, `recalibrate`/`regenerate`/`expand_scope` counts, `rule_based` vs. `full_system` only |

### 3.3 Model-vs-baseline (`reports/model_eval/graphs/`, built once by `scripts/plot_model_eval.py` from already-written CV results — never re-trains)

| file | content |
|---|---|
| `model_vs_baselines_<component>.{png,pdf,csv}` (×7) | grouped bar per held-out fold: model RMSE vs. persistence RMSE vs. 5s-MA RMSE |

---

## 4. Readable outputs per run

| file | content |
|---|---|
| `metrics_timeline.csv` | `ts, cycle, component, actual, predicted, abs_error, F_c, F_U, trigger_fired, model_version` — one row per component per cycle, raw units |
| `adaptation_events.jsonl` | one line per adaptation attempt: trigger, decision (strategy/confidence/root-cause), `gate_result`, `fidelity_before`/`after`, `candidate_version`, duration |
| `report.md` | header (mode/run/seeds/model versions/fidelity config) + per-component summary table + one block per adaptation event |
| `run_summary.json` | `n_triggers`/`n_accepted`/`n_rejected`/`mean_unified_fidelity`/`budget_spent_usd`/... |
| `llm_calls.jsonl` | one line per real LLM call: role, tokens, cached_tokens, cost_usd, latency, used_cache — absent for `static`/`periodic_retrain` (no LLM) |

`baselines/results/comparison/{comparison.md,comparison.csv}` is the single cross-mode summary,
built once from all of the above.
