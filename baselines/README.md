# Ablation ladder (baselines)

Four conditions, replayed over the real OAI TEST split (run 8, 61.5 min), all scored on the same
62 fixed 60-second windows of replayed data:

| step | mode | what adapts, and how | LLM used for |
|---|---|---|---|
| 0 | `static` | nothing — per-component triggers are detected and logged, never dispatched | nothing |
| 1 | `periodic_retrain` | all 7 components recalibrated every 10 min of data, **deployed without the verification gate** (blind) | nothing |
| 2 | `rule_based` | per-component fidelity trigger → severity rule (≤0.3 recalibrate, ≤0.7 regenerate, else expand scope) → agents → sandbox → verification gate → hot swap, with the live retry loop | codegen + gate explanation only |
| 3 | `full_system` | per-component fidelity trigger → LLM+RAG Decision Agent → agents → sandbox → verification gate → hot swap, with the live retry loop and rejection feedback; **3 runs** | decision + codegen + explanation |

## Controlled-ablation guarantees (enforced in `_lib.py` / `run_all.py`)

- **Same bootstrap models**: `bootstrap_components()` trains the 7 real components on the TRAIN pool once per run, identically for every mode.
- **Same data and windows**: `replay_test_split()` replays run 8; `window_edges()` defines the fixed 60 s windows of data time. Every mode is scored on exactly these windows.
- **No look-ahead**: run 8 is bulk-replayed up front, so agents read D1 through `CutoffD1View`. An adaptation started after window *k* only sees windows 1..*k*. (Before 2026-10-20 the agents saw all of run 8, including the windows they were later scored on.)
- **Same fidelity config and trigger policy**: one `FidelityEvaluator` and one `ComponentAdaptationPolicy` from `settings.fidelity` per run; no mode-specific fidelity settings.
- **Same adaptation inputs as the live system**: reference TRAIN data with ×5 recent weighting, retry loop up to `adaptation.max_adaptation_attempts` with rejection feedback, and the persistence baseline for expand-scope verification, all mirrored from `src/main.py`.
- **No external drift**: the live system's per-component fidelity policy is the only trigger source. Mock drift events are no longer injected.
- **Fresh state**: each run deletes and rebuilds its own `results/<mode>/` (or `seed_N/`) directory, so no registry, D1 or event state is shared.

Data time is frozen while an adaptation runs (the harness is synchronous). Recovery latency is therefore measured in data minutes; wall-clock and CPU cost of each attempt are recorded separately.

## Statistics (`stats.py`)

Each run's runtime `F_c` is normalized against **that run's own** rolling history. That suits triggering, but the scores are not comparable across modes. `stats.py` therefore re-scores every window with the same fidelity formula (square → min-max normalize → sum → `1 − S/4`), using **one frozen reference**: the min/max of each squared metric, pooled over every window of every run. The runtime formula in `src/fidelity` is unchanged.

- Window score = equal-weight mean over the 7 components, on windows every run scored.
- Comparisons, paired per window: Proposed vs Static, vs Periodic, vs Rule-based; Rule-based vs Periodic. The 3 full-system runs are averaged per window before pairing.
- Wilcoxon signed-rank test, two-sided. Effect size is the matched-pairs rank-biserial *r*.
- 95% moving-block bootstrap CI of the mean difference (block = 5 windows, 10,000 resamples, seed 0).
- Holm-Bonferroni adjustment across the 4 comparisons.
- Caveat: consecutive windows are autocorrelated, so Wilcoxon p-values are optimistic. The block-bootstrap CI is the primary evidence.

## Running it

```bash
python baselines/run_all.py                                     # all 4 modes, full run 8 (~2-3 h; LLM cost ~$0.1-0.5)
python baselines/run_all.py --modes static periodic_retrain --skip-preflight   # no LLM
python baselines/run_all.py --max-windows 15 --seeds 99 --skip-preflight      # smoke run
python baselines/stats.py                                       # -> results/stats/ (frozen-scale scores, tests, report.md)
python baselines/compare.py                                     # -> results/comparison/ (operational counts)
python scripts/generate_ieee_figures.py                         # -> figures/ (7 IEEE figures, PDF+PNG+CSV)
```

`results_archive_pre_2026-10-20/` holds the previous results. They were produced with the leaky, self-normalized, mock-drift protocol and must not be used for publication.
