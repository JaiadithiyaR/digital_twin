#!/usr/bin/env python3
"""baselines/stats.py — statistical validation of the ablation ladder, built ONLY from what
`run_all.py` already wrote (never re-runs anything). Writes `baselines/results/stats/`.

Why a separate scoring pass: each run's RUNTIME fidelity score F_c is min-max normalized against
that run's OWN rolling history (CLAUDE.md §5) — the right signal for triggering, but not a common
scale across modes (a uniformly bad model can score as high as a uniformly good one). For the
cross-mode comparison every run is re-scored with the SAME formula —

    D_m = metric_m ** 2;  D~_m = (D_m - min D_m) / (max D_m - min D_m + eps);  F = 1 - sum_m D~_m / 4

— but with ONE frozen normalization reference per (component, metric): the min/max of D_m pooled
over every window of every run of every mode being compared (`frozen_reference.json`). The
runtime formula in src/fidelity is untouched; only the reference set differs, and it is identical
for all conditions. Window score = equal-weight mean over the 7 components (the unified-score
default), on windows where every run scored all 7 components.

Statistics (all on the paired per-window series, 60 s windows of replayed run 8):
  - Wilcoxon signed-rank test (two-sided, zero differences dropped) per comparison pair,
    effect size = matched-pairs rank-biserial correlation r_rb in [-1, 1] (> 0 favours the first mode)
  - 95% moving-block bootstrap CI of the mean paired difference (windows are autocorrelated;
    block length and resample count from the CLI, fixed RNG seed)
  - Holm-Bonferroni adjustment across the comparison family
full_system's 3 runs are averaged per window before pairing; their spread is reported separately.

Usage:  python baselines/stats.py [--block-length 5] [--n-boot 10000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines._lib import MODES, ORDER, RAW_METRICS

RESULTS_DIR = Path(__file__).resolve().parent / "results"
STATS_DIR = RESULTS_DIR / "stats"
PAIRS: tuple[tuple[str, str], ...] = (
    ("full_system", "static"), ("full_system", "periodic_retrain"), ("full_system", "rule_based"), ("rule_based", "periodic_retrain"),
)
SEVERITY_BINS = ((0.0, 0.3, "minor"), (0.3, 0.7, "moderate"), (0.7, 1.0001, "major"))


def _settings_values() -> tuple[float, float, int]:
    import yaml

    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "settings.yaml").read_text(encoding="utf-8"))
    return float(cfg["fidelity"]["epsilon"]), float(cfg["fidelity"]["trigger_threshold"]), int(cfg["baselines"]["window_seconds"])


def load_runs() -> list[dict[str, Any]]:
    runs = []
    for mode in MODES:
        base = RESULTS_DIR / mode
        dirs = sorted(base.glob("seed_*")) or ([base] if (base / "run_summary.json").exists() else [])
        for d in dirs:
            if not (d / "metrics_timeline.csv").exists():
                continue
            timeline = pd.read_csv(d / "metrics_timeline.csv")
            if "window" not in timeline.columns:
                print(f"skipping {d}: pre-2026-10-20 output format (no 60 s windows) — re-run run_all.py")
                continue
            events = []
            if (d / "adaptation_events.jsonl").exists():
                events = [json.loads(line) for line in (d / "adaptation_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            calls = []
            if (d / "llm_calls.jsonl").exists():
                calls = [json.loads(line) for line in (d / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            summary = json.loads((d / "run_summary.json").read_text(encoding="utf-8"))
            runs.append({"mode": mode, "seed": summary.get("seed", 1), "dir": d, "timeline": timeline, "events": events, "calls": calls, "summary": summary})
    return runs


def frozen_reference(runs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    pooled = pd.concat([r["timeline"] for r in runs], ignore_index=True)
    ref: dict[str, dict[str, dict[str, float]]] = {}
    for comp in ORDER:
        sub = pooled[pooled["component"] == comp]
        ref[comp] = {m: {"min": float((sub[m] ** 2).min()), "max": float((sub[m] ** 2).max())} for m in RAW_METRICS}
    return ref


def rescore(timeline: pd.DataFrame, ref: dict, eps: float) -> pd.DataFrame:
    """Per (window, component): frozen-reference normalized metrics and frozen F_c."""
    df = timeline[timeline["component"].isin(ORDER)].copy()
    for m in RAW_METRICS:
        lo = df["component"].map(lambda c, m=m: ref[c][m]["min"])
        hi = df["component"].map(lambda c, m=m: ref[c][m]["max"])
        df[f"norm_{m}"] = ((df[m] ** 2) - lo) / (hi - lo + eps)
    df["F_frozen"] = 1.0 - df[[f"norm_{m}" for m in RAW_METRICS]].sum(axis=1) / 4.0
    return df


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = a - b
    d = d[d != 0]
    n = len(d)
    if n < 6:
        return {"n_nonzero": n, "W": math.nan, "p": math.nan, "r_rb": math.nan}
    res = sps.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
    ranks = sps.rankdata(np.abs(d))
    w_plus, w_minus = ranks[d > 0].sum(), ranks[d < 0].sum()
    return {"n_nonzero": n, "W": float(res.statistic), "p": float(res.pvalue), "r_rb": float((w_plus - w_minus) / (w_plus + w_minus))}


def block_bootstrap_ci(x: np.ndarray, block: int, n_boot: int, rng: np.random.Generator, alpha: float = 0.05) -> tuple[float, float]:
    """Moving-block bootstrap CI of the mean (Kunsch 1989): resample overlapping blocks of
    `block` consecutive windows, concatenate, truncate to n."""
    n = len(x)
    if n == 0:
        return math.nan, math.nan
    block = max(1, min(block, n))
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (NaN entries are left NaN and not counted)."""
    idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    m = len(idx)
    order = sorted(idx, key=lambda i: pvals[i])
    adjusted = [math.nan] * len(pvals)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * pvals[i]))
        adjusted[i] = running
    return adjusted


def recovery_latencies(runs: list[dict[str, Any]], threshold: float, window_seconds: int) -> pd.DataFrame:
    """Per first-attempt trigger: data minutes from the trigger window until the component's
    runtime F_c is next measured at-or-above its threshold (censored if it never is). Triggers
    are defined on the runtime F_c, so recovery is measured on the same signal."""
    rows = []
    for r in runs:
        fc = r["timeline"].pivot_table(index="window", columns="component", values="F_c", aggfunc="first")
        last = int(r["timeline"]["window"].max())
        for ev in r["events"]:
            if ev.get("trigger_type") != "fidelity_degradation" or ev.get("attempt", 1) != 1:
                continue
            comp, w0 = ev["component"], int(ev["window"])
            later = fc[comp].loc[w0 + 1:] if comp in fc.columns else pd.Series(dtype=float)
            ok = later[later >= threshold]
            recovered = len(ok) > 0
            rows.append({
                "mode": r["mode"], "seed": r["seed"], "component": comp, "trigger_window": w0, "severity": ev.get("severity"),
                "recovered": recovered,
                "latency_min": ((int(ok.index[0]) - w0) * window_seconds / 60.0) if recovered else ((last - w0) * window_seconds / 60.0),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--block-length", type=int, default=5, help="moving-block length in 60 s windows")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    eps, threshold, window_seconds = _settings_values()
    rng = np.random.default_rng(args.seed)

    runs = load_runs()
    if not runs:
        print("No 60 s-window results under baselines/results/ — run baselines/run_all.py first.")
        return
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    ref = frozen_reference(runs)
    (STATS_DIR / "frozen_reference.json").write_text(json.dumps({"epsilon": eps, "reference": ref}, indent=2), encoding="utf-8")

    scored = []
    for r in runs:
        s = rescore(r["timeline"], ref, eps)
        s["mode"], s["seed"] = r["mode"], r["seed"]
        scored.append(s)
    scored_df = pd.concat(scored, ignore_index=True)
    scored_df.to_csv(STATS_DIR / "windowed_component_scores.csv", index=False)

    # Common support: windows where every run scored all 7 components.
    counts = scored_df.groupby(["mode", "seed", "window"])["component"].nunique()
    complete = counts[counts == len(ORDER)].reset_index()[["mode", "seed", "window"]]
    n_runs = scored_df[["mode", "seed"]].drop_duplicates().shape[0]
    common = complete.groupby("window").size()
    common_windows = sorted(common[common == n_runs].index.tolist())
    per_run = (
        scored_df[scored_df["window"].isin(common_windows)]
        .groupby(["mode", "seed", "window"])[["F_frozen", *[f"norm_{m}" for m in RAW_METRICS]]].mean().reset_index()
    )
    per_run.to_csv(STATS_DIR / "windowed_scores_per_run.csv", index=False)
    per_mode = per_run.groupby(["mode", "window"]).agg(
        F_frozen=("F_frozen", "mean"), F_frozen_sd=("F_frozen", "std"), n_runs=("seed", "nunique"),
        **{f"norm_{m}": (f"norm_{m}", "mean") for m in RAW_METRICS},
    ).reset_index()
    per_mode.to_csv(STATS_DIR / "windowed_scores_per_mode.csv", index=False)

    # Per-mode summary with block-bootstrap CI of the mean windowed score.
    summary_rows = []
    modes_present = [m for m in MODES if m in set(per_mode["mode"])]
    for mode in modes_present:
        series = per_mode[per_mode["mode"] == mode].sort_values("window")
        x = series["F_frozen"].to_numpy()
        lo, hi = block_bootstrap_ci(x, args.block_length, args.n_boot, rng)
        run_means = per_run[per_run["mode"] == mode].groupby("seed")["F_frozen"].mean()
        row = {
            "mode": mode, "n_runs": int(series["n_runs"].max()), "n_windows": len(x), "mean_F_frozen": float(x.mean()),
            "ci95_low": lo, "ci95_high": hi, "sd_across_windows": float(x.std(ddof=1)) if len(x) > 1 else math.nan,
            "sd_across_runs": float(run_means.std(ddof=1)) if len(run_means) > 1 else math.nan,
        }
        for m in RAW_METRICS:
            v = series[f"norm_{m}"].to_numpy()
            row[f"mean_norm_{m}"], row[f"sd_norm_{m}"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else math.nan
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # Operational cost per mode (mean per run).
    cost_rows = []
    for mode in modes_present:
        mruns = [r for r in runs if r["mode"] == mode]
        cost_rows.append({
            "mode": mode,
            "adaptation_wall_seconds": float(np.mean([r["summary"].get("adaptation_wall_seconds", 0.0) for r in mruns])),
            "adaptation_cpu_seconds": float(np.mean([r["summary"].get("adaptation_cpu_seconds", 0.0) for r in mruns])),
            "llm_tokens": float(np.mean([sum(c.get("input_tokens", 0) + c.get("output_tokens", 0) for c in r["calls"]) for r in mruns])),
            "llm_cost_usd": float(np.mean([sum(c.get("cost_usd", 0.0) for c in r["calls"]) for r in mruns])),
            "n_attempts": float(np.mean([r["summary"].get("n_attempts", 0) for r in mruns])),
            "n_accepted": float(np.mean([r["summary"].get("n_accepted", 0) for r in mruns])),
            "n_rejected": float(np.mean([r["summary"].get("n_rejected", 0) for r in mruns])),
            "n_deployed_ungated": float(np.mean([r["summary"].get("n_deployed_ungated", 0) for r in mruns])),
            "n_agent_failures": float(np.mean([r["summary"].get("n_agent_failures", 0) for r in mruns])),
            "n_triggers": float(np.mean([r["summary"].get("n_triggers", 0) for r in mruns])),
        })
    summary_df = summary_df.merge(pd.DataFrame(cost_rows), on="mode")
    summary_df.to_csv(STATS_DIR / "mode_summary.csv", index=False)

    # Pairwise tests.
    pair_rows = []
    wide = per_mode.pivot(index="window", columns="mode", values="F_frozen")
    for a, b in PAIRS:
        if a not in wide.columns or b not in wide.columns:
            continue
        xa, xb = wide[a].to_numpy(), wide[b].to_numpy()
        diff = xa - xb
        lo, hi = block_bootstrap_ci(diff, args.block_length, args.n_boot, rng)
        pair_rows.append({"mode_a": a, "mode_b": b, "n_windows": len(diff), "mean_diff": float(diff.mean()), "ci95_low": lo, "ci95_high": hi, **wilcoxon_paired(xa, xb)})
    pairs_df = pd.DataFrame(pair_rows)
    if len(pairs_df):
        pairs_df["p_holm"] = holm(pairs_df["p"].tolist())
        pairs_df["significant_0.05"] = pairs_df["p_holm"] < 0.05
    pairs_df.to_csv(STATS_DIR / "pairwise_tests.csv", index=False)

    # Events, decisions, recovery.
    ev_rows = []
    for r in runs:
        for ev in r["events"]:
            if "attempt" not in ev:
                continue
            sev = ev.get("severity") or 0.0
            ev_rows.append({
                "mode": r["mode"], "seed": r["seed"], "window": ev.get("window"), "component": ev.get("component"),
                "attempt": ev.get("attempt"), "trigger_type": ev.get("trigger_type"), "severity": sev,
                "severity_bin": next(lbl for lo_, hi_, lbl in SEVERITY_BINS if lo_ <= sev < hi_),
                "strategy": (ev.get("decision") or {}).get("strategy"), "decision_source": (ev.get("decision") or {}).get("llm_or_fallback"),
                "gate_result": ev.get("gate_result"), "error": bool(ev.get("error")),
                "fidelity_before": ev.get("fidelity_before"), "fidelity_after": ev.get("fidelity_after"),
                "post_adaptation_fidelity": ev.get("post_adaptation_fidelity"),
                "duration_seconds": ev.get("duration_seconds"), "cpu_seconds": ev.get("cpu_seconds"),
            })
    events_df = pd.DataFrame(ev_rows)
    events_df.to_csv(STATS_DIR / "adaptation_attempts.csv", index=False)
    recovery_df = recovery_latencies(runs, threshold, window_seconds)
    recovery_df.to_csv(STATS_DIR / "recovery_latency.csv", index=False)

    # KPI traces for the RAN-domain figure (actual vs predicted window means, all runs).
    kpi = pd.concat([r["timeline"].assign(mode=r["mode"], seed=r["seed"]) for r in runs], ignore_index=True)
    kpi[["mode", "seed", "window", "component", "actual", "predicted"]].to_csv(STATS_DIR / "kpi_traces.csv", index=False)

    lines = [
        "# Ablation statistics (run 8, 60 s windows)", "",
        f"- Runs: {', '.join(f'{r['mode']}/seed{r['seed']}' for r in runs)}",
        f"- Common windows scored by every run: {len(common_windows)}",
        f"- Frozen normalization reference: pooled min/max of squared metrics over all runs (`frozen_reference.json`), epsilon={eps}",
        f"- Moving-block bootstrap: block={args.block_length} windows, {args.n_boot} resamples, seed={args.seed}", "",
        "## Mean windowed fidelity (frozen scale)", "", summary_df.to_string(index=False), "",
        "## Pairwise tests (Wilcoxon signed-rank, Holm-adjusted)", "", pairs_df.to_string(index=False) if len(pairs_df) else "(not enough modes)", "",
        (
            "Caveat: the Wilcoxon test assumes independent pairs; consecutive 60 s windows are autocorrelated, so "
            "its p-values are optimistic. The block-bootstrap CI of the mean difference accounts for that "
            "dependence and should be read as the primary evidence."
        ),
    ]
    (STATS_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {STATS_DIR}")


if __name__ == "__main__":
    main()
