#!/usr/bin/env python3
"""generate_ieee_figures.py — the 7 IEEE-format evaluation figures for the self-adaptive OAI
digital twin ablation (static / periodic / rule-based / proposed, OAI run 8, 60 s windows).

Data source, in order:
  1. MEASURED: the CSVs `baselines/stats.py` writes to `baselines/results/stats/` (built from
     `baselines/run_all.py`'s real runs). This is the only source fit for publication.
  2. SYNTHETIC placeholder (only when the stats CSVs are missing, or with `--synthetic`): data in
     the same schema so every figure can be laid out before the long runs finish. Placeholder
     figures are watermarked "SYNTHETIC PLACEHOLDER — NOT MEASURED" and saved with a
     `_SYNTHETIC` suffix. The synthetic generator deliberately gives every mode the SAME
     underlying fidelity distribution, so a placeholder can never be mistaken for evidence of an
     effect; its costs/counts are illustrative only.

Every figure is written as vector PDF + 300 dpi PNG, plus a CSV of exactly the plotted data.
Standalone: needs only numpy, pandas, scipy, matplotlib (seaborn optional, used for styling).

Usage:
    python scripts/generate_ieee_figures.py                         # measured if available
    python scripts/generate_ieee_figures.py --synthetic             # force placeholder data
    python scripts/generate_ieee_figures.py --drift-minutes 14 37   # mark KNOWN drift times
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

try:
    import seaborn as sns
except ImportError:  # styling only — every figure is pure matplotlib
    sns = None

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_STATS_DIR = REPO / "baselines" / "results" / "stats"
SINGLE_COL, DOUBLE_COL = 3.5, 7.16  # inches (IEEE Transactions column / page width)

MODES = ("static", "periodic_retrain", "rule_based", "full_system")
LABEL = {
    "static": "Static (Step 0)", "periodic_retrain": "Periodic (Step 1)",
    "rule_based": "Rule-based (Step 2)", "full_system": "Proposed (Step 3)",
}
SHORT = {"static": "Static", "periodic_retrain": "Periodic", "rule_based": "Rule-based", "full_system": "Proposed"}
# Okabe-Ito (colour-blind safe) + distinct line styles, markers and hatches so every series
# stays distinguishable in grayscale print.
STYLE = {
    "static": {"color": "#5A5A5A", "ls": ":", "marker": "s", "hatch": "..."},
    "periodic_retrain": {"color": "#E69F00", "ls": "-.", "marker": "^", "hatch": "//"},
    "rule_based": {"color": "#009E73", "ls": "--", "marker": "D", "hatch": "xx"},
    "full_system": {"color": "#D55E00", "ls": "-", "marker": "o", "hatch": ""},
}
STRATEGIES = ("recalibrate", "regenerate", "expand_scope")
STRAT_STYLE = {
    "recalibrate": {"color": "#56B4E9", "hatch": ""},
    "regenerate": {"color": "#0072B2", "hatch": "//"},
    "expand_scope": {"color": "#CC79A7", "hatch": "xx"},
}
METRICS = ("rmse", "mae", "wasserstein", "mk_mmd")
METRIC_LABEL = {"rmse": "RMSE", "mae": "MAE", "wasserstein": "Wasserstein", "mk_mmd": "MK-MMD"}
SEVERITY_BINS = ("minor", "moderate", "major")
PAIRS = (("full_system", "static"), ("full_system", "periodic_retrain"), ("full_system", "rule_based"), ("rule_based", "periodic_retrain"))


# ----------------------------------------------------------------------------- data


@dataclass
class Data:
    per_mode: pd.DataFrame  # mode, window, F_frozen, F_frozen_sd, n_runs, norm_<metric>
    per_run: pd.DataFrame  # mode, seed, window, F_frozen, norm_<metric>
    summary: pd.DataFrame  # mode, mean_F_frozen, ci95_low/high, costs, counts
    pairs: pd.DataFrame  # mode_a, mode_b, mean_diff, ci95_low/high, r_rb, p, p_holm
    attempts: pd.DataFrame  # mode, seed, window, attempt, trigger_type, severity_bin, strategy, gate_result, error, fidelity_before/after
    recovery: pd.DataFrame  # mode, seed, component, trigger_window, recovered, latency_min
    kpi: pd.DataFrame  # mode, seed, window, component, actual, predicted
    synthetic: bool
    window_minutes: float = 1.0


def load_measured(stats_dir: Path) -> Data | None:
    needed = ("windowed_scores_per_mode.csv", "windowed_scores_per_run.csv", "mode_summary.csv", "pairwise_tests.csv",
              "adaptation_attempts.csv", "recovery_latency.csv", "kpi_traces.csv")
    if not all((stats_dir / f).exists() for f in needed):
        return None

    def read(name: str) -> pd.DataFrame:
        try:
            return pd.read_csv(stats_dir / name)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    return Data(
        per_mode=read("windowed_scores_per_mode.csv"), per_run=read("windowed_scores_per_run.csv"),
        summary=read("mode_summary.csv"), pairs=read("pairwise_tests.csv"), attempts=read("adaptation_attempts.csv"),
        recovery=read("recovery_latency.csv"), kpi=read("kpi_traces.csv"), synthetic=False,
    )


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = a - b
    d = d[d != 0]
    if len(d) < 6:
        return math.nan, math.nan
    ranks = sps.rankdata(np.abs(d))
    r_rb = (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()
    return float(r_rb), float(sps.wilcoxon(d).pvalue)


def _block_ci(x: np.ndarray, rng: np.random.Generator, block: int = 5, n_boot: int = 4000) -> tuple[float, float]:
    n = len(x)
    starts = rng.integers(0, n - block + 1, size=(n_boot, math.ceil(n / block)))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _holm(p: list[float]) -> list[float]:
    order = sorted(range(len(p)), key=lambda i: p[i])
    out, running = [math.nan] * len(p), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        out[i] = running
    return out


def make_synthetic(n_windows: int = 62, seed: int = 7) -> Data:
    """Placeholder data in the measured schema. All modes share ONE fidelity distribution (no
    built-in effect); only independent noise differs. Costs and counts are illustrative."""
    rng = np.random.default_rng(seed)
    w = np.arange(1, n_windows + 1)
    regime = 0.72 + 0.08 * np.sin(w / 9.0) - 0.12 * ((w > 20) & (w < 32))  # shared network regime
    ar = np.zeros(n_windows)
    for i in range(1, n_windows):
        ar[i] = 0.6 * ar[i - 1] + rng.normal(0, 0.03)
    base = np.clip(regime + ar, 0.05, 0.98)

    run_rows = []
    for mode in MODES:
        for s in (1, 2, 3) if mode == "full_system" else (1,):
            f = np.clip(base + rng.normal(0, 0.025, n_windows), 0, 1)
            for i, wi in enumerate(w):
                norms = np.clip((1 - f[i]) * rng.uniform(0.6, 1.4, 4), 0, 1)
                norms = norms * (4 * (1 - f[i]) / max(norms.sum(), 1e-9))
                run_rows.append({"mode": mode, "seed": s, "window": int(wi), "F_frozen": f[i], **{f"norm_{m}": norms[k] for k, m in enumerate(METRICS)}})
    per_run = pd.DataFrame(run_rows)
    per_mode = per_run.groupby(["mode", "window"]).agg(
        F_frozen=("F_frozen", "mean"), F_frozen_sd=("F_frozen", "std"), n_runs=("seed", "nunique"),
        **{f"norm_{m}": (f"norm_{m}", "mean") for m in METRICS},
    ).reset_index()

    cost = {  # illustrative only
        "static": (0.0, 0.0, 0.0), "periodic_retrain": (1300.0, 0.0, 0.0),
        "rule_based": (420.0, 6.0e4, 0.01), "full_system": (520.0, 1.6e5, 0.03),
    }
    summary_rows, wide = [], per_mode.pivot(index="window", columns="mode", values="F_frozen")
    for mode in MODES:
        x = wide[mode].to_numpy()
        lo, hi = _block_ci(x, rng)
        summary_rows.append({
            "mode": mode, "n_runs": 3 if mode == "full_system" else 1, "n_windows": n_windows, "mean_F_frozen": x.mean(),
            "ci95_low": lo, "ci95_high": hi, "sd_across_windows": x.std(ddof=1),
            **{f"mean_norm_{m}": per_mode.loc[per_mode["mode"] == mode, f"norm_{m}"].mean() for m in METRICS},
            **{f"sd_norm_{m}": per_mode.loc[per_mode["mode"] == mode, f"norm_{m}"].std() for m in METRICS},
            "adaptation_cpu_seconds": cost[mode][0], "llm_tokens": cost[mode][1], "llm_cost_usd": cost[mode][2],
        })
    summary = pd.DataFrame(summary_rows)

    pair_rows = []
    for a, b in PAIRS:
        diff = wide[a].to_numpy() - wide[b].to_numpy()
        r_rb, p = _wilcoxon(wide[a].to_numpy(), wide[b].to_numpy())
        lo, hi = _block_ci(diff, rng)
        pair_rows.append({"mode_a": a, "mode_b": b, "mean_diff": diff.mean(), "ci95_low": lo, "ci95_high": hi, "r_rb": r_rb, "p": p})
    pairs = pd.DataFrame(pair_rows)
    pairs["p_holm"] = _holm(pairs["p"].tolist())

    att_rows = []
    for mode in ("rule_based", "full_system"):
        for s in (1, 2, 3) if mode == "full_system" else (1,):
            for _ in range(14):
                sev = float(rng.beta(2, 4))
                sbin = "minor" if sev < 0.3 else ("moderate" if sev < 0.7 else "major")
                strat = ({"minor": "recalibrate", "moderate": "regenerate", "major": "expand_scope"}[sbin]
                         if mode == "rule_based" else str(rng.choice(STRATEGIES, p=[0.5, 0.4, 0.1])))
                outcome = rng.choice(["ACCEPT", "REJECT", "ERROR"], p=[0.3, 0.6, 0.1])
                fb = float(rng.uniform(0.2, 0.5))
                att_rows.append({
                    "mode": mode, "seed": s, "window": int(rng.integers(13, n_windows)), "attempt": 1, "trigger_type": "fidelity_degradation",
                    "severity": sev, "severity_bin": sbin, "strategy": strat,
                    "gate_result": None if outcome == "ERROR" else outcome, "error": outcome == "ERROR",
                    "fidelity_before": fb, "fidelity_after": fb + (0.05 if outcome == "ACCEPT" else rng.normal(-0.03, 0.04)),
                })
    for k in range(6):  # periodic: 6 rounds x 7 components, deployed ungated
        att_rows += [{"mode": "periodic_retrain", "seed": 1, "window": 10 * (k + 1), "attempt": 1, "trigger_type": "scheduled",
                      "severity": 0.0, "severity_bin": "minor", "strategy": "recalibrate", "gate_result": "DEPLOYED_UNGATED",
                      "error": False, "fidelity_before": np.nan, "fidelity_after": np.nan}] * 7
    attempts = pd.DataFrame(att_rows)

    rec_rows = []
    for mode, scale in (("static", 6.0), ("rule_based", 6.0), ("full_system", 6.0)):
        for s in (1, 2, 3) if mode == "full_system" else (1,):
            for _ in range(10):
                lat = float(np.ceil(rng.exponential(scale)))
                rec_rows.append({"mode": mode, "seed": s, "component": "dl_bler", "trigger_window": 20, "recovered": lat < 30, "latency_min": min(lat, 30.0)})
    recovery = pd.DataFrame(rec_rows)

    kpi_rows = []
    truth = {
        "dl_throughput": np.clip(0.2 + 0.15 * np.sin(w / 5) + rng.normal(0, 0.05, n_windows), 0, None),
        "dl_rlc_delay": np.clip(0.5 + 0.4 * np.sin(w / 7 + 1) + rng.normal(0, 0.1, n_windows), 0, None),
        "dl_bler": np.clip(0.05 + 0.03 * np.sin(w / 6) + rng.normal(0, 0.01, n_windows), 0, 1),
    }
    for mode in ("static", "full_system"):
        for s in (1, 2, 3) if mode == "full_system" else (1,):
            for comp, y in truth.items():
                pred = y * rng.normal(1.0, 0.15, n_windows)
                kpi_rows += [{"mode": mode, "seed": s, "window": int(wi), "component": comp, "actual": y[i], "predicted": pred[i]} for i, wi in enumerate(w)]
    return Data(per_mode, per_run, summary, pairs, attempts, recovery, pd.DataFrame(kpi_rows), synthetic=True)


# ----------------------------------------------------------------------------- style/io


def set_ieee_style() -> None:
    if sns is not None:
        sns.set_theme(context="paper", style="ticks")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "axes.linewidth": 0.6, "grid.linewidth": 0.4, "grid.alpha": 0.35, "axes.grid": True,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "legend.handlelength": 2.2, "hatch.linewidth": 0.5,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig: plt.Figure, out_dir: Path, name: str, data: Data, plotted: pd.DataFrame) -> None:
    if data.synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC PLACEHOLDER — NOT MEASURED", transform=fig.transFigure, ha="center", va="center",
                 rotation=22, fontsize=14, color="red", alpha=0.22, zorder=100)
        name += "_SYNTHETIC"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plotted.to_csv(out_dir / f"{name}.csv", index=False)
    plt.close(fig)
    print(f"  wrote {out_dir / name}.{{pdf,png,csv}}")


def _present(data: Data, frame: pd.DataFrame) -> list[str]:
    have = set(frame["mode"]) if len(frame) and "mode" in frame.columns else set()
    return [m for m in MODES if m in have]


def _panel_label(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left")


# ----------------------------------------------------------------------------- figures


def fig1_fidelity_tracking(data: Data, out: Path, drift_minutes: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.4))
    plotted = []
    for mode in _present(data, data.per_mode):
        sub = data.per_mode[data.per_mode["mode"] == mode].sort_values("window")
        x = sub["window"].to_numpy() * data.window_minutes
        st = STYLE[mode]
        ax.plot(x, sub["F_frozen"], color=st["color"], ls=st["ls"], lw=1.2 if mode == "full_system" else 0.9,
                label=LABEL[mode] + (r" (mean $\pm1\sigma$, 3 runs)" if mode == "full_system" and sub["n_runs"].max() > 1 else ""))
        if mode == "full_system" and sub["n_runs"].max() > 1:
            sd = sub["F_frozen_sd"].fillna(0).to_numpy()
            ax.fill_between(x, sub["F_frozen"] - sd, sub["F_frozen"] + sd, color=st["color"], alpha=0.18, lw=0)
        plotted.append(sub.assign(minute=x))

    att = data.attempts
    if len(att):
        prop = att[(att["mode"] == "full_system")]
        trig = prop[(prop["attempt"] == 1) & (prop["trigger_type"] == "fidelity_degradation")]["window"].dropna().unique()
        acc = prop[prop["gate_result"] == "ACCEPT"]["window"].dropna().unique()
        for i, wv in enumerate(sorted(trig)):
            ax.plot(wv * data.window_minutes, 0.02, marker="v", color="#333333", ms=3.5, ls="none", label="Trigger fired (Proposed)" if i == 0 else None, clip_on=False)
        for i, wv in enumerate(sorted(acc)):
            ax.axvline(wv * data.window_minutes, color="#0072B2", ls="--", lw=0.7, alpha=0.8, label="Verified repair deployed (Proposed)" if i == 0 else None)
    for i, m in enumerate(drift_minutes):
        ax.axvline(m, color="black", ls=(0, (1, 1)), lw=0.9, label="Known drift event" if i == 0 else None)

    ax.set_xlabel("Elapsed time in OAI run 8 (min)")
    ax.set_ylabel("Composite fidelity score")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.2)
    save(fig, out, "fig1_fidelity_tracking", data, pd.concat(plotted, ignore_index=True) if plotted else pd.DataFrame())


def fig2_metric_breakdown(data: Data, out: Path) -> None:
    modes = _present(data, data.per_mode)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.3), gridspec_kw={"width_ratios": [4, 1.3]})
    n, width = len(modes), 0.8 / max(len(modes), 1)
    rows = []
    for i, mode in enumerate(modes):
        sub = data.per_mode[data.per_mode["mode"] == mode]
        means = [sub[f"norm_{m}"].mean() for m in METRICS]
        sds = [sub[f"norm_{m}"].std(ddof=1) for m in METRICS]
        xs = np.arange(len(METRICS)) + (i - (n - 1) / 2) * width
        st = STYLE[mode]
        ax_a.bar(xs, means, width, yerr=sds, color=st["color"], hatch=st["hatch"], edgecolor="black", linewidth=0.4,
                 error_kw={"elinewidth": 0.6, "capsize": 1.5}, label=LABEL[mode])
        f_mean, f_sd = sub["F_frozen"].mean(), sub["F_frozen"].std(ddof=1)
        ax_b.bar(i, f_mean, 0.7, yerr=f_sd, color=st["color"], hatch=st["hatch"], edgecolor="black", linewidth=0.4, error_kw={"elinewidth": 0.6, "capsize": 1.5})
        rows += [{"mode": mode, "metric": m, "mean": mu, "sd_windows": s} for m, mu, s in zip(METRICS, means, sds, strict=True)]
        rows.append({"mode": mode, "metric": "composite_fidelity", "mean": f_mean, "sd_windows": f_sd})
    ax_a.set_xticks(np.arange(len(METRICS)), [METRIC_LABEL[m] for m in METRICS])
    ax_a.set_ylabel(r"Normalized squared error $\tilde{D}_m$ (lower is better)")
    ax_a.set_ylim(bottom=0)
    ax_b.set_xticks(range(len(modes)), [SHORT[m] for m in modes], rotation=35, ha="right")
    ax_b.set_ylabel("Composite fidelity (higher is better)")
    ax_b.set_ylim(0, 1.05)
    _panel_label(ax_a, "(a)")
    _panel_label(ax_b, "(b)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(*ax_a.get_legend_handles_labels(), ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    save(fig, out, "fig2_metric_breakdown", data, pd.DataFrame(rows))


def fig3_strategy_distribution(data: Data, out: Path) -> None:
    att = data.attempts[data.attempts["mode"].isin(["rule_based", "full_system"])].copy() if len(data.attempts) else pd.DataFrame()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.4), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    rows = []
    for ax, mode, tag in ((axes[0], "rule_based", "(a)"), (axes[1], "full_system", "(b)")):
        sub = att[att["mode"] == mode] if len(att) else pd.DataFrame()
        n_runs = max(sub["seed"].nunique(), 1) if len(sub) else 1
        bottom = np.zeros(len(SEVERITY_BINS))
        for strat in STRATEGIES:
            counts = np.array([((sub["severity_bin"] == b) & (sub["strategy"] == strat)).sum() / n_runs if len(sub) else 0 for b in SEVERITY_BINS])
            ss = STRAT_STYLE[strat]
            ax.bar(range(len(SEVERITY_BINS)), counts, 0.65, bottom=bottom, color=ss["color"], hatch=ss["hatch"], edgecolor="black", linewidth=0.4, label=strat.replace("_", " ").capitalize())
            bottom += counts
            rows += [{"mode": mode, "severity_bin": b, "strategy": strat, "attempts_per_run": c} for b, c in zip(SEVERITY_BINS, counts, strict=True)]
        for j, b in enumerate(SEVERITY_BINS):
            bsub = sub[sub["severity_bin"] == b] if len(sub) else pd.DataFrame()
            if len(bsub):
                acc = (bsub["gate_result"] == "ACCEPT").sum()
                ax.text(j, bottom[j], f"{acc}/{len(bsub)}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(range(len(SEVERITY_BINS)), [s.capitalize() for s in SEVERITY_BINS])
        ax.set_xlabel("Drift severity at trigger")
        ax.set_title(f"{tag} {SHORT[mode]}" + (f" (mean of {n_runs} runs)" if n_runs > 1 else ""), loc="left")
        ax.set_ylim(0, max(bottom.max() * 1.25, 1))
    axes[0].set_ylabel("Repair attempts per run")

    ax = axes[2]
    width = 0.38
    for i, mode in enumerate(("rule_based", "full_system")):
        sub = att[att["mode"] == mode] if len(att) else pd.DataFrame()
        rates, ns = [], []
        for strat in STRATEGIES:
            s = sub[sub["strategy"] == strat] if len(sub) else pd.DataFrame()
            ns.append(len(s))
            rates.append(100.0 * (s["gate_result"] == "ACCEPT").mean() if len(s) else 0.0)
        xs = np.arange(len(STRATEGIES)) + (i - 0.5) * width
        st = STYLE[mode]
        ax.bar(xs, rates, width, color=st["color"], hatch=st["hatch"], edgecolor="black", linewidth=0.4, label=SHORT[mode])
        for x, r, n in zip(xs, rates, ns, strict=True):
            ax.text(x, r + 1, f"n={n}", ha="center", va="bottom", fontsize=5.5, rotation=90)
        rows += [{"mode": mode, "strategy": s, "gate_pass_rate_pct": r, "n_attempts": n} for s, r, n in zip(STRATEGIES, rates, ns, strict=True)]
    ax.set_xticks(range(len(STRATEGIES)), ["Recal.", "Regen.", "Expand"])
    ax.set_ylabel("Verification-gate pass rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("(c) Gate pass rate by strategy", loc="left")
    ax.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, loc="upper center", bbox_to_anchor=(0.36, 1.0), title="Selected strategy (bar label: accepted/attempts)")
    save(fig, out, "fig3_strategy_distribution", data, pd.DataFrame(rows))


def fig4_cost_tradeoff(data: Data, out: Path) -> None:
    s = data.summary.set_index("mode")
    modes = [m for m in MODES if m in s.index]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))
    cost = np.array([s.loc[m, "adaptation_cpu_seconds"] / 60.0 for m in modes])
    fid = np.array([s.loc[m, "mean_F_frozen"] for m in modes])
    lo = np.array([s.loc[m, "ci95_low"] for m in modes])
    hi = np.array([s.loc[m, "ci95_high"] for m in modes])
    for i, m in enumerate(modes):
        st = STYLE[m]
        ax.errorbar(cost[i], fid[i], yerr=[[fid[i] - lo[i]], [hi[i] - fid[i]]], fmt=st["marker"], color=st["color"],
                    ms=5, mec="black", mew=0.4, elinewidth=0.7, capsize=2, label=LABEL[m])
        tok = s.loc[m, "llm_tokens"] if "llm_tokens" in s.columns else 0
        if tok and tok > 0:
            # Name the mode in the label: near-coincident points otherwise read as swapped.
            ax.annotate(f"{SHORT[m]}: {tok / 1000:.0f}k LLM tok", (cost[i], fid[i]), xytext=(8, 9 if i % 2 else -13),
                        textcoords="offset points", fontsize=6, color=st["color"],
                        arrowprops={"arrowstyle": "-", "lw": 0.4, "color": st["color"]})
    # Pareto frontier (minimize cost, maximize fidelity) — computed, never assumed.
    order = np.argsort(cost)
    frontier, best = [], -np.inf
    for i in order:
        if fid[i] > best:
            frontier.append(i)
            best = fid[i]
    ax.step(cost[frontier], fid[frontier], where="post", color="black", lw=0.6, ls="--", alpha=0.7, label="Pareto frontier")
    ax.set_xlabel("Adaptation compute (local CPU-min, per run)")
    ax.set_ylabel("Mean windowed fidelity (95% CI)")
    ax.legend(fontsize=6.5, loc="best")
    save(fig, out, "fig4_cost_tradeoff", data, pd.DataFrame({
        "mode": modes, "cpu_minutes": cost, "mean_fidelity": fid, "ci95_low": lo, "ci95_high": hi,
        "llm_tokens": [s.loc[m, "llm_tokens"] for m in modes], "on_pareto_frontier": [i in frontier for i in range(len(modes))],
    }))


def fig5_recovery_and_gate(data: Data, out: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4))
    rows = []
    rec = data.recovery
    for mode in [m for m in ("static", "rule_based", "full_system") if len(rec) and m in set(rec["mode"])]:
        sub = rec[rec["mode"] == mode]
        ok = np.sort(sub.loc[sub["recovered"].astype(bool), "latency_min"].to_numpy())
        n, censored = len(sub), int((~sub["recovered"].astype(bool)).sum())
        st = STYLE[mode]
        if len(ok):
            y = np.arange(1, len(ok) + 1) / n  # censored triggers keep the ECDF below 1
            ax_a.step(np.r_[0, ok], np.r_[0, y], where="post", color=st["color"], ls=st["ls"], lw=1.1,
                      label=f"{SHORT[mode]} (n={n}, {censored} not recovered)")
        rows += [{"panel": "a", "mode": mode, "latency_min": v} for v in ok]
    ax_a.set_xlabel("Time to recovery, trigger to F$_c\\geq$ threshold (data-min)")
    ax_a.set_ylabel("Fraction of triggers recovered")
    ax_a.set_ylim(0, 1.02)
    ax_a.set_xlim(left=0)
    ax_a.legend(fontsize=6.5, loc="lower right")
    _panel_label(ax_a, "(a)")

    cats = (("ACCEPT", "Accepted", "#009E73", ""), ("REJ_REGRESS", "Rejected: would degrade fidelity", "#D55E00", "//"),
            ("REJ_SMALL", "Rejected: gain < $\\delta$", "#E69F00", ".."), ("NO_CAND", "No valid candidate", "#999999", "xx"),
            ("UNGATED", "Deployed unverified", "#56B4E9", "\\\\"))
    att = data.attempts
    modes = [m for m in MODES if m != "static"]
    bottom = np.zeros(len(modes))
    for key, label, color, hatch in cats:
        vals = []
        for m in modes:
            sub = att[att["mode"] == m] if len(att) else pd.DataFrame()
            runs = max(sub["seed"].nunique(), 1) if len(sub) else 1
            if not len(sub):
                vals.append(0.0)
                continue
            rej = sub["gate_result"] == "REJECT"
            regress = rej & (sub["fidelity_after"] <= sub["fidelity_before"])
            sel = {"ACCEPT": sub["gate_result"] == "ACCEPT", "REJ_REGRESS": regress, "REJ_SMALL": rej & ~regress,
                   "NO_CAND": sub["error"].astype(bool), "UNGATED": sub["gate_result"] == "DEPLOYED_UNGATED"}[key]
            vals.append(sel.sum() / runs)
        vals = np.array(vals)
        ax_b.bar(range(len(modes)), vals, 0.6, bottom=bottom, color=color, hatch=hatch, edgecolor="black", linewidth=0.4, label=label)
        bottom += vals
        rows += [{"panel": "b", "mode": m, "category": key, "per_run": v} for m, v in zip(modes, vals, strict=True)]
    ax_b.set_xticks(range(len(modes)), [SHORT[m] for m in modes])
    ax_b.set_ylabel("Candidate models per run")
    ax_b.legend(fontsize=6.2, loc="upper right", ncol=1)
    ax_b.set_ylim(0, max(bottom.max() * 1.1, 1))
    ax_b.set_xlim(-0.5, len(modes) + 0.9)  # room on the right for the legend
    ax_b.set_xlabel("(Static never adapts: no candidates)", style="italic", fontsize=6.5)
    _panel_label(ax_b, "(b)")
    fig.tight_layout()
    save(fig, out, "fig5_recovery_and_gate", data, pd.DataFrame(rows))


def fig6_significance(data: Data, out: Path) -> None:
    p = data.pairs
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 1.9))
    if not len(p):
        ax.text(0.5, 0.5, "No pairwise results", ha="center", va="center", transform=ax.transAxes)
        save(fig, out, "fig6_significance", data, pd.DataFrame())
        return
    p = p.iloc[::-1].reset_index(drop=True)
    for i, r in p.iterrows():
        sig = r["p_holm"] < 0.05 if not pd.isna(r["p_holm"]) else False
        ax.errorbar(r["mean_diff"], i, xerr=[[r["mean_diff"] - r["ci95_low"]], [r["ci95_high"] - r["mean_diff"]]], fmt="o",
                    color="black", mfc="black" if sig else "white", ms=4, elinewidth=0.8, capsize=2)
        ptxt = "n/a" if pd.isna(r["p_holm"]) else (f"{r['p_holm']:.3f}" if r["p_holm"] >= 0.001 else "<0.001")
        ax.text(1.02, i, f"$r$={r['r_rb']:+.2f}, $p_H$={ptxt}", transform=ax.get_yaxis_transform(), va="center", fontsize=6.5)
    ax.axvline(0, color="#777777", lw=0.7, ls="--")
    ax.set_yticks(range(len(p)), [f"{SHORT[a]} vs {SHORT[b]}" for a, b in zip(p["mode_a"], p["mode_b"], strict=True)])
    ax.set_xlabel("Mean paired difference in windowed fidelity\n(95% block-bootstrap CI)")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig6_significance", data, p)


def fig7_kpi_tracking(data: Data, out: Path) -> None:
    panels = (("dl_throughput", "(a) DL throughput (Mbps)", 1.0), ("dl_rlc_delay", "(b) DL RLC SDU delay (ms)", 1.0),
              ("dl_bler", "(c) DL BLER (%)", 100.0))
    fig, axes = plt.subplots(3, 1, figsize=(DOUBLE_COL, 4.3), sharex=True)
    rows = []
    kpi = data.kpi
    for ax, (comp, title, scale) in zip(axes, panels, strict=True):
        sub = kpi[kpi["component"] == comp] if len(kpi) else pd.DataFrame()
        if not len(sub):
            ax.text(0.5, 0.5, f"{comp}: no data", transform=ax.transAxes, ha="center")
            continue
        truth = sub[sub["mode"] == sub["mode"].iloc[0]].groupby("window")["actual"].mean() * scale
        x = truth.index.to_numpy() * data.window_minutes
        ax.plot(x, truth.to_numpy(), color="black", lw=1.2, label="Ground truth (OAI)")
        for mode in ("static", "full_system"):
            ms = sub[sub["mode"] == mode]
            if not len(ms):
                continue
            per_seed = ms.pivot_table(index="window", columns="seed", values="predicted") * scale
            mean = per_seed.mean(axis=1)
            st = STYLE[mode]
            ax.plot(mean.index * data.window_minutes, mean.to_numpy(), color=st["color"], ls=st["ls"], lw=1.0, label=f"{SHORT[mode]} twin")
            if per_seed.shape[1] > 1:
                ax.fill_between(mean.index * data.window_minutes, per_seed.min(axis=1), per_seed.max(axis=1), color=st["color"], alpha=0.15, lw=0)
            rows += [{"component": comp, "mode": mode, "window": wv, "predicted": v} for wv, v in mean.items()]
        rows += [{"component": comp, "mode": "ground_truth", "window": wv, "predicted": v} for wv, v in truth.items()]
        ax.set_title(title, loc="left")
        ax.set_ylim(bottom=0)
    axes[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.12))
    axes[-1].set_xlabel("Elapsed time in OAI run 8 (min); 60 s window means")
    fig.tight_layout()
    save(fig, out, "fig7_kpi_tracking", data, pd.DataFrame(rows))


# ----------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPO / "figures")
    parser.add_argument("--synthetic", action="store_true", help="force placeholder data (watermarked)")
    parser.add_argument("--drift-minutes", type=float, nargs="*", default=[],
                        help="minutes of independently KNOWN drift events to mark in Fig. 1 (none are inferred)")
    args = parser.parse_args()

    set_ieee_style()
    data = None if args.synthetic else load_measured(args.stats_dir)
    if data is None:
        print("!!! Using SYNTHETIC placeholder data — figures are watermarked and must not be published.")
        if not args.synthetic:
            print(f"    (no complete stats in {args.stats_dir}; run baselines/run_all.py then baselines/stats.py)")
        data = make_synthetic()
    else:
        ref = args.stats_dir / "frozen_reference.json"
        if ref.exists():
            print(f"Using MEASURED results from {args.stats_dir} (frozen reference: {len(json.loads(ref.read_text())['reference'])} components)")

    print(f"Writing figures to {args.out_dir}")
    fig1_fidelity_tracking(data, args.out_dir, args.drift_minutes)
    fig2_metric_breakdown(data, args.out_dir)
    fig3_strategy_distribution(data, args.out_dir)
    fig4_cost_tradeoff(data, args.out_dir)
    fig5_recovery_and_gate(data, args.out_dir)
    fig6_significance(data, args.out_dir)
    fig7_kpi_tracking(data, args.out_dir)


if __name__ == "__main__":
    main()
