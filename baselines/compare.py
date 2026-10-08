#!/usr/bin/env python3
"""baselines/compare.py — phase13 §4 comparison tables/graphs, built ONLY from what
`run_all.py` already wrote under `baselines/results/<mode>/` — never re-runs anything.

Writes `baselines/results/comparison/comparison.md`, `comparison.csv`, and the §5 comparison
graphs (items 4-8: unified_fidelity_all_modes, mean_fidelity_by_mode, recovery_time_by_mode,
adaptations_by_mode, strategy_distribution).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines._lib import MODES  # noqa: E402

BASELINES_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASELINES_DIR / "results"
COMPARISON_DIR = RESULTS_DIR / "comparison"


def _mode_run_dirs(mode: str) -> list[Path]:
    base = RESULTS_DIR / mode
    if not base.is_dir():
        return []
    seed_dirs = sorted(base.glob("seed_*"))
    if seed_dirs:
        return seed_dirs
    if (base / "run_summary.json").exists():
        return [base]
    return []


def _load_run(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    timeline_path = run_dir / "metrics_timeline.csv"
    timeline = pd.read_csv(timeline_path) if timeline_path.exists() else pd.DataFrame()
    events = []
    events_path = run_dir / "adaptation_events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    return {"summary": summary, "timeline": timeline, "events": events, "run_dir": run_dir}


def _recovery_time(timeline: pd.DataFrame, threshold: float) -> float | None:
    """Mean number of cycles from a below-threshold unified-fidelity cycle back above threshold
    — None if the run never dipped below threshold (nothing to recover from), honestly, not 0."""
    if len(timeline) == 0 or "F_U" not in timeline.columns:
        return None
    unified = timeline.drop_duplicates("cycle").sort_values("cycle")[["cycle", "F_U"]].dropna()
    if len(unified) == 0:
        return None
    below = unified["F_U"] < threshold
    recoveries: list[int] = []
    dip_start: int | None = None
    for cycle, is_below in zip(unified["cycle"], below, strict=True):
        if is_below and dip_start is None:
            dip_start = cycle
        elif not is_below and dip_start is not None:
            recoveries.append(cycle - dip_start)
            dip_start = None
    if not recoveries:
        return None
    return float(sum(recoveries) / len(recoveries))


def build_comparison(threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for mode in MODES:
        run_dirs = _mode_run_dirs(mode)
        if not run_dirs:
            continue
        loaded = [r for r in (_load_run(d) for d in run_dirs) if r is not None]
        if not loaded:
            continue

        mean_fus = [r["summary"].get("mean_unified_fidelity") for r in loaded if r["summary"].get("mean_unified_fidelity") is not None]
        n_triggers = [r["summary"].get("n_triggers", 0) for r in loaded]
        n_accepted = [r["summary"].get("n_accepted", 0) for r in loaded]
        n_rejected = [r["summary"].get("n_rejected", 0) for r in loaded]
        durations = [r["summary"].get("total_duration_seconds", 0.0) for r in loaded]
        recov = [_recovery_time(r["timeline"], threshold) for r in loaded]
        recov_defined = [r for r in recov if r is not None]
        n_fallbacks = [r["summary"].get("n_fallbacks", 0) for r in loaded]
        n_decisions = [r["summary"].get("n_decisions", 0) for r in loaded]
        any_invalid = any(r["summary"].get("invalid") for r in loaded)
        budget_spent = [r["summary"].get("budget_spent_usd", 0.0) for r in loaded]

        pct_above_threshold = None
        all_timelines = pd.concat([r["timeline"] for r in loaded if len(r["timeline"])], ignore_index=True) if any(len(r["timeline"]) for r in loaded) else pd.DataFrame()
        if len(all_timelines) and "F_U" in all_timelines.columns:
            fu = all_timelines.drop_duplicates(["cycle"])["F_U"].dropna()
            if len(fu):
                pct_above_threshold = float((fu >= threshold).mean() * 100.0)

        # LLM tokens/cost: aggregate llm_calls.jsonl across this mode's run(s), if any exist.
        total_input_tokens = total_output_tokens = total_cached_tokens = 0
        total_cost = 0.0
        total_llm_seconds = 0.0
        for r in loaded:
            calls_path = r["run_dir"] / "llm_calls.jsonl"
            if not calls_path.exists():
                continue
            for line in calls_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                total_input_tokens += rec.get("input_tokens", 0)
                total_output_tokens += rec.get("output_tokens", 0)
                total_cached_tokens += rec.get("cached_tokens", 0)
                total_cost += rec.get("cost_usd", 0.0)
                total_llm_seconds += rec.get("latency_seconds", 0.0)
        cache_hit_share = (total_cached_tokens / total_input_tokens) if total_input_tokens else 0.0

        rows.append(
            {
                "mode": mode, "n_runs": len(loaded),
                "mean_unified_fidelity": (sum(mean_fus) / len(mean_fus)) if mean_fus else None,
                "mean_unified_fidelity_std": (pd.Series(mean_fus).std() if len(mean_fus) > 1 else 0.0),
                "pct_time_above_threshold": pct_above_threshold,
                "mean_recovery_time_cycles": (sum(recov_defined) / len(recov_defined)) if recov_defined else None,
                "n_triggers": sum(n_triggers), "n_adaptations": sum(n_accepted) + sum(n_rejected),
                "n_accepted": sum(n_accepted), "n_rejected": sum(n_rejected),
                "total_adaptation_duration_seconds": sum(durations),
                "n_llm_decisions": sum(n_decisions), "n_llm_fallbacks": sum(n_fallbacks),
                "fallback_rate": (sum(n_fallbacks) / sum(n_decisions)) if sum(n_decisions) else 0.0,
                "run_invalid_fallback_rate_too_high": any_invalid,
                "llm_input_tokens": total_input_tokens, "llm_output_tokens": total_output_tokens,
                "llm_cached_tokens": total_cached_tokens, "llm_cache_hit_share": cache_hit_share,
                "llm_cost_usd": total_cost, "llm_time_seconds": total_llm_seconds,
                "budget_spent_usd": sum(budget_spent),
            }
        )
    return pd.DataFrame(rows)


def _plot_unified_fidelity_all_modes(out_dir: Path, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    plotted_any = False
    plot_rows = []
    for mode in MODES:
        run_dirs = _mode_run_dirs(mode)
        if not run_dirs:
            continue
        loaded = [r for r in (_load_run(d) for d in run_dirs) if r is not None and len(r["timeline"])]
        if not loaded:
            continue
        timeline = loaded[0]["timeline"].drop_duplicates("cycle").sort_values("cycle")
        if "F_U" not in timeline.columns or timeline["F_U"].dropna().empty:
            continue
        ax.plot(timeline["cycle"], timeline["F_U"], label=mode, marker="o", markersize=3)
        plotted_any = True
        for _, row in timeline[["cycle", "F_U"]].iterrows():
            plot_rows.append({"mode": mode, "cycle": row["cycle"], "F_U": row["F_U"]})
    ax.axhline(threshold, color="gray", linestyle=":", label=f"threshold ({threshold})")
    ax.set_xlabel("cycle")
    ax.set_ylabel("UnifiedFidelityScore (dimensionless, 0-1)")
    ax.set_title("Unified fidelity over time — all modes")
    if plotted_any:
        ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "unified_fidelity_all_modes.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    pd.DataFrame(plot_rows).to_csv(out_path.with_suffix(".csv"), index=False)


def _plot_mean_fidelity_by_mode(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = df[df["mean_unified_fidelity"].notna()]
    ax.bar(valid["mode"], valid["mean_unified_fidelity"], yerr=valid["mean_unified_fidelity_std"], capsize=5, color="tab:blue")
    ax.set_ylabel("mean UnifiedFidelityScore (dimensionless, 0-1)")
    ax.set_title("Mean unified fidelity by mode (error bars: std across seeds)")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mean_fidelity_by_mode.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    valid[["mode", "mean_unified_fidelity", "mean_unified_fidelity_std"]].to_csv(out_path.with_suffix(".csv"), index=False)


def _plot_recovery_time_by_mode(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = df[df["mean_recovery_time_cycles"].notna()]
    ax.bar(valid["mode"], valid["mean_recovery_time_cycles"], color="tab:orange")
    ax.set_ylabel("mean recovery time (cycles)")
    ax.set_title("Mean recovery time to above-threshold fidelity, by mode")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "recovery_time_by_mode.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    valid[["mode", "mean_recovery_time_cycles"]].to_csv(out_path.with_suffix(".csv"), index=False)


def _plot_adaptations_by_mode(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["mode"], df["n_accepted"], label="accepted", color="tab:green")
    ax.bar(df["mode"], df["n_rejected"], bottom=df["n_accepted"], label="rejected", color="tab:red")
    ax.set_ylabel("number of adaptations")
    ax.set_title("Accepted vs rejected adaptations, by mode")
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "adaptations_by_mode.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    df[["mode", "n_accepted", "n_rejected"]].to_csv(out_path.with_suffix(".csv"), index=False)


def _plot_strategy_distribution(out_dir: Path) -> None:
    counts: dict[str, dict[str, int]] = {"rule_based": {}, "full_system": {}}
    for mode in ("rule_based", "full_system"):
        for run_dir in _mode_run_dirs(mode):
            loaded = _load_run(run_dir)
            if loaded is None:
                continue
            for ev in loaded["events"]:
                strategy = (ev.get("decision") or {}).get("strategy")
                if strategy:
                    counts[mode][strategy] = counts[mode].get(strategy, 0) + 1
    strategies = sorted({s for m in counts.values() for s in m})
    if not strategies:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    x = range(len(strategies))
    ax.bar([i - width / 2 for i in x], [counts["rule_based"].get(s, 0) for s in strategies], width, label="rule_based", color="tab:purple")
    ax.bar([i + width / 2 for i in x], [counts["full_system"].get(s, 0) for s in strategies], width, label="full_system", color="tab:cyan")
    ax.set_xticks(list(x), strategies)
    ax.set_ylabel("count")
    ax.set_title("Strategy distribution — rule_based vs full_system")
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "strategy_distribution.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    pd.DataFrame([{"strategy": s, "rule_based": counts["rule_based"].get(s, 0), "full_system": counts["full_system"].get(s, 0)} for s in strategies]).to_csv(
        out_path.with_suffix(".csv"), index=False
    )


def write_comparison_md(df: pd.DataFrame, path: Path) -> None:
    lines = ["# Baseline comparison", "", "Built only from already-written `baselines/results/<mode>/` outputs — never re-run.", ""]
    try:
        lines.append(df.to_markdown(index=False))
    except ImportError:  # `tabulate` not installed — fall back to a plain aligned table, never crash
        lines.append("```\n" + df.to_string(index=False) + "\n```")
    invalid = df[df["run_invalid_fallback_rate_too_high"]]
    if len(invalid):
        lines.append("")
        lines.append("## Invalid runs")
        for _, row in invalid.iterrows():
            lines.append(f"- **{row['mode']}**: fallback_rate={row['fallback_rate']:.3f} exceeds the configured threshold — flagged invalid per phase13 §4.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    threshold = 0.5
    try:
        import yaml

        threshold = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "settings.yaml").read_text())["fidelity"]["trigger_threshold"]
    except Exception:  # noqa: BLE001 - fall back to the documented default, never crash comparison on a config-read issue
        pass

    df = build_comparison(threshold)
    if df.empty:
        print("No mode results found under baselines/results/ — run run_all.py first.")
        return

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMPARISON_DIR / "comparison.csv", index=False)
    write_comparison_md(df, COMPARISON_DIR / "comparison.md")

    graphs_dir = COMPARISON_DIR / "graphs"
    _plot_unified_fidelity_all_modes(graphs_dir, threshold)
    _plot_mean_fidelity_by_mode(df, graphs_dir)
    _plot_recovery_time_by_mode(df, graphs_dir)
    _plot_adaptations_by_mode(df, graphs_dir)
    _plot_strategy_distribution(graphs_dir)

    print(df.to_string(index=False))
    print(f"\nWrote {COMPARISON_DIR / 'comparison.md'}, {COMPARISON_DIR / 'comparison.csv'}, and graphs under {graphs_dir}")


if __name__ == "__main__":
    main()
