"""Phase13 §5/§6 — shared output writers used by `run_all.py` (per-mode) and `compare.py`
(comparison-only, reading already-written per-mode outputs, never re-running anything). Every
figure: matplotlib only, one figure per file, PNG 300dpi + PDF, titled/labeled axes with units,
legended — and a CSV of exactly the plotted data saved alongside every graph (phase13 §5).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TIMELINE_COLUMNS = ["ts", "cycle", "component", "actual", "predicted", "abs_error", "F_c", "F_U", "trigger_fired", "model_version"]

# Display-unit conversion (phase13 §0: "pick ONE display unit per KPI ... convert in one place,
# state units everywhere"). Raw OAI columns are kbps (throughput) / microseconds (RLC delay) /
# unitless ratios (BLER, PRB) — see docs/oai_twin.md. This is the ONLY place a raw stored value
# is converted for display; metrics_timeline.csv itself stores RAW units (so it stays consistent
# with everything else in this codebase that reads D1/model predictions directly) and graphs
# apply this conversion at plot time only.
DISPLAY_UNITS: dict[str, tuple[float, str]] = {
    "dl_throughput": (1.0 / 1000.0, "Mbps"),  # stored kbps -> displayed Mbps
    "ul_throughput": (1.0 / 1000.0, "Mbps"),
    "dl_rlc_delay": (1.0 / 1000.0, "ms"),  # stored microseconds -> displayed ms
    "dl_prb": (1.0, "ratio"),
    "ul_prb": (1.0, "ratio"),
    "dl_bler": (1.0, "ratio"),
    "ul_bler": (1.0, "ratio"),
}


def display_unit(component: str) -> tuple[float, str]:
    return DISPLAY_UNITS.get(component, (1.0, "raw"))


def write_metrics_timeline_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TIMELINE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in TIMELINE_COLUMNS})


def _save_plotted_csv(path_png: Path, df: pd.DataFrame) -> None:
    df.to_csv(path_png.with_suffix(".csv"), index=False)


def plot_actual_vs_predicted(component: str, timeline_df: pd.DataFrame, out_dir: Path) -> Path:
    scale, unit = display_unit(component)
    sub = timeline_df[timeline_df["component"] == component].sort_values("cycle")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sub["cycle"], sub["actual"] * scale, label="actual", color="tab:blue")
    ax.plot(sub["cycle"], sub["predicted"] * scale, label="predicted", color="tab:orange", linestyle="--")
    ax.set_xlabel("cycle (1 per replayed second, run 8)")
    ax.set_ylabel(f"{component} ({unit})")
    ax.set_title(f"Actual vs predicted — {component}")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f"actual_vs_predicted_{component}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    _save_plotted_csv(out_path, sub[["cycle", "actual", "predicted"]])
    return out_path


def plot_fidelity_component(component: str, timeline_df: pd.DataFrame, events: list[dict[str, Any]], threshold: float, out_dir: Path) -> Path:
    sub = timeline_df[timeline_df["component"] == component].sort_values("cycle")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sub["cycle"], sub["F_c"], label="F_c", color="tab:green")
    ax.axhline(threshold, color="gray", linestyle=":", label=f"threshold ({threshold})")
    trig = sub[sub["trigger_fired"].astype(bool)]
    if len(trig) > 0:
        ax.scatter(trig["cycle"], trig["F_c"], color="black", marker="x", s=60, label="trigger", zorder=5)
    for ev in events:
        if ev.get("component") != component:
            continue
        color = "green" if ev.get("gate_result") == "ACCEPT" else ("red" if ev.get("gate_result") == "REJECT" else "gray")
        ax.axvline(ev.get("cycle", -1), color=color, linestyle="--", alpha=0.6)
    ax.set_xlabel("cycle (1 per replayed second, run 8)")
    ax.set_ylabel("FidelityScore_c (dimensionless, 0-1)")
    ax.set_title(f"Fidelity over time — {component}")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f"fidelity_{component}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    _save_plotted_csv(out_path, sub[["cycle", "F_c", "trigger_fired"]])
    return out_path


def plot_unified_fidelity(timeline_df: pd.DataFrame, events: list[dict[str, Any]], threshold: float, out_dir: Path) -> Path:
    sub = timeline_df.drop_duplicates(subset=["cycle"]).sort_values("cycle")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sub["cycle"], sub["F_U"], label="F_U (unified fidelity)", color="tab:purple")
    ax.axhline(threshold, color="gray", linestyle=":", label=f"threshold ({threshold})")
    for ev in events:
        color = "green" if ev.get("gate_result") == "ACCEPT" else ("red" if ev.get("gate_result") == "REJECT" else "gray")
        ax.axvline(ev.get("cycle", -1), color=color, linestyle="--", alpha=0.6)
    ax.set_xlabel("cycle (1 per replayed second, run 8)")
    ax.set_ylabel("UnifiedFidelityScore (dimensionless, 0-1)")
    ax.set_title("Unified fidelity over time")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "unified_fidelity.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    _save_plotted_csv(out_path, sub[["cycle", "F_U"]])
    return out_path


def summarize_components(timeline_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for component, sub in timeline_df.groupby("component"):
        err = (sub["actual"] - sub["predicted"]).to_numpy()
        rmse = float((err**2).mean() ** 0.5) if len(err) else float("nan")
        mae = float(abs(err).mean()) if len(err) else float("nan")
        rows.append(
            {
                "component": component,
                "mean_actual": float(sub["actual"].mean()),
                "mean_predicted": float(sub["predicted"].mean()),
                "rmse": rmse,
                "mae": mae,
                "mean_F_c": float(sub["F_c"].dropna().mean()) if sub["F_c"].notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def write_report_md(
    path: Path,
    *,
    mode: str,
    run_ids: list[int],
    seeds: list[int],
    model_versions: dict[str, str],
    fidelity_config: dict[str, Any],
    timeline_df: pd.DataFrame,
    events: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append(f"# Baseline run report — mode `{mode}`")
    lines.append("")
    lines.append(f"- Replayed: OAI run(s) {run_ids} (TEST split, speed=0)")
    lines.append(f"- Seeds: {seeds}")
    lines.append(f"- Model versions at start: {model_versions}")
    lines.append(
        f"- Fidelity config: threshold={fidelity_config.get('trigger_threshold')}, "
        f"min_consecutive={fidelity_config.get('trigger_min_consecutive_evaluations')}, "
        f"rolling_window_length={fidelity_config.get('rolling_window_length')}"
    )
    lines.append("")
    lines.append("## Per-component summary")
    lines.append("")
    summary = summarize_components(timeline_df)
    lines.append("| component | mean actual | mean predicted | RMSE | MAE | mean F_c |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['component']} | {row['mean_actual']:.4f} | {row['mean_predicted']:.4f} | "
            f"{row['rmse']:.4f} | {row['mae']:.4f} | {row['mean_F_c']:.4f} |"
        )
    lines.append("")
    lines.append("## Adaptation Events")
    lines.append("")
    if not events:
        lines.append("_No adaptation events occurred during this run._")
    for i, ev in enumerate(events, start=1):
        ts = ev.get("trigger_time", "unknown")
        lines.append(f"### Event {i} — {ev.get('component')} — {ts}")
        lines.append("")
        lines.append(f"- **Trigger**: type={ev.get('trigger_type')}, severity={ev.get('severity')}, fidelity_at_trigger={ev.get('fidelity_at_trigger')}")
        decision = ev.get("decision", {})
        lines.append(f"- **Decision**: strategy={decision.get('strategy')}, source={decision.get('source')}, confidence={decision.get('confidence')}, llm_or_fallback={decision.get('llm_or_fallback')}")
        lines.append(f"- **Root cause**: {ev.get('root_cause_analysis')}")
        lines.append(f"- **Reasoning**: {ev.get('rationale')}")
        lines.append(f"- **References**: {ev.get('knowledge_refs')}")
        lines.append(f"- **Candidate**: version={ev.get('candidate_version')}")
        lines.append(
            f"- **Verification**: {ev.get('fidelity_before')} -> {ev.get('fidelity_after')} => "
            f"{ev.get('gate_result')} ({ev.get('gate_reason')})"
        )
        lines.append(f"- version {ev.get('version_before')} -> {ev.get('version_after')}, duration={ev.get('duration_seconds', 0):.2f}s")
        if ev.get("error"):
            lines.append(f"- **Error**: {ev.get('error')}")
        if ev.get("skipped_reason"):
            lines.append(f"- **Skipped**: {ev.get('skipped_reason')}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
