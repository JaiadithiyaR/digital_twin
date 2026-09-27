#!/usr/bin/env python
"""Standalone visualization program for real telemetry metrics and DT fidelity — reads whatever
this system has ALREADY genuinely produced (D1 telemetry history, the versioned model registry's
current production DT models, Module 19's lifecycle records) and renders PNG charts. Never a new
architecture module: it is an offline analysis tool that replays already-recorded/production
state through the SAME real Module 12 (`FidelityEvaluator`) code path used at runtime, never a
re-implementation, approximation, or synthetic stand-in for it (prompt.md's own "always compute
fidelity via the deterministic Module 12 evaluator" rule applies here exactly as it does inside
`src/main.py`'s own `run_prediction_and_fidelity_cycle`).

Usage:
    # production storage (config/settings.yaml's own paths — what `python -m src.main` writes to)
    python scripts/visualize_metrics.py

    # the dedicated real-NS-3 end-to-end demo's own storage subtree
    python scripts/visualize_metrics.py --preset e2e_ns3_demo

    # fully explicit paths (mix and match with the above)
    python scripts/visualize_metrics.py --history-path data/e2e_ns3_demo/d1_history.parquet \\
        --models-dir data/e2e_ns3_demo/models --lifecycle-path data/e2e_ns3_demo/lifecycle_records.jsonl

Output: PNG charts plus CSVs under --output-dir (default data/artifacts/plots/), plus a printed
summary — including telemetry_original.csv (the real D1 history, exactly as ingested) and
telemetry_predicted.csv (the DT's own predictions for that same history, from one real
DTOrchestrator.run_predictions() pass through the current production models — dependency-chained
live, not ground-truth-fed) as two separate, directly comparable files.
Nothing here writes back into D1, the model registry, or the lifecycle log — read-only throughout.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — this is a script, not a notebook; every plot is saved to disk
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adaptation.data_selection import with_dependency_ground_truth
from src.adaptation.lifecycle_agent import LifecycleAgent, LifecycleRecord
# Display-range clamp for the (theoretically unbounded-below) Module 12 fidelity score — a
# cosmetic choice for readable charts only, never a change to the underlying data (the full
# unclipped values are always written to the companion CSVs, see save_fidelity_csvs()). Previously
# imported from src/adaptation/rl_env.py, which no longer exists (LLM/RL design pivot — see
# CLAUDE.md §12's design-pivot notice); redefined locally here since this is purely a chart
# display concern now, not shared RL-training-stability infrastructure.
_FIDELITY_SCORE_CLIP: tuple[float, float] = (-3.0, 1.0)
from src.common.config import Settings, load_settings
from src.dt_models.base import DTComponent
from src.dt_models.jitter import JitterModel
from src.dt_models.latency import LatencyModel
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator, OrchestratorError
from src.dt_models.packet_loss import PacketLossModel
from src.dt_models.prb_utilization import PrbUtilizationModel
from src.dt_models.throughput import ThroughputModel
from src.fidelity.evaluator import FidelityEvaluator
from src.registry.model_registry import ModelRegistry

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("visualize_metrics")

# Reuses the exact same dispatch tables `src/main.py`'s `ContinuousOrchestrator` runs against —
# not re-declared independently, so this tool can never silently drift from what the real system
# actually predicts/targets per component.
_DT_COMPONENT_CLASSES: dict[str, type] = {
    "throughput": ThroughputModel,
    "packet_loss": PacketLossModel,
    "latency": LatencyModel,
    "prb_utilization": PrbUtilizationModel,
    "jitter": JitterModel,
}
_TARGET_COLUMNS: dict[str, str] = {
    "throughput": "throughput_mbps",
    "packet_loss": "packet_loss_pct",
    "latency": "latency_ms",
    "prb_utilization": "prb_utilization_pct",
    "jitter": "jitter_ms",
}
_OUTPUT_FIELDS: dict[str, str] = {name: cls.OUTPUT_FIELD for name, cls in _DT_COMPONENT_CLASSES.items()}

_TELEMETRY_PANELS = [
    ("throughput_mbps", "Throughput (Mbps)"),
    ("latency_ms", "Latency (ms)"),
    ("jitter_ms", "Jitter (ms)"),
    ("packet_loss_pct", "Packet Loss (%)"),
    ("prb_utilization_pct", "PRB Utilization (%)"),
]
_RADIO_PANELS = [
    ("sinr_db", "SINR (dB)"),
    ("rsrp_dbm", "RSRP (dBm)"),
    ("rsrq_db", "RSRQ (dB)"),
    ("ue_speed_mps", "UE Speed (m/s)"),
]

_COMPONENT_COLORS = {
    "throughput": "#1f77b4",
    "packet_loss": "#d62728",
    "latency": "#2ca02c",
    "prb_utilization": "#9467bd",
    "jitter": "#ff7f0e",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--preset",
        choices=["production", "e2e_ns3_demo"],
        default="production",
        help="production = config/settings.yaml's own storage paths (what `python -m src.main` writes); "
        "e2e_ns3_demo = the dedicated data/e2e_ns3_demo/ subtree scripts/run_e2e_demo.py writes to",
    )
    p.add_argument("--config", type=Path, default=None, help="config/settings.yaml override (default: repo's own)")
    p.add_argument("--history-path", type=Path, default=None, help="override the D1 history parquet path")
    p.add_argument("--models-dir", type=Path, default=None, help="override the model registry directory")
    p.add_argument("--lifecycle-path", type=Path, default=None, help="override the lifecycle_records.jsonl path")
    p.add_argument("--output-dir", type=Path, default=None, help="default: data/artifacts/plots/")
    p.add_argument(
        "--window",
        type=int,
        default=20,
        help="rows per fidelity-recompute window (default 20 — deliberately smaller than "
        "config.evaluation.window_size, which is for a single before/after verification "
        "comparison; this tool wants many sequential points to plot a real time series)",
    )
    p.add_argument("--stride", type=int, default=None, help="rows to advance between windows (default: --window, i.e. non-overlapping)")
    p.add_argument("--show", action="store_true", help="also open an interactive window per figure (needs a display)")
    return p.parse_args(argv)


def resolve_paths(args: argparse.Namespace, settings: Settings) -> tuple[Path, Path, Path, Path, Path]:
    """Returns (history_path, models_dir, lifecycle_path, reports_dir, output_dir), applying
    --preset defaults and then any explicit --history-path/--models-dir/--lifecycle-path/
    --output-dir overrides."""
    if args.preset == "e2e_ns3_demo":
        base = settings.resolve_path("data/e2e_ns3_demo")
        history_path = base / "d1_history.parquet"
        models_dir = base / "models"
        lifecycle_path = base / "lifecycle_records.jsonl"
        reports_dir = base / "maintenance_reports"
    else:
        history_path = settings.resolve_path(settings.storage.d1_history_path)
        models_dir = settings.resolve_path(settings.storage.models_dir)
        lifecycle_path = settings.resolve_path(settings.lifecycle.records_path)
        reports_dir = settings.resolve_path(settings.lifecycle.reports_dir)
    output_dir = settings.resolve_path("data/artifacts/plots")

    if args.history_path is not None:
        history_path = args.history_path
    if args.models_dir is not None:
        models_dir = args.models_dir
    if args.lifecycle_path is not None:
        lifecycle_path = args.lifecycle_path
    if args.output_dir is not None:
        output_dir = args.output_dir
    return history_path, models_dir, lifecycle_path, reports_dir, output_dir


def load_telemetry_history(history_path: Path) -> pd.DataFrame:
    if not history_path.exists():
        raise FileNotFoundError(
            f"no D1 history found at {history_path} — run the system first "
            f"(e.g. `python scripts/run_orchestrator_demo.py` or `python scripts/run_e2e_demo.py`)"
        )
    df = pd.read_parquet(history_path)
    if df.empty:
        raise ValueError(f"{history_path} exists but has zero rows — nothing to plot")
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def load_lifecycle_records(lifecycle_path: Path, reports_dir: Path) -> list[LifecycleRecord]:
    """Reuses the real `LifecycleAgent.list_records()` (never a hand-rolled JSONL parser here) so
    this tool automatically inherits its schema-compatibility handling — a real, load-bearing
    need: this project's own lifecycle log mixes lines from before and after the LLM/RL design
    pivot's field rename, and `list_records()` already skips/logs whichever lines don't match the
    CURRENT `LifecycleRecord` schema instead of crashing on them (see its own docstring)."""
    if not lifecycle_path.exists():
        logger.warning("no lifecycle records found at %s — the lifecycle panel will be empty", lifecycle_path)
        return []
    return LifecycleAgent(records_path=lifecycle_path, reports_dir=reports_dir).list_records()


# ---------------------------------------------------------------------------
# Telemetry plots
# ---------------------------------------------------------------------------
def plot_telemetry(history: pd.DataFrame, output_dir: Path, show: bool) -> list[Path]:
    """Per-timestamp mean across UEs (the same aggregation a real cell-wide view would use) for
    the five DT-predicted quantities and four radio-condition quantities. Real recorded values —
    no smoothing, no synthetic interpolation."""
    saved: list[Path] = []
    agg_cols = [c for c, _ in _TELEMETRY_PANELS + _RADIO_PANELS if c in history.columns]
    agg = history.groupby("timestamp", as_index=False)[agg_cols].mean().sort_values("timestamp")
    source_label = history["source"].mode().iat[0] if "source" in history.columns and not history.empty else "UNKNOWN"

    # Break the line across abnormally large gaps between telemetry ticks (e.g. a real pause
    # between retried drift-event attempts in scripts/run_e2e_demo.py) instead of letting
    # matplotlib draw a straight line across the gap — that line would visually invent a smooth
    # trend across a period with NO real data, which is exactly the kind of fabrication this
    # project's discipline forbids. A gap is "abnormal" if it's >8x the median tick spacing.
    gap_seconds = agg["timestamp"].diff().dt.total_seconds()
    median_gap = gap_seconds.median()
    if pd.notna(median_gap) and median_gap > 0:
        is_break = gap_seconds > max(8 * median_gap, 1.0)
        if is_break.any():
            for col in agg_cols:
                agg.loc[is_break, col] = np.nan
            logger.warning(
                "%d abnormally large gap(s) in telemetry ticks detected (median spacing %.3fs) — "
                "line breaks inserted rather than interpolating across missing data",
                int(is_break.sum()),
                median_gap,
            )

    for panels, filename, title in [
        (_TELEMETRY_PANELS, "telemetry_core_metrics.png", "DT-Predicted Telemetry Metrics"),
        (_RADIO_PANELS, "telemetry_radio_conditions.png", "Radio Condition Metrics"),
    ]:
        present = [(c, label) for c, label in panels if c in agg.columns]
        if not present:
            continue
        fig, axes = plt.subplots(len(present), 1, figsize=(11, 2.4 * len(present)), sharex=True)
        if len(present) == 1:
            axes = [axes]
        for ax, (col, label) in zip(axes, present):
            ax.plot(agg["timestamp"], agg[col], linewidth=1.1, color="#1f77b4")
            ax.set_ylabel(label, fontsize=9)
            ax.grid(alpha=0.3)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate()
        fig.suptitle(f"{title}  (source={source_label}, n={len(history)} rows, {agg['timestamp'].nunique()} ticks)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=140)
        saved.append(out_path)
        if show:
            plt.show(block=False)
        else:
            plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# Shared production-model loading — both the fidelity replay and the predicted-telemetry export
# below need the SAME "whichever of the 5 real components currently has a production version"
# set, loaded from the SAME real ModelRegistry artifacts; written once here so the two never drift
# against each other.
# ---------------------------------------------------------------------------
def load_production_components(models_dir: Path) -> dict[str, DTComponent]:
    index_path = models_dir / "registry_index.json"
    if not index_path.exists():
        logger.warning("no model registry found at %s", index_path)
        return {}
    registry = ModelRegistry(models_dir=models_dir, index_path=index_path)
    loaded: dict[str, DTComponent] = {}
    for name, cls in _DT_COMPONENT_CLASSES.items():
        version = registry.get_current_version(name)
        if version is None:
            logger.warning("no production version registered for %r — skipping", name)
            continue
        instance = cls()
        registry.load_artifact_into(instance, version)
        loaded[name] = instance
    return loaded


# ---------------------------------------------------------------------------
# Fidelity recomputation — replays real history through the real, current production models via
# the real Module 12 FidelityEvaluator, exactly mirroring ContinuousOrchestrator's own runtime
# prediction+fidelity cycle, just walked across the FULL stored history instead of one live batch.
# ---------------------------------------------------------------------------
def compute_fidelity_timeseries(
    history: pd.DataFrame, loaded: dict[str, DTComponent], settings: Settings, window: int, stride: int
) -> dict[str, pd.DataFrame]:
    if not loaded:
        return {}
    evaluator = FidelityEvaluator(settings.fidelity)  # ONE shared evaluator, matching runtime's single instance
    results: dict[str, list[dict]] = {name: [] for name in loaded}
    n = len(history)
    i = 0
    while i < n:
        chunk = history.iloc[i : i + window]
        i += stride
        if len(chunk) < 2:  # mk_mmd requires >=2 samples per side
            continue
        for name, instance in loaded.items():
            cls = _DT_COMPONENT_CLASSES[name]
            target_col = _TARGET_COLUMNS[name]
            try:
                feat = chunk
                if cls.DEPENDENCIES:
                    feat = with_dependency_ground_truth(chunk, cls.DEPENDENCIES, _OUTPUT_FIELDS)
                y_pred = instance.predict(feat)
                y_true = feat[target_col].to_numpy(dtype=float)
                result = evaluator.evaluate(name, y_true, y_pred, update_window=True)
            except Exception as exc:  # noqa: BLE001 — one bad window must never abort the whole replay
                logger.warning("fidelity replay window skipped for %r: %s", name, exc)
                continue
            results[name].append(
                {
                    "timestamp": chunk["timestamp"].iloc[-1],
                    "fidelity_score": result.fidelity_score,
                    "rmse": result.raw_metrics["rmse"],
                    "mae": result.raw_metrics["mae"],
                    "wasserstein": result.raw_metrics["wasserstein"],
                    "mk_mmd": result.raw_metrics["mk_mmd"],
                    "status": result.status,
                }
            )

    return {name: pd.DataFrame(rows) for name, rows in results.items() if rows}


# ---------------------------------------------------------------------------
# Original vs. predicted telemetry export — two separate, directly comparable CSVs: the real D1
# history exactly as ingested, and the DT's own predictions for that same history, computed via
# ONE real DTOrchestrator.run_predictions() pass (dependency-chained through each component's own
# LIVE prediction — throughput's prediction feeds latency/prb_utilization/jitter, matching how
# ContinuousOrchestrator actually serves predictions — never the ground-truth-fed shortcut
# compute_fidelity_timeseries above uses for independent per-component fidelity windows).
# ---------------------------------------------------------------------------
def compute_predicted_telemetry(history: pd.DataFrame, loaded: dict[str, DTComponent]) -> pd.DataFrame | None:
    if not loaded:
        logger.warning("no production components loaded — predicted-telemetry CSV will be skipped")
        return None
    dt_registry = DTModelRegistry()
    for component in loaded.values():
        dt_registry.register(component)
    try:
        predictions = DTOrchestrator(dt_registry).run_predictions(history)
    except OrchestratorError as exc:
        # A partially-populated registry (e.g. latency's production version exists but
        # throughput's doesn't) can produce an invalid dependency graph — fail soft here since
        # this is a read-only reporting tool, not the live system (prompt.md's "one bad thing
        # must never abort a whole replay" discipline, same as compute_fidelity_timeseries above).
        logger.warning("could not compute predicted telemetry: %s", exc)
        return None

    identity_cols = [c for c in ("timestamp", "ue_id", "cell_id") if c in history.columns]
    result = history[identity_cols].copy()
    for name, series in predictions.items():
        output_field = _DT_COMPONENT_CLASSES[name].OUTPUT_FIELD
        result[output_field] = series.to_numpy()
    return result


def save_telemetry_csvs(history: pd.DataFrame, predicted: pd.DataFrame | None, output_dir: Path) -> list[Path]:
    """The two files this tool exists to produce: the real, as-ingested telemetry and the DT's own
    predictions for it, kept as separate files (never merged into one) so each can be opened,
    diffed, or plotted independently — join them on (timestamp, ue_id, cell_id) if a side-by-side
    comparison is needed."""
    saved = []
    original_path = output_dir / "telemetry_original.csv"
    history.to_csv(original_path, index=False)
    saved.append(original_path)
    if predicted is not None and not predicted.empty:
        predicted_path = output_dir / "telemetry_predicted.csv"
        predicted.to_csv(predicted_path, index=False)
        saved.append(predicted_path)
    return saved


def _iqr_fence(values: np.ndarray, whisker: float = 3.0, floor: float = 1.0) -> float:
    """A robust (outlier-resistant) upper display bound — Tukey's IQR fence. Only affects what's
    VISIBLE in this one chart; the full unclipped values are always written to the companion CSVs
    (see `plot_fidelity`'s callers) so nothing is ever actually hidden, only kept off one crowded
    axis."""
    q1, q3 = np.nanpercentile(values, [25, 75])
    return max(float(q3 + whisker * (q3 - q1)), floor)


def plot_fidelity(fidelity_series: dict[str, pd.DataFrame], output_dir: Path, show: bool) -> Path | None:
    defined = {name: df for name, df in fidelity_series.items() if df["fidelity_score"].notna().any()}
    if not defined:
        logger.warning("no component crossed min_history_for_normalization in this replay — no fidelity plot")
        return None

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax_score, ax_rmse = axes

    n_score_clipped = 0
    n_rmse_clipped = 0
    all_rmse = np.concatenate([df["rmse"].to_numpy(dtype=float) for df in defined.values()])
    rmse_ylim = (0.0, _iqr_fence(all_rmse))

    for name, df in defined.items():
        color = _COMPONENT_COLORS.get(name, None)
        ok = df[df["status"] == "ok"]
        ax_score.plot(ok["timestamp"], ok["fidelity_score"], marker="o", markersize=3, linewidth=1.2, label=name, color=color)
        ax_rmse.plot(df["timestamp"], df["rmse"], marker=".", markersize=3, linewidth=1.0, label=name, color=color)
        n_score_clipped += int(((ok["fidelity_score"] < _FIDELITY_SCORE_CLIP[0]) | (ok["fidelity_score"] > _FIDELITY_SCORE_CLIP[1])).sum())
        n_rmse_clipped += int((df["rmse"] > rmse_ylim[1]).sum())

    # Display-range clipping only (never the underlying data — see companion CSVs): a single
    # extreme point (e.g. one out-of-range-flagged telemetry reading feeding the eps-stabilized
    # rolling-window normalization) can otherwise compress every other point into a flat line.
    ax_score.set_ylim(*_FIDELITY_SCORE_CLIP)
    ax_rmse.set_ylim(*rmse_ylim)

    clip_note = ""
    if n_score_clipped or n_rmse_clipped:
        clip_note = (
            f"{n_score_clipped} score pt(s) outside [{_FIDELITY_SCORE_CLIP[0]},{_FIDELITY_SCORE_CLIP[1]}] and "
            f"{n_rmse_clipped} RMSE pt(s) above {rmse_ylim[1]:.2f} run off-axis for readability — "
            f"exact values in fidelity_timeseries_<component>.csv"
        )

    ax_score.set_ylabel("Composite FidelityScore (Module 12, eq. 6-8)", fontsize=9)
    title = "Composite Fidelity Score Over Time (recomputed via the real FidelityEvaluator)"
    if clip_note:
        title += f"\n{clip_note}"
    ax_score.set_title(title, fontsize=9.5)
    ax_score.axhline(0.0, color="grey", linewidth=0.6, linestyle="--")
    ax_score.grid(alpha=0.3)
    ax_score.legend(fontsize=8, ncol=5, loc="lower left")

    ax_rmse.set_ylabel("Raw RMSE", fontsize=9)
    ax_rmse.set_title("Raw RMSE Over Time (component-native units)", fontsize=11)
    ax_rmse.grid(alpha=0.3)
    ax_rmse.legend(fontsize=8, ncol=5, loc="upper left")
    ax_rmse.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()

    fig.tight_layout()
    out_path = output_dir / "fidelity_over_time.png"
    fig.savefig(out_path, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)
    return out_path


def save_fidelity_csvs(fidelity_series: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    """The full, unclipped recomputed fidelity time series per component — always the source of
    truth; the PNG's display range is cosmetic only."""
    saved = []
    for name, df in fidelity_series.items():
        out_path = output_dir / f"fidelity_timeseries_{name}.csv"
        df.to_csv(out_path, index=False)
        saved.append(out_path)
    return saved


# ---------------------------------------------------------------------------
# Lifecycle records plot
# ---------------------------------------------------------------------------
def _annotate_bar(ax, rect, value: float, ylim: tuple[float, float]) -> bool:
    """Label one bar with its real, unclipped value. Returns True if the value fell outside
    `ylim` (the bar itself is visually truncated by matplotlib at the axis edge in that case —
    this places a small off-scale marker + the exact number right at that edge instead of letting
    the truncation silently look like the value simply stopped existing)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    lo, hi = ylim
    x = rect.get_x() + rect.get_width() / 2
    if value > hi:
        # Placed just INSIDE the top edge (va="top" growing downward), never above the axes —
        # so it can never collide with the title, regardless of how many bars are off-scale.
        ax.annotate(
            f"▲ {value:.3f}", (x, hi), xytext=(0, -3), textcoords="offset points",
            ha="center", va="top", fontsize=7.5, color="#8b0000", fontweight="bold",
        )
        return True
    if value < lo:
        ax.annotate(
            f"▼ {value:.3f}", (x, lo), xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.5, color="#8b0000", fontweight="bold",
        )
        return True
    va, dy = ("bottom", 2) if value >= 0 else ("top", -2)
    ax.annotate(
        f"{value:.3f}", (x, value), xytext=(0, dy), textcoords="offset points",
        ha="center", va=va, fontsize=7,
    )
    return False


def plot_lifecycle(records: list[LifecycleRecord], output_dir: Path, show: bool) -> Path | None:
    if not records:
        return None
    records = sorted(records, key=lambda r: r.timestamp)
    labels = [f"#{i+1}\n{r.affected_component}\n({r.decision_strategy})" for i, r in enumerate(records)]
    before = [r.fidelity_before if r.fidelity_before is not None else np.nan for r in records]
    after = [r.fidelity_after if r.fidelity_after is not None else np.nan for r in records]
    verdicts = [r.verification_result for r in records]

    x = np.arange(len(records))
    width = 0.35
    fig_w = max(7.0, 1.6 * len(records))
    fig, ax = plt.subplots(figsize=(fig_w, 5.6))
    bar_before = ax.bar(x - width / 2, before, width, label="fidelity_before", color="#9ecae1")
    bar_after = ax.bar(x + width / 2, after, width, label="fidelity_after", color="#fdae6b")
    for rect, verdict in zip(bar_after, verdicts):
        edge = "#2ca02c" if verdict == "ACCEPT" else "#d62728"
        rect.set_edgecolor(edge)
        rect.set_linewidth(2.2)

    # Display-range clipping only, same convention as `plot_fidelity` (never the underlying
    # data): FidelityScore is normalized against a ROLLING historical min/max, not a fixed 0-1
    # scale (unlike e.g. R^2/cosine similarity) — a point genuinely outside its own reference
    # window's prior min/max legitimately produces a score > 1.0 or < 0.0, which is correct
    # Module 12 behavior, not a bug. Every bar still gets its exact real value as a text label
    # (see `_annotate_bar`) regardless of whether it's clipped, and the full table is also
    # written to a companion CSV below — only the axis SCALE is bounded for readability.
    ax.set_ylim(*_FIDELITY_SCORE_CLIP)
    n_clipped = 0
    for rect, value in zip(bar_before, before):
        n_clipped += _annotate_bar(ax, rect, value, _FIDELITY_SCORE_CLIP)
    for rect, value in zip(bar_after, after):
        n_clipped += _annotate_bar(ax, rect, value, _FIDELITY_SCORE_CLIP)

    clip_note = f" {n_clipped} bar(s) run off-axis (▲/▼ marks the exact value)." if n_clipped else ""
    explanation = textwrap.fill(
        "FidelityScore is relative to each component's OWN rolling history, not a fixed 0-1 "
        f"scale — values above 1 or below 0 are real, expected outcomes.{clip_note}",
        width=max(46, int(fig_w * 9.5)),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f"FidelityScore, axis capped to [{_FIDELITY_SCORE_CLIP[0]},{_FIDELITY_SCORE_CLIP[1]}]", fontsize=9)
    ax.set_title(
        "Lifecycle Records — Adaptation Outcomes  (green outline = ACCEPT, red outline = REJECT)\n"
        f"{explanation}",
        fontsize=9,
    )
    ax.axhline(0.0, color="grey", linewidth=0.6)
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle=":")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "lifecycle_adaptation_outcomes.png"
    fig.savefig(out_path, dpi=140)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)

    # Full, unclipped numbers — the chart's axis is cosmetic only, this CSV is the source of truth.
    csv_path = output_dir / "lifecycle_outcomes.csv"
    pd.DataFrame(
        {
            "index": [i + 1 for i in range(len(records))],
            "event_id": [r.event_id for r in records],
            "timestamp": [r.timestamp for r in records],
            "affected_component": [r.affected_component for r in records],
            "decision_strategy": [r.decision_strategy for r in records],
            "fidelity_before": before,
            "fidelity_after": after,
            "verification_result": verdicts,
        }
    ).to_csv(csv_path, index=False)
    print(f"  lifecycle raw values (unclipped): {csv_path}")
    for i, (b, a, v) in enumerate(zip(before, after, verdicts), 1):
        print(f"    #{i} {records[i-1].affected_component:16s} before={b:.4f}  after={a:.4f}  {v}")

    return out_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.config) if args.config else load_settings()
    history_path, models_dir, lifecycle_path, reports_dir, output_dir = resolve_paths(args, settings)
    output_dir.mkdir(parents=True, exist_ok=True)
    stride = args.stride if args.stride is not None else args.window

    print(f"Reading telemetry history from: {history_path}")
    history = load_telemetry_history(history_path)
    print(f"  -> {len(history)} rows, {history['timestamp'].nunique()} distinct timestamps, "
          f"source={history['source'].unique().tolist() if 'source' in history.columns else 'n/a'}")

    print(f"Reading model registry from:    {models_dir}")
    print(f"Reading lifecycle records from: {lifecycle_path}")
    records = load_lifecycle_records(lifecycle_path, reports_dir)
    print(f"  -> {len(records)} lifecycle record(s)")

    saved: list[Path] = []
    saved += plot_telemetry(history, output_dir, args.show)

    loaded_components = load_production_components(models_dir)
    predicted = compute_predicted_telemetry(history, loaded_components)
    telemetry_csvs = save_telemetry_csvs(history, predicted, output_dir)
    saved += telemetry_csvs
    print(f"  original telemetry ({len(history)} rows):  {telemetry_csvs[0]}")
    if len(telemetry_csvs) > 1:
        print(f"  predicted telemetry ({len(predicted)} rows): {telemetry_csvs[1]}")
    else:
        print("  predicted telemetry: skipped — see warning above")

    print(f"Recomputing fidelity over {len(history)} rows (window={args.window}, stride={stride}) "
          f"via the real FidelityEvaluator against each component's current production model...")
    fidelity_series = compute_fidelity_timeseries(history, loaded_components, settings, args.window, stride)
    fid_path = plot_fidelity(fidelity_series, output_dir, args.show)
    if fid_path:
        saved.append(fid_path)
    saved += save_fidelity_csvs(fidelity_series, output_dir)
    for name, df in fidelity_series.items():
        n_ok = int((df["status"] == "ok").sum())
        latest = df[df["status"] == "ok"]["fidelity_score"]
        latest_str = f"{latest.iloc[-1]:.4f}" if not latest.empty else "n/a"
        print(f"  {name:16s} windows_evaluated={len(df):3d}  windows_with_score={n_ok:3d}  latest_score={latest_str}")

    lc_path = plot_lifecycle(records, output_dir, args.show)
    if lc_path:
        saved.append(lc_path)
        saved.append(output_dir / "lifecycle_outcomes.csv")
        n_accept = sum(1 for r in records if r.verification_result == "ACCEPT")
        print(f"  lifecycle: {n_accept}/{len(records)} ACCEPTed")

    print("\nSaved plots:")
    for p in saved:
        print(f"  {p}")
    if not saved:
        print("  (none — see warnings above)")

    if args.show:
        print("\n--show was passed: close the figure windows to exit.")
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
