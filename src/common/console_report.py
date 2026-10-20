"""Human-readable console output for the live/demo orchestrator (`python -m src.main`).

The structured JSON logger (`src/common/logging.py`) stays the system of record (full detail in
`logs/digital_twin.log`; WARNING+ also on the console). This module is the parallel, readable
channel: a per-cycle per-component fidelity table, one line per step of every adaptation
(trigger -> decision -> agent -> verification -> hot swap -> post-adaptation check -> dependency
revalidation -> lifecycle), and a framed summary block per adaptation event.

Every line is also appended (plain text, timestamped) to a per-run transcript file when one is
configured (`set_transcript()`), so a run's console output survives the terminal.

Two rules this module must keep:
- `markup=False` on every rich print: lines carry literal "[cycle]"/"[trigger]"/"[warning]" tags
  and arbitrary exception text, which rich would otherwise parse as markup (silently stripping
  tags, or raising MarkupError on e.g. "[/x]").
- explicit flush after every print: a redirected stdout is fully buffered, and a killed process
  would otherwise lose every unflushed line.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from src.fidelity.component_policy import ComponentFidelityStatus

try:
    from rich.console import Console

    _console: "Console | None" = Console()
except ImportError:  # pragma: no cover - rich is a declared dependency
    _console = None

_RULE_WIDTH = 100
_transcript_path: Path | None = None
_transcript_lock = threading.Lock()


def set_transcript(path: Path | None) -> None:
    """Mirror every subsequent console line into `path` (plain text). `None` disables."""
    global _transcript_path
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    _transcript_path = path


def _write_transcript(text: str) -> None:
    if _transcript_path is None:
        return
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _transcript_lock:
        with _transcript_path.open("a", encoding="utf-8") as f:
            for line in text.splitlines() or [""]:
                f.write(f"{stamp} {line}\n")


def _print(text: str, style: str | None = None) -> None:
    if _console is not None:
        # soft_wrap: never hard-wrap at the terminal width (80 cols when redirected) — a table row
        # split across lines is unreadable in a log/transcript.
        _console.print(text, style=style, markup=False, highlight=False, soft_wrap=True)
        _console.file.flush()
    else:
        print(text, flush=True)
    _write_transcript(text)


def _rule(title: str = "") -> str:
    if not title:
        return "─" * _RULE_WIDTH
    side = max(2, (_RULE_WIDTH - len(title) - 2) // 2)
    return f"{'─' * side} {title} {'─' * side}"


def _fmt(value: float | None, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


# --- startup / shutdown -------------------------------------------------------------------------


def startup_banner(
    mode: str, telemetry_source: str, components: list[str], llm_available: bool, rag_available: bool,
    *, policies: dict[str, tuple[float, int]] | None = None, log_paths: dict[str, str] | None = None,
    model_versions: dict[str, str] | None = None,
) -> None:
    lines = [
        _rule("startup"),
        "Digital Twin — continuous orchestration starting",
        f"  mode={mode}  telemetry_source={telemetry_source}",
        f"  LLM available={llm_available}  RAG available={rag_available}",
        "  adaptation policy: PER-COMPONENT (each component adapts on its own FidelityScore_c; "
        "unified F_U is reported only)",
        "  trigger source: internal per-component fidelity only (external drift permanently disabled)",
        f"  components ({len(components)}):",
    ]
    for name in components:
        threshold, min_consecutive = (policies or {}).get(name, (None, None))
        version = (model_versions or {}).get(name, "?")
        lines.append(
            f"    - {name:<16} version={version:<22} threshold={_fmt(threshold, 2)}  "
            f"min_consecutive_cycles={min_consecutive}"
        )
    for label, path in (log_paths or {}).items():
        lines.append(f"  {label:<22} {path}")
    lines.append(_rule())
    _print("\n".join(lines))


def shutdown_line(clean: bool) -> None:
    _print(
        "[shutdown] background sync thread confirmed exited, D1 flushed to disk"
        if clean else "[shutdown] timed out waiting for a background thread — see logs",
        style="bold" if clean else "bold red",
    )


# --- per-cycle fidelity report ------------------------------------------------------------------

_STATE_LABEL = {
    "ok": "ok",
    "below_threshold": "BELOW",
    "adapt": "ADAPT ->",
    "insufficient_history": "warming",
    "unavailable": "no data",
}


def cycle_report(
    cycle: int, *, window_start: Any, window_end: Any, window_rows: int, d1_rows: int,
    ingest_rate: float | None, statuses: "Iterable[ComponentFidelityStatus]", model_versions: dict[str, str],
    unified_score: float | None, triggers: list[str], backfilled_windows: int = 0,
) -> None:
    rate = f"{ingest_rate:.1f} rows/s" if ingest_rate is not None else "n/a"
    lines = [
        f"[cycle {cycle}] window {window_start} .. {window_end} ({window_rows} rows"
        + (f", +{backfilled_windows} backfilled windows" if backfilled_windows else "")
        + f") | D1 history {d1_rows:,} rows | ingest {rate}",
        f"   {'component':<16} {'model version':<24} {'F_c':>9} {'thresh':>7} {'gap':>8} {'below':>6}  state",
    ]
    for s in statuses:
        below = f"{s.consecutive_below}/{s.min_consecutive_evaluations}"
        state = _STATE_LABEL.get(s.state, s.state)
        if s.state == "adapt":
            state += f" trigger severity={_fmt(s.severity, 3)}"
        lines.append(
            f"   {s.component:<16} {model_versions.get(s.component, '?'):<24} {_fmt(s.fidelity_score):>9} "
            f"{_fmt(s.threshold, 2):>7} {_fmt(s.fidelity_gap):>8} {below:>6}  {state}"
        )
    lines.append(
        f"   F_U (report only, not an adaptation trigger) = {_fmt(unified_score)}"
        f" | triggers this cycle: {', '.join(triggers) if triggers else 'none'}"
    )
    _print("\n".join(lines), style=None)


# --- adaptation steps -------------------------------------------------------------------------


def trigger_seen(
    component: str, trigger_type: str, severity: float, *, fidelity_score: float | None = None,
    threshold: float | None = None, version: str | None = None, queue_remaining: int | None = None,
) -> None:
    _print(
        f"[trigger] component={component} type={trigger_type} severity={severity:.3f} "
        f"F_c={_fmt(fidelity_score)} threshold={_fmt(threshold, 2)} production={version or '?'}"
        + (f" | still pending: {queue_remaining}" if queue_remaining is not None else ""),
        style="yellow",
    )


def decision_made(
    component: str, strategy: str, confidence: float, root_cause: str | None = None, fallback: bool = False,
    source: str = "llm",
) -> None:
    label = " (DETERMINISTIC FALLBACK — LLM unavailable)" if fallback else (" (LLM)" if source == "llm" else f" ({source.upper()})")
    _print(f"[decision] component={component} strategy={strategy} confidence={confidence:.2f}{label}")
    if root_cause:
        excerpt = " ".join(root_cause.split())
        _print(f"           root cause: {excerpt[:300]}{'…' if len(excerpt) > 300 else ''}", style="dim")


def step(tag: str, message: str, style: str | None = None) -> None:
    """One adaptation-pipeline step line, e.g. step("agent", "...")."""
    _print(f"[{tag}] {message}", style=style)


def adaptation_event(
    component: str, action: str, gate_result: str, fidelity_before: float | None, fidelity_after: float | None,
    version_before: str | None, version_after: str | None, duration_seconds: float, *,
    threshold: float | None = None, post_adaptation_fidelity: float | None = None, remaining_gap: float | None = None,
    still_flagged: bool | None = None, dependents: list[dict[str, Any]] | None = None, event_id: str | None = None,
    attempt: str | None = None,
) -> None:
    lines = [
        _rule("adaptation event"),
        f"ADAPTATION EVENT — component={component} action={action}"
        + (f" attempt={attempt}" if attempt else "") + (f" event_id={event_id}" if event_id else ""),
        f"  gate_result={gate_result}  (verification fidelity {_fmt(fidelity_before)} -> {_fmt(fidelity_after)})",
        f"  version {version_before} -> {version_after}",
    ]
    if threshold is not None:
        lines.append(
            f"  post-adaptation F_c={_fmt(post_adaptation_fidelity)} vs threshold {_fmt(threshold, 2)} -> "
            + ("STILL BELOW threshold (gap " + _fmt(remaining_gap) + "), remains flagged by its policy"
               if still_flagged else ("meets threshold" if still_flagged is False else "not yet measurable"))
        )
    if dependents:
        for dep in dependents:
            lines.append(
                f"  dependent {dep['component']:<16} F_c={_fmt(dep.get('fidelity_score'))} "
                f"threshold={_fmt(dep.get('threshold'), 2)} -> {dep.get('status')}"
            )
    lines.append(f"  duration_seconds={duration_seconds:.2f}")
    lines.append(_rule())
    style = "green" if gate_result == "ACCEPT" else ("red" if gate_result == "REJECT" else None)
    _print("\n".join(lines), style=style)


def skipped(reason: str) -> None:
    _print(f"[skipped] {reason}", style="dim yellow")


def info(message: str) -> None:
    _print(f"[info] {message}", style="dim")


def warning(message: str) -> None:
    _print(f"[warning] {message}", style="bold yellow")
