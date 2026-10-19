"""Clean, human-readable console output for the live/demo orchestrator (`python -m src.main`).

The structured JSON logger (`src/common/logging.py`) stays the system of record — every detail
still goes to the log file (and to the console too, at WARNING+, so real errors are never
hidden) — but routine INFO-level pipeline chatter is far too dense to watch live. This module is
a second, parallel reporting channel: short status lines and framed adaptation-event blocks,
printed directly (never through `logging`), so a human watching the terminal can actually follow
what the system is doing. Mirrors `baselines/_console.py`'s proven rich-with-plain-fallback
pattern, retargeted at the live orchestrator's own concepts (trigger -> decision -> agent ->
verification -> hot swap) instead of a baseline experiment's cycle loop.
"""

from __future__ import annotations

try:
    from rich.console import Console

    _console: "Console | None" = Console()
except ImportError:  # pragma: no cover - rich is a declared dependency, but the fallback is
    # kept real and reachable rather than assumed unreachable.
    _console = None


def _print(text: str, style: str | None = None) -> None:
    # Explicit flush always — real bug, reproduced: when stdout is redirected to a file (not a
    # TTY), Python fully-buffers it, and neither bare print() nor rich.Console.print() forces a
    # flush on their own. A long-running process whose output is redirected and later killed
    # (SIGTERM, Ctrl-C) can lose every buffered line that never got flushed — confirmed directly:
    # a 7-minute `--mode live` run genuinely processed tens of thousands of rows and multiple
    # adaptation cycles, but its redirected log showed only the first status line. The structured
    # JSON logger doesn't have this problem (`logging.StreamHandler.emit()` flushes after every
    # record) — this module needs the same discipline since it never goes through `logging`.
    if _console is not None:
        _console.print(text, style=style)
        _console.file.flush()
    else:
        print(text, flush=True)


def startup_banner(mode: str, telemetry_source: str, components: list[str], llm_available: bool, rag_available: bool) -> None:
    lines = [
        "Digital Twin — continuous orchestration starting",
        f"  mode={mode}  telemetry_source={telemetry_source}",
        f"  components ({len(components)}): {', '.join(components)}",
        f"  LLM available={llm_available}  RAG available={rag_available}",
    ]
    if _console is not None:
        _console.rule("[bold]startup[/bold]")
        for line in lines:
            _console.print(line)
        _console.rule()
        _console.file.flush()
    else:
        print("=" * 70)
        for line in lines:
            print(line)
        print("=" * 70, flush=True)


def live_status(unified_fidelity: float | None, lowest_component: str | None, n_components_defined: int) -> None:
    fu = f"{unified_fidelity:.4f}" if unified_fidelity is not None else "insufficient_history"
    _print(f"[fidelity] F_U={fu}  lowest={lowest_component or '-'}  defined={n_components_defined}", style="dim")


def trigger_seen(component: str, trigger_type: str, severity: float) -> None:
    _print(f"[trigger] component={component} type={trigger_type} severity={severity:.3f}", style="yellow")


def decision_made(component: str, strategy: str, confidence: float) -> None:
    _print(f"[decision] component={component} strategy={strategy} confidence={confidence:.2f}")


def adaptation_event(
    component: str, action: str, gate_result: str, fidelity_before: float | None, fidelity_after: float | None,
    version_before: str | None, version_after: str | None, duration_seconds: float,
) -> None:
    before = f"{fidelity_before:.4f}" if fidelity_before is not None else "None"
    after = f"{fidelity_after:.4f}" if fidelity_after is not None else "None"
    lines = [
        f"ADAPTATION EVENT — component={component} action={action}",
        f"  gate_result={gate_result}  ({before} -> {after})",
        f"  version {version_before} -> {version_after}",
        f"  duration_seconds={duration_seconds:.2f}",
    ]
    style = "green" if gate_result == "ACCEPT" else ("red" if gate_result == "REJECT" else "white")
    if _console is not None:
        _console.rule(f"[bold {style}]adaptation event[/bold {style}]")
        for line in lines:
            _console.print(line, style=style if line.startswith("  gate_result") else None)
        _console.rule()
        _console.file.flush()
    else:
        border = "=" * 70
        print(border)
        for line in lines:
            print(line)
        print(border, flush=True)


def skipped(reason: str) -> None:
    _print(f"[skipped] {reason}", style="dim yellow")


def warning(message: str) -> None:
    _print(f"[warning] {message}", style="bold yellow")


def shutdown_line(clean: bool) -> None:
    _print("[shutdown] both background threads confirmed exited" if clean else "[shutdown] timed out waiting for a background thread — see logs", style="bold" if clean else "bold red")
