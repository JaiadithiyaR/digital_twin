"""Phase13 §6 console output: "uses `rich` if installed, else plain aligned text." A compact
live line every ~10 cycles plus a clearly framed block whenever an adaptation event happens."""

from __future__ import annotations

from typing import Any

try:
    from rich.console import Console

    _console: Console | None = Console()
except ImportError:  # pragma: no cover - rich is confirmed installed in this environment, but
    # the fallback is kept real and tested rather than assumed unreachable.
    _console = None


def live_line(mode: str, cycle: int, total_cycles: int, unified_fidelity: float | None, lowest_component: str | None) -> None:
    fu = f"{unified_fidelity:.4f}" if unified_fidelity is not None else "insufficient_history"
    text = f"[{mode}] window {cycle}/{total_cycles}  F_U={fu}  lowest={lowest_component or '-'}"
    # Explicit flush always — when stdout is redirected to a file (not a TTY), Python fully
    # buffers it and neither print() nor rich.Console.print() flushes on its own; a run killed
    # before a clean exit (Ctrl-C, a timeout, a crash) can silently lose every buffered line.
    if _console is not None:
        _console.print(text, style="dim", markup=False)
        _console.file.flush()
    else:
        print(text, flush=True)


def adaptation_event_block(mode: str, event: dict[str, Any]) -> None:
    lines = [
        f"ADAPTATION EVENT — mode={mode} component={event.get('component')}",
        f"  trigger_type={event.get('trigger_type')} severity={event.get('severity')}",
        f"  window={event.get('window')} attempt={event.get('attempt')} strategy={event.get('decision', {}).get('strategy')}",
        f"  gate_result={event.get('gate_result')}  ({event.get('fidelity_before')} -> {event.get('fidelity_after')})",
        f"  version {event.get('version_before')} -> {event.get('version_after')}",
        f"  duration_seconds={event.get('duration_seconds', 0):.2f}",
    ]
    if _console is not None:
        _console.rule(f"[bold]{mode} adaptation event[/bold]")
        for line in lines:
            _console.print(line, markup=False)
        _console.rule()
        _console.file.flush()
    else:
        border = "=" * 70
        print(border)
        for line in lines:
            print(line)
        print(border, flush=True)
