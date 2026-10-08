"""Phase13 §6 console output: "uses `rich` if installed, else plain aligned text." A compact
live line every ~10 cycles plus a clearly framed block whenever an adaptation event happens."""

from __future__ import annotations

from typing import Any

try:
    from rich.console import Console

    _console: "Console | None" = Console()
except ImportError:  # pragma: no cover - rich is confirmed installed in this environment, but
    # the fallback is kept real and tested rather than assumed unreachable.
    _console = None


def live_line(mode: str, cycle: int, total_cycles: int, unified_fidelity: float | None, lowest_component: str | None) -> None:
    fu = f"{unified_fidelity:.4f}" if unified_fidelity is not None else "insufficient_history"
    text = f"[{mode}] cycle {cycle}/{total_cycles}  F_U={fu}  lowest={lowest_component or '-'}"
    if _console is not None:
        _console.print(text, style="dim")
    else:
        print(text)


def adaptation_event_block(mode: str, event: dict[str, Any]) -> None:
    lines = [
        f"ADAPTATION EVENT — mode={mode} component={event.get('component')}",
        f"  trigger_type={event.get('trigger_type')} severity={event.get('severity')}",
        f"  strategy={event.get('decision', {}).get('strategy')}",
        f"  gate_result={event.get('gate_result')}  ({event.get('fidelity_before')} -> {event.get('fidelity_after')})",
        f"  version {event.get('version_before')} -> {event.get('version_after')}",
        f"  duration_seconds={event.get('duration_seconds', 0):.2f}",
    ]
    if _console is not None:
        _console.rule(f"[bold]{mode} adaptation event[/bold]")
        for line in lines:
            _console.print(line)
        _console.rule()
    else:
        border = "=" * 70
        print(border)
        for line in lines:
            print(line)
        print(border)
