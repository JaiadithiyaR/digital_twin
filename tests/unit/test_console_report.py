"""Unit tests for the human-readable console channel (`src/common/console_report.py`)."""

from __future__ import annotations

from datetime import UTC, datetime

from src.common import console_report as console
from src.fidelity.component_policy import ComponentFidelityStatus


def _status(component: str, score: float | None, state: str, consecutive: int = 0, severity: float | None = None):
    return ComponentFidelityStatus(
        component=component, fidelity_score=score, threshold=0.5, consecutive_below=consecutive,
        min_consecutive_evaluations=3, state=state, severity=severity, measured_at=datetime.now(UTC), reason="test",
    )


def test_bracketed_line_prefixes_are_printed_literally_not_parsed_as_rich_markup(capsys):
    console.trigger_seen("dl_bler", "fidelity_degradation", 0.25, fidelity_score=0.3, threshold=0.5, version="dl_bler-v1")
    console.decision_made("dl_bler", "recalibrate", 0.8, "stale parameters after load shift")
    console.step("verify", "ACCEPT: checks ok")
    out = capsys.readouterr().out
    assert "[trigger] component=dl_bler" in out and "threshold=0.50" in out and "production=dl_bler-v1" in out
    assert "[decision] component=dl_bler strategy=recalibrate confidence=0.80 (LLM)" in out
    assert "root cause: stale parameters after load shift" in out
    assert "[verify] ACCEPT" in out


def test_decision_source_label_never_mislabels_a_pinned_or_fallback_decision(capsys):
    console.decision_made("dl_bler", "regenerate", 1.0, source="pinned")
    console.decision_made("dl_bler", "recalibrate", 0.0, fallback=True)
    out = capsys.readouterr().out
    assert "strategy=regenerate confidence=1.00 (PINNED)" in out
    assert "(DETERMINISTIC FALLBACK — LLM unavailable)" in out and "(LLM)" not in out


def test_cycle_report_shows_every_component_and_marks_unified_score_report_only(capsys):
    console.cycle_report(
        7, window_start="t0", window_end="t1", window_rows=200, d1_rows=12_345, ingest_rate=58.2,
        statuses=[
            _status("dl_bler", 0.91, "ok"),
            _status("dl_prb", 0.31, "below_threshold", consecutive=2),
            _status("ul_prb", 0.12, "adapt", consecutive=3, severity=0.76),
            _status("ul_bler", None, "insufficient_history"),
        ],
        model_versions={"dl_bler": "dl_bler-v2"}, unified_score=0.4467, triggers=["ul_prb"],
    )
    out = capsys.readouterr().out
    assert "[cycle 7]" in out and "12,345 rows" in out and "58.2 rows/s" in out
    assert "dl_bler-v2" in out and "BELOW" in out and "2/3" in out
    assert "ADAPT -> trigger severity=0.760" in out and "warming" in out
    assert "F_U (report only, not an adaptation trigger) = 0.4467" in out
    assert "triggers this cycle: ul_prb" in out


def test_error_text_containing_markup_like_brackets_never_crashes(capsys):
    message = "recalibrate failed: missing column(s) ['dl_bler_pred'] near [/bold] and [red]"
    console.warning(message)  # rich would raise MarkupError on the stray closing tag
    console.skipped(message)
    out = capsys.readouterr().out
    assert "[/bold]" in out and "['dl_bler_pred']" in out


def test_transcript_mirrors_console_lines_to_a_file(tmp_path, capsys):
    transcript = tmp_path / "console" / "run.log"
    console.set_transcript(transcript)
    try:
        console.info("hello transcript")
    finally:
        console.set_transcript(None)
    console.info("not mirrored")
    text = transcript.read_text()
    assert "[info] hello transcript" in text and "not mirrored" not in text
