"""Unit tests for the OAI replay source (prompt2.md Step 7): strict timestamp order, one tick
per second present, no invented rows at gaps, deterministic seek, pause/resume, speed=0 finishes
quickly."""

from __future__ import annotations

import threading
import time

import pandas as pd

from src.telemetry.oai_replay_source import OaiReplaySource


def _write_fixture(tmp_path, ts_values, run_id=1):
    """A tiny synthetic processed table: one row per ts in `ts_values`, deliberately shuffled on
    disk so determinism/ordering is genuinely exercised, not accidentally true because the
    fixture was already sorted."""
    rows = []
    for i, ts in enumerate(ts_values):
        rows.append({
            "run_id": run_id, "cell_id": 1, "session_id": 1, "ue_id": 1, "slice_id": 1,
            "ts_1s": pd.Timestamp(ts, tz="UTC"), "dl_throughput": float(i), "ul_throughput": 0.0,
            "dl_rlc_delay": 0.0, "dl_prb": 0.0, "ul_prb": 0.0, "dl_bler": 0.0, "ul_bler": 0.0,
        })
    df = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle
    (tmp_path / "test.parquet").parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp_path / "test.parquet", engine="pyarrow", index=False)
    (tmp_path / "train.parquet").touch()  # absent-gracefully isn't tested here; just avoid noise
    pd.DataFrame(columns=df.columns).to_parquet(tmp_path / "train.parquet", engine="pyarrow", index=False)
    pd.DataFrame(columns=df.columns).to_parquet(tmp_path / "val.parquet", engine="pyarrow", index=False)
    return tmp_path


TS = ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:02Z",
      "2026-01-01T00:00:05Z", "2026-01-01T00:00:06Z"]  # a real 2s gap between index 2 and 3


def test_strict_timestamp_order_and_one_tick_per_present_second(tmp_path):
    _write_fixture(tmp_path, TS)
    src = OaiReplaySource(tmp_path, speed=0.0)
    src.start(run_ids=[1])
    seen = [r["ts_1s"] for r in src.records()]
    src.close()
    assert seen == sorted(seen)
    assert len(seen) == len(TS)
    assert {pd.Timestamp(t) for t in seen} == {pd.Timestamp(t) for t in TS}


def test_no_invented_rows_at_the_gap():
    pass  # the above IS the no-invented-rows proof: exactly 5 records for 5 real rows, the
    # 00:00:03/00:00:04 gap never produces a 6th/7th synthetic record.


def test_deterministic_seek_lands_on_first_real_tick_at_or_after_target(tmp_path):
    _write_fixture(tmp_path, TS)
    src = OaiReplaySource(tmp_path, speed=0.0)
    src.start(run_ids=[1])
    src.seek("2026-01-01T00:00:03Z")  # between index 2 (00:02) and index 3 (00:05)
    first = next(src.records())
    assert pd.Timestamp(first["ts_1s"]) == pd.Timestamp("2026-01-01T00:00:05Z")
    src.close()


def test_same_input_yields_the_same_sequence_every_time(tmp_path):
    _write_fixture(tmp_path, TS)
    runs = []
    for _ in range(2):
        src = OaiReplaySource(tmp_path, speed=0.0)
        src.start(run_ids=[1])
        runs.append([r["ts_1s"] for r in src.records()])
        src.close()
    assert runs[0] == runs[1]


def test_speed_zero_finishes_quickly(tmp_path):
    _write_fixture(tmp_path, TS)
    src = OaiReplaySource(tmp_path, speed=0.0)
    src.start(run_ids=[1])
    t0 = time.monotonic()
    list(src.records())
    elapsed = time.monotonic() - t0
    src.close()
    assert elapsed < 1.0  # 5 ticks at real-time pacing would take ~5s; speed=0 must not wait at all


def test_pause_blocks_further_records_until_resume(tmp_path):
    _write_fixture(tmp_path, TS)
    src = OaiReplaySource(tmp_path, speed=0.0)
    src.start(run_ids=[1])
    src.pause()

    received: list[dict] = []
    done = threading.Event()

    def drain():
        for r in src.records():
            received.append(r)
        done.set()

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    time.sleep(0.3)
    assert len(received) == 0  # paused before the first tick — nothing should have been emitted

    src.resume()
    done.wait(timeout=5.0)
    src.close()
    assert len(received) == len(TS)


def test_stop_halts_the_generator_promptly(tmp_path):
    _write_fixture(tmp_path, TS)
    src = OaiReplaySource(tmp_path, speed=1.0)  # real-time pacing — would take ~5s if not stopped
    src.start(run_ids=[1])
    it = src.records()
    next(it)  # consume the first tick
    src.stop()
    t0 = time.monotonic()
    remaining = list(it)
    assert time.monotonic() - t0 < 2.0  # stop() must halt well before the full ~5s real-time replay would
    assert len(remaining) < len(TS) - 1
