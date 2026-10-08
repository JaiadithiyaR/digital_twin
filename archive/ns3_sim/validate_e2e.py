"""End-to-end validation for Module 1 (prompt.md §0.2-0.4, §7): launches the real, compiled
ns3_sim/nr_5g_telemetry_sim binary, connects Module 2's actual `ZmqTelemetrySource` to it (no
mocking on either side), and runs every received record through the real `TelemetryPreprocessor`.

This is a manual/CI-with-ns3-built validation script, not part of `tests/` — it needs the
vendored ns-3 tree built (`./scripts/setup_ns3.sh`) and takes tens of seconds of real wall-clock
simulation time, which doesn't belong in the fast unit/integration suite run repeatedly during
development. Run it after building or changing `nr_5g_telemetry_sim.cc`:

    python ns3_sim/validate_e2e.py

Exits non-zero (and prints exactly what failed) if the real producer, the real ZeroMQ transport,
or the real preprocessing pipeline don't agree — never claims success without observing it
(prompt.md §0.24/§0.25).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_settings  # noqa: E402
from src.telemetry.preprocessing import TelemetryPreprocessor  # noqa: E402
from src.telemetry.zmq_source import ZmqTelemetrySource  # noqa: E402

BINARY = (
    REPO_ROOT
    / "ns3_sim"
    / "ns-3-dev"
    / "build"
    / "scratch"
    / "nr_5g_telemetry_sim"
    / "ns3.48-nr_5g_telemetry_sim"
)
UE_NUM = 6
SIM_TIME = "6s"
TELEMETRY_INTERVAL_MS = "200ms"
# 27 publish ticks (from ~0.6s to 6s at 200ms) x UE_NUM UEs, matching the scenario's own math.
EXPECTED_RECORDS = 27 * UE_NUM


def main() -> int:
    if not BINARY.exists():
        print(f"FAIL: {BINARY} not built. Run ./scripts/setup_ns3.sh first.", file=sys.stderr)
        return 1

    settings = load_settings()
    source = ZmqTelemetrySource(settings.telemetry.zmq)

    proc = subprocess.Popen(
        [
            str(BINARY),
            f"--ueNum={UE_NUM}",
            f"--simTime={SIM_TIME}",
            f"--telemetryIntervalMs={TELEMETRY_INTERVAL_MS}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    records: list[dict] = []
    try:
        for record in source.records():
            records.append(record)
            if len(records) >= EXPECTED_RECORDS:
                break
    finally:
        source.close()
        proc.wait(timeout=30)

    sim_output = proc.stdout.read() if proc.stdout else ""
    print(sim_output.strip())

    if proc.returncode != 0:
        print(f"FAIL: nr_5g_telemetry_sim exited {proc.returncode}", file=sys.stderr)
        return 1
    if len(records) != EXPECTED_RECORDS:
        print(
            f"FAIL: received {len(records)} records, expected exactly {EXPECTED_RECORDS}",
            file=sys.stderr,
        )
        return 1
    sources = {r.get("source") for r in records}
    if sources != {"NS3_5G_LENA"}:
        print(f"FAIL: unexpected source label(s): {sources}", file=sys.stderr)
        return 1

    preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)
    result = preprocessor.process_batch(records)
    if result.quarantined:
        print(f"FAIL: {len(result.quarantined)} real records were quarantined:", file=sys.stderr)
        for q in result.quarantined[:5]:
            print(f"  reason={q.reason} raw={q.raw}", file=sys.stderr)
        return 1
    if len(result.clean_records) != EXPECTED_RECORDS:
        print(
            f"FAIL: only {len(result.clean_records)}/{EXPECTED_RECORDS} records cleaned",
            file=sys.stderr,
        )
        return 1

    windows = preprocessor.build_feature_windows(result.clean_records)
    if len(windows) != UE_NUM:
        print(f"FAIL: expected {UE_NUM} feature windows (one per UE), got {len(windows)}", file=sys.stderr)
        return 1

    print(
        f"\nOK: {len(records)} real SOURCE=NS3_5G_LENA records from the compiled binary -> "
        f"real ZeroMQ transport -> real TelemetryPreprocessor -> {len(result.clean_records)} clean "
        f"records, 0 quarantined, {len(windows)} feature windows built."
    )
    return 0


if __name__ == "__main__":
    start = time.monotonic()
    code = main()
    print(f"(validate_e2e.py finished in {time.monotonic() - start:.1f}s)")
    sys.exit(code)
