# ns3_sim — NS-3 / 5G-LENA Simulator Integration (Module 1)

This directory holds the project-authored NS-3 simulator side of the Digital Twin: the custom
5G-LENA scenario that produces ground-truth network behaviour and streams it out as structured
telemetry (prompt.md §0.2, §0.3, §7).

## What is and isn't tracked in git

- **Tracked**: this README, our own scenario source (`*.cc`), any helper scripts here.
- **Not tracked** (`.gitignore`'d): `ns3_sim/ns-3-dev/`, the vendored full ns-3 + `contrib/nr`
  source trees. Those are an external build dependency (like a compiler), fetched at pinned
  versions by `scripts/setup_ns3.sh` — not code we authored. Re-run that script after a fresh
  clone of this repository to reconstruct them.

## Pinned versions

Looked up from the official 5G-LENA compatibility table
(`https://gitlab.com/cttc-lena/nr/-/raw/master/README.md`, "ns-3 + nr installation" section) on
2026-09-06, matched against ns-3 release tags on `gitlab.com/nsnam/ns-3-dev`:

| Component | Version | Git ref |
|---|---|---|
| ns-3 | ns-3.48 | tag `ns-3.48` on `https://gitlab.com/nsnam/ns-3-dev.git` |
| 5G-LENA (`nr` module) | 5g-lena-v5.1.y | tag `v5.1` on `https://gitlab.com/cttc-lena/nr.git`, cloned into `ns-3-dev/contrib/nr` |

This is the newest matched pair in the compatibility table as of the lookup date. `nr` is added
under `contrib/` (the standard ns-3 out-of-tree module location), not merged into ns-3 core.

## Environment inspection (this machine)

Checked directly (prompt.md §0.2 step 1-3, §59):

| Requirement | Found | ns-3.48 minimum | Status |
|---|---|---|---|
| g++ | 15.2.0 | >= 11.1 | OK |
| cmake | 4.2.3 | >= 3.25 | OK |
| ninja | 1.13.2 | (either make or ninja) | OK |
| python3 | 3.14.4 | >= 3.10 | OK |
| git | 2.53.0 | any | OK |

Core ns-3 has no unmet prerequisites on this machine — no root access is needed to build it.

### `nr` module (5G-LENA) prerequisites

From the nr README "ns-3 + nr prerequisites" section:

| Package | Purpose | Status found | Required or optional |
|---|---|---|---|
| `libc6-dev` | `semaphore.h` | already installed | required |
| `sqlite3`, `libsqlite3-dev` | optional example programs (`lena-lte-comparison`, etc.) | `libsqlite3-dev` present, `sqlite3` CLI missing | optional |
| `libeigen3-dev` | optional MIMO features | missing | optional |

None of these block building the `nr` module or running the throughput/latency/loss-relevant
examples we need for telemetry extraction — only MIMO-specific features and a few optional
example binaries are affected if left uninstalled.

### Our own telemetry exporter prerequisites (not an ns-3/nr requirement — our code only)

Module 1's ZeroMQ telemetry producer (the `TelemetryPublisher` class inside
`nr_5g_telemetry_sim.cc` — no separate exporter file was needed) links against `libzmq` from
C++, which required packages this environment did not have pre-installed and that only a user
with sudo could add:

| Package | Purpose |
|---|---|
| `libzmq3-dev`, `libzmq5` | ZeroMQ C library + headers |
| `cppzmq-dev` | header-only C++ ZeroMQ bindings (`zmq.hpp`) used by our exporter |

Exact command that failed autonomously (no cached sudo credentials in this non-interactive
session — prompt.md §0.4 step 7-8):

```
$ sudo -n apt-get install -y libzmq3-dev cppzmq-dev
sudo: interactive authentication is required
```

The user was asked to run `sudo apt-get update && sudo apt-get install -y sqlite3 libsqlite3-dev
libeigen3-dev libzmq3-dev libzmq5 cppzmq-dev` interactively. `scripts/setup_ns3.sh` checks for
these at build time and warns (not fails) if still absent, since they are not required to build
ns-3 core or the `nr` module itself.

## `nr_5g_telemetry_sim.cc` — our telemetry-producing scenario

The project-authored scenario (this directory's `nr_5g_telemetry_sim.cc`) is a 1 gNB + 6 UE
5G-LENA topology (UMa/`ThreeGpp` channel, shadow fading enabled, `RandomWalk2dMobilityModel`,
CBR UDP traffic) that publishes real per-UE telemetry over a ZeroMQ PUB socket every
`telemetryIntervalMs` (default 200ms). Every field is genuine simulator output — no field is
synthesized in the C++ layer:

| Telemetry field | Source (real NS-3/5G-LENA trace or computation) |
|---|---|
| `sinr_db` | `NrUePhy::"DlDataSinr"` trace (linear → dB) |
| `rsrp_dbm` / `rsrq_db` | `NrUePhy::"ReportUeMeasurements"` trace, serving-cell entries |
| `prb_utilization_pct` | `NrGnbPhy::"SlotDataStats"` + `"RBDataStats"` traces combined (see note below) |
| `throughput_mbps` / `offered_load_mbps` / `latency_ms` / `jitter_ms` / `packet_loss_pct` | `FlowMonitor` per-flow cumulative stats, polled every tick and converted to per-interval deltas |
| `ue_count` | live UE count in the scenario |
| `ue_position_x/y` / `ue_speed_mps` | the mobility model's real current position/velocity |

**Note on `prb_utilization_pct`**: it is deliberately computed from two separate PHY traces, not
one. `"SlotDataStats"`'s own `dataReg` field is an RBG×symbol product, not directly comparable to
its own `availRb` field (a pure frequency-axis count) — comparing them directly (an early version
of this scenario's own bug) saturates the ratio at 1.0 regardless of real load. The fix adds
`"RBDataStats"`, which gives a literal per-symbol used-RB bitmap in the same frequency-only unit
as `availRb`; the ratio is computed from `usedRbSymbols / (availRb * dataSymCount)`.

### Build wiring

`scripts/setup_ns3.sh` regenerates `ns-3-dev/scratch/nr_5g_telemetry_sim/` on every run: a
symlink to this directory's tracked `.cc` file, plus a generated `CMakeLists.txt` using
`build_exec(... LIBRARIES_TO_LINK ... zmq ...)` — the default top-level `scratch/CMakeLists.txt`
glob doesn't support linking extra libraries like libzmq, hence the subdirectory. The real ninja
target name (`scratch_nr_5g_telemetry_sim`) differs from what `./ns3 build <name>` expects; build
it directly from the build's `cmake-cache` directory:

```
cd ns3_sim/ns-3-dev/cmake-cache && ninja -j2 scratch_nr_5g_telemetry_sim
```

The resulting binary lands at
`ns3_sim/ns-3-dev/build/scratch/nr_5g_telemetry_sim/ns3.48-nr_5g_telemetry_sim`. Run it directly:

```
./ns3_sim/ns-3-dev/build/scratch/nr_5g_telemetry_sim/ns3.48-nr_5g_telemetry_sim \
    --ueNum=6 --simTime=20s --telemetryIntervalMs=200ms --zmqEndpoint=tcp://127.0.0.1:5556
```

### End-to-end validation

`ns3_sim/validate_e2e.py` is a permanent, repo-tracked validation script (not part of `pytest`,
since it needs the built ns-3 tree and takes real wall-clock simulation time). It launches the
compiled binary, connects the real `src/telemetry/zmq_source.py:ZmqTelemetrySource` (no mocking
on either side), and runs every received record through the real `TelemetryPreprocessor`:

```
python ns3_sim/validate_e2e.py
```

Last confirmed passing (2026-09-07): 162/162 real `SOURCE=NS3_5G_LENA` records received, 0
quarantined, 6 feature windows built (one per UE).

## Real vs mock telemetry (prompt.md §0.3, §49)

- `SOURCE=NS3_5G_LENA` — emitted only by the actual compiled-and-executed NS-3/5G-LENA scenario
  in this directory, over the ZeroMQ transport once built.
- `SOURCE=MOCK` — emitted only by `src/telemetry/mock_source.py`, a pure-Python generator used
  for development/testing when the real simulator is unavailable or not yet running. It is never
  reported as, or allowed to masquerade as, NS-3 output.

## Build / run

```
./scripts/setup_ns3.sh          # clone (if needed) + configure + build ns-3 + nr
cd ns3_sim/ns-3-dev
./ns3 run <example-name>        # e.g. an nr example, to validate the toolchain
```

## Status

See `IMPLEMENTATION_STATUS.md` (Module 1) for current build/validation status — updated as this
integration progresses from "toolchain builds" to "our scenario emits real telemetry over ZeroMQ
and is consumed by `src/telemetry/zmq_source.py`".
