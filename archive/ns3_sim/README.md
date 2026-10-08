# RETIRED — NS-3 / 5G-LENA Simulator Integration (formerly Module 1)

**This directory is retired and is not used anywhere in the current runtime.** It is kept for
historical reference only (prompt2.md: "Migrate the Digital Twin from NS-3 to OAI KPI replay").

As of the OAI migration, the Digital Twin's telemetry source is a replay of real exported OAI RAN
KPI CSVs (`src/telemetry/oai_replay_source.py`, `SOURCE=OAI_RAN_REPLAY`) — see `CLAUDE.md` and
`docs/oai_twin.md` for the current architecture. NS-3/5G-LENA, the ZeroMQ telemetry bridge
(`zmq_source.py`, also archived here), and the mock-telemetry path that stood in for them are no
longer part of the runtime:

- No module under `src/` imports anything from this directory or imports `zmq`
  (`tests/unit/test_no_archive_imports.py` enforces this as a real, permanent test).
- `requirements.txt` no longer lists `pyzmq`.
- `config/settings.yaml` no longer has a `telemetry.zmq` block or a `"zmq"` telemetry source
  option; `config/settings.yaml` validation fails at startup if it ever reappears.
- `scripts/setup_ns3.sh` and `scripts/run_e2e_demo.py`'s NS-3 launch path are removed.

## What used to be here

A real, compiled 5G-LENA scenario (`nr_5g_telemetry_sim.cc`) that produced genuine simulated 5G NR
telemetry and streamed it over a ZeroMQ PUB socket, consumed by `zmq_source.py`
(`SOURCE=NS3_5G_LENA`). See git history (pre-OAI-migration commits) and the original
`CLAUDE.md`/`IMPLEMENTATION_STATUS.md` entries for Module 1 for the full build/validation history
— that content was never a fabrication; it genuinely built and ran — it is simply superseded by
real exported OAI RAN KPI data, which is a strictly higher-fidelity, real-network-measured
telemetry source for this project's purposes.

The vendored `ns-3-dev/` build tree itself (`contrib/nr` 5G-LENA module included) is **not**
reconstructed by any script anymore; `scripts/setup_ns3.sh` (also retired) is kept here only as a
historical reference for how it was once built.
