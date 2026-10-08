# Pre-OAI-migration runtime data (archived, not deleted)

Real D1 state, lifecycle records, maintenance reports, and demo-script output genuinely produced
by this project's NS-3/mock-telemetry era, before the OAI migration (prompt2.md). Archived here —
not deleted — per the migration's own "provide a migration for existing local data, or start a
fresh store and archive the old one" instruction: the NS-3-era D1 schema (fixed `ue_id`/`cell_id`/
`throughput_mbps`/... columns) is structurally incompatible with the new OAI schema (dynamic
`run_id`/`session_id`/lagged-feature/target-dict shape), so a field-by-field migration would mean
inventing OAI-shaped values for data that was never OAI telemetry in the first place — exactly the
kind of fabrication this project's own conventions prohibit. Starting the live D1 store fresh,
with the old store's real history preserved here for reference, was the correct call.

None of this is read by any code path — it is historical reference only, gitignored like every
other runtime artifact.

## Known gap: the main production `data/models/` tree was not preserved (phase13 §8 item 5)

This directory preserves every DEMO-SCRIPT-scoped model tree that existed at migration time
(`demo_expand_scope/models/`, `demo_regeneration/models/`, `e2e_ns3_demo/models/`), but the
actual top-level, live-production `data/models/{throughput,latency,packet_loss,prb_utilization,
jitter}/` + `data/models/registry_index.json` tree — the one `src/main.py`'s real
`ContinuousOrchestrator` served predictions from before this migration — was genuinely NOT copied
into this archive before the OAI migration overwrote/replaced it with the new 7-component
registry. This was discovered, not assumed, while auditing this archive for phase13 §8's own
"fix the open issues" pass: no top-level `models/` subdirectory exists here, only the three
demo/e2e-scoped ones listed above.

**These specific artifacts cannot be recovered.** They were runtime-generated (gitignored,
never committed) and the NS-3-era live D1 state + lifecycle records that WERE preserved
(`d1_current_state.parquet`, `d1_history.parquet`, `lifecycle_records.jsonl` at this directory's
own top level) do not contain the trained model weights themselves — only the telemetry/
adaptation-event records that used to reference them. No other action is taken here per the
explicit instruction this entry responds to; this is a transparent record of a genuine, now
permanent data-loss gap from before this project adopted its current archive-don't-delete
discipline, not something to paper over or quietly work around.
