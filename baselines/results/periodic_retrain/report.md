# Baseline run report — mode `periodic_retrain`

- Replayed: OAI run(s) [8] (TEST split, speed=0)
- Seeds: [1]
- Model versions at start: {'dl_bler': 'dl_bler-v7', 'ul_bler': 'ul_bler-v7', 'dl_prb': 'dl_prb-v7', 'ul_prb': 'ul_prb-v7', 'dl_throughput': 'dl_throughput-v7', 'ul_throughput': 'ul_throughput-v7', 'dl_rlc_delay': 'dl_rlc_delay-v7'}
- Fidelity config: threshold=0.5, min_consecutive=3, rolling_window_length=200

## Per-component summary

| component | mean actual | mean predicted | RMSE | MAE | mean F_c |
|---|---|---|---|---|---|
| dl_bler | 0.0541 | 0.0539 | 0.0007 | 0.0003 | 0.7621 |
| dl_prb | 17.8606 | 18.1373 | 0.6218 | 0.4198 | 0.7482 |
| dl_rlc_delay | 0.4799 | 0.4913 | 0.4744 | 0.1621 | 0.7226 |
| dl_throughput | 0.2054 | 0.2061 | 0.0077 | 0.0035 | 0.8172 |
| ul_bler | 0.0075 | 0.0074 | 0.0004 | 0.0001 | 0.7513 |
| ul_prb | 0.9265 | 0.9214 | 0.0680 | 0.0308 | 0.9702 |
| ul_throughput | 0.0147 | 0.0147 | 0.0010 | 0.0004 | 0.9071 |

## Adaptation Events

### Event 1 — dl_bler — 2026-09-28T15:48:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_bler-v2
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_bler-v1 -> dl_bler-v2, duration=2.40s

### Event 2 — ul_bler — 2026-09-28T15:48:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_bler-v2
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_bler-v1 -> ul_bler-v2, duration=2.44s

### Event 3 — dl_prb — 2026-09-28T15:48:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_prb-v2
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_prb-v1 -> dl_prb-v2, duration=2.37s

### Event 4 — ul_prb — 2026-09-28T15:48:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_prb-v2
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_prb-v1 -> ul_prb-v2, duration=3.35s

### Event 5 — dl_throughput — 2026-09-28T15:48:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_throughput-v2
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_throughput-v1 -> dl_throughput-v2, duration=2.43s

### Event 6 — ul_throughput — 2026-09-28T15:48:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_throughput-v2
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_throughput-v1 -> ul_throughput-v2, duration=2.42s

### Event 7 — dl_rlc_delay — 2026-09-28T15:48:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_rlc_delay-v2
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_rlc_delay-v1 -> dl_rlc_delay-v2, duration=5.25s

### Event 8 — dl_bler — 2026-09-28T15:52:19+00:00

- **Trigger**: type=fidelity_degradation, severity=0.22720479402733318, fidelity_at_trigger=0.3863976029863334
- **Decision**: strategy=None, source=periodic_retrain_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: periodic_retrain: triggers are logged, never dispatched

### Event 9 — ul_bler — 2026-09-28T15:52:19+00:00

- **Trigger**: type=fidelity_degradation, severity=0.38148853546352623, fidelity_at_trigger=0.3092557322682369
- **Decision**: strategy=None, source=periodic_retrain_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: periodic_retrain: triggers are logged, never dispatched

### Event 10 — dl_rlc_delay — 2026-09-28T15:53:19+00:00

- **Trigger**: type=fidelity_degradation, severity=0.21305064689722863, fidelity_at_trigger=0.3934746765513857
- **Decision**: strategy=None, source=periodic_retrain_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: periodic_retrain: triggers are logged, never dispatched

### Event 11 — dl_bler — 2026-09-28T15:58:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.6186513130825368
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_bler-v3
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_bler-v2 -> dl_bler-v3, duration=2.84s

### Event 12 — ul_bler — 2026-09-28T15:58:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.8110520660665548
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_bler-v3
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_bler-v2 -> ul_bler-v3, duration=2.52s

### Event 13 — dl_prb — 2026-09-28T15:58:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.7612341034087227
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_prb-v3
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_prb-v2 -> dl_prb-v3, duration=2.45s

### Event 14 — ul_prb — 2026-09-28T15:58:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.9548594156843382
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_prb-v3
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_prb-v2 -> ul_prb-v3, duration=3.00s

### Event 15 — dl_throughput — 2026-09-28T15:58:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.7564427749530536
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_throughput-v3
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_throughput-v2 -> dl_throughput-v3, duration=2.33s

### Event 16 — ul_throughput — 2026-09-28T15:58:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.9999788744892188
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_throughput-v3
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_throughput-v2 -> ul_throughput-v3, duration=2.67s

### Event 17 — dl_rlc_delay — 2026-09-28T15:58:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.7514716609495496
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_rlc_delay-v3
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_rlc_delay-v2 -> dl_rlc_delay-v3, duration=2.46s

### Event 18 — dl_bler — 2026-09-28T16:08:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.7754804781802241
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_bler-v4
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_bler-v3 -> dl_bler-v4, duration=2.62s

### Event 19 — ul_bler — 2026-09-28T16:08:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.8838385233611085
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_bler-v4
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_bler-v3 -> ul_bler-v4, duration=2.59s

### Event 20 — dl_prb — 2026-09-28T16:08:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.7828221308966212
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_prb-v4
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_prb-v3 -> dl_prb-v4, duration=8.90s

### Event 21 — ul_prb — 2026-09-28T16:08:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.9912751671009538
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_prb-v4
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_prb-v3 -> ul_prb-v4, duration=2.83s

### Event 22 — dl_throughput — 2026-09-28T16:08:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.759726391398328
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_throughput-v4
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_throughput-v3 -> dl_throughput-v4, duration=2.89s

### Event 23 — ul_throughput — 2026-09-28T16:08:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.9996797562526413
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_throughput-v4
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_throughput-v3 -> ul_throughput-v4, duration=4.50s

### Event 24 — dl_rlc_delay — 2026-09-28T16:08:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.7499993170758147
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_rlc_delay-v4
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_rlc_delay-v3 -> dl_rlc_delay-v4, duration=3.13s

### Event 25 — dl_bler — 2026-09-28T16:18:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.7805804383613107
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_bler-v5
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_bler-v4 -> dl_bler-v5, duration=2.61s

### Event 26 — ul_bler — 2026-09-28T16:18:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.7508155448657561
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_bler-v5
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_bler-v4 -> ul_bler-v5, duration=3.06s

### Event 27 — dl_prb — 2026-09-28T16:18:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.7612511926487202
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_prb-v5
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_prb-v4 -> dl_prb-v5, duration=2.53s

### Event 28 — ul_prb — 2026-09-28T16:18:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.9998053927580464
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_prb-v5
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_prb-v4 -> ul_prb-v5, duration=3.39s

### Event 29 — dl_throughput — 2026-09-28T16:18:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.8877734405573986
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_throughput-v5
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_throughput-v4 -> dl_throughput-v5, duration=6.14s

### Event 30 — ul_throughput — 2026-09-28T16:18:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.97105503731788
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_throughput-v5
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_throughput-v4 -> ul_throughput-v5, duration=6.73s

### Event 31 — dl_rlc_delay — 2026-09-28T16:18:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.7499999799835831
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_rlc_delay-v5
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_rlc_delay-v4 -> dl_rlc_delay-v5, duration=6.29s

### Event 32 — dl_bler — 2026-09-28T16:28:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.8349909693315819
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_bler-v6
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_bler-v5 -> dl_bler-v6, duration=4.67s

### Event 33 — ul_bler — 2026-09-28T16:28:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.7745275657650709
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_bler-v6
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_bler-v5 -> ul_bler-v6, duration=5.80s

### Event 34 — dl_prb — 2026-09-28T16:28:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.7535323635818726
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_prb-v6
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_prb-v5 -> dl_prb-v6, duration=6.43s

### Event 35 — ul_prb — 2026-09-28T16:28:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.9999386571478344
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_prb-v6
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_prb-v5 -> ul_prb-v6, duration=4.84s

### Event 36 — dl_throughput — 2026-09-28T16:28:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.8790135018676759
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_throughput-v6
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_throughput-v5 -> dl_throughput-v6, duration=3.88s

### Event 37 — ul_throughput — 2026-09-28T16:28:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.9804582872346465
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_throughput-v6
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_throughput-v5 -> ul_throughput-v6, duration=3.50s

### Event 38 — dl_rlc_delay — 2026-09-28T16:28:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.7500000074375586
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_rlc_delay-v6
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_rlc_delay-v5 -> dl_rlc_delay-v6, duration=51.02s

### Event 39 — dl_bler — 2026-09-28T16:38:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.8186293333215455
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_bler-v7
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_bler-v6 -> dl_bler-v7, duration=49.30s

### Event 40 — ul_bler — 2026-09-28T16:38:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.7745298050094817
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_bler-v7
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_bler-v6 -> ul_bler-v7, duration=48.95s

### Event 41 — dl_prb — 2026-09-28T16:38:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.7471522390366834
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_prb-v7
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_prb-v6 -> dl_prb-v7, duration=47.15s

### Event 42 — ul_prb — 2026-09-28T16:38:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.9997861366872828
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_prb-v7
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_prb-v6 -> ul_prb-v7, duration=56.49s

### Event 43 — dl_throughput — 2026-09-28T16:38:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.8976590783887533
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_throughput-v7
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_throughput-v6 -> dl_throughput-v7, duration=51.51s

### Event 44 — ul_throughput — 2026-09-28T16:38:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.9994485794806223
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=ul_throughput-v7
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version ul_throughput-v6 -> ul_throughput-v7, duration=40.41s

### Event 45 — dl_rlc_delay — 2026-09-28T16:38:19+00:00

- **Trigger**: type=scheduled, severity=0.0, fidelity_at_trigger=0.7499992459891881
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=dl_rlc_delay-v7
- **Verification**: None -> None => DEPLOYED_UNGATED (None)
- version dl_rlc_delay-v6 -> dl_rlc_delay-v7, duration=47.49s
