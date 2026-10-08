# Baseline run report — mode `static`

- Replayed: OAI run(s) [8] (TEST split, speed=0)
- Seeds: [1]
- Model versions at start: {'dl_bler': 'dl_bler-v1', 'ul_bler': 'ul_bler-v1', 'dl_prb': 'dl_prb-v1', 'ul_prb': 'ul_prb-v1', 'dl_throughput': 'dl_throughput-v1', 'ul_throughput': 'ul_throughput-v1', 'dl_rlc_delay': 'dl_rlc_delay-v1'}
- Fidelity config: threshold=0.5, min_consecutive=3, rolling_window_length=200

## Per-component summary

| component | mean actual | mean predicted | RMSE | MAE | mean F_c |
|---|---|---|---|---|---|
| dl_bler | 0.0611 | 0.0614 | 0.0017 | 0.0008 | 0.7506 |
| dl_prb | 19.2771 | 19.3358 | 1.3012 | 0.7381 | 0.7518 |
| dl_rlc_delay | 0.2572 | 0.5390 | 0.6579 | 0.2818 | 0.7574 |
| dl_throughput | 0.2200 | 0.2209 | 0.0149 | 0.0068 | 0.8130 |
| ul_bler | 0.0075 | 0.0074 | 0.0005 | 0.0002 | 0.7500 |
| ul_prb | 0.9126 | 0.9108 | 0.0501 | 0.0288 | 0.8679 |
| ul_throughput | 0.0144 | 0.0145 | 0.0003 | 0.0002 | 0.8690 |

## Adaptation Events

### Event 1 — dl_prb — 2026-10-08T08:43:57.201610+00:00

- **Trigger**: type=external_drift, severity=0.9504636963259353, fidelity_at_trigger=None
- **Decision**: strategy=None, source=static_mode_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static mode: triggers logged, never dispatched

### Event 2 — dl_prb — 2026-10-08T08:43:57.340325+00:00

- **Trigger**: type=external_drift, severity=0.9486494471372439, fidelity_at_trigger=None
- **Decision**: strategy=None, source=static_mode_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static mode: triggers logged, never dispatched

### Event 3 — ul_bler — 2026-10-08T08:43:57.482403+00:00

- **Trigger**: type=external_drift, severity=0.8277025938204418, fidelity_at_trigger=None
- **Decision**: strategy=None, source=static_mode_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static mode: triggers logged, never dispatched

### Event 4 — dl_bler — 2026-10-08T08:43:57.768973+00:00

- **Trigger**: type=external_drift, severity=0.5381433132192782, fidelity_at_trigger=None
- **Decision**: strategy=None, source=static_mode_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static mode: triggers logged, never dispatched

### Event 5 — dl_prb — 2026-10-08T08:43:57.906285+00:00

- **Trigger**: type=external_drift, severity=0.303194829291645, fidelity_at_trigger=None
- **Decision**: strategy=None, source=static_mode_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static mode: triggers logged, never dispatched

### Event 6 — dl_bler — 2026-10-08T08:43:58.046002+00:00

- **Trigger**: type=external_drift, severity=0.13404169724716475, fidelity_at_trigger=None
- **Decision**: strategy=None, source=static_mode_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static mode: triggers logged, never dispatched

### Event 7 — ul_bler — 2026-10-08T08:43:58.209303+00:00

- **Trigger**: type=external_drift, severity=0.2623133404418495, fidelity_at_trigger=None
- **Decision**: strategy=None, source=static_mode_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static mode: triggers logged, never dispatched

### Event 8 — ul_throughput — 2026-10-08T08:43:58.346052+00:00

- **Trigger**: type=external_drift, severity=0.2804087579860399, fidelity_at_trigger=None
- **Decision**: strategy=None, source=static_mode_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static mode: triggers logged, never dispatched

### Event 9 — dl_throughput — 2026-10-08T08:43:58.486613+00:00

- **Trigger**: type=external_drift, severity=0.9616571936637868, fidelity_at_trigger=None
- **Decision**: strategy=None, source=static_mode_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static mode: triggers logged, never dispatched

### Event 10 — ul_bler — 2026-10-08T08:43:58.626250+00:00

- **Trigger**: type=external_drift, severity=0.5412268555474342, fidelity_at_trigger=None
- **Decision**: strategy=None, source=static_mode_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static mode: triggers logged, never dispatched

### Event 11 — dl_bler — 2026-10-08T08:43:58.770232+00:00

- **Trigger**: type=external_drift, severity=0.9699254132161326, fidelity_at_trigger=None
- **Decision**: strategy=None, source=static_mode_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static mode: triggers logged, never dispatched

### Event 12 — ul_throughput — 2026-10-08T08:43:58.905709+00:00

- **Trigger**: type=external_drift, severity=0.11586561247077032, fidelity_at_trigger=None
- **Decision**: strategy=None, source=static_mode_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static mode: triggers logged, never dispatched

### Event 13 — dl_prb — 2026-10-08T08:43:59.053549+00:00

- **Trigger**: type=external_drift, severity=0.6130033010530405, fidelity_at_trigger=None
- **Decision**: strategy=None, source=static_mode_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static mode: triggers logged, never dispatched

### Event 14 — dl_bler — 2026-10-08T08:43:59.190535+00:00

- **Trigger**: type=external_drift, severity=0.03959287666420286, fidelity_at_trigger=None
- **Decision**: strategy=None, source=static_mode_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static mode: triggers logged, never dispatched
