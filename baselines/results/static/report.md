# Baseline run report — mode `static`

- Replayed: OAI run(s) [8] (TEST split, speed=0)
- Seeds: [1]
- Model versions at start: {'dl_bler': 'dl_bler-v1', 'ul_bler': 'ul_bler-v1', 'dl_prb': 'dl_prb-v1', 'ul_prb': 'ul_prb-v1', 'dl_throughput': 'dl_throughput-v1', 'ul_throughput': 'ul_throughput-v1', 'dl_rlc_delay': 'dl_rlc_delay-v1'}
- Fidelity config: threshold=0.5, min_consecutive=3, rolling_window_length=200

## Per-component summary

| component | mean actual | mean predicted | RMSE | MAE | mean F_c |
|---|---|---|---|---|---|
| dl_bler | 0.0541 | 0.0539 | 0.0007 | 0.0003 | 0.7436 |
| dl_prb | 17.8606 | 18.0820 | 0.6291 | 0.3694 | 0.7470 |
| dl_rlc_delay | 0.4799 | 0.4988 | 0.4722 | 0.1590 | 0.7354 |
| dl_throughput | 0.2054 | 0.2060 | 0.0074 | 0.0031 | 0.7662 |
| ul_bler | 0.0075 | 0.0074 | 0.0004 | 0.0001 | 0.7113 |
| ul_prb | 0.9265 | 0.9300 | 0.0710 | 0.0394 | 0.9051 |
| ul_throughput | 0.0147 | 0.0147 | 0.0010 | 0.0004 | 0.9571 |

## Adaptation Events

### Event 1 — dl_bler — 2026-09-28T15:52:19+00:00

- **Trigger**: type=fidelity_degradation, severity=0.28922933461155886, fidelity_at_trigger=0.35538533269422057
- **Decision**: strategy=None, source=static_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static: triggers are logged, never dispatched

### Event 2 — ul_bler — 2026-09-28T15:52:19+00:00

- **Trigger**: type=fidelity_degradation, severity=0.3922715496916913, fidelity_at_trigger=0.30386422515415434
- **Decision**: strategy=None, source=static_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static: triggers are logged, never dispatched

### Event 3 — dl_rlc_delay — 2026-09-28T15:53:19+00:00

- **Trigger**: type=fidelity_degradation, severity=0.3087818198323935, fidelity_at_trigger=0.34560909008380325
- **Decision**: strategy=None, source=static_ignored, confidence=None, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version None -> None, duration=0.00s
- **Skipped**: static: triggers are logged, never dispatched
