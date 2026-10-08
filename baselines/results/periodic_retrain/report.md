# Baseline run report — mode `periodic_retrain`

- Replayed: OAI run(s) [8] (TEST split, speed=0)
- Seeds: [1]
- Model versions at start: {'dl_bler': 'dl_bler-v1', 'ul_bler': 'ul_bler-v1', 'dl_prb': 'dl_prb-v1', 'ul_prb': 'ul_prb-v1', 'dl_throughput': 'dl_throughput-v1', 'ul_throughput': 'ul_throughput-v1', 'dl_rlc_delay': 'dl_rlc_delay-v1'}
- Fidelity config: threshold=0.5, min_consecutive=3, rolling_window_length=200

## Per-component summary

| component | mean actual | mean predicted | RMSE | MAE | mean F_c |
|---|---|---|---|---|---|
| dl_bler | 0.0611 | 0.0615 | 0.0017 | 0.0008 | 0.8126 |
| dl_prb | 19.2771 | 19.2928 | 1.2975 | 0.6952 | 0.8945 |
| dl_rlc_delay | 0.2572 | 0.5384 | 0.6579 | 0.2811 | 0.7876 |
| dl_throughput | 0.2200 | 0.2209 | 0.0149 | 0.0068 | 0.8048 |
| ul_bler | 0.0075 | 0.0074 | 0.0005 | 0.0002 | 0.7500 |
| ul_prb | 0.9126 | 0.9108 | 0.0501 | 0.0288 | 0.8897 |
| ul_throughput | 0.0144 | 0.0145 | 0.0003 | 0.0002 | 0.8905 |

## Adaptation Events

### Event 1 — dl_bler — 2026-10-08T08:44:58.442000+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=dl_bler-v2
- **Verification**: None -> None => REJECT (Deterministic gate decision: REJECT for 'dl_bler' version 'dl_bler-v2' (adaptation_type='recalibrate'). fidelity_before=None, fidelity_after=None. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/dl_bler/dl_bler-v2.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='dl_bler'/OUTPUT_FIELD='dl_bler_pred' vs candidate component='dl_bler'/output_field='dl_bler_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version dl_bler-v1 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore before and/or after is not computable (insufficient rolling history for one or both) — cannot verify improvement, fail-safe REJECT))
- version dl_bler-v1 -> dl_bler-v1, duration=9.95s

### Event 2 — ul_bler — 2026-10-08T08:45:08.390167+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=ul_bler-v2
- **Verification**: None -> None => REJECT (Deterministic gate decision: REJECT for 'ul_bler' version 'ul_bler-v2' (adaptation_type='recalibrate'). fidelity_before=None, fidelity_after=None. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/ul_bler/ul_bler-v2.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='ul_bler'/OUTPUT_FIELD='ul_bler_pred' vs candidate component='ul_bler'/output_field='ul_bler_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version ul_bler-v1 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore before and/or after is not computable (insufficient rolling history for one or both) — cannot verify improvement, fail-safe REJECT))
- version ul_bler-v1 -> ul_bler-v1, duration=11.14s

### Event 3 — dl_prb — 2026-10-08T08:45:19.534838+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=dl_prb-v2
- **Verification**: None -> None => REJECT (Deterministic gate decision: REJECT for 'dl_prb' version 'dl_prb-v2' (adaptation_type='recalibrate'). fidelity_before=None, fidelity_after=None. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/dl_prb/dl_prb-v2.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='dl_prb'/OUTPUT_FIELD='dl_prb_pred' vs candidate component='dl_prb'/output_field='dl_prb_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version dl_prb-v1 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore before and/or after is not computable (insufficient rolling history for one or both) — cannot verify improvement, fail-safe REJECT))
- version dl_prb-v1 -> dl_prb-v1, duration=15.80s

### Event 4 — ul_prb — 2026-10-08T08:45:35.331703+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=ul_prb-v2
- **Verification**: None -> None => REJECT (Deterministic gate decision: REJECT for 'ul_prb' version 'ul_prb-v2' (adaptation_type='recalibrate'). fidelity_before=None, fidelity_after=None. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/ul_prb/ul_prb-v2.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='ul_prb'/OUTPUT_FIELD='ul_prb_pred' vs candidate component='ul_prb'/output_field='ul_prb_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version ul_prb-v1 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore before and/or after is not computable (insufficient rolling history for one or both) — cannot verify improvement, fail-safe REJECT))
- version ul_prb-v1 -> ul_prb-v1, duration=9.45s

### Event 5 — dl_throughput — 2026-10-08T08:45:44.779448+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=dl_throughput-v2
- **Verification**: None -> None => REJECT (Deterministic gate decision: REJECT for 'dl_throughput' version 'dl_throughput-v2' (adaptation_type='recalibrate'). fidelity_before=None, fidelity_after=None. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/dl_throughput/dl_throughput-v2.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='dl_throughput'/OUTPUT_FIELD='dl_throughput_pred' vs candidate component='dl_throughput'/output_field='dl_throughput_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version dl_throughput-v1 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore before and/or after is not computable (insufficient rolling history for one or both) — cannot verify improvement, fail-safe REJECT))
- version dl_throughput-v1 -> dl_throughput-v1, duration=15.44s

### Event 6 — ul_throughput — 2026-10-08T08:46:00.218729+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=ul_throughput-v2
- **Verification**: None -> None => REJECT (Deterministic gate decision: REJECT for 'ul_throughput' version 'ul_throughput-v2' (adaptation_type='recalibrate'). fidelity_before=None, fidelity_after=None. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/ul_throughput/ul_throughput-v2.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='ul_throughput'/OUTPUT_FIELD='ul_throughput_pred' vs candidate component='ul_throughput'/output_field='ul_throughput_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version ul_throughput-v1 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore before and/or after is not computable (insufficient rolling history for one or both) — cannot verify improvement, fail-safe REJECT))
- version ul_throughput-v1 -> ul_throughput-v1, duration=12.43s

### Event 7 — dl_rlc_delay — 2026-10-08T08:46:12.647981+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=dl_rlc_delay-v2
- **Verification**: None -> None => REJECT (Deterministic gate decision: REJECT for 'dl_rlc_delay' version 'dl_rlc_delay-v2' (adaptation_type='recalibrate'). fidelity_before=None, fidelity_after=None. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/dl_rlc_delay/dl_rlc_delay-v2.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='dl_rlc_delay'/OUTPUT_FIELD='dl_rlc_delay_pred' vs candidate component='dl_rlc_delay'/output_field='dl_rlc_delay_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version dl_rlc_delay-v1 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore before and/or after is not computable (insufficient rolling history for one or both) — cannot verify improvement, fail-safe REJECT))
- version dl_rlc_delay-v1 -> dl_rlc_delay-v1, duration=9.22s

### Event 8 — dl_bler — 2026-10-08T08:46:22.702747+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=0.750413369385543
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=dl_bler-v3
- **Verification**: 0.750934251029511 -> 0.8252177939082297 => ACCEPT (Deterministic gate decision: ACCEPT for 'dl_bler' version 'dl_bler-v3' (adaptation_type='recalibrate'). fidelity_before=0.750934251029511, fidelity_after=0.8252177939082297. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/dl_bler/dl_bler-v3.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='dl_bler'/OUTPUT_FIELD='dl_bler_pred' vs candidate component='dl_bler'/output_field='dl_bler_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version dl_bler-v1 evaluated on the same held-out set); fidelity_improved=PASS (FidelityScore_new (0.8252) > FidelityScore_old (0.7509) + verification_delta (0.01)))
- version dl_bler-v1 -> dl_bler-v3, duration=11.96s

### Event 9 — ul_bler — 2026-10-08T08:46:34.664331+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=0.7500177259434397
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=ul_bler-v3
- **Verification**: 0.7501567203221264 -> 0.7521847142254245 => REJECT (Deterministic gate decision: REJECT for 'ul_bler' version 'ul_bler-v3' (adaptation_type='recalibrate'). fidelity_before=0.7501567203221264, fidelity_after=0.7521847142254245. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/ul_bler/ul_bler-v3.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='ul_bler'/OUTPUT_FIELD='ul_bler_pred' vs candidate component='ul_bler'/output_field='ul_bler_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version ul_bler-v1 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore_new (0.7522) is NOT > FidelityScore_old (0.7502) + verification_delta (0.01)))
- version ul_bler-v1 -> ul_bler-v1, duration=13.36s

### Event 10 — dl_prb — 2026-10-08T08:46:48.022892+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=0.7556438987622006
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=dl_prb-v3
- **Verification**: 0.7499975343251859 -> 0.9243369427505485 => ACCEPT (Deterministic gate decision: ACCEPT for 'dl_prb' version 'dl_prb-v3' (adaptation_type='recalibrate'). fidelity_before=0.7499975343251859, fidelity_after=0.9243369427505485. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/dl_prb/dl_prb-v3.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='dl_prb'/OUTPUT_FIELD='dl_prb_pred' vs candidate component='dl_prb'/output_field='dl_prb_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version dl_prb-v1 evaluated on the same held-out set); fidelity_improved=PASS (FidelityScore_new (0.9243) > FidelityScore_old (0.7500) + verification_delta (0.01)))
- version dl_prb-v1 -> dl_prb-v3, duration=9.36s

### Event 11 — ul_prb — 2026-10-08T08:46:57.378263+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=0.999024768929501
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=ul_prb-v3
- **Verification**: 0.9919704478461338 -> 0.9697553280662201 => REJECT (Deterministic gate decision: REJECT for 'ul_prb' version 'ul_prb-v3' (adaptation_type='recalibrate'). fidelity_before=0.9919704478461338, fidelity_after=0.9697553280662201. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/ul_prb/ul_prb-v3.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='ul_prb'/OUTPUT_FIELD='ul_prb_pred' vs candidate component='ul_prb'/output_field='ul_prb_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version ul_prb-v1 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore_new (0.9698) is NOT > FidelityScore_old (0.9920) + verification_delta (0.01)))
- version ul_prb-v1 -> ul_prb-v1, duration=12.14s

### Event 12 — dl_throughput — 2026-10-08T08:47:09.522936+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=0.7638167280546575
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=dl_throughput-v3
- **Verification**: 0.7716901728630785 -> 0.771991313369906 => REJECT (Deterministic gate decision: REJECT for 'dl_throughput' version 'dl_throughput-v3' (adaptation_type='recalibrate'). fidelity_before=0.7716901728630785, fidelity_after=0.771991313369906. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/dl_throughput/dl_throughput-v3.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='dl_throughput'/OUTPUT_FIELD='dl_throughput_pred' vs candidate component='dl_throughput'/output_field='dl_throughput_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version dl_throughput-v1 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore_new (0.7720) is NOT > FidelityScore_old (0.7717) + verification_delta (0.01)))
- version dl_throughput-v1 -> dl_throughput-v1, duration=12.29s

### Event 13 — ul_throughput — 2026-10-08T08:47:21.812814+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=0.9984608067928029
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=ul_throughput-v3
- **Verification**: 0.9811673065786095 -> 0.9756726472137482 => REJECT (Deterministic gate decision: REJECT for 'ul_throughput' version 'ul_throughput-v3' (adaptation_type='recalibrate'). fidelity_before=0.9811673065786095, fidelity_after=0.9756726472137482. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/ul_throughput/ul_throughput-v3.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='ul_throughput'/OUTPUT_FIELD='ul_throughput_pred' vs candidate component='ul_throughput'/output_field='ul_throughput_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version ul_throughput-v1 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore_new (0.9757) is NOT > FidelityScore_old (0.9812) + verification_delta (0.01)))
- version ul_throughput-v1 -> ul_throughput-v1, duration=10.31s

### Event 14 — dl_rlc_delay — 2026-10-08T08:47:32.123135+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=0.7577836840502997
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=dl_rlc_delay-v3
- **Verification**: 0.7574016430991197 -> 0.793615828288673 => ACCEPT (Deterministic gate decision: ACCEPT for 'dl_rlc_delay' version 'dl_rlc_delay-v3' (adaptation_type='recalibrate'). fidelity_before=0.7574016430991197, fidelity_after=0.793615828288673. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/dl_rlc_delay/dl_rlc_delay-v3.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='dl_rlc_delay'/OUTPUT_FIELD='dl_rlc_delay_pred' vs candidate component='dl_rlc_delay'/output_field='dl_rlc_delay_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version dl_rlc_delay-v1 evaluated on the same held-out set); fidelity_improved=PASS (FidelityScore_new (0.7936) > FidelityScore_old (0.7574) + verification_delta (0.01)))
- version dl_rlc_delay-v1 -> dl_rlc_delay-v3, duration=14.78s

### Event 15 — dl_bler — 2026-10-08T08:47:47.724273+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=0.8250137829277755
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=dl_bler-v4
- **Verification**: 0.8251734842580631 -> 0.8251734842580631 => REJECT (Deterministic gate decision: REJECT for 'dl_bler' version 'dl_bler-v4' (adaptation_type='recalibrate'). fidelity_before=0.8251734842580631, fidelity_after=0.8251734842580631. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/dl_bler/dl_bler-v4.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='dl_bler'/OUTPUT_FIELD='dl_bler_pred' vs candidate component='dl_bler'/output_field='dl_bler_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version dl_bler-v3 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore_new (0.8252) is NOT > FidelityScore_old (0.8252) + verification_delta (0.01)))
- version dl_bler-v3 -> dl_bler-v3, duration=11.17s

### Event 16 — ul_bler — 2026-10-08T08:47:58.895256+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=0.7500177259172143
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=ul_bler-v4
- **Verification**: 0.7501567202959011 -> 0.7521847141991993 => REJECT (Deterministic gate decision: REJECT for 'ul_bler' version 'ul_bler-v4' (adaptation_type='recalibrate'). fidelity_before=0.7501567202959011, fidelity_after=0.7521847141991993. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/ul_bler/ul_bler-v4.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='ul_bler'/OUTPUT_FIELD='ul_bler_pred' vs candidate component='ul_bler'/output_field='ul_bler_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version ul_bler-v1 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore_new (0.7522) is NOT > FidelityScore_old (0.7502) + verification_delta (0.01)))
- version ul_bler-v1 -> ul_bler-v1, duration=9.48s

### Event 17 — dl_prb — 2026-10-08T08:48:08.374678+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=0.9255614718029148
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=dl_prb-v4
- **Verification**: 0.9242606433617662 -> 0.9242606433617662 => REJECT (Deterministic gate decision: REJECT for 'dl_prb' version 'dl_prb-v4' (adaptation_type='recalibrate'). fidelity_before=0.9242606433617662, fidelity_after=0.9242606433617662. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/dl_prb/dl_prb-v4.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='dl_prb'/OUTPUT_FIELD='dl_prb_pred' vs candidate component='dl_prb'/output_field='dl_prb_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version dl_prb-v3 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore_new (0.9243) is NOT > FidelityScore_old (0.9243) + verification_delta (0.01)))
- version dl_prb-v3 -> dl_prb-v3, duration=11.50s

### Event 18 — ul_prb — 2026-10-08T08:48:19.875742+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=0.9939767345342616
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=ul_prb-v4
- **Verification**: 0.9964957635233116 -> 0.9879756171594203 => REJECT (Deterministic gate decision: REJECT for 'ul_prb' version 'ul_prb-v4' (adaptation_type='recalibrate'). fidelity_before=0.9964957635233116, fidelity_after=0.9879756171594203. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/ul_prb/ul_prb-v4.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='ul_prb'/OUTPUT_FIELD='ul_prb_pred' vs candidate component='ul_prb'/output_field='ul_prb_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version ul_prb-v1 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore_new (0.9880) is NOT > FidelityScore_old (0.9965) + verification_delta (0.01)))
- version ul_prb-v1 -> ul_prb-v1, duration=9.79s

### Event 19 — dl_throughput — 2026-10-08T08:48:29.669228+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=0.7791537606299378
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=dl_throughput-v4
- **Verification**: 0.7716901728630785 -> 0.771991313369906 => REJECT (Deterministic gate decision: REJECT for 'dl_throughput' version 'dl_throughput-v4' (adaptation_type='recalibrate'). fidelity_before=0.7716901728630785, fidelity_after=0.771991313369906. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/dl_throughput/dl_throughput-v4.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='dl_throughput'/OUTPUT_FIELD='dl_throughput_pred' vs candidate component='dl_throughput'/output_field='dl_throughput_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version dl_throughput-v1 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore_new (0.7720) is NOT > FidelityScore_old (0.7717) + verification_delta (0.01)))
- version dl_throughput-v1 -> dl_throughput-v1, duration=11.75s

### Event 20 — ul_throughput — 2026-10-08T08:48:41.418244+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=0.974253170496801
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=ul_throughput-v4
- **Verification**: 0.9919903354159357 -> 0.9898207980221527 => REJECT (Deterministic gate decision: REJECT for 'ul_throughput' version 'ul_throughput-v4' (adaptation_type='recalibrate'). fidelity_before=0.9919903354159357, fidelity_after=0.9898207980221527. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/ul_throughput/ul_throughput-v4.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='ul_throughput'/OUTPUT_FIELD='ul_throughput_pred' vs candidate component='ul_throughput'/output_field='ul_throughput_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version ul_throughput-v1 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore_new (0.9898) is NOT > FidelityScore_old (0.9920) + verification_delta (0.01)))
- version ul_throughput-v1 -> ul_throughput-v1, duration=10.16s

### Event 21 — dl_rlc_delay — 2026-10-08T08:48:51.581719+00:00

- **Trigger**: type=external_drift, severity=0.0, fidelity_at_trigger=0.793614568785491
- **Decision**: strategy=recalibrate, source=periodic_schedule, confidence=1.0, llm_or_fallback=None
- **Root cause**: None
- **Reasoning**: None
- **References**: []
- **Candidate**: version=dl_rlc_delay-v4
- **Verification**: 0.793614568785491 -> 0.793614568785491 => REJECT (Deterministic gate decision: REJECT for 'dl_rlc_delay' version 'dl_rlc_delay-v4' (adaptation_type='recalibrate'). fidelity_before=0.793614568785491, fidelity_after=0.793614568785491. Checks: candidate_artifact_present=PASS (artifact at /home/harie06/dt_adaptation/baselines/results/periodic_retrain/models/dl_rlc_delay/dl_rlc_delay-v4.joblib exists and is non-empty); interface_identity_consistent=PASS (factory COMPONENT_NAME='dl_rlc_delay'/OUTPUT_FIELD='dl_rlc_delay_pred' vs candidate component='dl_rlc_delay'/output_field='dl_rlc_delay_pred'); evaluation_ran=PASS (eval_target has 5173 row(s), candidate_predictions has 5173 row(s)); no_invalid_output=PASS (all candidate predictions are finite); production_baseline_evaluable=PASS (production version dl_rlc_delay-v3 evaluated on the same held-out set); fidelity_improved=FAIL (FidelityScore_new (0.7936) is NOT > FidelityScore_old (0.7936) + verification_delta (0.01)))
- version dl_rlc_delay-v3 -> dl_rlc_delay-v3, duration=9.77s
