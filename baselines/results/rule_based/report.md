# Baseline run report — mode `rule_based`

- Replayed: OAI run(s) [8] (TEST split, speed=0)
- Seeds: [1]
- Model versions at start: {'dl_bler': 'dl_bler-v1', 'ul_bler': 'ul_bler-v1', 'dl_prb': 'dl_prb-v1', 'ul_prb': 'ul_prb-v1', 'dl_throughput': 'dl_throughput-v1', 'ul_throughput': 'ul_throughput-v1', 'dl_rlc_delay': 'dl_rlc_delay-v1'}
- Fidelity config: threshold=0.5, min_consecutive=3, rolling_window_length=200

## Per-component summary

| component | mean actual | mean predicted | RMSE | MAE | mean F_c |
|---|---|---|---|---|---|
| dl_bler | 0.0611 | 0.0614 | 0.0017 | 0.0008 | 0.7505 |
| dl_prb | 19.2771 | 19.3358 | 1.3012 | 0.7381 | 0.7560 |
| dl_rlc_delay | 0.2572 | 0.5390 | 0.6579 | 0.2818 | 0.7574 |
| dl_throughput | 0.2200 | 0.2209 | 0.0149 | 0.0068 | 0.8130 |
| ul_bler | 0.0075 | 0.0074 | 0.0005 | 0.0002 | 0.7500 |
| ul_prb | 0.9126 | 0.9108 | 0.0501 | 0.0288 | 0.8679 |
| ul_throughput | 0.0144 | 0.0145 | 0.0003 | 0.0002 | 0.8905 |

## Adaptation Events

### Event 1 — dl_prb — 2026-10-08T10:03:59.995547+00:00

- **Trigger**: type=external_drift, severity=0.9504636963259353, fidelity_at_trigger=None
- **Decision**: strategy=expand_scope, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version dl_prb-v1 -> None, duration=0.00s
- **Error**: ExpandScopeAgent.expand_scope() missing 1 required positional argument: 'example_component_factory'

### Event 2 — dl_prb — 2026-10-08T10:04:00.147721+00:00

- **Trigger**: type=external_drift, severity=0.9486494471372439, fidelity_at_trigger=None
- **Decision**: strategy=expand_scope, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version dl_prb-v1 -> None, duration=0.00s
- **Error**: ExpandScopeAgent.expand_scope() missing 1 required positional argument: 'example_component_factory'

### Event 3 — ul_bler — 2026-10-08T10:04:00.295833+00:00

- **Trigger**: type=external_drift, severity=0.8277025938204418, fidelity_at_trigger=None
- **Decision**: strategy=expand_scope, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version ul_bler-v1 -> None, duration=0.00s
- **Error**: ExpandScopeAgent.expand_scope() missing 1 required positional argument: 'example_component_factory'

### Event 4 — dl_bler — 2026-10-08T10:04:00.580101+00:00

- **Trigger**: type=external_drift, severity=0.5381433132192782, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: Deterministic severity-threshold rule: severity=0.5381 -> 'regenerate' (thresholds: recalibrate<=0.3, regenerate<=0.7).
- **Reasoning**: No LLM call for the decision itself (phase13 §4: rule_based).
- **References**: []
- **Candidate**: version=dl_bler-v2
- **Verification**: None -> None => REJECT (The regeneration adaptation candidate dl_bler-v2 for component 'dl_bler' was rejected by the deterministic verification gate because fidelity improvement could not be validated. Specifically, the FidelityScore before and/or after evaluation was not computable due to insufficient rolling history within the evaluation set. Applying the system's fail-safe policy, candidates without verifiable fidelity gains over the production baseline must be rejected. (1. Candidate artifact exists and passed interface identity and finite output validation checks (5,173 evaluation rows).
2. The fidelity comparison failed (fidelity_improved=FAIL) because rolling history was insufficient to compute a valid FidelityScore for one or both model versions.
3. Fail-safe safety protocol enforced an automatic REJECT decision to ensure baseline stability.))
- version dl_bler-v1 -> dl_bler-v1, duration=37.47s

### Event 5 — dl_prb — 2026-10-08T10:04:38.208126+00:00

- **Trigger**: type=external_drift, severity=0.303194829291645, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: Deterministic severity-threshold rule: severity=0.3032 -> 'regenerate' (thresholds: recalibrate<=0.3, regenerate<=0.7).
- **Reasoning**: No LLM call for the decision itself (phase13 §4: rule_based).
- **References**: []
- **Candidate**: version=dl_prb-v2
- **Verification**: None -> None => REJECT (The candidate model 'dl_prb-v2' generated during the regenerate adaptation process was rejected by the deterministic verification gate. Although the candidate passed artifact generation, interface consistency, evaluation execution, and finite prediction checks, the fidelity improvement check failed. Specifically, the FidelityScore before and/or after candidate execution could not be calculated due to insufficient rolling history on the held-out evaluation set. In accordance with the system's fail-safe adaptation rules (verification_delta=0.01), an uncomputable fidelity delta results in an automatic REJECT decision to ensure production stability. (1. Artifact generation and interface identity checks passed (dl_prb-v2.joblib exists, output field matches 'dl_prb_pred').
2. Evaluation set execution completed with 5,173 valid finite prediction samples.
3. FidelityScore comparison failed because rolling history was insufficient to reliably compute fidelity metrics before and/or after, triggering a fail-safe REJECT decision.))
- version dl_prb-v1 -> dl_prb-v1, duration=37.52s

### Event 6 — dl_bler — 2026-10-08T10:05:22.410960+00:00

- **Trigger**: type=external_drift, severity=0.13404169724716475, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: Deterministic severity-threshold rule: severity=0.1340 -> 'recalibrate' (thresholds: recalibrate<=0.3, regenerate<=0.7).
- **Reasoning**: No LLM call for the decision itself (phase13 §4: rule_based).
- **References**: []
- **Candidate**: version=dl_bler-v3
- **Verification**: None -> None => REJECT (The recalibrated candidate model 'dl_bler-v3' was rejected by the deterministic verification gate because the fidelity score before and/or after evaluation could not be computed. Although the candidate artifact was generated successfully, maintained interface identity, and produced finite predictions across all 5,173 evaluation rows, the lack of sufficient rolling evaluation history rendered the fidelity comparison non-computable. Following the system's fail-safe adaptation policy, any candidate that cannot verify a net improvement over production by at least the verification delta (0.01) must be rejected. (1. All structural and interface compliance checks passed (artifact present, interface identity consistent, valid non-null predictions for 5,173 samples).
2. Fidelity improvement check failed because FidelityScore before/after was non-computable due to insufficient rolling history.
3. The verification gate strictly enforces a fail-safe REJECT policy when fidelity gains cannot be quantitatively verified against the production baseline.))
- version dl_bler-v1 -> dl_bler-v1, duration=15.63s

### Event 7 — ul_bler — 2026-10-08T10:05:38.221436+00:00

- **Trigger**: type=external_drift, severity=0.2623133404418495, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: Deterministic severity-threshold rule: severity=0.2623 -> 'recalibrate' (thresholds: recalibrate<=0.3, regenerate<=0.7).
- **Reasoning**: No LLM call for the decision itself (phase13 §4: rule_based).
- **References**: []
- **Candidate**: version=ul_bler-v2
- **Verification**: None -> None => REJECT (The candidate recalibrated model for 'ul_bler' (version ul_bler-v2) was rejected by the deterministic verification gate under the fail-safe policy. Although candidate artifact existence, interface identity consistency, evaluation execution, finite predictions, and baseline evaluability checks all passed, the fidelity improvement check failed (fidelity_improved=FAIL). Specifically, the FidelityScore before and/or after candidate evaluation could not be computed due to insufficient rolling history within the evaluation set. Per adaptation safety guidelines, when fidelity improvement cannot be verified, the system enforces a mandatory REJECT to prevent unverified model promotion. (1. Artifact & Interface: candidate_artifact_present=PASS and interface_identity_consistent=PASS for ul_bler.
2. Evaluation Run: Both target and candidate predictions evaluated over 5,173 rows with valid finite outputs (no_invalid_output=PASS).
3. Rejection Root Cause: fidelity_improved=FAIL because FidelityScore is uncomputable due to insufficient rolling history, triggering a fail-safe REJECT.))
- version ul_bler-v1 -> ul_bler-v1, duration=15.83s

### Event 8 — ul_throughput — 2026-10-08T10:05:54.310808+00:00

- **Trigger**: type=external_drift, severity=0.2804087579860399, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: Deterministic severity-threshold rule: severity=0.2804 -> 'recalibrate' (thresholds: recalibrate<=0.3, regenerate<=0.7).
- **Reasoning**: No LLM call for the decision itself (phase13 §4: rule_based).
- **References**: []
- **Candidate**: version=ul_throughput-v2
- **Verification**: None -> 0.9756895579241149 => REJECT (The recalibration attempt for the 'ul_throughput' component was rejected during the deterministic verification gate. Although the candidate model artifact was correctly generated and produced finite predictions matching the evaluation target length, the fidelity score comparison could not be completed. Specifically, the fidelity check failed because either the baseline or candidate fidelity score was non-computable due to insufficient rolling history on the evaluated set, triggering a fail-safe REJECT decision. (1. Artifact and interface consistency checks passed (candidate artifact present and output field matching 'ul_throughput_pred').
2. Evaluation executed across 5,173 rows producing valid, finite predictions.
3. Candidate rejection was driven by 'fidelity_improved=FAIL' due to insufficient rolling evaluation history required to compute FidelityScore margin comparison against the verification delta threshold (0.01).))
- version ul_throughput-v1 -> ul_throughput-v1, duration=16.88s

### Event 9 — dl_throughput — 2026-10-08T10:06:11.417599+00:00

- **Trigger**: type=external_drift, severity=0.9616571936637868, fidelity_at_trigger=None
- **Decision**: strategy=expand_scope, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version dl_throughput-v1 -> None, duration=0.00s
- **Error**: ExpandScopeAgent.expand_scope() missing 1 required positional argument: 'example_component_factory'

### Event 10 — ul_bler — 2026-10-08T10:06:11.680812+00:00

- **Trigger**: type=external_drift, severity=0.5412268555474342, fidelity_at_trigger=0.7500354470768414
- **Decision**: strategy=regenerate, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: Deterministic severity-threshold rule: severity=0.5412 -> 'regenerate' (thresholds: recalibrate<=0.3, regenerate<=0.7).
- **Reasoning**: No LLM call for the decision itself (phase13 §4: rule_based).
- **References**: []
- **Candidate**: version=ul_bler-v3
- **Verification**: 0.7501567202959011 -> 0.751635818895963 => REJECT (The regenerated candidate model for 'ul_bler' (v3) was rejected by the deterministic verification gate because it failed to achieve the required improvement margin over the existing production baseline (v1). While the candidate achieved a marginal fidelity score increase from 0.7502 to 0.7516 (+0.0015), it fell short of the mandatory verification_delta threshold of 0.01 (requiring a minimum fidelity score of 0.7602). Consequently, the production system retains version ul_bler-v1. (- Deterministic decision: REJECT due to failure of the 'fidelity_improved' check.
- Baseline Fidelity Score (ul_bler-v1): 0.7502.
- Candidate Fidelity Score (ul_bler-v3): 0.7516 (improvement of +0.0015 vs required +0.0100).
- All structural integrity checks passed (artifact presence, interface identity consistency, evaluation row count matching at 5,173 samples, finite values, baseline evaluability).))
- version ul_bler-v1 -> ul_bler-v1, duration=34.18s

### Event 11 — dl_bler — 2026-10-08T10:06:46.104917+00:00

- **Trigger**: type=external_drift, severity=0.9699254132161326, fidelity_at_trigger=0.7504130668818338
- **Decision**: strategy=expand_scope, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version dl_bler-v1 -> None, duration=0.00s
- **Error**: ExpandScopeAgent.expand_scope() missing 1 required positional argument: 'example_component_factory'

### Event 12 — ul_throughput — 2026-10-08T10:06:46.263969+00:00

- **Trigger**: type=external_drift, severity=0.11586561247077032, fidelity_at_trigger=0.38366840519297307
- **Decision**: strategy=recalibrate, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: Deterministic severity-threshold rule: severity=0.1159 -> 'recalibrate' (thresholds: recalibrate<=0.3, regenerate<=0.7).
- **Reasoning**: No LLM call for the decision itself (phase13 §4: rule_based).
- **References**: []
- **Candidate**: version=ul_throughput-v3
- **Verification**: 0.9919903354159357 -> 0.9898207980221527 => REJECT (The recalibration candidate for 'ul_throughput' was rejected because it failed the deterministic fidelity improvement check. While all pre-verification structural and execution checks passed (including candidate artifact existence, interface identity consistency, evaluation execution, finite predictions, and baseline evaluation), the candidate achieved a FidelityScore of 0.9898 compared to the production baseline's 0.9920. Because the fidelity score decreased and did not exceed the production score plus the required verification delta of 0.01, the candidate was automatically rejected. (1. Candidate FidelityScore (0.9898) was lower than the production baseline FidelityScore (0.9920).
2. Failed the `fidelity_improved` rule since the margin fell below the required threshold of verification_delta=0.01.
3. All structural, sanity, and interface identity checks passed successfully prior to score comparison.))
- version ul_throughput-v1 -> ul_throughput-v1, duration=16.74s

### Event 13 — dl_prb — 2026-10-08T10:07:03.194825+00:00

- **Trigger**: type=external_drift, severity=0.6130033010530405, fidelity_at_trigger=0.7571425131760543
- **Decision**: strategy=regenerate, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: Deterministic severity-threshold rule: severity=0.6130 -> 'regenerate' (thresholds: recalibrate<=0.3, regenerate<=0.7).
- **Reasoning**: No LLM call for the decision itself (phase13 §4: rule_based).
- **References**: []
- **Candidate**: version=dl_prb-v3
- **Verification**: 0.7499975343251859 -> 0.6896193593224553 => REJECT (The candidate model generated for the 'dl_prb' component during the regeneration adaptation was REJECTED because it failed the deterministic fidelity improvement check. Prior to adaptation, the production version (dl_prb-v1) achieved a FidelityScore of 0.7500. The newly regenerated candidate version (dl_prb-v3) achieved a FidelityScore of 0.6896 on the same evaluation dataset of 5,173 rows. To be accepted under the active adaptation policy, the candidate's fidelity score must exceed the production score plus the required verification delta of 0.01 (threshold of 0.7600). Because candidate fidelity dropped by approximately 0.0604, the deterministic gate correctly rejected the candidate and retained the existing production model. (1. Adaptation trigger: Candidate regeneration attempted for 'dl_prb'.
2. Metric baseline vs. candidate: Fidelity decreased from 0.7500 (dl_prb-v1) to 0.6896 (dl_prb-v3).
3. Policy threshold: Required candidate fidelity score was >= 0.7600 (fidelity_before + verification_delta 0.01).
4. Verification gate status: 'fidelity_improved' failed, leading to an automatic REJECT decision.))
- version dl_prb-v1 -> dl_prb-v1, duration=38.18s

### Event 14 — dl_bler — 2026-10-08T10:07:41.535941+00:00

- **Trigger**: type=external_drift, severity=0.03959287666420286, fidelity_at_trigger=0.7504130668818338
- **Decision**: strategy=recalibrate, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: Deterministic severity-threshold rule: severity=0.0396 -> 'recalibrate' (thresholds: recalibrate<=0.3, regenerate<=0.7).
- **Reasoning**: No LLM call for the decision itself (phase13 §4: rule_based).
- **References**: []
- **Candidate**: version=dl_bler-v4
- **Verification**: 0.7509339485271057 -> 0.8252174913753535 => ACCEPT (The recalibrated candidate model for the 'dl_bler' component successfully passed all deterministic verification checks and was accepted into production. Model re-training on updated telemetry yielded a significant increase in FidelityScore from 0.7509 to 0.8252, comfortably exceeding the required verification delta threshold of 0.01 (+0.0743 improvement). All predictions were verified as finite and well-formed over the 5,173 evaluation samples. (FidelityScore improved from 0.7509 to 0.8252 (delta +0.0743 vs required threshold 0.01). Interface identity and component factory attributes matched target specifications. All 5,173 evaluated prediction samples were valid and finite, and the candidate model artifact was generated successfully at /home/harie06/dt_adaptation/baselines/results/rule_based/models/dl_bler/dl_bler-v4.joblib.))
- version dl_bler-v1 -> dl_bler-v4, duration=15.52s
