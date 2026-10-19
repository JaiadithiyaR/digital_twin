# Baseline run report — mode `full_system`

- Replayed: OAI run(s) [8] (TEST split, speed=0)
- Seeds: [2]
- Model versions at start: {'dl_bler': 'dl_bler-v1', 'ul_bler': 'ul_bler-v1', 'dl_prb': 'dl_prb-v1', 'ul_prb': 'ul_prb-v1', 'dl_throughput': 'dl_throughput-v1', 'ul_throughput': 'ul_throughput-v1', 'dl_rlc_delay': 'dl_rlc_delay-v1'}
- Fidelity config: threshold=0.5, min_consecutive=3, rolling_window_length=200

## Per-component summary

| component | mean actual | mean predicted | RMSE | MAE | mean F_c |
|---|---|---|---|---|---|
| dl_bler | 0.0611 | 0.0614 | 0.0017 | 0.0008 | 0.7505 |
| dl_prb | 19.2771 | 19.3358 | 1.3012 | 0.7381 | 0.7560 |
| dl_rlc_delay | 0.2572 | 0.5993 | 0.6664 | 0.3421 | 0.7931 |
| dl_throughput | 0.2200 | 0.2209 | 0.0149 | 0.0068 | 0.8130 |
| ul_bler | 0.0075 | 0.0074 | 0.0005 | 0.0002 | 0.7500 |
| ul_prb | 0.9126 | 0.9108 | 0.0501 | 0.0288 | 0.8897 |
| ul_throughput | 0.0144 | 0.0145 | 0.0003 | 0.0002 | 0.5448 |

## Adaptation Events

### Event 1 — dl_bler — 2026-10-09T05:43:13.351096+00:00

- **Trigger**: type=external_drift, severity=0.2984911434141233, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: The affected component dl_bler has triggered an external drift alert with a severity of 0.2985. The recent telemetry indicates elevated downlink block-error rates (dl_bler mean of 0.3094) along with high PRB usage (dl_prb mean of 127.2410) and significant PRB demand (requested_prb_ratio_lag1 ~49.67 across 7 UEs). The baseline model relationship and feature inputs remain valid within the existing 5G OAI RAN telemetry schema, but the model parameters need updating to reflect changed radio channel dynamics under high cell load.
- **Reasoning**: External drift on dl_bler without structural changes in the underlying feature schema or KPI definitions indicates that retraining the learned parameters over recent telemetry windows (recalibrate) is sufficient to restore prediction fidelity.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_bler-v2
- **Verification**: None -> None => REJECT (The candidate recalibration for component 'dl_bler' (v2) was REJECTED by the deterministic verification gate. Although the candidate model artifact was successfully generated, satisfied interface identity checks, produced valid finite predictions, and passed sandbox execution on the held-out dataset, the fidelity improvement check failed. Specifically, the FidelityScore before and/or after adaptation could not be calculated due to insufficient rolling evaluation history in the evaluation window. Under the system's safety policy, uncomputable fidelity triggers a fail-safe REJECT decision to ensure unverified models are not deployed to production. (1. Artifact, interface consistency, execution, and output validity checks all passed successfully for 'dl_bler' v2.
2. The candidate failed the 'fidelity_improved' check because FidelityScore metrics could not be computed due to insufficient rolling history.
3. The system enforced a fail-safe REJECT to maintain Digital Twin operational safety when quantitative verification margin cannot be established.))
- version dl_bler-v1 -> dl_bler-v1, duration=12.44s

### Event 2 — ul_throughput — 2026-10-09T05:43:29.588262+00:00

- **Trigger**: type=external_drift, severity=0.0919159421350969, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: External drift detected on the ul_throughput component (severity 0.0919). Network state telemetry indicates high uplink resource demand (requested_prb_ratio_lag1 = 49.7705, ul_prb = 0.3588) resulting in stale model parameters for uplink throughput estimation under shifted traffic conditions.
- **Reasoning**: The model architecture remains valid for predicting UL throughput from uplink block error rates and PRBs, but the learned feature weights have degraded due to external environmental drift. Retraining the model via recalibrate using recent telemetry data will restore fidelity.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'digital_twin/internal:CLAUDE.md']
- **Candidate**: version=ul_throughput-v2
- **Verification**: None -> None => REJECT (The recalibration attempt for component 'ul_throughput' was rejected by the deterministic verification gate because the fidelity score improvement check failed ('fidelity_improved=FAIL'). Although all prerequisite validation checks passed—including candidate artifact verification, interface consistency, execution completion, output validity, and baseline evaluability—the FidelityScore before and/or after could not be reliably computed due to insufficient rolling history in the evaluation window. Under the system's safety-first design, uncomputable fidelity scores trigger an automatic fail-safe rejection to prevent unverified candidate models from entering production. (1. All standard pre-qualification checks passed (artifact presence, interface identity matching, evaluation execution over 5173 rows, valid numeric outputs, and baseline evaluability).
2. The adaptation failed specifically on 'fidelity_improved' because rolling history requirements for fidelity calculation were not met.
3. Fail-safe policy correctly rejected the candidate model to maintain system integrity when fidelity delta cannot be verified.))
- version ul_throughput-v1 -> ul_throughput-v1, duration=14.96s

### Event 3 — dl_bler — 2026-10-09T05:43:50.426737+00:00

- **Trigger**: type=external_drift, severity=0.18790107336660344, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: External drift was detected on dl_bler with severity 0.1879. A previous attempt to recalibrate this component was rejected by the verification gate, indicating that simply retraining existing model weights on recent data is insufficient to capture the altered radio link behaviour. Therefore, regenerating the component model architecture is required.
- **Reasoning**: Recalibration previously failed and was REJECTed during candidate verification. Escalating from recalibrate to regenerate is necessary when retrained parameters cannot resolve the drift in MAC-layer block-error rate dynamics.
- **References**: ['digital_twin/internal:CLAUDE.md']
- **Candidate**: version=dl_bler-v3
- **Verification**: None -> None => REJECT (The regenerate adaptation candidate for the 'dl_bler' component was rejected by the deterministic verification gate because fidelity improvement could not be validated. While the candidate model successfully built and generated 5,173 valid, finite prediction outputs matching the expected interface contract, the FidelityScore before and/or after evaluation could not be calculated due to insufficient rolling history. Under the system's fail-safe adaptation safety rules, uncomputable fidelity metrics trigger an automatic REJECT verdict to protect the production baseline. (1. All basic artifact checks, interface identity constraints, finite output validations, and baseline evaluability checks passed with 5,173 evaluated target rows.
2. The fidelity_improved check failed because FidelityScore computation requires a continuous rolling window history that was incomplete or missing for this evaluation run.
3. In accordance with safety policies, missing or non-computable fidelity score metrics result in a mandatory deterministic REJECT.))
- version dl_bler-v1 -> dl_bler-v1, duration=39.30s

### Event 4 — dl_bler — 2026-10-09T05:44:33.032562+00:00

- **Trigger**: type=external_drift, severity=0.2749693679060381, fidelity_at_trigger=None
- **Decision**: strategy=expand_scope, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version dl_bler-v1 -> None, duration=12.41s
- **Error**: component has DEPENDENCIES=('dl_bler', 'dl_throughput') but no dependency_output_fields mapping was supplied

### Event 5 — dl_rlc_delay — 2026-10-09T05:44:51.381838+00:00

- **Trigger**: type=external_drift, severity=0.15006226330533612, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: An external drift trigger was detected on the dl_rlc_delay component with severity 0.1501. Telemetry indicates shift in uplink/downlink load dynamics (high requested PRB ratio lag1 ~48.9, 7 active UEs), which affects RLC SDU queueing delay downstream. The current pipeline structure and dependencies (dl_rlc_delay depending on dl_throughput and dl_bler) remain valid, but model parameters require recalibration against recent data distributions.
- **Reasoning**: External drift without architectural model breakdown indicates parameter stale-ness. Recalibrating the model parameters using the recent training window is the appropriate first-line strategy before considering more complex architectural changes like regeneration or adding new components.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_rlc_delay-v2
- **Verification**: None -> None => REJECT (The recalibration candidate for 'dl_rlc_delay' (v2) was rejected by the deterministic verification gate because the FidelityScore could not be computed for evaluation. The gate enforces a fail-safe REJECT rule when baseline or candidate fidelity metrics cannot be calculated due to insufficient rolling history in the evaluation slice. Even though structural checks, interface consistency, and prediction output validity checks passed, the lack of a measurable fidelity improvement prevented adoption. (1. Artifact presence, component interface identity, prediction finiteness, and evaluation sample count (5173 rows) all passed validation successfully.
2. Verification failed at 'fidelity_improved' because rolling window requirements prevented calculating pre- or post-adaptation FidelityScores.
3. Under system adaptation policy, unquantifiable fidelity changes trigger an automatic fail-safe rejection to preserve system stability.))
- version dl_rlc_delay-v1 -> dl_rlc_delay-v1, duration=13.21s

### Event 6 — dl_prb — 2026-10-09T05:45:08.465383+00:00

- **Trigger**: type=external_drift, severity=0.6692972985745202, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: External drift was detected on component 'dl_prb' with a severity of 0.6693, driven by a surge in traffic demand (requested_prb_ratio_lag1 reached 48.8680 with 7 active UEs in the cell). The existing PRB allocation pipeline structure remains valid, but its model parameters require recalibration against recent network load statistics.
- **Reasoning**: Because the component architecture and feature definitions are well-established for dl_prb and the issue is non-structural external drift under changed traffic conditions, recalibrating parameters on the recent 24-hour training window is sufficient to restore prediction accuracy without requiring a full code regeneration or scope expansion.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_prb-v2
- **Verification**: None -> None => REJECT (The candidate recalibration for component 'dl_prb' was rejected by the deterministic verification gate because the FidelityScore could not be computed. Specifically, there was insufficient rolling history in the evaluation dataset to evaluate the baseline or candidate fidelity scores, triggering a fail-safe REJECT decision under the adaptation policies. (Candidate artifact generation, interface identity, evaluation execution, and numerical validity checks all passed successfully (5173 evaluation samples). However, fidelity improvement could not be verified due to uncomputable fidelity metrics resulting from missing rolling history, mandating an automatic fail-safe rejection.))
- version dl_prb-v1 -> dl_prb-v1, duration=13.63s

### Event 7 — ul_throughput — 2026-10-09T05:45:25.880754+00:00

- **Trigger**: type=external_drift, severity=0.9674359524936766, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: An external drift event with severe magnitude (0.9674) impacted 'ul_throughput'. A previous 'recalibrate' adaptation attempt was REJECTED, indicating that simply updating model parameters on recent data is insufficient to capture the altered relationship between uplink metrics (ul_prb: 0.3157, requested_prb_ratio_lag1: 48.868) and uplink throughput under high cell load (ue_count_cell_lag1: 7). Therefore, a structural pipeline regeneration is required to rebuild the predictive architecture.
- **Reasoning**: Recalibration previously failed and was rejected by the verification gate. According to adaptation policy, when simple retraining fails following high-severity external drift, regenerating the component model architecture from scratch is the necessary next escalation step.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_throughput-v3
- **Verification**: None -> None => REJECT (The candidate model 'ul_throughput-v3' generated during the regeneration adaptation was rejected by the deterministic verification gate. Although the candidate artifact was present, syntactically valid, and generated finite predictions matching the target schema length, the fidelity improvement check failed. Specifically, the FidelityScore before and/or after evaluation could not be reliably computed due to insufficient rolling history samples in the verification set. Under the system's fail-safe policy, an uncomputable or non-verifiable fidelity score automatically triggers a deterministic REJECT decision to ensure candidate models do not compromise production stability. (1. All initial structural, schema, and baseline execution checks passed (candidate_artifact_present, interface_identity_consistent, evaluation_ran, no_invalid_output, production_baseline_evaluable).
2. Verification failed on 'fidelity_improved' because FidelityScore computation requires sufficient rolling history that was missing/incomplete for the evaluation window.
3. The system enforced its fail-safe verification policy, rejecting the candidate to prevent unverified model deployment.))
- version ul_throughput-v1 -> ul_throughput-v1, duration=32.94s

### Event 8 — ul_prb — 2026-10-09T05:46:02.976653+00:00

- **Trigger**: type=external_drift, severity=0.39162483308002616, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: The trigger type is external_drift on the ul_prb component with a severity of 0.3916. Telemetry indicates elevated uplink resource allocation (ul_prb average 0.3178, requested_prb_ratio_lag1 at 49.0190 across 7 UEs). Because this represents a parameter shift in uplink PRB demand rather than a structural failure or new unmodeled phenomenon, recalibrating the existing component on recent operational data is appropriate.
- **Reasoning**: An external drift trigger on ul_prb indicates feature/target distribution drift under changing cell load conditions. The existing pipeline structure is sound, making parameter recalibration the standard and sufficient strategy under the system's adaptation policy.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=ul_prb-v2
- **Verification**: None -> None => REJECT (The recalibration attempt for the 'ul_prb' component (version ul_prb-v2) was rejected during deterministic verification. Although the candidate model artifact was generated successfully, maintained interface consistency, and produced valid finite predictions on the evaluation set, the verification gate could not establish an improvement in fidelity score. Specifically, the fidelity score before and/or after evaluation was non-computable due to an insufficient rolling history on the held-out dataset. Under system safety policies, unquantifiable fidelity improvements trigger a fail-safe REJECT decision. (1. Artifact and interface checks passed: Model file ul_prb-v2.joblib exists and interface metadata (COMPONENT_NAME='ul_prb', OUTPUT_FIELD='ul_prb_pred') aligns perfectly with expectations.
2. Evaluation set completeness: Both ground truth target and candidate predictions contained 5,173 valid, non-null samples.
3. Fail-safe trigger: FidelityScore before or after could not be computed due to insufficient rolling window history, causing the required fidelity improvement check to fail-safe REJECT.))
- version ul_prb-v1 -> ul_prb-v1, duration=12.64s

### Event 9 — dl_rlc_delay — 2026-10-09T05:46:19.344338+00:00

- **Trigger**: type=external_drift, severity=0.5110659735695771, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: The component dl_rlc_delay experienced external drift causing degraded fidelity. A previous recalibration attempt was REJECTED by the verification gate, indicating that simple parameter retraining on the existing model pipeline/architecture is insufficient to capture the altered queueing behavior under the updated network state. Thus, a complete structural rebuild via model regeneration is required.
- **Reasoning**: Recalibration previously failed and was REJECTED, meaning parameter updating alone cannot resolve the shift in RLC SDU delay characteristics. Rebuilding the component model architecture via regeneration is the appropriate next step under the adaptation policy.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_rlc_delay-v3
- **Verification**: 0.7574016430991197 -> 0.793138662171483 => ACCEPT (The regeneration adaptation for the 'dl_rlc_delay' model was ACCEPTED because it successfully passed all deterministic checks and exceeded the required fidelity threshold. The candidate model artifact 'dl_rlc_delay-v3.joblib' was properly produced, maintained interface consistency (dl_rlc_delay_pred), and output valid, finite predictions across all 5,173 test rows. The candidate achieved a FidelityScore of 0.7931, representing a net gain of +0.0357 over the production baseline score of 0.7574. This improvement comfortably satisfies the verification_delta margin requirement of 0.01. (1. All 6 deterministic gate checks passed without errors, including interface contract verification and prediction sanity checks.
2. FidelityScore increased from 0.7574 (dl_rlc_delay-v1) to 0.7931 (dl_rlc_delay-v3), exceeding the required minimum improvement margin (+0.0100) by an additional +0.0257.
3. Candidate predictions remained strictly valid across all 5,173 evaluation samples on the held-out test split, confirming robust model stability for RLC queueing/transmission delay prediction within the downstream dependency graph.))
- version dl_rlc_delay-v1 -> dl_rlc_delay-v3, duration=36.06s

### Event 10 — dl_rlc_delay — 2026-10-09T05:46:59.003026+00:00

- **Trigger**: type=external_drift, severity=0.7755639424726894, fidelity_at_trigger=0.7930666095203093
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: The affected component dl_rlc_delay experienced external_drift (severity 0.7756). A previous model architecture regeneration was already performed and accepted (fidelity improved from 0.757 to 0.793). However, under recent network conditions marked by high requested PRB ratio (48.87) and increased active UE count (7.0), parameter staleness has caused performance drift relative to predicted inputs from upstream dependencies (dl_throughput and dl_bler). Since the recent regeneration successfully validated the model architecture, parameter recalibration on recent operational telemetry is sufficient.
- **Reasoning**: Model architecture was recently updated via regeneration and verified successfully. Recalibrating learned parameters against recent 24-hour training windows will realign the model to recent network load shifts without requiring additional iteration cycles.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_rlc_delay-v4
- **Verification**: 0.793138662171483 -> 0.793138662171483 => REJECT (The recalibration candidate for 'dl_rlc_delay' was rejected because it failed the deterministic fidelity improvement check. Re-training the model yielded a FidelityScore of 0.7931, which is identical to the production version (0.7931) and fails to exceed the required verification threshold margin of 0.01 above the current baseline. (1. All initial sanity checks passed, including model artifact presence, interface consistency, evaluation execution, and finite prediction outputs.
2. The candidate fidelity score (0.793139) equaled the production model score (0.793139), producing zero improvement.
3. Because the improvement margin (0.0) fell short of the required verification_delta (0.01), the deterministic verification gate rejected candidate dl_rlc_delay-v4 in favor of keeping dl_rlc_delay-v3 active in production.))
- version dl_rlc_delay-v3 -> dl_rlc_delay-v3, duration=12.14s

### Event 11 — ul_bler — 2026-10-09T05:47:16.190657+00:00

- **Trigger**: type=external_drift, severity=0.4709098854157575, fidelity_at_trigger=0.7500354470768414
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: The ul_bler component experienced an external drift trigger (severity 0.4709) causing MAC-layer uplink block error rate prediction accuracy to decrease relative to real OAI telemetry. Given that no previous adaptation attempts exist and the feature dependency graph remains unchanged, the underlying distribution of radio link parameters (such as uplink MCS and PUSCH SNR) has shifted due to network environment variations, making existing model parameters stale without invalidating the core architecture.
- **Reasoning**: Since the pipeline structure and dependency graph for ul_bler remain valid and no prior adaptation has failed, retraining the model on recent replayed OAI telemetry (recalibrate) is the optimal first-line strategy before considering complex architecture rebuilds or scope additions.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_bler-v2
- **Verification**: 0.7501567202959011 -> 0.7521847141991993 => REJECT (The candidate model ul_bler-v2 failed the deterministic verification gate and was rejected because its performance improvement did not satisfy the required margin. The recalibrated model achieved a FidelityScore of 0.7522, representing an increment of +0.0020 over the production model ul_bler-v1 (0.7502). However, this increase falls short of the verification_delta threshold of 0.01 required for adoption. Consequently, the production model remains active to prevent unnecessary churn for marginal gains. (1. All safety and compliance checks passed (artifact present, interface identity consistent, valid predictions generated across 5173 test rows).
2. Candidate FidelityScore (0.7522) failed the improvement gate against production baseline (0.7502) + verification_delta (0.01 = 0.7602).
3. Rejection is final; production version ul_bler-v1 remains deployed.))
- version ul_bler-v1 -> ul_bler-v1, duration=12.16s

### Event 12 — ul_bler — 2026-10-09T05:47:32.200313+00:00

- **Trigger**: type=external_drift, severity=0.10720730845358806, fidelity_at_trigger=0.7500000029308993
- **Decision**: strategy=regenerate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: External drift in the ul_bler component caused fidelity degradation (score 0.7500). A previous recalibration attempt was REJECTED because retraining existing parameters only yielded a 0.0020 gain in fidelity (0.75015 to 0.75218), failing the mandatory verification_delta threshold of 0.01. This indicates that the parameter weights are not stale, but rather the underlying model pipeline architecture is insufficient to represent the shifted uplink radio-link error characteristics under current channel conditions.
- **Reasoning**: Since recalibration failed to achieve the necessary fidelity improvement margin, escalating to regeneration is required to rebuild the ul_bler prediction pipeline architecture from scratch.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_bler-v3
- **Verification**: 0.7501567202959011 -> 0.7515867731143009 => REJECT (The regenerate adaptation candidate for component 'ul_bler' was REJECTED because it failed to achieve the required minimum fidelity improvement threshold. While all deterministic sanity and safety checks passed (artifact creation, interface identity, evaluation run completion, and finite output validation), the candidate achieved a FidelityScore of 0.7516 compared to the production baseline of 0.7502. This represents an improvement of only 0.0014, which falls short of the required verification delta threshold of 0.01 (target score >= 0.7602). Per system adaptation policies, candidates must beat production by more than verification_delta to be accepted. (FidelityScore improved marginally from 0.750157 to 0.751587 (+0.00143), failing the fidelity_improved check against verification_delta=0.01. All prerequisite deterministic checks passed: candidate artifact was present, interface identity was consistent (ul_bler/ul_bler_pred), evaluation ran successfully across 5,173 samples, outputs were valid/finite, and production baseline was evaluable.))
- version ul_bler-v1 -> ul_bler-v1, duration=39.53s

### Event 13 — ul_bler — 2026-10-09T05:48:17.383995+00:00

- **Trigger**: type=external_drift, severity=0.8844496736882935, fidelity_at_trigger=0.7500000029308993
- **Decision**: strategy=expand_scope, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version ul_bler-v1 -> None, duration=29.05s
- **Error**: component has DEPENDENCIES=('dl_prb', 'ul_prb') but no dependency_output_fields mapping was supplied

### Event 14 — ul_throughput — 2026-10-09T05:48:53.201441+00:00

- **Trigger**: type=external_drift, severity=0.8492363236144346, fidelity_at_trigger=0.9939079628115425
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: An external drift trigger was detected for ul_throughput with high severity (0.8492). Telemetry reveals high cell resource contention (requested_prb_ratio_lag1 at 48.87 and ue_count_cell_lag1 at 7.0) alongside near-zero UL throughput (0.0049 Mbps). A previous attempt to regenerate the model architecture was REJECTed by the verification gate. Thus, the underlying feature pipeline remains sound, but the model parameters require retraining on recent data to adapt to the shifted load dynamics.
- **Reasoning**: Given that model architecture regeneration was previously rejected and the standard feature vector already includes uplink load indicators, recalibration on recent 24-hour training data is the optimal strategy to update model weights without risk of pipeline instability.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=ul_throughput-v4
- **Verification**: 0.9919903354159357 -> 0.9898207980221527 => REJECT (The recalibration candidate for the 'ul_throughput' component was REJECTED because it failed the deterministic fidelity improvement check (fidelity_improved=FAIL). To be accepted, the candidate model was required to exceed the baseline FidelityScore (0.9920) by a minimum verification delta margin of 0.01 (requiring a score > 1.0020). Instead, the candidate achieved a FidelityScore of 0.9898, showing a performance degradation compared to the active production version (ul_throughput-v1). Although all structural and safety checks passed (artifact validity, interface identity consistency, finite prediction values, and baseline evaluation), the loss in predictive accuracy on the evaluation set prohibits deployment. (- Deterministic Decision: REJECTED (Fidelity score failed to improve).
- Baseline Fidelity Score: 0.9920 (ul_throughput-v1).
- Candidate Fidelity Score: 0.9898 (ul_throughput-v4).
- Required Threshold: > 1.0020 (Baseline + verification_delta of 0.01).
- Verification Check Results: Structural checks (artifact presence, interface identity, prediction finiteness) passed, but fidelity_improved failed.))
- version ul_throughput-v1 -> ul_throughput-v1, duration=13.50s

### Event 15 — dl_prb — 2026-10-09T05:49:13.271735+00:00

- **Trigger**: type=external_drift, severity=0.5165781941249967, fidelity_at_trigger=0.757266901219243
- **Decision**: strategy=regenerate, source=full_system, confidence=0.9, llm_or_fallback=llm
- **Root cause**: The affected component `dl_prb` experienced an external drift trigger with severity 0.5166. A previous recalibration attempt failed (verification_result: REJECT), indicating that retraining parameter weights on the existing pipeline is insufficient. Recent telemetry shows high cell load context (requested_prb_ratio_lag1 = 49.02, ue_count_cell_lag1 = 7.0), altering PRB allocation dynamics beyond what the current model architecture can capture.
- **Reasoning**: Because the previous `recalibrate` action failed to yield an acceptable candidate and external drift persists under significant cell UE load variations, recalibrating again is likely to fail. Regenerating the component architecture allows structural learning of the non-linear relationship between requested PRB ratio, cell UE counts, and actual allocated downlink PRBs.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_prb-v3
- **Verification**: 0.7499975343251859 -> 0.8299381961883656 => ACCEPT (The regeneration adaptation for component 'dl_prb' was ACCEPTED because the candidate model (dl_prb-v3) demonstrated a significant fidelity score improvement, rising from 0.74999 to 0.82994 on the held-out evaluation dataset. This gain of ~0.0799 easily clears the required verification threshold delta of 0.01. Furthermore, all deterministic validation gates passed successfully: the candidate artifact was generated and stored properly, output fields aligned with system identity specs, evaluation returned finite non-null predictions across all 5,173 test samples, and performance exceeded the production baseline (dl_prb-v1). (1. Fidelity Score improved from 0.749997 (dl_prb-v1) to 0.829938 (dl_prb-v3), exceeding the minimum margin threshold (verification_delta=0.01).
2. Candidate artifact exists at the designated output path and passed structural interface consistency checks (COMPONENT_NAME='dl_prb', OUTPUT_FIELD='dl_prb_pred').
3. Evaluation was executed across 5,173 valid rows with all finite predictions and no invalid outputs or runtime exceptions.))
- version dl_prb-v1 -> dl_prb-v3, duration=66.76s
