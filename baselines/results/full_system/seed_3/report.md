# Baseline run report — mode `full_system`

- Replayed: OAI run(s) [8] (TEST split, speed=0)
- Seeds: [3]
- Model versions at start: {'dl_bler': 'dl_bler-v1', 'ul_bler': 'ul_bler-v1', 'dl_prb': 'dl_prb-v1', 'ul_prb': 'ul_prb-v1', 'dl_throughput': 'dl_throughput-v1', 'ul_throughput': 'ul_throughput-v1', 'dl_rlc_delay': 'dl_rlc_delay-v1'}
- Fidelity config: threshold=0.5, min_consecutive=3, rolling_window_length=200

## Per-component summary

| component | mean actual | mean predicted | RMSE | MAE | mean F_c |
|---|---|---|---|---|---|
| dl_bler | 0.0916 | 0.0922 | 0.0020 | 0.0012 | 0.7504 |
| dl_prb | 28.9014 | 28.9023 | 1.5888 | 1.0200 | 0.7562 |
| dl_rlc_delay | 0.3858 | 0.7999 | 0.8057 | 0.4140 | 0.7576 |
| dl_throughput | 0.3299 | 0.3310 | 0.0182 | 0.0101 | 0.7638 |
| ul_bler | 0.0113 | 0.0111 | 0.0006 | 0.0003 | 0.7500 |
| ul_prb | 1.2255 | 1.2192 | 0.0611 | 0.0396 | nan |
| ul_throughput | 0.0190 | 0.0192 | 0.0003 | 0.0002 | nan |

## Adaptation Events

### Event 1 — dl_bler — 2026-10-09T06:00:36.269059+00:00

- **Trigger**: type=external_drift, severity=0.2368105065960997, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.91, llm_or_fallback=llm
- **Root cause**: External drift was detected on the dl_bler component with a severity of 0.2368. Telemetry shows a high downlink BLER (0.3094) along with high cell UE contention (7 UEs) and high requested PRB ratio (49.67%). Because dl_bler is an upstream radio-link KPI depending on channel conditions and scheduling parameters whose baseline structural feature relationships remain valid, the fidelity shift is driven by staleness in learned model parameters under the shifted channel environment.
- **Reasoning**: Since the feature structure for MAC-layer downlink block error rate remains intact and valid, retraining the existing pipeline on recent 24-hour telemetry window data via recalibration will restore prediction fidelity without needing full model architecture regeneration.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_bler-v2
- **Verification**: None -> None => REJECT (The recalibration candidate for the dl_bler component was rejected because the deterministic fidelity check failed. Although the candidate model passed all structural, interface, and evaluation checks—generating valid predictions across 5,173 rows—the FidelityScore before and/or after could not be computed due to insufficient rolling history in the evaluation window. Under the system's safety policy, when fidelity improvement cannot be verified quantitatively against the production baseline, a fail-safe REJECT decision is enforced. (1. Artifact presence, interface identity, evaluation execution, and prediction validity checks all passed successfully for dl_bler-v2.
2. Fidelity score evaluation failed because rolling history was insufficient to compute a valid FidelityScore metric for comparison.
3. The deterministic verification gate triggered a fail-safe REJECT decision due to unquantifiable fidelity improvement.))
- version dl_bler-v1 -> dl_bler-v1, duration=13.06s

### Event 2 — dl_throughput — 2026-10-09T06:00:53.638114+00:00

- **Trigger**: type=external_drift, severity=0.5821620360643678, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: An external drift event with severity 0.5822 was detected for the dl_throughput component. Recent network telemetry indicates moderate radio link degradation (dl_bler: 0.1131) and high cell resource pressure (requested_prb_ratio_lag1: 49.77, ue_count_cell_lag1: 7.0), causing a shift in statistical relationships. The feature space and dependency structure remain valid, but model weights require retraining on recent telemetry.
- **Reasoning**: External drift without structural failure in feature dependencies indicates stale model parameters. Recalibrating the model over the recent training window is sufficient to regain fidelity before considering architectural changes.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_throughput-v2
- **Verification**: None -> None => REJECT (The recalibration candidate for 'dl_throughput' was rejected because the system was unable to compute a valid FidelityScore comparison between the candidate and production models. Although the candidate model artifact was generated, satisfied interface checks, and produced valid finite predictions on the evaluation set (5,173 rows), the evaluation metric requires a minimum rolling history window that was not met by the evaluation sample. In accordance with the system's fail-safe verification policy, any candidate whose fidelity improvement cannot be deterministically proven above the verification_delta threshold (0.01) must be REJECTED. (1. All deterministic prerequisite checks passed (artifact presence, interface consistency, execution completion, and non-NaN outputs).
2. Fidelity comparison failed because FidelityScore before and/or after was non-computable due to insufficient rolling history in the evaluation slice.
3. Fail-safe safety policy triggered automatically to reject the candidate due to unverified fidelity performance.))
- version dl_throughput-v1 -> dl_throughput-v1, duration=15.83s

### Event 3 — dl_rlc_delay — 2026-10-09T06:01:13.345686+00:00

- **Trigger**: type=external_drift, severity=0.479051298140834, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: An external drift event (severity 0.4791) was detected for the dl_rlc_delay component. Based on recent telemetry, average downlink PRB utilization is high (102.63 PRBs/s) alongside an elevated downlink BLER (0.3175) and cell UE count (7.0), causing shifted distribution dynamics in RLC queueing and transmission delays. Because this is the first adaptation trigger without prior failed attempts and the fundamental input-output relationship remains within the feature space of the model, recalibrating the model parameters on recent data is the most appropriate initial response.
- **Reasoning**: External drift indicates distribution shift in telemetry features (e.g. high PRB usage and MAC layer block error rate impacting RLC delays). Recalibration updates model weights using recent data to adjust to current network load without requiring architectural redesign or scope expansion.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_rlc_delay-v2
- **Verification**: None -> None => REJECT (The recalibrated candidate artifact for 'dl_rlc_delay' (version v2) passed all baseline structural and interface checks (artifact presence, schema consistency, finite evaluation outputs, and production evaluation parity). However, the adaptation was REJECTED because the FidelityScore could not be computed for the evaluation window due to insufficient rolling history for the before or after state. Per the fail-safe policy, an unverified fidelity improvement results in an automatic rejection. (1. All structural, schema, and interface consistency tests PASSED (5,173 valid evaluation rows evaluated against baseline v1).
2. The 'fidelity_improved' check FAILED because the rolling history window was insufficient to calculate a definitive FidelityScore margin relative to the verification_delta threshold of 0.01.
3. The system correctly applied a fail-safe REJECT decision to ensure candidate model stability prior to system integration.))
- version dl_rlc_delay-v1 -> dl_rlc_delay-v1, duration=13.25s

### Event 4 — dl_prb — 2026-10-09T06:01:30.521991+00:00

- **Trigger**: type=external_drift, severity=0.7345771514092145, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: The affected component `dl_prb` exhibits external drift with high severity (0.7346). Under recent network conditions, the requested PRB ratio lag (`requested_prb_ratio_lag1` = 48.94) and average cell UE count (`ue_count_cell_lag1` = 7.0) indicate shifts in PRB allocation demands, causing the current production model's learned parameters for `dl_prb` to become stale relative to active radio resource scheduling dynamics. Recalibrating the model over recent data will re-fit the learned weights to the current allocation patterns.
- **Reasoning**: Because the model pipeline structure for physical resource block (`dl_prb`) allocation remains mathematically appropriate for the O-RAN/OAI RAN layer features, but parameters have drifted due to changed network load and UE scheduling conditions, retraining via `recalibrate` is the minimal and most effective strategy.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_prb-v2
- **Verification**: None -> None => REJECT (The recalibration candidate `dl_prb-v2` for component `dl_prb` was rejected by the deterministic verification gate. Although the artifact was successfully generated, passed schema and non-null checks, and ran evaluation over 5,173 rows, the fidelity improvement check failed (`fidelity_improved=FAIL`). Specifically, the FidelityScore before and/or after evaluation could not be reliably computed due to insufficient rolling history in the test evaluation window. Under the system's fail-safe adaptation policy, unquantifiable fidelity changes mandate an automated rejection to prevent unverified model promotion into production. (1. Artifact Verification: `candidate_artifact_present`, `interface_identity_consistent`, `evaluation_ran`, and `no_invalid_output` all PASSED for 5,173 evaluation rows.
2. Verification Failure: `fidelity_improved` failed because rolling history metrics required for calculating the FidelityScore were missing or incomplete for one or both model versions.
3. Decision Outcome: REJECT enforced by the deterministic gate due to strict fail-safe rules requiring quantifiable metric improvement exceeding `verification_delta=0.01`.))
- version dl_prb-v1 -> dl_prb-v1, duration=14.58s

### Event 5 — dl_prb — 2026-10-09T06:01:49.142919+00:00

- **Trigger**: type=external_drift, severity=0.5167401826213637, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: The previous adaptation attempt using recalibration for the 'dl_prb' component failed (verification_result=REJECT). Additionally, telemetry reveals high resource contention with requested_prb_ratio_lag1 at ~48.94 and 7 active UEs in the cell, while current PRB allocation (dl_prb=0.029) is extremely low. Simple model recalibration cannot capture these structural non-linear traffic dynamics, necessitating an architectural rebuild/regeneration.
- **Reasoning**: Recalibration was previously attempted and rejected by the deterministic verification gate. Per system adaptation policies, when parameter tuning fails to resolve external drift under significant resource load shifts, regenerating the model architecture is required to learn new functional mappings.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_prb-v3
- **Verification**: None -> None => REJECT (The regeneration attempt for the dl_prb component was rejected by the deterministic verification gate. Although the candidate model (dl_prb-v3.joblib) successfully passed artifact presence, interface consistency, execution evaluation, and valid finite-output checks, the overall fidelity score could not be computed due to insufficient rolling history in the evaluation dataset. Per the system's deterministic safety policy, when baseline or candidate fidelity scores are non-computable, the system cannot verify a strictly positive improvement exceeding the verification_delta threshold (0.01). Consequently, the decision defaults to a fail-safe REJECT, keeping production version dl_prb-v1 active. (Candidate code execution, interface validation, and raw model evaluation passed with 5,173 valid finite prediction rows. However, fidelity_improved failed because the FidelityScore before and/or after evaluation returned None due to insufficient rolling history windows, triggering an automatic fail-safe rejection.))
- version dl_prb-v1 -> dl_prb-v1, duration=28.79s

### Event 6 — dl_rlc_delay — 2026-10-09T06:02:21.757809+00:00

- **Trigger**: type=external_drift, severity=0.5867985714381407, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.91, llm_or_fallback=llm
- **Root cause**: The affected component `dl_rlc_delay` suffered an external_drift event with severity 0.5868. A previous recalibration attempt was rejected by the verification gate (`verification_result=REJECT`), indicating that retraining learned parameters on recent data is insufficient to capture the altered underlying queueing dynamics. Furthermore, telemetry shows high requested PRB ratio (48.87) with 7 active UEs despite low throughput, suggesting non-linear congestion effects that require regenerating the model pipeline/architecture rather than simple recalibration.
- **Reasoning**: Recalibration has already failed and been rejected by the verification gate. According to adaptation policy, when parameter tuning (recalibrate) fails to resolve drift, regenerating the model architecture/pipeline is required to handle structural shifts in feature interaction.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_rlc_delay-v3
- **Verification**: None -> None => REJECT (The regeneration adaptation candidate dl_rlc_delay-v3 was REJECTED because the fidelity score improvement could not be evaluated. Specifically, there was insufficient rolling history during the evaluation window to compute a valid FidelityScore for either the baseline (dl_rlc_delay-v1) or the candidate model. Per system adaptation safety policy, if the fidelity improvement cannot be verified quantitatively against the production threshold (verification_delta=0.01), the system fails safe and rejects the candidate. (1. All deterministic interface and execution checks passed: candidate artifact exists, interface identity matched 'dl_rlc_delay', model output had 5,173 valid finite prediction rows, and baseline model was evaluable.
2. The decision failed on the fidelity_improved check due to uncomputable FidelityScore metrics from insufficient rolling history.
3. The component 'dl_rlc_delay' is downstream of 'dl_throughput' and 'dl_bler' in the Digital Twin dependency graph and is scheduled last.))
- version dl_rlc_delay-v1 -> dl_rlc_delay-v1, duration=34.62s

### Event 7 — dl_bler — 2026-10-09T06:03:00.094459+00:00

- **Trigger**: type=external_drift, severity=0.28420116374879145, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: The previous recalibration attempt for dl_bler failed (REJECTed by the verification gate), indicating that simple parameter retraining on the existing model pipeline is insufficient to handle the detected external drift under current network dynamics (e.g., active cell UE load of 7 UEs and high requested PRB ratio). Since recalibrating failed and scope expansion is not warranted for a core base KPI, the model architecture itself must be regenerated to capture the updated radio link error behavior.
- **Reasoning**: A prior recalibration action was rejected, proving that recalibrating learned parameters on the same architecture cannot resolve the drift. Per standard self-adaptation policy, when retraining fails, rebuilding the component model structure via regeneration is the appropriate step.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_bler-v3
- **Verification**: None -> None => REJECT (The regeneration candidate for component 'dl_bler' was rejected by the deterministic verification gate because the FidelityScore could not be computed for evaluation. Under the system's fail-safe adaptation rules, if either the baseline or candidate fidelity score cannot be evaluated—in this case due to insufficient rolling history in the evaluation window—the system cannot verify any improvement exceeding the required verification_delta threshold of 0.01. Consequently, the candidate is safely rejected and production remains unchanged. (1. Rejection triggered by fidelity_improved check failure: FidelityScores (before/after) are uncomputable (None).
2. Candidate artifact exists and passed initial interface and valid prediction sanity checks (5,173 valid finite prediction rows).
3. Rejection is governed by fail-safe verification policy requiring explicit proof of fidelity improvement exceeding the 0.01 margin.))
- version dl_bler-v1 -> dl_bler-v1, duration=38.20s

### Event 8 — ul_bler — 2026-10-09T06:03:42.240332+00:00

- **Trigger**: type=external_drift, severity=0.6962159966701554, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: External drift detected in the uplink block error rate (ul_bler) with severity 0.6962. Telemetry indicates active cell load with 7 UEs and high requested PRB ratio (49.02%), leading to altered uplink interference and radio link propagation dynamics. The model structure for ul_bler remains valid, but its learned parameters are stale relative to current channel conditions.
- **Reasoning**: External drift in MAC-layer UL BLER is best addressed by recalibrating (retraining) the existing component pipeline on recent operational data to capture shifted interference distributions without changing model architecture.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=ul_bler-v2
- **Verification**: None -> None => REJECT (The candidate recalibration artifact for component 'ul_bler' (v2) passed all baseline integrity, schema consistency, and output validation checks. However, the candidate was rejected under the fail-safe acceptance policy because the FidelityScore before and/or after evaluation could not be computed due to insufficient rolling history in the evaluation dataset. Without a computable fidelity score margin exceeding the required verification delta (0.01), improvement cannot be deterministically verified, triggering an automatic safety rejection. (1. All initial deterministic validation checks passed (artifact presence, interface identity, row alignment at 5,173 samples, finite predictions, baseline evaluability).
2. Evaluation failed the 'fidelity_improved' gate because rolling history was insufficient to calculate the FidelityScore for comparison.
3. Safe operation was enforced via deterministic fail-safe REJECT, keeping production version 'ul_bler-v1' active.))
- version ul_bler-v1 -> ul_bler-v1, duration=13.99s

### Event 9 — ul_bler — 2026-10-09T06:04:02.396032+00:00

- **Trigger**: type=external_drift, severity=0.9734602747664127, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.91, llm_or_fallback=llm
- **Root cause**: The ul_bler component suffered severe external drift (severity 0.9735), and a previous recalibration attempt was REJECTED by the verification gate. A simple model retraining (recalibrate) was insufficient to handle the structural distribution shift in UL channel dynamics or telemetry patterns, requiring model architecture/pipeline regeneration.
- **Reasoning**: Given that recalibration has already failed to pass the deterministic verification gate for ul_bler under high external drift severity, repeating recalibration is unlikely to succeed. Regenerating the model component allows building a fresh model architecture better equipped to capture the shifted uplink block error rate distribution.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_bler-v3
- **Verification**: 0.7501567203221264 -> 0.7516539696943318 => REJECT (The candidate model for component 'ul_bler' generated via regeneration was rejected because it failed the deterministic fidelity improvement threshold. Although the candidate achieved a slight increase in FidelityScore from 0.7502 to 0.7517 (+0.0015), it did not meet the required verification_delta margin of 0.01 above the production baseline (required score >= 0.7602). All structural, interface, and execution safety checks passed successfully, but the marginal accuracy gain was insufficient to justify replacing the existing production model. (Fidelity increased slightly (+0.0015 from 0.7502 to 0.7517) but failed the required verification_delta threshold of 0.01. All sanity and structural checks passed (candidate_artifact_present, interface_identity_consistent, evaluation_ran, no_invalid_output, production_baseline_evaluable).))
- version ul_bler-v1 -> ul_bler-v1, duration=44.81s

### Event 10 — dl_throughput — 2026-10-09T06:04:51.027245+00:00

- **Trigger**: type=external_drift, severity=0.3139860020343368, fidelity_at_trigger=0.7638167280546575
- **Decision**: strategy=regenerate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: The component 'dl_throughput' suffered from external_drift resulting in degraded performance. A previous attempt to 'recalibrate' was REJECTED by the verification gate, indicating that simple retraining on recent data is insufficient to capture changed non-linear relations or feature shifts. Regenerating the model pipeline from scratch is required to adapt to the modified network conditions under high PRB demand and UE density.
- **Reasoning**: Previous recalibration attempt failed (REJECTED), demonstrating that retraining learned parameters on existing model architecture does not restore required fidelity delta. Regeneration is the mandated strategy when recalibration fails.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_throughput-v3
- **Verification**: 0.7716901728630785 -> 0.9065370845638588 => ACCEPT (The candidate model generation for component 'dl_throughput' was successfully accepted by the deterministic verification gate. The model achieved a FidelityScore of 0.9065, significantly exceeding the previous baseline fidelity of 0.7717 by 0.1348, which easily surpasses the required verification_delta margin of 0.01. Furthermore, all deterministic checks passed without issue: candidate artifacts were verified, interface standards were met, output predictions were finite across all 5173 test rows, and the model architecture aligns with downstream component requirements. (1. FidelityScore improved from 0.7717 to 0.9065 (a net increase of +0.1348, exceeding the minimum threshold delta of 0.01).
2. All 6 deterministic gate checks passed (candidate artifact presence, interface consistency, non-empty evaluation datasets, prediction finiteness, production baseline evaluability, and fidelity gain).
3. Re-generation of the 'dl_throughput' model successfully captured primary source relationships while maintaining system-wide target schema consistency.))
- version dl_throughput-v1 -> dl_throughput-v3, duration=45.84s

### Event 11 — ul_prb — 2026-10-09T06:05:43.362328+00:00

- **Trigger**: type=external_drift, severity=0.47130966518183137, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: The affected component ul_prb experienced external drift (severity 0.4713) under shifted uplink scheduling conditions, as indicated by a high requested PRB ratio (requested_prb_ratio_lag1 = 48.9435) and 7 active UEs per cell. The underlying model structure for uplink PRB allocation remains valid, but the learned feature weights are stale due to changed traffic demand patterns.
- **Reasoning**: External drift without architectural failure or unrepresented state variables indicates parameter staleness. Recalibrating the model over recent 24-hour training windows will align weights with current uplink PRB demand dynamics.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=ul_prb-v2
- **Verification**: 0.9919704478461338 -> 0.9697553280662201 => REJECT (The candidate recalibrated model for 'ul_prb' (ul_prb-v2.joblib) was REJECTED because it failed the deterministic fidelity improvement check. While all structural, isolation, and interface checks passed, the candidate achieved a FidelityScore of 0.9698, which is lower than the production baseline score of 0.9920. Under the system's adaptation policy (verification_delta=0.01), a candidate must outperform the production baseline by at least 0.01 to be accepted. (1. Baseline Fidelity: 0.9920 (ul_prb-v1).
2. Candidate Fidelity: 0.9698 (ul_prb-v2).
3. Required Threshold: > 1.0020 (Baseline + verification_delta 0.01).
4. Structural Verification: All prerequisite checks (artifact presence, interface identity consistency, evaluation execution, non-NaN/finite predictions) PASSED.
5. Outcome: REJECTED due to fidelity degradation (-0.0222 delta vs baseline).))
- version ul_prb-v1 -> ul_prb-v1, duration=15.11s

### Event 12 — ul_prb — 2026-10-09T06:06:08.357003+00:00

- **Trigger**: type=external_drift, severity=0.030346007662471197, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: The `ul_prb` component experienced an external drift trigger with a severity of 0.0303. A prior recalibration attempt was rejected by the verification gate because fidelity decreased from 0.9920 to 0.9698. Analysis of recent telemetry reveals high requested PRB ratio (48.94) and elevated uplink PRB consumption (0.3225) across 7 active UEs. Retraining on recent data failed to resolve the fidelity loss, indicating that simple recalibration of existing parameters is insufficient to capture the underlying shift in uplink resource scheduling dynamics.
- **Reasoning**: Recalibration was previously attempted and explicitly rejected due to decreased fidelity score. Per the digital twin adaptation policy, when parameter recalibration fails to improve model performance during persistent drift, the architecture/pipeline must be regenerated to capture non-linear relationships in uplink PRB allocation under high contention.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_prb-v3
- **Verification**: 0.9919704478461338 -> 0.8805385859406911 => REJECT (The regenerated candidate for 'ul_prb' (v3) was REJECTED because it failed the deterministic fidelity improvement gate. Although the candidate passed all prerequisite structural, interface, and execution sanity checks (artifact existence, interface identity consistency, evaluation execution, and finite prediction checks), its evaluation FidelityScore dropped to 0.8805 compared to the baseline production model's score of 0.9920. To meet the acceptance criteria, the candidate was required to achieve a FidelityScore strictly greater than 1.0020 (baseline + verification_delta of 0.01). (1. Quantitative Degradation: Candidate FidelityScore (0.8805) represented a significant performance drop of ~0.1114 compared to the existing production model (0.9920).
2. Failed Gate: Verification check 'fidelity_improved' evaluated to FAIL as FidelityScore_new (0.8805) <= FidelityScore_old (0.9920) + verification_delta (0.01).
3. Structural Integrity Passed: All pre-evaluation sanity checks (candidate_artifact_present, interface_identity_consistent, evaluation_ran, no_invalid_output, production_baseline_evaluable) PASSED across 5,173 evaluated target rows.))
- version ul_prb-v1 -> ul_prb-v1, duration=34.60s

### Event 13 — dl_prb — 2026-10-09T06:06:46.907706+00:00

- **Trigger**: type=external_drift, severity=0.09085271350425783, fidelity_at_trigger=None
- **Decision**: strategy=expand_scope, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version dl_prb-v1 -> None, duration=33.65s
- **Error**: no valid component design produced after 2 attempt(s): required_features ['dl_bler_pred'] are not available D1 columns

### Event 14 — dl_rlc_delay — 2026-10-09T06:07:24.369450+00:00

- **Trigger**: type=external_drift, severity=0.9314638547413545, fidelity_at_trigger=None
- **Decision**: strategy=expand_scope, source=full_system, confidence=0.91, llm_or_fallback=llm
- **Root cause**: An external drift event with high severity (0.9315) occurred on dl_rlc_delay. A prior adaptation attempt using 'regenerate' was REJECTed by the verification gate. Telemetry reveals high requested_prb_ratio_lag1 (48.868) and multiple active UEs (ue_count_cell_lag1=7.0), indicating severe RAN congestion/buffer bloat impacting RLC SDU delay. Because structural model regeneration failed to pass verification, the current feature schema lacks representation of cell-level scheduling pressure and buffer dynamics, requiring scope expansion to incorporate new features.
- **Reasoning**: Regeneration was previously attempted and rejected by the verification gate, ruling it out under typical retry dynamics when underlying feature representations are insufficient. Recalibration only retrains on existing stale parameters. The presence of high PRB request ratios and cell UE count shows that queueing delay in the RLC layer is driven by queueing/scheduling congestion metrics not fully captured by existing inputs, making 'expand_scope' the appropriate strategy.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=channel_quality-v1
- **Verification**: None -> None => REJECT (The candidate component 'channel_quality' was deterministically REJECTED due to multiple critical validation failures. First, it failed interface identity consistency because the model factory expected COMPONENT_NAME='dl_rlc_delay' with OUTPUT_FIELD='dl_rlc_delay_pred', whereas the candidate declared component='channel_quality' and output_field='wb_cqi_avg_lag1_pred'. Second, fidelity improvement could not be demonstrated: no baseline existed and the candidate's FidelityScore was non-computable due to insufficient rolling evaluation history, preventing valid score comparison under the required verification delta threshold. (1. Interface Identity Mismatch: Candidate output field ('wb_cqi_avg_lag1_pred') and name ('channel_quality') collided with the target factory contract ('dl_rlc_delay_pred' / 'dl_rlc_delay').
2. Fidelity Evaluation Failure: FidelityScore was non-computable owing to insufficient rolling window data, failing the fidelity improvement check.
3. Artifact & Output Integrity: Candidate model artifact generation and output finiteness checks passed successfully (5,173 rows evaluated), but structural and fidelity policy gates forced a rejection.))
- version dl_rlc_delay-v1 -> dl_rlc_delay-v1, duration=83.06s

### Event 15 — ul_prb — 2026-10-09T06:08:53.810558+00:00

- **Trigger**: type=external_drift, severity=0.29816309065742475, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: The affected component ul_prb experienced external drift (severity 0.2982), but a recent attempt to rebuild the model architecture via 'regenerate' failed verification (rejected due to a drop in fidelity score from 0.992 to 0.881). The high requested_prb_ratio_lag1 (49.02) and active UE count (7.0) indicate shifting traffic demand and uplink resource allocation patterns. Since structural regeneration degraded baseline performance, the existing model architecture remains valid but its parameters require retrain-based alignment with recent telemetry state.
- **Reasoning**: Regeneration failed previously with significant performance loss (fidelity reduced from 0.992 to 0.881), proving that structural changes are counterproductive. Recalibrating retrains model weights on recent 24-hour window data while preserving the model architecture, directly addressing the external telemetry drift.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=ul_prb-v4
- **Verification**: 0.9919704478461338 -> 0.9697553280662201 => REJECT (The recalibration candidate 'ul_prb-v4' was rejected by the deterministic verification gate because it failed the strict performance improvement check (fidelity_improved=FAIL). The existing production model 'ul_prb-v1' achieved a high FidelityScore of 0.9920 (0.99197), whereas the recalibrated candidate achieved a lower FidelityScore of 0.9698 (0.96976). To be accepted, the candidate needed to exceed the production score plus the required verification delta of 0.01 (a minimum target of ~1.0020). Because the candidate caused a degradation in overall prediction fidelity on the held-out set rather than an improvement, the system safely rejected the candidate and retained 'ul_prb-v1' in production. (1. All initial integrity, interface, artifact, and safety checks passed (candidate_artifact_present, interface_identity_consistent, evaluation_ran, no_invalid_output, production_baseline_evaluable).
2. Candidate FidelityScore (0.9698) fell short of the production FidelityScore (0.9920) by 0.0222, resulting in a failure of the fidelity_improved check against the verification_delta margin (0.01).
3. Rejection ensures the active production model remains on the higher-performing baseline ('ul_prb-v1') without risking model performance degradation.))
- version ul_prb-v1 -> ul_prb-v1, duration=12.73s
