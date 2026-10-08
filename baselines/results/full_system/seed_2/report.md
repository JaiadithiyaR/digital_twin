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
| dl_rlc_delay | 0.2572 | 0.5759 | 0.6615 | 0.3187 | 0.7934 |
| dl_throughput | 0.2200 | 0.2209 | 0.0149 | 0.0068 | 0.8130 |
| ul_bler | 0.0075 | 0.0074 | 0.0005 | 0.0002 | 0.7500 |
| ul_prb | 0.9126 | 0.9108 | 0.0501 | 0.0288 | 0.8897 |
| ul_throughput | 0.0144 | 0.0145 | 0.0003 | 0.0002 | 0.5448 |

## Adaptation Events

### Event 1 — dl_bler — 2026-10-08T10:41:11.517689+00:00

- **Trigger**: type=external_drift, severity=0.2984911434141233, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: External drift was detected on component dl_bler with severity 0.2985 (dl_bler mean at 0.3094, reflecting high downlink block error rate under elevated cell load with ue_count_cell_lag1=7 and dl_prb=127.24). The current model structure remains valid for representing MAC-layer block error rates, but its parameters require retraining on recent data to adapt to shifted channel conditions.
- **Reasoning**: External drift on dl_bler without structural pipeline failure indicates parameter staleness rather than architecture invalidity. Recalibration is the minimal and most appropriate strategy to re-fit model weights on recent 24-hour training data while maintaining the established dependency graph.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_bler-v2
- **Verification**: None -> None => REJECT (The candidate recalibrated model for 'dl_bler' (version dl_bler-v2.joblib) passed basic sanity and interface verification checks, including artifact existence, schema identity consistency, non-null output evaluation, and baseline alignment. However, the candidate failed the mandatory fidelity improvement check because the FidelityScore before and/or after evaluation could not be computed due to insufficient rolling window history in the evaluation set. Under system safety rules, an uncomputable fidelity delta triggers a deterministic fail-safe REJECT decision to ensure unverified models are never promoted to production. (Candidate artifact and interface verification checks passed (PASS).
FidelityScore was uncomputable due to insufficient rolling evaluation history, triggering the safety rule fidelity_improved=FAIL.
Deterministic verification gate executed a fail-safe REJECT to prevent unverified candidate promotion.))
- version dl_bler-v1 -> dl_bler-v1, duration=14.14s

### Event 2 — ul_throughput — 2026-10-08T10:41:29.771606+00:00

- **Trigger**: type=external_drift, severity=0.0919159421350969, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: The external drift trigger on ul_throughput indicates statistical drift in the underlying uplink data distribution, likely due to shifting traffic patterns or UL PRB allocation dynamics under high requested PRB ratio (49.77). The current model structure remains valid, but its parameters require retrain recalibration on recent operational telemetry.
- **Reasoning**: External drift without fundamental structural component failure or schema missingness is best addressed by retraining model weights via recalibration on recent telemetry windows.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=ul_throughput-v2
- **Verification**: None -> None => REJECT (The recalibration candidate for ul_throughput was rejected because the fidelity score improvement could not be deterministically verified. Specifically, the FidelityScore before and/or after candidate evaluation was non-computable due to insufficient rolling history within the evaluation dataset, triggering the fail-safe rejection policy. (Candidate artifact creation, interface consistency, candidate evaluation execution, and prediction validity checks all PASSED. However, fidelity_improved check FAILED due to uncomputable rolling fidelity metrics, preventing verification against the required verification_delta margin.))
- version ul_throughput-v1 -> ul_throughput-v1, duration=14.29s

### Event 3 — dl_bler — 2026-10-08T10:41:49.054692+00:00

- **Trigger**: type=external_drift, severity=0.18790107336660344, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: The dl_bler component experienced an external drift trigger (severity 0.1879). A previous recalibration attempt was rejected by the verification gate, indicating that simple parameter tuning on recent data is insufficient to adapt to the changed radio channel dynamics and BLER behavior. Regeneration of the underlying model architecture is required.
- **Reasoning**: Recalibration was already attempted for dl_bler and failed verification gate checks. According to systemic adaptation guidelines, when recalibration fails to improve fidelity following a drift event, the model structure itself is likely inadequate for the new signal conditions, necessitating component regeneration.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_bler-v3
- **Verification**: None -> None => REJECT (The regeneration candidate for component 'dl_bler' (v3) was rejected because the system was unable to verify an improvement in fidelity score. Although all static candidate validation checks passed (artifact presence, interface identity, evaluation completeness, and finite output), the FidelityScore could not be computed for both production and candidate versions due to insufficient rolling history on the held-out evaluation set. Adhering to the fail-safe verification protocol, the deterministic gate rejected the candidate. (1. Artifact check, interface identity consistency, evaluation execution, and non-invalid output checks all passed successfully for candidate artifact dl_bler-v3.joblib.
2. The check 'fidelity_improved' failed because FidelityScore before and/or after evaluation was not computable due to lack of required rolling history in the evaluation dataset.
3. Under the system's fail-safe adaptation safety rules, unverified candidates cannot be promoted, resulting in an automatic deterministic REJECT verdict.))
- version dl_bler-v1 -> dl_bler-v1, duration=33.49s

### Event 4 — dl_bler — 2026-10-08T10:42:26.214900+00:00

- **Trigger**: type=external_drift, severity=0.2749693679060381, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: The affected component `dl_bler` triggered an external drift alert (severity 0.2750) under ongoing channel fluctuations. A previous complex `regenerate` attempt was REJECTED by the verification gate. Given that the feature pipeline structure remains correct but parameters have drifted under recent radio conditions (DL BLER mean 0.1763, PRB load ~58.9), recalibrating the model parameters on recent 24-hour training window data is the safest and most effective adaptation strategy.
- **Reasoning**: Prior regeneration attempt failed verification, indicating that structural redesign or pipeline rewrite is ineffective or overfitting. Recalibration retrains model coefficients on recent telemetry, recovering prediction accuracy without introducing code syntax or boundary regression risks.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_bler-v4
- **Verification**: None -> None => REJECT (The candidate model dl_bler-v4 was rejected because it failed the fidelity improvement check (fidelity_improved=FAIL). Specifically, the FidelityScore could not be computed for either the production baseline or the candidate model due to insufficient rolling history within the evaluation set (5,173 rows). Per safety policy, when a metric cannot be computed to verify improvement over the 0.01 verification_delta margin, a fail-safe REJECT decision is enforced. (1. Artifact presence, interface consistency, evaluation execution, and prediction validity all passed.
2. The candidate failed the critical deterministic check `fidelity_improved` because rolling history was insufficient to calculate the FidelityScore.
3. Safe operational standards require verifiable quantitative improvement over production, making REJECT the required fail-safe outcome.))
- version dl_bler-v1 -> dl_bler-v1, duration=12.33s

### Event 5 — dl_rlc_delay — 2026-10-08T10:42:42.875904+00:00

- **Trigger**: type=external_drift, severity=0.15006226330533612, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: External drift detected in the dl_rlc_delay KPI model due to shift in operational network state (e.g., elevated requested_prb_ratio_lag1 of 48.94 and 7 active UEs in cell impacting downlink queueing dynamics). The model's feature space and dependency graph (dl_throughput and dl_bler inputs) remain structurally valid, but model weights need parameter re-estimation on recent data.
- **Reasoning**: An external drift event was triggered for dl_rlc_delay with no structural change in telemetry features or schema invalidation. According to operational policies, retraining the current model architecture on the updated 24-hour training window (recalibrate) is the initial and least disruptive action to resolve parametric drift.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_rlc_delay-v2
- **Verification**: None -> None => REJECT (The recalibration attempt for component 'dl_rlc_delay' was REJECTED by the deterministic verification gate. Although the candidate model artifact ('dl_rlc_delay-v2.joblib') was properly generated, interface-compatible, and yielded 5,173 valid finite predictions, the evaluation failed the fidelity improvement check. Specifically, the FidelityScore before and/or after evaluation could not be calculated due to insufficient rolling history within the evaluation dataset. Because the deterministic rule requires a candidate to exceed the baseline FidelityScore by at least verification_delta=0.01, an uncomputable fidelity score triggers a fail-safe REJECT decision. (1. Artifact & Interface Check: PASS — candidate artifact exists, non-empty, and has consistent output fields ('dl_rlc_delay_pred').
2. Evaluation Sanity: PASS — candidate generated 5,173 finite predictions matching the target row count without NaNs or infs.
3. Baseline Comparison: FAIL — fidelity_improved check failed because baseline/candidate FidelityScore could not be computed due to insufficient rolling evaluation window history, triggering a deterministic fail-safe REJECT.))
- version dl_rlc_delay-v1 -> dl_rlc_delay-v1, duration=12.51s

### Event 6 — dl_prb — 2026-10-08T10:42:59.588284+00:00

- **Trigger**: type=external_drift, severity=0.6692972985745202, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: The external drift trigger on 'dl_prb' with severity 0.6693 indicates that recent telemetry patterns (such as a high requested_prb_ratio_lag1 of 48.8680 across 7 active UEs) have shifted from the original baseline distribution. However, the existing model architecture and features remain valid for predicting PRB allocation, meaning the issue stems from stale model parameters under the shifted traffic load rather than missing network dimensions or structural model failure.
- **Reasoning**: Because the existing feature schema (including lagged features, rolling stats, and requested_prb_ratio) already accounts for cell load and PRB demand, recalibrating the 'dl_prb' model on a recent 24-hour training window will adjust its learned weights to match the shifted downlink resource consumption patterns without requiring architectural regeneration or feature scope expansion.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_prb-v2
- **Verification**: None -> None => REJECT (The recalibration candidate for 'dl_prb' (version dl_prb-v2) was rejected by the deterministic verification gate. Although the model artifact was generated successfully, passed interface consistency checks, and produced valid finite predictions over all 5,173 test rows, the fidelity score evaluation failed. Specifically, the FidelityScore before and/or after could not be computed due to insufficient rolling history required for the metric calculation. Under the system's fail-safe adaptation policy, unverified fidelity changes result in an automatic rejection to prevent unvalidated models from entering production. (1. Artifact & Schema Validation: Candidate artifact exists at the designated path, maintains consistent interface identifiers ('dl_prb' / 'dl_prb_pred'), and yielded 5,173 valid finite prediction rows.
2. Verification Failure Cause: Fidelity improvement evaluation failed (fidelity_improved=FAIL) because fidelity metrics could not be calculated due to inadequate rolling history length on the evaluation slice.
3. Final Deterministic Action: Fail-safe safety rules triggered a deterministic REJECT decision, preserving the production baseline (dl_prb-v1).))
- version dl_prb-v1 -> dl_prb-v1, duration=13.94s

### Event 7 — ul_throughput — 2026-10-08T10:43:17.636096+00:00

- **Trigger**: type=external_drift, severity=0.9674359524936766, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: The affected component `ul_throughput` experienced severe external drift (severity 0.9674). A prior recalibration attempt failed (REJECT), indicating that retrain-in-place parameter tuning is insufficient to capture the altered underlying uplink traffic or channel dynamics. Under high PRB demand (requested_prb_ratio_lag1 = 48.868) and cell load (ue_count_cell_lag1 = 7), the existing model structure cannot adequately represent the behavior, necessitating model architecture regeneration.
- **Reasoning**: Recalibration was previously attempted and rejected for this component. Since parameter retrain failed to achieve verification acceptance during severe drift, policy dictates regenerating the component structure/pipeline.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_throughput-v3
- **Verification**: None -> None => REJECT (The regenerated candidate for 'ul_throughput' (version v3) was rejected by the deterministic verification gate despite passing all basic structural, interface, and output validity checks. Although the model artifact was validly created, evaluation was executed successfully over 5,173 held-out rows, and all predicted values were finite, the fidelity improvement check failed. Specifically, the FidelityScore before and/or after could not be reliably computed due to insufficient rolling history in the evaluation window. Under the system's strict safety policy, uncomputable fidelity metrics prevent verification of the required +0.01 improvement threshold, triggering an automatic fail-safe REJECT decision to preserve production stability. (1. All initial deterministic pre-checks passed: artifact presence, interface contract identity (component 'ul_throughput' / target 'ul_throughput_pred'), evaluation execution (5,173 rows), prediction validity, and baseline evaluability. 2. Failure occurred at the fidelity gate (`fidelity_improved: FAIL`) because rolling history was insufficient to calculate the FidelityScore for comparison against production version v1. 3. System deterministic policy strictly enforces REJECT when fidelity gain cannot be verified quantitatively.))
- version ul_throughput-v1 -> ul_throughput-v1, duration=39.35s

### Event 8 — ul_prb — 2026-10-08T10:44:00.926237+00:00

- **Trigger**: type=external_drift, severity=0.39162483308002616, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: The affected component `ul_prb` experienced an external drift event with a severity of 0.3916. The recent network state telemetry indicates high resource contention or demand changes with `requested_prb_ratio_lag1` reaching 49.019 and an active cell UE count of 7.0. While `ul_prb` has insufficient history score data, the feature schema and dependency topology show that `ul_prb` is a base KPI component relying on lagged features. Because the underlying network structure and feature dependencies remain unchanged, retraining the existing model parameters on recent data will capture the updated distribution of uplink PRB allocation without needing pipeline architectural changes or scope expansion.
- **Reasoning**: External drift on `ul_prb` is caused by shifts in telemetry distributions (e.g., changes in user count and PRB demand). Recalibration allows the existing model pipeline to adapt its parameters to these shifted distributions using recent data windows.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=ul_prb-v2
- **Verification**: None -> None => REJECT (The candidate recalibration for the 'ul_prb' component was rejected by the deterministic verification gate because fidelity improvement could not be verified (`fidelity_improved=FAIL`). While the candidate model artifact was successfully generated, passed interface consistency checks, and produced valid finite predictions, the FidelityScore before and/or after the evaluation window could not be computed due to insufficient rolling evaluation history. Following the system's fail-safe adaptation policy, any candidate whose relative fidelity gain cannot be mathematically confirmed over the required threshold must be rejected. (1. All structural and sandbox checks passed (candidate artifact present at /home/harie06/dt_adaptation/baselines/results/full_system/seed_2/models/ul_prb/ul_prb-v2.joblib, valid interface identity, finite predictions on 5173 rows).
2. Production baseline was evaluable on the held-out dataset.
3. Verification failed on the `fidelity_improved` criterion due to missing/insufficient rolling evaluation history required to calculate the FidelityScore delta.
4. Fail-safe deterministic policy strictly enforced REJECT to prevent unverified model deployment into production.))
- version ul_prb-v1 -> ul_prb-v1, duration=13.41s

### Event 9 — dl_rlc_delay — 2026-10-08T10:44:18.982824+00:00

- **Trigger**: type=external_drift, severity=0.5110659735695771, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: The component 'dl_rlc_delay' experienced external drift with severity 0.5111. A previous adaptation attempt using 'recalibrate' was REJECTED by the verification gate, indicating that simply retraining learned parameters on existing model pipelines is insufficient to overcome feature drift or architectural mismatch. Therefore, the pipeline architecture itself must be regenerated.
- **Reasoning**: A prior recalibration attempt failed to improve model fidelity above the required verification margin. Selecting 'regenerate' allows rebuilding the candidate model pipeline to better capture the changed RLC queueing and transmission delay dynamics under updated traffic patterns.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_rlc_delay-v3
- **Verification**: 0.7574016430991197 -> 0.7932722481167573 => ACCEPT (The regenerate adaptation for the 'dl_rlc_delay' component was ACCEPTED after successfully passing all deterministic verification checks. The newly regenerated candidate model achieved a FidelityScore of 0.7933 compared to the production baseline of 0.7574 on the same 5,173 held-out evaluation samples. This yield a fidelity improvement of +0.0359, comfortably exceeding the required verification delta threshold of 0.01. (1. Quantitative Improvement: Fidelity score increased from 0.7574 to 0.7933 (+0.0359 improvement vs 0.01 required delta).
2. Verification Gates: All deterministic checks passed, including model artifact presence (dl_rlc_delay-v3.joblib), component/interface identity consistency (dl_rlc_delay_pred), evaluation dataset size alignment (5,173 rows), and numeric validity (all finite outputs).
3. Component Context: 'dl_rlc_delay' measures RAN queueing/transmission delay in ms and depends downstream on predicted 'dl_throughput' and 'dl_bler'.))
- version dl_rlc_delay-v1 -> dl_rlc_delay-v3, duration=66.57s

### Event 10 — dl_rlc_delay — 2026-10-08T10:45:31.654695+00:00

- **Trigger**: type=external_drift, severity=0.7755639424726894, fidelity_at_trigger=0.7933849325940784
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version dl_rlc_delay-v3 -> None, duration=0.10s
- **Error**: HistGradientBoostingRegressor.__init__() got an unexpected keyword argument 'n_estimators'

### Event 11 — ul_bler — 2026-10-08T10:45:36.022500+00:00

- **Trigger**: type=external_drift, severity=0.4709098854157575, fidelity_at_trigger=0.7500354470768414
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: The uplink block error rate (ul_bler) experienced external drift (severity 0.4709) causing component fidelity to drop to 0.7500. Telemetry shows a high cell UE count (7 UEs) and elevated uplink requested PRB ratio (48.94), shifting channel conditions and interference dynamics. The current model architecture remains suitable, but its learned parameters require retraining on recent data to adapt to the changed uplink interference profile.
- **Reasoning**: External drift in ul_bler is driven by changing cell density and PRB contention rather than structural model invalidity. Recalibrating (retraining) the existing model on recent 24-hour window data will restore fidelity without requiring full model regeneration or scope expansion.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_bler-v2
- **Verification**: 0.7501567202959011 -> 0.7521847141991993 => REJECT (The recalibrated candidate model for the 'ul_bler' component was rejected because it failed the deterministic fidelity improvement threshold test. Although the candidate achieved a slight improvement in FidelityScore (0.7522 vs. the production baseline score of 0.7502), the margin (+0.0020) fell short of the mandatory verification_delta requirement of 0.01. Per system safety rules, a candidate must demonstrate a significant improvement beyond the margin of error before replacing a production component. (1. All safety and structural checks (artifact existence, interface identity, evaluation completeness, finite output, baseline comparability) passed successfully.
2. FidelityScore increased slightly from 0.750156 to 0.752185 (+0.002028).
3. Rejection occurred strictly due to the fidelity_improved check failing, as the candidate failed to achieve the required +0.01 verification_delta improvement (target required >= 0.760157).))
- version ul_bler-v1 -> ul_bler-v1, duration=12.79s

### Event 12 — ul_bler — 2026-10-08T10:45:53.025638+00:00

- **Trigger**: type=external_drift, severity=0.10720730845358806, fidelity_at_trigger=0.7500000029308993
- **Decision**: strategy=regenerate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: The uplink block error rate (ul_bler) experiencing external drift showed insufficient fidelity improvement following a recalibration attempt (fidelity changed from 0.7501 to 0.7521, failing the required verification delta of 0.01 and leading to a REJECT decision). Retraining parameter weights alone is inadequate to capture the non-linear MAC-layer block error dynamics under the changed network state, indicating the underlying model architecture or feature processing pipeline needs to be regenerated.
- **Reasoning**: Recalibration was previously attempted for ul_bler and was REJECTED by the verification gate because the fidelity score improvement (0.0020) fell well below the required threshold of 0.01. According to system adaptation guidelines, when simple parameter retraining fails to restore component fidelity, the next step is to regenerate the component architecture.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_bler-v3
- **Verification**: 0.7501567202959011 -> 0.7512947253187687 => REJECT (The candidate model 'ul_bler-v3' generated during the 'regenerate' adaptation attempt was rejected by the deterministic verification gate. Although the candidate passed all structural, interface, and validity checks (candidate artifact present, interface identity consistent, evaluation ran without errors, and non-nan outputs), it failed the mandatory fidelity improvement check. Specifically, the candidate achieved a Fidelity Score of 0.7513 compared to the production baseline's 0.7502. Because the improvement of 0.0011 fell below the required verification delta threshold of 0.01 (requiring a score of at least 0.7602), the production model remains active. (1. Marginal Improvement Insufficient: The candidate model showed a negligible fidelity gain (+0.0011) over the existing production baseline (0.75015 vs 0.75129), failing the verification delta hurdle (+0.01).
2. Passed All Sanity and Interface Checks: The regenerated pipeline maintained interface consistency (OUTPUT_FIELD='ul_bler_pred') and produced finite, valid predictions over all 5,173 test rows.
3. Baseline Maintained: In accordance with adaptation safety rules, the previous production model (ul_bler-v1) is retained to avoid unnecessary churn without significant accuracy gains.))
- version ul_bler-v1 -> ul_bler-v1, duration=69.72s

### Event 13 — ul_bler — 2026-10-08T10:47:07.391086+00:00

- **Trigger**: type=external_drift, severity=0.8844496736882935, fidelity_at_trigger=0.7500000029308993
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: The affected component ul_bler experienced external drift with a severity of 0.8844. A previous attempt to regenerate the model architecture was rejected because the verification result failed to meet the verification_delta margin (+0.0011 gain vs 0.01 required). Given that regenerating the model pipeline yielded minimal improvement and was rejected, the underlying model architecture remains adequate, but the learned model parameters have grown stale due to shifting uplink channel characteristics (indicated by high requested PRB ratio of 48.87 and cell UE load). Recalibrating (retraining on recent data) is the appropriate next strategy.
- **Reasoning**: Regeneration was recently tried and failed the deterministic verification gate (delta 0.0011 < 0.01 threshold). Re-trying regeneration would violate adaptation efficiency. Recalibration is selected to update stale parameters over the recent training window.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_bler-v4
- **Verification**: 0.7501567202959011 -> 0.7521847141991993 => REJECT (The candidate recalibration for the ul_bler component was rejected because it failed to satisfy the fidelity improvement threshold required by the deterministic acceptance gate. Although the candidate model (ul_bler-v4) showed a minor increase in FidelityScore from 0.7502 to 0.7522 over the 5,173 held-out evaluation samples, the margin of improvement (+0.0020) fell short of the mandatory verification_delta threshold of 0.01 (required minimum score: ~0.7602). All artifact, schema, and numerical sanity checks passed successfully, but under system adaptation safety rules, a candidate model must demonstrate significant improvement over the production baseline (ul_bler-v1) to justify promotion. (1. All structural and numerical checks passed: candidate artifact was non-empty, output schemas matched, and predictions were finite across all 5,173 evaluation rows.
2. The candidate achieved a FidelityScore of 0.75218, improving slightly over the production score of 0.75016 (+0.0020).
3. Rejection occurred strictly at the deterministic threshold check (fidelity_improved=FAIL) because the score delta (+0.0020) was below the required verification_delta of 0.01.))
- version ul_bler-v1 -> ul_bler-v1, duration=13.21s

### Event 14 — ul_throughput — 2026-10-08T10:47:24.809187+00:00

- **Trigger**: type=external_drift, severity=0.8492363236144346, fidelity_at_trigger=0.9939079628115425
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: External drift trigger on ul_throughput with high severity (0.8492) indicates a shift in uplink channel conditions and scheduling dynamics, marked by a high requested PRB ratio (48.868) across 7 cell UEs. A prior attempt to regenerate the model architecture was rejected by the verification gate. Because the underlying feature schema already includes the necessary lagged MAC and PRB metrics, the degradation is caused by stale model parameters under the shifted operational distribution rather than structural model inadequacy.
- **Reasoning**: Given that a previous 'regenerate' adaptation was rejected and the required telemetry features (e.g., requested_prb_ratio_lag1, ue_count_cell_lag1) are present in the dataset schema, retraining learned parameters on recent telemetry via 'recalibrate' is the most effective and safe strategy to restore fidelity without repeating structural failure.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'digital_twin/internal:CLAUDE.md']
- **Candidate**: version=ul_throughput-v4
- **Verification**: 0.9919903354159357 -> 0.9898207980221527 => REJECT (The candidate model 'ul_throughput-v4' yielded a FidelityScore of 0.9898 during evaluation on 5,173 held-out rows, failing to outperform the existing production version 'ul_throughput-v1' (FidelityScore 0.9920). Because the candidate decreased overall fidelity and fell short of the required verification_delta margin (0.01), it was rejected by the deterministic gate. (1. Baseline evaluation completed on 5,173 valid samples with finite outputs across all checks.
2. Recalibration produced a FidelityScore of 0.9898, which is lower than the production score of 0.9920.
3. The fidelity_improved requirement (FidelityScore_new > FidelityScore_old + verification_delta) failed, triggering an automatic REJECT verdict.))
- version ul_throughput-v1 -> ul_throughput-v1, duration=13.51s

### Event 15 — dl_prb — 2026-10-08T10:47:46.734694+00:00

- **Trigger**: type=external_drift, severity=0.5165781941249967, fidelity_at_trigger=0.757266901219243
- **Decision**: strategy=expand_scope, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version dl_prb-v1 -> None, duration=0.00s
- **Error**: ExpandScopeAgent.expand_scope() missing 1 required positional argument: 'example_component_factory'
