# Baseline run report — mode `full_system`

- Replayed: OAI run(s) [8] (TEST split, speed=0)
- Seeds: [1]
- Model versions at start: {'dl_bler': 'dl_bler-v1', 'ul_bler': 'ul_bler-v1', 'dl_prb': 'dl_prb-v1', 'ul_prb': 'ul_prb-v1', 'dl_throughput': 'dl_throughput-v1', 'ul_throughput': 'ul_throughput-v1', 'dl_rlc_delay': 'dl_rlc_delay-v1'}
- Fidelity config: threshold=0.5, min_consecutive=3, rolling_window_length=200

## Per-component summary

| component | mean actual | mean predicted | RMSE | MAE | mean F_c |
|---|---|---|---|---|---|
| dl_bler | 0.0611 | 0.0615 | 0.0017 | 0.0008 | 0.7825 |
| dl_prb | 19.2771 | 19.3272 | 1.3004 | 0.7295 | 0.7802 |
| dl_rlc_delay | 0.2572 | 0.5390 | 0.6579 | 0.2818 | 0.7574 |
| dl_throughput | 0.2200 | 0.2209 | 0.0149 | 0.0068 | 0.8130 |
| ul_bler | 0.0075 | 0.0074 | 0.0005 | 0.0002 | 0.7500 |
| ul_prb | 0.9126 | 0.9108 | 0.0501 | 0.0288 | 0.8679 |
| ul_throughput | 0.0144 | 0.0145 | 0.0003 | 0.0002 | 0.8905 |

## Adaptation Events

### Event 1 — dl_prb — 2026-10-08T10:33:56.286298+00:00

- **Trigger**: type=external_drift, severity=0.9504636963259353, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: An external drift event with high severity (0.9505) occurred on the dl_prb component. Telemetry shows substantial PRB allocation demand (mean dl_prb of 127.24 with requested_prb_ratio_lag1 at 49.67 and 7 active cell UEs), leading to distributional shift in radio resource utilization. Since the model structure and feature dependencies remain sound, retraining on recent data via recalibration is required to re-align parameter weights with current traffic conditions.
- **Reasoning**: High severity external drift indicates that existing parameter weights for dl_prb no longer accurately map current radio resource request features to PRB allocation. Recalibrating the model parameters using a recent training window restores prediction fidelity without needing pipeline redesign.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_prb-v2
- **Verification**: None -> None => REJECT (The candidate recalibration model for 'dl_prb' was rejected by the deterministic verification gate because the FidelityScore before and/or after candidate execution could not be computed due to insufficient rolling history in the evaluation dataset. In accordance with the system's safety rules, when fidelity improvement cannot be verified, the gate applies a fail-safe REJECT decision to ensure unverified models are never promoted to production. (1. All structural and sanity checks passed: the artifact exists, interface identity is consistent, evaluation ran across 5173 rows, and predictions were finite.
2. The fidelity_improved check failed because rolling history was insufficient to calculate the metric, triggering an automatic fail-safe rejection.
3. Production baseline 'dl_prb-v1' remains active.))
- version dl_prb-v1 -> dl_prb-v1, duration=15.58s

### Event 2 — dl_prb — 2026-10-08T10:34:15.996928+00:00

- **Trigger**: type=external_drift, severity=0.9486494471372439, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.9, llm_or_fallback=llm
- **Root cause**: An external drift event with severe magnitude (0.9486) impacted component dl_prb. A previous recalibration attempt failed (REJECT), indicating that merely updating parameters on the existing model structure is insufficient to handle the structural distribution shift, requiring architecture regeneration.
- **Reasoning**: High drift severity and a failed prior recalibration rule out simple retraining. Regenerating the model architecture is necessary to capture changed PRB allocation relationships.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_prb-v3
- **Verification**: None -> None => REJECT (The candidate model dl_prb-v3 was rejected by the deterministic verification gate despite passing basic artifact existence, interface consistency, execution, and output validity checks. The gate failed the fidelity_improved check because the FidelityScore could not be computed for the evaluation dataset due to insufficient rolling history required to calculate the metric. Per the adaptation safety policy, when fidelity improvement cannot be quantitatively verified, the system enforces a fail-safe REJECT decision to ensure production models are not replaced by unverified candidates. (1. Artifact, interface consistency, evaluation execution (5,173 rows), and prediction sanity checks all PASSED.
2. The candidate failed the fidelity_improved requirement because FidelityScore before/after was not computable due to insufficient rolling history on the evaluation set.
3. The deterministic verification gate executed a fail-safe REJECT to prevent unverified model deployment into production.))
- version dl_prb-v1 -> dl_prb-v1, duration=31.30s

### Event 3 — ul_bler — 2026-10-08T10:34:50.856543+00:00

- **Trigger**: type=external_drift, severity=0.8277025938204418, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: An external drift event with high severity (0.8277) was detected on the uplink block error rate (ul_bler) component. Grounded in the recent network state summary, the cell maintains active traffic (e.g., ue_count_cell_lag1=7.0, requested_prb_ratio_lag1=49.07, pusch/pucch conditions) where uplink channel characteristics have shifted relative to the baseline training distribution. Because the existing model pipeline structure for ul_bler remains valid but parameter performance has degraded due to radio channel condition drift, retraining the model on recent telemetry data is required.
- **Reasoning**: The trigger type is external_drift affecting ul_bler with high severity. Since the fundamental network behavior and dependency structure for uplink BLER estimation remain unchanged, recalibrating (retraining) the existing model parameters against recent training windows is the appropriate and minimal intervention strategy.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_bler-v2
- **Verification**: None -> None => REJECT (The recalibration attempt for the 'ul_bler' model was rejected by the deterministic verification gate because the FidelityScore could not be computed. Specifically, the evaluation dataset (5,173 rows) provided insufficient rolling history required to calculate the baseline or candidate fidelity metrics accurately. In accordance with the system's safety policies, when fidelity improvement cannot be verified, the deterministic gate defaults to a fail-safe REJECT. (1. Artifact generation, interface identity checks, evaluation execution, and prediction validity checks all passed successfully.
2. The candidate model artifact 'ul_bler-v2.joblib' exists and produced 5,173 finite predictions matching the evaluation target size.
3. The evaluation failed at the 'fidelity_improved' step due to missing/insufficient rolling history required to calculate FidelityScore before and after adaptation, resulting in a safe rejection of the candidate model.))
- version ul_bler-v1 -> ul_bler-v1, duration=13.54s

### Event 4 — dl_bler — 2026-10-08T10:35:08.616477+00:00

- **Trigger**: type=external_drift, severity=0.5381433132192782, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: External drift trigger on dl_bler (severity 0.5381) indicates parameter decay/shift caused by changing radio link environment or traffic load patterns (requested_prb_ratio_lag1 = 48.94%, ue_count_cell_lag1 = 7). The underlying pipeline structure for predicting downlink block error rate remains valid, but model weights need retraining on recent telemetry.
- **Reasoning**: An external drift event with moderate severity on dl_bler is best resolved by recalibrating (retraining) the existing model parameters on the updated training window, as there is no indication that the feature schema or component architecture itself is insufficient.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_bler-v2
- **Verification**: None -> None => REJECT (The candidate recalibration for component 'dl_bler' (version dl_bler-v2) was REJECTED by the deterministic verification gate. Although the model artifact was valid, interface-consistent, and generated finite predictions across all 5,173 evaluation rows, the FidelityScore before and/or after could not be computed due to insufficient rolling evaluation history. Under the system's fail-safe adaptation policy, uncomputable fidelity metrics automatically fail the fidelity improvement check (fidelity_improved=FAIL), triggering an immediate rejection to prevent unverified model updates from entering production. (1. Artifact & Interface: Candidate artifact exists at dl_bler-v2.joblib and matches the required COMPONENT_NAME ('dl_bler') and OUTPUT_FIELD ('dl_bler_pred').
2. Execution & Data Quality: Model evaluation executed successfully across 5,173 rows with zero NaN/infinite prediction errors.
3. Failure Root Cause: Baseline/candidate FidelityScore computation failed due to insufficient rolling history, preventing verification against the required verification_delta (0.01) and resulting in a fail-safe REJECT.))
- version dl_bler-v1 -> dl_bler-v1, duration=16.01s

### Event 5 — dl_prb — 2026-10-08T10:35:28.375855+00:00

- **Trigger**: type=external_drift, severity=0.303194829291645, fidelity_at_trigger=None
- **Decision**: strategy=expand_scope, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version dl_prb-v1 -> None, duration=0.00s
- **Error**: ExpandScopeAgent.expand_scope() missing 1 required positional argument: 'example_component_factory'

### Event 6 — dl_bler — 2026-10-08T10:35:34.434939+00:00

- **Trigger**: type=external_drift, severity=0.13404169724716475, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.9, llm_or_fallback=llm
- **Root cause**: External drift occurred on dl_bler with severity 0.1340 under increased load conditions (requested PRB ratio lag 1 reaching 48.868 with 7 UEs in the cell). A prior recalibration attempt failed verification and was rejected, indicating that simply retraining existing model parameters on recent data is insufficient to reflect changed MAC-layer block error rate dynamics.
- **Reasoning**: Because recalibration was previously attempted and rejected by the deterministic verification gate, the underlying model architecture is unable to represent the new channel and load characteristics. Regenerating the model architecture from scratch is required.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_bler-v3
- **Verification**: None -> None => REJECT (The regenerated candidate model for the 'dl_bler' component was rejected by the deterministic acceptance gate due to a failure in the fidelity improvement check (fidelity_improved=FAIL). While structural, interface, and evaluation execution checks passed (including candidate artifact existence, field name consistency, prediction validity, and baseline evaluation), the FidelityScore before and/or after could not be computed due to insufficient rolling history during evaluation. In accordance with safety policies, uncomputable fidelity metrics trigger a fail-safe REJECT to prevent unverified candidate models from entering production. (1. All initial deterministic validation gates passed: artifact present, interface definitions matched ('dl_bler' / 'dl_bler_pred'), evaluation completed across 5,173 samples, and all candidate predictions were finite.
2. The production baseline (dl_bler-v1) was evaluable on the held-out set.
3. The evaluation failed the 'fidelity_improved' criteria because FidelityScore metrics could not be computed (insufficient rolling history in evaluation window), triggering a mandatory fail-safe rejection.))
- version dl_bler-v1 -> dl_bler-v1, duration=46.44s

### Event 7 — ul_bler — 2026-10-08T10:36:26.907723+00:00

- **Trigger**: type=external_drift, severity=0.2623133404418495, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: External drift was detected on ul_bler (severity 0.2623), and a previous recalibration attempt was REJECTED. Under lock policies and past failure of standard retraining, the existing model architecture/pipeline is no longer sufficient to handle the changed uplink radio link dynamics, necessitating architectural structural regeneration.
- **Reasoning**: Recalibration previously failed and was rejected, indicating that simply updating model parameters on recent data is insufficient to adapt to the external drift on ul_bler. Upgrading/rebuilding the model via structural regeneration is required.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_bler-v3
- **Verification**: None -> 0.7514312918375797 => REJECT (The candidate model generation for the 'ul_bler' component was rejected by the deterministic verification gate. Although the candidate artifact was generated successfully, maintained interface identity, and produced valid finite predictions across 5,173 test rows, the fidelity improvement check failed. Specifically, the production baseline fidelity score could not be computed due to insufficient rolling history, making it impossible to satisfy the required fidelity improvement criteria (verification_delta=0.01). Under the system's fail-safe adaptation safety rules, unverified candidates cannot be promoted to production. (1. Artifact & Evaluation Checks: Candidate model ul_bler-v3 was created and produced 5,173 valid finite predictions.
2. Interface Identity: COMPONENT_NAME ('ul_bler') and OUTPUT_FIELD ('ul_bler_pred') matched expected specifications.
3. Gate Failure: 'fidelity_improved' check failed because baseline fidelity_before was None (insufficient rolling history), triggering an automatic fail-safe REJECT.))
- version ul_bler-v1 -> ul_bler-v1, duration=53.04s

### Event 8 — ul_throughput — 2026-10-08T10:37:23.750020+00:00

- **Trigger**: type=external_drift, severity=0.2804087579860399, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: External drift trigger detected on ul_throughput (severity 0.2804) indicating that recent uplink channel conditions, traffic demand, or scheduling dynamics have shifted relative to the trained model's parameters. The existing feature set and component dependencies remain valid, but model parameters require retraining on recent telemetry to restore predictive fidelity.
- **Reasoning**: The external drift signal reflects parameter staleness under changing uplink network conditions rather than a structural failure of the component or missing domain features. Retraining via recalibrate is the standard response to re-align model weights with current telemetry distributions.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'digital_twin/internal:CLAUDE.md']
- **Candidate**: version=ul_throughput-v2
- **Verification**: None -> 0.9756895579241149 => REJECT (The recalibrated candidate model for 'ul_throughput' (v2) was rejected by the deterministic acceptance gate because the fidelity score improvement could not be verified. Specifically, the fidelity_improved check failed because either the production baseline or the candidate model lacked sufficient rolling evaluation history to compute a valid FidelityScore. Per the adaptation policy safety rules (verification_delta=0.01), when fidelity comparison is uncomputable, the system defaults to a fail-safe REJECT decision. (1. Artifact generation and interface identity checks passed successfully.
2. Both evaluation target and candidate prediction sets contained 5173 finite rows.
3. Baseline model 'ul_throughput-v1' evaluation was attempted, but insufficient rolling history prevented computing a comparable FidelityScore before/after.
4. Fail-safe policy triggered automatically to reject candidate 'ul_throughput-v2'.))
- version ul_throughput-v1 -> ul_throughput-v1, duration=14.25s

### Event 9 — dl_throughput — 2026-10-08T10:37:43.486509+00:00

- **Trigger**: type=external_drift, severity=0.9616571936637868, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: The affected component dl_throughput experienced a high severity external drift trigger (0.9617) alongside an insufficient history flag during fidelity evaluation. Telemetry shows high requested PRB ratio (48.868) and multiple active UEs (7.0) with very low throughput values (0.0004 Mbps), indicating shifts in traffic distribution and channel condition distribution that render existing learned model parameters stale without invalidating the core pipeline representation.
- **Reasoning**: Because the model pipeline and feature dependency structure (dl_throughput depending on dl_bler and dl_prb) remain valid for OAI KPI predictions, retraining the model on recent telemetry via recalibration is the appropriate strategy to restore predictive fidelity before considering structural regeneration.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_throughput-v2
- **Verification**: 0.7716901882749765 -> 0.771991313369906 => REJECT (The candidate model for the 'dl_throughput' component was rejected because it failed the deterministic fidelity improvement check. Although the recalibrated model achieved a minor fidelity increase from 0.7717 to 0.7720 (+0.0003), this improvement fell short of the required verification delta threshold of 0.01 (target score >= 0.7817). Consequently, the existing production version (v1) remains active. (Candidate fidelity (0.7720) vs production fidelity (0.7717) yielded a marginal gain of 0.0003, which is below the mandatory verification_delta margin of 0.01. All prerequisite deterministic checks passed (artifact presence, interface identity consistency, finite outputs, and baseline evaluability).))
- version dl_throughput-v1 -> dl_throughput-v1, duration=18.38s

### Event 10 — ul_bler — 2026-10-08T10:38:06.091484+00:00

- **Trigger**: type=external_drift, severity=0.5412268555474342, fidelity_at_trigger=0.7500354470768414
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: The affected component ul_bler experienced external drift resulting in degraded fidelity (0.7500). A previous attempt to resolve the issue using the 'regenerate' strategy failed and was rejected (yielding minimal gain to 0.7514, which failed the verification_delta margin of 0.01). Given that changing the model structure/architecture through regeneration did not pass the verification gate, the underlying model architecture remains valid but the model parameters have become stale due to shifting uplink channel characteristics and interference patterns under current cell load (ue_count_cell_lag1 = 7.0). Retraining the existing model pipeline on recent data via recalibration is the appropriate strategy.
- **Reasoning**: Regeneration was previously attempted for ul_bler and rejected due to failing the verification gate margin (fidelity_after 0.7514 vs required threshold margin). Consequently, regenerating the model architecture is inefficient. Recalibration is indicated to update the learned parameters on recent telemetry.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_bler-v4
- **Verification**: 0.7501567202959011 -> 0.7521847141991993 => REJECT (The recalibrated candidate model for 'ul_bler' was REJECTED because it failed the deterministic fidelity improvement threshold check. Although the candidate's fidelity score increased slightly from 0.7502 to 0.7522 (+0.0020), this improvement fell short of the required verification delta margin of 0.01 (which required a minimum score of 0.7602). All structural and execution checks passed successfully, but the marginal gain in performance did not justify replacing the existing production baseline. (Fidelity increased slightly from 0.750157 to 0.752185 (+0.002028), but failed to meet the required verification threshold delta of 0.01 over production. All prerequisite validation checks (artifact presence, interface identity, evaluation execution, finite predictions, and production baseline availability) passed.))
- version ul_bler-v1 -> ul_bler-v1, duration=13.91s

### Event 11 — dl_bler — 2026-10-08T10:38:24.292305+00:00

- **Trigger**: type=external_drift, severity=0.9699254132161326, fidelity_at_trigger=0.7504130668818338
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: The affected component `dl_bler` triggered an external drift alert with high severity (0.9699). A previous attempt to `regenerate` this component was REJECTED by the verification gate. Given that architectural regeneration failed to yield an acceptable candidate, the underlying pipeline structure remains valid, but the model parameters have degraded or drifted due to environmental shift (e.g., changes in requested PRB ratio or cell UE density). Recalibrating (retraining on recent data) is the appropriate strategy to update the parameters without risking another structural failure.
- **Reasoning**: A prior `regenerate` attempt failed and was rejected, indicating that changing model architecture/features is ineffective or prone to rejection. Recalibration retrains learned parameters on the updated telemetry window, resolving external drift while avoiding pipeline instability.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_bler-v4
- **Verification**: 0.7509339485271057 -> 0.8252174913753535 => ACCEPT (The recalibration of the dl_bler component succeeded as the new candidate model version (dl_bler-v4.joblib) demonstrated significant fidelity recovery over the existing production baseline (dl_bler-v1). Re-training the model on the updated telemetry window effectively mitigated baseline drift, improving the FidelityScore from 0.7509 to 0.8252 across 5,173 evaluation rows. The observed improvement of +0.0743 comfortably surpassed the mandatory verification_delta threshold of 0.01. Furthermore, all deterministic validation checks—including candidate artifact verification, interface field alignment (dl_bler_pred), finite output checks, and evaluation dataset completeness—passed without errors, justifying the candidate's automatic promotion. (1. Fidelity Score Improvement: Increased from 0.7509 to 0.8252 (+0.0743 margin, exceeding the 0.01 verification threshold).
2. Deterministic Verification Gates: Passed 100% of candidate checks including artifact existence (dl_bler-v4.joblib), non-null evaluation over 5,173 samples, and zero non-finite values.
3. Interface Sanity: Output field identity 'dl_bler_pred' remained consistent across production and candidate models.))
- version dl_bler-v1 -> dl_bler-v4, duration=15.30s

### Event 12 — ul_throughput — 2026-10-08T10:38:44.106846+00:00

- **Trigger**: type=external_drift, severity=0.11586561247077032, fidelity_at_trigger=0.38366840519297307
- **Decision**: strategy=regenerate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: The affected component `ul_throughput` has low fidelity (0.3837) following an external drift trigger (severity 0.1159). A previous `recalibrate` action was rejected by the verification gate despite achieving post-training fidelity, indicating that simple hyperparameter or coefficient re-fitting on stale architecture is insufficient. Furthermore, `ul_prb` exhibits degraded fidelity (0.3490) while uplink channel dynamics show significant resource demand (`requested_prb_ratio_lag1` = 48.8680) and multi-UE cell load (`ue_count_cell_lag1` = 7.0). Since `ul_throughput` depends downstream on `ul_prb` and `ul_bler` in the DT dependency graph, the existing pipeline structure cannot capture the structural shift under current load conditions, requiring model regeneration.
- **Reasoning**: A previous recalibration attempt failed verification gate acceptance. Regenerating the component architecture allows structural adjustments and non-linear re-mapping of upstream uplink features (`ul_prb` and `ul_bler`) under high PRB demand and multi-UE interference.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'digital_twin/internal:CLAUDE.md']
- **Candidate**: version=ul_throughput-v3
- **Verification**: 0.9919903354159357 -> 0.9865933594054506 => REJECT (The candidate model generated for the 'ul_throughput' component via regeneration was REJECTED by the deterministic verification gate. Although the candidate passed all functional, interface, and validity checks (such as artifact existence, non-null outputs, and shape matching), it failed the mandatory fidelity improvement check. The candidate achieved a FidelityScore of 0.9866, which is lower than the production baseline fidelity of 0.9920 and far below the required margin threshold (0.9920 + verification_delta of 0.01 = 1.0020). Consequently, the production model ul_throughput-v1 remains active. (1. All deterministic structural and execution sanity checks passed (candidate_artifact_present, interface_identity_consistent, evaluation_ran, no_invalid_output, production_baseline_evaluable).
2. The candidate fidelity score (0.9866) represents a degradation compared to the current production fidelity score (0.9920).
3. The candidate failed the threshold requirement (fidelity_improved=FAIL), as it did not surpass the current baseline by the required verification delta margin of 0.01.))
- version ul_throughput-v1 -> ul_throughput-v1, duration=39.26s

### Event 13 — dl_prb — 2026-10-08T10:39:27.732026+00:00

- **Trigger**: type=external_drift, severity=0.6130033010530405, fidelity_at_trigger=0.7571425131760543
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: External drift trigger on dl_prb with severity 0.6130 reflects parameter staleness under shifting radio resource demand conditions, as shown by high requested_prb_ratio_lag1 (48.8680) across 7 active cell UEs while physical dl_prb usage remains low (0.0265).
- **Reasoning**: The current model inputs (lagged PRB features, cell UE count, and requested PRB ratio) remain appropriate, but the model's learned weights need recalibration over the recent 24-hour training window to adapt to the shifted downlink PRB allocation dynamics.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_prb-v4
- **Verification**: 0.7499975343251859 -> 0.9243369427505485 => ACCEPT (The recalibration candidate for the dl_prb component was ACCEPTED because it demonstrated a substantial improvement in FidelityScore on the evaluation dataset. Prior to recalibration, the production baseline (dl_prb-v1) achieved a FidelityScore of 0.7500. The newly calibrated model (dl_prb-v4) achieved a FidelityScore of 0.9243, representing a metric gain of +0.1743. This improvement easily exceeds the required verification delta threshold of 0.01. Furthermore, all deterministic checks passed successfully, including candidate artifact verification, interface identity consistency, finite output validation, and complete target row alignment across all 5,173 evaluation samples. (1. High Fidelity Gain: FidelityScore increased significantly from 0.7500 to 0.9243, surpassing the required delta (0.01) by 0.1643 margin. 2. Deterministic Verification Passed: Model artifact exists, interface inputs/outputs match expectations, all 5,173 predicted values are finite, and predictions align with ground-truth evaluations. 3. Lifecycle Action: Candidate dl_prb-v4 is promoted to active deployment in accordance with adaptation policy safety rules.))
- version dl_prb-v1 -> dl_prb-v4, duration=14.64s

### Event 14 — dl_bler — 2026-10-08T10:39:48.855775+00:00

- **Trigger**: type=external_drift, severity=0.03959287666420286, fidelity_at_trigger=0.8250137829277755
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: An external drift event was detected on dl_bler (severity 0.0396). Recent network telemetry indicates high UE count (7.0 UEs in cell) and elevated requested PRB ratio (49.02), creating shifted radio link interference conditions. However, the previous recalibration for dl_bler was successful (increasing fidelity from 0.7509 to 0.8252), confirming that the model architecture remains sound and that retrain/recalibration on the updated feature distribution will suffice to adjust learned parameters to current channel drift.
- **Reasoning**: External drift on dl_bler reflects shifted channel dynamics under high load. Because the feature set and model pipeline previously demonstrated strong recovery under recalibration, updating the model parameters on recent 24-hour window data is the optimal and lowest-risk strategy.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'digital_twin/internal:CLAUDE.md']
- **Candidate**: version=dl_bler-v5
- **Verification**: 0.8251734842580631 -> 0.8251734842580631 => REJECT (The recalibration candidate for component 'dl_bler' was REJECTED because it failed the deterministic fidelity improvement check. Re-training on recent telemetry resulted in an identical FidelityScore of 0.8252 (fidelity_before = 0.8252, fidelity_after = 0.8252). To be accepted under the active adaptation safety rules, the candidate must outperform the current production model by at least the required verification_delta margin of 0.01 (requiring a score > 0.8352). Because there was zero improvement in prediction fidelity on the evaluation set, the production model 'dl_bler-v4' is retained. (1. All prerequisite deterministic checks passed: candidate artifact existed, interface identity matched 'dl_bler', evaluation ran on 5,173 rows with finite predictions, and the production baseline was evaluated on identical held-out data.
2. The candidate fidelity score (0.825173) was identical to the production baseline (0.825173), yielding a net improvement of 0.0000, which fell short of the required verification_delta threshold (+0.01).
3. Retaining the existing production version 'dl_bler-v4' prevents unnecessary model churn and ensures model stability when recalibration yields no accuracy gains.))
- version dl_bler-v4 -> dl_bler-v4, duration=15.33s
