# Baseline run report — mode `full_system`

- Replayed: OAI run(s) [8] (TEST split, speed=0)
- Seeds: [3]
- Model versions at start: {'dl_bler': 'dl_bler-v1', 'ul_bler': 'ul_bler-v1', 'dl_prb': 'dl_prb-v1', 'ul_prb': 'ul_prb-v1', 'dl_throughput': 'dl_throughput-v1', 'ul_throughput': 'ul_throughput-v1', 'dl_rlc_delay': 'dl_rlc_delay-v1'}
- Fidelity config: threshold=0.5, min_consecutive=3, rolling_window_length=200

## Per-component summary

| component | mean actual | mean predicted | RMSE | MAE | mean F_c |
|---|---|---|---|---|---|
| dl_bler | 0.0611 | 0.0614 | 0.0017 | 0.0008 | 0.7505 |
| dl_prb | 19.2771 | 19.3186 | 1.2997 | 0.7209 | 0.8042 |
| dl_rlc_delay | 0.2572 | 0.5390 | 0.6579 | 0.2818 | 0.7575 |
| dl_throughput | 0.2200 | 0.2209 | 0.0149 | 0.0068 | 0.8048 |
| ul_bler | 0.0075 | 0.0074 | 0.0005 | 0.0002 | 0.7500 |
| ul_prb | 0.9126 | 0.9108 | 0.0501 | 0.0288 | 0.8679 |
| ul_throughput | 0.0144 | 0.0145 | 0.0003 | 0.0002 | 0.8690 |

## Adaptation Events

### Event 1 — dl_bler — 2026-10-08T10:48:47.137639+00:00

- **Trigger**: type=external_drift, severity=0.2368105065960997, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: The affected component dl_bler experienced an external drift trigger (severity 0.2368) due to shifts in underlying RF link dynamics, as evidenced by an elevated mean dl_bler of 0.3094 alongside significant PRB allocation (dl_prb = 127.24) across 7 active UEs in the cell. The existing feature set and model pipeline structure remain valid, but the model parameters require updating to reflect recent telemetry trends.
- **Reasoning**: External drift on dl_bler indicates feature distribution shift rather than missing domain variables or structural invalidity. Recalibrating the model over recent training data will adjust model weights to match current channel conditions without requiring an architectural regeneration or feature scope expansion.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_bler-v2
- **Verification**: None -> None => REJECT (The recalibration attempt for component 'dl_bler' (candidate version dl_bler-v2) was REJECTED by the deterministic verification gate. Although the model artifact was generated, loaded successfully, passed interface consistency, and produced valid finite predictions on the evaluation target set (5,173 rows), the fidelity check failed (fidelity_improved=FAIL). Specifically, the FidelityScore could not be reliably computed for either the baseline or candidate due to insufficient rolling evaluation history. Under the system's fail-safe adaptation policy, unverified fidelity improvements cannot be promoted to production. (1. All prerequisite code and interface structural checks passed: artifact presence, schema identity ('dl_bler' / 'dl_bler_pred'), non-empty predictions, and production baseline evaluability.
2. The adaptation failed the 'fidelity_improved' rule because the FidelityScore before and/or after adaptation was non-computable due to inadequate rolling history length.
3. Safe operational standards dictate an automatic REJECT when fidelity delta cannot be deterministically verified against the required threshold (verification_delta=0.01).))
- version dl_bler-v1 -> dl_bler-v1, duration=12.51s

### Event 2 — dl_throughput — 2026-10-08T10:49:03.449220+00:00

- **Trigger**: type=external_drift, severity=0.5821620360643678, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: An external drift trigger was detected for dl_throughput with severity 0.5822. The recent network state shows high requested PRB ratio (49.77) and elevated cell UE count (7.0), indicating shift in radio resource demand and cell load distribution, which stale model parameters fail to accurately map to downlink throughput.
- **Reasoning**: The feature schema and dependency graph for dl_throughput remain valid, but parameter recalibration on recent training split data is sufficient to adjust for the environmental shift in PRB demand and UE density without requiring structural regeneration.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_throughput-v2
- **Verification**: None -> None => REJECT (The recalibration adaptation for the dl_throughput component was REJECTED by the deterministic verification gate. Although the candidate model artifact was successfully generated, met all interface signature checks, and produced valid finite predictions across all 5,173 test rows, the fidelity improvement check failed. Specifically, the FidelityScore before and/or after calibration could not be computed due to insufficient rolling history, triggering the system's fail-safe policy requiring proven fidelity improvement prior to deployment. (Candidate artifact exists and is structurally valid, passing all schema, interface, and output validity checks. The adaptation failed exclusively on the fidelity improvement criteria because FidelityScore metrics were uncomputable due to inadequate rolling evaluation context, leading to an automatic fail-safe rejection.))
- version dl_throughput-v1 -> dl_throughput-v1, duration=14.06s

### Event 3 — dl_rlc_delay — 2026-10-08T10:49:21.542707+00:00

- **Trigger**: type=external_drift, severity=0.479051298140834, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: External drift was detected on component 'dl_rlc_delay' with a severity of 0.4791. Recent telemetry shows elevated queueing/transmission delay (0.9126 ms) amidst high cell PRB demand (49.07 requested PRB ratio, 7 active UEs) and high downlink block-error rate (dl_bler = 0.3175). Since the core feature relationship between dl_throughput, dl_bler, and dl_rlc_delay remains structurally sound according to the dependency graph, the observed performance degradation is due to stale model parameters under changing traffic and channel conditions rather than a structural failure.
- **Reasoning**: Recalibrating the model on recent training data will adjust parameter weights to reflect current RLC queue dynamics and high BLER conditions without requiring an architectural rebuild or scope expansion.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_rlc_delay-v2
- **Verification**: None -> None => REJECT (The recalibrated candidate model for 'dl_rlc_delay' (version v2) successfully passed initial verification checks, including candidate artifact verification, interface identity matching, non-null prediction validation, and baseline alignment over 5,173 evaluation rows. However, the adaptation was deterministically rejected during the final performance gate because the FidelityScore before and/or after adaptation could not be calculated (fidelity_before=None, fidelity_after=None) due to insufficient rolling history in the evaluation dataset. Because the framework strictly enforces a fail-safe rule requiring a verified improvement margin exceeding the verification_delta threshold of 0.01 before promoting a model, uncomputable fidelity metrics trigger an automatic REJECT decision. (1. Deterministic gate decision: REJECT due to fidelity_improved=FAIL.
2. All preliminary structural and interface checks passed (artifact presence, output field alignment, output finiteness, and baseline evaluability).
3. Both baseline and candidate FidelityScores were returned as None due to insufficient rolling window history in the evaluation evaluation set, preventing margin comparison against the 0.01 verification threshold.))
- version dl_rlc_delay-v1 -> dl_rlc_delay-v1, duration=14.42s

### Event 4 — dl_prb — 2026-10-08T10:49:40.006075+00:00

- **Trigger**: type=external_drift, severity=0.7345771514092145, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: The affected component dl_prb exhibits severe external drift (severity 0.7346). Recent network state data indicates high physical resource block usage (dl_prb = 58.9156) driven by high requested PRB ratio (48.9435) and elevated cell UE count (7.0). The model's existing feature structure is still valid for radio resource consumption prediction, but the underlying traffic and scheduling parameters have shifted relative to historical baseline data.
- **Reasoning**: External drift without fundamental structural changes in the KPI dependency chain or missing features indicates parameter staleness. Recalibrating the model on recent data windows will update the learned parameters to accurately reflect current PRB allocation dynamics.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_prb-v2
- **Verification**: None -> None => REJECT (The recalibration candidate for 'dl_prb' (v2) was rejected by the deterministic verification gate because the fidelity score improvement check failed (fidelity_improved=FAIL). Specifically, the FidelityScore before and/or after evaluation was not computable due to insufficient rolling history within the held-out evaluation set (5,173 rows). Per system safety policy, when fidelity improvement cannot be quantitatively verified, the system enforces a fail-safe REJECT decision. (1. Artifact & Interface Checks: Candidate artifact dl_prb-v2.joblib was successfully generated and interface identity consistency checks passed (component_name='dl_prb', output_field='dl_prb_pred').
2. Execution & Data Validity: Both evaluation target and candidate prediction sets contained 5,173 finite rows, successfully passing invalid output checks.
3. Gate Failure: The fidelity_improved check failed because rolling history was insufficient to calculate baseline or candidate FidelityScore metrics, triggering the fail-safe rejection rule (verification_delta=0.01 margin required).))
- version dl_prb-v1 -> dl_prb-v1, duration=12.92s

### Event 5 — dl_prb — 2026-10-08T10:49:56.529577+00:00

- **Trigger**: type=external_drift, severity=0.5167401826213637, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: The affected component `dl_prb` experienced an external drift trigger with a severity of 0.5167. A prior `recalibrate` strategy was executed and REJECTed by the deterministic verification gate. Furthermore, recent network telemetry shows a high requested PRB ratio (`requested_prb_ratio_lag1` = 48.9435) and active UEs (`ue_count_cell_lag1` = 7.0), while actual `dl_prb` usage remains low (0.0290). Because simple retraining (recalibration) failed to fix the performance gap, the model architecture/pipeline is no longer able to capture the relationship under this load regime, requiring component regeneration.
- **Reasoning**: Recalibration was previously attempted and rejected by the verification gate. According to adaptation policy, when parameter retraining (recalibrate) fails to improve model fidelity following drift, regenerating the component model pipeline from scratch is necessary to learn the updated structural relationships.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_prb-v3
- **Verification**: None -> None => REJECT (The regeneration attempt for component 'dl_prb' (candidate model dl_prb-v3.joblib) was REJECTED by the deterministic verification gate. Although the candidate model artifact was generated successfully, maintained interface identity, produced valid finite predictions, and completed evaluation, the fidelity check failed (fidelity_improved=FAIL). Specifically, the FidelityScore before and/or after evaluation could not be computed due to insufficient rolling history on the evaluated test set. Following the system's fail-safe adaptation policy, any candidate whose fidelity improvement cannot be explicitly verified must be rejected. (1. Artifact & Interface Check: Candidate artifact dl_prb-v3.joblib was successfully built and passed interface consistency checks (COMPONENT_NAME='dl_prb', output_field='dl_prb_pred').
2. Model Evaluation: 5,173 test rows were processed without invalid or non-finite outputs.
3. Fail-Safe Rejection Gate: FidelityScore computation returned non-computable (None) due to inadequate rolling history length, triggering a mandatory safety rejection under the deterministic verification delta policy (verification_delta=0.01).))
- version dl_prb-v1 -> dl_prb-v1, duration=31.88s

### Event 6 — dl_rlc_delay — 2026-10-08T10:50:32.591474+00:00

- **Trigger**: type=external_drift, severity=0.5867985714381407, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.91, llm_or_fallback=llm
- **Root cause**: An external drift event (severity 0.5868) affected 'dl_rlc_delay', likely driven by high cell load (requested_prb_ratio_lag1 = 48.87, ue_count = 7) and non-linear RLC SDU buffer queueing dynamics. A previous recalibration attempt was rejected during verification, indicating that simple parameter retraining on the existing model pipeline is insufficient to capture the changed RAN queueing behaviour.
- **Reasoning**: Because the prior recalibration attempt failed validation, retraining the current weights cannot resolve the structural fidelity drift. Regenerating the model architecture for dl_rlc_delay is required to learn updated functional representations of RAN delay under high resource utilization.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_rlc_delay-v3
- **Verification**: None -> None => REJECT (The regeneration candidate for the dl_rlc_delay component was rejected during deterministic verification due to an uncomputable fidelity score. While all preliminary structural checks passed—including artifact existence, interface consistency, finite output checks, and baseline evaluation—the system failed the fidelity improvement check (fidelity_improved=FAIL). Specifically, the fidelity score before and/or after adaptation could not be calculated due to insufficient rolling history across the evaluation period. Per the system's fail-safe adaptation policy, any candidate whose fidelity improvement cannot be deterministically verified must be rejected. (1. Preliminary deterministic validation passed: candidate model artifact dl_rlc_delay-v3.joblib existed, interface naming matched, predictions were non-null/finite across 5,173 evaluation rows, and production baseline dl_rlc_delay-v1 was evaluated on the same set.
2. Fidelity check failed: FidelityScore before/after could not be computed due to insufficient rolling window history in the evaluation dataset.
3. Fail-safe decision: Without verifiable fidelity improvement exceeding the required verification_delta (0.01), the candidate was deterministically rejected to ensure operational safety.))
- version dl_rlc_delay-v1 -> dl_rlc_delay-v1, duration=58.64s

### Event 7 — dl_bler — 2026-10-08T10:51:35.674124+00:00

- **Trigger**: type=external_drift, severity=0.28420116374879145, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: The affected component dl_bler experienced an external drift trigger (severity 0.2842). A previous recalibration attempt was REJECTED, indicating that retraining learned parameters on existing model architecture is insufficient to capture the underlying shift in radio link quality characteristics or interference dynamics. Regenerating the model architecture from scratch is necessary to adapt to the changed behavior.
- **Reasoning**: Recalibration previously failed and was rejected by the verification gate for dl_bler. Per adaptation escalation policies, when parameter retrain (recalibrate) fails to resolve drift, regenerating the model architecture is the appropriate next level strategy to handle persistent feature pattern shifts.
- **References**: ['digital_twin/internal:CLAUDE.md']
- **Candidate**: version=dl_bler-v3
- **Verification**: None -> None => REJECT (The deterministic verification gate rejected the regenerated candidate model (dl_bler-v3) because fidelity scores could not be successfully calculated. Specifically, the system encountered insufficient rolling history when attempting to compute FidelityScore for the baseline or candidate predictions across the 5,173 evaluation rows. In accordance with the fail-safe adaptation policy (fidelity_improved=FAIL), any candidate whose fidelity improvement cannot be definitively verified against the production model (dl_bler-v1) is safely rejected to ensure operational stability. (1. All initial deterministic validation checks passed, including candidate artifact non-emptiness, interface identity consistency ('dl_bler' / 'dl_bler_pred'), prediction finiteness, and dataset length alignment (5,173 rows).
2. The fidelity check failed (fidelity_improved=FAIL) due to uncomputable FidelityScore values resulting from insufficient rolling history context during evaluation.
3. Following safety protocols, unverified fidelity improvements trigger an automatic deterministic REJECT to prevent potentially unstable models from reaching production.))
- version dl_bler-v1 -> dl_bler-v1, duration=39.30s

### Event 8 — ul_bler — 2026-10-08T10:52:19.037884+00:00

- **Trigger**: type=external_drift, severity=0.6962159966701554, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: External drift was detected on uplink block error rate (ul_bler) with a severity of 0.6962. Telemetry indicates significant uplink resource requests (requested_prb_ratio_lag1 ~49.02, ue_count_cell_lag1 = 7) relative to low actual throughput. This mismatch in radio link conditions and uplink allocation trends has caused the existing ul_bler statistical model to experience distributional drift, making parameter recalibration over recent data necessary.
- **Reasoning**: External drift indicates that while the underlying dependency graph and feature set for uplink BLER remain valid, the learned parameters no longer match recent radio channel dynamics. Recalibrating (retraining on the 24-hour window) is the appropriate low-overhead intervention to restore model accuracy.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_bler-v2
- **Verification**: None -> None => REJECT (The recalibration attempt for the 'ul_bler' component was rejected by the deterministic verification gate. Although the candidate artifact was generated successfully, maintained interface consistency, and produced valid predictions across all 5,173 test rows, the fidelity evaluation check failed (fidelity_improved=FAIL). Specifically, the FidelityScore before and/or after evaluation could not be computed due to insufficient rolling history in the evaluation dataset. Per system safety rules, when fidelity improvement cannot be deterministically verified, the system defaults to a fail-safe REJECT decision. (1. Candidate model artifact 'ul_bler-v2.joblib' exists, is non-empty, and passed interface consistency checks.
2. Evaluation ran across 5,173 rows with zero NaN/infinite predictions from the candidate model.
3. Recalibration failed verification because FidelityScore was uncomputable (insufficient rolling history for metric calculation), triggering a fail-safe deterministic rejection.))
- version ul_bler-v1 -> ul_bler-v1, duration=12.76s

### Event 9 — ul_bler — 2026-10-08T10:52:38.408952+00:00

- **Trigger**: type=external_drift, severity=0.9734602747664127, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version ul_bler-v1 -> None, duration=128.10s
- **Error**: LLM code generation failed for 'ul_bler': transient transport error calling Google AI: [Errno -3] Temporary failure in name resolution

### Event 10 — dl_throughput — 2026-10-08T10:54:50.449078+00:00

- **Trigger**: type=external_drift, severity=0.3139860020343368, fidelity_at_trigger=0.7638167280546575
- **Decision**: strategy=regenerate, source=full_system, confidence=0.91, llm_or_fallback=llm
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version dl_throughput-v1 -> None, duration=116.35s
- **Error**: regeneration for 'dl_throughput' produced no accepted candidate after 2 attempt(s): stage=conformance error=ValueError: candidate.py defines no attribute named 'DlThroughputModel'

### Event 11 — ul_prb — 2026-10-08T10:57:14.007712+00:00

- **Trigger**: type=external_drift, severity=0.47130966518183137, fidelity_at_trigger=0.9993734489428555
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: The affected component ul_prb experienced an external drift trigger with a severity of 0.4713, despite currently maintaining a high per-component fidelity score of 0.9994. Telemetry indicates shift in uplink radio resource usage (ul_prb average at 0.3165 with requested_prb_ratio_lag1 at 48.9435 and ue_count_cell_lag1 at 7.0). The underlying feature relationships and model architecture for uplink PRB prediction remain sound, but the model parameters need updating to adapt to the recent shift in network traffic and resource request patterns.
- **Reasoning**: Because the model structure for ul_prb is intact and component fidelity remains high, a recalibration (retraining on the recent 24-hour window) is sufficient to adjust learned parameters to the detected external distribution shift without requiring architectural regeneration or feature scope expansion.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=ul_prb-v2
- **Verification**: 0.9919704478461338 -> 0.9697553280662201 => REJECT (The candidate recalibrated model for 'ul_prb' was REJECTED because it failed the required fidelity improvement threshold. The production baseline achieved a FidelityScore of 0.9920, whereas the candidate recalibrated model achieved a lower FidelityScore of 0.9698. According to the adaptation policy, a candidate must outperform the production model by at least the verification delta (0.01) to be accepted (requiring a score > 1.0020). Since the candidate's fidelity actually regressed by ~0.0222, the deterministic gate failed the fidelity_improved check and kept the production version in service. (1. Production model 'ul_prb-v1' fidelity score: 0.9920.
2. Candidate model 'ul_prb-v2' fidelity score: 0.9698.
3. Required acceptance threshold: 1.0020 (fidelity_before + verification_delta 0.01).
4. All structural, schema, and safety verification checks passed (candidate_artifact_present, interface_identity_consistent, evaluation_ran, no_invalid_output, production_baseline_evaluable).
5. The rejection was strictly driven by the deterministic fidelity_improved check failing due to performance regression.))
- version ul_prb-v1 -> ul_prb-v1, duration=14.23s

### Event 12 — ul_prb — 2026-10-08T10:57:32.371802+00:00

- **Trigger**: type=external_drift, severity=0.030346007662471197, fidelity_at_trigger=0.999722506119868
- **Decision**: strategy=regenerate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: The affected component ul_prb experienced external drift and a previous recalibration attempt was rejected by the verification gate because fidelity decreased (from 0.992 to 0.970). Given high requested_prb_ratio_lag1 (48.94) and cell UE count (7.0), the parameter-only recalibration is insufficient to capture the non-linear uplink resource demand shifts, requiring model regeneration.
- **Reasoning**: Recalibration previously failed to beat the verification threshold and lowered fidelity score. Under the system's policy, when recalibration fails to capture behavioral shift under external drift, regenerating the model architecture/pipeline is the appropriate strategy.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'digital_twin/internal:CLAUDE.md']
- **Candidate**: version=ul_prb-v3
- **Verification**: 0.9919704478461338 -> 0.8672875908375304 => REJECT (The regenerated candidate for 'ul_prb' was rejected because it failed the performance verification step. Although all functional, artifact, and interface checks passed, the candidate achieved a FidelityScore of 0.8673, which represents a noticeable performance regression compared to the production model's baseline FidelityScore of 0.9920. To be accepted, the candidate needed to exceed 1.0020 (baseline + verification_delta of 0.01). (1. Artifact, interface identity, and prediction validity checks all passed successfully with 5,173 evaluation rows processed.
2. Fidelity dropped significantly from 0.9920 (production) to 0.8673 (candidate).
3. The candidate failed the mandatory deterministic threshold check (fidelity_improved=FAIL), resulting in an automated REJECT decision.))
- version ul_prb-v1 -> ul_prb-v1, duration=38.08s

### Event 13 — dl_prb — 2026-10-08T10:58:14.581332+00:00

- **Trigger**: type=external_drift, severity=0.09085271350425783, fidelity_at_trigger=0.7566900173887696
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: The dl_prb component experienced external drift with a severity of 0.0909. A previous attempt to regenerate the model architecture was rejected by the verification gate. Given that the underlying feature relationships remain intact but recent network telemetry exhibits shifted operational dynamics (e.g., requested PRB ratio around 48.87 with 7 connected UEs), recalibrating the model parameters on recent data is appropriate.
- **Reasoning**: Regeneration was previously attempted and REJECTed by the deterministic gate. Attempting another structural regeneration without updated candidate code patterns is unlikely to pass verification. Recalibrating learned parameters on recent training/tuning distributions provides a lower-risk adaptation to absorb the observed external drift for dl_prb.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_prb-v4
- **Verification**: 0.7499975343251859 -> 0.9243369427505485 => ACCEPT (The recalibration adaptation for the dl_prb component was ACCEPTED after successfully passing all deterministic verification gates. Re-training the model on the updated 24-hour training window improved the baseline FidelityScore from 0.7500 to 0.9243 on the evaluation dataset. This gain of 0.1743 significantly exceeds the required verification_delta margin of 0.01. Furthermore, all structural checks passed: the model artifact was properly serialized at /home/harie06/dt_adaptation/baselines/results/full_system/seed_3/models/dl_prb/dl_prb-v4.joblib, interface signatures remained consistent, predictions were finite across all 5,173 test rows, and baseline comparisons were successfully computed. (1. FidelityScore increased from 0.7500 to 0.9243 (+0.1743), easily passing the minimum required threshold gain of 0.01.
2. All deterministic verification checks passed (candidate_artifact_present, interface_identity_consistent, evaluation_ran, no_invalid_output, production_baseline_evaluable, fidelity_improved).
3. Re-training over the updated data window successfully mitigated model drift in predicting downlink PRB utilization without schema or interface violations.))
- version dl_prb-v1 -> dl_prb-v4, duration=14.24s

### Event 14 — dl_rlc_delay — 2026-10-08T10:58:32.839061+00:00

- **Trigger**: type=external_drift, severity=0.9314638547413545, fidelity_at_trigger=0.7577836840502997
- **Decision**: strategy=expand_scope, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: None
- **Reasoning**: None
- **References**: None
- **Candidate**: version=None
- **Verification**: None -> None => None (None)
- version dl_rlc_delay-v1 -> None, duration=0.00s
- **Error**: ExpandScopeAgent.expand_scope() missing 1 required positional argument: 'example_component_factory'

### Event 15 — ul_prb — 2026-10-08T10:58:37.465204+00:00

- **Trigger**: type=external_drift, severity=0.29816309065742475, fidelity_at_trigger=0.9939767345342616
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.95, llm_or_fallback=llm
- **Root cause**: The affected component ul_prb experienced external drift (severity 0.2982), but its base fidelity remains very high (0.9940). A previous attempt to 'regenerate' the architecture was explicitly REJECTED by the verification gate because candidate fidelity dropped from 0.992 to 0.867. Given high cell load (requested_prb_ratio_lag1 = 49.019, ue_count_cell_lag1 = 7.0) causing parameter drift, retraining the existing architecture on recent telemetry data (recalibrate) is required rather than restructuring the model.
- **Reasoning**: Because regenerate was rejected and degraded fidelity significantly while ul_prb's current baseline fidelity is still high (0.9940), the existing model pipeline structure remains sound. Recalibrating learned parameters on recent window data will address the external drift without architectural regression.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_prb-v4
- **Verification**: 0.9964957635233116 -> 0.9879756171594203 => REJECT (The candidate recalibration for component 'ul_prb' was REJECTED because it failed the required fidelity improvement check. The candidate achieved a FidelityScore of 0.9880, which is lower than the production baseline score of 0.9965 and does not meet the minimum required improvement margin (verification_delta = 0.01). All preliminary deterministic checks (artifact presence, interface identity consistency, finite output evaluation) passed successfully, but the model performance degraded compared to the production baseline. (1. Production Baseline FidelityScore: 0.9965.
2. Candidate Recalibration FidelityScore: 0.9880.
3. Required Threshold: > 1.0065 (Baseline + verification_delta of 0.01).
4. Deterministic Checks Passed: Candidate artifact existence, interface identity consistency, dataset row match (5173 rows), and output validity (all finite).
5. Primary Failure Cause: Fidelity improvement check failed (`fidelity_improved=FAIL`), indicating performance degradation post-recalibration.))
- version ul_prb-v1 -> ul_prb-v1, duration=15.57s
