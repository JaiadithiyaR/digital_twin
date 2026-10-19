# Baseline run report — mode `full_system`

- Replayed: OAI run(s) [8] (TEST split, speed=0)
- Seeds: [1]
- Model versions at start: {'dl_bler': 'dl_bler-v1', 'ul_bler': 'ul_bler-v1', 'dl_prb': 'dl_prb-v1', 'ul_prb': 'ul_prb-v1', 'dl_throughput': 'dl_throughput-v1', 'ul_throughput': 'ul_throughput-v1', 'dl_rlc_delay': 'dl_rlc_delay-v1'}
- Fidelity config: threshold=0.5, min_consecutive=3, rolling_window_length=200

## Per-component summary

| component | mean actual | mean predicted | RMSE | MAE | mean F_c |
|---|---|---|---|---|---|
| dl_bler | 0.0611 | 0.0615 | 0.0017 | 0.0008 | 0.7825 |
| dl_prb | 19.2771 | 19.3358 | 1.3012 | 0.7381 | 0.7561 |
| dl_rlc_delay | 0.2572 | 0.5390 | 0.6579 | 0.2818 | 0.7574 |
| dl_throughput | 0.2200 | 0.2209 | 0.0149 | 0.0068 | 0.8130 |
| ul_bler | 0.0075 | 0.0074 | 0.0005 | 0.0002 | 0.7500 |
| ul_prb | 0.9126 | 0.9108 | 0.0501 | 0.0288 | 0.8679 |
| ul_throughput | 0.0144 | 0.0145 | 0.0003 | 0.0002 | 0.8905 |

## Adaptation Events

### Event 1 — dl_prb — 2026-10-09T05:35:52.430276+00:00

- **Trigger**: type=external_drift, severity=0.9504636963259353, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: An external drift trigger was detected for the dl_prb component with a high severity of 0.9505. Analysis of recent network telemetry indicates heavy cell-level demand (dl_prb averaging 127.24 PRBs, requested_prb_ratio_lag1 at 49.67, and 7 UEs in the cell), alongside elevated DL BLER (0.3094). The existing model pipeline structure remains valid for predicting physical resource block allocation, but the relationship between traffic demand and allocated PRBs has shifted due to current network conditions, making model parameters stale.
- **Reasoning**: Because the component architecture and features remain valid for PRB prediction under current O-RAN/OAI data structures, rebuilding the model architecture (regenerate) or adding new components (expand_scope) is unnecessary. Retraining the model parameters on the latest 24-hour window via recalibration is the standard, minimal strategy to correct telemetry drift.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_prb-v2
- **Verification**: None -> None => REJECT (The candidate model dl_prb-v2 was rejected by the deterministic verification gate following a recalibration attempt. Although all structural, interface, and prediction validity checks passed (candidate artifact present, interface identity consistent, non-empty evaluation dataset of 5,173 samples, and all predictions finite), the fidelity improvement check failed. Specifically, the FidelityScore before and/or after adaptation could not be calculated due to insufficient rolling history within the evaluation segment. Operating under the fail-safe policy, unverified candidate performance defaults to REJECT to prevent potentially degrading production state. (1. All basic validation gates passed: artifact verification, component identity alignment ('dl_prb'), finite output checks, and dataset size consistency (5,173 rows).
2. The adaptation failed specifically at the 'fidelity_improved' verification check.
3. FidelityScore computation was blocked due to incomplete rolling history in the held-out evaluation window, triggering a deterministic fail-safe rejection.))
- version dl_prb-v1 -> dl_prb-v1, duration=13.87s

### Event 2 — dl_prb — 2026-10-09T05:36:10.879562+00:00

- **Trigger**: type=external_drift, severity=0.9486494471372439, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.91, llm_or_fallback=llm
- **Root cause**: The component dl_prb suffered severe external drift (severity 0.9486) amidst high cell congestion (7 active UEs with an average requested PRB ratio of 49.77). A prior recalibration attempt was rejected by the verification gate, indicating that simple parameter retraining on the existing model pipeline is insufficient to capture the altered resource allocation dynamics. Rebuilding the feature pipeline and model architecture from scratch via regeneration is required.
- **Reasoning**: Recalibration previously failed and was rejected. Given the high drift severity and high load conditions (requested_prb_ratio_lag1 ~ 49.77), the feature relationships for downlink PRB demand have undergone a structural shift that requires model regeneration.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=dl_prb-v3
- **Verification**: None -> None => REJECT (The regenerated candidate for 'dl_prb' was rejected by the deterministic verification gate due to a fail-safe check on fidelity metrics. Although the candidate model artifact was generated successfully, maintained interface consistency, and produced valid finite predictions across all 5,173 evaluation rows, the overall FidelityScore could not be computed for both production and candidate models due to insufficient rolling history windowing on the evaluation slice. Because improvement over the production baseline could not be verified quantitatively, the system applied a mandatory fail-safe rejection. (1. All initial structural and functional validation checks passed (candidate artifact present, interface identity aligned, 5173 predictions generated without invalid/non-finite values).
2. Production baseline model 'dl_prb-v1' was available and evaluable on the same dataset.
3. The 'fidelity_improved' gate check failed because rolling history requirements prevented computing valid FidelityScore metrics before or after, triggering a strict fail-safe REJECT decision.))
- version dl_prb-v1 -> dl_prb-v1, duration=33.10s

### Event 3 — ul_bler — 2026-10-09T05:36:48.306498+00:00

- **Trigger**: type=external_drift, severity=0.8277025938204418, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.91, llm_or_fallback=llm
- **Root cause**: External drift was detected on uplink BLER (ul_bler) with high severity (0.8277). Uplink block error rates are sensitive to changing radio link conditions, channel SNR, and interference parameters (such as pusch_snr / pucch_snr). The existing model structure remains appropriate for predicting MAC-layer uplink block errors, but its parameters have grown stale relative to the recent telemetry state.
- **Reasoning**: External drift on ul_bler indicates a shift in the underlying distribution of uplink radio quality metrics. Recalibrating the component on the latest window of OAI RAN telemetry allows the learned model weights to adapt to the shifted uplink channel conditions without requiring structural redesign.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_bler-v2
- **Verification**: None -> None => REJECT ( sensitivity analysis and evaluation gate checked candidate version ul_bler-v2 against production version ul_bler-v1. Although the candidate artifact was generated correctly, interface identities matched, and evaluation executed across 5,173 test rows, the fidelity metric check failed. Specifically, the FidelityScore before and/or after could not be computed due to insufficient rolling history window data in the evaluation slice, making it impossible to verify the required verification_delta margin (+0.01). Per the fail-safe deterministic gate rules, candidate promotion was rejected. (Candidate artifact exists and is finite across 5173 prediction rows. Evaluation gate failed on fidelity_improved check due to uncomputable rolling history metrics, triggering a mandatory fail-safe rejection.))
- version ul_bler-v1 -> ul_bler-v1, duration=13.95s

### Event 4 — dl_bler — 2026-10-09T05:37:06.359525+00:00

- **Trigger**: type=external_drift, severity=0.5381433132192782, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: An external drift event with severity 0.5381 was detected for the dl_bler component. Telemetry summary indicates high resource contention (requested_prb_ratio_lag1 ~48.9) with 7 active UEs per cell, leading to distribution shift in MAC-layer downlink block-error rates. Because the core feature graph and dependencies remain valid, recalibrating the model parameters on recent deployment data is sufficient to restore prediction fidelity.
- **Reasoning**: External drift severity 0.5381 indicates parameter staleness due to changing radio link/traffic conditions rather than a structural model failure. Recalibration is the least disruptive and most targeted response before attempting full model regeneration.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_bler-v2
- **Verification**: None -> None => REJECT (The candidate recalibration for the 'dl_bler' component was rejected by the deterministic verification gate because the fidelity improvement check failed (fidelity_improved=FAIL). Specifically, the FidelityScore before and/or after evaluation could not be computed due to insufficient rolling history across the evaluated window. Per system safety policies, when fidelity improvement cannot be verified quantitatively, a fail-safe REJECT is enforced to prevent unverified model artifacts from entering production. (All functional and interface integrity checks passed (artifact presence, interface identity consistency, evaluation execution, finite output validation, and production baseline evaluability). However, fidelity scores before and after adaptation evaluated to None due to insufficient rolling history, causing the fidelity_improved check to FAIL under fail-safe rules.))
- version dl_bler-v1 -> dl_bler-v1, duration=13.14s

### Event 5 — dl_prb — 2026-10-09T05:37:23.661088+00:00

- **Trigger**: type=external_drift, severity=0.303194829291645, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: The affected component dl_prb experienced external drift (severity 0.3032). A previous regeneration attempt was REJECTED by the verification gate. Furthermore, there is insufficient evaluation history to justify structural changes, but network telemetry indicates altered downlink resource demand patterns (requested_prb_ratio_lag1 of 48.8680 across 7 active UEs). Retraining the existing model on recent data via recalibration is the appropriate strategy.
- **Reasoning**: Regeneration was previously attempted and rejected. Given the current drift and shift in telemetry, recalibrating the parameters of the existing dl_prb model on updated historical windows avoids unnecessary pipeline restructuring while addressing domain shift.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_prb-v4
- **Verification**: None -> None => REJECT (The candidate recalibration model 'dl_prb-v4' was rejected by the deterministic verification gate because the fidelity score could not be reliably calculated for either the baseline or candidate predictions due to insufficient rolling history in the evaluation dataset. Following the system's fail-safe adaptation policy, any candidate whose fidelity improvement cannot be strictly verified against the verification delta threshold (0.01) must be rejected to ensure production stability. (1. All initial structural and safety checks passed: candidate artifact presence, interface identity consistency, non-null candidate predictions (5,173 rows), and production baseline evaluation.
2. Verification gate check 'fidelity_improved' failed because FidelityScore before/after was non-computable due to lack of required rolling window history.
3. Fail-safe policy triggered a deterministic REJECT decision to prevent unverified model deployment.))
- version dl_prb-v1 -> dl_prb-v1, duration=13.15s

### Event 6 — dl_bler — 2026-10-09T05:37:40.876780+00:00

- **Trigger**: type=external_drift, severity=0.13404169724716475, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: The affected component dl_bler experienced external drift, and a previous recalibrate attempt was REJECTED. Under severe radio channel state drift where parameter retraining alone fails to restore prediction fidelity, recalibration is insufficient. Rebuilding the model architecture via regenerate is necessary to handle the non-linear distribution shift in MAC-layer downlink block error rate.
- **Reasoning**: A previous recalibration attempt was rejected by the verification gate. According to the adaptation strategy progression rules, when parameter tuning fails to mitigate observed drift, the agent must escalate to regenerating the model architecture rather than repeating a failed recalibration.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_bler-v3
- **Verification**: None -> None => REJECT (The regeneration candidate for component 'dl_bler' (version dl_bler-v3) was rejected by the deterministic verification gate because the fidelity score improvement check failed (fidelity_improved=FAIL). Specifically, the fidelity score before and/or after candidate evaluation was not computable due to insufficient rolling history on the evaluated test set. Following the system's fail-safe adaptation safety rules, a non-computable fidelity delta automatically triggers a REJECT decision to ensure unverified model updates are not promoted to production. (1. Artifact existence, interface identity consistency, evaluation execution, non-invalid output, and production baseline evaluability checks all PASSED.
2. The candidate failed the fidelity_improved check because the FidelityScore could not be computed (insufficient rolling window history on one or both sides).
3. In accordance with safety policies (verification_delta=0.01 margin required), unverified or non-computable fidelity changes default safely to REJECT.))
- version dl_bler-v1 -> dl_bler-v1, duration=32.59s

### Event 7 — ul_bler — 2026-10-09T05:38:16.960726+00:00

- **Trigger**: type=external_drift, severity=0.2623133404418495, fidelity_at_trigger=None
- **Decision**: strategy=regenerate, source=full_system, confidence=0.91, llm_or_fallback=llm
- **Root cause**: The ul_bler component suffered an external drift event (severity 0.2623). A previous recalibration attempt was rejected by the verification gate, demonstrating that simply retraining learned parameters on recent data is insufficient to restore accuracy under the current radio condition shift. The current model architecture/pipeline is no longer able to capture the non-linear relationship during severe uplink link quality changes (reflected by high requested PRB ratio and multiple UEs per cell), requiring a structural model regeneration.
- **Reasoning**: Recalibration has already failed and was rejected by the deterministic verification gate. Per system adaptation policies, when parameter update via recalibration fails to improve candidate performance beyond the verification delta threshold, the component structure itself must be rebuilt from scratch (regenerate) to handle the updated telemetry distribution.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_bler-v3
- **Verification**: None -> 0.7515468623323398 => REJECT (The regeneration candidate for the 'ul_bler' component was rejected by the deterministic verification gate. Although the candidate model artifact was generated, met interface consistency checks, and produced valid finite predictions, the baseline evaluation failed the fidelity improvement check. Specifically, the FidelityScore prior to adaptation was non-computable due to insufficient rolling history on the held-out evaluation set. Per system safety rules, if fidelity improvement cannot be deterministically computed and verified against the verification delta threshold (0.01), a fail-safe REJECT is enforced. (1. Candidate artifact 'ul_bler-v3.joblib' exists and interface definitions matched (ul_bler / ul_bler_pred).
2. Model evaluation produced 5,173 valid finite prediction rows on the evaluation target set.
3. Baseline fidelity score was uncomputable (None), causing the 'fidelity_improved' check to FAIL and triggering an automatic fail-safe rejection.))
- version ul_bler-v1 -> ul_bler-v1, duration=38.80s

### Event 8 — ul_throughput — 2026-10-09T05:38:59.748823+00:00

- **Trigger**: type=external_drift, severity=0.2804087579860399, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: External drift event detected on component 'ul_throughput' with severity 0.2804. The current baseline model parameters are stale relative to recent telemetry changes (e.g., elevated requested_prb_ratio_lag1 at 48.94 with 7 active UEs), causing divergence from ground truth while the underlying model structure remains valid.
- **Reasoning**: An external drift trigger on ul_throughput without structural schema or functional dependency changes indicates statistical parameter drift rather than architectural invalidity. Recalibrating the existing model on recent 24-hour training data is sufficient to restore prediction fidelity.
- **References**: ["digital_twin/internal:docs/oai_twin.md (this repository's own OAI migration design document)", "policies/internal:config/settings.yaml (this repository's own enforced configuration)"]
- **Candidate**: version=ul_throughput-v2
- **Verification**: None -> 0.9756895579241149 => REJECT (The recalibrated candidate model for 'ul_throughput' was rejected by the deterministic verification gate. Although the candidate model evaluated successfully and achieved a post-calibration fidelity score of 0.9757 over 5,173 evaluation rows, the system was unable to compute or verify a baseline fidelity score due to insufficient rolling history required for a valid before-and-after comparison. Under the fail-safe policy, any adaptation where fidelity improvement cannot be definitively established relative to production must be rejected. (Candidate artifact creation, interface identity, and prediction evaluation checks all passed with 5,173 valid rows. However, the 'fidelity_improved' check failed because the baseline fidelity score could not be calculated (insufficient rolling window context), triggering a mandatory fail-safe rejection.))
- version ul_throughput-v1 -> ul_throughput-v1, duration=13.13s

### Event 9 — dl_throughput — 2026-10-09T05:39:16.591795+00:00

- **Trigger**: type=external_drift, severity=0.9616571936637868, fidelity_at_trigger=None
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: External network drift detected for dl_throughput with high severity (0.9617). Recent telemetry indicates high requested PRB ratio (48.87) and increased cell UE count (7.0), but downlink throughput remains near zero (0.0004 Mbps), reflecting shifting network conditions or stale model parameters.
- **Reasoning**: The feature relationships remain conceptually consistent with the existing digital twin dependency structure (dl_throughput depending on dl_bler and dl_prb), but recent network conditions show substantial feature drift. Recalibrating the model parameters on the updated telemetry window is the optimal initial strategy.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_throughput-v2
- **Verification**: 0.7716901882749765 -> 0.771991313369906 => REJECT (The recalibrated candidate model for 'dl_throughput' was REJECTED because it failed to meet the required performance threshold established by the verification gate. Although the candidate model passed all structural, sanity, and output validity checks—demonstrating complete predictions and consistent interface parameters—its FidelityScore only marginally increased from 0.77169 to 0.77199 (a delta of +0.00030). Under the active system adaptation policy, a candidate component must surpass the production model's FidelityScore by at least the verification_delta margin of 0.01 to be accepted for deployment. Because 0.7720 is not strictly greater than 0.7817 (0.7717 + 0.01), the deterministic gate rejected the candidate. (1. Verification Gate Failure: fidelity_improved evaluated to FAIL. The candidate's FidelityScore improvement (+0.00030) fell short of the mandatory verification_delta threshold of 0.01.
2. Validation Checks Passed: Candidate artifact existence, interface identity consistency, evaluation row count alignment (5,173 samples), output finiteness, and production baseline availability all passed without error.
3. Model Behavior: Recalibrating 'dl_throughput' on the updated dataset yielded negligible fidelity gains over the production baseline (dl_throughput-v1), indicating that parameter recalibration alone is insufficient to address current drift in downlink throughput predictions.))
- version dl_throughput-v1 -> dl_throughput-v1, duration=15.63s

### Event 10 — ul_bler — 2026-10-09T05:39:38.089674+00:00

- **Trigger**: type=external_drift, severity=0.5412268555474342, fidelity_at_trigger=0.7500354470768414
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.88, llm_or_fallback=llm
- **Root cause**: An external drift event (severity 0.5412) affected the uplink block error rate (ul_bler) model, causing the fidelity score to fall to 0.7500. A prior regeneration attempt was REJECTED by the verification gate as architectural re-generation failed to yield a sufficient delta improvement over production (fidelity_after 0.7515). The underlying cause is staleness in the learned parameters under recent uplink channel conditions, high PRB demand (requested_prb_ratio_lag1 ~48.94), and active UE load (7 UEs in cell).
- **Reasoning**: Because structural regeneration previously failed acceptance verification, retraining model parameters on recent telemetry window via recalibration is the required recovery strategy to adapt the existing ul_bler pipeline to current radio link statistics.
- **References**: ['policies/internal:config/settings.yaml', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_bler-v4
- **Verification**: 0.7501567202959011 -> 0.7521847141991993 => REJECT (The recalibration candidate for 'ul_bler' was REJECTED because it failed the deterministic fidelity improvement check. Although all functional, artifact, and interface checks passed, the candidate achieved a FidelityScore of 0.7522 compared to the production model's 0.7502. This represents an improvement of only ~0.0020, which falls below the strict safety threshold required by verification_delta (0.0100). Consequently, the production baseline remains active. (Candidate fidelity score (0.7522) failed to beat production baseline (0.7502) by the required verification_delta margin of 0.0100 (delta achieved: +0.0020). All structural checks (candidate artifact presence, interface identity consistency, finite outputs) passed successfully.))
- version ul_bler-v1 -> ul_bler-v1, duration=13.42s

### Event 11 — dl_bler — 2026-10-09T05:39:57.600410+00:00

- **Trigger**: type=external_drift, severity=0.9699254132161326, fidelity_at_trigger=0.7504130668818338
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.85, llm_or_fallback=llm
- **Root cause**: The affected component `dl_bler` experienced an external drift trigger with high severity (0.9699) and lower fidelity (0.7504). A previous `regenerate` action was attempted and rejected by the verification gate, indicating that altering the underlying architectural pipeline was unsuccessful. Recalibration on recent OAI RAN training data is the appropriate next step to adjust model parameters to shifting channel state distributions (such as requested PRB ratios and UE count fluctuations) without introducing risky pipeline modifications.
- **Reasoning**: Since structural regeneration previously failed (REJECT) and the model scope already encompasses the required features, recalibrating the existing architecture on recent telemetry is the most stable and reliable strategy to recover fidelity above the acceptance gate.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_bler-v4
- **Verification**: 0.7509339485271057 -> 0.8252174913753535 => ACCEPT (The recalibration of the 'dl_bler' model was ACCEPTED because the candidate model (dl_bler-v4) successfully passed all deterministic integrity checks and demonstrated a significant improvement in overall fidelity on the evaluation dataset. Specifically, the FidelityScore increased from 0.7509 to 0.8252, yielding an absolute improvement of +0.0743, which comfortably exceeds the strict verification delta threshold of 0.01. (1. Quantitative Verification: FidelityScore improved from 0.7509 (dl_bler-v1) to 0.8252 (dl_bler-v4), satisfying the fidelity_improved check (+0.0743 vs required +0.01 delta).
2. Integrity & Compatibility: Passed all deterministic checks including candidate artifact existence, interface identity consistency ('dl_bler' / 'dl_bler_pred'), finite predictions across all 5,173 evaluation rows, and comparative baseline evaluation.
3. Operational Outcome: The candidate artifact has been accepted and promoted to production as the active model for 'dl_bler'.))
- version dl_bler-v1 -> dl_bler-v4, duration=13.40s

### Event 12 — ul_throughput — 2026-10-09T05:40:14.534659+00:00

- **Trigger**: type=external_drift, severity=0.11586561247077032, fidelity_at_trigger=0.38366840519297307
- **Decision**: strategy=regenerate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: An external drift trigger caused severe fidelity degradation in ul_throughput (0.3837) and its upstream dependency ul_prb (0.3490). A prior recalibration attempt was rejected by the verification gate, indicating that simple retraining on existing model parameters is insufficient to capture the structural shift in uplink radio resource usage and throughput metrics. Rebuilding the component architecture via regeneration is required.
- **Reasoning**: Recalibration previously failed and was rejected, showing parameter updates alone cannot resolve the persistent fidelity drop under current network conditions.
- **References**: ['digital_twin/internal:CLAUDE.md', 'digital_twin/internal:docs/oai_twin.md']
- **Candidate**: version=ul_throughput-v3
- **Verification**: 0.9919903354159357 -> 0.992598312036884 => REJECT (The regenerate candidate for 'ul_throughput' was rejected because its fidelity improvement failed to meet the required verification threshold. Although the candidate model slightly increased the FidelityScore from 0.99199 to 0.99260 (a gain of 0.00061), this improvement is below the required verification_delta margin of 0.01. Consequently, the candidate failed the deterministic fidelity_improved check and the existing production version ul_throughput-v1 remains active. (1. All baseline integrity checks passed (artifact presence, interface identity consistency, evaluation row alignment, and valid non-null output predictions).
2. Candidate FidelityScore (0.99260) vs. Production FidelityScore (0.99199) yielded an incremental gain of +0.00061.
3. The candidate was rejected because the gain fell short of the mandatory verification_delta threshold of +0.01000.))
- version ul_throughput-v1 -> ul_throughput-v1, duration=36.79s

### Event 13 — dl_prb — 2026-10-09T05:41:01.330819+00:00

- **Trigger**: type=external_drift, severity=0.6130033010530405, fidelity_at_trigger=0.7571425131760543
- **Decision**: strategy=regenerate, source=full_system, confidence=0.9, llm_or_fallback=llm
- **Root cause**: The affected component dl_prb experienced external drift (severity 0.6130) driven by high resource contention (requested_prb_ratio_lag1 of 48.87 and cell UE count of 7). A previous recalibration attempt was rejected by the verification gate, indicating that simply updating parameters on the existing model architecture cannot resolve the distribution shift.
- **Reasoning**: Because recalibration has already failed to yield a sufficient fidelity gain for dl_prb, the current model architecture is inadequate for representing the altered downlink PRB allocation behavior. Standard policy requires escalating to model regeneration to rebuild the pipeline structure.
- **References**: ['digital_twin/internal:CLAUDE.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_prb-v5
- **Verification**: 0.7499975343251859 -> 0.72827964809806 => REJECT (The regenerated candidate model for 'dl_prb' (v5) was rejected by the deterministic verification gate because it failed the required fidelity improvement check. The candidate achieved a FidelityScore of 0.7283, which is lower than the production baseline score of 0.7500 and fails to satisfy the required improvement threshold of verification_delta=0.01 (requiring a score > 0.7600). Consequently, the production model 'dl_prb-v1' remains active in deployment. (1. Deterministic check 'fidelity_improved' failed (candidate score 0.7283 vs production score 0.7500 + margin 0.01).
2. All preliminary candidate integrity and sandbox evaluation checks passed (artifact presence, interface identity consistency, finite prediction values, and baseline comparability).
3. Regenerating 'dl_prb' did not yield enhanced performance under current drift conditions; production version dl_prb-v1 is retained.))
- version dl_prb-v1 -> dl_prb-v1, duration=52.26s

### Event 14 — dl_bler — 2026-10-09T05:41:59.240314+00:00

- **Trigger**: type=external_drift, severity=0.03959287666420286, fidelity_at_trigger=0.8250137829277755
- **Decision**: strategy=recalibrate, source=full_system, confidence=0.92, llm_or_fallback=llm
- **Root cause**: The affected component dl_bler experienced external drift, leading to a small fidelity degradation. Given that a prior recalibration attempt successfully improved fidelity from ~0.751 to ~0.825 and the model architecture remains appropriate for predicting MAC-layer downlink block error rates, updating the model parameters with recent telemetry data is sufficient to adapt to the environmental shift.
- **Reasoning**: External drift in radio link performance metrics (dl_bler) is typically driven by evolving channel conditions and UE traffic patterns. Since recalibration previously succeeded in boosting fidelity for this component and the feature schema/pipeline covers the relevant lagged parameters, retraining on recent data is the minimal and most effective response before considering structural regeneration.
- **References**: ['digital_twin/internal:docs/oai_twin.md', 'policies/internal:config/settings.yaml']
- **Candidate**: version=dl_bler-v5
- **Verification**: 0.8251734842580631 -> 0.8251734842580631 => REJECT (The recalibration attempt for the 'dl_bler' component was REJECTED because it failed the required fidelity improvement check. While the candidate passed all functional checks (artifact presence, interface identity consistency, finite output prediction, and baseline evaluation), its fidelity score of 0.8252 showed zero improvement over the existing production baseline (0.8252). Per the adaptation policy, a candidate must exceed the current production fidelity by at least the verification delta threshold of 0.01. (1. Functional Verification: PASS for artifact presence, interface consistency, non-empty evaluations (5,173 rows), and finite predictions. 2. Fidelity Evaluation: Fidelity Score remained identical at 0.825173 (delta = 0.0000), failing the required +0.01 margin (fidelity_improved=FAIL). 3. Outcome: Production model dl_bler-v4 remains active; candidate dl_bler-v5 rejected.))
- version dl_bler-v4 -> dl_bler-v4, duration=14.07s
