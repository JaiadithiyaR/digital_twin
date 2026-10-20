# Baseline run report — mode `rule_based`

- Replayed: OAI run(s) [8] (TEST split, speed=0)
- Seeds: [1]
- Model versions at start: {'dl_bler': 'dl_bler-v2', 'ul_bler': 'ul_bler-v1', 'dl_prb': 'dl_prb-v1', 'ul_prb': 'ul_prb-v1', 'dl_throughput': 'dl_throughput-v1', 'ul_throughput': 'ul_throughput-v1', 'dl_rlc_delay': 'dl_rlc_delay-v2'}
- Fidelity config: threshold=0.5, min_consecutive=3, rolling_window_length=200

## Per-component summary

| component | mean actual | mean predicted | RMSE | MAE | mean F_c |
|---|---|---|---|---|---|
| dl_bler | 0.0541 | 0.0539 | 0.0007 | 0.0003 | 0.7313 |
| dl_prb | 17.8606 | 18.0820 | 0.6291 | 0.3694 | 0.7470 |
| dl_rlc_delay | 0.4799 | 0.6109 | 0.4930 | 0.2758 | 0.7345 |
| dl_throughput | 0.2054 | 0.2060 | 0.0074 | 0.0031 | 0.7662 |
| ul_bler | 0.0075 | 0.0074 | 0.0004 | 0.0001 | 0.7113 |
| ul_prb | 0.9265 | 0.9300 | 0.0710 | 0.0394 | 0.9051 |
| ul_throughput | 0.0147 | 0.0147 | 0.0010 | 0.0004 | 0.9571 |

## Adaptation Events

### Event 1 — dl_bler — 2026-09-28T15:52:19+00:00

- **Trigger**: type=fidelity_degradation, severity=0.28922933461155886, fidelity_at_trigger=0.35538533269422057
- **Decision**: strategy=recalibrate, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: Deterministic severity rule: severity=0.2892 -> 'recalibrate'.
- **Reasoning**: No LLM call for the decision itself (ablation step 2).
- **References**: []
- **Candidate**: version=dl_bler-v2
- **Verification**: 0.4058032436210799 -> 0.434136365429445 => ACCEPT (The candidate model dl_bler-v2 for the dl_bler component was ACCEPTED because it successfully improved the FidelityScore from 0.4058 to 0.4341 on held-out evaluation data (1,176 rows). This gain of +0.0283 surpasses the mandatory verification threshold delta of 0.01. Furthermore, all deterministic validation checks were satisfied, including model artifact non-emptiness, schema/interface consistency, full evaluation dataset alignment, and zero non-finite outputs. Recalibrating the component effectively mitigated the observed fidelity degradation (severity 0.289). (- Target Component: dl_bler
- Adaptation Strategy: Recalibrate
- Pre-adaptation FidelityScore: 0.4058
- Post-adaptation FidelityScore: 0.4341 (Delta: +0.0283, Exceeds threshold: 0.01)
- Verification Checks: Candidate artifact present (PASS), Interface identity consistent (PASS), Scorable rows matched at 1,176 (PASS), Valid outputs verified (PASS), Baseline baseline evaluation (PASS)
- Decision: ACCEPTED and promoted to production as dl_bler-v2.))
- version dl_bler-v1 -> dl_bler-v2, duration=6.83s

### Event 2 — ul_bler — 2026-09-28T15:52:19+00:00

- **Trigger**: type=fidelity_degradation, severity=0.3922715496916913, fidelity_at_trigger=0.30386422515415434
- **Decision**: strategy=regenerate, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: Deterministic severity rule: severity=0.3923 -> 'regenerate'.
- **Reasoning**: No LLM call for the decision itself (ablation step 2).
- **References**: []
- **Candidate**: version=ul_bler-v2
- **Verification**: 0.31480542174319903 -> 0.3122785336232017 => REJECT (The regenerated candidate model for the 'ul_bler' component failed the deterministic acceptance verification gate because its fidelity score (0.3123) did not achieve the required improvement margin over the existing production baseline (0.3148). Per the system's adaptation policy, a candidate model must exceed the baseline score by at least verification_delta (0.01) to be accepted. Since the new model actually resulted in a slight reduction in fidelity during evaluation, the rejection decision was enforced. (1. Production fidelity score was 0.3148, whereas the regenerated candidate model scored 0.3123.
2. The candidate failed the 'fidelity_improved' gate check as it did not surpass the threshold (baseline + 0.01 = 0.3248).
3. All structural and operational deterministic checks (artifact presence, interface identity consistency, evaluation execution, baseline evaluability, and output validity) passed without issue.))
- version ul_bler-v1 -> ul_bler-v1, duration=48.23s

### Event 3 — ul_bler — 2026-09-28T15:52:19+00:00

- **Trigger**: type=fidelity_degradation, severity=0.3922715496916913, fidelity_at_trigger=0.30386422515415434
- **Decision**: strategy=regenerate, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: Deterministic severity rule: severity=0.3923 -> 'regenerate'.
- **Reasoning**: No LLM call for the decision itself (ablation step 2).
- **References**: []
- **Candidate**: version=ul_bler-v3
- **Verification**: 0.31480542174319903 -> 0.27149558345284464 => REJECT (The candidate component 'ul_bler-v3' generated during attempt 2 of the adaptation loop was rejected because it failed the deterministic fidelity improvement check. While all baseline artifact, interface identity, and scorable prediction checks passed successfully over 1,176 test rows, the candidate's FidelityScore (0.2715) dropped below the active production model's baseline (0.3148). Because the score failed to exceed the required improvement threshold of 0.3248 (production score + verification_delta of 0.01), the deterministic gate automatically rejected the model, retaining production version 'ul_bler-v1'. (1. Quantitative performance regression: Fidelity declined from 0.3148 (production) to 0.2715 (candidate), missing the acceptance margin (0.3248) by 0.0533.
2. Deterministic gate integrity: Structural checks (artifact presence, interface consistency, non-null finite predictions across 1,176 rows) passed, isolating the failure strictly to fidelity regression.
3. Adaptation trajectory: Attempt 2 showed worse performance than Attempt 1 (0.3123) and baseline (0.3148), indicating that LLM regeneration prompts for 'ul_bler' are struggling to capture underlying physical channel state patterns.))
- version ul_bler-v1 -> ul_bler-v1, duration=59.30s

### Event 4 — ul_bler — 2026-09-28T15:52:19+00:00

- **Trigger**: type=fidelity_degradation, severity=0.3922715496916913, fidelity_at_trigger=0.30386422515415434
- **Decision**: strategy=regenerate, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: Deterministic severity rule: severity=0.3923 -> 'regenerate'.
- **Reasoning**: No LLM call for the decision itself (ablation step 2).
- **References**: []
- **Candidate**: version=ul_bler-v4
- **Verification**: 0.31480542174319903 -> 0.21896843558939616 => REJECT (The regenerate adaptation attempt 3 for component 'ul_bler' was rejected because the candidate model (ul_bler-v4) failed the required fidelity improvement check. The candidate achieved a FidelityScore of 0.2190, which represents a degradation compared to the current production baseline (ul_bler-v1) score of 0.3148, and falls far short of the minimum required threshold of 0.3248 (production + verification_delta of 0.01). Although all structural and execution checks passed (artifact creation, interface identity, row alignment, and valid finite outputs), the candidate model's predictive accuracy degraded performance by ~0.0958 points relative to production, prompting an automatic rejection by the deterministic verification gate. (Candidate ul_bler-v4 failed the deterministic gate on 'fidelity_improved' with a fidelity score of 0.2190 vs production's 0.3148. This marks the 3rd consecutive failed regeneration attempt for ul_bler (v2 scored 0.3123, v3 scored 0.2715), indicating that simple regeneration attempts are progressively degrading accuracy rather than compensating for the observed 0.392 severity fidelity drift.))
- version ul_bler-v1 -> ul_bler-v1, duration=39.70s

### Event 5 — ul_bler — 2026-09-28T15:52:19+00:00

- **Trigger**: type=fidelity_degradation, severity=0.3922715496916913, fidelity_at_trigger=0.30386422515415434
- **Decision**: strategy=regenerate, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: Deterministic severity rule: severity=0.3923 -> 'regenerate'.
- **Reasoning**: No LLM call for the decision itself (ablation step 2).
- **References**: []
- **Candidate**: version=ul_bler-v5
- **Verification**: 0.31480542174319903 -> 0.11540903342458886 => REJECT (The regenerate adaptation attempt for component 'ul_bler' (candidate version ul_bler-v5) was REJECTED by the deterministic verification gate. Although the candidate passed all preliminary checks (artifact presence, interface identity consistency, finite scorable output evaluation on 1,176 rows, and baseline comparability), it failed the mandatory fidelity improvement check. Specifically, the candidate achieved a FidelityScore of 0.1154, which significantly degraded relative to the existing production model's score of 0.3148 and fell far short of the required threshold (0.3148 + 0.01 = 0.3248). (1. Preliminary deterministic checks PASS: Candidate artifact was successfully created, interface contract remained consistent (COMPONENT_NAME='ul_bler'/OUTPUT_FIELD='ul_bler_pred'), and all 1,176 predictions were scorable and finite.
2. Verification gate failure: Candidate fidelity dropped to 0.1154 vs production fidelity of 0.3148 (delta threshold of +0.01 required).
3. Persistent degradation trend: Attempt 4 shows worsening fidelity compared to prior failed regeneration attempts (v2: 0.3123, v3: 0.2715, v4: 0.2190, v5: 0.1154), indicating that code/model regeneration alone is insufficient to address the underlying drift context.))
- version ul_bler-v1 -> ul_bler-v1, duration=50.22s

### Event 6 — ul_bler — 2026-09-28T15:52:19+00:00

- **Trigger**: type=fidelity_degradation, severity=0.3922715496916913, fidelity_at_trigger=0.30386422515415434
- **Decision**: strategy=regenerate, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: Deterministic severity rule: severity=0.3923 -> 'regenerate'.
- **Reasoning**: No LLM call for the decision itself (ablation step 2).
- **References**: []
- **Candidate**: version=ul_bler-v6
- **Verification**: 0.31480542174319903 -> 0.1644112140793711 => REJECT (The regenerate adaptation candidate 'ul_bler-v6' was rejected during attempt 5 because it failed the required deterministic fidelity improvement check. To be accepted, the candidate's FidelityScore needed to exceed the production score (0.314805) by at least the verification delta (0.01), targeting a score above 0.324805. However, the candidate achieved a FidelityScore of only 0.164411 on the 1176 scorable evaluation rows, representing a severe drop in fidelity compared to production. (1. All prerequisite deterministic checks (candidate_artifact_present, interface_identity_consistent, evaluation_ran, no_invalid_output, production_baseline_evaluable) passed without issue.
2. The candidate score of 0.164411 failed the fidelity_improved check against the baseline score of 0.314805 + 0.01 verification margin.
3. This marks the fifth consecutive failed attempt to regenerate the 'ul_bler' model following persistent fidelity degradation (severity 0.392).))
- version ul_bler-v1 -> ul_bler-v1, duration=22.85s

### Event 7 — dl_rlc_delay — 2026-09-28T15:53:19+00:00

- **Trigger**: type=fidelity_degradation, severity=0.308558682980381, fidelity_at_trigger=0.3457206585098095
- **Decision**: strategy=regenerate, source=rule_based, confidence=1.0, llm_or_fallback=rule
- **Root cause**: Deterministic severity rule: severity=0.3086 -> 'regenerate'.
- **Reasoning**: No LLM call for the decision itself (ablation step 2).
- **References**: []
- **Candidate**: version=dl_rlc_delay-v2
- **Verification**: 0.35943147719976654 -> 0.4251544281665661 => ACCEPT (The regeneration adaptation for component 'dl_rlc_delay' was ACCEPTed because it successfully passed all deterministic validation gates and achieved a statistically significant fidelity improvement. Fidelity rose from 0.3594 to 0.4252 on the 1,260 held-out evaluation samples, yielding an absolute improvement of +0.0657, which easily satisfies the required verification delta threshold of 0.01. Interface identity and artifact checks confirmed that the newly generated model candidate 'dl_rlc_delay-v2.joblib' strictly matches the expected target schema without producing non-finite predictions. (1. Trigger Reason: Severe fidelity degradation detected on 'dl_rlc_delay' (severity 0.309) in the rule-based component line.
2. Fidelity Gain: Candidate improved FidelityScore from 0.3594 to 0.4252 (+0.0657 margin vs required 0.01 delta).
3. Operational Validity: Model artifact 'dl_rlc_delay-v2.joblib' is verified non-empty, output fields align strictly with 'dl_rlc_delay_pred', and all 1,260 evaluation predictions are finite.))
- version dl_rlc_delay-v1 -> dl_rlc_delay-v2, duration=28.14s
