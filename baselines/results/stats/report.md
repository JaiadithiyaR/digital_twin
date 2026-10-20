# Ablation statistics (run 8, 60 s windows)

- Runs: static/seed1, periodic_retrain/seed1, rule_based/seed1, full_system/seed1, full_system/seed2, full_system/seed3
- Common windows scored by every run: 62
- Frozen normalization reference: pooled min/max of squared metrics over all runs (`frozen_reference.json`), epsilon=1e-08
- Moving-block bootstrap: block=5 windows, 10000 resamples, seed=0

## Mean windowed fidelity (frozen scale)

            mode  n_runs  n_windows  mean_F_frozen  ci95_low  ci95_high  sd_across_windows  sd_across_runs  mean_norm_rmse  sd_norm_rmse  mean_norm_mae  sd_norm_mae  mean_norm_wasserstein  sd_norm_wasserstein  mean_norm_mk_mmd  sd_norm_mk_mmd  adaptation_wall_seconds  adaptation_cpu_seconds    llm_tokens  llm_cost_usd  n_attempts  n_accepted  n_rejected  n_deployed_ungated  n_agent_failures  n_triggers
          static       1         62       0.819636  0.772999   0.858419           0.095747             NaN        0.082826      0.159675       0.110820     0.225548               0.086803             0.170129          0.441007        0.178429                 0.000000                0.000000      0.000000      0.000000         0.0         0.0         0.0                 0.0               0.0         3.0
periodic_retrain       1         62       0.812190  0.767634   0.849022           0.093566             NaN        0.082597      0.159296       0.110832     0.225804               0.090457             0.178012          0.467354        0.196811               518.584137             4779.260000      0.000000      0.000000        42.0         0.0         0.0                42.0               0.0         3.0
      rule_based       1         62       0.817365  0.772173   0.855786           0.094448             NaN        0.082676      0.159397       0.110524     0.224899               0.086582             0.169609          0.450759        0.182637               255.257915              148.610000 174662.000000      0.016326         7.0         2.0         5.0                 0.0               0.0         3.0
     full_system       3         62       0.815530  0.770791   0.852503           0.092788        0.000179        0.082641      0.159353       0.110085     0.224166               0.085794             0.167885          0.459362        0.187604               116.387835              137.526667  92325.333333      0.008110         4.0         3.0         1.0                 0.0               0.0         3.0

## Pairwise tests (Wilcoxon signed-rank, Holm-adjusted)

     mode_a           mode_b  n_windows  mean_diff  ci95_low  ci95_high  n_nonzero     W            p      r_rb       p_holm  significant_0.05
full_system           static         62  -0.004107 -0.005750  -0.002133         52  65.5 1.075071e-08 -0.904935 4.300284e-08              True
full_system periodic_retrain         62   0.003340  0.001224   0.006454         55 357.0 5.393712e-04  0.536364 5.393712e-04              True
full_system       rule_based         62  -0.001835 -0.002742  -0.000675         52  65.5 1.102766e-08 -0.904935 4.300284e-08              True
 rule_based periodic_retrain         62   0.005175  0.003028   0.008337         52 143.0 6.613709e-07  0.792453 1.322742e-06              True

Caveat: the Wilcoxon test assumes independent pairs; consecutive 60 s windows are autocorrelated, so its p-values are optimistic. The block-bootstrap CI of the mean difference accounts for that dependence and should be read as the primary evidence.