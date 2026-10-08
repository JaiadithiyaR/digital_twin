# Model Evaluation — Leave-One-Run-Out CV (phase13 §1.5)

Never uses TEST (run 8) — every number below is from the TRAIN/TUNE pool {3, 6, 7}, holding out one whole run per fold.

## dl_bler
| held_out_run | model RMSE | model MAE | persistence RMSE | persistence MAE | MA(5s) RMSE | MA(5s) MAE | beats persistence |
|---|---|---|---|---|---|---|---|
| 3 | 0.0098 | 0.0041 | 0.0112 | 0.0043 | 0.0194 | 0.0070 | True |
| 6 | 0.0081 | 0.0028 | 0.0086 | 0.0026 | 0.0142 | 0.0040 | True |
| 7 | 0.0146 | 0.0085 | 0.0152 | 0.0077 | 0.0242 | 0.0115 | True |
| **avg** | **0.0108** | **0.0052** | **0.0117** | **0.0049** | **0.0193** | **0.0075** | **3/3 folds** |

## ul_bler
| held_out_run | model RMSE | model MAE | persistence RMSE | persistence MAE | MA(5s) RMSE | MA(5s) MAE | beats persistence |
|---|---|---|---|---|---|---|---|
| 3 | 0.0056 | 0.0021 | 0.0072 | 0.0027 | 0.0095 | 0.0037 | True |
| 6 | 0.0047 | 0.0014 | 0.0059 | 0.0018 | 0.0078 | 0.0025 | True |
| 7 | 0.0080 | 0.0043 | 0.0094 | 0.0049 | 0.0122 | 0.0065 | True |
| **avg** | **0.0061** | **0.0026** | **0.0075** | **0.0032** | **0.0098** | **0.0042** | **3/3 folds** |

## dl_prb
| held_out_run | model RMSE | model MAE | persistence RMSE | persistence MAE | MA(5s) RMSE | MA(5s) MAE | beats persistence |
|---|---|---|---|---|---|---|---|
| 3 | 13.9414 | 4.9448 | 15.8307 | 4.7493 | 20.6485 | 6.9349 | True |
| 6 | 14.1114 | 3.9527 | 14.3754 | 3.6181 | 18.4845 | 4.9651 | True |
| 7 | 20.6129 | 9.0402 | 20.3775 | 8.0968 | 24.6883 | 10.5566 | False |
| **avg** | **16.2219** | **5.9792** | **16.8612** | **5.4881** | **21.2737** | **7.4855** | **2/3 folds** |

## ul_prb
| held_out_run | model RMSE | model MAE | persistence RMSE | persistence MAE | MA(5s) RMSE | MA(5s) MAE | beats persistence |
|---|---|---|---|---|---|---|---|
| 3 | 2.1791 | 0.3671 | 2.2627 | 0.3345 | 3.4299 | 0.5375 | True |
| 6 | 1.9770 | 0.3191 | 2.0186 | 0.2866 | 2.9531 | 0.4397 | True |
| 7 | 1.3008 | 0.3904 | 1.3597 | 0.3851 | 1.8984 | 0.5109 | True |
| **avg** | **1.8190** | **0.3589** | **1.8804** | **0.3354** | **2.7605** | **0.4960** | **3/3 folds** |

## dl_throughput
| held_out_run | model RMSE | model MAE | persistence RMSE | persistence MAE | MA(5s) RMSE | MA(5s) MAE | beats persistence |
|---|---|---|---|---|---|---|---|
| 3 | 0.1255 | 0.0259 | 0.1935 | 0.0550 | 0.2486 | 0.0825 | True |
| 6 | 0.6320 | 0.0894 | 0.4833 | 0.0779 | 0.7675 | 0.1307 | False |
| 7 | 0.2323 | 0.0396 | 0.3466 | 0.1011 | 0.4883 | 0.1427 | True |
| **avg** | **0.3299** | **0.0516** | **0.3411** | **0.0780** | **0.5015** | **0.1186** | **2/3 folds** |

## ul_throughput
| held_out_run | model RMSE | model MAE | persistence RMSE | persistence MAE | MA(5s) RMSE | MA(5s) MAE | beats persistence |
|---|---|---|---|---|---|---|---|
| 3 | 0.0312 | 0.0030 | 0.0361 | 0.0048 | 0.0610 | 0.0090 | True |
| 6 | 0.0302 | 0.0034 | 0.0330 | 0.0044 | 0.0550 | 0.0077 | True |
| 7 | 0.0186 | 0.0036 | 0.0225 | 0.0064 | 0.0338 | 0.0090 | True |
| **avg** | **0.0267** | **0.0033** | **0.0305** | **0.0052** | **0.0499** | **0.0086** | **3/3 folds** |

## dl_rlc_delay
| held_out_run | model RMSE | model MAE | persistence RMSE | persistence MAE | MA(5s) RMSE | MA(5s) MAE | beats persistence |
|---|---|---|---|---|---|---|---|
| 3 | 10.1366 | 1.7197 | 11.9148 | 1.2119 | 9.1556 | 1.2096 | True |
| 6 | 7.1072 | 1.1556 | 7.5677 | 0.5771 | 5.8532 | 0.5773 | True |
| 7 | 15.9520 | 3.9150 | 19.9140 | 2.8020 | 15.5659 | 2.8468 | True |
| **avg** | **11.0653** | **2.2634** | **13.1322** | **1.5303** | **10.1916** | **1.5446** | **3/3 folds** |
