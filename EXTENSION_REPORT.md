# Repeated-split extension report

Completed MPS experiments with independent baseline selection. Results are descriptive split means and sample standard deviations; three overlapping partitions do not support population-level significance claims.

The main extension has 27 tasks on 25 underlying datasets, 81 task/split evaluations and 1044 neural training trajectories. It adds four tasks from two raw sensor corpora. The transfer study adds three subset/toy-video tasks; its two image datasets overlap the original collection.

## Main comparisons

Baseline | JEPA wins | Ties | JEPA losses
---|---:|---:|---:
Fixed mask, matched budget | 10/27 | 1 | 16
Fixed mask, 30 epochs | 16/27 | 5 | 6
Supervised encoder | 4/27 | 0 | 23
TabM | 3/27 | 0 | 24
Raw linear | 10/27 | 0 | 17
Extra Trees | 5/27 | 0 | 22
CatBoost | 3/18 | 0 | 15
Filter + linear | 11/27 | 0 | 16
Filter + Extra Trees | 7/27 | 0 | 20
Embedded + linear | 8/27 | 0 | 19
Embedded + Extra Trees | 5/27 | 0 | 22

## Per-task results

Metrics are macro F1 (higher is better) for classification and original-unit RMSE (lower is better) for forecasting. Every method chooses its own mask on validation.

Task | Metric | JEPA | Fixed matched | Supervised | TabM | Extra Trees | CatBoost
---|---|---:|---:|---:|---:|---:|---:
Wine | macro_f1 | 0.914242 +/- 0.054641 | 0.868246 +/- 0.030467 | 0.903588 +/- 0.043803 | 0.913371 +/- 0.031121 | 0.964396 +/- 0.042718 | 0.901170 +/- 0.065326
Digits | macro_f1 | 0.845752 +/- 0.013617 | 0.862480 +/- 0.004582 | 0.885747 +/- 0.037476 | 0.890759 +/- 0.008401 | 0.887298 +/- 0.024771 | 0.885597 +/- 0.017425
Oscillatory sensors | rmse | 0.293980 +/- 0.004069 | 0.220358 +/- 0.005809 | 0.218401 +/- 0.016121 | 0.234611 +/- 0.016605 | 0.301015 +/- 0.018234 | n/a
WDBC | macro_f1 | 0.955168 +/- 0.025457 | 0.923812 +/- 0.007279 | 0.968232 +/- 0.023121 | 0.958627 +/- 0.023618 | 0.945902 +/- 0.022259 | 0.935830 +/- 0.031081
Ionosphere | macro_f1 | 0.812410 +/- 0.014303 | 0.839848 +/- 0.084069 | 0.860059 +/- 0.022998 | 0.870110 +/- 0.036837 | 0.871776 +/- 0.020316 | 0.846350 +/- 0.053487
Sonar | macro_f1 | 0.707973 +/- 0.077313 | 0.720670 +/- 0.036639 | 0.742770 +/- 0.051617 | 0.711478 +/- 0.108073 | 0.792243 +/- 0.058484 | 0.799497 +/- 0.065581
Semeion | macro_f1 | 0.771404 +/- 0.016962 | 0.771404 +/- 0.016962 | 0.844176 +/- 0.022294 | 0.822324 +/- 0.006663 | 0.840342 +/- 0.026918 | 0.829272 +/- 0.021453
Letter | macro_f1 | 0.651522 +/- 0.011143 | 0.368130 +/- 0.031411 | 0.646302 +/- 0.019373 | 0.708732 +/- 0.019983 | 0.810720 +/- 0.001041 | 0.775215 +/- 0.008663
PenDigits | macro_f1 | 0.891450 +/- 0.008865 | 0.861451 +/- 0.008484 | 0.925191 +/- 0.007525 | 0.935308 +/- 0.007181 | 0.926166 +/- 0.007927 | 0.928620 +/- 0.004093
Satimage | macro_f1 | 0.792727 +/- 0.018950 | 0.816279 +/- 0.017690 | 0.805736 +/- 0.003843 | 0.824099 +/- 0.013520 | 0.855629 +/- 0.013649 | 0.841827 +/- 0.015250
HAR features | macro_f1 | 0.765314 +/- 0.017328 | 0.781413 +/- 0.018696 | 0.917502 +/- 0.009364 | 0.930502 +/- 0.019643 | 0.899434 +/- 0.034001 | 0.901209 +/- 0.001970
Dry Bean | macro_f1 | 0.926923 +/- 0.007607 | 0.914652 +/- 0.012238 | 0.914708 +/- 0.013318 | 0.926689 +/- 0.009916 | 0.917536 +/- 0.011583 | 0.919813 +/- 0.008397
ISOLET | macro_f1 | 0.695945 +/- 0.022638 | 0.556814 +/- 0.019234 | 0.912461 +/- 0.012979 | 0.935326 +/- 0.004494 | 0.920171 +/- 0.004187 | 0.889979 +/- 0.011120
Lorenz | rmse | 0.925932 +/- 0.043915 | 0.665515 +/- 0.057082 | 0.558376 +/- 0.047285 | 0.892047 +/- 0.045856 | 0.577183 +/- 0.032994 | n/a
Delayed feedback | rmse | 0.114601 +/- 0.003639 | 0.084432 +/- 0.004751 | 0.090922 +/- 0.005905 | 0.092833 +/- 0.006202 | 0.093146 +/- 0.003658 | n/a
Switching VAR | rmse | 0.178192 +/- 0.004336 | 0.177126 +/- 0.004700 | 0.180673 +/- 0.004676 | 0.176438 +/- 0.004005 | 0.176243 +/- 0.004382 | n/a
Chirp/seasonal | rmse | 0.256155 +/- 0.002465 | 0.173836 +/- 0.003287 | 0.159343 +/- 0.006192 | 0.163486 +/- 0.004145 | 0.209899 +/- 0.007220 | n/a
Coupled oscillators | rmse | 0.151323 +/- 0.025695 | 0.096120 +/- 0.025289 | 0.049573 +/- 0.003295 | 0.070929 +/- 0.003476 | 0.153623 +/- 0.048257 | n/a
Nonlinear AR | rmse | 0.106476 +/- 0.006012 | 0.109528 +/- 0.003027 | 0.102107 +/- 0.003839 | 0.101549 +/- 0.003247 | 0.120480 +/- 0.007249 | n/a
Nonlinear multiview | macro_f1 | 0.871403 +/- 0.021886 | 0.874664 +/- 0.016022 | 0.947943 +/- 0.006353 | 0.949462 +/- 0.007729 | 0.913757 +/- 0.006901 | 0.906387 +/- 0.008780
Hierarchical classes | macro_f1 | 0.962480 +/- 0.008771 | 0.963391 +/- 0.003659 | 0.968390 +/- 0.006303 | 0.962432 +/- 0.010022 | 0.970014 +/- 0.006861 | 0.969664 +/- 0.009259
Sparse interactions | macro_f1 | 0.705350 +/- 0.009669 | 0.593093 +/- 0.063127 | 0.894587 +/- 0.010331 | 0.928966 +/- 0.011871 | 0.859655 +/- 0.008374 | 0.929785 +/- 0.008922
Manifold/nuisance | macro_f1 | 0.855571 +/- 0.034994 | 0.842810 +/- 0.017624 | 0.966781 +/- 0.011598 | 0.958716 +/- 0.008087 | 0.957901 +/- 0.015019 | 0.954680 +/- 0.002878
MHEALTH activity | macro_f1 | 0.602897 +/- 0.071885 | 0.641167 +/- 0.037985 | 0.724771 +/- 0.044691 | 0.743805 +/- 0.038145 | 0.791457 +/- 0.090451 | 0.797185 +/- 0.067440
MHEALTH forecast | rmse | 3.970502 +/- 0.410399 | 3.970193 +/- 0.415387 | 3.630558 +/- 0.290257 | 3.515510 +/- 0.319033 | 3.500053 +/- 0.417639 | n/a
PAMAP2 activity | macro_f1 | 0.357232 +/- 0.052462 | 0.333453 +/- 0.059221 | 0.388194 +/- 0.139571 | 0.375876 +/- 0.174532 | 0.409205 +/- 0.106767 | 0.422063 +/- 0.122703
PAMAP2 forecast | rmse | 3.643899 +/- 0.608546 | 3.643152 +/- 0.609408 | 3.425885 +/- 0.646571 | 3.239555 +/- 0.599002 | 3.502880 +/- 0.698001 | n/a

## Official-model transfer

Task | JEPA | Supervised | Extra Trees | Official pretrained
---|---:|---:|---:|---:
Digits | 0.807708 +/- 0.015763 | 0.822212 +/- 0.024293 | 0.892213 +/- 0.026716 | 0.914597 +/- 0.042634
Semeion | 0.668702 +/- 0.053417 | 0.680708 +/- 0.059786 | 0.689924 +/- 0.076389 | 0.765425 +/- 0.021761
Synthetic motion video | 0.403852 +/- 0.028986 | 0.621017 +/- 0.075347 | 0.582748 +/- 0.149163 | 0.771346 +/- 0.091462

Official backbones are frozen, externally pretrained transfer baselines. Their pretraining cost, sizes and resizing are not matched to the scratch pilot. Video evidence is synthetic and uses eight-frame inference; no natural-video or official benchmark reproduction is claimed.

## Validation and reproduction

1284 main-study checkpoint snapshots and 108 transfer-study scratch snapshots were replayed on eight training examples each on CPU and MPS. Maximum absolute differences were 4.3869019e-05 and 2.4795532e-05; atol/rtol were 5e-4. All 945 main and 99 transfer saved prediction files reproduced the reported metrics.

See docs/EXTENSION_PROTOCOL.md and docs/VISION_TRANSFER_PROTOCOL.md for exact budgets, grouping, sampling and limits. Compressed evidence and validation reports are under results/. The paper sources and result-derived tables are under paper/neurips2026/.

Model selection remains supervised through development labels. Raw sensor forecasts exclude null/transition windows using source annotations; PAMAP2 is decimated without antialias filtering. Most prior datasets are reused, only one neural initialization is used per split, and broad foundation-model tuning remains outside this bounded experiment.
