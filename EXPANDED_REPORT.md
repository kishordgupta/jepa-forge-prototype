# JEPA-FORGE: expanded 23-dataset prototype report

**23 datasets and 138 completed GPU training runs:** the original three-dataset, 18-run study plus **20 additional datasets and 120 new GPU runs**. The expansion contains 10 public datasets and 10 distinct synthetic generators, each with two declared task policies and three model seeds.

These are fresh prototype measurements. Public data are bounded samples with new internal partitions; synthetic mechanisms test controlled behavior. This is not an official benchmark reproduction, a scientific-realism validation, or evidence of general JEPA superiority.

## Evidence and observed comparisons

Independent validation confirms 40 exported tasks, 120 checkpoint hashes, and 120 CPU checkpoint replays. Training tensors, parameters, losses, and gradients were recorded on `mps:0`. The full test report has **120 passed** and 0 skipped tests.

| Primary-task comparison (20 datasets) | JEPA wins | Ties | Losses |
| --- | ---: | ---: | ---: |
| JEPA vs Raw linear | 9 | 0 | 11 |
| JEPA vs Extra Trees | 3 | 0 | 17 |
| JEPA vs Random encoder | 9 | 0 | 11 |

Wins compare seed-mean test metrics within the same predeclared primary policy: higher macro F1 for classification, lower RMSE for forecasting. Counts weight datasets equally and are descriptive; they are not significance tests or model-selection decisions.

![JEPA versus the matched raw-context linear baseline across the 20 new primary tasks](results/figures/expanded_comparison.png)

The figure shows macro-F1 percentage-point changes for classification and relative RMSE reduction for forecasting. Positive values favor JEPA. Whiskers show one model-seed standard deviation, not confidence intervals. Forecast percentages remain within-system comparisons and are not averaged across generators.

The representation heuristic flagged **83 of 120 new runs** and 90 of 138 combined runs for low standard deviation or effective rank. A flag is a diagnostic, not an independent finding of complete collapse; its absence does not prove a useful representation.

## Fixed protocol

- Model seeds: [7, 17, 27]; split/generator seed: 2026; 60 fixed epochs per neural run; final checkpoint retained.
- First declared policy is primary before test evaluation. Both policies are retained. Comparisons stay within a policy because masks, available context, and forecast horizons differ.
- Training-only normalization and PCA. Encoders train without class labels. Frozen linear probes use validation macro F1 or RMSE to select the declared grid and remain training-fitted; test data do not select checkpoints, probes, or policies.
- Logistic C / ridge alpha: 0.01, 0.1, 1, 10, 100. PCA: min(8, input dimensions, training rows minus 1). Extra Trees: 100 trees and depth 8 or unlimited.
- JEPA-style latent prediction with an EMA target encoder and variance regularization; latent dimension 32, batch 128, Adam learning rate 0.001, EMA 0.99, variance weight 1.0. This is a small pilot adaptation, not a reproduction of a named published JEPA architecture.
- PyTorch model training and encoding use the Apple GPU. Scikit-learn preprocessing/probes/baselines and covariance diagnostics use CPU. Validator checkpoint replay also uses CPU, with atol=rtol=1e-4.
- Values below are mean +/- sample standard deviation over three model seeds on one fixed split. Deterministic baselines may have zero standard deviation; seeds are not independent dataset replications.

## All 20 new primary tasks

Macro F1 is a fraction and higher is better. RMSE is in each dataset's original synthetic signal units and lower is better; RMSE values are not comparable across different generators.

| Dataset | Policy | Metric | JEPA | Raw linear | PCA | Extra Trees | Random encoder | Reference |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Breast cancer (WDBC) | `mean_to_worst` | macro_f1 | 0.921 +/- 0.001 | 0.919 +/- 0.000 | 0.919 +/- 0.000 | 0.899 +/- 0.012 | 0.908 +/- 0.006 | 0.980 +/- 0.000 |
| Ionosphere | `alternating_pairs` | macro_f1 | 0.770 +/- 0.046 | 0.746 +/- 0.000 | 0.672 +/- 0.000 | 0.896 +/- 0.015 | 0.811 +/- 0.057 | 0.813 +/- 0.000 |
| Sonar | `alternating` | macro_f1 | 0.597 +/- 0.075 | 0.708 +/- 0.000 | 0.686 +/- 0.000 | 0.809 +/- 0.024 | 0.704 +/- 0.053 | 0.708 +/- 0.000 |
| Semeion | `center` | macro_f1 | 0.849 +/- 0.006 | 0.909 +/- 0.000 | 0.794 +/- 0.000 | 0.908 +/- 0.007 | 0.722 +/- 0.021 | 0.940 +/- 0.000 |
| Letter | `alternating` | macro_f1 | 0.522 +/- 0.002 | 0.498 +/- 0.000 | 0.497 +/- 0.000 | 0.700 +/- 0.011 | 0.602 +/- 0.012 | 0.696 +/- 0.000 |
| PenDigits | `alternating_pairs` | macro_f1 | 0.898 +/- 0.005 | 0.833 +/- 0.000 | 0.833 +/- 0.000 | 0.911 +/- 0.001 | 0.901 +/- 0.001 | 0.925 +/- 0.000 |
| SatImage | `neighbor_to_center` | macro_f1 | 0.799 +/- 0.020 | 0.806 +/- 0.000 | 0.792 +/- 0.000 | 0.868 +/- 0.008 | 0.839 +/- 0.010 | 0.813 +/- 0.000 |
| Human activity (HAR) | `time_to_frequency` | macro_f1 | 0.796 +/- 0.014 | 0.929 +/- 0.000 | 0.804 +/- 0.000 | 0.916 +/- 0.001 | 0.789 +/- 0.008 | 0.912 +/- 0.000 |
| Dry Bean | `alternating` | macro_f1 | 0.914 +/- 0.002 | 0.914 +/- 0.000 | 0.914 +/- 0.000 | 0.916 +/- 0.003 | 0.923 +/- 0.008 | 0.943 +/- 0.000 |
| ISOLET | `alternating` | macro_f1 | 0.748 +/- 0.005 | 0.929 +/- 0.000 | 0.711 +/- 0.000 | 0.932 +/- 0.006 | 0.703 +/- 0.013 | 0.934 +/- 0.000 |
| Lorenz trajectories | `past24_future8` | rmse | 0.850 +/- 0.072 | 2.384 +/- 0.000 | 5.051 +/- 0.000 | 0.589 +/- 0.002 | 1.885 +/- 0.094 | 5.673 +/- 0.000 |
| Mackey-Glass delay | `past24_future8` | rmse | 0.089 +/- 0.005 | 0.078 +/- 0.000 | 0.109 +/- 0.000 | 0.087 +/- 0.001 | 0.094 +/- 0.007 | 0.204 +/- 0.000 |
| Switching VAR | `past24_future8` | rmse | 0.183 +/- 0.001 | 0.185 +/- 0.000 | 0.184 +/- 0.000 | 0.179 +/- 0.000 | 0.188 +/- 0.000 | 0.226 +/- 0.000 |
| Chirp + seasonal | `past24_future8` | rmse | 0.211 +/- 0.015 | 0.162 +/- 0.000 | 0.343 +/- 0.000 | 0.197 +/- 0.000 | 0.257 +/- 0.010 | 0.746 +/- 0.000 |
| Coupled oscillators | `past24_future8` | rmse | 0.138 +/- 0.007 | 0.036 +/- 0.000 | 0.043 +/- 0.000 | 0.131 +/- 0.001 | 0.079 +/- 0.004 | 0.231 +/- 0.000 |
| Nonlinear AR | `past24_future8` | rmse | 0.115 +/- 0.008 | 0.098 +/- 0.000 | 0.101 +/- 0.000 | 0.121 +/- 0.000 | 0.107 +/- 0.002 | 0.122 +/- 0.000 |
| Nonlinear multiview | `view_a_to_view_b` | macro_f1 | 0.880 +/- 0.011 | 0.849 +/- 0.000 | 0.846 +/- 0.000 | 0.919 +/- 0.007 | 0.888 +/- 0.018 | 0.859 +/- 0.000 |
| Hierarchical classes | `view_a_to_view_b` | macro_f1 | 0.963 +/- 0.005 | 0.950 +/- 0.000 | 0.950 +/- 0.000 | 0.961 +/- 0.003 | 0.952 +/- 0.003 | 0.960 +/- 0.000 |
| Sparse interactions | `view_a_to_view_b` | macro_f1 | 0.578 +/- 0.018 | 0.665 +/- 0.000 | 0.507 +/- 0.000 | 0.657 +/- 0.011 | 0.600 +/- 0.027 | 0.935 +/- 0.000 |
| Manifold + nuisance | `view_a_to_view_b` | macro_f1 | 0.765 +/- 0.028 | 0.736 +/- 0.000 | 0.751 +/- 0.000 | 0.958 +/- 0.003 | 0.859 +/- 0.004 | 0.932 +/- 0.000 |

The classification reference uses all input features and is not a matched context-only baseline or theoretical ceiling. The forecasting reference is persistence using the last observed value of each channel.

## Dataset catalog, provenance, and limits

Detailed dataset documentation: [public datasets](docs/PUBLIC_DATASETS_EXPANDED.md) and [synthetic generators](docs/SYNTHETIC_DATASETS_EXPANDED.md).

### Breast cancer (WDBC) (`breast_cancer`)

569 retained rows, 30 flattened features; input shape `[30]`. Training/validation/test rows: 341/114/114. Primary task: `mean_to_worst` (10 context / 10 target coordinates).
Split policy: held-out source groups.
Groups by partition: train=341, val=114, test=114.
Source size: 569 rows; bounded, label-independent sample after the documented duplicate handling. Source SHA-256: `bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af`.
[Source or mechanism reference](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
Attribution: Wolberg, W., Mangasarian, O., Street, N. & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic). https://doi.org/10.24432/C5DW2B
Recorded data license: CC BY 4.0.
Thirty correlated morphology measurements across mean, standard-error and worst-value summaries; small real biomedical classification set.
Bounded new pilot partition; not an official benchmark reproduction or proof of domain transfer.

Recorded dataset limitations:

- Public historical feature dataset; not clinical validation or a diagnostic tool.
- Unique source record IDs are retained for grouping; no repeated-patient history is provided.
Realized feature-array SHA-256: `eb208df8ce11610b70c234582b5d57bbea37003b6bb047ca94b032a774097705`.

### Ionosphere (`ionosphere`)

350 retained rows, 34 flattened features; input shape `[17, 2]`. Training/validation/test rows: 210/70/70. Primary task: `alternating_pairs` (18 context / 16 target coordinates).
Split policy: label-independent random record split.
Source size: 351 rows; bounded, label-independent sample after the documented duplicate handling. Source SHA-256: `4d218ece62756c99659011a13052d06464f13cb2b3d0410ce2aef16de1403860`.
[Source or mechanism reference](https://archive.ics.uci.edu/dataset/52/ionosphere).
Attribution: Sigillito, V., Wing, S., Hutton, L. & Baker, K. (1989). Ionosphere. https://doi.org/10.24432/C5W01B
Recorded data license: CC BY 4.0.
Seventeen complex-valued radar pulse pairs, mixed constant and informative channels, and nonlinear signal structure.
Bounded new pilot partition; not an official benchmark reproduction or proof of domain transfer.

Recorded dataset limitations:

- Source acquisition/session identifiers are absent; random record splitting cannot establish independent-session generalization.
- The reconstruction mask uses the documented feature order; downstream task is classification, not forecasting.
Realized feature-array SHA-256: `f3b29f4e04dc506e89fa3ccc4ab4041348153d91e9271f640299a9c885acf3b2`.

### Sonar (`sonar`)

208 retained rows, 60 flattened features; input shape `[60, 1]`. Training/validation/test rows: 124/42/42. Primary task: `alternating` (30 context / 30 target coordinates).
Split policy: label-independent random record split.
Source size: 208 rows; bounded, label-independent sample after the documented duplicate handling. Source SHA-256: `088b0b6813fb5ab84736bc2a1e3d6bb886e317520d2f2c04152e2ed65af7a18d`.
[Source or mechanism reference](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks).
Attribution: Sejnowski, T. & Gorman, R. (1988). Connectionist Bench (Sonar, Mines vs. Rocks). https://doi.org/10.24432/C5T01Q
Recorded data license: CC BY 4.0.
Sixty ordered spectral energy bands with only 208 observations; a high-feature-to-sample real sensing problem.
Bounded new pilot partition; not an official benchmark reproduction or proof of domain transfer.

Recorded dataset limitations:

- Source acquisition/session identifiers are absent; random record splitting cannot establish independent-session generalization.
- The reconstruction mask uses the documented feature order; downstream task is classification, not forecasting.
Realized feature-array SHA-256: `45e340369853b4ea83f7bebd979c7ebf3c58fc607e453b27ec4ba55606878f99`.

### Semeion (`semeion`)

1,593 retained rows, 256 flattened features; input shape `[16, 16]`. Training/validation/test rows: 955/319/319. Primary task: `center` (192 context / 64 target coordinates).
Split policy: label-independent random record split.
Source size: 1,593 rows; bounded, label-independent sample after the documented duplicate handling. Source SHA-256: `6fb091394714cddda5751d4e1c2781ab094e7cf15de07917fb40e581f19efc75`.
[Source or mechanism reference](https://archive.ics.uci.edu/dataset/178/semeion+handwritten+digit).
Attribution: Semeion Handwritten Digit (1998). UCI Machine Learning Repository. https://doi.org/10.24432/C5SC8V
Recorded data license: CC BY 4.0.
256 binary image pixels and ten classes at 16x16 resolution, larger spatial masks than the original 8x8 Digits pilot.
Bounded new pilot partition; not an official benchmark reproduction or proof of domain transfer.

Recorded dataset limitations:

- Writer IDs are not present in the numeric records; random record splits do not demonstrate writer-independent performance.
Realized feature-array SHA-256: `30bdb6c394907e3a811704a929f4739d542187f829967a9793c158171c5e35a5`.

### Letter (`letter`)

3,000 retained rows, 16 flattened features; input shape `[16]`. Training/validation/test rows: 1,800/600/600. Primary task: `alternating` (8 context / 8 target coordinates).
Split policy: label-independent random record split.
Source size: 20,000 rows; bounded, label-independent sample after the documented duplicate handling. Source SHA-256: `3b5f07a334697b6cace4fbae22940393a18fee596e73f68d97ce5973d52dc60f`.
[Source or mechanism reference](https://archive.ics.uci.edu/dataset/59/letter+recognition).
Attribution: Slate, D. (1991). Letter Recognition. https://doi.org/10.24432/C5ZP40
Recorded data license: CC BY 4.0.
Twenty-six classes with geometric moments and edge summaries extracted from distorted font images.
Bounded new pilot partition; not an official benchmark reproduction or proof of domain transfer.

Recorded dataset limitations:

- Source font identifiers are absent; this is a new bounded row split, not the original 16000/4000 benchmark or font-independent test.
Realized feature-array SHA-256: `27ecfbe19a4c5e8e260b71c1a999a22b5e3234535109dc987f80aeb3cba4265d`.

### PenDigits (`pendigits`)

2,992 retained rows, 16 flattened features; input shape `[8, 2]`. Training/validation/test rows: 1,768/612/612. Primary task: `alternating_pairs` (8 context / 8 target coordinates).
Split policy: held-out source groups.
Groups by partition: train=26, val=9, test=9.
Source size: 10,992 rows; bounded, label-independent sample after the documented duplicate handling. Source SHA-256: `1e02bea023613c2b11c9492f6f34caf975420455934f3527d270cee9a1f03b64`.
[Source or mechanism reference](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits).
Attribution: Alpaydin, E. & Alimoglu, F. (1996). Pen-Based Recognition of Handwritten Digits. https://doi.org/10.24432/C5MG6K
Recorded data license: CC BY 4.0.
Eight ordered two-coordinate trajectory points; ten classes and preserved source grouping across 44 writer-like groups.
Bounded new pilot partition; not an official benchmark reproduction or proof of domain transfer.

Recorded dataset limitations:

- The source comment grouping is preserved conservatively; source numeric records do not themselves name writers.
- Original partitions are pooled and new source-group splits are made; this is not the published 7494/3498 benchmark.
Realized feature-array SHA-256: `ede5a3649061d67e847abd66d14a1d57e9775e6857e3e55785b8557500157b68`.

### SatImage (`satimage`)

3,000 retained rows, 36 flattened features; input shape `[36]`. Training/validation/test rows: 1,800/600/600. Primary task: `neighbor_to_center` (32 context / 4 target coordinates).
Split policy: label-independent random record split.
Source size: 6,435 rows; bounded, label-independent sample after the documented duplicate handling. Source SHA-256: `7c54e0e11c872a1b0b647da370d596dcb06746159cce4121d92ccd70b7d7ce3c`.
[Source or mechanism reference](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite).
Attribution: Srinivasan, A. (1993). Statlog (Landsat Satellite). https://doi.org/10.24432/C55887
Recorded data license: CC BY 4.0.
Nine spatial neighbors times four spectral bands; six land-cover classes and interdependent spatial/spectral features.
Bounded new pilot partition; not an official benchmark reproduction or proof of domain transfer.

Recorded dataset limitations:

- Geographic IDs are unavailable and neighboring records may overlap; row splits do not establish spatially independent generalization.
- Physical pixel/band masks are retained, but the existing pilot uses a tabular encoder, not a multispectral CNN.
- Original training/test files are pooled for a new bounded split.
Realized feature-array SHA-256: `253f29681abb0acc54b4cb5e8a597b96d2a7ae24b1a9800208dc7e756e851d72`.

### Human activity (HAR) (`har`)

3,000 retained rows, 561 flattened features; input shape `[561]`. Training/validation/test rows: 1,800/600/600. Primary task: `time_to_frequency` (265 context / 289 target coordinates).
Split policy: held-out source groups.
Groups by partition: train=18, val=6, test=6.
Source size: 10,299 rows; bounded, label-independent sample after the documented duplicate handling. Source SHA-256: `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.
[Source or mechanism reference](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
Attribution: Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L. & Parra, X. (2013). Human Activity Recognition Using Smartphones. https://doi.org/10.24432/C54S4K
Recorded data license: CC BY 4.0.
561 engineered inertial time/frequency features with six activities and explicit subject-held-out evaluation.
Bounded new pilot partition; not an official benchmark reproduction or proof of domain transfer.

Recorded dataset limitations:

- Source-provided engineered features are already normalized; this pilot cannot reconstruct or audit every original preprocessing step.
- Explicit subjects are held out; exact recording intervals are not available with the feature vectors.
- The original subject split is replaced by a declared new group split; no original benchmark score is claimed.
Realized feature-array SHA-256: `686612e6fbb781352d3b13bfa7537de4a6701a0662ada5637dc6ee5c19cea287`.

### Dry Bean (`dry_bean`)

3,000 retained rows, 16 flattened features; input shape `[16]`. Training/validation/test rows: 1,800/600/600. Primary task: `alternating` (8 context / 8 target coordinates).
Split policy: label-independent random record split.
Source size: 13,611 rows; bounded, label-independent sample after the documented duplicate handling. Source SHA-256: `0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f`.
[Source or mechanism reference](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).
Attribution: Dry Bean (2020). UCI Machine Learning Repository. Koklu, M. & Ozkan, I. A. (2020), Computers and Electronics in Agriculture 174, 105507. https://doi.org/10.24432/C50S4B
Recorded data license: CC BY 4.0.
Seven varieties with correlated geometric and shape features, imbalance and redundant derived attributes.
Bounded new pilot partition; not an official benchmark reproduction or proof of domain transfer.

Recorded dataset limitations:

- Several shape features are algebraically related; predictable context-target views need not imply semantic representation quality.
- No acquisition/batch identifier is available; row splits do not establish new-batch generalization.
Realized feature-array SHA-256: `015f68df64305717c7ebc5ebfdaada63c6a5eb22f601e7a87e1cdda3b9cdeaa4`.

### ISOLET (`isolet`)

3,000 retained rows, 617 flattened features; input shape `[617]`. Training/validation/test rows: 1,800/600/600. Primary task: `alternating` (309 context / 308 target coordinates).
Split policy: label-independent random record split.
Source size: 7,797 rows; bounded, label-independent sample after the documented duplicate handling. Source SHA-256: `fe2e0d45f1057d112e051d309070992fc178e7f6922cc7b4e8b2bd310303a053`.
[Source or mechanism reference](https://archive.ics.uci.edu/dataset/54/isolet).
Attribution: Cole, R. & Fanty, M. (1991). ISOLET. https://doi.org/10.24432/C51G69
Recorded data license: CC BY 4.0.
617 acoustic summary features with 26 spoken-letter classes; substantially higher dimensional than the initial benchmark.
Bounded new pilot partition; not an official benchmark reproduction or proof of domain transfer.

Recorded dataset limitations:

- Explicit speaker IDs are not in numeric records; the source reports missing recordings, so identities are not guessed from fixed row blocks.
- The published speaker-separated partitions are pooled for the bounded row-split pilot; results do not establish speaker-independent performance.
- The source says exact acoustic feature order is unknown; masks are numerical partitions, not semantic frequency groups.
Realized feature-array SHA-256: `97da5c68fdee4a2888dac8a9dd25da5b3bf47d842bbc86a8bf1602152f48b275`.

### Lorenz trajectories (`synthetic_lorenz`)

2,000 retained rows, 96 flattened features; input shape `[32, 3]`. Training/validation/test rows: 1,200/400/400. Primary task: `past24_future8` (72 context / 24 target coordinates).
Split policy: held-out independent entities; every overlapping window from one entity stays in one split.
Groups by partition: train=60, val=20, test=20.
[Source or mechanism reference](https://journals.ametsoc.org/view/journals/atsc/20/2/1520-0469_1963_020_0130_dnf_2_0_co_2.xml).
Recorded data license: CC0-1.0.
Generator: Lorenz differential equations integrated by RK4; independent initial states/rho and noisy three-state observations

```json
{
  "entities": 100,
  "windows_per_entity": 20,
  "window_timesteps": 32,
  "stride": 8,
  "channels": 3,
  "retained_timesteps_per_entity": 184,
  "equations": [
    "dx/dt=10(y-x)",
    "dy/dt=x(rho-z)-y",
    "dz/dt=x*y-(8/3)z"
  ],
  "rho_range": [
    26.0,
    30.0
  ],
  "sigma": 10.0,
  "beta": 2.6666666666666665,
  "dt": 0.02,
  "burn_in_steps": 500,
  "measurement_noise_std": 0.05,
  "initial_state_min": [
    -12.0,
    -12.0,
    8.0
  ],
  "initial_state_max": [
    12.0,
    12.0,
    28.0
  ],
  "adaptation": "Entity rho variation, observation noise, RK4, and windowing are pilot choices, not a reproduction of the original paper."
}
```
Controlled synthetic mechanism; no claim of calibrated scientific realism, physical validity, or real-world effectiveness.
Realized feature-array SHA-256: `bcc33504b98d8797eda1d79773bdeaa3f89ad6f764fa611b48b3f96929cc5f13`.

### Mackey-Glass delay (`synthetic_mackey_glass`)

2,000 retained rows, 96 flattened features; input shape `[32, 3]`. Training/validation/test rows: 1,200/400/400. Primary task: `past24_future8` (72 context / 24 target coordinates).
Split policy: held-out independent entities; every overlapping window from one entity stays in one split.
Groups by partition: train=60, val=20, test=20.
[Source or mechanism reference](https://doi.org/10.1126/science.267326).
Recorded data license: CC0-1.0.
Generator: Noisy discrete Euler Mackey-Glass-inspired delayed feedback with three causal observation channels

```json
{
  "entities": 100,
  "windows_per_entity": 20,
  "window_timesteps": 32,
  "stride": 8,
  "channels": 3,
  "retained_timesteps_per_entity": 184,
  "equation": "x[t]=max(.01,(1-gamma)*x[t-1]+beta*x[t-1-delay]/(1+x[t-1-delay]^10)+epsilon)",
  "dt": 1.0,
  "delay_integer_range_inclusive": [
    14,
    22
  ],
  "beta_range": [
    0.18,
    0.23
  ],
  "gamma_range": [
    0.08,
    0.12
  ],
  "exponent": 10,
  "burn_in_steps": 400,
  "initial_history_range": [
    0.8,
    1.3
  ],
  "initial_history_noise_std": 0.02,
  "innovation_std": 0.002,
  "measurement_noise_std": 0.005,
  "observations": [
    "x[t]",
    "x[t-4]",
    "x[t]^2+.2*x[t-8]"
  ],
  "adaptation": "Discrete dt=1 recursion, parameter ranges, positivity floor, process noise and sensor transforms are deliberate adaptations; not a validated DDE solver."
}
```
Controlled synthetic mechanism; no claim of calibrated scientific realism, physical validity, or real-world effectiveness.
Realized feature-array SHA-256: `e8554ae6c05218c76222720f0f3ae140fd33e982f6ff188e66172fe1edd5bb46`.

### Switching VAR (`synthetic_switching_var`)

2,000 retained rows, 128 flattened features; input shape `[32, 4]`. Training/validation/test rows: 1,200/400/400. Primary task: `past24_future8` (96 context / 32 target coordinates).
Split policy: held-out independent entities; every overlapping window from one entity stays in one split.
Groups by partition: train=60, val=20, test=20.
Recorded data license: CC0-1.0.
Generator: Three-regime four-channel vector autoregression with persistent latent Markov switching and entity-specific noise scale

```json
{
  "entities": 100,
  "windows_per_entity": 20,
  "window_timesteps": 32,
  "stride": 8,
  "channels": 4,
  "retained_timesteps_per_entity": 184,
  "equation": "x[t]=A[regime[t]]@x[t-1]+b[regime[t]]+entity_scale*epsilon[t]",
  "transition_rule": "With entity probability p switch uniformly to one of the two other regimes; otherwise retain regime",
  "switch_probability_range": [
    0.025,
    0.09
  ],
  "transition_matrices": [
    [
      [
        0.7,
        0.15,
        0.0,
        0.0
      ],
      [
        -0.1,
        0.65,
        0.1,
        0.0
      ],
      [
        0.0,
        -0.15,
        0.6,
        0.15
      ],
      [
        0.1,
        0.0,
        0.0,
        0.65
      ]
    ],
    [
      [
        0.4,
        -0.35,
        0.0,
        0.1
      ],
      [
        0.3,
        0.35,
        0.1,
        0.0
      ],
      [
        0.1,
        0.0,
        0.5,
        -0.25
      ],
      [
        0.0,
        0.1,
        0.25,
        0.45
      ]
    ],
    [
      [
        -0.45,
        0.15,
        0.0,
        0.0
      ],
      [
        0.1,
        -0.4,
        0.2,
        0.0
      ],
      [
        0.0,
        0.1,
        -0.35,
        0.2
      ],
      [
        0.15,
        0.0,
        0.1,
        -0.4
      ]
    ]
  ],
  "regime_offsets": [
    [
      0.1,
      -0.08,
      0.05,
      0.0
    ],
    [
      -0.12,
      0.08,
      0.0,
      0.1
    ],
    [
      0.0,
      0.0,
      -0.1,
      -0.08
    ]
  ],
  "entity_noise_scale_range": [
    0.8,
    1.3
  ],
  "innovation_std": 0.11,
  "measurement_noise_std": 0.025,
  "burn_in_steps": 100,
  "latent_regime_is_input": false
}
```
Controlled synthetic mechanism; no claim of calibrated scientific realism, physical validity, or real-world effectiveness.
Realized feature-array SHA-256: `3bcb30a94db5f688719e14d04f9f9f0ed57306347578aa0babff34b29e574c9d`.

### Chirp + seasonal (`synthetic_chirp_seasonal`)

2,000 retained rows, 96 flattened features; input shape `[32, 3]`. Training/validation/test rows: 1,200/400/400. Primary task: `past24_future8` (72 context / 24 target coordinates).
Split policy: held-out independent entities; every overlapping window from one entity stays in one split.
Groups by partition: train=60, val=20, test=20.
Recorded data license: CC0-1.0.
Generator: Nonstationary chirp phase plus slow amplitude modulation, fixed seasonal component, linear trend and colored noise

```json
{
  "entities": 100,
  "windows_per_entity": 20,
  "window_timesteps": 32,
  "stride": 8,
  "channels": 3,
  "retained_timesteps_per_entity": 184,
  "chirp_phase": "2*pi*(f0*t+.5*drift*t^2)+phase",
  "initial_frequency_range": [
    0.012,
    0.035
  ],
  "frequency_drift_range": [
    4e-05,
    0.00018
  ],
  "amplitude_range": [
    0.7,
    1.5
  ],
  "envelope": "amplitude*(1+.25*sin(2*pi*t/90+phase))",
  "season": ".35*sin(2*pi*t/18+phase/2)",
  "slope_range": [
    -0.004,
    0.004
  ],
  "ar_coefficient": 0.75,
  "ar_innovation_std": 0.04,
  "measurement_noise_std": 0.03,
  "phase_range": [
    -3.141592653589793,
    3.141592653589793
  ]
}
```
Controlled synthetic mechanism; no claim of calibrated scientific realism, physical validity, or real-world effectiveness.
Realized feature-array SHA-256: `63633cd23b7efe1ee8b88314f297f06ea7b67c1859c6c588bbf2cbd8ba3c9519`.

### Coupled oscillators (`synthetic_coupled_oscillators`)

2,000 retained rows, 128 flattened features; input shape `[32, 4]`. Training/validation/test rows: 1,200/400/400. Primary task: `past24_future8` (96 context / 32 target coordinates).
Split policy: held-out independent entities; every overlapping window from one entity stays in one split.
Groups by partition: train=60, val=20, test=20.
Recorded data license: CC0-1.0.
Generator: Two bidirectionally coupled damped cubic oscillators with independent driving phases and noisy semi-implicit Euler integration

```json
{
  "entities": 100,
  "windows_per_entity": 20,
  "window_timesteps": 32,
  "stride": 8,
  "channels": 4,
  "retained_timesteps_per_entity": 184,
  "acceleration": "-omega_i^2*q_i-.12*q_i^3-.10*v_i+k*(q_j-q_i)+.35*sin(omega_drive*t+phase_i)",
  "integration": "v_next=v+dt*acceleration+Normal(0,.008); q_next=q+dt*v_next",
  "dt": 0.08,
  "natural_frequency_range": [
    0.7,
    1.2
  ],
  "coupling_range": [
    0.12,
    0.35
  ],
  "drive_frequency_range": [
    0.5,
    0.9
  ],
  "channel_order": [
    "q0",
    "q1",
    "v0",
    "v1"
  ],
  "measurement_noise_std": 0.015,
  "burn_in_steps": 250,
  "adaptation": "Authored coupled cubic oscillator benchmark; numerical trajectories are not physical validation."
}
```
Controlled synthetic mechanism; no claim of calibrated scientific realism, physical validity, or real-world effectiveness.
Realized feature-array SHA-256: `b42b8d12e2e080a80cf3cbb648384367c2f987018c6fc99d1abf04e913699063`.

### Nonlinear AR (`synthetic_nonlinear_ar`)

2,000 retained rows, 96 flattened features; input shape `[32, 3]`. Training/validation/test rows: 1,200/400/400. Primary task: `past24_future8` (72 context / 24 target coordinates).
Split policy: held-out independent entities; every overlapping window from one entity stays in one split.
Groups by partition: train=60, val=20, test=20.
Recorded data license: CC0-1.0.
Generator: Three-channel nonlinear autoregression with multiplicative cross-lag interactions and state-dependent innovation variance

```json
{
  "entities": 100,
  "windows_per_entity": 20,
  "window_timesteps": 32,
  "stride": 8,
  "channels": 3,
  "retained_timesteps_per_entity": 184,
  "equations_before_gain_bias_noise": [
    "x[t]=.60*x[t-1]+.40*tanh(y[t-2]*z[t-1])",
    "y[t]=.45*y[t-1]+.35*sin(x[t-1])-.15*z[t-3]",
    "z[t]=.50*z[t-1]+.30*tanh(x[t-1]-y[t-2])"
  ],
  "entity_gain_range": [
    0.75,
    1.15
  ],
  "entity_bias_std": 0.1,
  "innovation_std": ".06+.06*abs(tanh(previous_channel_state))",
  "measurement_noise_std": 0.01,
  "burn_in_steps": 200,
  "maximum_lag": 3
}
```
Controlled synthetic mechanism; no claim of calibrated scientific realism, physical validity, or real-world effectiveness.
Realized feature-array SHA-256: `931668fa22eea2691b9891a55e6205af789e31254276ab6a5fc7e94e7a76aa0d`.

### Nonlinear multiview (`synthetic_nonlinear_multiview`)

2,000 retained rows, 64 flattened features; input shape `[64]`. Training/validation/test rows: 1,200/400/400. Primary task: `view_a_to_view_b` (32 context / 32 target coordinates).
Split policy: random independent rows, independent of labels.
Recorded data license: CC0-1.0.
Generator: Two nonlinear noisy measurements of six shared Gaussian latent factors; three evaluation classes from latent nonlinear scores

```json
{
  "rows": 2000,
  "latent_dimensions": 6,
  "view_dimensions": [
    32,
    32
  ],
  "measurement_noise_std": 0.08,
  "view_a": "tanh(z@unit_column_A)+.15*z0*z1*w_a+noise",
  "view_b": "sin(z@unit_column_B)+.20*(z2^2-z3^2)*w_b+noise",
  "class_rule": "argmax(z0+.8*z1*z2, -z0+.6*sin(z3), z4-.5*z5+.4*z1^2)"
}
```
Controlled synthetic mechanism; no claim of calibrated scientific realism, physical validity, or real-world effectiveness.
Realized feature-array SHA-256: `b2d37f79f8971f562e3a18a4159a90fd5348fa679fc17e753a8e8a3978ce19e4`.

### Hierarchical classes (`synthetic_hierarchical_multiclass`)

2,000 retained rows, 80 flattened features; input shape `[80]`. Training/validation/test rows: 1,200/400/400. Primary task: `view_a_to_view_b` (40 context / 40 target coordinates).
Split policy: random independent rows, independent of labels.
Recorded data license: CC0-1.0.
Generator: Six latent Gaussian subclasses nested under three separated parent centers, observed through linear and nonlinear views

```json
{
  "rows": 2000,
  "classes": 6,
  "parent_classes": 3,
  "subclasses_per_parent": 2,
  "latent_dimensions": 8,
  "parent_circle_radius": 2.2,
  "child_offset": 0.9,
  "latent_noise_std": 0.45,
  "measurement_noise_std": 0.12,
  "view_dimensions": [
    40,
    40
  ],
  "latent_rule": "z=parent_center[y//2]+(2*(y%2)-1)*.9*parent_child_direction+Normal(0,.45)",
  "view_a": "z@unit_column_A+noise",
  "view_b": "tanh(z@unit_column_B)+.12*z2*z3*w+noise",
  "label_note": "Class IDs determine latent mixture membership only; no class ID or parent ID is appended to X."
}
```
Controlled synthetic mechanism; no claim of calibrated scientific realism, physical validity, or real-world effectiveness.
Realized feature-array SHA-256: `914464aec3d58a82c0d88e23a0e4d2b58aea34a7da1e4f2af8c7f19610f6b919`.

### Sparse interactions (`synthetic_sparse_interactions`)

2,000 retained rows, 96 flattened features; input shape `[96]`. Training/validation/test rows: 1,200/400/400. Primary task: `view_a_to_view_b` (48 context / 48 target coordinates).
Split policy: random independent rows, independent of labels.
Recorded data license: CC0-1.0.
Generator: Sparse nonlinear classification signal among 70 independent nuisance columns; source factors and pairwise products occupy separate views

```json
{
  "rows": 2000,
  "latent_dimensions": 10,
  "view_dimensions": [
    48,
    48
  ],
  "view_a_signal_columns": 10,
  "view_b_interaction_columns": 16,
  "independent_nuisance_columns": 70,
  "interaction_pairs": [
    [
      0,
      1
    ],
    [
      1,
      2
    ],
    [
      2,
      3
    ],
    [
      3,
      4
    ],
    [
      4,
      5
    ],
    [
      5,
      6
    ],
    [
      6,
      7
    ],
    [
      7,
      8
    ],
    [
      8,
      9
    ],
    [
      9,
      0
    ],
    [
      0,
      4
    ],
    [
      1,
      5
    ],
    [
      2,
      6
    ],
    [
      3,
      7
    ],
    [
      4,
      8
    ],
    [
      5,
      9
    ]
  ],
  "source_noise_std": 0.04,
  "interaction_noise_std": 0.06,
  "label_noise_std": 0.1,
  "class_rule": "1[z0*z1+.8*z2*z3-.5*z4+.35*sin(z5)+Normal(0,.1)>0]"
}
```
Controlled synthetic mechanism; no claim of calibrated scientific realism, physical validity, or real-world effectiveness.
Realized feature-array SHA-256: `18bd7c82a978d6ae09da22bbfc5ff30e45b1f5f4f5cc2df06f31d7d3ae5eb689`.

### Manifold + nuisance (`synthetic_manifold_nuisance`)

2,000 retained rows, 96 flattened features; input shape `[96]`. Training/validation/test rows: 1,200/400/400. Primary task: `view_a_to_view_b` (48 context / 48 target coordinates).
Split policy: random independent rows, independent of labels.
Recorded data license: CC0-1.0.
Generator: Rolled two-dimensional manifold observed through two different embeddings with 64 independent-of-label correlated nuisance measurements

```json
{
  "rows": 2000,
  "intrinsic_dimensions": 2,
  "view_dimensions": [
    48,
    48
  ],
  "signal_columns_per_view": 16,
  "nuisance_columns_per_view": 32,
  "theta_range": [
    4.71238898038469,
    14.137166941154069
  ],
  "height_range": [
    -1.0,
    1.0
  ],
  "view_a_manifold": "(theta*cos(theta)/8, height, theta*sin(theta)/8)",
  "view_b_basis": "(sin(theta),cos(theta),height,height*sin(theta))",
  "nuisance_latent_dimensions_per_view": 4,
  "nuisance_scale": 2.5,
  "signal_noise_std": 0.08,
  "nuisance_noise_std": 0.2,
  "class_rule": "2*floor((theta-1.5*pi)/pi)+1[height>0]",
  "nuisance_note": "Each view has independently drawn nuisance factors; abundance remains a distractor after train-fitted per-column scaling."
}
```
Controlled synthetic mechanism; no claim of calibrated scientific realism, physical validity, or real-world effectiveness.
Realized feature-array SHA-256: `fcdbcdf052b9b8e84e3c7f168efabc2811e56d537a3ec3c6f130270d069767af`.

## Every baseline under every new policy

This appendix retains all 40 policy tasks and all six methods. Full per-seed accuracy, macro F1, MAE/RMSE, validation histories, and diagnostics are preserved in the result JSON.

### Breast cancer (WDBC) / `mean_to_worst` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.919 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.919 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.899 +/- 0.012 | 3 | Context only |
| Random encoder | macro_f1 | 0.908 +/- 0.006 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.921 +/- 0.001 | 3 | Context only |
| Full-input reference | macro_f1 | 0.980 +/- 0.000 | 3 | Full input reference |

### Breast cancer (WDBC) / `mean_to_error` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.919 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.919 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.899 +/- 0.012 | 3 | Context only |
| Random encoder | macro_f1 | 0.908 +/- 0.006 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.921 +/- 0.001 | 3 | Context only |
| Full-input reference | macro_f1 | 0.980 +/- 0.000 | 3 | Full input reference |

### Ionosphere / `alternating_pairs` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.746 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.672 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.896 +/- 0.015 | 3 | Context only |
| Random encoder | macro_f1 | 0.811 +/- 0.057 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.770 +/- 0.046 | 3 | Context only |
| Full-input reference | macro_f1 | 0.813 +/- 0.000 | 3 | Full input reference |

### Ionosphere / `first_half` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.833 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.796 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.872 +/- 0.009 | 3 | Context only |
| Random encoder | macro_f1 | 0.866 +/- 0.028 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.798 +/- 0.018 | 3 | Context only |
| Full-input reference | macro_f1 | 0.813 +/- 0.000 | 3 | Full input reference |

### Sonar / `alternating` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.708 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.686 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.809 +/- 0.024 | 3 | Context only |
| Random encoder | macro_f1 | 0.704 +/- 0.053 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.597 +/- 0.075 | 3 | Context only |
| Full-input reference | macro_f1 | 0.708 +/- 0.000 | 3 | Full input reference |

### Sonar / `first_half` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.638 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.641 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.761 +/- 0.023 | 3 | Context only |
| Random encoder | macro_f1 | 0.661 +/- 0.022 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.573 +/- 0.009 | 3 | Context only |
| Full-input reference | macro_f1 | 0.708 +/- 0.000 | 3 | Full input reference |

### Semeion / `center` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.909 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.794 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.908 +/- 0.007 | 3 | Context only |
| Random encoder | macro_f1 | 0.722 +/- 0.021 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.849 +/- 0.006 | 3 | Context only |
| Full-input reference | macro_f1 | 0.940 +/- 0.000 | 3 | Full input reference |

### Semeion / `right_half` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.781 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.675 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.836 +/- 0.008 | 3 | Context only |
| Random encoder | macro_f1 | 0.640 +/- 0.037 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.746 +/- 0.045 | 3 | Context only |
| Full-input reference | macro_f1 | 0.940 +/- 0.000 | 3 | Full input reference |

### Letter / `alternating` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.498 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.497 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.700 +/- 0.011 | 3 | Context only |
| Random encoder | macro_f1 | 0.602 +/- 0.012 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.522 +/- 0.002 | 3 | Context only |
| Full-input reference | macro_f1 | 0.696 +/- 0.000 | 3 | Full input reference |

### Letter / `first_half` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.298 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.298 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.447 +/- 0.003 | 3 | Context only |
| Random encoder | macro_f1 | 0.396 +/- 0.011 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.329 +/- 0.011 | 3 | Context only |
| Full-input reference | macro_f1 | 0.696 +/- 0.000 | 3 | Full input reference |

### PenDigits / `alternating_pairs` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.833 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.833 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.911 +/- 0.001 | 3 | Context only |
| Random encoder | macro_f1 | 0.901 +/- 0.001 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.898 +/- 0.005 | 3 | Context only |
| Full-input reference | macro_f1 | 0.925 +/- 0.000 | 3 | Full input reference |

### PenDigits / `first_half` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.703 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.703 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.827 +/- 0.008 | 3 | Context only |
| Random encoder | macro_f1 | 0.828 +/- 0.004 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.819 +/- 0.015 | 3 | Context only |
| Full-input reference | macro_f1 | 0.925 +/- 0.000 | 3 | Full input reference |

### SatImage / `neighbor_to_center` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.806 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.792 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.868 +/- 0.008 | 3 | Context only |
| Random encoder | macro_f1 | 0.839 +/- 0.010 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.799 +/- 0.020 | 3 | Context only |
| Full-input reference | macro_f1 | 0.813 +/- 0.000 | 3 | Full input reference |

### SatImage / `spectral_halves` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.746 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.734 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.836 +/- 0.003 | 3 | Context only |
| Random encoder | macro_f1 | 0.802 +/- 0.012 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.761 +/- 0.023 | 3 | Context only |
| Full-input reference | macro_f1 | 0.813 +/- 0.000 | 3 | Full input reference |

### Human activity (HAR) / `time_to_frequency` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.929 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.804 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.916 +/- 0.001 | 3 | Context only |
| Random encoder | macro_f1 | 0.789 +/- 0.008 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.796 +/- 0.014 | 3 | Context only |
| Full-input reference | macro_f1 | 0.912 +/- 0.000 | 3 | Full input reference |

### Human activity (HAR) / `alternating` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.897 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.830 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.915 +/- 0.014 | 3 | Context only |
| Random encoder | macro_f1 | 0.783 +/- 0.032 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.773 +/- 0.015 | 3 | Context only |
| Full-input reference | macro_f1 | 0.912 +/- 0.000 | 3 | Full input reference |

### Dry Bean / `alternating` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.914 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.914 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.916 +/- 0.003 | 3 | Context only |
| Random encoder | macro_f1 | 0.923 +/- 0.008 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.914 +/- 0.002 | 3 | Context only |
| Full-input reference | macro_f1 | 0.943 +/- 0.000 | 3 | Full input reference |

### Dry Bean / `first_half` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.920 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.922 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.899 +/- 0.006 | 3 | Context only |
| Random encoder | macro_f1 | 0.921 +/- 0.001 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.902 +/- 0.007 | 3 | Context only |
| Full-input reference | macro_f1 | 0.943 +/- 0.000 | 3 | Full input reference |

### ISOLET / `alternating` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.929 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.711 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.932 +/- 0.006 | 3 | Context only |
| Random encoder | macro_f1 | 0.703 +/- 0.013 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.748 +/- 0.005 | 3 | Context only |
| Full-input reference | macro_f1 | 0.934 +/- 0.000 | 3 | Full input reference |

### ISOLET / `first_half` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.668 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.452 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.642 +/- 0.012 | 3 | Context only |
| Random encoder | macro_f1 | 0.522 +/- 0.013 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.537 +/- 0.030 | 3 | Context only |
| Full-input reference | macro_f1 | 0.934 +/- 0.000 | 3 | Full input reference |

### Lorenz trajectories / `past24_future8` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | rmse | 2.384 +/- 0.000 | 3 | Context only |
| PCA + linear | rmse | 5.051 +/- 0.000 | 3 | Context only |
| Extra Trees | rmse | 0.589 +/- 0.002 | 3 | Context only |
| Random encoder | rmse | 1.885 +/- 0.094 | 3 | Context only |
| JEPA + linear | rmse | 0.850 +/- 0.072 | 3 | Context only |
| Persistence | rmse | 5.673 +/- 0.000 | 3 | Context only |

### Lorenz trajectories / `past16_future16` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | rmse | 4.934 +/- 0.000 | 3 | Context only |
| PCA + linear | rmse | 5.981 +/- 0.000 | 3 | Context only |
| Extra Trees | rmse | 0.782 +/- 0.010 | 3 | Context only |
| Random encoder | rmse | 3.773 +/- 0.069 | 3 | Context only |
| JEPA + linear | rmse | 1.297 +/- 0.057 | 3 | Context only |
| Persistence | rmse | 8.637 +/- 0.000 | 3 | Context only |

### Mackey-Glass delay / `past24_future8` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | rmse | 0.078 +/- 0.000 | 3 | Context only |
| PCA + linear | rmse | 0.109 +/- 0.000 | 3 | Context only |
| Extra Trees | rmse | 0.087 +/- 0.001 | 3 | Context only |
| Random encoder | rmse | 0.094 +/- 0.007 | 3 | Context only |
| JEPA + linear | rmse | 0.089 +/- 0.005 | 3 | Context only |
| Persistence | rmse | 0.204 +/- 0.000 | 3 | Context only |

### Mackey-Glass delay / `past16_future16` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | rmse | 0.141 +/- 0.000 | 3 | Context only |
| PCA + linear | rmse | 0.152 +/- 0.000 | 3 | Context only |
| Extra Trees | rmse | 0.126 +/- 0.001 | 3 | Context only |
| Random encoder | rmse | 0.134 +/- 0.002 | 3 | Context only |
| JEPA + linear | rmse | 0.131 +/- 0.003 | 3 | Context only |
| Persistence | rmse | 0.335 +/- 0.000 | 3 | Context only |

### Switching VAR / `past24_future8` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | rmse | 0.185 +/- 0.000 | 3 | Context only |
| PCA + linear | rmse | 0.184 +/- 0.000 | 3 | Context only |
| Extra Trees | rmse | 0.179 +/- 0.000 | 3 | Context only |
| Random encoder | rmse | 0.188 +/- 0.000 | 3 | Context only |
| JEPA + linear | rmse | 0.183 +/- 0.001 | 3 | Context only |
| Persistence | rmse | 0.226 +/- 0.000 | 3 | Context only |

### Switching VAR / `past16_future16` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | rmse | 0.190 +/- 0.000 | 3 | Context only |
| PCA + linear | rmse | 0.189 +/- 0.000 | 3 | Context only |
| Extra Trees | rmse | 0.186 +/- 0.000 | 3 | Context only |
| Random encoder | rmse | 0.191 +/- 0.001 | 3 | Context only |
| JEPA + linear | rmse | 0.188 +/- 0.001 | 3 | Context only |
| Persistence | rmse | 0.241 +/- 0.000 | 3 | Context only |

### Chirp + seasonal / `past24_future8` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | rmse | 0.162 +/- 0.000 | 3 | Context only |
| PCA + linear | rmse | 0.343 +/- 0.000 | 3 | Context only |
| Extra Trees | rmse | 0.197 +/- 0.000 | 3 | Context only |
| Random encoder | rmse | 0.257 +/- 0.010 | 3 | Context only |
| JEPA + linear | rmse | 0.211 +/- 0.015 | 3 | Context only |
| Persistence | rmse | 0.746 +/- 0.000 | 3 | Context only |

### Chirp + seasonal / `past16_future16` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | rmse | 0.283 +/- 0.000 | 3 | Context only |
| PCA + linear | rmse | 0.396 +/- 0.000 | 3 | Context only |
| Extra Trees | rmse | 0.265 +/- 0.001 | 3 | Context only |
| Random encoder | rmse | 0.347 +/- 0.007 | 3 | Context only |
| JEPA + linear | rmse | 0.279 +/- 0.010 | 3 | Context only |
| Persistence | rmse | 0.979 +/- 0.000 | 3 | Context only |

### Coupled oscillators / `past24_future8` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | rmse | 0.036 +/- 0.000 | 3 | Context only |
| PCA + linear | rmse | 0.043 +/- 0.000 | 3 | Context only |
| Extra Trees | rmse | 0.131 +/- 0.001 | 3 | Context only |
| Random encoder | rmse | 0.079 +/- 0.004 | 3 | Context only |
| JEPA + linear | rmse | 0.138 +/- 0.007 | 3 | Context only |
| Persistence | rmse | 0.231 +/- 0.000 | 3 | Context only |

### Coupled oscillators / `past16_future16` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | rmse | 0.064 +/- 0.000 | 3 | Context only |
| PCA + linear | rmse | 0.065 +/- 0.000 | 3 | Context only |
| Extra Trees | rmse | 0.145 +/- 0.001 | 3 | Context only |
| Random encoder | rmse | 0.094 +/- 0.005 | 3 | Context only |
| JEPA + linear | rmse | 0.147 +/- 0.019 | 3 | Context only |
| Persistence | rmse | 0.432 +/- 0.000 | 3 | Context only |

### Nonlinear AR / `past24_future8` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | rmse | 0.098 +/- 0.000 | 3 | Context only |
| PCA + linear | rmse | 0.101 +/- 0.000 | 3 | Context only |
| Extra Trees | rmse | 0.121 +/- 0.000 | 3 | Context only |
| Random encoder | rmse | 0.107 +/- 0.002 | 3 | Context only |
| JEPA + linear | rmse | 0.115 +/- 0.008 | 3 | Context only |
| Persistence | rmse | 0.122 +/- 0.000 | 3 | Context only |

### Nonlinear AR / `past16_future16` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | rmse | 0.103 +/- 0.000 | 3 | Context only |
| PCA + linear | rmse | 0.104 +/- 0.000 | 3 | Context only |
| Extra Trees | rmse | 0.124 +/- 0.000 | 3 | Context only |
| Random encoder | rmse | 0.112 +/- 0.004 | 3 | Context only |
| JEPA + linear | rmse | 0.115 +/- 0.008 | 3 | Context only |
| Persistence | rmse | 0.128 +/- 0.000 | 3 | Context only |

### Nonlinear multiview / `view_a_to_view_b` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.849 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.846 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.919 +/- 0.007 | 3 | Context only |
| Random encoder | macro_f1 | 0.888 +/- 0.018 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.880 +/- 0.011 | 3 | Context only |
| Full-input reference | macro_f1 | 0.859 +/- 0.000 | 3 | Full input reference |

### Nonlinear multiview / `alternating` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.846 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.846 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.901 +/- 0.008 | 3 | Context only |
| Random encoder | macro_f1 | 0.884 +/- 0.007 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.880 +/- 0.014 | 3 | Context only |
| Full-input reference | macro_f1 | 0.859 +/- 0.000 | 3 | Full input reference |

### Hierarchical classes / `view_a_to_view_b` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.950 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.950 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.961 +/- 0.003 | 3 | Context only |
| Random encoder | macro_f1 | 0.952 +/- 0.003 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.963 +/- 0.005 | 3 | Context only |
| Full-input reference | macro_f1 | 0.960 +/- 0.000 | 3 | Full input reference |

### Hierarchical classes / `alternating` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.960 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.962 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.964 +/- 0.002 | 3 | Context only |
| Random encoder | macro_f1 | 0.945 +/- 0.001 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.962 +/- 0.004 | 3 | Context only |
| Full-input reference | macro_f1 | 0.960 +/- 0.000 | 3 | Full input reference |

### Sparse interactions / `view_a_to_view_b` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.665 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.507 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.657 +/- 0.011 | 3 | Context only |
| Random encoder | macro_f1 | 0.600 +/- 0.027 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.578 +/- 0.018 | 3 | Context only |
| Full-input reference | macro_f1 | 0.935 +/- 0.000 | 3 | Full input reference |

### Sparse interactions / `alternating` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.935 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.564 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.889 +/- 0.018 | 3 | Context only |
| Random encoder | macro_f1 | 0.710 +/- 0.020 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.757 +/- 0.020 | 3 | Context only |
| Full-input reference | macro_f1 | 0.935 +/- 0.000 | 3 | Full input reference |

### Manifold + nuisance / `view_a_to_view_b` (primary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.736 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.751 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.958 +/- 0.003 | 3 | Context only |
| Random encoder | macro_f1 | 0.859 +/- 0.004 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.765 +/- 0.028 | 3 | Context only |
| Full-input reference | macro_f1 | 0.932 +/- 0.000 | 3 | Full input reference |

### Manifold + nuisance / `alternating` (secondary)

| Method | Test metric | Mean +/- SD | Seeds | Information |
| --- | --- | ---: | ---: | --- |
| Raw linear | macro_f1 | 0.908 +/- 0.000 | 3 | Context only |
| PCA + linear | macro_f1 | 0.659 +/- 0.000 | 3 | Context only |
| Extra Trees | macro_f1 | 0.953 +/- 0.009 | 3 | Context only |
| Random encoder | macro_f1 | 0.788 +/- 0.022 | 3 | Context only |
| JEPA + linear | macro_f1 | 0.843 +/- 0.031 | 3 | Context only |
| Full-input reference | macro_f1 | 0.932 +/- 0.000 | 3 | Full input reference |

## Original three-dataset snapshot

The original 18 measurements are preserved rather than represented as reruns under the expanded source tree. Their runtime source files are archived in `results/original_runtime.zip`. Its historic hashes are audited separately; current runtime hash checks apply to the new 120-run expansion.

| Dataset / primary policy | Metric | JEPA | Raw linear | Extra Trees |
| --- | --- | ---: | ---: | ---: |
| Wine / `alternating` | macro_f1 | 0.914 +/- 0.000 | 0.941 +/- 0.000 | 0.980 +/- 0.018 |
| Digits / `center` | macro_f1 | 0.886 +/- 0.008 | 0.875 +/- 0.000 | 0.913 +/- 0.008 |
| Original sensors / `past16_future8` | rmse | 0.321 +/- 0.021 | 0.259 +/- 0.000 | 0.358 +/- 0.004 |

## Interpretation boundaries and next research work

This expansion broadens structural coverage and exposes where a small encoder is competitive or limited. Dataset labels, sampling caps, synthetic equations, and one fixed split constrain the interpretation. Results do not establish clinical utility, physical calibration, generalization to unseen populations, real-world deployment quality, developer productivity, or broad state-of-the-art performance.

Priority follow-up studies are independent splits or external cohorts; targeted analysis of low-rank runs and generator-specific failures; and stronger modality-specific architectures under matched compute and tuning budgets. Do not choose among the reported task policies using their test scores and then report that choice as independently validated.

## Reproduction and evidence

```bash
python -m pip install '.[dev,report]'
python -m pytest -q --junitxml=artifacts/expanded_pytest.xml
python scripts/run_experiments.py --config configs/expanded.json --output artifacts/expanded
python scripts/validate_expanded.py --config configs/expanded.json --results artifacts/expanded/results.json --output results/expanded_validation.json
python scripts/plot_expanded.py
python scripts/build_expanded_report.py
```

To rebuild the report from the published measured JSON without rerunning training or downloading datasets:

```bash
python scripts/build_expanded_report.py --expanded results/expanded_benchmark.json.gz
```

The gzip file preserves the original JSON bytes exactly and uses a deterministic timestamp of zero. Report validation compares the SHA-256 of its decompressed payload with the independent validation record. The compressed-file SHA is recorded separately in `results/expanded_publication.json`. Full binary export/checkpoint validation requires the local experiment artifacts or their reproducibility bundle.

Use a process with access to the Metal GPU; the benchmark rejects silent CPU fallback. The validator requires 20 datasets, two policies each, three seeds, complete artifacts, and current runtime hashes. It performs no GPU training.

Expanded result SHA-256: `8f223e56d42f1a23bda989cdfa1419c2e261843627fd1e4db8f007dc1c8d90a5`.
Expanded config SHA-256: `77d37d71486cd416447d9f3867333bd2ea941eeb2dbbff6d83c3b32a28be5ef1`.
The exported manifests retain precise context/target indices, split identities, normalization, source attribution, and generator parameters. Checksums detect modifications; they do not authenticate authorship.
