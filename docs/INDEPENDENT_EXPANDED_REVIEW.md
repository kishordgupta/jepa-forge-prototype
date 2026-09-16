# Independent review of the expanded benchmark

**Verdict: accept as a bounded, honestly reported prototype experiment.** No blocking mismatch was found between the frozen protocol, recorded dataset metadata and completed results. This is an independent implementation/evidence review within the project, not an external scientific replication.

Reviewed `artifacts/expanded/results.json` (SHA-256 `8f223e56d42f1a23bda989cdfa1419c2e261843627fd1e4db8f007dc1c8d90a5`), `configs/expanded.json`, the expanded protocol, public/synthetic dataset documentation, runtime source fingerprints and primary compiled NPZ files. No runtime code was changed and no GPU experiment was rerun.

## Integrity and scope

- The recorded configuration exactly matches the declared configuration: 20 additional datasets, two policies each, seeds 7/17/27, split/generator seed 2026, 60 epochs and MPS. The matrix contains exactly 120 unique runs, 40 task audits and 240 method/policy aggregates with three observations each.
- All recorded online/teacher parameters, context/target batches, losses and online gradients use `mps:0`; teacher gradients and CPU fallback are disabled. All runs record 60 epochs, label-free encoder training and no test use in training/model selection. Runtime source/config hashes match the recorded inventory.
- The expansion is 10 real public source datasets plus 10 distinct synthetic generator families: 14 classification datasets and six forecasting datasets. With the original three datasets/18 runs preserved separately, the project totals 23 datasets and 138 runs.
- All ten public archive hashes were independently compared with the downloaded cache. Public loaded shapes, sample indices/count metadata and split counts match the earlier live loader audit. The twenty primary NPZ exports have matching dimensions and label presence; every supplied group is disjoint across train, validation and test. All six synthetic series retain 100 entities and 1200/400/400 rows.
- All recorded task audits pass; warnings remain visible. Every method uses the declared train/validation/test roles. Recomputing probe selection from every candidate history reproduces the chosen parameters and validation scores within floating-point rounding (largest discrepancy 1.11e-16). Context-only inference does not consume hidden targets; full-input classification references remain separately labelled.
- The complete binary/checkpoint replay is a separate validator run; this review checked its recorded round-trip outcomes and inspected the twenty primary exports, rather than claiming a second replay of all 120 checkpoints.

## Independently recomputed primary outcomes

Using unrounded three-seed means, JEPA improves raw-context linear on **9/20**, the untrained encoder on **9/20**, and ExtraTrees on **3/20** primary policies. It exceeds the best measured context-only comparator on **2/20**. Across all forty policies it exceeds raw-context linear on **19/40**. These are descriptive counts on one split, not significance tests or evidence of universal superiority. Full-input references are excluded from the matched-comparator counts.

Classification values are macro-F1 (higher is better); forecast values are original-unit RMSE (lower is better). RMSE is not averaged across systems.

| Dataset / primary policy | Metric | JEPA mean ± sample SD | Raw linear | ExtraTrees |
|---|---|---:|---:|---:|
| `breast_cancer` / `mean_to_worst` | macro_f1 | 0.9207 ± 0.0006 | 0.9188 | 0.8995 |
| `ionosphere` / `alternating_pairs` | macro_f1 | 0.7696 ± 0.0463 | 0.7464 | 0.8956 |
| `sonar` / `alternating` | macro_f1 | 0.5968 ± 0.0747 | 0.7083 | 0.8091 |
| `semeion` / `center` | macro_f1 | 0.8485 ± 0.0064 | 0.9087 | 0.9079 |
| `letter` / `alternating` | macro_f1 | 0.5223 ± 0.0022 | 0.4983 | 0.7000 |
| `pendigits` / `alternating_pairs` | macro_f1 | 0.8976 ± 0.0053 | 0.8329 | 0.9105 |
| `satimage` / `neighbor_to_center` | macro_f1 | 0.7991 ± 0.0198 | 0.8062 | 0.8678 |
| `har` / `time_to_frequency` | macro_f1 | 0.7961 ± 0.0144 | 0.9291 | 0.9159 |
| `dry_bean` / `alternating` | macro_f1 | 0.9138 ± 0.0022 | 0.9140 | 0.9156 |
| `isolet` / `alternating` | macro_f1 | 0.7476 ± 0.0055 | 0.9286 | 0.9318 |
| `synthetic_lorenz` / `past24_future8` | rmse | 0.8504 ± 0.0723 | 2.3842 | 0.5890 |
| `synthetic_mackey_glass` / `past24_future8` | rmse | 0.0893 ± 0.0046 | 0.0783 | 0.0866 |
| `synthetic_switching_var` / `past24_future8` | rmse | 0.1831 ± 0.0012 | 0.1849 | 0.1789 |
| `synthetic_chirp_seasonal` / `past24_future8` | rmse | 0.2106 ± 0.0151 | 0.1619 | 0.1968 |
| `synthetic_coupled_oscillators` / `past24_future8` | rmse | 0.1384 ± 0.0069 | 0.0357 | 0.1306 |
| `synthetic_nonlinear_ar` / `past24_future8` | rmse | 0.1152 ± 0.0075 | 0.0980 | 0.1206 |
| `synthetic_nonlinear_multiview` / `view_a_to_view_b` | macro_f1 | 0.8804 ± 0.0108 | 0.8487 | 0.9192 |
| `synthetic_hierarchical_multiclass` / `view_a_to_view_b` | macro_f1 | 0.9631 ± 0.0053 | 0.9499 | 0.9614 |
| `synthetic_sparse_interactions` / `view_a_to_view_b` | macro_f1 | 0.5779 ± 0.0184 | 0.6647 | 0.6572 |
| `synthetic_manifold_nuisance` / `view_a_to_view_b` | macro_f1 | 0.7650 ± 0.0280 | 0.7365 | 0.9578 |

## Diagnostics and interpretation

**83/120 runs trigger the final low-rank heuristic.** All 83 flags are attributable to covariance effective rank below two; none has mean coordinate standard deviation below 0.01. The variance regularizer therefore did not ensure diverse embeddings. These are heuristic rank warnings, not numerical execution failures. The compiler warnings are {'constant_context': 2, 'trivial_target_copy': 1}.

JEPA has useful but uneven outcomes: Lorenz RMSE improves substantially over raw linear (0.8504 versus 2.3842), while ExtraTrees is still better (0.5890). HAR and ISOLET remain materially behind strong raw-context baselines. Breast Cancer and hierarchical synthetic classification are the only primary cases above every measured matched comparator by mean; their small margins do not establish statistical superiority.

## Required limitations in the release

- Preserve the distinction between the ten public datasets and ten authored synthetic mechanisms. Do not call all twenty real observations or claim all twenty are larger than the original datasets.
- Public exact-feature deduplication, bounded label-independent sampling and pooling of original source partitions are explicitly recorded; results are new pilot partitions, not official UCI benchmark reproductions.
- HAR uses explicit subjects but source-provided engineered/normalized features; original preprocessing cannot be fully audited. PenDigits retains conservative source-comment groups whose writer interpretation is inferred. Semeion/ISOLET and Satimage lack explicit writer/speaker/geographic identifiers, so corresponding independence claims remain unsupported.
- Three model seeds share one fixed data split; standard deviations characterize optimization variability, not population uncertainty. No cross-policy ranking should ignore different context/horizon budgets.
- Report all low-rank warnings and unfavorable outcomes. No label leakage or protocol mismatch was found, but the results do not establish broad representation superiority, clinical utility, physical simulator validity or real-world deployment value.
