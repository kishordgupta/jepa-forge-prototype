# Independent review of expanded model quality

**The expanded experiment is complete, but it does not establish a general
advantage for the JEPA-style learner.** It demonstrates a functioning dataset
compiler and GPU experiment pipeline, with useful gains on selected tasks and a
substantial representation-quality limitation: learned variance often
concentrates in fewer than two effective dimensions despite 32 output
coordinates. Strong execution evidence should be reported separately from
learning quality.

This review covers the **120 completed expanded runs**: 20 additional datasets,
two predetermined policies each, and model seeds 7, 17 and 27. It does not include
the original three-dataset pilot. All comparisons below use the mean across
those three model seeds within one dataset-policy case. Classification uses
test macro-F1, where higher is better; forecasting uses test RMSE, where lower
is better. The test results are retrospective review evidence, not a basis for
changing these runs' hyperparameters or selecting a policy.

Sources are [`expanded_benchmark.json.gz`](../results/expanded_benchmark.json.gz), its saved
embeddings and compiled splits, and
[`expanded_model_review_evidence.json`](../results/expanded_model_review_evidence.json).
The completed results file's SHA-256 is
`8f223e56d42f1a23bda989cdfa1419c2e261843627fd1e4db8f007dc1c8d90a5`.
Additional checks used only saved artifacts and CPU calculations. No new GPU
experiments, training changes, or hyperparameter searches were performed.

## Comparison with context-only baselines

| Evaluation category | Dataset-policy cases | JEPA better than random encoder | Better than raw linear | Better than ExtraTrees |
|---|---:|---:|---:|---:|
| Public classification | 20 | 7 | 8 | 3 |
| Synthetic classification | 8 | 4 | 5 | 1 |
| Synthetic forecasting | 12 | 8 | 6 | 2 |
| Total, counting each case once | 40 | 19 | 19 | 6 |

These are strict directional comparisons of case means, not significance tests,
independent replications, or a pooled metric across incompatible scales. Two
policies on the same dataset are correlated, and three model seeds share the
same data split. A tiny numerical lead counts in this table and should not be
described as a robust improvement. The random encoder is particularly useful:
it has the same architecture, context inputs, latent dimension, initialization
seed and supervised probe procedure as the trained encoder.

Selected results show why broad superiority would overstate the evidence:

- **Semeion** improves over its random encoder by 0.1266 macro-F1 for `center`
  and 0.1057 for `right_half`, but remains below the corresponding raw linear
  probes by 0.0602 and 0.0348. Learning helps this compressed architecture; it
  does not outperform using the available raw context.
- **Sonar** loses 0.1068 and 0.0881 macro-F1 to the random encoder for
  `alternating` and `first_half`. These are substantial degradations for the
  declared pilot, although the dataset and held-out sample are small.
- **Lorenz** reduces random-encoder RMSE by 54.9% and 65.6% across its two
  horizons. JEPA's mean RMSE is 0.8504 and 1.2969, compared with 1.8849 and 3.7731
  for the random encoder. ExtraTrees still performs better at 0.5890 and 0.7815.
- **Coupled oscillators** shows the opposite behavior: JEPA's RMSE is 0.1384
  and 0.1472, worse than the random encoder's 0.0785 and 0.0939 and the raw linear
  probe's 0.0357 and 0.0641. This is a concrete failure case for the current
  representation objective and compression choice.
- **Synthetic sparse interactions** is strongly policy-dependent. The
  `view_a_to_view_b` JEPA probe scores 0.5779 macro-F1 versus 0.6001 random and
  0.6647 raw linear; `alternating` reaches 0.7566 versus 0.7099 random, but
  remains well below raw linear at 0.9350. The policies reveal different
  information, so their difference cannot be attributed solely to task quality.

The full-information linear reference has additional input columns and is not
an information-matched competitor. The main table above excludes that reference.
All reported forecasts are supervised probes of frozen context embeddings, not
direct physical-variable predictions by the latent JEPA predictor.

## Representation concentration is a material limitation

Effective rank here is `exp(entropy(p))`, with `p` the normalized nonnegative
covariance eigenvalues. It measures how variance is distributed across latent
directions. It is not an algebraic matrix rank and does not prove that small
residual directions are useless. A rank below two is a disclosed heuristic
warning, not a universal failure criterion; some generated systems have low
intrinsic dimensionality by design.

| Category | Runs | Recorded rank < 2 | Full-training rank < 2 | Test rank < 2 | Median full-training rank |
|---|---:|---:|---:|---:|---:|
| Public classification | 60 | 48 | 48 | 49 | 1.181 |
| Synthetic classification | 24 | 16 | 16 | 16 | 1.713 |
| Synthetic forecasting | 36 | 19 | 19 | 18 | 1.932 |
| Total | 120 | 83 | 83 | 83 | 1.555 |

The training monitor uses up to the first 512 training rows. This review
independently recomputed ranks using **every training row and every test row**
from saved context embeddings; the same aggregate count of 83 warnings persists
in both full partitions. The issue is therefore not explained by the small
monitoring subset. Across the 120 runs, median recorded effective rank falls
from 6.153 at initialization to 1.505 after training. Public classification falls
from 6.242 to 1.147.

No final run has mean coordinate standard deviation below 0.01. These results
do **not** indicate all-constant representations. They indicate severe variance
concentration in many otherwise varying embeddings. The current variance
regularizer acts on each coordinate separately and cannot prevent correlated
coordinates: if `z_j = a_j*s`, every coordinate can have substantial standard
deviation while covariance has only one direction. That mathematical limitation
is consistent with the measurements, although this observational review does not
isolate the causal contribution of each loss component.

The matched-target validation MSE is lower than its one deterministic shuffled
comparison in 117 of 120 runs. All 60 public runs pass that pairing check, even
though 48 have rank below two. **Passing the pairing diagnostic does not rule
out dimensional collapse or demonstrate downstream superiority.** The three
non-improving shuffle cases are all synthetic sparse-interaction runs:
`view_a_to_view_b`, seed 7, and `alternating`, seeds 17 and 27. Their shuffled to
matched MSE ratios are 0.9901, 0.9807 and 0.9931. A single shuffle is a heuristic,
not a significance test.

## Why breast-cancer policies give identical test outcomes

The two policies use the same ten mean-measurement context columns, with
different target blocks: worst-value measurements versus standard errors. The
same seed therefore produces **exactly identical untrained context embeddings**.
Raw-context baseline equality is expected for the same reason.

The learned representations are different, but close:

| Seed | Relative embedding difference | Flattened correlation | Different reconstructed test predictions | Selected logistic C, both policies |
|---|---:|---:|---:|---:|
| 7 | 1.61% | 0.999872 | 0 / 114 | 0.1 |
| 17 | 2.06% | 0.999789 | 0 / 114 | 100 |
| 27 | 2.53% | 0.999706 | 0 / 114 | 10 |

Relative difference is the Frobenius norm of the embedding difference divided
by the first policy's embedding norm. Predictions were reproduced on CPU from
the saved embeddings using each run's already-selected regularization and the
original training rows; no parameters were searched again. Both policies give
test macro-F1 values of approximately 0.9200, 0.9211 and 0.9211 for the three
seeds. Their identical test scores are explained by identical classifications,
not copied result files or identical trained embeddings.

There is also a real concentration warning: their effective ranks are
1.0445–1.0933, down from 3.6782–4.2813 initially, and the first principal component
accounts for **98.38%–99.36% of training embedding variance**. Context mean
standard deviations remain 2.49–2.64. The target teachers also remain varying:
their validation mean standard deviations are 0.0379–0.1262, with effective ranks
1.626–3.467. Matched-target MSE is better than shuffled-target MSE by ratios of
1.69–6.79, depending on the target policy and seed.

The supported interpretation is **near-identical useful decision boundaries
with severe latent variance concentration**, rather than total constant collapse
or evidence that the target tasks are equivalent. Coordinate variance and a
positive pairing check are insufficient to dismiss the rank warning. Absolute
latent losses across these target policies also have different target scales
and should not be used as an unqualified policy-quality ranking.

## Next experiments, to be specified before new confirmatory evaluation

1. **Separate the learning signals.** Compare the current objective against
   variance-only training, latent-prediction-only training, and an explicitly
   decorrelated or whitened representation objective. Keep architecture,
   context, optimizer budget and probe selection matched. This would test
   whether downstream gains depend on cross-view prediction or mostly on the
   variance adaptation.
2. **Audit all three representations.** Log online context, target teacher and
   predictor variance, effective rank, leading eigenvalue fraction and norm
   during training. Evaluate target normalization or scale-aware losses as
   declared ablations; the present large context/teacher scale differences make
   raw MSE alone hard to interpret.
3. **Strengthen correspondence controls.** Use several predetermined validation
   permutations, with a distribution of shuffled losses, and a negative-control
   training task with target pairing randomized independently of labels. These
   controls should supplement, not replace, downstream baselines.
4. **Measure the compression tradeoff.** Match PCA and other compressed baselines
   to the JEPA feature budget where feasible; the current PCA baseline uses at
   most eight components while JEPA uses 32. Include the existing strong raw
   linear and ExtraTrees comparisons, and explicitly report context-feature
   count, latent dimension and trainable parameters.
5. **Confirm across independent data splits.** Three initialization seeds on one
   split quantify only part of uncertainty. Predeclare new row or entity splits,
   ensure the public dataset's available grouping supports the claimed scope,
   and report paired variation across splits. Preserve the present results as
   a completed exploratory benchmark; select any revised objective using
   training/validation data and reserve fresh confirmatory evaluation.

No algorithm changes are required to make the existing evidence honest: retain
the matched baselines, rank diagnostics, failure cases, exact task definitions,
and execution records. The next research milestone is improved representation
quality under those controls, rather than a claim that a larger dataset count
alone demonstrates general scientific utility.
