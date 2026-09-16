# Experiment protocol

This protocol specifies the pilot before interpreting results. A completed experiment requires per-run artifacts from the actual implementation. No numerical claim in a proposal or an earlier feasibility estimate is reused as a result.

## Fixed controls

| Control | Setting |
| --- | --- |
| Split seed | `2026`; save exact record and group memberships |
| Model seeds | `7`, `17`, `27`; keep data and splits identical across these runs |
| Synthetic seed | `2026`, passed to the dataset loader from the split-seed setting; unchanged across model seeds |
| Candidate tasks | Wine `alternating`, `first_half`; Digits `center`, `right_half`; sensors `past16_future8`, `past12_future12` |
| Training budget | Default 60 epochs per configuration; record the actual budget and reason for any deviation |
| Neural compute | Actual GPU when available; record `mps`, `cuda`, or `cpu` explicitly |
| Classical compute | Scikit-learn preprocessing, PCA, and downstream estimators on CPU |
| Split strategy | Label-independent random record split for Wine/Digits; disjoint source-group split for sensors; nominal 60/20/20 proportions |
| Primary task | First declared policy, chosen a priori; retain both policies without selecting an automatic winner |

The model configuration is latent dimension **32**, batch size **128**, Adam learning rate **0.001**, EMA decay **0.99**, variance weight **1.0**, and gradient-norm clipping at **10**. The tabular MLP uses two hidden layers of width 64. Digits use a two-dimensional convolutional encoder with value and mask channels; sensors use a one-dimensional convolutional encoder preserving time-major input order with value and mask channels. Exact parameter counts are saved per run.

The prediction term is MSE between the predictor output and the unnormalized, stop-gradient target latent. The regularizer is `mean(relu(1 - sqrt(var(context_latents) + 1e-4)))`, with population variance over each batch. The target branch receives target-only values and their mask. The encoder checkpoint is the **fixed final epoch**, not the best validation epoch; validation losses are logged for inspection only.

Write split proportions and exact counts to the exported manifest. Keep the declared settings in the run configuration and do not silently change them after seeing test metrics.

## Training and selection boundaries

1. Construct and audit the dataset and each task. Persist split identifiers. Confirm that no record crosses partitions and that no sensor group crosses partitions.
2. Fit normalization on training records, then apply it unchanged to validation and test. Fit PCA on training context inputs only. Compiler means/scales and split arrays are saved in local NPZ exports. Probe/PCA estimators are reproduced by refitting the recorded grids, inputs and seeds; these estimators are not serialized.
3. Train each online encoder and predictor on training context/target pairs. Compute the target representation with a frozen EMA target encoder. Optimize latent prediction MSE plus the declared variance regularizer. Do not feed class labels to this stage.
4. Freeze the encoder for downstream evaluation. Fit supervised probes using training labels or training future targets. Tune probe hyperparameters on validation data only; apply the same declared tuning budget to comparable methods. Keep the selected estimator fitted on training data; do not refit it on training plus validation.
5. Choose probe hyperparameters by validation **macro F1** for classification and validation **RMSE** for regression. Record every candidate and its validation metrics, not just the winner. The first candidate wins an exact tie. Neural checkpoints use the fixed final epoch; task policies are retained separately and are not chosen by performance.
6. Freeze the selected procedure before evaluating the test partition. Test observations must not be used for gradient updates, preprocessing, early stopping, feature selection, policy selection, or probe tuning. Report test results even when a baseline wins.

A validation set used repeatedly for tuning is not an independent estimate of performance. Three model seeds characterize optimization variability on one fixed split; they do not supply three independent datasets or establish population-level statistical significance.

## Probe grids

| Probe | Declared candidates and fitting |
| --- | --- |
| Logistic regression | `C` in `[0.01, 0.1, 1, 10, 100]`, `max_iter=2000`; a train-fitted `StandardScaler` precedes each fit |
| Ridge regression | `alpha` in `[0.01, 0.1, 1, 10, 100]`; a train-fitted `StandardScaler` precedes each fit |
| PCA then linear probe | Train-fitted scaler, PCA with `min(8, input_dimensions, n_train - 1)` components and full SVD, then the same logistic/ridge grid |
| Extra Trees | Classifier or regressor with 100 trees; `max_depth` in `[8, None]`; model seed is the declared run seed; `n_jobs=2` |

All preprocessing stays inside training-fitted pipelines. Extra Trees use the same compiled context as the other primary comparators. Classification reports both accuracy and macro F1 even though selection uses macro F1. The linear predictability diagnostic uses `Ridge(alpha=10)` on training context and reports standardized target MSE on validation versus a training-mean predictor; it does not select the primary policy.

## Context-matched comparisons

All primary comparisons must use the **same observed context** within a task policy. Withheld pixels/features or future observations must not enter any of these representations.

| Method | Training and information available | Evaluation |
| --- | --- | --- |
| Raw-context linear model | Linear classifier or regressor fitted on the visible training context | Accuracy and macro F1 for Wine/Digits; MAE and RMSE for sensor targets |
| PCA plus linear probe | PCA fitted on training context, followed by a supervised linear probe | Same partition, target, and metrics as the learned representation |
| Random encoder plus linear probe | Same modality encoder architecture and seed, frozen before representation training | Tests how much the architecture alone contributes |
| Nonlinear supervised baseline | Extra Trees trained directly on context and training labels/targets; validation-only depth tuning | A stronger context-only comparator, with its estimator and budget reported |
| Learned encoder plus linear probe | GPU representation training without class labels; frozen context embedding; supervised probe | Primary representation-learning result |
| Persistence, sensors only | Repeat the last observed value of each channel across the future horizon | MAE and RMSE on exactly the same future steps |

For classification, also provide a **full-information reference** trained with all original input features or pixels. Label it separately in tables and plots because it has more information than context-only methods; it is neither a matched baseline nor a theoretical ceiling. A classifier may use training labels, but no method may use validation or test labels as fitted training examples under this protocol.

## Metrics and interpretation

Report classification accuracy and macro F1 as fractions or percentages with an explicit unit. Preserve per-seed scores and summarize the arithmetic mean and sample standard deviation across the three fixed model seeds. Do not duplicate a deterministic baseline score to suggest additional independent experiments.

For forecasting, compute MAE and RMSE over the same held-out samples, future time steps, and two channels. The probes fit original-unit future targets, and persistence uses the last original-unit context observation; no target inverse transform is required. The implementation uses uniform averaging over samples and all scalar target coordinates, followed by a square root for RMSE. Report per-policy results; an 8-step horizon and a 12-step horizon are different tasks.

Compare methods **within each policy**. `center` and `right_half` can expose different pixel counts, and the sensor policies have different history lengths and horizons. Cross-policy rankings must identify these differences. If the software proposes an automatic policy winner, describe the validation-only criterion and treat it as a pilot recommendation rather than proof of a universally better task design.

Record training loss and a representation-collapse diagnostic alongside predictive metrics. The pilot computes coordinate standard deviations and covariance effective rank on at most the first 512 training context embeddings, using CPU float64 covariance arithmetic. It also records effective rank divided by `min(latent_dimension, n_samples - 1)`. It flags mean coordinate standard deviation below `0.01` or effective rank below `2`. This is a heuristic: a low latent loss or the absence of that flag is not sufficient evidence of useful learning.

The final checkpoint also compares validation latent-prediction MSE for matched targets with MSE after one seeded target shuffle. A random cycle ensures every target changes position. This diagnostic is computed after training, does not select the model or policy, and supplies neither a significance test nor a robust estimate over many permutations.

## Required run artifacts

For each dataset, policy, seed, and method, preserve configuration, dataset/split identifiers, validation metric, final test metric, actual compute device, wall-clock time, and failure status. Save package versions and a source revision identifier when available. For a GPU claim, retain evidence that model parameters and training tensors actually used that GPU; a detected device or isolated smoke check alone is insufficient.

Keep any exploratory changes separate from the final declared run. If a defect is found after test evaluation, document the defect and rerun all affected comparisons consistently. The final report should explain corrections instead of silently replacing unfavorable results.

This pilot does not measure real-world deployment, human utility, developer productivity, graph/audio/event performance, state-of-the-art performance, or general superiority across datasets. Larger external benchmarks and independent replications remain future work.
