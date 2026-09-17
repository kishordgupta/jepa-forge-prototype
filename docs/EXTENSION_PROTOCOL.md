# Repeated-split extension protocol

Declared before opening the new test partitions. The executable configuration is `configs/extension.json`. The historical 273-run study remains immutable and is reported separately.

## Scope and partitions

The extension repeats all 23 prior tasks and adds activity classification and forecasting for each of two public raw sensor corpora, MHEALTH and PAMAP2: **27 tasks on 25 underlying datasets**. Data/generation seed is 2026. Three label-independent outer split seeds are 4026, 5026, and 6026. Model initialization is fixed to seed 7, so the new spreads measure split variation conditional on one initialization, not both sources of uncertainty. Repeated partitions can overlap; their standard deviations are descriptive, not independent confidence intervals.

Public raw accelerometers retain nine channels, original source member hashes, sample intervals, and public pseudonymous subject IDs. MHEALTH contributes 1,600 windows from 10 subjects; PAMAP2 contributes 1,280 windows from eight protocol subjects. PAMAP2 subject 109 has only a limited activity recording and is excluded before sampling. The same sampled windows supply both tasks, so these are two outcomes on one corpus, not independent datasets. Windows contain 64 readings at an effective 50 Hz. PAMAP2 retains every second reading without interpolation or antialias filtering. All 128 original samples must be finite and contiguous. MHEALTH uses consecutive file rows; its time bounds are row-index/50 source-relative estimates because the files have no timestamp column. Only PAMAP2 timestamp gaps can be checked directly. No window crosses a file, timestamp gap, null activity, or activity transition. This source-label filtering applies to forecasting too; the forecast population is segmented activities. Each subject contributes a label-independent random sample of 160 valid nonoverlapping windows.

Splits hold out whole subjects or supplied synthetic/source groups. The floor-based 60/20/20 algorithm yields 6/2/2 subjects for MHEALTH and 4/2/2 for PAMAP2; exact groups and rows are published in locks. Existing tasks retain their documented sampling and missing-group limitations. Newly shuffled partitions of previously inspected data are not external confirmation. All preprocessing and feature rankings fit training rows only.

## Models and feature selection

Each method independently chooses among the same schema masks using its own validation score. The existing four-mask families are retained, with three masks for Breast Cancer. Context counts match within a task. Forecasts share the final-quarter target and only observe past coordinates. The methods are:

- The existing JEPA-style pilot, trained label-free for 30 epochs per mask, followed by a training-fitted logistic/ridge probe.
- A supervised encoder with exactly the pilot's online architecture and a linear prediction head, trained end to end with cross-entropy or standardized-target MSE.
- Official TabM package 0.0.3: two blocks of width 128, eight ensemble members, dropout 0.1, numeric features without learned numeric embeddings. Member losses are averaged during training; class probabilities are averaged at inference. This is a bounded TabM configuration, not a reproduction of its tuned benchmark results.
- Raw logistic/ridge regression, Extra Trees, and CatBoost. CatBoost is classification-only in this protocol. Extra Trees uses 100 trees, minimum leaf size 2, and depths 8/16. CatBoost uses 100 iterations, learning rate 0.1, depths 4/6, and two CPU threads. Both choose depth on validation independently of JEPA.
- Independent feature-selection controls: training-only ANOVA F rankings for classification or mean squared feature/target correlation for forecasting; and training-only Extra Trees feature importance. Each chooses exactly the JEPA observation count from all legal training features. A separately tuned linear or Extra Trees model evaluates each selected subset. These selectors can choose individual sensor coordinates; schema-mask candidates retain whole timesteps. Their extra ranking fits and search domains are explicitly different.

Neural methods use Adam at 0.001, batch size 128, gradient clipping at 10, and MPS with no fallback. JEPA uses dimension 32, EMA 0.99, and the existing variance penalty. All neural models use 30 epochs per candidate and final-epoch checkpoints. Equal epochs do not imply equal FLOPs across different architectures. CatBoost, tree models, probes, preprocessing and ranking use CPU. Timings and trainable parameter counts are recorded; no wall-clock or pretraining-cost parity is claimed across model families.

## Matched policy compute control

For a dataset with K schema masks, search trains K independent JEPA runs for E=30 epochs and fits five probe strengths per final checkpoint. The fixed-mask policy observes the first declared mask throughout, trains for K*E epochs, and considers checkpoints E, 2E, ..., K*E with the same five probe strengths. It selects its best checkpoint/probe by validation. Both policies therefore spend the same total optimizer updates and K*5 probe fits, have the same architecture, training rows, observation count and forecast horizon, and receive the same number of validation choices. Different random batch histories and elapsed time are measured, not assumed identical. A separate short fixed-mask result reuses the first 30-epoch candidate; it is clearly labeled as cheaper. The extended fixed trajectory is not stopped early even when validation favors an earlier checkpoint.

## Selection, test gate and reporting

Each selector receives physically copied train/validation rows and no test-array interface. All 81 task/split selections must be locked, including baseline masks, checkpoints, probes and hashes, before any extension test evaluation. The runner freezes executed source hashes and package versions at start. A failed method stops the study. Resume checks reject changed source/config/data. Final inference verifies artifact hashes and replays every chosen validation metric before scoring test predictions. Models are not refitted on train plus validation. Raw scores, class/forecast metrics, masks and prediction hashes are retained.

Report mean and sample standard deviation over the three splits and per-split results. Macro F1 follows the standard scikit-learn observed/predicted-class union convention; rare classes may be absent from a partition. Forecast RMSE averages original-unit errors over all target coordinates; do not average RMSE across datasets with different units. Dataset-level win counts use unrounded split means and are descriptive. No best-test mask, best-test seed, or post-test hyperparameter selection is permitted.

## Official-model transfer study

A separate, predeclared vision study uses official Meta I-JEPA ViT-H/14 weights (ImageNet-1K) and official V-JEPA ViT-L/16 weights (VideoMix2M). It uses image/video inputs and identical visible pixels within each task. External pretraining, model sizes and interpolation differ from the small pilot, so this is a frozen-feature transfer comparison, not a compute-matched architecture experiment. `docs/VISION_TRANSFER_PROTOCOL.md` records that study's exact subsets, masks and settings before evaluation.

Sources: [MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset), [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring), [TabM](https://github.com/yandex-research/tabm), [CatBoost](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier).
