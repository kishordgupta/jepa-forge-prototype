# Expanded 23-dataset evaluation protocol

The user requested 20 more complex datasets after the original three-dataset pilot. The original Wine, Digits and synthetic sensor measurements remain a separate, unchanged result set: 18 GPU training runs. The expansion declares **10 additional public datasets and 10 distinct synthetic generator families**, with two task policies each and three model seeds, for **120 additional GPU runs** and **138 total runs**.

## Scope fixed before expanded test evaluation

- Configuration: `configs/expanded.json`; split and generator seed 2026; model seeds 7, 17 and 27; 60 epochs; MPS GPU; batch size 128; latent dimension 32; Adam learning rate 0.001; EMA 0.99; variance penalty weight 1.0.
- Both registered context/target policies are evaluated. The first declared policy is primary before seeing expanded test results. Policies are never selected using the test set.
- All original matched comparison methods and probe grids from [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) are retained. Classification reports macro-F1/accuracy; forecasting reports original-unit RMSE/MAE. Forecasting channel counts and horizons now vary by dataset.
- Public datasets are deterministically sampled without labels to at most 3,000 observations. Available subject/source groups stay within a single partition. Synthetic time series use independent entities with overlapping windows confined to their own entity's partition.
- One fixed approximately 60/20/20 split is shared across model seeds and methods. Sample standard deviations describe model-seed variation only; they are not confidence intervals across independently sampled datasets or splits.
- One-epoch GPU execution checks are saved separately under `artifacts/expanded_smoke`. They verify device/shape compatibility and do not select training budgets, masks or hyperparameters.

## Why these are additional and more challenging

The public suite spans medical measurements, radar returns, sonar spectra, handwritten symbols, letters, pen trajectories, multispectral pixels, human activity, bean morphology and spoken-letter acoustic features. Some datasets have many more observations, features or classes; others stress noisy high-dimensional classification with few observations. The suite does not assert that every dataset is harder under every metric.

The ten synthetic families use different mechanisms, including chaotic continuous dynamics, delayed feedback, regime switching, frequency variation, coupled oscillators, nonlinear autoregression, multiview nonlinear observations, hierarchical classes, sparse feature interactions and manifold nuisance variables. They are distinct generator families, not ten random seeds of the original sensor generator. They remain controlled simulations, not evidence of performance on real scientific systems.

See [PUBLIC_DATASETS_EXPANDED.md](PUBLIC_DATASETS_EXPANDED.md) for exact upstream sources, licenses, row selection, feature semantics and group limitations. See [SYNTHETIC_DATASETS_EXPANDED.md](SYNTHETIC_DATASETS_EXPANDED.md) for equations, parameters and generated dimensions.

## Integrity and limits

Raw inputs are downloaded from pinned public sources or deterministically generated. Labels remain separate and are used only for evaluation probes. Compiler statistics and estimator preprocessing are fitted on training rows. Known identifiers are split metadata, never model inputs. All task exports and model checkpoints must reload successfully; actual parameter, tensor, loss and gradient devices must be MPS for the GPU claim. Numerical failures are retained as failures rather than replaced with favorable subsets.

The original runtime is archived at `results/original_runtime.zip`, with its entries checked against the SHA-256 inventory recorded in `results/benchmark.json`. Expanded runs record the updated runtime, including the new adapters and a faster exact-duplicate-column audit. The optimization retains train-only comparisons and signed-zero equality; no loss/model/probe selection rule is changed for the expansion.

The benchmark does not establish state-of-the-art results, clinical utility, speaker/writer independence when identifiers are unavailable, population-level significance, or superiority of representation learning across all data. Dataset-specific metrics are compared within each policy; RMSE values from different synthetic systems are not averaged into a single unitless score. Public subsets and one fixed split limit generalization.

## Reproduce

```bash
python -m pip install '.[dev,report]'
pytest -q --junitxml=artifacts/pytest.xml
python scripts/verify_environment.py
python scripts/smoke_expanded.py
python scripts/run_experiments.py --config configs/expanded.json --output artifacts/expanded
python scripts/validate_expanded.py
python scripts/plot_expanded.py
python scripts/build_expanded_report.py
python scripts/package_artifacts.py
```

Neural training and encoding use GPU; scikit-learn probes, baselines and data processing use CPU. The experiment configs require an available MPS device and refuse silent CPU fallback. CUDA is supported by the model on an appropriate host, but the recorded measurements here use the local Apple M3 Max GPU.
