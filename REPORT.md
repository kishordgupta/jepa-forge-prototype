# JEPA-FORGE prototype: implementation and experiment report

Generated from measured run artifacts on 2026-09-16.

This working prototype implements a typed task compiler, train-only normalization, structural/leakage diagnostics, numeric exports with checksums, a context-only inference API, and small JEPA-style GPU pilots. It does not implement the complete proposed ecosystem or reproduce the proposal's earlier result table.

## Outcome

- wine / alternating: JEPA macro_f1 0.914 +/- 0.000; raw-context linear 0.941 +/- 0.000.
- digits / center: JEPA macro_f1 0.886 +/- 0.008; raw-context linear 0.875 +/- 0.000.
- synthetic_sensors / past16_future8: JEPA rmse 0.321 +/- 0.021; raw-context linear 0.259 +/- 0.000.

JEPA improved primary Digits macro-F1 over raw-context linear, but remained below ExtraTrees. It underperformed the raw-context linear model on Wine and sensor forecasting. Seven of 18 runs triggered a low-rank embedding warning (all six Wine runs and one 12-step sensor run). These results support pipeline feasibility, not general JEPA superiority.

## Implementation and protocol

18 neural runs cover three datasets and six policies, with model seeds 7, 17 and 27, a fixed split/generator seed of 2026, and 60 epochs per run. The first declared policy for each dataset was designated primary before evaluating the test set. Sample standard deviations measure model-seed variation on one fixed split, not population uncertainty.

GPU: Apple M3 Max / PyTorch MPS. Training and encoding use the GPU; scikit-learn baselines, probes and data processing use CPU. No paid cloud GPU was used.

All primary inference receives context columns only. Hidden targets are used for training, forecast supervision, diagnostics and scoring; they never enter context-only inference. Full raw classification is a separate full-information reference.

## Test results

Mean +/- sample SD over the three model seeds. Compare methods within each policy. Macro-F1: higher is better; RMSE: lower is better, in original synthetic signal units.

| Dataset | Policy | Method | Metric | Mean +/- SD |
|---|---|---|---|---|
| wine | alternating | Raw context + linear | macro_f1 | 0.941 +/- 0.000 |
| wine | alternating | PCA + linear | macro_f1 | 0.941 +/- 0.000 |
| wine | alternating | Raw context + ExtraTrees | macro_f1 | 0.980 +/- 0.018 |
| wine | alternating | Random encoder + linear | macro_f1 | 0.962 +/- 0.044 |
| wine | alternating | JEPA context + linear | macro_f1 | 0.914 +/- 0.000 |
| wine | alternating | Full raw reference* | macro_f1 | 0.970 +/- 0.000 |
| wine | first_half | Raw context + linear | macro_f1 | 0.892 +/- 0.000 |
| wine | first_half | PCA + linear | macro_f1 | 0.892 +/- 0.000 |
| wine | first_half | Raw context + ExtraTrees | macro_f1 | 0.990 +/- 0.018 |
| wine | first_half | Random encoder + linear | macro_f1 | 0.950 +/- 0.017 |
| wine | first_half | JEPA context + linear | macro_f1 | 0.907 +/- 0.032 |
| wine | first_half | Full raw reference* | macro_f1 | 0.970 +/- 0.000 |
| digits | center | Raw context + linear | macro_f1 | 0.875 +/- 0.000 |
| digits | center | PCA + linear | macro_f1 | 0.741 +/- 0.000 |
| digits | center | Raw context + ExtraTrees | macro_f1 | 0.913 +/- 0.008 |
| digits | center | Random encoder + linear | macro_f1 | 0.834 +/- 0.014 |
| digits | center | JEPA context + linear | macro_f1 | 0.886 +/- 0.008 |
| digits | center | Full raw reference* | macro_f1 | 0.947 +/- 0.000 |
| digits | right_half | Raw context + linear | macro_f1 | 0.848 +/- 0.000 |
| digits | right_half | PCA + linear | macro_f1 | 0.760 +/- 0.000 |
| digits | right_half | Raw context + ExtraTrees | macro_f1 | 0.921 +/- 0.003 |
| digits | right_half | Random encoder + linear | macro_f1 | 0.846 +/- 0.020 |
| digits | right_half | JEPA context + linear | macro_f1 | 0.870 +/- 0.004 |
| digits | right_half | Full raw reference* | macro_f1 | 0.947 +/- 0.000 |
| synthetic_sensors | past16_future8 | Raw context + linear | rmse | 0.259 +/- 0.000 |
| synthetic_sensors | past16_future8 | PCA + linear | rmse | 0.274 +/- 0.000 |
| synthetic_sensors | past16_future8 | Raw context + ExtraTrees | rmse | 0.358 +/- 0.004 |
| synthetic_sensors | past16_future8 | Random encoder + linear | rmse | 0.322 +/- 0.018 |
| synthetic_sensors | past16_future8 | JEPA context + linear | rmse | 0.321 +/- 0.021 |
| synthetic_sensors | past16_future8 | Persistence | rmse | 1.033 +/- 0.000 |
| synthetic_sensors | past12_future12 | Raw context + linear | rmse | 0.339 +/- 0.000 |
| synthetic_sensors | past12_future12 | PCA + linear | rmse | 0.343 +/- 0.000 |
| synthetic_sensors | past12_future12 | Raw context + ExtraTrees | rmse | 0.451 +/- 0.001 |
| synthetic_sensors | past12_future12 | Random encoder + linear | rmse | 0.411 +/- 0.017 |
| synthetic_sensors | past12_future12 | JEPA context + linear | rmse | 0.404 +/- 0.042 |
| synthetic_sensors | past12_future12 | Persistence | rmse | 1.149 +/- 0.000 |

*Full raw reference sees all features/pixels, including those withheld from context-only methods. It is not a matched baseline or theoretical ceiling.

![Primary results](results/figures/primary_results.png)

## Validation

```json
{
  "status": "passed",
  "tests_passed": 44,
  "test_failures": 0,
  "completed_gpu_runs": 18,
  "audited_tasks": 6,
  "export_reload_checks": 6,
  "checkpoint_reload_checks": 18,
  "actual_device": "mps:0",
  "runtime_source_hashes_match": true,
  "low_rank_warning_runs": 7
}
```

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev,report]'
pytest -q --junitxml=artifacts/pytest.xml
jepa-forge inspect wine
python scripts/verify_environment.py
python scripts/run_experiments.py --config configs/benchmark.json --output artifacts/benchmark
python scripts/validate_artifacts.py
python scripts/build_report.py
```

The supplied benchmark config explicitly requires MPS. Change `device` to `cuda` for an available NVIDIA GPU or `cpu` for a disclosed CPU run. GPU unavailability raises an error. Large compiled arrays and checkpoints remain in the local artifact bundle and can be regenerated.

## Limits and next steps

- Small fixed-split feasibility experiments; no broad JEPA superiority or state-of-the-art claim.
- Wine is tiny; Digits does not establish writer-disjoint generalization; synthetic signals do not establish real sensor performance.
- Rule-based leakage checks cannot prove semantic safety or discover every target-derived variable. Checksums detect changed bytes, not trusted provenance.
- Only numeric tabular, small image and grouped sequence adapters are present. Graph/audio/event/multimodal extensions, plugin isolation, community governance and usability studies remain future work.
- The small latent predictor uses a variance regularizer and custom encoders; it is a JEPA-style pilot, not an official I-JEPA implementation.
- Separate future work: real multivariate sensor benchmarks, broader held-out splits, ablations, independent reproduction and user studies.

## Sources

- [I-JEPA paper](https://arxiv.org/abs/2301.08243) and [official implementation](https://github.com/facebookresearch/ijepa).
- [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html).
- [Wine dataset](https://archive.ics.uci.edu/dataset/109/wine), [Optical digits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).
- [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html). See docs/DATASETS.md for dataset attribution.
