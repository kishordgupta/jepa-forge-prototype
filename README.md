# JEPA-FORGE prototype

A working research prototype for converting scientific datasets into explicit, validated context-target tasks and evaluating a small JEPA-style learner. It implements a bounded technical slice of the foundation-world-model ecosystem proposal.

**Measured status: 23 datasets, 138 full GPU benchmark runs, and 120 passing local tests.** The original three-dataset study contains 18 runs; a requested expansion adds ten public datasets and ten distinct synthetic families with **120 additional runs**. Each dataset has two fixed policies and three model seeds, trained for 60 epochs on the Apple M3 Max GPU through PyTorch MPS. Neural training and encoding use GPU; classical baselines use CPU.

The expansion passed all **40 task export reloads and 120 checkpoint hash/replay checks**. Across the 20 predeclared primary tasks, JEPA's seed-mean result beats the matched raw linear baseline on **9/20**, Extra Trees on **3/20**, and a matched random encoder on **9/20**. **83/120 new runs trigger the low-rank heuristic.** These measured limitations are central to the result: the compiler and experiment pipeline work, while the small representation learner does not show a general advantage.

- [Expanded 23-dataset report](EXPANDED_REPORT.md) · [Expanded PDF](output/pdf/JEPA_FORGE_Expanded_23_Dataset_Report.pdf)
- [All expanded per-run evidence (gzip JSON)](results/expanded_benchmark.json.gz) · [240-row summary](results/expanded_summary.json) · [Independent validation](results/expanded_validation.json)
- [Expanded protocol](docs/EXPANDED_PROTOCOL.md) · [10 public sources and sampling](docs/PUBLIC_DATASETS_EXPANDED.md) · [10 synthetic mechanisms](docs/SYNTHETIC_DATASETS_EXPANDED.md)
- [Original three-dataset report](REPORT.md) · [Original results](results/benchmark.json) · [GPU environment](results/environment.json)
- [Independent experiment review](docs/INDEPENDENT_EXPANDED_REVIEW.md) · [Model-quality review](docs/MODEL_QUALITY_REVIEW.md)
- [Architecture](docs/ARCHITECTURE.md) · [Probe/training protocol](docs/EXPERIMENT_PROTOCOL.md) · [Requirements](docs/REQUIREMENTS.md)

![Expanded primary comparisons](results/figures/expanded_comparison.png)

The additional public datasets are Breast Cancer (WDBC), Ionosphere, Sonar, Semeion, Letter, PenDigits, Satimage, Human Activity Recognition, Dry Bean, and ISOLET. The ten synthetic families cover Lorenz trajectories, Mackey–Glass-inspired delayed dynamics, switching VAR, chirp/seasonal signals, coupled oscillators, nonlinear autoregression, nonlinear multiview factors, hierarchical classes, sparse interactions, and manifold data with nuisance variables. Public samples contain 208–3,000 rows and up to 617 features; each synthetic dataset has 2,000 rows. Exact sample sizes, partitions, deduplication and missing identifier limitations are documented above.


## What works

1. Inspect typed numeric data, modality/shape, separate labels, groups, intervals and provenance.
2. Propose explicit context-target policies for tabular, spatial and grouped sequence examples.
3. Audit malformed indices, overlap, duplicate observations across splits, declared labels/identifiers in inputs, group/window leakage and forecasting direction. Constant/copy targets produce warnings requiring review.
4. Fit normalization on training rows and export numeric NPZ plus JSON manifests/checksums.
5. Train a masked context encoder, predictor and frozen EMA target encoder on GPU. Context-only inference accepts context columns only.
6. Compare frozen representations against matched raw linear, PCA, random-encoder and ExtraTrees baselines; use persistence for forecasting and a separately labelled full-input classification reference.

This is a **JEPA-style pilot**, with custom small encoders and an explicit variance penalty. It is not an official I-JEPA reproduction. It does not provide graph/audio/event adapters, arbitrary file-format inference, plugin isolation, usability-study evidence or a complete governed ecosystem.

## Install and inspect

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev,report]'
pytest -q --junitxml=artifacts/pytest.xml

jepa-forge inspect wine
jepa-forge inspect digits
jepa-forge inspect synthetic_sensors
jepa-forge inspect har
jepa-forge inspect synthetic_lorenz
jepa-forge compile digits --task center --output artifacts/compiled/digits-center
jepa-forge verify artifacts/compiled/digits-center
```

Python 3.11+ is supported. `requirements-lock.txt` records the measured macOS environment; `pyproject.toml` supplies portable dependency requirements. CPU unit tests do not require a GPU.

## GPU experiments

```bash
python scripts/verify_environment.py
python scripts/run_experiments.py --config configs/benchmark.json --output artifacts/benchmark
python scripts/validate_artifacts.py
python scripts/build_report.py

# Twenty additional, more complex dataset families:
python scripts/run_experiments.py --config configs/expanded.json --output artifacts/expanded
python scripts/validate_expanded.py
python scripts/plot_expanded.py
python scripts/build_expanded_report.py
```

The fixed benchmark uses three model seeds (`7,17,27`), two policies per dataset, one split/generator seed (`2026`) and 60 epochs. `configs/smoke.json` runs two epochs with one seed to check execution. The supplied configs explicitly request MPS; requested GPU unavailability raises an error. Sandboxed hosts may need permission for GPU device access. For another machine, explicitly set `device` to `cuda` or `cpu` in a copied config and label those results accordingly. The original three-dataset PDF template requires the complete declared MPS benchmark; adapt its compute/template wording for a different benchmark.

Neural training and encoding run on the GPU. Data preparation, covariance diagnostics and scikit-learn estimators run on CPU. No cloud GPU service or paid resource is required.

| Dataset | Shape | Train / validation / test | Primary policy |
|---|---:|---:|---|
| Wine | 178 x 13 | 106 / 36 / 36 | 7 alternating fields -> 6 hidden fields |
| Digits | 1,797 x 64 | 1,078 / 359 / 360 | outer 48 pixels -> central 16 pixels |
| Synthetic sensors | 1,600 x 48 | 960 / 320 / 320 | past 16 steps -> future 8 steps, two channels |

Sensor partitions hold out complete entities: 60/20/20 groups. Wine/Digits use a label-independent random row split. Exact indices and compiler means/scales are saved in local NPZ exports. Probes/PCA estimators are reproduced by refitting the recorded grids, inputs and seeds. The first declared policy is primary a priori; policies have different input/horizon budgets and are not ranked using test scores.

## Original three-dataset primary results

Mean +/- sample standard deviation over three model seeds on one fixed split. F1 is higher-is-better; RMSE is lower-is-better.

| Task / metric | JEPA context + linear | Raw context + linear | Raw context + ExtraTrees |
|---|---:|---:|---:|
| Wine / macro-F1 | 0.914 +/- 0.000 | 0.941 +/- 0.000 | 0.980 +/- 0.018 |
| Digits / macro-F1 | 0.886 +/- 0.008 | 0.875 +/- 0.000 | 0.913 +/- 0.008 |
| Sensors / RMSE | 0.321 +/- 0.021 | 0.259 +/- 0.000 | 0.358 +/- 0.004 |

JEPA improved Digits over the linear context baseline but did not beat ExtraTrees. It underperformed raw-context linear on Wine and sensor forecasting. Sensor persistence RMSE was 1.033. Seven of 18 runs triggered the predeclared low-rank heuristic: all six Wine runs and one 12-step sensor run. These findings support pipeline feasibility and demonstrate why diagnostics and strong baselines are necessary; they do not establish general JEPA superiority.

![Primary results](results/figures/primary_results.png)

## Python API and custom data

```python
from jepa_forge.datasets import load_dataset, make_splits, propose_tasks
from jepa_forge.compiler import compile_task, export_task, load_export

dataset = load_dataset("digits", seed=2026)
compiled = compile_task(dataset, propose_tasks(dataset)[0], make_splits(dataset, seed=2026))
export_task(compiled, "artifacts/compiled/digits-center")
restored = load_export("artifacts/compiled/digits-center")
context = restored.X[:, restored.task.context]
```

For a custom numeric dataset, construct `RawDataset`, `TaskSpec` and explicit train/validation/test index arrays through the Python API. Declare labels, identifiers and entity/time metadata correctly; automated checks cannot discover every semantic shortcut. Labels stay separate from the unsupervised task and are used only by downstream supervised probes. Hidden targets never enter context-only inference; they are used in training, forecast supervision, diagnostics and scoring.

## Artifacts and evidence

`results/` contains measured evidence and hashes, including the complete expanded JSON compressed with gzip. Use `gzip.open("results/expanded_benchmark.json.gz", "rt")` with `json.load` to inspect it. `results/original_runtime.zip` preserves the exact original runtime; the expansion records its updated runtime separately. Large compiled arrays, per-run histories and tensor-only checkpoints are retained under local `artifacts/benchmark/` and `artifacts/expanded/`, excluded from Git, and reproducible from source. `scripts/package_artifacts.py` creates a local archive with code, report, public/synthetic compiled arrays and trained checkpoints. Hashes detect changed bytes, not authorship or trusted provenance.

The reports describe limits including tiny test sets, missing writer/speaker/geographic identifiers, synthetic-only forecasting claims, one fixed split and incomplete modality/community support. The proposal's existing numeric results are not reused as measurements here.

Original prototype code is Apache-2.0 licensed. Public datasets retain their own licenses and attribution in [DATASETS.md](docs/DATASETS.md) and [expanded public dataset documentation](docs/PUBLIC_DATASETS_EXPANDED.md).
