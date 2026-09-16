# JEPA-FORGE prototype requirements

This repository is a limited feasibility pilot for compiling typed data into context/target prediction tasks and testing learned representations. It is not an official reproduction of a published JEPA model or evidence that the complete research program has been delivered. Numerical examples from a proposal are not experimental results for this implementation.

## Scope and verification status

The table separates implementation and local checks from completed GPU benchmark evidence. The compiler, model, and evaluation paths have been implemented and checked; the final experiment matrix must still be confirmed from its run artifacts before reporting measured performance.

| Capability | Required prototype behavior | Verification status |
| --- | --- | --- |
| GPU runtime | Actual PyTorch MPS tensor forward and backward computation; record device and versions | **Verified** in `artifacts/environment.json` and `scripts/verify_environment.py` |
| Typed raw data | `RawDataset` carries a typed schema, metadata, numeric `X`, downstream `y`, and group/interval information when applicable | Implemented; compiler tests cover array, role, group, and interval constraints |
| Typed task | `TaskSpec` identifies context indices, target indices, and policy; `RawDataset` supplies the modality | Implemented; all six declared tasks inspected and audited locally |
| Audit and compilation | `audit_task` rejects invalid tasks; `compile_task` materializes audited views | Implemented and tested, including overlap, duplicate observations, temporal direction, and train-only normalization |
| Portable export | `export_task` and `load_export` round-trip JSON metadata and NPZ arrays with checksums; no pickle-dependent payload | Implemented and tested for round-trip equivalence, tampering, and rejection of object payloads |
| Tabular adapter | Wine with `alternating` and `first_half` policies | Implemented; both policies compile and audit successfully; verified in the completed 18-run GPU matrix (`results/benchmark.json`) |
| Spatial adapter | Digits with `center` and `right_half` policies | Implemented; both policies compile and audit successfully; verified in the completed 18-run GPU matrix (`results/benchmark.json`) |
| Sequence adapter | Synthetic grouped sensor windows, shape `24 x 2`, with `past16_future8` and `past12_future12` policies | Implemented; both policies compile and group/interval checks pass; verified in the completed 18-run GPU matrix (`results/benchmark.json`) |
| Representation learning | Modality-specific online encoder, frozen EMA target encoder, predictor, latent prediction loss, variance regularizer | Implemented; CPU tests cover context-only inference, label exclusion, EMA, diagnostics, and checkpoint reload; verified in the completed 18-run GPU matrix (`results/benchmark.json`) |
| Evaluation | Fixed splits and three model seeds; context-matched baselines; validation-only tuning; final held-out metrics | Implemented and reviewed; six held-out perturbation checks verify probe selection is unaffected by test data; complete measured evidence in `results/benchmark.json` |
| Research extensions | Graph, audio, and event plugins; user studies; broad benchmark or deployment claims | Outside this pilot |

Relevant automated checks are in [`test_compiler.py`](../tests/test_compiler.py), [`test_model.py`](../tests/test_model.py), and [`test_evaluation.py`](../tests/test_evaluation.py). Passing local checks establishes the tested contracts; it does not establish benchmark superiority or validate every possible input.

## Compiler acceptance criteria

1. Validate schema, array shape, numeric type, finite values, sample count, and downstream-label alignment. The numeric compiler validates explicit index tasks; the model backend rejects unsupported modalities and the CLI rejects unregistered policy names.
2. Require context and target index sets to be nonempty, unique, in range, and disjoint. Preserve index order and the complete policy definition in exported metadata.
3. For forecasting, require every context time to precede every target time. Two channels at one time step belong to the same temporal position.
4. Keep raw record identities in disjoint train, validation, and test sets. For sensor windows, assign all windows from one source group to one split and check the interval metadata consistently with that assignment.
5. Fit normalization and any learned transformation on training records only. Labels for classification may train downstream probes, but do not enter unsupervised encoder training.
6. Record enough schema, policy, split, preprocessing, and checksum information to load an export and detect a mismatched or modified payload. Checksums provide integrity checking, not authentication or proof of provenance.

## Model and evidence requirements

Use an MLP for tabular data, a two-dimensional convolutional encoder for Digits values plus their observation mask, and a one-dimensional convolutional encoder for sensor sequences. A context representation predicts a stop-gradient target representation. The target encoder updates by exponential moving average rather than backpropagation. A variance penalty is intended to discourage degenerate constant embeddings; report embedding diagnostics because the penalty alone does not prove that collapse was avoided.

GPU training evidence must identify the actual device used. Scikit-learn preprocessing and downstream estimators may run on CPU and must be reported that way. Save configuration, seeds, split identifiers, per-run metrics, and any failed runs. The environment smoke check is evidence of GPU capability, not evidence that model experiments have run.

Use only the public bundled datasets and the documented synthetic generator. Public repository content must exclude private proposal text, support letters, budgets, credentials, and collaborator details. The report should map this pilot's verified components to these requirements and describe unfinished items directly.
