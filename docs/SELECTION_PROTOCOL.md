# Automatic context/target selection protocol

Declared before running the selection study. The configuration is `configs/selection.json`.

The extension proposes a small set of schema-based feature masks, trains one JEPA-style encoder per candidate and seed, and selects the candidate with the highest mean validation score. Classification uses macro F1; forecasting uses negative RMSE in the original target units. Exact ties select the first declared candidate. Classification labels fit training probes and rank validation performance; the JEPA encoder remains label-free. This is supervised task selection, not fully unsupervised feature selection, causal discovery, or an exhaustive search of feature subsets.

Candidate families:

* Images: left, right, top, or bottom half observed; the complement is the prediction target.
* Ordered classification sequences: first, last, alternating, or middle half of tokens, preserving every channel in each chosen token. Odd-length sequences use the same floor-half budget in all candidates.
* Forecasting: predict the final quarter of timesteps for every candidate. Observe half of the earlier timesteps using recent, early, spaced, or middle blocks. All context timesteps strictly precede every target timestep.
* Breast Cancer: use the documented mean, standard-error, or worst measurement family, each with ten context features.
* Satellite: four pairs of spectral bands, each with eighteen context features.
* Other tabular datasets: first, last, alternating, or middle half of columns. These are generic ordered-column hypotheses; their scientific usefulness needs domain review. They do not infer semantic importance from column names.

All candidates within a dataset have the same context budget. Forecast candidates also have identical target coordinates and horizon, making their RMSE values comparable. The training budget, probe grid, seeds, and candidate order are fixed. Any failed candidate stops the run; failures are not silently excluded.

The original 23 datasets are reused with data seed 2026 and a newly predeclared label-independent split seed 3026. Splits are 60/20/20 by row or whole entity where available. This is a fresh partition of previously studied datasets, not independent new external data. Existing public-data provenance and sampling limitations still apply.

Before search, the orchestrator copies only training and validation rows into a development object. The encoder's dataset has no labels, and labels are held separately for probes. Arbitrary source metadata are excluded. Development normalization fits training rows only. The development compiler audits feature roles, groups, exact duplicates, and temporal direction without receiving test rows. The selection probe accepts only train/validation partitions.

Each candidate trains for 60 epochs with seeds 7, 17, and 27. A training-fitted standardized logistic/ridge probe chooses among strengths 0.01, 0.1, 1, 10, and 100 on validation. Candidate selection averages validation scores across all three seeds; it does not pick the best seed. Reuse of validation for probe and mask selection can make validation scores optimistic. Test scores are the final evaluation, not ranking inputs. No significance or general superiority claim is planned from three seeds on one split.

Each dataset's ranking, feature indices, validation evidence, checkpoint hashes, and development fingerprints are written to a selection lock. All 23 selections finish and a global lock is written before any test evaluation. Only the selected task reaches final test evaluation. Its checkpoint is reused, with no train+validation refit. Raw linear, PCA, Extra Trees, random-encoder, and full-information/persistence baselines are then evaluated for that same selected task. The final JEPA probe must reproduce the locked validation choice.

JEPA training and encoding use the requested GPU without silent CPU fallback. Scikit-learn probes and preprocessing use CPU. CPU-only unit tests and checkpoint replay are separate from the GPU experiment evidence.
