# Ten additional public datasets

These adapters download and parse actual UCI archives. They do not replace unavailable observations with synthetic data. All ten official archives were downloaded, their hashes checked, and both task policies compiled successfully on 2026-09-16. The local execution record is `artifacts/public_dataset_checks.json`; this loader check is separate from GPU experiment completion.

The selection broadens dimensions, class counts, sensing structure and grouping. It does **not** mean every dataset has more observations: Sonar has only 208 records and is challenging because it has 60 measurements per record. All results remain bounded pilot results on new partitions, not reproductions of UCI's original benchmarks.

## Data and fixed policies

The first policy below is the declared primary. Policies are specified from feature layout or fixed column positions before training; labels, test scores and fitted correlations do not choose masks.

| Registry name | Original rows × features | Loaded rows × features | Classes | Primary policy, context → target | Secondary policy | Structure and evaluation boundary |
|---|---:|---:|---:|---|---|---|
| `breast_cancer` | 569 × 30 | 569 × 30 | 2 | `mean_to_worst`, 10 → 10 | `mean_to_error`, 10 → 10 | Mean/error/worst morphology groups; original record ID is metadata only. Historical public features do not validate clinical use. |
| `ionosphere` | 351 × 34 | 350 × 34 | 2 | `alternating_pairs`, 18 → 16 | `first_half`, 16 → 18 | Sequence encoder over 17 real/imaginary radar pulse pairs. Session IDs are absent. |
| `sonar` | 208 × 60 | 208 × 60 | 2 | `alternating`, 30 → 30 | `first_half`, 30 → 30 | Sequence encoder respects spectral-band order; the axis is not equally spaced time and the task is classification. |
| `semeion` | 1,593 × 256 | 1,593 × 256 | 10 | `center`, outer 192 → central 64 | `right_half`, 128 → 128 | Binary 16 × 16 image, spatial encoder; writer IDs are unavailable. |
| `letter` | 20,000 × 16 | 3,000 × 16 | 26 | `alternating`, 8 → 8 | `first_half`, 8 → 8 | Geometric and edge features from distorted font images; font identities are unavailable. |
| `pendigits` | 10,992 × 16 | 2,992 × 16 | 10 | `alternating_pairs`, 8 → 8 | `first_half`, 8 → 8 | Sequence of eight x/y trajectory points, with 44 preserved source groups. |
| `satimage` | 6,435 × 36 | 3,000 × 36 | 6 | `neighbor_to_center`, 32 → 4 | `spectral_halves`, 18 → 18 | Four spectral bands at nine neighboring pixels; physical masks with a tabular encoder. Geographic IDs are absent. |
| `har` | 10,299 × 561 | 3,000 × 561 | 6 | `time_to_frequency`, 265 → 289 | `alternating`, 281 → 280 | Engineered time/frequency summaries with 30 explicit subject IDs; subjects are held out. Seven angle features are unused by the primary mask. |
| `dry_bean` | 13,611 × 16 | 3,000 × 16 | 7 | `alternating`, 8 → 8 | `first_half`, 8 → 8 | Correlated image-derived size/shape summaries; derived relationships can make prediction trivial. |
| `isolet` | 7,797 × 617 | 3,000 × 617 | 26 | `alternating`, 309 → 308 | `first_half`, 308 → 309 | High-dimensional acoustic summaries; exact semantic feature ordering and explicit speaker IDs are unavailable. |

Ionosphere's constant input column remains valid and receives a compiler warning. HAR's alternating policy receives an exact-context/target-copy warning. These warnings are retained in the evidence; an audit pass does not certify scientific usefulness.

## Source attribution and licensing

Each linked UCI record displayed **Creative Commons Attribution 4.0 International** when checked on 2026-09-16. The public data keep that license; the repository's code license does not replace it. The source pages, citation, DOI, license URL, archive hash, original dimensions and parser limitations are also embedded in every `RawDataset.metadata` and exported manifest.

| Dataset and primary source | Attribution and DOI |
|---|---|
| [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). DOI [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). |
| [Ionosphere](https://archive.ics.uci.edu/dataset/52/ionosphere) | Sigillito, V., Wing, S., Hutton, L., & Baker, K. (1989). DOI [10.24432/C5W01B](https://doi.org/10.24432/C5W01B). |
| [Connectionist Bench: Sonar](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks) | Sejnowski, T., & Gorman, R. (1988). DOI [10.24432/C5T01Q](https://doi.org/10.24432/C5T01Q). |
| [Semeion Handwritten Digit](https://archive.ics.uci.edu/dataset/178/semeion+handwritten+digit) | Semeion Handwritten Digit (1998). DOI [10.24432/C5SC8V](https://doi.org/10.24432/C5SC8V). |
| [Letter Recognition](https://archive.ics.uci.edu/dataset/59/letter+recognition) | Slate, D. (1991). DOI [10.24432/C5ZP40](https://doi.org/10.24432/C5ZP40). |
| [Pen-Based Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits) | Alpaydin, E., & Alimoglu, F. (1996). DOI [10.24432/C5MG6K](https://doi.org/10.24432/C5MG6K). |
| [Statlog: Landsat Satellite](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite) | Srinivasan, A. (1993). DOI [10.24432/C55887](https://doi.org/10.24432/C55887). |
| [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). DOI [10.24432/C54S4K](https://doi.org/10.24432/C54S4K). |
| [Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) | Dry Bean (2020), DOI [10.24432/C50S4B](https://doi.org/10.24432/C50S4B). Also credit Koklu, M., & Ozkan, I. A. (2020), *Computers and Electronics in Agriculture* 174, 105507, [original paper](https://doi.org/10.1016/j.compag.2020.105507). |
| [ISOLET](https://archive.ics.uci.edu/dataset/54/isolet) | Cole, R., & Fanty, M. (1991). DOI [10.24432/C51G69](https://doi.org/10.24432/C51G69). |

## Sampling, grouping and source preprocessing

The fixed loader seed is `2026`. Exact duplicate **feature vectors** are reduced to their first source occurrence, without consulting labels, before partitioning. The source rows removed are: Ionosphere **1**, Letter **1,332**, Dry Bean **68**, and **0** in the other seven datasets. This changes the observation distribution and is recorded, rather than presented as the unmodified official benchmark.

For ungrouped sets larger than 3,000 unique records, the loader selects 3,000 rows uniformly without replacement with NumPy's seeded generator. For grouped sets, it uses a common per-group budget: HAR retains 100 windows per subject, and PenDigits retains 68 records per source group. Breast Cancer retains all 569 distinct record IDs separately. Original row indices and their SHA-256 are saved; output rows are restored to source order. Sampling never uses class labels, so class balance is not artificially fixed.

`make_splits` assigns whole source groups to nominal 60/20/20 partitions when groups exist. Otherwise it uses a label-independent random row split. At seed 2026:

| Dataset | Train / validation / test rows | Group handling |
|---|---:|---|
| Breast Cancer | 341 / 114 / 114 | Distinct original record IDs; no patient-history inference |
| Ionosphere | 210 / 70 / 70 | No session IDs |
| Sonar | 124 / 42 / 42 | No acquisition IDs |
| Semeion | 955 / 319 / 319 | No writer IDs |
| Letter | 1,800 / 600 / 600 | No font IDs |
| PenDigits | 1,768 / 612 / 612 | 26 / 9 / 9 source groups |
| Satimage | 1,800 / 600 / 600 | No geographic IDs |
| HAR | 1,800 / 600 / 600 | 18 / 6 / 6 explicit subjects |
| Dry Bean | 1,800 / 600 / 600 | No acquisition/batch IDs |
| ISOLET | 1,800 / 600 / 600 | No explicit speaker IDs |

HAR uses the archive's explicit `subject_train.txt` and `subject_test.txt` fields. Its windows overlap in the source, making subject grouping necessary. The 561 inputs are already source-engineered and normalized; this project applies an additional training-only normalization, but cannot claim to audit every original preprocessing operation. Exact recording intervals are not supplied with those feature rows, so no interval coordinates are invented.

PenDigits' normalized feature files omit group IDs, but aligned original UNIPEN files contain a `.COMMENT` record with three fields for each sample. The second field has 30 training groups and 14 test groups of approximately 250 records, matching the donor's writer counts. The adapter verifies the row count and the complete label alignment between those records and the normalized features, then retains the second field under `tra:`/`tes:` namespaces. Its interpretation as writer identity is an inference from the archive structure, not an independently authenticated identity claim. Group-based sampling itself never consults labels.

ISOLET's documentation describes speaker cohorts and three missing recordings, but its numeric files contain no explicit speaker field. The adapter does not guess speaker IDs from fixed row blocks. Semeion similarly lacks explicit writer fields. Satimage can contain neighboring source windows, but geographic positions were removed from its source records. Results for these datasets therefore cannot establish independence across speakers, writers or geographic areas.

Where UCI supplies separate original train/test files, the loader concatenates them in documented order, retains partition counts as provenance, and makes this pilot's new splits. This is a deliberate bounded experiment, not the original published evaluation protocol.

## Download integrity and cache

The cache defaults to `.cache/public_datasets/` under the current working directory. Set `JEPA_FORGE_DATA_HOME` to choose another writable cache directory. Cached and newly downloaded archives must match the pinned SHA-256 below. A mismatch fails clearly and is not silently accepted or replaced. Downloads use official UCI HTTPS URLs, and ZIP members are read without extracting paths. ISOLET and original PenDigits use Unix-compress `.Z` members; the standard `gzip` command must be installed to decode them. No new Python parser dependency beyond NumPy and SciPy is required.

| Archive | Reviewed SHA-256 |
|---|---|
| `breast_cancer.zip` | `bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af` |
| `ionosphere.zip` | `4d218ece62756c99659011a13052d06464f13cb2b3d0410ce2aef16de1403860` |
| `sonar.zip` | `088b0b6813fb5ab84736bc2a1e3d6bb886e317520d2f2c04152e2ed65af7a18d` |
| `semeion.zip` | `6fb091394714cddda5751d4e1c2781ab094e7cf15de07917fb40e581f19efc75` |
| `letter.zip` | `3b5f07a334697b6cace4fbae22940393a18fee596e73f68d97ce5973d52dc60f` |
| `pendigits.zip` | `1e02bea023613c2b11c9492f6f34caf975420455934f3527d270cee9a1f03b64` |
| `satimage.zip` | `7c54e0e11c872a1b0b647da370d596dcb06746159cce4121d92ccd70b7d7ce3c` |
| `har.zip` | `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031` |
| `dry_bean.zip` | `0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f` |
| `isolet.zip` | `fe2e0d45f1057d112e051d309070992fc178e7f6922cc7b4e8b2bd310303a053` |

Hashes establish which downloaded bytes this experiment used. They do not authenticate the original collection process or prove that metadata are scientifically complete.

```python
from jepa_forge.public_datasets import load_public_dataset, propose_public_tasks
from jepa_forge.datasets import make_splits
from jepa_forge.compiler import compile_task

dataset = load_public_dataset("har", seed=2026)
task = propose_public_tasks(dataset)[0]
compiled = compile_task(dataset, task, make_splits(dataset, seed=2026))
```

`tests/test_public_datasets.py` runs its small fixtures offline. Official-data integration cases run only when their archives are already cached; they never download during tests. The separate live loader audit verifies all ten real downloads and twenty compilations in the development workspace.
