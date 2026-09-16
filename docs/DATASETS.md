# Dataset provenance and task definitions

Source pages were checked on 2026-09-16. Wine and Digits are bundled with scikit-learn; the prototype uses `load_wine` and `load_digits` rather than fetching private or account-connected data. Package installation may require network access, but these loaders do not require a separate dataset download. The synthetic data are generated locally.

| Dataset | Loader and size | Prototype role | Source and dataset license |
| --- | --- | --- | --- |
| Wine | `sklearn.datasets.load_wine`; 178 records, 13 numeric features, 3 classes; class counts 59, 71, 48 | Typed tabular context/target compilation; cultivar classification from frozen context representations | [scikit-learn loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html); [UCI Wine](https://archive.ics.uci.edu/dataset/109/wine), whose current record states CC BY 4.0 |
| Digits | `sklearn.datasets.load_digits`; 1,797 images of shape `8 x 8`, 64 integer-valued pixels in 0–16, 10 classes | Spatial mask compilation; digit classification from visible-context representations | [scikit-learn loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html); [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), whose current record states CC BY 4.0 |
| Synthetic sensors | Local generator seed `2026`; 100 groups, 16 windows per group, 1,600 samples of shape `24 x 2` (48 flattened features) | Group-separated temporal context/target compilation and multistep forecasting | Generated for this repository; metadata declares CC0-1.0 for generated data; no measured people, devices, or private records |

UCI's full optical-digits collection contains 5,620 records. The 1,797 examples returned by scikit-learn are a copy of that collection's original test subset. This pilot creates a new internal split within those examples; it does not reproduce the original UCI training/test benchmark. The bundled loader does not provide writer IDs suitable for this pilot's group split, so a random record split cannot establish generalization to unseen writers. [scikit-learn description](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), [UCI provenance](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).

## Attribution and transformations

For Wine, credit **Stefan Aeberhard and M. Forina**, *Wine* (1992), UCI Machine Learning Repository, [DOI: 10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J). For Digits, credit **E. Alpaydin and C. Kaynak**, *Optical Recognition of Handwritten Digits* (1998), UCI Machine Learning Repository, [DOI: 10.24432/C50P49](https://doi.org/10.24432/C50P49). UCI lists both datasets under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). This is the dataset attribution; the repository's software license does not replace it.

The prototype transforms these data by internal train/validation/test splitting, train-fitted normalization where configured, context/target masking, and learned representations. Preserve these changes and the source attribution with any redistributed task export. Dataset metadata from UCI establishes the stated source license; this documentation does not make a broader certification claim about arbitrary future datasets.

## Policies and visibility

Indices are zero-based. The following are the actual `TaskSpec.name` values. An export includes exact context and target indices, shape, policy name, and feature or coordinate order; a policy name alone is insufficient to reconstruct the task.

| Modality | Candidate policies | Required interpretation |
| --- | --- | --- |
| Wine | `alternating` (primary) | Even columns `[0,2,4,6,8,10,12]` form 7 context features; odd columns `[1,3,5,7,9,11]` form 6 targets. |
| Wine | `first_half` | Columns 0–5 form 6 context features; columns 6–12 form 7 targets. |
| Digits | `center` (primary) | Central rows 2–5 and columns 2–5 form the 16-pixel target; the outer 48 pixels form context. Flattening is row-major. |
| Digits | `right_half` | Left columns 0–3 form 32 context pixels; right columns 4–7 form 32 targets. Flattening is row-major. |
| Sensors | `past16_future8` (primary) | Context times 0–15; future target times 16–23; both channels at each time, giving 32 context and 16 target coordinates. |
| Sensors | `past12_future12` | Context times 0–11; future target times 12–23; both channels at each time, giving 24 context and 24 target coordinates. |

These are controlled toy tasks. Wine feature order does not imply a causal or spatial ordering. Digits masks test small-image behavior rather than natural-image performance. Synthetic series test known generator dynamics rather than measured sensor behavior.

## Synthetic generation and splitting

The experiment runner passes seed **2026** to the dataset loader and the split constructor; model seeds are separately **7, 17, 27**. The generator seed is also recorded in dataset metadata. The same generated dataset is used across model seeds so differences reflect model randomness rather than different data.

For each of 100 independent groups, draw frequency `f ~ Uniform(0.035, 0.085)`, amplitude `A ~ Uniform(0.7, 1.6)`, phase `phi ~ Uniform(-pi, pi)`, and two offsets `b_c ~ Normal(0, 0.25^2)`. Generate 144 time steps. A shared AR(1) process starts at `a_0 ~ Normal(0, 0.08^2)` and evolves as `a_t = 0.82*a_(t-1) + e_t`, with `e_t ~ Normal(0, 0.055^2)`. The two channels are:

```text
angle_t = 2*pi*f*t + phi
x_t,0 = A*sin(angle_t) + a_t + b_0 + noise_t,0
x_t,1 = 0.75*A*sin(angle_t + 0.45) + 0.6*a_t + b_1 + noise_t,1
noise_t,c ~ Normal(0, 0.035^2), independently
```

Extract 16 windows of length 24 at stride 8 from each group. Window `w` covers the inclusive source interval `[8*w, 8*w + 23]`. Flatten each window in time-major order: the two channel values for time 0, then the two for time 1, and so on. The dataset has no classification label (`y=None`); future measurement coordinates provide forecasting targets.

Split sensor groups before training: every window from a source group belongs wholly to training, validation, or test. This prevents related or overlapping windows from the same generated trajectory from entering multiple partitions. Store group IDs and source intervals in each task export so this property can be audited.

All splits use the fixed seed `2026` and nominal 60/20/20 proportions. Wine and Digits use a label-independent random record permutation, with cut points rounded down; they are **not stratified**. Sensor groups use a random group permutation and the same proportions. The realized counts were checked against the loader and split code:

| Dataset | Training | Validation | Test | Split unit |
| --- | ---: | ---: | ---: | --- |
| Wine | 106 | 36 | 36 | Records |
| Digits | 1,078 | 359 | 360 | Records |
| Synthetic sensors | 960 | 320 | 320 | 60 / 20 / 20 source groups, each with 16 windows |

Training-only preprocessing also applies to public data. Classification labels are held separately from inputs and used only for downstream probes and evaluation; they do not determine the split or task templates. Test features and labels do not fit normalization, encoders, probes, policy selection, or hyperparameters. Structural audits may inspect partition membership and duplicate observations without fitting a model or using test performance.
