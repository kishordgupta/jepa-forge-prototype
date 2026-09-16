# JEPA-FORGE automatic task-selection report

Automatic context/target selection is implemented and validated across 23 public/synthetic datasets. The study evaluates 91 schema-based candidates with three seeds each: 273 additional 60-epoch GPU training runs. All 23 selections were locked before test evaluation. Only selected tasks were evaluated on the test partition (69 seed evaluations). Together with the separate earlier 138-run study, this gives 411 full GPU training runs; smoke tests are excluded.

## What the extension does

The selector receives copied training/validation rows only. It compares equal context budgets and identical forecasting targets. Training labels fit classification probes, and validation labels rank candidates; JEPA training remains label-free. Candidate choice uses mean validation macro F1 or negative RMSE across all seeds, with fixed-order ties. The test partition is used only after a durable global selection lock.

```mermaid
flowchart LR
 A["Dataset and schema"] --> B["Train / validation copies"]
 B --> C["Equal-budget candidates"]
 C --> D["GPU training + validation probes"]
 D --> E["Lock all 23 choices"]
 E --> F["Final selected-task test evaluation"]
```

## Final selected-task results

Mean +/- sample standard deviation over three seeds. Macro F1 is higher-is-better; RMSE is lower-is-better. Each baseline receives the same context as its selected task.

| Dataset | Chosen context | Features | Metric | JEPA + linear | Raw linear | Extra Trees |
|---|---|---:|---|---:|---:|---:|
| Wine | last_columns | 6 | macro_f1 | 0.899 +/- 0.015 | 0.891 +/- 0.000 | 0.936 +/- 0.014 |
| Digits | right_half | 32 | macro_f1 | 0.883 +/- 0.022 | 0.857 +/- 0.000 | 0.902 +/- 0.011 |
| Original sensors | recent_past | 18 | rmse | 0.276 +/- 0.017 | 0.239 +/- 0.000 | 0.351 +/- 0.003 |
| Breast cancer (WDBC) | worst_measurements | 10 | macro_f1 | 0.969 +/- 0.005 | 0.963 +/- 0.000 | 0.953 +/- 0.009 |
| Ionosphere | alternating_tokens | 16 | macro_f1 | 0.897 +/- 0.036 | 0.883 +/- 0.000 | 0.978 +/- 0.010 |
| Sonar | alternating_tokens | 30 | macro_f1 | 0.730 +/- 0.014 | 0.712 +/- 0.000 | 0.776 +/- 0.014 |
| Semeion | left_half | 128 | macro_f1 | 0.754 +/- 0.013 | 0.768 +/- 0.000 | 0.824 +/- 0.009 |
| Letter | last_columns | 8 | macro_f1 | 0.665 +/- 0.036 | 0.598 +/- 0.000 | 0.823 +/- 0.004 |
| PenDigits | alternating_tokens | 8 | macro_f1 | 0.904 +/- 0.002 | 0.807 +/- 0.000 | 0.948 +/- 0.004 |
| SatImage | bands_1_3 | 18 | macro_f1 | 0.828 +/- 0.007 | 0.766 +/- 0.000 | 0.874 +/- 0.004 |
| Human activity (HAR) | first_columns | 280 | macro_f1 | 0.785 +/- 0.024 | 0.946 +/- 0.000 | 0.895 +/- 0.008 |
| Dry Bean | last_columns | 8 | macro_f1 | 0.918 +/- 0.005 | 0.926 +/- 0.000 | 0.916 +/- 0.006 |
| ISOLET | alternating_columns | 308 | macro_f1 | 0.764 +/- 0.012 | 0.943 +/- 0.000 | 0.932 +/- 0.012 |
| Lorenz trajectories | recent_past | 36 | rmse | 0.754 +/- 0.034 | 2.500 +/- 0.000 | 0.617 +/- 0.005 |
| Mackey-Glass delay | recent_past | 36 | rmse | 0.083 +/- 0.002 | 0.079 +/- 0.000 | 0.082 +/- 0.001 |
| Switching VAR | recent_past | 48 | rmse | 0.178 +/- 0.001 | 0.180 +/- 0.000 | 0.177 +/- 0.000 |
| Chirp + seasonal | recent_past | 36 | rmse | 0.226 +/- 0.021 | 0.191 +/- 0.000 | 0.220 +/- 0.001 |
| Coupled oscillators | recent_past | 48 | rmse | 0.109 +/- 0.016 | 0.035 +/- 0.000 | 0.135 +/- 0.001 |
| Nonlinear AR | spaced_past | 36 | rmse | 0.098 +/- 0.001 | 0.094 +/- 0.000 | 0.114 +/- 0.000 |
| Nonlinear multiview | first_columns | 32 | macro_f1 | 0.881 +/- 0.005 | 0.840 +/- 0.000 | 0.907 +/- 0.010 |
| Hierarchical classes | alternating_columns | 40 | macro_f1 | 0.968 +/- 0.006 | 0.965 +/- 0.000 | 0.968 +/- 0.006 |
| Sparse interactions | alternating_columns | 48 | macro_f1 | 0.730 +/- 0.007 | 0.907 +/- 0.000 | 0.867 +/- 0.012 |
| Manifold + nuisance | first_columns | 48 | macro_f1 | 0.807 +/- 0.047 | 0.773 +/- 0.000 | 0.973 +/- 0.004 |

The selected JEPA encoder beats raw linear on 13/23 datasets, Extra Trees on 5/23, and the random encoder on 12/23. The low-rank heuristic triggers in 198/273 candidate runs. These descriptive comparisons do not establish statistical significance or a general JEPA advantage. Unselected candidates were not scored on test, so this study does not estimate test-set improvement over every possible feature division.

## Candidate rankings and exact feature choices

Validation scores below determine selection. Negative RMSE is reported as positive RMSE for readability; lower is better. Classification uses higher macro F1.

### Wine

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| first_columns | 0.891312 | 0.057177 |  |
| last_columns | 0.967963 | 0.000000 | yes |
| alternating_columns | 0.928286 | 0.015244 |  |
| middle_columns | 0.865497 | 0.044791 |  |

Context columns (zero-based): `[7, 8, 9, 10, 11, 12]`.

Target columns (zero-based): `[0, 1, 2, 3, 4, 5, 6]`.

### Digits

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| left_half | 0.873783 | 0.004668 |  |
| right_half | 0.886894 | 0.010120 | yes |
| top_half | 0.857005 | 0.017399 |  |
| bottom_half | 0.855676 | 0.009024 |  |

Context columns (zero-based): `[4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63]`.

Target columns (zero-based): `[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59]`.

### Original sensors

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| recent_past | 0.245175 | 0.015291 | yes |
| early_past | 0.547922 | 0.010839 |  |
| spaced_past | 0.270862 | 0.018906 |  |
| middle_past | 0.488355 | 0.021076 |  |

Context columns (zero-based): `[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]`.

Target columns (zero-based): `[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]`.

### Breast cancer (WDBC)

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| mean_measurements | 0.916177 | 0.004538 |  |
| standard_error_measurements | 0.804222 | 0.018403 |  |
| worst_measurements | 0.963849 | 0.009167 | yes |

Context columns (zero-based): `[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]`.

Target columns (zero-based): `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`.

### Ionosphere

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| first_tokens | 0.830986 | 0.040100 |  |
| last_tokens | 0.705098 | 0.061423 |  |
| alternating_tokens | 0.879556 | 0.010672 | yes |
| middle_tokens | 0.792759 | 0.010801 |  |

Context columns (zero-based): `[0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29]`.

Target columns (zero-based): `[2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 32, 33]`.

### Sonar

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| first_tokens | 0.780380 | 0.050167 |  |
| last_tokens | 0.683486 | 0.077541 |  |
| alternating_tokens | 0.831518 | 0.024033 | yes |
| middle_tokens | 0.596526 | 0.033150 |  |

Context columns (zero-based): `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58]`.

Target columns (zero-based): `[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59]`.

### Semeion

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| left_half | 0.781432 | 0.038278 | yes |
| right_half | 0.708072 | 0.021324 |  |
| top_half | 0.738823 | 0.011205 |  |
| bottom_half | 0.720071 | 0.039181 |  |

Context columns (zero-based): `[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55, 64, 65, 66, 67, 68, 69, 70, 71, 80, 81, 82, 83, 84, 85, 86, 87, 96, 97, 98, 99, 100, 101, 102, 103, 112, 113, 114, 115, 116, 117, 118, 119, 128, 129, 130, 131, 132, 133, 134, 135, 144, 145, 146, 147, 148, 149, 150, 151, 160, 161, 162, 163, 164, 165, 166, 167, 176, 177, 178, 179, 180, 181, 182, 183, 192, 193, 194, 195, 196, 197, 198, 199, 208, 209, 210, 211, 212, 213, 214, 215, 224, 225, 226, 227, 228, 229, 230, 231, 240, 241, 242, 243, 244, 245, 246, 247]`.

Target columns (zero-based): `[8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63, 72, 73, 74, 75, 76, 77, 78, 79, 88, 89, 90, 91, 92, 93, 94, 95, 104, 105, 106, 107, 108, 109, 110, 111, 120, 121, 122, 123, 124, 125, 126, 127, 136, 137, 138, 139, 140, 141, 142, 143, 152, 153, 154, 155, 156, 157, 158, 159, 168, 169, 170, 171, 172, 173, 174, 175, 184, 185, 186, 187, 188, 189, 190, 191, 200, 201, 202, 203, 204, 205, 206, 207, 216, 217, 218, 219, 220, 221, 222, 223, 232, 233, 234, 235, 236, 237, 238, 239, 248, 249, 250, 251, 252, 253, 254, 255]`.

### Letter

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| first_columns | 0.334377 | 0.002597 |  |
| last_columns | 0.648409 | 0.027875 | yes |
| alternating_columns | 0.544103 | 0.012952 |  |
| middle_columns | 0.552895 | 0.013395 |  |

Context columns (zero-based): `[8, 9, 10, 11, 12, 13, 14, 15]`.

Target columns (zero-based): `[0, 1, 2, 3, 4, 5, 6, 7]`.

### PenDigits

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| first_tokens | 0.840346 | 0.009690 |  |
| last_tokens | 0.877605 | 0.010528 |  |
| alternating_tokens | 0.901452 | 0.004347 | yes |
| middle_tokens | 0.878673 | 0.009985 |  |

Context columns (zero-based): `[0, 1, 4, 5, 8, 9, 12, 13]`.

Target columns (zero-based): `[2, 3, 6, 7, 10, 11, 14, 15]`.

### SatImage

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| bands_0_1 | 0.780523 | 0.016950 |  |
| bands_2_3 | 0.686935 | 0.029665 |  |
| bands_0_2 | 0.783149 | 0.011750 |  |
| bands_1_3 | 0.801806 | 0.005791 | yes |

Context columns (zero-based): `[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]`.

Target columns (zero-based): `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]`.

### Human activity (HAR)

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| first_columns | 0.775103 | 0.004267 | yes |
| last_columns | 0.745127 | 0.016112 |  |
| alternating_columns | 0.752960 | 0.024877 |  |
| middle_columns | 0.653991 | 0.007628 |  |

Context columns (zero-based): `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279]`.

Target columns (zero-based): `[280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560]`.

### Dry Bean

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| first_columns | 0.898103 | 0.001543 |  |
| last_columns | 0.930579 | 0.007044 | yes |
| alternating_columns | 0.921186 | 0.006511 |  |
| middle_columns | 0.916615 | 0.006976 |  |

Context columns (zero-based): `[8, 9, 10, 11, 12, 13, 14, 15]`.

Target columns (zero-based): `[0, 1, 2, 3, 4, 5, 6, 7]`.

### ISOLET

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| first_columns | 0.546079 | 0.028050 |  |
| last_columns | 0.727892 | 0.018148 |  |
| alternating_columns | 0.757993 | 0.011392 | yes |
| middle_columns | 0.757287 | 0.013283 |  |

Context columns (zero-based): `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 260, 262, 264, 266, 268, 270, 272, 274, 276, 278, 280, 282, 284, 286, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336, 338, 340, 342, 344, 346, 348, 350, 352, 354, 356, 358, 360, 362, 364, 366, 368, 370, 372, 374, 376, 378, 380, 382, 384, 386, 388, 390, 392, 394, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414, 416, 418, 420, 422, 424, 426, 428, 430, 432, 434, 436, 438, 440, 442, 444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474, 476, 478, 480, 482, 484, 486, 488, 490, 492, 494, 496, 498, 500, 502, 504, 506, 508, 510, 512, 514, 516, 518, 520, 522, 524, 526, 528, 530, 532, 534, 536, 538, 540, 542, 544, 546, 548, 550, 552, 554, 556, 558, 560, 562, 564, 566, 568, 570, 572, 574, 576, 578, 580, 582, 584, 586, 588, 590, 592, 594, 596, 598, 600, 602, 604, 606, 608, 610, 612, 614]`.

Target columns (zero-based): `[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 255, 257, 259, 261, 263, 265, 267, 269, 271, 273, 275, 277, 279, 281, 283, 285, 287, 289, 291, 293, 295, 297, 299, 301, 303, 305, 307, 309, 311, 313, 315, 317, 319, 321, 323, 325, 327, 329, 331, 333, 335, 337, 339, 341, 343, 345, 347, 349, 351, 353, 355, 357, 359, 361, 363, 365, 367, 369, 371, 373, 375, 377, 379, 381, 383, 385, 387, 389, 391, 393, 395, 397, 399, 401, 403, 405, 407, 409, 411, 413, 415, 417, 419, 421, 423, 425, 427, 429, 431, 433, 435, 437, 439, 441, 443, 445, 447, 449, 451, 453, 455, 457, 459, 461, 463, 465, 467, 469, 471, 473, 475, 477, 479, 481, 483, 485, 487, 489, 491, 493, 495, 497, 499, 501, 503, 505, 507, 509, 511, 513, 515, 517, 519, 521, 523, 525, 527, 529, 531, 533, 535, 537, 539, 541, 543, 545, 547, 549, 551, 553, 555, 557, 559, 561, 563, 565, 567, 569, 571, 573, 575, 577, 579, 581, 583, 585, 587, 589, 591, 593, 595, 597, 599, 601, 603, 605, 607, 609, 611, 613, 615, 616]`.

### Lorenz trajectories

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| recent_past | 0.662429 | 0.057622 | yes |
| early_past | 2.265936 | 0.151511 |  |
| spaced_past | 0.881483 | 0.062106 |  |
| middle_past | 1.287239 | 0.079892 |  |

Context columns (zero-based): `[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]`.

Target columns (zero-based): `[72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]`.

### Mackey-Glass delay

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| recent_past | 0.081222 | 0.003042 | yes |
| early_past | 0.179167 | 0.003597 |  |
| spaced_past | 0.088003 | 0.002067 |  |
| middle_past | 0.136285 | 0.001873 |  |

Context columns (zero-based): `[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]`.

Target columns (zero-based): `[72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]`.

### Switching VAR

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| recent_past | 0.179304 | 0.000443 | yes |
| early_past | 0.193655 | 0.000572 |  |
| spaced_past | 0.183221 | 0.000409 |  |
| middle_past | 0.190342 | 0.000048 |  |

Context columns (zero-based): `[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]`.

Target columns (zero-based): `[96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]`.

### Chirp + seasonal

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| recent_past | 0.202674 | 0.017779 | yes |
| early_past | 0.411396 | 0.011475 |  |
| spaced_past | 0.235431 | 0.012814 |  |
| middle_past | 0.315064 | 0.004864 |  |

Context columns (zero-based): `[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]`.

Target columns (zero-based): `[72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]`.

### Coupled oscillators

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| recent_past | 0.125785 | 0.020759 | yes |
| early_past | 0.180765 | 0.012917 |  |
| spaced_past | 0.133601 | 0.015118 |  |
| middle_past | 0.158552 | 0.011253 |  |

Context columns (zero-based): `[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]`.

Target columns (zero-based): `[96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]`.

### Nonlinear AR

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| recent_past | 0.109546 | 0.005176 |  |
| early_past | 0.115823 | 0.004416 |  |
| spaced_past | 0.104595 | 0.001510 | yes |
| middle_past | 0.120671 | 0.010853 |  |

Context columns (zero-based): `[0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26, 30, 31, 32, 36, 37, 38, 42, 43, 44, 48, 49, 50, 54, 55, 56, 60, 61, 62, 66, 67, 68]`.

Target columns (zero-based): `[72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]`.

### Nonlinear multiview

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| first_columns | 0.890535 | 0.012679 | yes |
| last_columns | 0.835876 | 0.001329 |  |
| alternating_columns | 0.877376 | 0.016752 |  |
| middle_columns | 0.873252 | 0.012171 |  |

Context columns (zero-based): `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]`.

Target columns (zero-based): `[32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]`.

### Hierarchical classes

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| first_columns | 0.968067 | 0.006706 |  |
| last_columns | 0.957471 | 0.005630 |  |
| alternating_columns | 0.971236 | 0.002284 | yes |
| middle_columns | 0.969714 | 0.003533 |  |

Context columns (zero-based): `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78]`.

Target columns (zero-based): `[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79]`.

### Sparse interactions

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| first_columns | 0.594957 | 0.009956 |  |
| last_columns | 0.715574 | 0.037771 |  |
| alternating_columns | 0.755795 | 0.046195 | yes |
| middle_columns | 0.706073 | 0.008622 |  |

Context columns (zero-based): `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94]`.

Target columns (zero-based): `[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95]`.

### Manifold + nuisance

| Candidate | Validation mean | Sample SD | Selected |
|---|---:|---:|---|
| first_columns | 0.825404 | 0.024293 | yes |
| last_columns | 0.629464 | 0.006974 |  |
| alternating_columns | 0.821146 | 0.046537 |  |
| middle_columns | 0.610451 | 0.009660 |  |

Context columns (zero-based): `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]`.

Target columns (zero-based): `[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]`.

## Verification

169 local tests passed. Independent validation replayed all 273 checkpoints on CPU training samples and all 273 development probes. Maximum GPU-to-CPU embedding error was 1.34e-05, within the 1e-4 absolute/relative tolerance. All 23 selected exports reloaded successfully.

GPU tensor, gradient and loss evidence identifies MPS execution without fallback. Data preparation, covariance checks and sklearn probes use CPU. Wall time includes monitoring and evaluation; it is not a hardware benchmark.

## Privacy and publication

The privacy audit found no credentials under the applied scans. Two historical JUnit reports contained a personal machine name; their published replacements remove that metadata. A sanitized replacement for the affected branch ancestry is prepared; main-history replacement is pending explicit approval. Gitleaks and metadata checks cover staged files, branch history, archives, and PDF text; a fake credential was correctly blocked. Local commit/push hooks and GitHub CI repeat the checks. GitHub's settings state branch protection is not enforced for this private repository's current account setup; native secret-scanning/push-protection controls were not offered in the observed settings. This cannot erase copies or platform-retained objects, or guarantee every future manual/API upload. See SECURITY_AUDIT.md for scope and final publication verification.

## Limits

These are structure-based candidate templates, not learned causal feature importance or an exhaustive search. Generic tabular blocks need domain review. Probe tuning and mask ranking share validation, so validation can be optimistic. Data are the same 23 previously studied datasets with a new predeclared split (3026); this is not independent external replication. One split, three seeds, bounded public samples, missing subject identifiers in some sources, and synthetic forecasting limit generalization.

## Reproduce

```bash
python scripts/run_selection.py --config configs/selection.json --output artifacts/selection
python scripts/validate_selection.py
python scripts/run_tests_private.py
python scripts/build_selection_report.py
python scripts/check_privacy.py --worktree
```

Use a fresh output directory for each rerun; existing selection locks are not overwritten. Public-data sources and licenses: docs/DATASETS.md and docs/PUBLIC_DATASETS_EXPANDED.md. Synthetic mechanisms: docs/SYNTHETIC_DATASETS_EXPANDED.md. Full protocol: docs/SELECTION_PROTOCOL.md.
