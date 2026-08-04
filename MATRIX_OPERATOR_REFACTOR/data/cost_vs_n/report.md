# Task 030 analysis

Campaign: 42 measured cases from /Users/ryan/Dropbox/research/projects/tmp3/FastMultipole/MATRIX_OPERATOR_REFACTOR/data/cost_vs_n.

## Stage reconciliation

Median-class columns only. `verdict - (refresh + eval + finalize + euler)`,
as a share of the verdict step:

| n | worst residual | note |
|---|---|---|
| 1000 | 6.7% | usable with care |
| 3162 | 6.3% | usable with care |
| 10000 | 6.5% | usable with care |
| 31623 | 3.9% | usable with care |
| 100000 | 3.7% | usable with care |
| 316228 | 16.3% | **per-stage split not usable** (each stage carries its own sync) |
| 1000000 | 19.3% | **per-stage split not usable** (each stage carries its own sync) |

`eval_ms - sum(stages)` is negative throughout (most negative -1.194 ms): the measured nearfield/L2B overlap gain, since the pipeline
overlaps them while `l2b_ms` times the standalone fused kernel.

Single-shot `route_gen_ms` telemetry, for scale only: 1000=>0.331ms, 3162=>0.419ms, 10000=>0.594ms, 31623=>0.881ms, 100000=>1.14ms, 316228=>1.181ms, 1000000=>1.194ms.

## Per-stage breakdown (median-class columns, ms)

`M2L` includes window generation. `M2M+L2L` is per-level launch-bound work,
which is what dominates the small-`n` floor. Rows are the six series at each `n`.

| n | ell | prec | verdict | refresh | B2M | M2M+L2L | M2L | near+L2B | fin+eu | dominant |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 3 | fp16 | 1.926 | 0.351 | 0.066 | 1.155 | 0.192 | 0.032 | 0.046 | M2M+L2L (per-level launches) (60.0%) |
| 1000 | 3 | f64 | 1.97 | 0.351 | 0.106 | 1.143 | 0.252 | 0.041 | 0.048 | M2M+L2L (per-level launches) (58.0%) |
| 1000 | 4 | fp16 | 2.495 | 0.406 | 0.047 | 1.516 | 0.281 | 0.028 | 0.046 | M2M+L2L (per-level launches) (61.0%) |
| 1000 | 4 | f64 | 2.58 | 0.401 | 0.073 | 1.539 | 0.353 | 0.036 | 0.051 | M2M+L2L (per-level launches) (60.0%) |
| 1000 | 5 | fp16 | 2.989 | 0.425 | 0.03 | 1.915 | 0.376 | 0.026 | 0.045 | M2M+L2L (per-level launches) (64.0%) |
| 1000 | 5 | f64 | 3.122 | 0.43 | 0.046 | 1.925 | 0.439 | 0.034 | 0.049 | M2M+L2L (per-level launches) (62.0%) |
| 3162 | 3 | fp16 | 2.043 | 0.4 | 0.141 | 1.16 | 0.195 | 0.037 | 0.045 | M2M+L2L (per-level launches) (57.0%) |
| 3162 | 3 | f64 | 1.985 | 0.378 | 0.23 | 1.146 | 0.274 | 0.053 | 0.05 | M2M+L2L (per-level launches) (58.0%) |
| 3162 | 4 | fp16 | 2.714 | 0.565 | 0.056 | 1.507 | 0.352 | 0.057 | 0.044 | M2M+L2L (per-level launches) (56.0%) |
| 3162 | 4 | f64 | 2.915 | 0.567 | 0.091 | 1.542 | 0.629 | 0.077 | 0.049 | M2M+L2L (per-level launches) (53.0%) |
| 3162 | 5 | fp16 | 3.355 | 0.626 | 0.047 | 1.892 | 0.524 | 0.044 | 0.046 | M2M+L2L (per-level launches) (56.0%) |
| 3162 | 5 | f64 | 3.726 | 0.639 | 0.074 | 1.935 | 0.88 | 0.059 | 0.048 | M2M+L2L (per-level launches) (52.0%) |
| 10000 | 3 | fp16 | 2.009 | 0.384 | 0.331 | 1.128 | 0.193 | 0.063 | 0.045 | M2M+L2L (per-level launches) (56.0%) |
| 10000 | 3 | f64 | 2.061 | 0.394 | 0.546 | 1.138 | 0.273 | 0.102 | 0.048 | M2M+L2L (per-level launches) (55.0%) |
| 10000 | 4 | fp16 | 2.83 | 0.583 | 0.093 | 1.525 | 0.465 | 0.104 | 0.046 | M2M+L2L (per-level launches) (54.0%) |
| 10000 | 4 | f64 | 2.985 | 0.57 | 0.153 | 1.531 | 0.87 | 0.16 | 0.048 | M2M+L2L (per-level launches) (51.0%) |
| 10000 | 5 | fp16 | 3.744 | 0.63 | 0.057 | 1.914 | 0.872 | 0.113 | 0.046 | M2M+L2L (per-level launches) (51.0%) |
| 10000 | 5 | f64 | 4.212 | 0.622 | 0.09 | 1.921 | 1.557 | 0.169 | 0.049 | M2M+L2L (per-level launches) (46.0%) |
| 31623 | 3 | fp16 | 2.481 | 0.392 | 0.781 | 1.141 | 0.19 | 0.216 | 0.045 | M2M+L2L (per-level launches) (46.0%) |
| 31623 | 3 | f64 | 3.273 | 0.392 | 1.414 | 1.148 | 0.277 | 0.406 | 0.052 | B2M (43.0%) |
| 31623 | 4 | fp16 | 2.857 | 0.575 | 0.186 | 1.516 | 0.493 | 0.157 | 0.046 | M2M+L2L (per-level launches) (53.0%) |
| 31623 | 4 | f64 | 3.136 | 0.566 | 0.305 | 1.532 | 0.943 | 0.29 | 0.049 | M2M+L2L (per-level launches) (49.0%) |
| 31623 | 5 | fp16 | 4.339 | 0.654 | 0.069 | 1.91 | 1.576 | 0.328 | 0.046 | M2M+L2L (per-level launches) (44.0%) |
| 31623 | 5 | f64 | 6.046 | 0.674 | 0.111 | 1.978 | 3.147 | 0.519 | 0.05 | M2L (incl. window gen) (52.0%) |
| 100000 | 3 | fp16 | 5.542 | 0.386 | 2.561 | 1.147 | 0.195 | 1.482 | 0.05 | B2M (46.0%) |
| 100000 | 3 | f64 | 8.606 | 0.406 | 4.125 | 1.155 | 0.275 | 3.0 | 0.055 | B2M (48.0%) |
| 100000 | 4 | fp16 | 3.13 | 0.591 | 0.415 | 1.534 | 0.494 | 0.353 | 0.05 | M2M+L2L (per-level launches) (49.0%) |
| 100000 | 4 | f64 | 3.63 | 0.558 | 0.687 | 1.542 | 0.942 | 0.717 | 0.056 | M2M+L2L (per-level launches) (42.0%) |
| 100000 | 5 | fp16 | 5.479 | 0.69 | 0.145 | 1.914 | 2.408 | 0.738 | 0.049 | M2L (incl. window gen) (44.0%) |
| 100000 | 5 | f64 | 9.176 | 0.673 | 0.235 | 1.93 | 5.402 | 1.437 | 0.06 | M2L (incl. window gen) (59.0%) |
| 316228 | 3 | fp16 | 22.435 | 0.485 | 7.415 | 1.135 | 0.192 | 13.311 | 0.075 | nearfield+L2B (59.0%) |
| 316228 | 3 | f64 | 56.616 | 0.503 | 11.923 | 1.134 | 0.274 | 28.556 | 0.091 | nearfield+L2B (50.0%) |
| 316228 | 4 | fp16 | 5.655 | 0.613 | 1.097 | 1.527 | 0.495 | 2.289 | 0.082 | nearfield+L2B (40.0%) |
| 316228 | 4 | f64 | 9.047 | 0.62 | 1.949 | 1.53 | 0.942 | 4.757 | 0.148 | nearfield+L2B (53.0%) |
| 316228 | 5 | fp16 | 6.25 | 0.728 | 0.255 | 1.922 | 2.567 | 1.314 | 0.078 | M2L (incl. window gen) (41.0%) |
| 316228 | 5 | f64 | 11.301 | 0.719 | 0.422 | 1.929 | 5.805 | 3.1 | 0.146 | M2L (incl. window gen) (51.0%) |
| 1000000 | 3 | fp16 | 213.878 | 0.919 | 22.654 | 1.141 | 0.193 | 143.959 | 0.258 | nearfield+L2B (67.0%) |
| 1000000 | 3 | f64 | 582.508 | 1.019 | 36.523 | 1.141 | 0.277 | 391.755 | 0.424 | nearfield+L2B (67.0%) |
| 1000000 | 4 | fp16 | 25.046 | 0.876 | 3.333 | 1.521 | 0.49 | 19.118 | 0.254 | nearfield+L2B (76.0%) |
| 1000000 | 4 | f64 | 65.175 | 0.944 | 5.362 | 1.538 | 0.94 | 41.817 | 0.424 | nearfield+L2B (64.0%) |
| 1000000 | 5 | fp16 | 9.591 | 0.961 | 0.586 | 1.914 | 2.544 | 4.243 | 0.254 | nearfield+L2B (44.0%) |
| 1000000 | 5 | f64 | 20.74 | 1.031 | 1.02 | 1.942 | 5.797 | 11.707 | 0.424 | nearfield+L2B (56.0%) |

## Per-n recommendations vs the shipped ell=5 default

Accuracy target: 0.00119 gradient relative RMS (the unchanged 028 gate).

| n | prec | shipped ell=5 | best admissible | saving | accuracy | evidence |
|---|---|---|---|---|---|---|
| 1000 | fp16 | 2.989 ms (off-target) | ell=3, 1.926 ms | 1.063 ms (36.0%) | **fixes** off-target default (1.4x -> 0.37x) | measured |
| 1000 | f64 | 3.122 ms (off-target) | ell=3, 1.97 ms | 1.152 ms (37.0%) | **fixes** off-target default (1.4x -> 0.31x) | measured |
| 3162 | fp16 | 3.355 ms (off-target) | ell=3, 2.043 ms | 1.312 ms (39.0%) | **fixes** off-target default (1.5x -> 0.85x) | measured |
| 3162 | f64 | 3.726 ms (off-target) | ell=3, 1.985 ms | 1.741 ms (47.0%) | **fixes** off-target default (1.5x -> 0.62x) | measured |
| 10000 | fp16 | 3.744 ms (**off-target** 1.36x) | none admissible | — | — | measured |
| 10000 | f64 | 4.212 ms (off-target) | ell=3, 2.061 ms | 2.151 ms (51.0%) | **fixes** off-target default (1.36x -> 0.8x) | measured |
| 31623 | fp16 | 4.339 ms (off-target) | ell=4, 2.857 ms | 1.482 ms (34.0%) | **fixes** off-target default (1.04x -> 0.94x) | measured |
| 31623 | f64 | 6.046 ms (off-target) | ell=4, 3.136 ms | 2.911 ms (48.0%) | **fixes** off-target default (1.03x -> 0.91x) | measured |
| 100000 | fp16 | 5.479 ms | ell=4, 3.13 ms | 2.349 ms (43.0%) | 0.89x target | measured |
| 100000 | f64 | 9.176 ms | ell=4, 3.63 ms | 5.546 ms (60.0%) | 0.86x target | measured |
| 316228 | fp16 | 6.25 ms | ell=4, 5.655 ms | 0.594 ms (10.0%) | 0.85x target | measured |
| 316228 | f64 | 11.301 ms | ell=4, 9.047 ms | 2.254 ms (20.0%) | 0.8x target | measured |
| 1000000 | fp16 | 9.591 ms | keep ell=5, 9.591 ms | — | 0.89x target | measured |
| 1000000 | f64 | 20.74 ms | keep ell=5, 20.74 ms | — | 0.88x target | measured |

## Best admissible configuration at each n, across both precisions

Shipped default is `ell=5` + FP16 (the `expansion_order=3` rule).

| n | shipped default | best admissible anywhere | speedup | FP16 admissible at any depth? |
|---|---|---|---|---|
| 1000 | 2.989 ms (off-target 1.4x) | ell=3 fp16, 1.926 ms (0.37x target) | 1.55x | yes |
| 3162 | 3.355 ms (off-target 1.5x) | ell=3 f64, 1.985 ms (0.62x target) | 1.69x | yes |
| 10000 | 3.744 ms (off-target 1.36x) | ell=3 f64, 2.061 ms (0.8x target) | 1.82x | **no** |
| 31623 | 4.339 ms (off-target 1.04x) | ell=4 fp16, 2.857 ms (0.94x target) | 1.52x | yes |
| 100000 | 5.479 ms | ell=4 fp16, 3.13 ms (0.89x target) | 1.75x | yes |
| 316228 | 6.25 ms | ell=4 fp16, 5.655 ms (0.85x target) | 1.11x | yes |
| 1000000 | 9.591 ms | ell=5 fp16, 9.591 ms (0.89x target) | 1.0x | yes |

## FP16 arithmetic penalty vs depth

`E_arith = sqrt(err_fp16^2 - err_f64^2)`, from matched (n, ell) pairs.

| n | ell=3 | ell=4 | ell=5 |
|---|---|---|---|
| 1000 | 0.000228 | 4.63e-5 | 4.89e-5 |
| 3162 | 0.000694 | 0.000168 | 8.62e-5 |
| 10000 | 0.000983 | 0.000288 | 9.14e-5 |
| 31623 | 0.00106 | 0.000256 | 9.35e-5 |
| 100000 | 0.00111 | 0.000289 | 0.000137 |
| 316228 | 0.00124 | 0.000339 | 0.000104 |
| 1000000 | 0.00116 | 0.000239 | 0.000142 |

The penalty grows as the grid coarsens: at `ell=5` it is at or below the
sampling resolution, while at `ell=3` it is comparable to the whole error
budget. Coarse depth concentrates more source mass per route and widens the
operator dynamic range, which is what the FP16 input format cannot hold — the
scale-invariance caveat recorded in the 028 review, now quantified against `ell`.

wrote /Users/ryan/Dropbox/research/projects/tmp3/FastMultipole/MATRIX_OPERATOR_REFACTOR/data/cost_vs_n/recommendations.csv
