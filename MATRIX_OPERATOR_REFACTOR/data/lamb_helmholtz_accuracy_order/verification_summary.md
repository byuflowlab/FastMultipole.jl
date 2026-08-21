# Lamb-Helmholtz Accuracy Order Verification Summary

- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/lamb_helmholtz_accuracy_order_verify.jl`
- Status: `PASS`
- Final rule: `P_chi = P_phi + 1` for `Val(true)` M2L/evaluation; `Val(false)` remains single-order `P`.
- Tested offsets: `P_chi = P_phi + delta`, delta in `[-1, 0, 1, 2]`.
- Configurations: 3 deterministic source coefficient clouds, 2 source centers, 3 directions, 3 separation ratios, 2 target offsets.
- Scalar `Val(false)` unchanged behavior: `PASS`

## Convergence Slopes

Slopes are least-squares slopes of `log(rms error)` versus `P_phi`; more negative is faster convergence.

| Case | Slope |
| --- | ---: |
| `chi_only` | -0.646844 |
| `delta_-1` | -1.659254 |
| `delta_0` | -1.747962 |
| `delta_1` | -1.781671 |
| `delta_2` | -1.797323 |
| `phi_only` | -1.466331 |

## Aggregate Errors

| Case | P_phi | Count | Max error | RMS error | Median error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chi_only` | 4 | 108 | 5.239486e-13 | 1.431027e-13 | 2.175019e-14 |
| `chi_only` | 6 | 108 | 8.203122e-17 | 1.432864e-17 | 1.167386e-18 |
| `chi_only` | 8 | 108 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `chi_only` | 10 | 108 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `chi_only` | 12 | 108 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `delta_-1` | 4 | 108 | 2.196990e-05 | 8.234406e-06 | 3.078479e-06 |
| `delta_-1` | 6 | 108 | 9.682673e-07 | 2.334516e-07 | 2.065120e-08 |
| `delta_-1` | 8 | 108 | 1.816449e-08 | 4.879126e-09 | 2.950604e-10 |
| `delta_-1` | 10 | 108 | 2.565686e-09 | 4.887037e-10 | 2.732300e-12 |
| `delta_-1` | 12 | 108 | 5.836900e-11 | 1.119847e-11 | 6.775929e-14 |
| `delta_0` | 4 | 108 | 5.438645e-06 | 1.521162e-06 | 2.919559e-07 |
| `delta_0` | 6 | 108 | 4.760010e-07 | 1.045692e-07 | 6.682278e-09 |
| `delta_0` | 8 | 108 | 7.328371e-09 | 1.484694e-09 | 2.083955e-11 |
| `delta_0` | 10 | 108 | 3.322971e-10 | 7.087195e-11 | 4.835131e-13 |
| `delta_0` | 12 | 108 | 8.275867e-12 | 1.497399e-12 | 3.036319e-15 |
| `delta_1` | 4 | 108 | 6.317705e-06 | 1.558580e-06 | 1.986285e-07 |
| `delta_1` | 6 | 108 | 1.828408e-07 | 4.025178e-08 | 2.510258e-09 |
| `delta_1` | 8 | 108 | 6.905409e-09 | 1.340013e-09 | 1.921251e-11 |
| `delta_1` | 10 | 108 | 2.255523e-10 | 4.381842e-11 | 3.581800e-13 |
| `delta_1` | 12 | 108 | 4.648642e-12 | 8.641607e-13 | 1.260949e-15 |
| `delta_2` | 4 | 108 | 5.863443e-06 | 1.486998e-06 | 2.015090e-07 |
| `delta_2` | 6 | 108 | 1.835278e-07 | 4.018803e-08 | 2.348299e-09 |
| `delta_2` | 8 | 108 | 5.705200e-09 | 1.127663e-09 | 1.628541e-11 |
| `delta_2` | 10 | 108 | 2.067512e-10 | 4.029009e-11 | 3.685919e-13 |
| `delta_2` | 12 | 108 | 3.968680e-12 | 7.346575e-13 | 9.391986e-16 |
| `phi_only` | 4 | 108 | 3.151789e-10 | 7.788120e-11 | 1.209448e-11 |
| `phi_only` | 6 | 108 | 4.235392e-14 | 9.836629e-15 | 6.971941e-16 |
| `phi_only` | 8 | 108 | 5.545617e-18 | 1.301381e-18 | 0.000000e+00 |
| `phi_only` | 10 | 108 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `phi_only` | 12 | 108 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |

## Decision Checks

- At `P_phi = 12`, `delta_1` RMS / `delta_0` RMS = `5.771079e-01`.
- At `P_phi = 12`, `delta_2` RMS / `delta_1` RMS = `8.501400e-01`.
- `delta_1` slope improvement over `delta_0`: `3.370896e-02`.
- At `P_phi = 12`, isolated `chi_only(P_chi=P_phi)` RMS = `0.000000e+00`.
- The `phi_only` and `chi_only` rows are diagnostic isolated-tail rows; the pass decision is based on paired M2L policies.
