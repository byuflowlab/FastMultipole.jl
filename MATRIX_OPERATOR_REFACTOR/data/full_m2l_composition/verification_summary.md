# Full M2L Composition Verification Summary

- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/full_m2l_composition_verify.jl`
- Composition tolerance: `atol <= 1.0e-9`, `rtol <= 2.0e-11`
- Scaled-block tolerance: finite entries compare with `rtol <= 1.0e-12`; absolute errors are reported for scale context.
- Point-mass convergence target: final `rtol <= 2.5e-7` in `1/r` normalization
- Status: `PASS`
- Max composition absolute error: `5.165929906070232e-10`
- Max composition relative error: `1.5897417339721547e-12`
- Max scaled-block absolute error: `2.3539131507700053e57`
- Max scaled-block relative error: `4.092898796994199e-16`
- Final point-mass relative error: `3.6155107851558025e-16`
- Coverage: axis-aligned `+z`, axis-aligned `-z`, positive off-axis, negative off-axis, layouts `Val(false)` and `Val(true)`, expansion orders `0`, `1`, `3`, `6`, `9`.

## Composition Cases

| Layout | Case | P | Offset | Max abs error | Max rel error |
| --- | --- | ---: | --- | ---: | ---: |
| `Val(false)` | axis-aligned +z | 0 | `(0, 0, 1.75)` | 0.000000e+00 | 0.000000e+00 |
| `Val(true)` | axis-aligned +z | 0 | `(0, 0, 1.75)` | 0.000000e+00 | 0.000000e+00 |
| `Val(false)` | axis-aligned -z | 1 | `(0, 0, -2.25)` | 0.000000e+00 | 0.000000e+00 |
| `Val(true)` | axis-aligned -z | 1 | `(0, 0, -2.25)` | 0.000000e+00 | 0.000000e+00 |
| `Val(false)` | positive off-axis | 3 | `(1.7, 0.8, 2.4)` | 2.775558e-16 | 1.042898e-15 |
| `Val(true)` | positive off-axis | 3 | `(1.7, 0.8, 2.4)` | 6.661338e-16 | 4.970341e-15 |
| `Val(false)` | negative off-axis | 6 | `(-2.3, -0.9, 1.6)` | 5.093170e-11 | 3.356601e-14 |
| `Val(true)` | negative off-axis | 6 | `(-2.3, -0.9, 1.6)` | 4.365575e-11 | 1.333631e-13 |
| `Val(false)` | positive off-axis | 9 | `(2.8, -1.1, 3.3)` | 1.673470e-10 | 3.576919e-13 |
| `Val(true)` | positive off-axis | 9 | `(2.8, -1.1, 3.3)` | 5.165930e-10 | 1.589742e-12 |

## Scaled M2L Blocks

| P | t | Finite entries | Max abs error | Max rel error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.001 | 1 | 0.000000e+00 | 0.000000e+00 |
| 0 | 0.25 | 1 | 0.000000e+00 | 0.000000e+00 |
| 0 | 2 | 1 | 0.000000e+00 | 0.000000e+00 |
| 0 | 1000 | 1 | 0.000000e+00 | 0.000000e+00 |
| 1 | 0.001 | 5 | 2.384186e-07 | 1.192093e-16 |
| 1 | 0.25 | 5 | 0.000000e+00 | 0.000000e+00 |
| 1 | 2 | 5 | 0.000000e+00 | 0.000000e+00 |
| 1 | 1000 | 5 | 0.000000e+00 | 0.000000e+00 |
| 3 | 0.001 | 30 | 1.342177e+08 | 1.864135e-16 |
| 3 | 0.25 | 30 | 0.000000e+00 | 0.000000e+00 |
| 3 | 2 | 30 | 0.000000e+00 | 0.000000e+00 |
| 3 | 1000 | 30 | 3.155444e-30 | 1.314768e-16 |
| 6 | 0.001 | 140 | 1.622593e+32 | 3.721561e-16 |
| 6 | 0.25 | 140 | 0.000000e+00 | 0.000000e+00 |
| 6 | 2 | 140 | 0.000000e+00 | 0.000000e+00 |
| 6 | 1000 | 140 | 3.155444e-30 | 1.314768e-16 |
| 9 | 0.001 | 385 | 2.353913e+57 | 4.092899e-16 |
| 9 | 0.25 | 385 | 0.000000e+00 | 0.000000e+00 |
| 9 | 2 | 385 | 0.000000e+00 | 0.000000e+00 |
| 9 | 1000 | 385 | 3.155444e-30 | 1.314768e-16 |

## Unit Point-Mass M2L Example

Production scalar-potential results were multiplied by `-4π` before comparison so this table uses the theory `1/r` normalization.

| P | M2L local evaluation | Analytic `1/r` | Abs error | Rel error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 6.1256389183168891e-01 | 6.1414449608802035e-01 | 1.580604e-03 | 2.573668e-03 |
| 1 | 6.1503856465385875e-01 | 6.1414449608802035e-01 | 8.940686e-04 | 1.455795e-03 |
| 2 | 6.1415246965956105e-01 | 6.1414449608802035e-01 | 7.973572e-06 | 1.298322e-05 |
| 3 | 6.1414351236224074e-01 | 6.1414449608802035e-01 | 9.837258e-07 | 1.601782e-06 |
| 5 | 6.1414449726941245e-01 | 6.1414449608802035e-01 | 1.181392e-09 | 1.923639e-09 |
| 7 | 6.1414449608662858e-01 | 6.1414449608802035e-01 | 1.391776e-12 | 2.266202e-12 |
| 9 | 6.1414449608802157e-01 | 6.1414449608802035e-01 | 1.221245e-15 | 1.988531e-15 |
| 12 | 6.1414449608802013e-01 | 6.1414449608802035e-01 | 2.220446e-16 | 3.615511e-16 |
