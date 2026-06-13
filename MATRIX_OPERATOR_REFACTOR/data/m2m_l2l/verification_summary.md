# M2M and L2L Extension Verification Summary

- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/m2m_l2l_verify.jl`
- Operator tolerance: `atol <= 1.0e-9`, `rtol <= 2.0e-11`
- Z-block tolerance: finite entries compare with `rtol <= 1.0e-12`; absolute errors are reported for scale context.
- Point-mass chain convergence target: final `rtol <= 5.0e-7` in `1/r` normalization
- Status: `PASS`
- Max M2M absolute error: `5.417888360170764e-14`
- Max M2M relative error: `1.7094382274525772e-12`
- Max L2L absolute error: `7.460698725481052e-14`
- Max L2L relative error: `2.6067926229863967e-13`
- Max z-block absolute error: `262144.0`
- Max z-block relative error: `1.3682805004681272e-13`
- Final point-chain relative error: `8.556935334046337e-16`
- Coverage: axis-aligned `+z`, axis-aligned `-z`, positive off-axis, negative off-axis, layouts `Val(false)` and `Val(true)`, expansion orders `0`, `1`, `3`, `6`, `9`.

## Operator Cases

| Operator | Layout | Case | P | Offset | Max abs error | Max rel error |
| --- | --- | --- | ---: | --- | ---: | ---: |
| `M2M` | `Val(false)` | axis-aligned +z | 0 | `(0, 0, 1.75)` | 0.000000e+00 | 0.000000e+00 |
| `M2M` | `Val(true)` | axis-aligned +z | 0 | `(0, 0, 1.75)` | 0.000000e+00 | 0.000000e+00 |
| `M2M` | `Val(false)` | axis-aligned -z | 1 | `(0, 0, -2.25)` | 0.000000e+00 | 0.000000e+00 |
| `M2M` | `Val(true)` | axis-aligned -z | 1 | `(0, 0, -2.25)` | 0.000000e+00 | 0.000000e+00 |
| `M2M` | `Val(false)` | positive off-axis | 3 | `(1.7, 0.8, 2.4)` | 6.661338e-16 | 3.987473e-15 |
| `M2M` | `Val(true)` | positive off-axis | 3 | `(1.7, 0.8, 2.4)` | 1.776357e-15 | 4.213524e-15 |
| `M2M` | `Val(false)` | negative off-axis | 6 | `(-2.3, -0.9, 1.6)` | 4.440892e-15 | 1.146160e-14 |
| `M2M` | `Val(true)` | negative off-axis | 6 | `(-2.3, -0.9, 1.6)` | 6.661338e-15 | 1.709438e-12 |
| `M2M` | `Val(false)` | positive off-axis | 9 | `(2.8, -1.1, 3.3)` | 2.464695e-14 | 1.300126e-13 |
| `M2M` | `Val(true)` | positive off-axis | 9 | `(2.8, -1.1, 3.3)` | 5.417888e-14 | 1.587838e-13 |
| `L2L` | `Val(false)` | axis-aligned +z | 0 | `(0, 0, 1.75)` | 0.000000e+00 | 0.000000e+00 |
| `L2L` | `Val(true)` | axis-aligned +z | 0 | `(0, 0, 1.75)` | 0.000000e+00 | 0.000000e+00 |
| `L2L` | `Val(false)` | axis-aligned -z | 1 | `(0, 0, -2.25)` | 0.000000e+00 | 0.000000e+00 |
| `L2L` | `Val(true)` | axis-aligned -z | 1 | `(0, 0, -2.25)` | 0.000000e+00 | 0.000000e+00 |
| `L2L` | `Val(false)` | positive off-axis | 3 | `(1.7, 0.8, 2.4)` | 4.440892e-16 | 4.709746e-16 |
| `L2L` | `Val(true)` | positive off-axis | 3 | `(1.7, 0.8, 2.4)` | 1.776357e-15 | 5.802350e-14 |
| `L2L` | `Val(false)` | negative off-axis | 6 | `(-2.3, -0.9, 1.6)` | 7.327472e-15 | 3.754367e-14 |
| `L2L` | `Val(true)` | negative off-axis | 6 | `(-2.3, -0.9, 1.6)` | 1.110223e-14 | 2.273591e-13 |
| `L2L` | `Val(false)` | positive off-axis | 9 | `(2.8, -1.1, 3.3)` | 3.907985e-14 | 6.139199e-14 |
| `L2L` | `Val(true)` | positive off-axis | 9 | `(2.8, -1.1, 3.3)` | 7.460699e-14 | 2.606793e-13 |

## Z-Translation Blocks

| P | t | M2M abs | M2M rel | L2L abs | L2L rel |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.001 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 0 | 0.25 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 0 | 2 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 0 | 1000 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 1 | 0.001 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 1 | 0.25 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 1 | 2 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 1 | 1000 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 3 | 0.001 | 1.110223e-16 | 1.645996e-16 | 0.000000e+00 | 0.000000e+00 |
| 3 | 0.25 | 5.551115e-17 | 1.385182e-16 | 0.000000e+00 | 0.000000e+00 |
| 3 | 2 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 3 | 1000 | 7.450581e-09 | 1.899993e-16 | 0.000000e+00 | 0.000000e+00 |
| 6 | 0.001 | 1.110223e-16 | 2.142693e-16 | 0.000000e+00 | 0.000000e+00 |
| 6 | 0.25 | 1.110223e-16 | 2.216172e-16 | 0.000000e+00 | 0.000000e+00 |
| 6 | 2 | 9.020562e-17 | 7.386057e-15 | 0.000000e+00 | 0.000000e+00 |
| 6 | 1000 | 1.250000e-01 | 1.963114e-16 | 0.000000e+00 | 0.000000e+00 |
| 9 | 0.001 | 2.220446e-16 | 2.176444e-16 | 0.000000e+00 | 0.000000e+00 |
| 9 | 0.25 | 2.220446e-16 | 3.056085e-16 | 0.000000e+00 | 0.000000e+00 |
| 9 | 2 | 1.387779e-16 | 1.368281e-13 | 0.000000e+00 | 0.000000e+00 |
| 9 | 1000 | 2.621440e+05 | 3.038420e-16 | 0.000000e+00 | 0.000000e+00 |

## Unit Point-Mass M2M-M2L-L2L Chain

Production scalar-potential results were multiplied by `-4π` before comparison so this table uses the theory `1/r` normalization.

| P | Chain local evaluation | Analytic `1/r` | Abs error | Rel error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5.1091610452638725e-01 | 5.1898161259104103e-01 | 8.065508e-03 | 1.554103e-02 |
| 1 | 5.1950072793688573e-01 | 5.1898161259104103e-01 | 5.191153e-04 | 1.000258e-03 |
| 2 | 5.1899209588530271e-01 | 5.1898161259104103e-01 | 1.048329e-05 | 2.019974e-05 |
| 3 | 5.1898121248099571e-01 | 5.1898161259104103e-01 | 4.001100e-07 | 7.709523e-07 |
| 5 | 5.1898161301938872e-01 | 5.1898161259104103e-01 | 4.283477e-10 | 8.253620e-10 |
| 7 | 5.1898161259049813e-01 | 5.1898161259104103e-01 | 5.428991e-13 | 1.046085e-12 |
| 9 | 5.1898161259104203e-01 | 5.1898161259104103e-01 | 9.992007e-16 | 1.925310e-15 |
| 12 | 5.1898161259104147e-01 | 5.1898161259104103e-01 | 4.440892e-16 | 8.556935e-16 |
