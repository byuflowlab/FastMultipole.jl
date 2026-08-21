# M2L Z-Translation Verification Summary

- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/m2l_z_translation_verify.jl`
- Tolerance: `atol <= 1.0e-12`, `rtol <= 1.0e-12`
- Status: `PASS`
- Max absolute error: `3.469446951953614e-18`
- Max relative error: `4.823830462317249e-16`
- Max inactive-channel overwrite error: `0.0`

| Layout | P | t | Max abs error | Max rel error | Inactive channel |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Val(false)` | 0 | 1.75 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(true)` | 0 | 1.75 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(false)` | 1 | 2.25 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(true)` | 1 | 2.25 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(false)` | 3 | 4.5 | 3.469447e-18 | 2.098912e-16 | 0.000000e+00 |
| `Val(true)` | 3 | 4.5 | 3.469447e-18 | 2.098912e-16 | 0.000000e+00 |
| `Val(false)` | 6 | 8 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(true)` | 6 | 8 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(false)` | 9 | 13 | 5.082198e-21 | 4.823830e-16 | 0.000000e+00 |
| `Val(true)` | 9 | 13 | 5.082198e-21 | 4.823830e-16 | 0.000000e+00 |
