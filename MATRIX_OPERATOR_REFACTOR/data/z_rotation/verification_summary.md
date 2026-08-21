# Z-Rotation Verification Summary

- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/z_rotation_verify.jl`
- Tolerance: `atol <= 1.0e-12`
- Status: `PASS`
- Max forward error: `3.552713678800501e-15`
- Max back/inverse accumulation error: `3.1086244689504383e-15`
- Max `m = 0` forward identity error: `0.0`
- Max `m = 0` back accumulation error: `0.0`

| Layout | P | phi | Forward max error | Back max error | m=0 forward | m=0 back |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Val(false)` | 0 | 0 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(true)` | 0 | 0 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(false)` | 1 | 0.25 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(true)` | 1 | 0.25 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(false)` | 3 | -1.125 | 2.220446e-16 | 4.440892e-16 | 0.000000e+00 | 0.000000e+00 |
| `Val(true)` | 3 | -1.125 | 4.440892e-16 | 4.440892e-16 | 0.000000e+00 | 0.000000e+00 |
| `Val(false)` | 6 | 1.047197551196598 | 1.110223e-15 | 1.110223e-15 | 0.000000e+00 | 0.000000e+00 |
| `Val(true)` | 6 | 1.047197551196598 | 1.110223e-15 | 1.110223e-15 | 0.000000e+00 | 0.000000e+00 |
| `Val(false)` | 9 | 2.4 | 3.330669e-15 | 2.664535e-15 | 0.000000e+00 | 0.000000e+00 |
| `Val(true)` | 9 | 2.4 | 3.552714e-15 | 3.108624e-15 | 0.000000e+00 | 0.000000e+00 |
