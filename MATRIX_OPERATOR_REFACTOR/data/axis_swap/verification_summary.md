# Axis-Swap Verification Summary

- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/axis_swap_verify.jl`
- Tolerance: `atol <= 1.0e-12`
- Status: `PASS`
- Max multipole-axis-swap error: `0.0`
- Max local-axis-swap error: `0.0`
- Max `T` reconstruction error: `0.0`
- Max inactive-channel reset error: `0.0`
- Max reset/back-rotation target-independence error: `0.0`
- Layout coverage: `Val(false)`, `Val(true)`
- Case coverage: axis-aligned `theta = 0`, `theta = pi`; off-axis positive and negative angles.

| Layout | Case | P | theta | Multipole error | Local error | T error | Inactive reset | Reset/back target independence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Val(false)` | axis-aligned +z | 0 | 0 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(true)` | axis-aligned +z | 0 | 0 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(false)` | axis-aligned +z | 4 | 0 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(true)` | axis-aligned +z | 4 | 0 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(false)` | axis-aligned -z | 5 | 3.141592653589793 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(true)` | axis-aligned -z | 5 | 3.141592653589793 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(false)` | off-axis positive | 6 | 0.4487989505128276 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(true)` | off-axis positive | 6 | 0.4487989505128276 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(false)` | off-axis negative | 7 | -1.256637061435917 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| `Val(true)` | off-axis negative | 7 | -1.256637061435917 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
