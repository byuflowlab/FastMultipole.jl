# Constant-`P` Error Stencil Verification Summary

- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/constant_p_error_stencil_verify.jl`
- Status: `PASS`
- Geometry mapping: `PASS`
- `c <= 2` rejection: `PASS`
- Analytic/generated stencil agreement: `PASS`
- Monotonicity: `PASS`
- Paths covered: `Val(false)`, `Val(true)`

## Agreement Case

| P | Epsilon | Max abs offset | Scalar accepted | LH accepted |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1.000000e-03 | 6 | 1946 | 1760 |

## Monotonicity Case

| Path | Base accepted | Larger P accepted | Looser epsilon accepted |
| --- | ---: | ---: | ---: |
| `Val(false)` | 2138 | 3124 | 2788 |
| `Val(true)` | 104 | 2986 | 1584 |
