# Coefficient Buffer Layout Verification Summary

- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/coefficient_buffer_layout_verify.jl`
- Status: `PASS`
- Expansion orders: `0`, `1`, `3`, `6`, `9`
- Layouts: `Val(false)`, `Val(true)`
- Batch counts: `1`, `2`, `5`
- Max complex legacy/native round-trip error: `0.0`
- Fixed-channel slab requirement: `stride(view, 1) == 1`, `stride(view, 2) == basis_dof`
- Production `harmonic_index` cross-check: `PASS`
- Production `flat_basis_index` cross-check: `PASS`

## Compressed Complex Cases

| P | Layout | Batch count | Channels | Basis dof | Max round-trip error | Indices contiguous/unique | Slabs dense | Slab stride 1 | Slab stride 2 |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 0 | `Val(false)` | 1 | 1 | 2 | 0.0 | `true` | `true` | `1` | `2` |
| 0 | `Val(false)` | 2 | 1 | 2 | 0.0 | `true` | `true` | `1` | `2` |
| 0 | `Val(false)` | 5 | 1 | 2 | 0.0 | `true` | `true` | `1` | `2` |
| 0 | `Val(true)` | 1 | 2 | 2 | 0.0 | `true` | `true` | `1,1` | `2,2` |
| 0 | `Val(true)` | 2 | 2 | 2 | 0.0 | `true` | `true` | `1,1` | `2,2` |
| 0 | `Val(true)` | 5 | 2 | 2 | 0.0 | `true` | `true` | `1,1` | `2,2` |
| 1 | `Val(false)` | 1 | 1 | 6 | 0.0 | `true` | `true` | `1` | `6` |
| 1 | `Val(false)` | 2 | 1 | 6 | 0.0 | `true` | `true` | `1` | `6` |
| 1 | `Val(false)` | 5 | 1 | 6 | 0.0 | `true` | `true` | `1` | `6` |
| 1 | `Val(true)` | 1 | 2 | 6 | 0.0 | `true` | `true` | `1,1` | `6,6` |
| 1 | `Val(true)` | 2 | 2 | 6 | 0.0 | `true` | `true` | `1,1` | `6,6` |
| 1 | `Val(true)` | 5 | 2 | 6 | 0.0 | `true` | `true` | `1,1` | `6,6` |
| 3 | `Val(false)` | 1 | 1 | 20 | 0.0 | `true` | `true` | `1` | `20` |
| 3 | `Val(false)` | 2 | 1 | 20 | 0.0 | `true` | `true` | `1` | `20` |
| 3 | `Val(false)` | 5 | 1 | 20 | 0.0 | `true` | `true` | `1` | `20` |
| 3 | `Val(true)` | 1 | 2 | 20 | 0.0 | `true` | `true` | `1,1` | `20,20` |
| 3 | `Val(true)` | 2 | 2 | 20 | 0.0 | `true` | `true` | `1,1` | `20,20` |
| 3 | `Val(true)` | 5 | 2 | 20 | 0.0 | `true` | `true` | `1,1` | `20,20` |
| 6 | `Val(false)` | 1 | 1 | 56 | 0.0 | `true` | `true` | `1` | `56` |
| 6 | `Val(false)` | 2 | 1 | 56 | 0.0 | `true` | `true` | `1` | `56` |
| 6 | `Val(false)` | 5 | 1 | 56 | 0.0 | `true` | `true` | `1` | `56` |
| 6 | `Val(true)` | 1 | 2 | 56 | 0.0 | `true` | `true` | `1,1` | `56,56` |
| 6 | `Val(true)` | 2 | 2 | 56 | 0.0 | `true` | `true` | `1,1` | `56,56` |
| 6 | `Val(true)` | 5 | 2 | 56 | 0.0 | `true` | `true` | `1,1` | `56,56` |
| 9 | `Val(false)` | 1 | 1 | 110 | 0.0 | `true` | `true` | `1` | `110` |
| 9 | `Val(false)` | 2 | 1 | 110 | 0.0 | `true` | `true` | `1` | `110` |
| 9 | `Val(false)` | 5 | 1 | 110 | 0.0 | `true` | `true` | `1` | `110` |
| 9 | `Val(true)` | 1 | 2 | 110 | 0.0 | `true` | `true` | `1,1` | `110,110` |
| 9 | `Val(true)` | 2 | 2 | 110 | 0.0 | `true` | `true` | `1,1` | `110,110` |
| 9 | `Val(true)` | 5 | 2 | 110 | 0.0 | `true` | `true` | `1,1` | `110,110` |

## Real Basis Cases

| P | Basis dof | Indices contiguous/unique | Degree blocks contiguous |
| ---: | ---: | --- | --- |
| 0 | 1 | `true` | `true` |
| 1 | 4 | `true` | `true` |
| 3 | 16 | `true` | `true` |
| 6 | 49 | `true` | `true` |
| 9 | 100 | `true` | `true` |
