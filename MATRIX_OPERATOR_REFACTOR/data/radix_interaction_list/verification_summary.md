# Radix-Path M2L Interaction-List Verification Summary

- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/radix_interaction_list_verify.jl`
- Status: `PASS`
- No expansion translations performed; cell/pair identification and coverage only.
- Root domain: center `(0, 0, 0)`, half-width `1`; source and target share one grid.

## Cases

| Case | LH | Points | Level | P | eps | Cells | Accepted offsets | M2L batches | Far | Near | Self | Cell partition | Stencil agreement | Body coverage ==1 | Grid sharing | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| grid-aligned | `false` | 6 | 3 | 4 | 1 | 6 | 3318 | 30 | 30 | 0 | 6 | `true` | `true` | `true` | `true` | `PASS` |
| clustered | `false` | 7 | 4 | 4 | 1 | 4 | 29710 | 8 | 12 | 0 | 4 | `true` | `true` | `true` | `true` | `PASS` |
| sparse occupancy | `false` | 4 | 3 | 4 | 1 | 4 | 3318 | 12 | 12 | 0 | 4 | `true` | `true` | `true` | `true` | `PASS` |
| fixed-seed random | `false` | 64 | 3 | 4 | 1 | 60 | 3318 | 1516 | 3198 | 342 | 60 | `true` | `true` | `true` | `true` | `PASS` |
| fixed-seed random (LH) | `true` | 64 | 3 | 4 | 1 | 60 | 3294 | 1492 | 3082 | 458 | 60 | `true` | `true` | `true` | `true` | `PASS` |
| grid-aligned (LH) | `true` | 6 | 3 | 4 | 1 | 6 | 3294 | 30 | 30 | 0 | 6 | `true` | `true` | `true` | `true` | `PASS` |

Coverage target: for each case the far/near/self sets partition every ordered
occupied cell pair exactly once, the swept M2L batches equal the classified far
set, and every ordered body pair is covered exactly once (count == 1).
