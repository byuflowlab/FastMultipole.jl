# Radix-Sort Clustering Verification Summary

- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/radix_sort_clustering_verify.jl`
- Status: `PASS`
- Key construction: `PASS`
- Boundary quantization: `PASS`
- Root domain: center `(0, 0, 0)`, half-width `1`

## Cases

| Case | Points | Level | Occupied cells | Keys sorted | Stable ties | Ranges compressed | Inverse round trip | Geometry | Offsets | Status |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| grid-aligned | 4 | 2 | 4 | `true` | `true` | `true` | `true` | `true` | `true` | `PASS` |
| boundary/tie | 6 | 2 | 4 | `true` | `true` | `true` | `true` | `true` | `true` | `PASS` |
| clustered | 6 | 4 | 3 | `true` | `true` | `true` | `true` | `true` | `true` | `PASS` |
| sparse occupancy | 4 | 5 | 4 | `true` | `true` | `true` | `true` | `true` | `true` | `PASS` |
| fixed-seed random | 64 | 4 | 63 | `true` | `true` | `true` | `true` | `true` | `true` | `PASS` |
