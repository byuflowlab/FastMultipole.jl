# Hierarchical rigid stencil verification

Generated deterministically by `hierarchical_rigid_stencil_verify.jl`.

- supported shells: `3, 4, 5, 6, 8, 9, 10, 11, 12`; near/V/union counts are recorded per shell
- level-1/2 and downward monotonicity checks: PASS for every supported shell
- exact-once ordered-pair coverage: 36 cases PASS
- production dense scalar and Lamb–Helmholtz scaling: 32 cases PASS at rtol 1e-13
- campaign audit: occupancy reconstructed exactly from seed 24025; route comparisons are labeled model estimates because 024b CSVs contain no route telemetry
