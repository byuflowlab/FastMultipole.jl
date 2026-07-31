# Hierarchical rigid stencil verification

Generated deterministically by `hierarchical_rigid_stencil_verify.jl`.

- near/V/union counts: q=3 -> 27/189/316; q=12 -> 179/1253/1740
- level-1/2 and downward monotonicity checks: PASS (minimum child norms 9 and 34)
- exact-once ordered-pair coverage: 8 cases PASS
- production dense scalar and Lamb–Helmholtz scaling: 32 cases PASS at rtol 1e-13
- campaign audit: occupancy reconstructed exactly from seed 24025; route comparisons are labeled model estimates because 024b CSVs contain no route telemetry
