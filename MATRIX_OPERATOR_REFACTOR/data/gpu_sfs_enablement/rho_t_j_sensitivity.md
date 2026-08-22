# Local `rho_t` J sensitivity

Command (run from the sibling `FLOWVPM.jl` repository):

```bash
FLOWVPM_JREF_RHO_SWEEP=1 \
FLOWVPM_JREF_RHO_CSV="../FastMultipole/MATRIX_OPERATOR_REFACTOR/data/gpu_sfs_enablement/rho_t_j_sensitivity.csv" \
julia --project=test test/runtests_unit_j_reference.jl
```

The deterministic case has 96 particles (`MersenneTwister(48117)`), Float64,
`P=8`, `q=20`, `ell=2`, `sigma=0.08`, and the partitioned direct kernel.
Only `rho_t` changes. The independent reference differentiates the mathematical
Gaussian-erf Biot-Savart velocity with ForwardDiff; it does not reuse the
production analytic J implementation.

The non-round sweep points come from the repository's epsilon=`1e-3` cutoff
theory: `3.668` is the U RMS radius, `4.211` the U per-pair radius, `4.252` the
J RMS radius, and `4.789` the J per-pair radius. The half-step points are kept
to show the overall sensitivity curve.

For target `i`, the reported target-relative error is

```text
norm(J_radix[i] - J_AD[i], Frobenius) /
max(norm(J_AD[i], Frobenius), sqrt(eps(Float64))*maximum_j norm(J_AD[j], Frobenius))
```

The CSV reports the maximum and p95 of this value over all 96 targets. Its
global relative RMS column is `norm(J_radix-J_AD)/norm(J_AD)` over the complete
9-by-96 arrays. The denominator floor in this run was
`1.7977706581871235e-8`.

Local run result: 15/15 sweep checks passed in 83 seconds. Of the sampled
points, the largest target-relative error first falls just below `1e-3` at the
theoretical U per-pair radius `rho_t=4.211`; the J RMS radius `4.252` retains
the same worst target while lowering global RMS and p95 error. The theoretical
J per-pair radius `4.789` is close to the error floor, and improvement is
effectively saturated by `rho_t=5.0` on this case.
