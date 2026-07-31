# Compact CUDA Factored-Plan Storage

## Summary

Remove heavyweight per-offset `ResidentOperatorGroup` construction from the CUDA
factored cache while leaving the optimized whole-pass algorithm and the `023a`
host implementation unchanged.

## Implementation Changes

- Keep full groups on the host; construct no per-class groups for CUDA factored
  plans.
- Drive CUDA class ranges from `host_class_counts`, `class_starts`, and compact
  `phi`, `theta`, `r`, and `z_flat` tables.
- Preserve the independent per-class CUDA reference using the existing
  scalar-angle kernels, shared scratch, and unit Lamb-Helmholtz rows scaled by
  class `r`.
- Leave the optimized whole-pass execution and public API unchanged.
- Update residency checks and benchmarks for the compact representation.

## Correctness and Performance Tests

- Validate every class's stored `phi`, `theta`, and `r` against its Cartesian
  offset and compare every `z_flat` column with a fresh `m2l_z_blocks!` result.
- Verify emitted `route_class` values, histograms, prefix starts, route ranges,
  sources, and targets exactly match the host route builder after construction
  and moving-body updates, including empty classes and partial final chunks.
- Compare the CUDA per-class reference against `023a` at the M2L output-buffer
  level (`locals.phi` and `locals.chi`), before L2L/L2B can hide errors.
- Compare the CUDA whole-pass against both the per-class reference and host
  factored output for `P = 4, 8, 12`; Float32 and Float64; Lamb-Helmholtz on and
  off; empty routes; repeated steps; and capacity reuse.
- Retain end-to-end GPU-versus-`direct!` potential and gradient tolerances,
  transfer-counter assertions, zero M2L device allocation, and unchanged concat
  behavior.
- Run the complete local suite.
- Before submitting an H200 job, show the Slurm script to the user and obtain
  explicit permission.
- Run the full H200 lifecycle/integration suite and benchmark sweep after
  permission. Record corrected construction time, device footprint,
  M2L/full-step timing, and numerical error data.
- Update the `023b` implementation notes and leave clear-context approval to a
  different reviewer.

## Assumptions

- The per-class path prioritizes independent numerical validation over speed, so
  one scalar-fill launch per class is acceptable.
- H200 submission remains separately permission-gated.

## Fresh-Context Start

Read `MATRIX_OPERATOR_REFACTOR/START_HERE.md` first, then this plan and
`MATRIX_OPERATOR_REFACTOR/023b-impl-factored-resident-m2l-cuda.md`. Treat the
current worktree as authoritative, preserve unrelated user changes, and resume
implementation from the first incomplete item above.
