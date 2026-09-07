# Codebase Map — Higher Derivatives

Refreshed 2026-09-07 from a repository-wide census. Verify locations immediately before
editing; inspect connected interfaces when this map is stale.

## Switch and layouts

- `src/containers.jl`: `DerivativesSwitch`, probe structs, and `ThirdDerivativeTensor`.
- `src/derivativesswitch.jl`: constructors and compact ranges. Standard widths are 1, 3,
  9, and 18; no-switch output rows are 4, 5:7, 8:16, and 17:34.
- `src/compatibility.jl`: matrix getters/setters and the capability trait.
- `src/tree.jl`: target allocation and derivative-coefficient scratch.
- `src/FastMultipole.jl`: public exports.

## Public execution paths

- `src/direct.jl`: standalone single/multithread keywords, switch creation, cached direct.
- `src/fmm.jl`: allocating/cache/plan entry points, structural keywords, rotation checks,
  near-field execution, and Radix guards.
- `src/nearfield_cache.jl`: derives output rows from `output_range(switch)`; construction,
  reuse, mismatch, and estimates need explicit third-order coverage.
- `src/solve.jl` and `src/error.jl`: internal switch construction only; `solve!` does not
  offer third-order output.

## Expansion evaluation

- `src/evaluate_expansions.jl`: complex and real L2B paths. Both form vector-field
  coefficients, differentiate again for Hessians, and require one further recurrence for
  third derivatives. LH coefficients use the same spatial recurrence after their initial
  scalar/vector coupling.
- `src/fmm.jl`: allocates worker-local scratch for the downward pass.

## Reference systems and probes

- `test/gravitational.jl`: scalar point-source direct formulas, result storage/writeback,
  metadata, and capability opt-in.
- `test/vortex.jl`: point-vortex velocity, nonsymmetric 9-component gradient, second
  spatial derivatives, writeback, and capability opt-in.
- `src/probes.jl`: static/array storage, reset, and writeback; four-field constructors remain
  compatible.

## Documentation and tests

- `docs/src/guided_examples.md`, `advanced_usage.md`, `advanced_usage_2.md`, and
  `device_interface.md` describe switches and layouts.
- `test/metadata_extra_test.jl`, `nearfield_cache_test.jl`, `fmm_plan_test.jl`,
  `transform_plan_test.jl`, and the higher-derivative tests cover interface behavior.

## GPU/Radix census boundary

The fixed layout is independent of `DerivativesSwitch`. G1 must inspect by body/kernel/route:
`translate_batched_cuda.jl`, `translate_batched_resident.jl`, `cross_stencil_cuda.jl`,
`direct_rectangular.jl`, and `interaction_list_batched.jl`. Existing cache output is 4 rows
through gradient or 13 through Hessian; the reviewed level-three target is 31 rows.
