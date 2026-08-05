# 032 Implementation: Generalized Device-Resident System Interface in FastMultipole

## Status and Entry Gate

**Added by user request on `2026-08-04`.** Not started.

Entry gate: `031-integration-api-design.md` must be Done and clear-context
approved, **including its recorded user sign-off on the interface spec**. Do
not start from the spec draft alone.

## Objective

Implement the `031`-approved interface in FastMultipole `src/` so an external
device-resident system — FLOWVPM first — can drive the resident GPU lifecycle
with vector strengths, the Lamb-Helmholtz channel, and full hessian output,
with zero per-step allocation and no per-step body transfers.

Deliverables:

1. **Vector-strength / Lamb-Helmholtz body packing and device B2M**: extend
   the resident body layout beyond scalar `[x, y, z, radius, strength]` per
   the approved spec (vector `Γ`, smoothing radius `σ`, extensible extra
   states), with device B2M for vortex point sources and the Lamb-Helmholtz
   channel exercised end-to-end (B2M → M2M → M2L → L2L → L2B) on the device
   lifecycle.
   Implement the canonical all-`data_per_body`-row packed layout first. Add
   the 5-row core + side-buffer alternative only if a profile or bandwidth
   model predicts at least a 5% end-to-end U/J-solve improvement, and retain
   it only if measurement confirms that gain.
2. **9-component hessian output on the resident path**: remove the
   `hessian=true` throw; produce the full velocity-gradient/hessian block in
   the resident output buffer and through `buffer_to_target!`, for both the
   scalar-potential and Lamb-Helmholtz channels. No 6-component symmetric
   variant (user decision `2026-08-04`).
3. **First-class device-system API**: promote the `fm028_device_system.jl`
   pattern into `src/` as a documented, exported surface (residency trait,
   device `source_to_buffer!`/`buffer_to_target!` contracts, capacity
   pre-sizing with `max_n_bodies`), plus docs and a worked example.
4. **External-code connection guidelines**: a documentation artifact for
   consumers, covering both the device-resident coupling and the
   transfer-based fallback with the `031` decision framework.

**Nearfield scope note (user direction `2026-08-05`)**: `RegularizedVortex`
supports only the FLOWVPM default `gaussianerf` kernel (the sole
`CoreSpreading`-compatible kernel; `winckelmans` dropped). It borrows the
fused per-pair U+J math from `../FLOWVPM.jl/ext/FLOWVPMCUDAExt.jl:172-234`.
This single-pass strategy is
the **baseline**; the partitioned regularized/singular alternative and the final
default selection are row `032a` (derivation in `031a`). This row's scope is
otherwise unchanged.

**How to evaluate `g` (added by the `031`/`031a` approval review,
`2026-08-05`).** The spec's §5 sentence directing a straight port of FLOWVPM's
FDLIBM `custom_erf` predates `031a` §6.2, which shows the `erf` can be dropped
from the production nearfield **entirely**, in every candidate strategy:

- below `ρ ≈ 2` the `031a` §3 Horner series for `g` and `h = ρg' − 3g` is
  already erf-free (six terms Float32, ten Float64) and is required there
  anyway for cancellation safety;
- above `ρ ≈ 2` the retained result is `O(1)`, so `ḡ = 1 − g` is needed only to
  *absolute* tolerance. Writing `ḡ = e^{−ρ²/2}(Aρ + s(ρ))` with
  `s = erfc(ρ/√2)e^{ρ²/2}`, a degree-3 least-squares polynomial in `u = 1/ρ²`
  holds `|δḡ| = 2.1e-4` against the `3.69e-4` budget at `ε=1e-3`
  (`data/kernel_splitting/cheap_gbar_fit.csv`) — one hardware `exp` plus four
  FMAs, with `ρg' = Aρ³e^{−ρ²/2}` reusing the same exponential.

`RegularizedVortex` must therefore **benchmark the §6.2 erf-free form against
the `custom_erf` port and ship whichever is faster at equal measured
accuracy**, rather than porting `custom_erf` unconditionally. `031a` §6.1 ranks
this above the `032a` strategy choice, so it belongs in this row's baseline,
not in `032a`. If the erf-free form is shipped, no FDLIBM code is copied into
FastMultipole at all.
5. **Tests and regression gate**: parity tests against the legacy host path
   for gradient and hessian, Float64 and Float32, scalar and Lamb-Helmholtz,
   **including `P=4`**; and a re-run of the shipped `028`/`030` harness
   configuration demonstrating the scalar path is not regressed.

## Dependencies

- `031-integration-api-design.md`, Done and clear-context approved with user
  sign-off recorded.
- Transitively: the resident lifecycle rows (`022`, `023`, `023b`–`023f`,
  `026`, `027`) and the `028` shipped defaults.

## Mandatory Reading Gate

1. `START_HERE.md`, including the Integration Phase preamble.
2. `031-integration-api-design.md` in full, including the sign-off record.
3. `theory/kernel-splitting-nearfield.md` §§3, 5.1-5.2, 6.2 (approved with
   `031a`): the erf-free evaluation of `g`, the cancellation-safe small-`ρ`
   series `RegularizedVortex` needs regardless of strategy, and the near-set
   adequacy test this row must assert (spec §5 acceptance item).
4. The Implementation Code Placement rules in `START_HERE.md`.

## Task-Local Requirements

- Code placement follows the existing Implementation Code Placement rules:
  new types in `src/containers.jl`, `_batched`/`_cuda` file naming, CUDA code
  behind the extension/flag so the CPU path and public API are unaffected
  when CUDA is absent.
- The `023` residency counter contract must hold for a device-resident
  consumer: `body_uploads = 0`, `expansion_host_copies = 0`, route/operator
  uploads construction-only, zero steady-state allocation.
- All FastMultipole edits are committed in this repository (never in
  `../FLOWVPM.jl`); this row makes no FLOWVPM changes.
- H200 validation of the new packing/hessian kernels (correctness plus a
  before/after cost check on the `028` workload) is required before Done.

## Work Record

### Mandatory Reading Gate

Completed 2026-08-05 (staging session) and re-completed 2026-08-05 by the
continuing agent: `START_HERE.md` incl. Integration Phase preamble and
placement rules, `031-integration-api-design.md` in full incl. the sign-off
record, `integration-api-spec.md` in full, and
`theory/kernel-splitting-nearfield.md` §§1, 3, 5.1–5.2, 6.2 (plus 6.3 for the
032a divergence constraint context).

Execution plan with user decisions (staged with checkpoints; `recenter!`
fallback-first; adequacy gate rejects rather than enlarges):
`032-implementation-plan.md` (this directory).

### Stage 1 — generalized packing, vortex B2M + LH, 9-component hessian (Done)

Committed as `e61fb95` ("stage 1 of 032"). Delivered: canonical
all-`data_per_body`-row packed layout on both resident paths (radius row 4 now
live; pack kernels `translate_batched_cuda.jl:922-941`,
`translate_batched_resident.jl:1621`); `body_type` trait
(`compatibility.jl:24`) carried in `CUDARadixLifecycleOptions` as a type
parameter; `Point{Vortex}` B2M writing φ+χ on device
(`translate_batched_cuda.jl:1199-1249`) and host
(`translate_batched_resident.jl:292`), with `Point{Vortex}` + `LH=false` a
construction-time `ArgumentError`; 13-row output (potential, gradient,
9-component hessian) selected at cache construction via
`RadixFMMCache(...; hessian=true)`, with `Val(HS)`-specialized L2B/direct
kernels, switch-relative scatter, widened pinned staging, and the `fmm!`
hessian throw replaced by validation against the cache flag; vortex direct
nearfield kernels on both paths; tests
`test/device_system_interface_test.jl` (host, no CUDA) and
`test/cuda_radix_interface_test.jl` (device-vs-host parity) registered in both
runners, sharing `test/interface_test_systems.jl` (`ExtendedVortex`,
`data_per_body = 9`).

### Checkpoint 1 (2026-08-05) — local host-path verification

`test/device_system_interface_test.jl`: 930/930 pass. Full local suite
(`Pkg.test()`, 4 threads, CUDA tests auto-skipped without a GPU): all pass,
including the pre-existing radix/resident suites (023/023a/026 etc.).
Measured parity numbers (n = 600, ell = 3, ConcatenatedFixedZM2L, host
resident path):

| case | max abs err | rel to field scale |
|---|---|---|
| scalar P=8 F64: potential / gradient / hessian | 4.7e-8 / 4.2e-6 / 3.7e-4 | — |
| scalar P=4 F64: gradient / hessian | 2.4e-4 / 1.1e-2 | — |
| hessian=true cache vs plain cache (P=4, P=8): potential, gradient | 0.0 (bit-identical) | — |
| vortex P=8 F64: U / J | 2.8e-5 / 3.0e-3 | 7.9e-6 / 4.3e-6 |
| vortex P=4 F64: U / J | 6.7e-4 / 6.2e-2 | 1.9e-4 / 8.7e-5 |
| vortex P=4 F32: U / J | 6.7e-4 / 6.2e-2 | 1.9e-4 / 8.7e-5 |
| ExtendedVortex (dpb=9) P=8 F64: U | 2.0e-5 | — |

No accuracy surprises: vortex P=4 errors are truncation-dominated (Float32
indistinguishable from Float64, consistent with the task-024 ~5e-4 F32 floor)
and sit well inside the phase's 1e-3 relative-gradient gate; the
hessian-capable cache leaves the shipped 4-row outputs bit-identical.

Stage-1 residuals carried into Stages 2–3 (found in the continuing agent's
code review): five `>= 5` row validators not yet relaxed to
`>= 4 + strength_dims` (`translate_batched_cuda.jl:967, 975, 984`,
`translate_batched_resident.jl:115` + one host sibling); output-row counts
(4/13) recomputed locally rather than stored once on the cache/ctx; the
one-shot reference kernel `_cuda_direct_source_output_kernel!` remains 4-row
assign-only (reference path only). Device mirrors of the Stage-1 kernels
remain unvalidated on hardware until the Stage-4 H200 run.
