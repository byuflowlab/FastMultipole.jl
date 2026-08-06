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

### Stage 2 — `direct_kernel` functor trait, `RegularizedVortex`, adequacy gate

Committed as `704ab83`. Design decisions and deviations:

- **Trait/functor plumbing.** `direct_kernel(system)` trait
  (`compatibility.jl`) defaults per `body_type` (`SingularSource` /
  `SingularVortex`); resolved once at cache construction and stamped into
  `CUDARadixLifecycleOptions` as a sixth type parameter, so the pair kernels
  specialize at compile time. An explicit `options.direct_kernel` is honored;
  a conflict between a non-default trait and a non-default options choice is
  an `ArgumentError`.
- **Signature deviation from spec §5** (documented in the trait docstring):
  the functor methods are `_direct_pair_ug(kernel, dx, dy, dz, r2, invr,
  source_bodies, j)` / `_direct_pair_ugh(...)` — flat arguments plus the
  caller-computed reciprocal sqrt rather than a column view, so one generic
  method body compiles as both the host loop and the CUDA device function
  (the CUDA caller supplies `_cuda_fast_rsqrt`, the host `inv(sqrt(r2))`).
- **Hard-coded kernels retained** verbatim (host + CUDA, scalar + vortex) as
  the functor-abstraction benchmark reference; production dispatch routes all
  kernels through the generic functor kernels. The symmetric Newton-pair
  kernel is gated to `SingularSource` (its shared-work trick assumes the
  same-source/target scalar kernel).
- **Erf-free g/h (task-file amendment honored).** The theory-§3 series is run
  to ρ = 2, which required re-measuring term counts (the §3 6/10-term counts
  hold only to its ρ = 0.5 partitioning switch): 13 terms (Float32) and 19
  (Float64) hold ≤ 6.8e-7 / 1.7e-12 relative over the whole series branch;
  above ρ = 2 the §6.2 one-`exp` degree-3 `s(u)` fit holds |δg| ≤ 2.1e-4
  against the 3.69e-4 absolute budget, with the exact singular limit
  (g → 1, h → −3) beyond the fit range. Constants measured by
  `scripts/fit_032_nearfield_g.jl` → `data/kernel_splitting/nearfield_g_eval.csv`.
  **Measured switch point: ρ_c = 2.0, both precisions.** No FDLIBM code is in
  `src/`; the `custom_erf` candidate lives only in the benchmark script until
  the H200 A/B decides (job 13058104). The §2 `r2 > 0` self-pair guard is kept
  (scale-free) rather than FLOWVPM's absolute `r2 > 1e-6`.
- **Adequacy gate (user decision: reject, don't enlarge).** Per-step, both
  residency paths, regularized kernels only: `g_min·h_leaf > ρ_t·σ_max` with
  σ_max from the packed σ row (host loop / device reduction) and `g_min` from
  the live stencil (deepest-level near radius for hierarchical policies —
  the constraint binds only at the leaf level — or the closest accepted
  offset class for the flat stencil). Failure throws naming the measured
  gap/cutoff ratio and the admissible depth. The `n/8^ℓ` form is not
  implemented anywhere.
- **`>= 5` validator resolution.** The three `_radix_body_matrix` validators
  and the `_host_radix_body_matrix` sibling take bare pre-packed matrices on
  one-shot/device-origin paths with no system object in scope, so
  `4 + strength_dims` is not computable there; they are minimum-floor checks
  only. The cache path validates `data_per_body >= 4 + strength_dims` per
  system at construction. Left as-is, documented here.

Local verification: `device_system_interface_test.jl` 930 + 22 pass (series /
outer-branch accuracy vs a BigFloat reference — the naive Float64 reference
itself cancels at small ρ, confirming the §3 hazard; functor-vs-hard-coded
parity exact for the scalar kernel and ≤ 1e-13 relative for the vortex form
change; `RegularizedVortex` end-to-end vs an erf-based O(N²) reference at
≤ 1e-3 (F64) / 3e-3 (F32) relative max; gate + validation error paths). Full
local `Pkg.test()` passes.

### Checkpoint 2 (2026-08-05) — H200 validation + mandated benchmarks

Job **13058240** (H200 `m13h-2-2`, 6m13s; two earlier submissions failed on
environment: 13058104 used the CUDA-less `test` project, 13058142 hit `set -u`
vs `/etc/profile`; 13058191 exposed that the cluster's default julia module
moved from 1.11.7 to **1.12.6, which segfaults in host LLVM while
JIT-compiling the device step** — `cuda_032_run.sh`/`cuda_032_submit.sh` now
pin `julia/1.11.7-6bmogfl`, the toolchain of every H200 result of record).

Device correctness — first hardware validation of the 032 device mirrors:
`test/cuda_radix_interface_test.jl` **1238/1238** pass (stage-1 packing /
vortex B2M+χ / 13-row hessian device-vs-host parity incl. P=4 and Float32,
stage-2 `RegularizedVortex` device parity, device adequacy-gate rejection,
counter contract); shipped lifecycle regression `cuda_radix_lifecycle_test.jl`
215/215 + 37/37 pass.

Benchmark A — functor abstraction (031 sign-off (b); n = 1e6, shipped
hierarchical defaults, ell = 5, median of 9):

| config | hard-coded | functor `SingularSource` | delta |
|---|---:|---:|---:|
| F32, 4-row (shipped default) | 3.825 ms | 3.826 ms | +0.0% |
| F32, 13-row | 9.597 ms | 9.972 ms | +3.9% |
| F64, 4-row | 10.977 ms | 10.844 ms | −1.2% |
| F64, 13-row | 15.382 ms | 16.852 ms | +9.6% |

The shipped scalar default (4-row) pays nothing for the abstraction; the
13-row hessian path (new in 032, no shipped baseline) pays 4–10%, likely the
13-value tuple return vs in-place accumulation — noted as a small future
optimization, not a regression.

Benchmark B — vortex kernel ladder (n = 1e6, ell = 4, `near_radius2 = 12`,
β = 2, adequacy margin verified; 13-row output):

| kernel | Float32 | Float64 |
|---|---:|---:|
| hard-coded singular vortex | 129.6 ms | 239.2 ms |
| functor `SingularVortex` (crss/a/b form) | 114.3 ms | 227.4 ms |
| functor `RegularizedVortex` (erf-free) | 190.2 ms | 432.6 ms |
| functor `CustomErfVortex` (FDLIBM port) | 284.8 ms | 663.3 ms |

**Decisions from the numbers:** (1) the functor `SingularVortex` is *faster*
than the stage-1 hard-coded kernel (the crss/a/b factorization beats the
expanded form) — functor dispatch stays the production path; (2) **erf-free
wins the mandated A/B decisively — 1.50× (F32) / 1.53× (F64) faster than the
`custom_erf` port at equal delivered accuracy** (device outputs agree to
4.3e-5 relative, far inside the 1e-3 phase gate; host reference check confirms
the erf-free |δg| = 2.07e-4 sits inside its 3.69e-4 §6.2 budget). The shipped
erf-free form stands; **no FDLIBM code enters `src/`** (the port remains
benchmark-only in `benchmark_032_nearfield.jl`). (3) Regularized-everywhere
costs ~1.7× the singular kernel per pair — the baseline-vs-partitioned
trade-off is 032a's question, unchanged. Data:
`data/feasibility_1m_10ms/nearfield032_m13h-2-2_20260805-191230.csv` + job log
`fm032-13058240.out` (same directory).

### Cluster toolchain note (2026-08-05)

A clean-env probe (throwaway 1.12.6 env, current CUDA.jl stack, job 13058336)
reproduced the device-step host-LLVM segfault exactly, while a CUDA smoke test
passed — the crash is Julia 1.12.6's LLVM-18 JIT on FastMultipole's device
step, not a stale-manifest artifact (the CUDA.jl full suite could not run on
the offline compute node; verdict rests on the smoke test + repro). All 22
cluster run scripts now pin `julia/1.11.7-6bmogfl` (commit `5bf0524`);
1.12.x migration is blocked until the upstream crash is resolved. Evidence
kept at `orc:~/fm112-13058336.out`.

### Stage 3 — persistent buffers, hook removal, `recenter!`, API surface (Done)

Committed as `ad1bef4` (core src, subagent), `3d2524b` (connection guide
`docs/src/device_interface.md` + worked example
`examples/device_resident_system.jl` + docs page registration), `af3d3df`
(stage-4 validation script draft `scripts/cuda_032_validation.jl`, not yet
submitted). Delivered:

1. **Gap 5 closed**: persistent per-system device source buffers allocated at
   construction for every residency; the recurring refresh fills the valid
   prefix in place through the consumer's `source_to_buffer!` — no per-step
   `CuArray` allocation, `body_uploads` untouched for device-resident systems.
   One-shot builders keep the allocating path via a shared fill helper.
2. **Gap 7 closed**: deprecated `source_system_to_device_buffer!` /
   `target_system_from_device_buffer!` fully removed (definitions, exports,
   `hasmethod` probes, depwarn shims); lifecycle regression test updated.
3. **`recenter!(cache, systems; bounds=nothing, padding=0.05)`** with the full
   spec-§4 validation contract (all error paths tested, failure leaves the
   cache bit-identical); derived bounds via host `get_position` or a device
   min/max reduction (six scalars downloaded). **Documented deviation**: the
   geometry-rebuild fallback is construct-and-swap into the existing mutable
   cache (object identity preserved) — a `recenter!` costs about one
   construction, transiently ~2× memory, and restarts the transfer counters,
   rather than the spec's zero-allocation ideal. The normalized-unit-cube
   variant that would make `recenter!` a pure restamp is recorded as the `035`
   lever (user decision: fallback first). Subtlety: a hierarchical policy's
   stencil tolerance is box-derived, so the policy ε is re-derived at the new
   box (near set/schedule preserved exactly).
4. **API surface**: `body_type`, `direct_kernel`, `data_per_body`,
   `strength_dims`, `has_vector_potential`, `get_position`,
   `source_to_buffer!`, `buffer_to_target!`, `recenter!` exported with
   ownership/lifecycle/allocation docstrings incl. total-influence delivery
   semantics; connection guide + runnable worked example per spec §§2-8.

Local verification: interface tests 930 + 22 + 21 pass; full `Pkg.test()`
passes. CUDA mirrors (persistent-buffer identity, flat `body_uploads` across
device-resident steps, device `recenter!` parity) are in
`cuda_radix_interface_test.jl` awaiting the Stage-4 H200 job.
