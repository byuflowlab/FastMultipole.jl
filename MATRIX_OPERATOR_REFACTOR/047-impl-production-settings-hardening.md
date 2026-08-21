# 047 Impl: Production Settings Consolidation and Hardening

## Status and entry gate

**Staged `2026-08-20` (user direction). Not started.**

Entry gate: `046` complete and approved (all work lands on the unified
branches). Runs in parallel with `048` — both block only on `046`. No new
performance work in this row: consolidation, validation, documentation,
robustness only.

## Motivation

The GPU production surface accumulated ~25 process-global `Ref` tunables
across the 019–042 arc. Several are read at cache-construction time only —
flipping them afterwards **silently keeps the old mechanism**
(`translate_batched_cuda.jl:2247-2252`), a documented hazard. Before FLOWVPM
and FLOWPanel production work builds on this surface (`048`–`052`), the
tunables need one validated, documented settings surface, and the
construction-time-lock contract needs to be enforced loudly and tested.
The Future Dispatch Cleanup Notes in START_HERE.md (boolean residency flags
→ dispatch-on-object) are fodder for this row.

## Objective

One consolidated, documented, validated settings surface for the GPU
production path; every construction-time-locked tunable errors loudly on a
late flip (with a regression test); the dispatch-cleanup items addressed;
generalization/robustness sweeps green; full FastMultipole suite green.

## Method

### Stage 1 — inventory and classification

For each tunable in the Recorded-context list below, classify:
(a) genuinely runtime-flippable, (b) construction-time-locked (must error on
late flip), (c) should become a per-cache option on
`CUDARadixLifecycleOptions` / `AdaptiveTreePolicy`
(`containers.jl:2119-2179`, `:578-603`), or (d) dead/vestigial (remove with
user approval). FLOWVPM-side overrides flow through the non-exported
`radix_fmm_settings!` (`FLOWVPM_fmm_radix.jl:86-113`) — keep that surface
coherent with the consolidation.

### Stage 2 — consolidation

Move class-(c) tunables to per-cache options; wrap the remainder behind one
documented settings module/API with validation (type/range/enum checks,
docstrings). For class (b): record the construction-time snapshot on the
cache and **throw a loud, actionable error** if the global is flipped and
then used with an existing cache (today it silently keeps the old
mechanism).

### Stage 3 — dispatch cleanup

Address the Future Dispatch Cleanup Notes items:
`CUDARadixLifecycleOptions.allow_host_bodies` → source-buffer dispatch or
explicit residency capability checks; the legacy `nearfield_device::Bool`
(if revived) → nearfield execution policy tag; `target::Bool` tree-role arg
and radix route-selection flags → dispatch on source/destination/policy
objects.

### Stage 4 — robustness sweeps

- Parity sweep: F32/F64 × adaptive/uniform × P=4/P=8 (the standing P=4 test
  rule: same-P parity checks at P=4; direct-accuracy tolerances gate at
  P=8).
- Error-path exercises: capacity overflow, out-of-box bodies (must throw, no
  silent rebuild), recenter contract (explicit `recenter!` only).
- Regression test that a post-construction flag flip errors loudly.
- Full FastMultipole test suite green.

## Gates and verdict

- All tunables classified and consolidated; no undocumented process-global
  `Ref` remains on the production GPU path.
- The late-flip regression test exists and passes.
- Sweeps and full suite green. Any default-behavior change requires explicit
  user approval (deferred to the `053` checkpoint if ambiguous).

## Artifacts

- Source changes on the unified branch (settings surface, per-cache options,
  dispatch cleanup, tests).
- A classification table (tunable → class → disposition) appended to this
  doc or committed under `MATRIX_OPERATOR_REFACTOR/data/`.

## Verification

- `Pkg.test()` green on the unified FastMultipole branch; FLOWVPM
  `runtests_gpu_fmm.jl` Part A green (device parts on cluster).
- Grep audit: no production-path read of a raw tunable `Ref` outside the
  consolidated surface.

## Recorded context (2026-08-20 staging)

**Tunables to consolidate** (all process-global `Ref`s except env
`FASTMULTIPOLE_FORCE_CUDA_LOAD` at `FastMultipole.jl:143`):

- Nearfield: `CUDA_NEARFIELD_GH_MODE(:fp32)`, `CUDA_NEARFIELD_BINNING(:classsplit)`,
  `CUDA_NEARFIELD_SHAPE(:pairs)`, `CUDA_NEARFIELD_FUSED_MIN_BODIES(400k)`,
  `CUDA_NEARFIELD_SUBSORT(true)`, `CUDA_NEARFIELD_PAIR_AABB(false)`,
  `CUDA_TWOPASS_*`, `CUDA_SYMMETRIC_NEARFIELD(false)`,
  `DIRECT_CUDA_MAX_BLOCKS`.
- M2L: `FACTORED_/PRECOMPUTED_/DENSE_CUDA_*` (whole-pass, chunk, fused,
  tiled, `DENSE_CUDA_TENSOR_FORMAT(:fp16)`).
- Tree/lifecycle: `RADIX_CUDA_COUNTING_SORT(±MAX_ELL)`,
  `CUDA_CACHED_WINDOWS(true)`, `CUDA_GRAPH_LIFECYCLE(true)`,
  `CUDA_OVERLAP_NEARFIELD(true)`.

**Hazard (already documented):** several are read at cache-construction
only — flipping later silently keeps the old mechanism
(`translate_batched_cuda.jl:2247-2252`).

**Per-cache options that exist today:** `CUDARadixLifecycleOptions` +
`AdaptiveTreePolicy` (`containers.jl:2119-2179`, `:578-603`); FLOWVPM-side
overrides via non-exported `radix_fmm_settings!`
(`FLOWVPM_fmm_radix.jl:86-113`).

**Signed contracts (integration-api-spec.md, 2026-08-04):** counters
`body_uploads=0`/`expansion_host_copies=0`; zero per-step alloc; full
9-component hessian; cache-lifetime-fixed set includes **`n_systems`**;
out-of-box throws (no silent rebuild); explicit `recenter!` only; normalized
unit-cube internal coords with 1/L,1/L²,1/L³ output rescale.

**034 integration state** (Done+tested): adapter `FLOWVPM_fmm_radix.jl`
(520 lines) + `ext/FLOWVPMCUDAExt.jl`; traits Point{Vortex}, residency from
`particles isa Array`, `RegularizedVortex(sigma_row=8)`, gaussianerf-only
hard check (`:62,:260`); one lazily-built `RadixFMMCache` per pfield
(capacity = maxparticles, hessian=true, WeakKeyDict registry); derived ell
from near-set adequacy + n^(1/3) occupancy heuristic (`:300-:370`);
recenter-retry policy; zero per-step body H2D/D2H. Hard errors: rbf
(`:497`), sfs (`:499-500`), autotune flags. 034 punch list: `update_U_prev`
scalar-indexes CuArray; per-np-change scatter-buffer realloc; SFS/rbf
loud-unsupported.

**START_HERE Future Dispatch Cleanup Notes:** prefer dispatch on source,
destination, and policy objects over boolean residency flags; cleanup
candidates = `allow_host_bodies`, legacy `nearfield_device::Bool`,
`target::Bool` tree-role arg, radix route-selection flags.

**P=4 test rule (standing):** all future tests must cover P=4;
direct-accuracy tolerances gate on P=8, use same-P parity checks at P=4.
