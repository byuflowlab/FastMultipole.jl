# Correct and Simplify the Higher-Derivatives Project

## Clean-Context Handoff

This document is the reviewed source of truth for the `HIGHER_DERIVATIVES/` project. It
supersedes conflicting decisions in `START_HERE.md` and the existing T/I/G task files,
especially the proposed Hessian 9→6 repack and packed-10 public layout.

Initial review snapshot: 2026-09-07, branch `higher-derivatives`, commit `1c81e310`; at that
point the complete `HIGHER_DERIVATIVES/` directory was untracked and implementation had not
begun. The worktree has since gained the theory artifacts, CPU tensor/switch/direct/L2B/probe
implementation, focused tests, documentation, and the G1 capability census described by the
authoritative per-item checklist in `START_HERE.md`; I6, I7, I9, and G2–G4 remain open there.
Unrelated `MATRIX_OPERATOR_REFACTOR/` work remains present and must not be changed.

Agents must begin with `START_HERE.md` and the selected T/I/G file, using the checklist there
for current status and this document for the detailed acceptance contract. Update each table
description as one sentence by overwriting it in place, never by appending progress history.

Authority order is: current user instruction, this handoff, revised `START_HERE.md`, selected
task file, then theory artifacts/codebase map. Verify a cited location immediately before
editing because line numbers drift, but do not repeat the broad exploration recorded here
unless the current code contradicts the snapshot.

## Baseline review summary

The initial roadmap had a sound phased structure, and this review identified the corrections now
reflected in the contract and task map below:

- **Accuracy:** The proposed 9→6 Hessian repack cannot represent the existing Lamb–Helmholtz velocity gradient. Preserve the current 9-component Hessian and store third-order output in a uniform packed-18 layout.
- **Robustness:** The baseline census missed standalone `direct!`, probe storage, tree allocation, exports, documentation, exact switch signatures, near-field caches, and vortex direct formulas. Old user kernels could otherwise silently return zero near-field third derivatives.
- **Performance:** Packed-18 removes the nine redundancies required by dense 27-component storage without introducing scalar/LH-dependent buffer layouts. Specialize scalar evaluation to exploit full tensor symmetry internally.
- **User experience:** Keep existing Hessian behavior unchanged, expose an array-like compressed tensor type, document tensor orientation, and fail early when a user-defined interaction has not opted into third-order support.
- The per-item prohibition on inspecting related code was replaced with a dependency-aware census; targeted tests belong to each work item, while full-suite runs are reserved for interface changes and milestones.

## Public Contract and Interfaces

- Keep Hessians as the existing dense 3×3, 9-row representation.
- Define third-order output as `T[i,j,k] = ∂H[i,j]/∂x[k]`. It is symmetric in `j,k` for every supported field and fully symmetric for scalar-potential systems.
- Store 18 values using component-major ordering: for each `i = x,y,z`, store derivative pairs `(xx, xy, xz, yy, yz, zz)`.
- Add an immutable `ThirdDerivativeTensor{T} <: AbstractArray{T,3}` backed by `SVector{18,T}`. It remains symmetry-compressed, reports size `(3,3,3)`, and provides allocation-free `tensor[i,j,k]` access by mapping `(j,k)` and `(k,j)` to the same packed slot. Provide `packed_data(tensor)` for allocation-free access to the canonical 18 values; do not implicitly materialize a dense 27-value array.
- `get_third_derivative` returns `ThirdDerivativeTensor` directly from the 18 buffer rows. Hot-path `set_third_derivative!` methods accept `ThirdDerivativeTensor` or `SVector{18}` and write the packed values without unpacking. A separate explicit `dense(tensor)` convenience conversion may return a 3×3×3 static array for diagnostics and interoperability, but production code and examples use the compressed type. The no-switch matrix convention uses rows `17:34`.
- Extend `DerivativesSwitch` with compile-time `TS`, while preserving all existing three-argument constructors and defaulting `third_derivative=false`. Update every exact type signature and layout helper found by a fresh repository-wide census.
- Add `third_derivative` to both `fmm!` and standalone `direct!`, including tuple-valued target switches, `FmmPlan` structural keywords, rotation checks, cached near-field compatibility, and the Radix guard.
- Add `supports_third_derivative(target, source) = false`. A requested third derivative must fail before computation unless every active target/source pair opts in; this prevents legacy `direct!` overloads from silently omitting near-field contributions.
- Extend both probe implementations with third-order storage and backward-compatible four-argument constructors.
- Do not add third-order output to `solve!`; only make internal switch construction compile after the type change. Document the solver exclusion explicitly.

### Canonical packed-18 order

For each vector component `i`, pack the symmetric derivative pair `(j,k)` as
`(xx, xy, xz, yy, yz, zz)`. The slots are:

| Slots | Components |
|---:|:---|
| 1–6 | `T[x,x,x]`, `T[x,x,y]`, `T[x,x,z]`, `T[x,y,y]`, `T[x,y,z]`, `T[x,z,z]` |
| 7–12 | `T[y,x,x]`, `T[y,x,y]`, `T[y,x,z]`, `T[y,y,y]`, `T[y,y,z]`, `T[y,z,z]` |
| 13–18 | `T[z,x,x]`, `T[z,x,y]`, `T[z,x,z]`, `T[z,y,y]`, `T[z,y,z]`, `T[z,z,z]` |

Cartesian indexing maps `(j,k)` and `(k,j)` to the same slot. With metadata disabled and
all preceding outputs enabled, the no-switch matrix convention is potential row `4`,
gradient rows `5:7`, Hessian rows `8:16`, and third-derivative rows `17:34`. Compact
switch-relative buffers continue to omit disabled groups.

`ThirdDerivativeTensor` must implement `size == (3,3,3)`, axes, length, `eltype`,
`IndexStyle(IndexCartesian())`, and allocation-free Cartesian indexing. Generic
`AbstractArray` operations are not promised to be allocation-free; construction,
`packed_data`, scalar indexing, and packed setters are.

### Compatibility safety

Use `DerivativesSwitch{PS,GS,HS,NO,NM,TS}`: append `TS` so the historical first five type
parameters retain their meaning. Preserve all three-Boolean outer constructors, with TS
defaulting false. Update exact five-parameter signatures and constructors throughout the
repository; methods parameterized by only `DerivativesSwitch{PS,GS,HS}` remain compatible.

That compatibility creates a correctness hazard: an old `direct!` method will still dispatch
while silently omitting TS. Add the public opt-in
`supports_third_derivative(target_system, source_system) = false`. Before allocation or
execution, `fmm!` and `direct!` check every active target/source pair and throw an informative
`ArgumentError` unless it opts in. Reference pairs opt in only after their direct kernel and
target writeback exist. Perform the check once per call or plan construction, never inside
interaction loops.

## Baseline codebase snapshot

These historical facts were checked at commit `1c81e310` before implementation began; use the
refreshed map and status table rather than treating present-tense details here as current state:

- `src/containers.jl:46-63` defines `DerivativesSwitch{PS,GS,HS,NO,NM}` and its compact
  layout. `src/derivativesswitch.jl:1-159` contains its constructors/ranges;
  `_standard_output_rows` currently counts `1 + 3 + 9`.
- `src/tree.jl:393-395` allocates target buffers via `target_buffer_rows`. Near line 2147 it
  allocates the current `2 × 3 × n_harmonics` `gradient_n_m` scratch.
- Switch-aware matrix getters/setters are in `src/compatibility.jl:711-801`. No-switch
  Hessian access uses rows `8:16`. Reset uses `output_range(switch)`, so correct range
  plumbing propagates to reset behavior.
- Standalone direct keyword plumbing is in `src/direct.jl:48-136` and currently exposes only
  PS/GS/HS. The original project map omitted this necessary public entry point.
- FMM entry/cache variants are in `src/fmm.jl:841-1074`; structural plan keywords are near
  1147, rotation-sensitive output logic near 1237, and downward-pass scratch plumbing near
  657-798 and 1533-1535.
- Complex-basis L2B is in `src/evaluate_expansions.jl:33-399`: it forms gradient
  coefficients, stores them in `gradient_n_m`, and reapplies the recurrence for a
  9-component Hessian. The real-basis path near 404-583 performs the second contraction
  inline through `_complex_gradient_contract`.
- Probe structs are in `src/containers.jl:1014-1030`; constructors/reset/writeback are in
  `src/probes.jl`. Adding storage changes their generated positional constructor, so retain
  explicit compatible four-argument outer constructors.
- `test/gravitational.jl:9-13,31,95-136` owns a separate 16-row result matrix. Its `direct!`
  computes only potential and gradient despite destructuring HS. It needs analytic Hessian,
  packed-18 third order, widened result storage, and writeback.
- `test/vortex.jl:83-136` contains point-vortex direct velocity and a full nonsymmetric
  9-component velocity gradient. This is direct evidence that packed-6 Hessian storage is
  invalid. It also needs the LH third-order direct formula before LH acceptance.
- `src/nearfield_cache.jl` derives matrix rows from `output_range(switch)`, so basic storage
  should generalize. Construction, compatibility, estimates, plan reuse, and standalone
  direct integration still require explicit TS tests.
- `src/FastMultipole.jl:200-254` collects public exports. Export the tensor type,
  `packed_data`, range/get/set helpers, and capability trait.
- Public three-output documentation appears in `docs/src/guided_examples.md`,
  `docs/src/advanced_usage*.md`, and `docs/src/device_interface.md`; examples also teach
  switch-relative layout.
- GPU/Radix has an independent fixed 4- or 13-output-row layout controlled by
  `RadixFMMCache.hessian::Bool`. The five principal surfaces are
  `translate_batched_cuda.jl`, `translate_batched_resident.jl`, `cross_stencil_cuda.jl`,
  `direct_rectangular.jl`, and `interaction_list_batched.jl`. Do not read those large files
  during CPU phases. G1 must census them by kernel/body/route.
- `ForwardDiff` is already declared as a test extra and should be the independent derivative
  oracle; finite differences are secondary corroboration.

Run this focused census once while revising the codebase map, then inspect only locations
relevant to the selected task:

```sh
rg -n "DerivativesSwitch|scalar_potential.*gradient.*hessian|hessian_range|\
get_hessian|set_hessian|target_buffer_rows|output_range|nearfield_cache" \
  src test docs examples --glob '*.jl' --glob '*.md'
```

## High-Level Implementation Sequence

1. **Contract and census**
   - Rewrite `START_HERE.md` around the 9/18 contract.
   - Replace the Hessian-repack tasks with a compatibility-regression task.
   - Rebuild `reference/codebase-map.md` using repository-wide searches covering source, tests, examples, and docs.
   - Remove unavailable agent-specific routing requirements and allow task owners to inspect connected interfaces when the map may be stale.

2. **Theory**
   - Derive the scalar third recurrence and packed-18 mapping; prove full scalar symmetry.
   - Derive analytic scalar Hessian and third-order point-source formulas.
   - Derive the LH tensor as the second spatial derivative of velocity, including analytic point-vortex direct formulas and its `j,k` symmetry.
   - Derive both complex- and real-basis contractions.
   - Verify formulas primarily with `ForwardDiff`; retain finite differences only as a secondary, scale-aware cross-check.
   - Complete an independent theory milestone before implementation.

3. **CPU scalar implementation**
   - Land switch/range/accessor/direct/FMM/cache/probe plumbing and capability checks first.
   - Implement complex and real L2B paths. Support `TS=true` independently of `PS`, `GS`, and `HS`.
   - For `LH=false`, compute only the ten fully independent scalar components internally and expand them into packed-18 storage.
   - Extend the gravitational reference `direct!`, permanent tests, user documentation, and guided example.
   - Finish with scalar accuracy, compatibility, and performance review.

4. **CPU Lamb–Helmholtz implementation**
   - Extend both L2B paths and the vorton reference `direct!` to all 18 independent components.
   - Add vortex, mixed target/source, dynamic-order, and FLOWVPM-shaped tests.
   - Complete an LH milestone before beginning GPU implementation.

5. **GPU/Radix implementation**
   - Gate the GPU phase on the completed scalar and LH CPU milestones.
   - Census every kernel variant and produce a capability matrix by body type, near-field implementation, and resident/nonresident path.
   - Replace the Boolean cache Hessian flag with a construction-time maximum derivative level. Level three allocates the fixed `1 + 3 + 9 + 18 = 31` output rows.
   - Implement scalar and vortex near/far kernels, then add permanent CPU/GPU parity and performance coverage.

## Detailed task map

The existing task files use these decision-complete assignments.

### Theory

- **T1 — scalar recurrence:** Derive the third complex solid-harmonic differentiation pass,
  coefficient orientation, and full scalar symmetry. Prototype packed output and compare
  with nested `ForwardDiff` derivatives of `1/r`.
- **T2 — packed-18 contract:** Replace the 6/10 task with the exact 18-slot table,
  `ThirdDerivativeTensor` indexing, legacy/compact layouts, and compatibility rules. Do not
  alter Hessian storage.
- **T3 — real-basis contraction:** Derive the deeper real-basis contraction, compare it with
  T1 on identical coefficients, and check it against `ForwardDiff`.
- **T4 — scalar direct oracle:** Derive factored analytic Hessian and third derivatives of
  `1/r`, emitting packed-18 directly. Validate primarily with `ForwardDiff` over multiple
  scales.
- **T5 — LH theory/direct oracle:** Derive the second spatial derivative of point-vortex
  velocity, its `j,k` symmetry, complex/real L2B flow, and CPU/GPU near-field formulas.
- **T6 — theory milestone:** Independently rerun T1–T5 checks and audit signs,
  `ONE_OVER_4π`, tensor orientation, packing, and implementability. T1–T4 gate scalar CPU;
  approved T5 gates I8.

Theory work modifies only `HIGHER_DERIVATIVES/theory`, `scripts`, `data`, and task documents.

### CPU implementation

- **I1 — compatibility/tensor API:** Replace Hessian repacking with the compressed tensor
  type, indexing, buffer helpers, exports, probe storage, compatible probe constructors, and
  bitwise Hessian-layout regression tests.
- **I2 — switch/public plumbing:** Add final-position TS; all scalar/vector constructor
  forms; compact ranges; `fmm!` and `direct!` keywords; capability preflight; structural-plan,
  rotation, cache, and Radix guards; and documentation. Do not claim kernel support while
  TS rows are intentionally zero.
- **I3 — complex scalar L2B:** Make TS independent of lower switches. Allocate scratch once
  per active worker only when any target requests TS. Compute the 10 unique scalar values
  internally and emit their packed-18 repetitions without a dense temporary.
- **I4 — gravitational reference:** Add analytic Hessian and packed-18 third-order direct
  accumulation, storage/writeback, standalone-direct behavior, and capability opt-in. I3 and
  I4 may run in parallel after I2; whichever finishes second owns their integration check.
- **I5 — real scalar L2B:** Implement the T3 design without dense intermediates. Require
  real/complex agreement near roundoff and agreement with direct to truncation accuracy.
- **I6 — scalar integration/performance:** Add permanent switch-combination, system, probe,
  P-mode, conditioning, near-field-cache, plan, metadata, threading, API, and performance
  coverage.
- **I7 — scalar milestone:** Run all scalar scripts and the full suite at one and four
  threads; record convergence, disabled-path performance, scratch/output memory, and docs.
- **I8 — LH implementation/reference:** Add packed-18 complex and real L2B plus vorton
  direct formulas and capability opt-in. Remove the temporary LH+TS rejection only when all
  three agree.
- **I9 — LH milestone/tests:** Cover vortex convergence, symmetry, dynamic P, threading,
  mixed pairs, and FLOWVPM-shaped usage. I9 gates G1.

### GPU/Radix

- **G1 — census/layout:** Build a capability matrix for every output site by body type,
  kernel, near/far route, and resident/nonresident path. Specify cache levels of 4 rows
  through gradient, 13 through Hessian, and 31 through third order.
- **G2 — cache/layout infrastructure:** Replace the Hessian-repack task with a maximum
  derivative-level cache field, 4/13/31 allocation/clear/copy behavior, request guards, and
  proof that existing 4/13 behavior and performance are unchanged.
- **G3 — kernels:** Implement packed-18 scalar and vortex near/far kernels for every supported
  G1 route. Unsupported combinations fail clearly. Compare with CPU FMM and direct oracles.
- **G4 — parity/performance:** Permanently cover levels, P, sizes, scalar/vortex routes, and
  device lifecycle. Record kernel-only and end-to-end performance and memory under existing
  HPC campaign rules.

## Test and Acceptance Plan

- Preserve all existing Hessian values and layouts bit-for-bit when `third_derivative=false`.
- Test all 16 combinations of the four output switches, including `TS=true` with every lower-order output disabled.
- Check every packed slot and both `(j,k)` permutations, explicit dense conversion, scalar
  full symmetry, and LH `j,k` symmetry without assuming LH `i,j` symmetry.
- Compare analytic direct kernels against `ForwardDiff` over randomized directions and logarithmically separated distance scales; exclude singular self-pairs explicitly.
- Verify FMM convergence across expansion orders for complex and real bases, scalar and LH systems, single/multi-system trees, target/source calls, probes, dynamic order, and one- versus four-thread execution.
- Cover standalone `direct!`, conditioned direct paths, near-field cache build/reuse/mismatch, `FmmPlan` reuse, transformed-plan rejection, metadata, extra outputs, and unsupported user-kernel errors.
- Explicitly test a legacy three-switch direct overload: ordinary requests still work, while
  `TS=true` fails during capability preflight instead of silently returning zeros.
- Require `@allocated == 0` after warm-up for tensor construction from a buffer,
  `get_third_derivative`, Cartesian scalar indexing, `packed_data`, and both packed setters.
- Benchmark the disabled path with paired alternating baseline/feature runs: at least 30
  warmed samples and at least 0.2 seconds of work per sample. Median feature/baseline must be
  at most 1.03. Rerun one failure; a repeated failure blocks the milestone and requires
  profiling.
- Record enabled-path time, allocations, scratch bytes per worker, and output-buffer bytes,
  without imposing a speed threshold before a correct baseline exists.
- GPU acceptance covers every supported kernel route, output-level cache configuration, scalar/vortex parity, and 4/13/31-row memory and throughput results.

Targeted tests run after every item. The full suite runs after I1, I2, and each phase
milestone; I7 and I9 run it with one and four Julia threads. GPU timing separates kernel
execution from transfer/initialization and also reports end-to-end throughput.

## Assumptions

- Lamb–Helmholtz support remains a later phase of this project, completed before GPU work.
- The 18-row representation is uniform across scalar and LH systems; no runtime field-mode-dependent buffer layout is introduced.
- Higher-order output is opt-in and defaults off.
- Existing error estimators are not claimed to bound third-order error. Documentation will state this limitation, while convergence tests verify behavior under fixed and dynamic expansion order.
- Approval is independent at phase milestones; individual tasks require recorded targeted verification but not a separate reviewer and full-suite rerun each time.

## Why the Original Roadmap Must Be Revised

1. Packed-6 Hessians cannot represent the nonsymmetric velocity gradient already produced
   and consumed by vortex systems.
2. Packed-10 third derivatives apply only to scalar φ. LH needs 18 independent components
   because only the last two derivative indices are symmetric.
3. The original plumbing omitted public standalone `direct!`, though later tests depended
   on `direct!(...; third_derivative=true)`.
4. Existing three-switch direct overloads would keep dispatching and could silently omit all
   near-field third derivatives; capability preflight is required.
5. Probe structs/writeback lacked third-order storage despite promised ProbeSystem tests.
6. LH acceptance required comparison with vorton direct evaluation, but no task implemented
   the corresponding direct formula.
7. GPU work could begin before LH CPU completion even though its fixed layout and kernels
   also claimed LH support.
8. Finite-difference-only verification is fragile; `ForwardDiff` is already available as a
   stronger independent oracle.
9. A dense inferred static tensor need not heap-allocate, but it still materializes 27
   values, increases register/code pressure, and can escape. The default API therefore stays
   compressed.
10. Full-suite runs and separate approval for every small task cost substantially more than
    targeted item checks plus independent full-suite milestone reviews.
