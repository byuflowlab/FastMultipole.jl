# 013 Implementation Axis Swap Operators (Materialized Y-Rotation Variant)

## Objective

Implement the materialized arbitrary y-rotation building blocks used by the
near-term `MaterializedYRotationM2L` path.

> **Scope note (2026-06-19 roadmap update).** This task supplies the **materialized
> arbitrary y-rotation** building blocks. It precomputes the angle-independent
> invariant `H(pi/2)` / `S_pos` / `S_neg` data into the operator-invariant cache,
> then *reconstructs a per-angle dense Wigner `Ts(theta)`* from that data each call
> (`build_Ts_from_S!`) and applies it through production-parity y-rotation wrappers.
> It is bit-exact (Float64) vs production and low-risk, but it deliberately
> re-collapses the `S · Z_theta · S_inv` factorization into an angle-dependent dense
> matrix. The explicit factored-stage counterpart is
> `013b-impl-fixed-y-swap-primitives.md`. The two near-term full-M2L variants
> are composed in `014`, compared in `015`, and compared end-to-end in `024`.

## Dependencies

- `004-theory-axis-swap-conventions.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `010-impl-z-rotation-operators.md`
- `012a-milestone-review-impl-009-012.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/axis-swap-conventions.md`
- Existing axis-alignment and rotation production code

## Artifacts or Production Surface

- Production axis-swap and y-rotation operator code
- Tests comparing invariant axis swaps with current behavior

## Deliverables

- Cached invariant `H(pi/2)` / `S_pos` / `S_neg` data.
- `build_Ts_from_S!` to materialize arbitrary `Ts(theta)`.
- Production-parity y-rotation apply wrappers using the materialized `Ts(theta)`.
- Parity tests for active/passive convention handling.

## Verification

Run parity tests for representative axis-aligned and off-axis interactions.
Record commands and result summaries.

## Implementation Notes

Completed on 2026-06-18.

### What was implemented

Additive, non-exported operators in `src/rotate_batched.jl` (per the
`rotate.jl -> rotate_batched.jl` placement rule). No production hot-path call
site changed; legacy `update_Ts!` / `rotate_multipole_y!` / `rotate_local_y!` in
`src/rotate.jl` are untouched.

- **Cached angle-independent axis-swap blocks** (`S_pos`, `S_neg`): the
  `T_n(θ) = S_n · Z_n(θ) · S_n^{-1}` decomposition from
  `theory/axis-swap-conventions.md`. `update_S_blocks!` is the angle-independent
  skeleton of production `update_Ts!` with the `cos/sin(νθ)` factors removed,
  keeping the identical `(n, m, mp, ν)` traversal and sign recurrences
  (`_1_n`, `_1_n_mp`, `_1_n_mp_ν`, `_1_mp_odd`) and the parity-gated `ν=0`
  (`zero_mode`) term.
- Index helpers `length_S_block`, `length_Ss`, `S_block_offset`, `S_index`
  mirror the existing `Hs`/`Ts` flat-buffer conventions.
- **`build_Ts_from_S!`** reconstructs the per-call Wigner `Ts` from the cached
  blocks using only the cheap `cos/sin(νβ)` recurrence, matching production
  `update_Ts!` to ~1e-12 (same accumulate→×2→add-`ν=0` order). This removes the
  per-call H(π/2) product rebuild that `008c` flagged as the dominant M2L cost.
- **`Ts` is shared** between the multipole and local paths; they differ only in
  the sign table (`ζ` vs `η`) passed to the reused production apply kernels.
- Four thin wrappers `rotate_multipole_y_op!`, `back_rotate_multipole_y_op!`,
  `rotate_local_y_op!`, `back_rotate_local_y_op!` build `Ts` then delegate to the
  existing `_rotate_multipole_y!` / `_rotate_local_y!` (which reset their
  destination — back-alignment resets rather than accumulates, per the theory
  convention; final accumulation stays at `back_rotate_z!`).
- `OperatorInvariantCache` (`src/containers.jl`) gained `S_pos`/`S_neg`, built in
  the constructor from the just-populated `Hs_pi2` at `P_active` (reads no module
  globals, so cache construction stays side-effect-free).

### Tests

- `test/rotate_batched_test.jl`: new `axis-swap y-rotation operators (batched)`
  testset — non-export guard; `Ts` reconstruction parity vs `update_Ts!`
  (`Ts[1]` exact, body isapprox); multipole forward parity vs
  `rotate_multipole_y!` (ζ); local forward parity vs `rotate_local_y!` (η) reusing
  the same reconstructed `Ts` to prove sharing; `Val(false)` inactive-channel
  reset; back-rotation reset-not-accumulate semantics and parity vs
  `back_rotate_*_y!`. Coverage: `P ∈ (0,1,3,6,9)`, `θ ∈ (0, π, π/7, -2π/5, 1.3,
  -0.4)` (axis-aligned + off-axis ±), `Val(false)`/`Val(true)`, Float64/Float32.
- `test/operator_cache_types_test.jl`: `expected_invariants` extended to recompute
  `S_pos`/`S_neg`; added `eltype`, exact `==`, and `length_Ss` assertions for the
  two new fields.

### Verification

```
julia --project=. test/rotate_batched_test.jl
# z-rotation operators (batched):        25684 Pass / 25684 Total
# axis-swap y-rotation operators (batched): 1084 Pass / 1084 Total

julia --project=. --threads=4 test/rotate_batched_test.jl
# z-rotation operators (batched):        25684 Pass / 25684 Total
# axis-swap y-rotation operators (batched): 1084 Pass / 1084 Total

julia --project=. -e 'using Pkg; Pkg.test()'
# Testing FastMultipole tests passed   (0 Test Failed / Error During Test;
# includes operator_cache_types_test.jl with the new S_pos/S_neg assertions)
```

Note: the full-suite run prints CUDA-related precompile markers before the
package tests; the package test command completed successfully (exit 0).

### Notes for review / later tasks

- The `S` cache is O(P⁴) for both polarities (a factor ~P larger than `Ts`/`Hs`),
  sized once at `P_active` (~320 KB at P=20, Float64). Acceptable as a
  precompute-once cache for this milestone; flagged for `015`/`019` to revisit if
  `P_active` grows large.
- Parity tolerance: the `2026-06-19` review (below) established that in **Float64**
  the reconstruction is in fact **bit-exact** vs `update_Ts!` (the only
  reassociation is multiplication by `get_scalar ∈ {0, ±1}`, exact under IEEE), so
  the test now asserts exact `==` for Float64. **Float32** differs by ~2 ulp
  (~2e-7), which is intrinsic Float32 vs Float64 rounding present in production
  `update_Ts!` itself (confirmed: `update_Ts!(Float32)` vs `update_Ts!(Float64)`
  differ by 2.07e-7; `build_Ts_from_S!(Float32)` lands at 2.29e-7 from the Float64
  reference and within 0.9e-7 of production Float32), not an artifact of the `S`
  decomposition. The earlier "~1e-4, not bit-for-bit" note was over-conservative.

### Follow-up review changes on 2026-06-19

- Replaced loop-based `length_Ss` / `S_block_offset` helper computations with the
  closed-form integer sum of per-degree `S` block lengths. This keeps the same
  layout while avoiding repeated prefix-sum loops in construction/reconstruction
  helper paths.
- Strengthened `test/rotate_batched_test.jl` coverage for the `S` index layout:
  closed-form lengths are checked against an explicit sum, adjacent degree
  offsets are checked against `length_S_block`, and all `(n, m, mp, ν)` slots are
  verified to be unique and complete.
- Added non-export guards for all four y-rotation wrapper operators and direct
  parity coverage for `rotate_local_y_op!`, while retaining the shared-`Ts`
  local-kernel check.

### Follow-up verification on 2026-06-19

```
julia --project=. test/rotate_batched_test.jl
# z-rotation operators (batched):        25684 Pass / 25684 Total
# axis-swap y-rotation operators (batched): 5727 Pass / 5727 Total

julia --project=. --threads=4 test/rotate_batched_test.jl
# z-rotation operators (batched):        25684 Pass / 25684 Total
# axis-swap y-rotation operators (batched): 5727 Pass / 5727 Total

julia --project=. -e 'using Test; using FastMultipole; using StaticArrays; include("test/operator_cache_types_test.jl")'
# operator cache support types: 176 Pass / 176 Total
# operator cache construction is side-effect-free: 5 Pass / 5 Total
# cache and scratch drive translations identically to legacy workspace: 4 Pass / 4 Total

julia --project=. -e 'using Pkg; Pkg.test()'
# FastMultipole tests passed
```

### Review pass: performance + test hardening on 2026-06-19

Performed as a correctness/performance/test-robustness review of the completed
task (the reviewer did not author the original implementation).

- **Correctness (verified, no change needed):** `update_S_blocks!` /
  `build_Ts_from_S!` are a faithful port of `update_Ts!` — same `(n,m,mp,ν)`
  traversal, sign recurrences, parity-gated `ν=0` term, and accumulate→×2→add-`ν=0`
  order. Confirmed **Float64 bit-exact** vs `update_Ts!` across
  `P ∈ {0,1,3,6,9,12,20}` and all test angles; Float32 deviation is intrinsic FP
  rounding (root cause: `get_scalar` returns Float64 `±1.0` literals, so production
  promotes within a triple while the `S` cache stores at the working precision —
  both land within ~2 ulp of the Float64 reference). `S_block_offset` closed form
  re-derived and checked against the explicit per-degree sum.
- **Performance:** `build_Ts_from_S!` previously re-ran the `cos/sin(νβ)` two-term
  recurrence inside every `(n,m,mp)` triple — the same redundant work that makes
  `update_Ts!` the dominant M2L cost. The `cos(νβ)`/`sin(νβ)` values depend only on
  `β`, so they are now materialized once into a `trig` scratch and reused, and the
  `m+mp` parity branch is hoisted out of the innermost `ν` loop. Result is
  **bit-identical to the prior implementation** for all TF/P/θ (no accuracy change)
  while cutting the rebuild to **2.1× (P=10) / 4.1× (P=20)** faster than production
  `update_Ts!` (was ~1.3-1.4×). A 6-arg form takes a caller-owned `trig` buffer and
  is **0 bytes allocated**; a 5-arg convenience method allocates one (used by the
  four `_op!` wrappers and tests). Hot-path wiring (task 014) should thread a scratch
  `trig` so the wrappers stay allocation-free.
- **Tests (`test/rotate_batched_test.jl`):** added a Float64 exact-`==`
  reconstruction assertion (locks the bit-exactness property), an identity check
  between the convenience and scratch `build_Ts_from_S!` forms, and a
  `@allocated == 0` assertion on the scratch form. Axis-swap testset: 5727 → 6003
  passing.

```
julia --project=. test/rotate_batched_test.jl            # 25684 + 6003 Pass
julia --project=. --threads=4 test/rotate_batched_test.jl # 25684 + 6003 Pass
julia --project=. -e 'using Pkg; Pkg.test()'             # FastMultipole tests passed
```

### Follow-up review/fix pass on 2026-06-19

Performed a second review pass focused on correctness, stated objective
alignment, performance, and test robustness.

- **Correctness / robustness:** `build_Ts_from_S!` now validates that `Ts`,
  `S_pos`, `S_neg`, and caller-owned `trig` storage are large enough before
  entering the `@inbounds` reconstruction loop. This keeps the hot loop unchanged
  but converts undersized scratch/cache inputs into deterministic `ArgumentError`s
  instead of unchecked access.
- **Performance / objective alignment:** the y-rotation operator wrappers now
  have caller-owned `trig` scratch overloads, so the variant-A operator path can
  use the already allocation-free reconstruction form directly. The convenience
  wrappers are retained for tests and low-frequency calls.
- **Scratch support:** `OperatorScratch` now carries `y_trig`, sized to
  `2 * max(P_active, 1)`, so later composition tasks can thread this scratch
  through the y-rotation stages without adding per-call allocations.
- **Specialization:** wrapper signatures now dispatch on `Val{LH}` rather than
  abstract `Val`, preserving specialization into the reused production y-apply
  kernels.
- **Tests:** `test/rotate_batched_test.jl` now checks the scratch overloads for
  all four y-rotation wrappers against the convenience path, asserts Float64
  zero-allocation behavior for the scratch path, and checks the undersized-trig
  guard. Float32 remains covered for parity; its apply-kernel allocations are
  inherited from the reused legacy y-rotation kernels. `test/operator_cache_types_test.jl`
  now verifies `OperatorScratch.y_trig` sizing, eltype, and per-thread uniqueness.

Verification:

```
julia --project=. test/rotate_batched_test.jl
# z-rotation operators (batched):        25684 Pass / 25684 Total
# axis-swap y-rotation operators (batched): 6819 Pass / 6819 Total

julia --project=. -e 'using Test; using FastMultipole; using StaticArrays; include("test/operator_cache_types_test.jl")'
# operator cache support types: 188 Pass / 188 Total
# operator cache construction is side-effect-free: 5 Pass / 5 Total
# cache and scratch drive translations identically to legacy workspace: 4 Pass / 4 Total

julia --project=. -e 'using Pkg; Pkg.test()'
# FastMultipole tests passed
```

## Approval Notes

**Approved on 2026-06-19** by clear-context review (a different agent from the
implementer; reviewed `START_HERE.md`, this task file, `theory/axis-swap-conventions.md`,
`src/rotate_batched.jl`, the `src/containers.jl` cache/scratch additions,
`test/rotate_batched_test.jl`, and `test/operator_cache_types_test.jl`).

Findings, in the START_HERE priority order:

1. **Objective alignment.** Delivers variant A as specified: angle-independent `S`
   blocks cached, per-call `Ts` reconstructed via `build_Ts_from_S!`, applied through
   the reused production `_rotate_*_y!` kernels with the `ζ`/`η` sign tables. No
   production hot-path call site changed; legacy `update_Ts!`/`rotate_*_y!` untouched.
   All required deliverables (construction, y-rotation composition, active/passive
   parity tests) are present.
2. **Correctness.** Verified `update_S_blocks!`/`build_Ts_from_S!` line-by-line against
   `update_Ts!` (`src/rotate.jl`): identical `(n,m,mp,ν)` traversal, sign recurrences,
   parity-gated `ν=0` term, and accumulate→×2→add-`ν=0` order. The factorization splits
   only the `get_scalar ∈ {0,±1}` factor, so Float64 is bit-exact — confirmed by the
   `Ts_op == Ts_ref` assertion. Re-derived the closed-form `S_block_offset`
   (`Σ k²(k+1)/2`) and confirmed all three integer divisions are exact; the layout
   uniqueness/completeness test corroborates it.
3. **Performance.** `trig` precompute + hoisted parity branch; allocation-free 6-arg
   form with `@allocated == 0` assertions across all four wrappers; `OperatorScratch`
   carries a correctly-sized `y_trig`. O(P⁴) `S` cache is flagged for `015`/`019`,
   acceptable for this milestone.
4. **Robustness.** `ArgumentError` guards on undersized `Ts`/`S_pos`/`S_neg`/`trig`;
   parity coverage spans `P ∈ {0,1,3,6,9}`, axis-aligned + off-axis ± angles,
   `Val(false)`/`Val(true)`, Float64/Float32, plus reset-not-accumulate and
   shared-`Ts` checks.
5. **Minimally invasive.** Additive, non-exported, correct file placement
   (`rotate.jl → rotate_batched.jl`); cache construction stays side-effect-free.
6. **Readable.** Thorough docstrings tying each operator back to the theory artifact.

Verification reproduced independently:

```
julia --project=. test/rotate_batched_test.jl
# z-rotation operators (batched):        25684 / 25684
# axis-swap y-rotation operators (batched): 6819 / 6819
julia --project=. -e 'using Test; using FastMultipole; using StaticArrays; include("test/operator_cache_types_test.jl")'
# operator cache support types: 188/188; side-effect-free: 5/5; legacy parity: 4/4
```

No blocking issues. Approved.
