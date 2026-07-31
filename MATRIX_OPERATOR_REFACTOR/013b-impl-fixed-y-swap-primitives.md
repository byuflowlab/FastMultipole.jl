# 013b Implementation Fixed Y-Swap Primitives

## Objective

Implement the fixed `90 degree` y-axis swap primitives (`R_y(+π/2)` and
`R_y(−π/2)`) used by the near-term `FactoredRotationM2L` path. These are the dense,
angle/geometry-independent `S` / `S⁻¹` stages of the factored rotation alignment:

```text
R_z(φ) · R_y(θ) = R_z(φ) · R_y(π/2) · R_z(θ) · R_y(−π/2)
                    Z_φ        S          Z_θ       S⁻¹
```

(`theory/axis-swap-conventions.md`, the `Z_phi -> S -> Z_theta -> S_inv` form).
This task delivers only the fixed `S` and `S⁻¹` swap primitives at
single-interaction granularity, shaped for global batched GEMM, and preserving
FastMultipole's current sign and extra-`pi` convention. **Assembling** these
primitives with the `010` z rotations into the full `Z_phi -> S -> Z_theta -> S_inv`
alignment and return alignment is task `013c`; whole-M2L composition (and the
shared `011` z-axis M2L block / `012` Lamb-Helmholtz coupling) is task `014`. The
benchmark-gated comparison against the materialized `Ts(theta)` path is staged in
`015` and the definitive post-`023` end-to-end row `024`.

## Relationship To Task 013

- `013` (**done**) supplies the materialized arbitrary y-rotation path: cache the
  invariant `H(pi/2)` / `S_pos` / `S_neg` data, rebuild `Ts(theta)` with
  `build_Ts_from_S!`, and apply through production-parity wrappers.
- This task (`013b`) supplies the fixed ±π/2 `S` / `S⁻¹` swap primitives for the
  factored path; `013c` composes them with the `010` z rotations into the assembled
  factored alignment.

Both the materialized (`013`) and factored (`013b` + `013c`) paths must hit the
identical production M2L parity target, including the extra `pi` z-axis convention,
so `014` can put them behind swappable whole-M2L operator interfaces.

The folded no-`Ts` y-rotation path is **not** a product or benchmark variant. It may
exist only as temporary debug scaffolding while validating the explicit stages and
must not become a roadmap deliverable.

## Dependencies

- `004-theory-axis-swap-conventions.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `010-impl-z-rotation-operators.md` (diagonal z-rotation apply, reused for `Z_θ`)
- `012a-milestone-review-impl-009-012.md`
- `013-impl-axis-swap-operators.md` (cached `S_pos`/`S_neg`; same parity target)
- `013a-spike-m2l-batching-and-dynamic-p-feasibility.md` (historical batching
  evidence, superseded in near-term scope by the 2026-06-19 roadmap update)

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/axis-swap-conventions.md`
- `src/rotate_batched.jl` (materialized y-rotation operators and the `S` block
  layout)
- Existing axis-alignment and rotation production code in `src/rotate.jl`

## Artifacts or Production Surface

- Fixed `90 degree` y-axis swap and inverse-swap primitives in `src/rotate_batched.jl`
  (same `_batched` placement rule; additive, non-exported, no hot-path call-site
  change).
- Separately callable stage APIs suitable for global batched GEMM.
- Tests comparing the explicit staged alignment with current production behavior
  and with the materialized `Ts(theta)` path from `013`.

## Deliverables

- Explicit fixed `90 degree` y-axis swap stage and inverse-swap stage for multipole
  coefficients, preserving the current production sign convention.
- Explicit fixed `90 degree` y-axis swap stage and inverse-swap stage for local
  coefficients, preserving the current production sign convention.
- The fixed swap stages honor the reset-not-accumulate convention; final
  accumulation remains the responsibility of inverse z rotation.
- Fixed `R_y(±π/2)` matrices cached on the operator-invariant cache, built once from
  the `013` `S_pos` / `S_neg` blocks via `build_Ts_from_S!`.
- Primitive APIs shaped for global batched GEMM and compatible with the cache/scratch
  conventions from `009`/`010`/`013`.

**Out of scope (owned by `013c`).** Assembling these primitives with the `010` z
rotations into the `Z_phi -> S -> Z_theta -> S_inv` alignment and the return
alignment is task `013c`. This task does not deliver `Z_phi`/`Z_theta` composition,
the assembled alignment stages, or any z-axis M2L application (the shared `011`
z-axis M2L block is applied by `014`).

## Verification

- Parity tests for the fixed ±π/2 swap primitives vs production `rotate_*_y!` /
  `back_rotate_*_y!` evaluated at `±π/2`, for representative axis-aligned and
  off-axis source inputs, both `Val(false)` and `Val(true)`, Float64 and Float32,
  over the `P` grid used in `test/rotate_batched_test.jl`.
- Identity check that the cached `T_y_±90` matrices equal an on-the-fly
  `build_Ts_from_S!` rebuild at `±π/2`.
- Reset-not-accumulate behavior of the swap primitives (preloaded destination
  overwritten) and inactive-`χ`-channel zeroing for `Val(false)`.
- Record commands and result summaries. (Cross-parity of the *assembled* factored
  alignment vs production M2L alignment is verified in `013c`.)

## Implementation Notes

Implemented as the split-now primitive task only; the assembled
`Z_phi -> S -> Z_theta -> S_inv` alignment chain is deferred to
`013c-impl-factored-rotation-alignment.md`.

- `OperatorInvariantCache` now owns fixed production-parity y matrices:
  `T_y_pos90` for `R_y(+pi/2)` and `T_y_neg90` for `R_y(-pi/2)`.
- Both fixed matrices are built once during cache construction from the existing
  cached `S_pos` / `S_neg` blocks via `build_Ts_from_S!`, using constructor-local
  `y_trig` scratch.
- Added internal, non-exported fixed-stage primitives in `src/rotate_batched.jl`:
  `multipole_y_swap_pos90!`, `multipole_y_swap_neg90!`,
  `local_y_swap_pos90!`, and `local_y_swap_neg90!`.
- The primitives accept caller-provided fixed `T_y_*90` matrices plus the
  production sign tables, the active operator order, and `Val{LH}`, and apply through
  `_rotate_multipole_y!` / `_rotate_local_y!`.
- The y stages reset destination storage through the existing production-parity
  kernels. They do not accumulate; later return alignment should leave final
  accumulation to inverse z rotation.
- The focused tests exercise `Val(true)` at `P_active = P_phi + 1` so the padded
  `chi` degree is covered before `014` composes the full M2L path.
- No production hot-path call sites changed.

Verification run:

```text
julia --project=. test/rotate_batched_test.jl
  fixed y-swap primitives (013b): 548 passed
  z-rotation operators (batched): 25684 passed
  axis-swap y-rotation operators (batched): 6819 passed

julia --project=. -e 'using Test; using FastMultipole; using StaticArrays; include("test/operator_cache_types_test.jl")'
  operator cache support types: 212 passed
  operator cache construction is side-effect-free: 5 passed
  cache and scratch drive translations identically to legacy workspace: 4 passed

julia --project=. -e 'using Pkg; Pkg.test()'
  FastMultipole tests passed
```

## Review Pass (2026-06-22)

Clear-context review by a different agent from the implementer (reviewed
`START_HERE.md`, this task file, `src/rotate_batched.jl`, the `src/containers.jl`
`T_y_pos90`/`T_y_neg90` additions, and `test/rotate_batched_test.jl`). Findings in
`START_HERE.md` priority order:

1. **Objective alignment / correctness.** Delivers the fixed ±π/2 y-swap primitives
   as specified; assembly correctly deferred to `013c`. Both fixed matrices are built
   once from the cached `S_pos`/`S_neg` via `build_Ts_from_S!`; primitives apply
   through the reused production-parity kernels. Parity tests confirm
   `apply(T_y_±90) == rotate_*_y!(±π/2)` over the full P/angle/`Val{LH}`/precision grid.
2. **Naming corrected.** The file was renamed from
   `013b-impl-invariant-axis-swap-operators.md` to this name: the *invariant*
   axis-swap blocks (`S_pos`/`S_neg`) live in `013`, so the old name was misleading.
3. **Clarity.** Added a comment in `src/rotate_batched.jl` noting that the
   `pos90`/`neg90` function bodies are identical — the ±π/2 sign is carried entirely by
   the supplied matrix; the distinct names exist for `013c` staged-composition
   readability (naming the forward `S` vs inverse `S⁻¹` swap). The duplication is
   therefore intentional staged-API surface, not an oversight.
4. **Granularity (retain split).** This task is deliberately thin plumbing; its tests
   re-validate little beyond `013`. The split is retained because `013c`'s factored
   assembly is where the genuinely new, testable behavior lands, and the named fixed-swap
   stages give `013c` a clean composition surface.

Final clear-context **approval is still pending** a separate pass: the reviewer above
also applied the naming/comment cleanups to `013b`'s surface, so per the
"must not approve its own work" rule the Approved checkbox is left unset for an
independent agent.

## Approval Notes

**Approved (2026-06-22).** Independent clear-context approval by an agent that did
not implement `013b` nor edit its surface (the prior review-pass agent applied the
naming/comment cleanups and so was barred from approving). Reviewed `START_HERE.md`,
this task file, the fixed-swap primitives in `src/rotate_batched.jl`, the
`T_y_pos90`/`T_y_neg90` cache additions in `src/containers.jl`, and the `013b`
testset in `test/rotate_batched_test.jl`. Re-ran `julia --project=.
test/rotate_batched_test.jl`: `fixed y-swap primitives (013b)` 548 passed, with the
`z-rotation` (25684) and `axis-swap y-rotation` (6819) regressions green.

Findings in `START_HERE.md` priority order:

1. **Objective alignment.** Delivers exactly the fixed ±π/2 `S`/`S⁻¹` swap stages
   (multipole + local, forward + inverse); the `Z_phi -> S -> Z_theta -> S_inv`
   assembly is correctly deferred to `013c`, with no z composition or M2L
   application leaking in.
2. **Correctness.** Both fixed matrices are built once at cache construction from
   the cached `S_pos`/`S_neg` via `build_Ts_from_S!` at ±π/2; the primitives
   dispatch through the production-parity `_rotate_*_y!` kernels, inheriting the sign
   and extra-`pi` conventions. Parity tests confirm
   `apply(T_y_±90) == rotate_*_y!(±π/2) == back_rotate_*_y!` and that the cached
   matrices equal an on-the-fly rebuild.
3. **Performance.** Fixed matrices materialized once (not per call); primitives are
   allocation-free passthroughs.
4. **Robustness.** Tests cover reset-not-accumulate (preloaded destination
   overwritten), inactive-`χ` zeroing for `Val(false)`, the `Val(true)` padded
   `χ` degree at `P_active = P+1`, and non-export — over the full
   P/`Val{LH}`/Float64+Float32 grid.
5. **Minimally invasive.** Two cache fields plus four internal functions; additive,
   non-exported, no hot-path call sites changed.
6. **Readability.** The comment explaining the deliberately-identical
   `pos90`/`neg90` bodies (sign carried by the matrix; distinct names for `013c`
   composition) resolves the only apparent duplication.

Non-blocking note: `T_y_neg90` could be derived from `T_y_pos90` rather than rebuilt
independently, but the independent rebuild is a one-time cost and is clearer; fine
as-is. `013c` is now unblocked.

## Superseded As The 013c Building Block (`2026-06-23`)

User-directed roadmap amendment ("Plain-H"). A `013c` spike proved that composing
these **ζ-dressed** `T_y_pos90` / `T_y_neg90` primitives (applied through the
production `_rotate_*_y!` kernel) with a `Z_theta` z-rotation **cannot** reproduce the
production `R_y(theta)`: the kernel re-applies the ζ dressing around the swap, so
`(ζS)·Z·(ζS⁻¹) ≠ ζ·(S·Z·S⁻¹)` — ζ does not commute through `S`. `013c` therefore does
**not** use these primitives as its `S`/`S_inv`. The shipped factored swap is the
**plain** fixed per-degree mode matrices `V_n`/`U_n` (rank-1 Fourier modes of the
production y-operator, with ζ/η dressing absorbed); see the `013c` Revised
Implementation Notes and `theory/axis-swap-conventions.md`.

These `013b` primitives remain valid as a fixed-`±π/2` parity reference for the
materialized (`013`) path and are still covered by the `013b` tests; they are simply
not the load-bearing swap for the factored (`013c`) path. The task `013b` deliverable
(cache the fixed `±π/2` matrices and expose the primitives) stands; only its intended
downstream *use* in `013c` changed.
