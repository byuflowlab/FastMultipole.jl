# 011 Implementation M2L Z Translation Blocks

## Objective

Implement fixed-`m` M2L z-translation blocks with approved distance scaling.

## Dependencies

- `002-theory-m2l-z-translation-scaling.md`
- `007-theory-coefficient-buffer-layout.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `008d-theory-dynamic-p-error-m2l-integration.md`
- `008h-theory-lamb-helmholtz-accuracy-order.md`
- `009-impl-basis-and-operator-cache-types.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/m2l-z-translation-scaling.md`
- Approved `theory/dynamic-p-error-m2l-integration.md`
- Existing M2L z-translation production code and tests

## Artifacts or Production Surface

- Production M2L z-translation operator code
- Tests comparing fixed-`m` blocks with current z-aligned M2L behavior

## Deliverables

- Fixed-`m` block construction
- Stable distance-scaling implementation
- Cache integration compatible with task `009`
- For `Val(true)`, block/cache sizing must support the active padded order
  `P_active = P_phi + 1` so the `chi` channel is translated through
  `P_chi = P_phi + 1`; `phi` rows above `P_phi` are padding/scratch, not
  requested physical output.

## Implementation Summary

- New file `src/translate_batched.jl` (per START_HERE Code Placement rule 2,
  `translate.jl` -> `translate_batched.jl`); `src/translate.jl` and all
  production call sites were left unmodified (additive only, mirroring task
  `010`).
  - `m2l_z_blocks!(blocks, t, P)` materializes the fixed-`m` blocks
    `K_m[n, np] = (n + np)! / t^(n + np + 1)` for all `0 <= m <= n, np <= P`
    into a flat column-major buffer, using the approved stable recurrence with
    `rho = inv(t)` (no separate factorial/power). The multiply chain and order
    match `translate_multipole_to_local_z!` so the apply is bit-for-bit
    identical to production.
  - `m2l_z_block_length(P) = (P+1)(P+2)(2P+3)/6` and
    `m2l_z_block_offset(m, P)` / `m2l_z_block_index(n, np, m, P)` define the flat
    layout (each `m`-block is dense `(P-m+1) x (P-m+1)`, column-major,
    `row = n-m+1`, `col = np-m+1`).
  - `apply_m2l_z!(out, in, blocks, P, lamb_helmholtz::Val, ::Val{:overwrite})` —
    symmetric path: for each output `(n, m)`, sums `np = m:P` in increasing order
    of `K_m[n, np] * in[:, c, harmonic_index(np, m)]` into
    `out[:, c, harmonic_index(n, m)]`, overwrite semantics. Same real scalar
    multiplies both real/imag lanes and (for `Val(true)`) both `φ`/`χ` channels.
  - `apply_m2l_z!(out, in, blocks, basis_info::OperatorBasisInfo, ::Val{:overwrite})`
    — order-aware path driven only through the `009` accessors: `φ` (component 1)
    translated through `orders.P_phi`; for `Val(true)`, `χ` (component 2)
    translated through the padded `orders.P_active = P_chi = P_phi + 1`. `blocks`
    sized at `P_active`; `φ` rows above `P_phi` are scratch/padding, not written.
  - Per approved theory, only `:overwrite` is provided; accumulation into the
    target occurs later via the inverse z-rotation, not in the z-translation.
- Operates on the existing production layout
  `weights[real_or_imag, component, harmonic_index]` (flat native buffers are
  task `017`). No new structs (the `OperatorBasisInfo`/order types from task
  `009` are reused); the `009` type surface is untouched. Functions are
  internal/non-exported.
- Registered: `include("translate_batched.jl")` in `src/FastMultipole.jl` after
  `translate.jl`. New test `test/translate_batched_test.jl`, registered in
  `test/runtests.jl` after `translate_multipole_to_local_test.jl`.
- No new verification script was added: the theory artifact
  `MATRIX_OPERATOR_REFACTOR/scripts/m2l_z_translation_verify.jl` (task `002`)
  remains untouched and was re-run as a sanity check (the same convention task
  `010` followed for `z_rotation_verify.jl`).

## Verification

Commands run and results:

```
julia --project=. test/translate_batched_test.jl
# M2L z-translation blocks (batched): 7163 Pass / 7163 Total
```

Covers `P ∈ {0,1,3,6,9}` × distances `t ∈ {5.408…, 1.3, -2.7, 11.0}`, both
`Val(false)`/`Val(true)`, `Float64`/`Float32`. Tests: block-length/offset
accounting; **bit-for-bit** (`===`) symmetric parity vs
`translate_multipole_to_local_z!`; overwrite of a preloaded sentinel; and the
`Val(true)` padded-order check (φ rows `n ≤ P_phi` equal the `P`-order
production result and χ rows `n ≤ P_active` equal the `(P+1)`-order production
result, both bit-for-bit). Also verifies `m2l_z_blocks!`/`apply_m2l_z!` are not
exported while remaining callable as module internals.

```
julia --check-bounds=yes --project=. test/translate_batched_test.jl
# M2L z-translation blocks (batched): 7163 Pass / 7163 Total
```

```
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/m2l_z_translation_verify.jl
# m2l_z_translation_verify: PASS  (existing task-002 theory artifact, re-run)
# max_abs_error: 3.47e-18, max_rel_error: 4.82e-16, max_inactive_channel_error: 0.0
```

```
julia --project=. -e 'using Pkg; Pkg.test()'
# Testing FastMultipole tests passed (full suite, no regressions)
```

## Approval Notes

Clear-context approval performed by a separate agent (different from the
completing agent), reviewing only `START_HERE.md`, this task file, the listed
production/test artifacts, and the verification notes.

Findings (correctness → performance → robustness → minimally invasive →
human-readable):

- **Correctness — confirmed.** Read `m2l_z_blocks!`/`apply_m2l_z!`
  (`src/translate_batched.jl`) against production `translate_multipole_to_local_z!`
  (`src/translate.jl:181`). The distance-scaling recurrence (init `rho = inv(t)`,
  down-degree advance `*(n+1)*rho`, across-source advance `*n_np*rho`), the flat
  block fill order, and the increasing-`np` summation order match production
  exactly, so materialize-then-apply is genuinely bit-for-bit. Block-length
  `(P+1)(P+2)(2P+3)/6` and `m2l_z_block_offset`/`m2l_z_block_index` accounting
  verified. The `Val(true)` padded order `P_active = P_phi + 1` matches
  `OperatorOrders` (`src/containers.jl:376`); `φ` is correctly capped at `P_phi`
  and `χ` carried to `P_active`. Re-ran `test/translate_batched_test.jl`:
  7163/7163 Pass.
- **Performance — appropriate.** Materialize-then-apply is the right shape for
  the later offset-class batching (020/021) and GPU (022) reuse. The per-`(n,m)`
  `m2l_z_block_offset` recompute is O(m), bounded by the O(P³) apply work.
- **Robustness — adequate.** Covers Float64/Float32, `Val(false)`/`Val(true)`,
  multiple distances incl. negative `t`; overwrite verified via `-99` sentinel.
- **Minimally invasive — confirmed.** Additive-only new file per Code Placement
  rule 2; no production call sites modified; functions non-exported; `009` type
  surface untouched; include placed after `translate.jl` (`FastMultipole.jl:95`).
- **Human-readable — good.** Clear header rationale and per-function docstrings.

Minor, non-blocking: the order-aware `apply_m2l_z!` `Val(false)` branch is not
directly exercised via `OperatorBasisInfo` (it degenerates to `P_active = P_phi`
and the symmetric path covers `Val(false)` bit-for-bit). Optional future
coverage; not a defect.

**Approved.**
