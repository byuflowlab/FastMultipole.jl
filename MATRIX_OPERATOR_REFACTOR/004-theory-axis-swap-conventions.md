# 004 Theory Axis Swap Conventions

## Objective

Derive invariant axis-swap signs and active/passive rotation conventions for
the explicit operator pipeline.

## Dependencies

- `001-theory-z-rotation-operators.md`

## Required Reading

- `START_HERE.md`
- `001-theory-z-rotation-operators.md`
- Current rotation and axis-alignment code used by M2L, M2M, and L2L paths

## Artifacts or Production Surface

- `theory/axis-swap-conventions.md`
- `scripts/axis_swap_*.jl`
- `data/axis_swap/`

## Deliverables

- [x] Axis-swap sign table and convention notes
- [x] Active/passive rotation interpretation matched to current behavior
- [x] Parity targets for future y-rotation and full-pipeline tests
- [x] Invariant axis-swap matrix requirements showing that all non-z rotation
  effects are captured by angle-independent matrices, with z-axis rotations as
  the only angle-dependent rotations in later M2M, M2L, and L2L compositions

## Verification

Check sign and convention formulas against deterministic axis-aligned and
off-axis examples, including examples that validate the invariant-matrix plus
z-axis-rotation-only decomposition. Record commands, tolerances, and result
summaries.

Completed verification:

- Command: `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/axis_swap_verify.jl`
- Tolerance: `atol <= 1.0e-12`
- Status: `PASS`
- Max multipole-axis-swap error: `0.0`
- Max local-axis-swap error: `0.0`
- Max `T` reconstruction error: `0.0`
- Max inactive-channel reset error: `0.0`
- Max reset/back-rotation target-independence error: `0.0`
- Coverage: axis-aligned `theta = 0`, axis-aligned `theta = pi`, positive
  off-axis, and negative off-axis cases for both `Val(false)` and `Val(true)`.
- Generated data: `data/axis_swap/verification_summary.md`

## Approval Notes

Approved by clear-context review.

Review confirmed that:

- the active coefficient convention and passive alignment interpretation match
  the production `rotate_z!`, y-alignment, z-translation, y-back-rotation, and
  final `back_rotate_z!` flow;
- the fixed axis-swap plus z-phase decomposition is documented with
  angle-independent `H(pi/2)`/axis-swap structure and z-axis phases as the only
  angle-dependent rotation terms;
- the distinction between multipole `zeta` signs and local `eta` signs is
  captured, including the reversed modulo difference used by production;
- forward and back y rotations reset their destination buffers, including the
  inactive `Val(false)` channel, while final z back-rotation accumulates into
  the target;
- verification covers `Val(false)`, `Val(true)`, axis-aligned, off-axis, reset,
  and `T` reconstruction cases at `atol <= 1.0e-12`.
