# 001 Theory Z Rotation Operators

## Objective

Derive explicit real `2x2` z-rotation blocks for the current compressed
complex basis, including inverse signs and overwrite versus accumulate
semantics.

## Dependencies

- None

## Required Reading

- `START_HERE.md`
- Current `rotate_z!`, `back_rotate_z!`, and `harmonic_index(n, m)` definitions

## Artifacts or Production Surface

- `theory/z-rotation-operators.md`
- `scripts/z_rotation_*.jl`
- `data/z_rotation/`

## Deliverables

- Formula for forward rotation of each `(n, m)` real/imaginary pair
- Formula for inverse rotation and `m = 0` behavior
- Notes on overwrite semantics for forward rotation and accumulation semantics
  for back rotation
- Compatibility notes for the existing harmonic indexing layout

## Verification

Compare derived blocks against deterministic coefficient examples and the
current z-rotation formulas. Record commands, tolerances, and result summaries.

## Approval Notes

To be filled by a different agent after artifacts and verification are
complete.
