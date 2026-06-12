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

- Axis-swap sign table and convention notes
- Active/passive rotation interpretation matched to current behavior
- Parity targets for future y-rotation and full-pipeline tests

## Verification

Check sign and convention formulas against deterministic axis-aligned and
off-axis examples. Record commands, tolerances, and result summaries.

## Approval Notes

To be filled by a different agent after artifacts and verification are
complete.
